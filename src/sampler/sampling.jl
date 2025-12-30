include("primes.jl")

struct Distribution1D{V<:AbstractVector{Float32}}
    func::V
    cdf::V
    func_int::Float32

    """Inner constructor for pre-computed distributions (used by to_gpu)."""
    function Distribution1D(func::V, cdf::V, func_int::Float32) where {V<:AbstractVector{Float32}}
        new{V}(func, cdf, func_int)
    end

    function Distribution1D(func::Vector{Float32})
        n = length(func)
        cdf = Vector{Float32}(undef, n + 1)
        # Compute integral of step function at `xᵢ`.
        cdf[1] = 0f0
        @_inbounds for i in 2:length(cdf)
            cdf[i] = cdf[i-1] + func[i-1] / n
        end
        # Transform step function integral into CDF.
        func_int = cdf[n+1]
        if func_int ≈ 0f0
            @_inbounds for i in 2:n+1
                cdf[i] = i / n
            end
        else
            @_inbounds for i in 2:n+1
                cdf[i] /= func_int
            end
        end

        new{typeof(func)}(func, cdf, func_int)
    end
end

function sample_discrete(d::Distribution1D, u::Float32)
    # Find interval.
    # TODO replace current `find_interval` function.
    offset = findlast(i -> d.cdf[i] ≤ u, 1:length(d.cdf))
    offset = clamp(offset, 1, length(d.cdf) - 1)

    pdf = d.func_int > 0 ? d.func[offset] / (d.func_int * length(d.func)) : 0f0
    u_remapped = (u - d.cdf[offset]) / (d.cdf[offset+1] - d.cdf[offset])
    offset, pdf, u_remapped
end

"""
GPU-compatible binary search to find last index where cdf[i] ≤ u.
Returns index in [1, n] where n = length(cdf).
"""
@inline function find_interval_binary(cdf, u::Float32)
    n = length(cdf)
    lo = Int32(1)
    hi = Int32(n)
    # Binary search for last index where cdf[i] ≤ u
    while lo < hi
        mid = (lo + hi + Int32(1)) ÷ Int32(2)
        if @inbounds cdf[mid] ≤ u
            lo = mid
        else
            hi = mid - Int32(1)
        end
    end
    return lo
end

"""
Sample continuous value from Distribution1D.
Returns (sampled value in [0,1], pdf, offset index).
"""
@inline function sample_continuous(d::Distribution1D, u::Float32)
    # Find interval using GPU-compatible binary search
    offset = find_interval_binary(d.cdf, u)
    offset = clamp(offset, Int32(1), Int32(length(d.cdf) - 1))

    # Compute offset within CDF segment
    du = u - @inbounds d.cdf[offset]
    denom = @inbounds d.cdf[offset + 1] - d.cdf[offset]
    if denom > 0f0
        du /= denom
    end

    # Compute continuous position
    n = length(d.func)
    sampled = (offset - Int32(1) + du) / n

    # Compute PDF: for piecewise-constant over [0,1], pdf = f[i] / func_int
    pdf = d.func_int > 0f0 ? (@inbounds d.func[offset]) / d.func_int : 0f0

    sampled, pdf, offset
end

"""
Compute PDF for sampling a specific value from Distribution1D.
"""
@inline function pdf(d::Distribution1D, u::Float32)::Float32
    n = length(d.func)
    offset = clamp(Int32(floor(u * n)) + Int32(1), Int32(1), Int32(n))
    d.func_int > 0f0 ? (@inbounds d.func[offset]) / d.func_int : 0f0
end

"""
2D piecewise-constant distribution for importance sampling.
Built from a 2D function (e.g., environment map luminance).
"""
struct Distribution2D{D<:Distribution1D, VD<:AbstractVector{D}}
    """Conditional distributions p(v|u) for each row."""
    p_conditional_v::VD
    """Marginal distribution p(u) over rows."""
    p_marginal::D

    """Inner constructor for pre-computed distributions (used by to_gpu)."""
    function Distribution2D(p_conditional_v::VD, p_marginal::D) where {D<:Distribution1D, VD<:AbstractVector{D}}
        new{D, VD}(p_conditional_v, p_marginal)
    end

    function Distribution2D(func::Matrix{Float32})
        nv, nu = size(func)  # nv = height (rows), nu = width (columns)

        # Build conditional distributions for each row
        p_conditional_v = Vector{Distribution1D{Vector{Float32}}}(undef, nv)
        marginal_func = Vector{Float32}(undef, nv)

        for v in 1:nv
            # Extract row and create distribution
            row = Float32[func[v, u] for u in 1:nu]
            p_conditional_v[v] = Distribution1D(row)
            # Marginal is the integral of each row
            marginal_func[v] = p_conditional_v[v].func_int
        end

        p_marginal = Distribution1D(marginal_func)
        new{Distribution1D{Vector{Float32}}, Vector{Distribution1D{Vector{Float32}}}}(p_conditional_v, p_marginal)
    end
end

"""
Sample a 2D point from the distribution.
Returns (Point2f(u, v), pdf).
"""
@inline function sample_continuous(d::Distribution2D, u::Point2f)
    # Sample v (row) from marginal distribution
    v_sampled, pdf_v, v_offset = sample_continuous(d.p_marginal, u[2])

    # Sample u (column) from conditional distribution for that row
    u_sampled, pdf_u, _ = sample_continuous(@inbounds(d.p_conditional_v[v_offset]), u[1])

    Point2f(u_sampled, v_sampled), pdf_u * pdf_v
end

"""
Compute PDF for sampling a specific 2D point.
"""
@inline function pdf(d::Distribution2D, uv::Point2f)::Float32
    nu = length(@inbounds(d.p_conditional_v[1]).func)
    nv = length(d.p_marginal.func)

    # Find indices
    iu = clamp(Int32(floor(uv[1] * nu)) + Int32(1), Int32(1), Int32(nu))
    iv = clamp(Int32(floor(uv[2] * nv)) + Int32(1), Int32(1), Int32(nv))

    @inbounds(d.p_conditional_v[iv]).func[iu] / d.p_marginal.func_int
end

function radical_inverse(base_index::Int64, a::UInt64)::Float32
    @real_assert base_index < 1024 "Limit for radical inverse is 1023"
    base_index == 0 && return reverse_bits(a) * 5.4210108624275222e-20

    base = PRIMES[base_index]
    inv_base = 1f0 / base
    reversed_digits = UInt64(0)
    inv_base_n = 1f0

    while a > 0
        next = UInt64(floor(a / base))
        digit = UInt64(a - next * base)
        reversed_digits = reversed_digits * base + digit
        inv_base_n *= inv_base
        a = next
    end
    min(reversed_digits * inv_base_n, 1f0)
end

@inline function reverse_bits(n::UInt32)::UInt32
    n = (n << 16) | (n >> 16)
    n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8)
    n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4)
    n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2)
    ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1)
end

@inline function reverse_bits(n::UInt64)::UInt64
    n0 = UInt64(reverse_bits(UInt32((n << 32) >> 32)))
    n1 = UInt64(reverse_bits(UInt32(n >> 32)))
    return (n0 << 32) | n1
end
