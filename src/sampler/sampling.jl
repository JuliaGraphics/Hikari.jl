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
    hi = u_int32(n)
    # Binary search for last index where cdf[i] ≤ u
    while lo < hi
        mid = (lo + hi + Int32(1)) ÷ Int32(2)
        if @_inbounds cdf[mid] ≤ u
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
    offset = clamp(offset, Int32(1), u_int32(length(d.cdf) - 1))

    # Compute offset within CDF segment
    du = u - @_inbounds d.cdf[offset]
    denom = @_inbounds d.cdf[offset + 1] - d.cdf[offset]
    if denom > 0f0
        du /= denom
    end

    # Compute continuous position
    n = length(d.func)
    sampled = (offset - Int32(1) + du) / n

    # Compute PDF: for piecewise-constant over [0,1], pdf = f[i] / func_int
    pdf = d.func_int > 0f0 ? (@_inbounds d.func[offset]) / d.func_int : 0f0

    sampled, pdf, offset
end

"""
Compute PDF for sampling a specific value from Distribution1D.
"""
@inline function pdf(d::Distribution1D, u::Float32)::Float32
    n = length(d.func)
    offset = clamp(floor_int32(u * n) + Int32(1), Int32(1), u_int32(n))
    d.func_int > 0f0 ? (@_inbounds d.func[offset]) / d.func_int : 0f0
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
    u_sampled, pdf_u, _ = sample_continuous(@_inbounds(d.p_conditional_v[v_offset]), u[1])

    Point2f(u_sampled, v_sampled), pdf_u * pdf_v
end

"""
Compute PDF for sampling a specific 2D point.
"""
@inline function pdf(d::Distribution2D, uv::Point2f)::Float32
    nu = length(@_inbounds(d.p_conditional_v[1]).func)
    nv = length(d.p_marginal.func)

    # Find indices
    iu = clamp(floor_int32(uv[1] * nu) + Int32(1), Int32(1), u_int32(nu))
    iv = clamp(floor_int32(uv[2] * nv) + Int32(1), Int32(1), u_int32(nv))

    @_inbounds(d.p_conditional_v[iv]).func[iu] / d.p_marginal.func_int
end

# ============================================================================
# FlatDistribution2D - GPU-compatible version without nested device arrays
# ============================================================================

"""
    FlatDistribution2D{V<:AbstractVector{Float32}, M<:AbstractMatrix{Float32}}

GPU-compatible 2D distribution that stores all data in flat arrays/matrices.
Avoids nested device arrays which cause SPIR-V validation errors on OpenCL.

The conditional distribution data is stored as 2D matrices where each column
represents one conditional distribution:
- `conditional_func[i, v]` = func value at index i for row v
- `conditional_cdf[i, v]` = cdf value at index i for row v
- `conditional_func_int[v]` = func_int for row v
"""
struct FlatDistribution2D{V<:AbstractVector{Float32}, M<:AbstractMatrix{Float32}}
    # Conditional distribution data stored as matrices (nu x nv) and (nu+1 x nv)
    conditional_func::M      # (nu, nv) - func values for all rows
    conditional_cdf::M       # (nu+1, nv) - cdf values for all rows
    conditional_func_int::V  # (nv,) - func_int for each row

    # Marginal distribution data
    marginal_func::V        # (nv,)
    marginal_cdf::V         # (nv+1,)
    marginal_func_int::Float32

    # Dimensions for indexing
    nu::Int32  # Number of columns (width)
    nv::Int32  # Number of rows (height)
end

"""
Convert a Distribution2D to FlatDistribution2D for GPU use.
"""
function FlatDistribution2D(d::Distribution2D)
    nv = length(d.p_conditional_v)
    nu = length(d.p_conditional_v[1].func)

    # Allocate flat arrays
    conditional_func = Matrix{Float32}(undef, nu, nv)
    conditional_cdf = Matrix{Float32}(undef, nu + 1, nv)
    conditional_func_int = Vector{Float32}(undef, nv)

    # Copy conditional distribution data
    for v in 1:nv
        cond = d.p_conditional_v[v]
        conditional_func[:, v] .= cond.func
        conditional_cdf[:, v] .= cond.cdf
        conditional_func_int[v] = cond.func_int
    end

    # Copy marginal distribution data
    marginal_func = copy(d.p_marginal.func)
    marginal_cdf = copy(d.p_marginal.cdf)
    marginal_func_int = d.p_marginal.func_int

    FlatDistribution2D(
        conditional_func, conditional_cdf, conditional_func_int,
        marginal_func, marginal_cdf, marginal_func_int,
        Int32(nu), Int32(nv)
    )
end

"""
Sample a 2D point from the flat distribution.
Returns (Point2f(u, v), pdf).
"""
@inline function sample_continuous(d::FlatDistribution2D, u::Point2f)
    # Sample v (row) from marginal distribution
    v_offset = find_interval_binary_flat(d.marginal_cdf, u[2])
    v_offset = clamp(v_offset, Int32(1), d.nv)

    # Compute v_sampled
    du_v = u[2] - @_inbounds d.marginal_cdf[v_offset]
    denom_v = @_inbounds d.marginal_cdf[v_offset + 1] - d.marginal_cdf[v_offset]
    if denom_v > 0f0
        du_v /= denom_v
    end
    v_sampled = (v_offset - Int32(1) + du_v) / d.nv

    # PDF for v
    pdf_v = d.marginal_func_int > 0f0 ? (@_inbounds d.marginal_func[v_offset]) / d.marginal_func_int : 0f0

    # Sample u (column) from conditional distribution for row v_offset
    # Binary search in the v_offset column of conditional_cdf
    u_offset = find_interval_binary_col(d.conditional_cdf, v_offset, u[1])
    u_offset = clamp(u_offset, Int32(1), d.nu)

    # Compute u_sampled
    du_u = u[1] - @_inbounds d.conditional_cdf[u_offset, v_offset]
    denom_u = @_inbounds d.conditional_cdf[u_offset + 1, v_offset] - d.conditional_cdf[u_offset, v_offset]
    if denom_u > 0f0
        du_u /= denom_u
    end
    u_sampled = (u_offset - Int32(1) + du_u) / d.nu

    # PDF for u
    func_int_v = @_inbounds d.conditional_func_int[v_offset]
    pdf_u = func_int_v > 0f0 ? (@_inbounds d.conditional_func[u_offset, v_offset]) / func_int_v : 0f0

    Point2f(u_sampled, v_sampled), pdf_u * pdf_v
end

"""
Binary search in a column of a 2D array (for conditional CDF).
"""
@inline function find_interval_binary_col(cdf::AbstractMatrix{Float32}, col::Int32, u::Float32)
    n = size(cdf, 1)
    lo = Int32(1)
    hi = u_int32(n)
    while lo < hi
        mid = (lo + hi + Int32(1)) ÷ Int32(2)
        if @_inbounds cdf[mid, col] ≤ u
            lo = mid
        else
            hi = mid - Int32(1)
        end
    end
    return lo
end

"""
Binary search in a flat vector (for marginal CDF).
Same as find_interval_binary but named differently for clarity.
"""
@inline function find_interval_binary_flat(cdf::AbstractVector{Float32}, u::Float32)
    n = length(cdf)
    lo = Int32(1)
    hi = u_int32(n)
    while lo < hi
        mid = (lo + hi + Int32(1)) ÷ Int32(2)
        if @_inbounds cdf[mid] ≤ u
            lo = mid
        else
            hi = mid - Int32(1)
        end
    end
    return lo
end

"""
Compute PDF for sampling a specific 2D point from flat distribution.
"""
@inline function pdf(d::FlatDistribution2D, uv::Point2f)::Float32
    # Find indices
    iu = clamp(floor_int32(uv[1] * d.nu) + Int32(1), Int32(1), d.nu)
    iv = clamp(floor_int32(uv[2] * d.nv) + Int32(1), Int32(1), d.nv)

    @_inbounds(d.conditional_func[iu, iv]) / d.marginal_func_int
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
