include("primes.jl")

struct Distribution1D{V<:AbstractVector{Float32}}
    func::V
    cdf::V
    func_int::Float32
end

function Distribution1D(func::Vector{Float32})
    n = length(func)
    cdf = Vector{Float32}(undef, n + 1)
    # Compute integral of step function at `xᵢ`.
    cdf[1] = 0f0
    for i in 2:length(cdf)
        cdf[i] = cdf[i-1] + func[i-1] / n
    end
    # Transform step function integral into CDF.
    func_int = cdf[n+1]
    if func_int ≈ 0f0
        for i in 2:n+1
            cdf[i] = i / n
        end
    else
        for i in 2:n+1
            cdf[i] /= func_int
        end
    end

    Distribution1D{typeof(func)}(func, cdf, func_int)
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
@propagate_inbounds function find_interval_binary(cdf, u::Float32)
    n = length(cdf)
    lo = Int32(1)
    hi = u_int32(n)
    # Fully unrolled branchless binary search (20 iterations enough for 2^20 elements)
    Base.Cartesian.@nexprs 20 _ -> begin
        mid = (lo + hi + Int32(1)) ÷ Int32(2)
        cond = cdf[mid] ≤ u
        lo = ifelse(cond, mid, lo)
        hi = ifelse(cond, hi, mid - Int32(1))
    end
    return lo
end

"""
Sample continuous value from Distribution1D.
Returns (sampled value in [0,1], pdf, offset index).
"""
@propagate_inbounds function sample_continuous(d::Distribution1D, u::Float32)
    # Find interval using GPU-compatible binary search
    offset = find_interval_binary(d.cdf, u)
    offset = clamp(offset, Int32(1), u_int32(length(d.cdf) - 1))

    # Compute offset within CDF segment
    du = u -  d.cdf[offset]
    denom =  d.cdf[offset + 1] - d.cdf[offset]
    if denom > 0f0
        du /= denom
    end

    # Compute continuous position
    n = length(d.func)
    sampled = (offset - Int32(1) + du) / n

    # Compute PDF: for piecewise-constant over [0,1], pdf = f[i] / func_int
    pdf = d.func_int > 0f0 ? ( d.func[offset]) / d.func_int : 0f0

    sampled, pdf, offset
end

"""
Compute PDF for sampling a specific value from Distribution1D.
"""
@propagate_inbounds function pdf(d::Distribution1D, u::Float32)::Float32
    n = length(d.func)
    offset = clamp(floor_int32(u * n) + Int32(1), Int32(1), u_int32(n))
    d.func_int > 0f0 ? ( d.func[offset]) / d.func_int : 0f0
end

# ============================================================================
# Distribution2D - GPU-compatible 2D distribution with flat storage
# ============================================================================

"""
    Distribution2D{V<:AbstractVector{Float32}, M<:AbstractMatrix{Float32}}

GPU-compatible 2D distribution that stores all data in flat arrays/matrices.
Avoids nested device arrays which cause SPIR-V validation errors on OpenCL.

The conditional distribution data is stored as 2D matrices where each column
represents one conditional distribution:
- `conditional_func[i, v]` = func value at index i for row v
- `conditional_cdf[i, v]` = cdf value at index i for row v
- `conditional_func_int[v]` = func_int for row v
"""
struct Distribution2D{V<:AbstractVector{Float32}, M<:AbstractMatrix{Float32}}
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
    Distribution2D(func::Matrix{Float32})

Construct a GPU-friendly 2D distribution directly from a function matrix.
The matrix has dimensions (nv, nu) where nv is height (rows) and nu is width (columns).
"""
function Distribution2D(func::Matrix{Float32})
    nv, nu = size(func)  # nv = height (rows), nu = width (columns)

    # Allocate flat arrays
    conditional_func = Matrix{Float32}(undef, nu, nv)
    conditional_cdf = Matrix{Float32}(undef, nu + 1, nv)
    conditional_func_int = Vector{Float32}(undef, nv)

    # Build conditional distributions for each row
    for v in 1:nv
        # Copy function values (transposed: row v -> column v)
        for u in 1:nu
            conditional_func[u, v] = func[v, u]
        end

        # Compute CDF
        conditional_cdf[1, v] = 0f0
        for u in 2:(nu + 1)
            conditional_cdf[u, v] = conditional_cdf[u-1, v] + conditional_func[u-1, v] / nu
        end

        # func_int is the last CDF value (before normalization)
        func_int = conditional_cdf[nu + 1, v]
        conditional_func_int[v] = func_int

        # Normalize CDF
        if func_int ≈ 0f0
            for u in 2:(nu + 1)
                conditional_cdf[u, v] = Float32(u - 1) / nu
            end
        else
            for u in 2:(nu + 1)
                conditional_cdf[u, v] /= func_int
            end
        end
    end

    # Build marginal distribution from row integrals
    marginal_func = copy(conditional_func_int)
    marginal_cdf = Vector{Float32}(undef, nv + 1)
    marginal_cdf[1] = 0f0
    for v in 2:(nv + 1)
        marginal_cdf[v] = marginal_cdf[v-1] + marginal_func[v-1] / nv
    end
    marginal_func_int = marginal_cdf[nv + 1]

    # Normalize marginal CDF
    if marginal_func_int ≈ 0f0
        for v in 2:(nv + 1)
            marginal_cdf[v] = Float32(v - 1) / nv
        end
    else
        for v in 2:(nv + 1)
            marginal_cdf[v] /= marginal_func_int
        end
    end

    Distribution2D(
        conditional_func, conditional_cdf, conditional_func_int,
        marginal_func, marginal_cdf, marginal_func_int,
        Int32(nu), Int32(nv)
    )
end

"""
Sample a 2D point from the flat distribution.
Returns (Point2f(u, v), pdf).
"""
@propagate_inbounds function sample_continuous(d::Distribution2D, u::Point2f)
    # Sample v (row) from marginal distribution
    v_offset = find_interval_binary_flat(d.marginal_cdf, u[2])
    v_offset = clamp(v_offset, Int32(1), d.nv)

    # Compute v_sampled
    du_v = u[2] -  d.marginal_cdf[v_offset]
    denom_v =  d.marginal_cdf[v_offset + 1] - d.marginal_cdf[v_offset]
    if denom_v > 0f0
        du_v /= denom_v
    end
    v_sampled = (v_offset - Int32(1) + du_v) / d.nv

    # PDF for v
    pdf_v = d.marginal_func_int > 0f0 ? ( d.marginal_func[v_offset]) / d.marginal_func_int : 0f0

    # Sample u (column) from conditional distribution for row v_offset
    # Binary search in the v_offset column of conditional_cdf
    u_offset = find_interval_binary_col(d.conditional_cdf, v_offset, u[1])
    u_offset = clamp(u_offset, Int32(1), d.nu)

    # Compute u_sampled
    du_u = u[1] -  d.conditional_cdf[u_offset, v_offset]
    denom_u =  d.conditional_cdf[u_offset + 1, v_offset] - d.conditional_cdf[u_offset, v_offset]
    if denom_u > 0f0
        du_u /= denom_u
    end
    u_sampled = (u_offset - Int32(1) + du_u) / d.nu

    # PDF for u
    func_int_v =  d.conditional_func_int[v_offset]
    pdf_u = func_int_v > 0f0 ? ( d.conditional_func[u_offset, v_offset]) / func_int_v : 0f0

    Point2f(u_sampled, v_sampled), pdf_u * pdf_v
end

"""
Binary search in a column of a 2D array (for conditional CDF).
"""
@propagate_inbounds function find_interval_binary_col(cdf::AbstractMatrix{Float32}, col::Int32, u::Float32)
    n = size(cdf, 1)
    lo = Int32(1)
    hi = u_int32(n)
    # Fully unrolled branchless binary search (20 iterations)
    Base.Cartesian.@nexprs 20 _ -> begin
        mid = (lo + hi + Int32(1)) ÷ Int32(2)
        cond = cdf[mid, col] ≤ u
        lo = ifelse(cond, mid, lo)
        hi = ifelse(cond, hi, mid - Int32(1))
    end
    return lo
end

"""
GPU-compatible fully unrolled branchless binary search in a flat vector (for marginal CDF).
"""
@propagate_inbounds function find_interval_binary_flat(cdf::AbstractVector{Float32}, u::Float32)
    n = length(cdf)
    lo = Int32(1)
    hi = u_int32(n)
    # Fully unrolled branchless binary search (20 iterations)
    Base.Cartesian.@nexprs 20 _ -> begin
        mid = (lo + hi + Int32(1)) ÷ Int32(2)
        cond = cdf[mid] ≤ u
        lo = ifelse(cond, mid, lo)
        hi = ifelse(cond, hi, mid - Int32(1))
    end
    return lo
end

"""
Compute PDF for sampling a specific 2D point from flat distribution.
"""
@propagate_inbounds function pdf(d::Distribution2D, uv::Point2f)::Float32
    # Find indices
    iu = clamp(floor_int32(uv[1] * d.nu) + Int32(1), Int32(1), d.nu)
    iv = clamp(floor_int32(uv[2] * d.nv) + Int32(1), Int32(1), d.nv)

    (d.conditional_func[iu, iv]) / d.marginal_func_int
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

@propagate_inbounds function reverse_bits(n::UInt32)::UInt32
    n = (n << 16) | (n >> 16)
    n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8)
    n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4)
    n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2)
    ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1)
end

@propagate_inbounds function reverse_bits(n::UInt64)::UInt64
    n0 = UInt64(reverse_bits(UInt32((n << 32) >> 32)))
    n1 = UInt64(reverse_bits(UInt32(n >> 32)))
    return (n0 << 32) | n1
end
