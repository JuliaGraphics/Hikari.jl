# ============================================================================
# Basic sampling primitives
# ============================================================================

"""
A point on a lens of radius 1: a disk for no `blades`, else a regular polygon of `blades`
corners on the unit circle, the first at `rotation` radians.
"""
@propagate_inbounds function sample_aperture(u::Point2f, blades::Int32, rotation::Float32)::Point2f
    blades == Int32(0) && return concentric_sample_disk(u)
    n = Float32(blades)
    slice = min(floor(u[1] * n), n - 1f0)
    along = u[1] * n - slice
    a0 = rotation + 2f0 * Float32(π) * slice / n
    a1 = a0 + 2f0 * Float32(π) / n
    corner0 = Point2f(cos(a0), sin(a0))
    corner1 = Point2f(cos(a1), sin(a1))
    return sqrt(u[2]) * (corner0 + along * (corner1 - corner0))
end

# ============================================================================
# Distributions
# ============================================================================

include("primes.jl")

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
The `textures` parameter is used to deref TextureRef fields when Distribution2D is stored in a MultiTypeSet.
"""
@propagate_inbounds function sample_continuous(d::Distribution2D, u::Point2f, textures)
    # Deref arrays from TextureRef (no-op if already arrays)
    marginal_cdf = Raycore.deref(textures, d.marginal_cdf)
    marginal_func = Raycore.deref(textures, d.marginal_func)
    conditional_cdf = Raycore.deref(textures, d.conditional_cdf)
    conditional_func = Raycore.deref(textures, d.conditional_func)
    conditional_func_int = Raycore.deref(textures, d.conditional_func_int)

    # Sample v (row) from marginal distribution
    # Each offset is assigned once: the second search's closure captures `v_offset`,
    # and a captured variable assigned twice is boxed, a dynamic call on the device.
    v_offset = clamp(find_interval(i -> marginal_cdf[i] <= u[2], u_int32(length(marginal_cdf))), Int32(1), d.nv)

    # Compute v_sampled
    du_v = u[2] - marginal_cdf[v_offset]
    denom_v = marginal_cdf[v_offset + 1] - marginal_cdf[v_offset]
    if denom_v > 0f0
        du_v /= denom_v
    end
    v_sampled = (v_offset - Int32(1) + du_v) / d.nv

    # PDF for v
    pdf_v = d.marginal_func_int > 0f0 ? marginal_func[v_offset] / d.marginal_func_int : 0f0

    # Sample u (column) from conditional distribution for row v_offset
    # Binary search in the v_offset column of conditional_cdf
    u_offset = clamp(find_interval(i -> conditional_cdf[i, v_offset] <= u[1], u_int32(size(conditional_cdf, 1))),
                     Int32(1), d.nu)

    # Compute u_sampled
    du_u = u[1] - conditional_cdf[u_offset, v_offset]
    denom_u = conditional_cdf[u_offset + 1, v_offset] - conditional_cdf[u_offset, v_offset]
    if denom_u > 0f0
        du_u /= denom_u
    end
    u_sampled = (u_offset - Int32(1) + du_u) / d.nu

    # PDF for u
    func_int_v = conditional_func_int[v_offset]
    pdf_u = func_int_v > 0f0 ? conditional_func[u_offset, v_offset] / func_int_v : 0f0

    Point2f(u_sampled, v_sampled), pdf_u * pdf_v
end

"""
Compute PDF for sampling a specific 2D point from flat distribution.
The `textures` parameter is used to deref TextureRef fields when Distribution2D is stored in a MultiTypeSet.
"""
@propagate_inbounds function pdf(d::Distribution2D, uv::Point2f, textures)::Float32
    # Deref array from TextureRef (no-op if already array)
    conditional_func = Raycore.deref(textures, d.conditional_func)

    # Find indices
    iu = clamp(floor_int32(uv[1] * d.nu) + Int32(1), Int32(1), d.nu)
    iv = clamp(floor_int32(uv[2] * d.nv) + Int32(1), Int32(1), d.nv)

    conditional_func[iu, iv] / d.marginal_func_int
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
