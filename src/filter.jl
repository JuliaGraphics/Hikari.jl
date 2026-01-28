# ============================================================================
# Filter System - pbrt-v4 Compatible
# ============================================================================
# Implements filter types and FilterSampler for importance sampling the filter
# footprint during camera ray generation.

abstract type AbstractFilter end

# ============================================================================
# FilterSample - Result of sampling a filter
# ============================================================================

"""
    FilterSample

Result of sampling a filter. Contains the offset position and the weight
for this sample (filter value / pdf).
"""
struct FilterSample
    p::Point2f      # Offset from pixel center
    weight::Float32 # Weight = filter(p) / pdf(p)
end

# ============================================================================
# Box Filter
# ============================================================================

"""
    BoxFilter(radius=Point2f(0.5, 0.5))

Simple box filter with constant weight 1.0 within the support region.
This is the simplest filter - uniform sampling within the pixel.
"""
struct BoxFilter <: AbstractFilter
    radius::Point2f
end

BoxFilter() = BoxFilter(Point2f(0.5f0, 0.5f0))

@inline function filter_radius(f::BoxFilter)
    f.radius
end

@inline function filter_evaluate(f::BoxFilter, p::Point2f)::Float32
    (abs(p[1]) <= f.radius[1] && abs(p[2]) <= f.radius[2]) ? 1f0 : 0f0
end

# Make filter callable (functor pattern for film.jl compatibility)
(f::BoxFilter)(p::Point2f) = filter_evaluate(f, p)

@inline function filter_integral(f::BoxFilter)::Float32
    2f0 * f.radius[1] * 2f0 * f.radius[2]
end

"""
Sample the box filter - uniform distribution with weight = 1.0
"""
@inline function filter_sample(f::BoxFilter, u::Point2f)::FilterSample
    p = Point2f(
        lerp(-f.radius[1], f.radius[1], u[1]),
        lerp(-f.radius[2], f.radius[2], u[2])
    )
    FilterSample(p, 1f0)
end

# ============================================================================
# Triangle Filter
# ============================================================================

"""
    TriangleFilter(radius=Point2f(2.0, 2.0))

Triangle (tent) filter - linear falloff from center to edges.
Can be importance sampled analytically with weight = 1.0.
"""
struct TriangleFilter <: AbstractFilter
    radius::Point2f
end

TriangleFilter() = TriangleFilter(Point2f(2f0, 2f0))

@inline function filter_radius(f::TriangleFilter)
    f.radius
end

@inline function filter_evaluate(f::TriangleFilter, p::Point2f)::Float32
    max(0f0, f.radius[1] - abs(p[1])) * max(0f0, f.radius[2] - abs(p[2]))
end

# Make filter callable (functor pattern for film.jl compatibility)
(f::TriangleFilter)(p::Point2f) = filter_evaluate(f, p)

@inline function filter_integral(f::TriangleFilter)::Float32
    f.radius[1]^2 * f.radius[2]^2
end

"""
Sample tent distribution for one dimension.
Uses inverse CDF sampling of the triangle distribution.
"""
@inline function sample_tent(u::Float32, r::Float32)::Float32
    # Triangle distribution: PDF(x) = (1 - |x|/r) / r for |x| < r
    # Split into two halves: left (negative) and right (positive)
    if u < 0.5f0
        # Left half: remap u to [0,1] for this half
        u_remapped = 2f0 * u
        # Inverse CDF for left half: x = r * (sqrt(u) - 1)
        return -r + r * sqrt(u_remapped)
    else
        # Right half: remap u to [0,1] for this half
        u_remapped = 2f0 * (1f0 - u)
        # Inverse CDF for right half: x = r * (1 - sqrt(u))
        return r * (1f0 - sqrt(u_remapped))
    end
end

"""
Sample the triangle filter - analytical importance sampling with weight = 1.0
"""
@inline function filter_sample(f::TriangleFilter, u::Point2f)::FilterSample
    p = Point2f(sample_tent(u[1], f.radius[1]), sample_tent(u[2], f.radius[2]))
    FilterSample(p, 1f0)
end

# ============================================================================
# Gaussian Filter
# ============================================================================

"""
    GaussianFilter(radius=Point2f(1.5, 1.5), sigma=0.5)

Gaussian filter with configurable sigma.
The filter is truncated at the radius and normalized by subtracting
the value at the boundary.
"""
struct GaussianFilter <: AbstractFilter
    radius::Point2f
    sigma::Float32
    exp_x::Float32  # Gaussian at radius.x (for normalization)
    exp_y::Float32  # Gaussian at radius.y
end

function GaussianFilter(radius::Point2f, sigma::Float32)
    exp_x = gaussian_1d(radius[1], sigma)
    exp_y = gaussian_1d(radius[2], sigma)
    GaussianFilter(radius, sigma, exp_x, exp_y)
end

GaussianFilter(; radius=Point2f(1.5f0, 1.5f0), sigma=0.5f0) =
    GaussianFilter(radius, Float32(sigma))

@inline function gaussian_1d(x::Float32, sigma::Float32)::Float32
    exp(-x^2 / (2f0 * sigma^2))
end

@inline function filter_radius(f::GaussianFilter)
    f.radius
end

@inline function filter_evaluate(f::GaussianFilter, p::Point2f)::Float32
    gx = max(0f0, gaussian_1d(p[1], f.sigma) - f.exp_x)
    gy = max(0f0, gaussian_1d(p[2], f.sigma) - f.exp_y)
    gx * gy
end

# Make filter callable (functor pattern for film.jl compatibility)
(f::GaussianFilter)(p::Point2f) = filter_evaluate(f, p)

@inline function gaussian_integral(a::Float32, b::Float32, sigma::Float32)::Float32
    # Integral of Gaussian from a to b
    # Use approximation since erf isn't available without SpecialFunctions
    # For filter integral we don't need high precision
    sqrt_2 = Float32(sqrt(2))
    sigma * Float32(sqrt(π)) / sqrt_2 * (erf_approx(b / (sqrt_2 * sigma)) - erf_approx(a / (sqrt_2 * sigma)))
end

# Approximation of erf for filter integral calculation
# Abramowitz and Stegun approximation (7.1.26), max error ~1.5e-7
@inline function erf_approx(x::Float32)::Float32
    sign_x = sign(x)
    x = abs(x)

    # Constants
    a1 = 0.254829592f0
    a2 = -0.284496736f0
    a3 = 1.421413741f0
    a4 = -1.453152027f0
    a5 = 1.061405429f0
    p  = 0.3275911f0

    t = 1f0 / (1f0 + p * x)
    y = 1f0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x)

    sign_x * y
end

@inline function filter_integral(f::GaussianFilter)::Float32
    (gaussian_integral(-f.radius[1], f.radius[1], f.sigma) - 2f0 * f.radius[1] * f.exp_x) *
    (gaussian_integral(-f.radius[2], f.radius[2], f.sigma) - 2f0 * f.radius[2] * f.exp_y)
end

# Gaussian filter requires tabulated importance sampling (FilterSampler)
# filter_sample is implemented via FilterSampler

# ============================================================================
# Mitchell Filter
# ============================================================================

"""
    MitchellFilter(radius=Point2f(2.0, 2.0), B=1/3, C=1/3)

Mitchell-Netravali filter - a family of cubic filters parameterized by B and C.
Default B=C=1/3 gives a good balance between ringing and blurring.
"""
struct MitchellFilter <: AbstractFilter
    radius::Point2f
    B::Float32
    C::Float32
end

MitchellFilter(; radius=Point2f(2f0, 2f0), B=1f0/3f0, C=1f0/3f0) =
    MitchellFilter(radius, Float32(B), Float32(C))

@inline function filter_radius(f::MitchellFilter)
    f.radius
end

@inline function mitchell_1d(x::Float32, B::Float32, C::Float32)::Float32
    x = abs(x)
    if x <= 1f0
        return ((12f0 - 9f0*B - 6f0*C) * x^3 +
                (-18f0 + 12f0*B + 6f0*C) * x^2 +
                (6f0 - 2f0*B)) / 6f0
    elseif x <= 2f0
        return ((-B - 6f0*C) * x^3 +
                (6f0*B + 30f0*C) * x^2 +
                (-12f0*B - 48f0*C) * x +
                (8f0*B + 24f0*C)) / 6f0
    else
        return 0f0
    end
end

@inline function filter_evaluate(f::MitchellFilter, p::Point2f)::Float32
    mitchell_1d(2f0 * p[1] / f.radius[1], f.B, f.C) *
    mitchell_1d(2f0 * p[2] / f.radius[2], f.B, f.C)
end

# Make filter callable (functor pattern for film.jl compatibility)
(f::MitchellFilter)(p::Point2f) = filter_evaluate(f, p)

@inline function filter_integral(f::MitchellFilter)::Float32
    # Mitchell integral is radius.x * radius.y / 4
    f.radius[1] * f.radius[2] / 4f0
end

# Mitchell filter requires tabulated importance sampling (FilterSampler)

# ============================================================================
# Lanczos Sinc Filter
# ============================================================================

"""
    LanczosSincFilter(radius=Point2f(4.0, 4.0), tau=3.0)

Lanczos-windowed sinc filter for high-quality reconstruction.
The sinc function is windowed by another sinc to reduce ringing.
"""
struct LanczosSincFilter <: AbstractFilter
    radius::Point2f
    τ::Float32
end

LanczosSincFilter(; radius=Point2f(4f0, 4f0), tau=3f0) =
    LanczosSincFilter(radius, Float32(tau))

@inline function filter_radius(f::LanczosSincFilter)
    f.radius
end

@inline function sinc_func(x::Float32)::Float32
    x = abs(x)
    x < 1f-5 && return 1f0
    x *= Float32(π)
    sin(x) / x
end

@inline function windowed_sinc(x::Float32, r::Float32, τ::Float32)::Float32
    x = abs(x)
    x > r && return 0f0
    sinc_func(x) * sinc_func(x / τ)
end

@inline function filter_evaluate(f::LanczosSincFilter, p::Point2f)::Float32
    windowed_sinc(p[1], f.radius[1], f.τ) * windowed_sinc(p[2], f.radius[2], f.τ)
end

# Make filter callable (functor pattern for film.jl compatibility)
(f::LanczosSincFilter)(p::Point2f) = filter_evaluate(f, p)

@inline function filter_integral(f::LanczosSincFilter)::Float32
    # Monte Carlo estimate of integral (computed once, matches pbrt-v4)
    # For tau=3, radius=4, integral ≈ 1.0
    sqrt_samples = 64
    n_samples = sqrt_samples * sqrt_samples
    area = 2f0 * f.radius[1] * 2f0 * f.radius[2]

    sum = 0f0
    for y in 0:sqrt_samples-1
        for x in 0:sqrt_samples-1
            ux = (Float32(x) + 0.5f0) / Float32(sqrt_samples)
            uy = (Float32(y) + 0.5f0) / Float32(sqrt_samples)
            px = lerp(-f.radius[1], f.radius[1], ux)
            py = lerp(-f.radius[2], f.radius[2], uy)
            sum += filter_evaluate(f, Point2f(px, py))
        end
    end
    sum / n_samples * area
end

# Lanczos filter requires tabulated importance sampling (FilterSampler)

# Note: lerp(v1, v2, t) is defined in spectrum.jl

# ============================================================================
# PiecewiseConstant2D - For importance sampling arbitrary 2D functions
# ============================================================================

"""
    PiecewiseConstant2D

2D piecewise constant distribution for importance sampling.
Stores a 2D function tabulated on a grid and allows efficient sampling
proportional to the function values.
"""
struct PiecewiseConstant2D{A<:AbstractMatrix{Float32}, V<:AbstractVector{Float32}}
    # Function values on grid
    func::A

    # Marginal CDF for y (sum of each row)
    marginal_cdf::V
    marginal_func::V

    # Conditional CDFs for x given y (one per row)
    conditional_cdfs::A

    # Domain
    domain_min::Point2f
    domain_max::Point2f

    # Grid size
    nx::Int32
    ny::Int32
end

"""
Build a PiecewiseConstant2D from a 2D function array.
func[y, x] = function value at grid cell (x, y).
"""
function PiecewiseConstant2D(func::AbstractMatrix{Float32}, domain_min::Point2f, domain_max::Point2f)
    ny, nx = size(func)

    # Compute conditional CDFs for each row (sampling x given y)
    conditional_cdfs = similar(func)
    row_integrals = zeros(Float32, ny)

    for y in 1:ny
        # Compute CDF for this row
        row_sum = 0f0
        for x in 1:nx
            row_sum += func[y, x]
            conditional_cdfs[y, x] = row_sum
        end
        row_integrals[y] = row_sum

        # Normalize CDF
        if row_sum > 0f0
            for x in 1:nx
                conditional_cdfs[y, x] /= row_sum
            end
        else
            # Uniform fallback if row is all zeros
            for x in 1:nx
                conditional_cdfs[y, x] = Float32(x) / Float32(nx)
            end
        end
    end

    # Compute marginal CDF for y
    marginal_cdf = zeros(Float32, ny)
    marginal_sum = 0f0
    for y in 1:ny
        marginal_sum += row_integrals[y]
        marginal_cdf[y] = marginal_sum
    end

    # Normalize marginal CDF
    if marginal_sum > 0f0
        marginal_cdf ./= marginal_sum
    else
        for y in 1:ny
            marginal_cdf[y] = Float32(y) / Float32(ny)
        end
    end

    PiecewiseConstant2D(
        func,
        marginal_cdf,
        row_integrals,
        conditional_cdfs,
        domain_min,
        domain_max,
        Int32(nx),
        Int32(ny)
    )
end

"""
Sample from the 2D distribution using inverse CDF sampling.
Returns (point, pdf, grid_index).
"""
@inline function sample_piecewise_2d(d::PiecewiseConstant2D, u::Point2f)
    # Sample y from marginal distribution
    y_idx = binary_search_cdf(d.marginal_cdf, u[2])
    y_idx = clamp(y_idx, Int32(1), d.ny)

    # Sample x from conditional distribution given y
    x_idx = binary_search_cdf_row(d.conditional_cdfs, y_idx, u[1])
    x_idx = clamp(x_idx, Int32(1), d.nx)

    # Compute continuous position within cell
    # Map grid index to domain coordinates
    dx = (d.domain_max[1] - d.domain_min[1]) / Float32(d.nx)
    dy = (d.domain_max[2] - d.domain_min[2]) / Float32(d.ny)

    # Center of the selected cell
    px = d.domain_min[1] + (Float32(x_idx) - 0.5f0) * dx
    py = d.domain_min[2] + (Float32(y_idx) - 0.5f0) * dy

    # Compute PDF at this point
    total_sum = d.marginal_cdf[end] * sum(d.marginal_func)
    if total_sum > 0f0
        pdf = d.func[y_idx, x_idx] / total_sum * Float32(d.nx * d.ny)
    else
        pdf = 1f0 / (Float32(d.nx * d.ny))
    end

    Point2f(px, py), pdf, Point2{Int32}(x_idx, y_idx)
end

# GPU-compatible branchless binary search - fully unrolled (SPIR-V requires no loops)
# Uses @nexprs to generate 20 iterations of branchless binary search at compile time
@inline function binary_search_cdf(cdf::AbstractVector{Float32}, u::Float32)::Int32
    n = length(cdf)
    lo = Int32(1)
    hi = Int32(n)

    # Fully unrolled branchless binary search (20 iterations enough for 2^20 elements)
    Base.Cartesian.@nexprs 20 _ -> begin
        mid = (lo + hi) ÷ Int32(2)
        cond = cdf[mid] < u
        lo = ifelse(cond, mid + Int32(1), lo)
        hi = ifelse(cond, hi, mid)
    end
    lo
end

# GPU-compatible branchless binary search for matrix rows - fully unrolled
@inline function binary_search_cdf_row(cdfs::AbstractMatrix{Float32}, row::Int32, u::Float32)::Int32
    n = size(cdfs, 2)
    lo = Int32(1)
    hi = Int32(n)

    # Fully unrolled branchless binary search (20 iterations enough for 2^20 elements)
    Base.Cartesian.@nexprs 20 _ -> begin
        mid = (lo + hi) ÷ Int32(2)
        cond = cdfs[row, mid] < u
        lo = ifelse(cond, mid + Int32(1), lo)
        hi = ifelse(cond, hi, mid)
    end
    lo
end

# ============================================================================
# FilterSampler - Importance sampling for arbitrary filters
# ============================================================================

"""
    FilterSampler{F<:AbstractFilter}

Importance sampler for filters that don't have analytical sampling.
Tabulates the filter function and uses PiecewiseConstant2D for sampling.

For BoxFilter and TriangleFilter, direct analytical sampling is used instead.
"""
struct FilterSampler{F<:AbstractFilter, D}
    filter::F
    distrib::D  # PiecewiseConstant2D or nothing for analytically samplable filters
    func_table::Matrix{Float32}  # Tabulated filter values
end

"""
Create a FilterSampler for the given filter.
"""
function FilterSampler(filter::F) where F<:AbstractFilter
    # Box and Triangle filters have analytical sampling - no table needed
    if filter isa BoxFilter || filter isa TriangleFilter
        return FilterSampler{F, Nothing}(filter, nothing, Matrix{Float32}(undef, 0, 0))
    end

    # Tabulate filter function
    r = filter_radius(filter)
    # Use 32 samples per unit radius (matching pbrt-v4)
    nx = max(Int32(32 * r[1]), Int32(8))
    ny = max(Int32(32 * r[2]), Int32(8))

    func_table = Matrix{Float32}(undef, ny, nx)
    domain_min = Point2f(-r[1], -r[2])
    domain_max = Point2f(r[1], r[2])

    dx = (domain_max[1] - domain_min[1]) / Float32(nx)
    dy = (domain_max[2] - domain_min[2]) / Float32(ny)

    for iy in 1:ny
        for ix in 1:nx
            px = domain_min[1] + (Float32(ix) - 0.5f0) * dx
            py = domain_min[2] + (Float32(iy) - 0.5f0) * dy
            func_table[iy, ix] = filter_evaluate(filter, Point2f(px, py))
        end
    end

    distrib = PiecewiseConstant2D(func_table, domain_min, domain_max)

    FilterSampler{F, typeof(distrib)}(filter, distrib, func_table)
end

"""
Sample the filter using the FilterSampler.
Returns FilterSample with position and weight.
"""
@inline function sample_filter(fs::FilterSampler{BoxFilter, Nothing}, u::Point2f)::FilterSample
    filter_sample(fs.filter, u)
end

@inline function sample_filter(fs::FilterSampler{TriangleFilter, Nothing}, u::Point2f)::FilterSample
    filter_sample(fs.filter, u)
end

@inline function sample_filter(fs::FilterSampler{F, D}, u::Point2f)::FilterSample where {F<:AbstractFilter, D<:PiecewiseConstant2D}
    p, pdf, grid_idx = sample_piecewise_2d(fs.distrib, u)

    # Weight = f(p) / pdf
    f_val = fs.func_table[grid_idx[2], grid_idx[1]]
    weight = pdf > 0f0 ? f_val / pdf : 0f0

    FilterSample(p, weight)
end

# ============================================================================
# GPU-Compatible Filter Sampling with Importance Sampling (pbrt-v4 style)
# ============================================================================

# For GPU kernels, we use tabulated importance sampling for Gaussian/Mitchell/Lanczos
# filters, matching pbrt-v4's FilterSampler approach. Box and Triangle use analytical
# sampling with weight=1.

"""
    GPUFilterParams

GPU-compatible filter parameters for kernel use.
For filters requiring tabulated importance sampling (Gaussian, Mitchell, Lanczos),
the table data must be passed separately to the kernel.
"""
struct GPUFilterParams
    filter_type::Int32  # 1=Box, 2=Triangle, 3=Gaussian, 4=Mitchell, 5=Lanczos
    radius::Point2f
    param1::Float32     # sigma for Gaussian, B for Mitchell, tau for Lanczos
    param2::Float32     # C for Mitchell
    exp_x::Float32      # For Gaussian
    exp_y::Float32      # For Gaussian
end

function GPUFilterParams(f::BoxFilter)
    GPUFilterParams(Int32(1), f.radius, 0f0, 0f0, 0f0, 0f0)
end

function GPUFilterParams(f::TriangleFilter)
    GPUFilterParams(Int32(2), f.radius, 0f0, 0f0, 0f0, 0f0)
end

function GPUFilterParams(f::GaussianFilter)
    GPUFilterParams(Int32(3), f.radius, f.sigma, 0f0, f.exp_x, f.exp_y)
end

function GPUFilterParams(f::MitchellFilter)
    GPUFilterParams(Int32(4), f.radius, f.B, f.C, 0f0, 0f0)
end

function GPUFilterParams(f::LanczosSincFilter)
    GPUFilterParams(Int32(5), f.radius, f.τ, 0f0, 0f0, 0f0)
end

"""
    GPUFilterSamplerData

GPU-compatible tabulated data for importance sampling filters.
Stores the distribution data needed for sampling Gaussian/Mitchell/Lanczos filters.

This matches pbrt-v4's FilterSampler which uses PiecewiseConstant2D.
"""
struct GPUFilterSamplerData{V<:AbstractVector{Float32}, M<:AbstractMatrix{Float32}}
    # Tabulated filter values (ny × nx) - for computing weight = f[pi] / pdf
    func::M

    # Marginal distribution (for y sampling)
    # marginal_cdf[i] = CDF value at cell boundary i (size ny+1, starts with 0)
    marginal_cdf::V
    marginal_func::V  # Row integrals (size ny)

    # Conditional distributions (for x|y sampling)
    # conditional_cdf[y, x] = CDF value at cell boundary x for row y (size ny × (nx+1))
    conditional_cdf::M

    # Domain and grid info
    domain_min::Point2f
    domain_max::Point2f
    nx::Int32
    ny::Int32

    # Total integral (for PDF computation)
    func_integral::Float32
end

"""
Build GPU-compatible filter sampler data from a filter.
Returns nothing for Box/Triangle filters (they use analytical sampling).
"""
function GPUFilterSamplerData(filter::AbstractFilter)
    # Box and Triangle use analytical sampling
    if filter isa BoxFilter || filter isa TriangleFilter
        return nothing
    end

    r = filter_radius(filter)
    # Use 32 samples per unit radius (matching pbrt-v4)
    nx = max(Int32(ceil(32 * r[1])), Int32(8))
    ny = max(Int32(ceil(32 * r[2])), Int32(8))

    domain_min = Point2f(-r[1], -r[2])
    domain_max = Point2f(r[1], r[2])
    dx = (domain_max[1] - domain_min[1]) / Float32(nx)
    dy = (domain_max[2] - domain_min[2]) / Float32(ny)

    # Tabulate filter function
    func = Matrix{Float32}(undef, ny, nx)
    for iy in 1:ny
        for ix in 1:nx
            # Center of cell (ix, iy)
            px = domain_min[1] + (Float32(ix) - 0.5f0) * dx
            py = domain_min[2] + (Float32(iy) - 0.5f0) * dy
            func[iy, ix] = max(0f0, filter_evaluate(filter, Point2f(px, py)))
        end
    end

    # Build marginal distribution (row sums, NOT including dx - that's handled in func_integral)
    marginal_func = zeros(Float32, ny)
    for iy in 1:ny
        for ix in 1:nx
            marginal_func[iy] += func[iy, ix]
        end
    end

    # Build marginal CDF (size ny+1, starts with 0, ends with 1)
    marginal_cdf = zeros(Float32, ny + 1)
    marginal_cdf[1] = 0f0
    for iy in 1:ny
        marginal_cdf[iy + 1] = marginal_cdf[iy] + marginal_func[iy]
    end
    # func_integral = sum of all func values * cell_area (proper 2D integral)
    # This is what pbrt computes: integral of filter over domain
    func_integral = marginal_cdf[end] * dx * dy
    # Normalize CDF by its endpoint so it goes 0 to 1
    marginal_cdf_end = marginal_cdf[end]
    if marginal_cdf_end > 0f0
        marginal_cdf ./= marginal_cdf_end
    else
        # Uniform fallback
        for iy in 1:ny+1
            marginal_cdf[iy] = Float32(iy - 1) / Float32(ny)
        end
    end

    # Build conditional CDFs (size ny × (nx+1))
    conditional_cdf = zeros(Float32, ny, nx + 1)
    for iy in 1:ny
        conditional_cdf[iy, 1] = 0f0
        for ix in 1:nx
            conditional_cdf[iy, ix + 1] = conditional_cdf[iy, ix] + func[iy, ix]
        end
        # Normalize
        row_sum = conditional_cdf[iy, nx + 1]
        if row_sum > 0f0
            for ix in 1:nx+1
                conditional_cdf[iy, ix] /= row_sum
            end
        else
            # Uniform fallback
            for ix in 1:nx+1
                conditional_cdf[iy, ix] = Float32(ix - 1) / Float32(nx)
            end
        end
    end

    GPUFilterSamplerData(
        func,
        marginal_cdf,
        marginal_func,
        conditional_cdf,
        domain_min,
        domain_max,
        nx,
        ny,
        func_integral
    )
end

"""
Binary search in CDF to find interval containing u.
Returns index o such that cdf[o] <= u < cdf[o+1] (1-based indexing).
CDF has size n+1 with cdf[1]=0 and cdf[n+1]=1.
GPU-compatible fully unrolled branchless implementation.
"""
@inline function find_interval(cdf::AbstractVector{Float32}, u::Float32, n::Int32)::Int32
    # Fully unrolled branchless binary search for interval containing u
    lo = Int32(1)
    hi = n + Int32(1)

    # 20 iterations (enough for 2^20 elements)
    Base.Cartesian.@nexprs 20 _ -> begin
        mid = (lo + hi) >> Int32(1)
        cond = cdf[mid] <= u
        lo = ifelse(cond, mid, lo)
        hi = ifelse(cond, hi, mid)
    end
    lo
end

"""
Sample 1D piecewise constant distribution (pbrt-v4 compatible).
Returns (continuous_position, pdf, cell_index).
"""
@inline function sample_piecewise_1d(
    cdf::AbstractVector{Float32},
    func::AbstractVector{Float32},
    func_integral::Float32,
    u::Float32,
    n::Int32,
    domain_min::Float32,
    domain_max::Float32
)
    # Find interval containing u
    o = find_interval(cdf, u, n)
    o = clamp(o, Int32(1), n)

    # Compute offset within interval
    du = u - cdf[o]
    diff = cdf[o + 1] - cdf[o]
    if diff > 0f0
        du /= diff
    else
        du = 0f0
    end

    # Compute PDF: pdf = func[o] / func_integral
    pdf = func_integral > 0f0 ? func[o] / func_integral : 0f0

    # Compute continuous position: lerp((o-1 + du) / n, min, max)
    # Note: o is 1-based, so use (o-1 + du) / n to get [0, 1] range
    t = (Float32(o - 1) + du) / Float32(n)
    pos = lerp(domain_min, domain_max, t)

    pos, pdf, o
end

"""
Sample 1D from conditional distribution (row of 2D distribution).
"""
@inline function sample_conditional_1d(
    conditional_cdf::AbstractMatrix{Float32},
    func::AbstractMatrix{Float32},
    row_integral::Float32,
    row::Int32,
    u::Float32,
    nx::Int32,
    domain_min::Float32,
    domain_max::Float32
)
    # Find interval in this row's CDF (fully unrolled branchless binary search)
    lo = Int32(1)
    hi = nx + Int32(1)

    Base.Cartesian.@nexprs 20 _ -> begin
        mid = (lo + hi) >> Int32(1)
        cond = conditional_cdf[row, mid] <= u
        lo = ifelse(cond, mid, lo)
        hi = ifelse(cond, hi, mid)
    end
    o = clamp(lo, Int32(1), nx)

    # Compute offset within interval
    du = u - conditional_cdf[row, o]
    diff = conditional_cdf[row, o + 1] - conditional_cdf[row, o]
    if diff > 0f0
        du /= diff
    else
        du = 0f0
    end

    # Compute PDF: pdf = func[row, o] / row_integral
    pdf = row_integral > 0f0 ? func[row, o] / row_integral : 0f0

    # Compute continuous position
    t = (Float32(o - 1) + du) / Float32(nx)
    pos = lerp(domain_min, domain_max, t)

    pos, pdf, o
end

"""
Sample filter using tabulated importance sampling (pbrt-v4 compatible).
This is used for Gaussian, Mitchell, and Lanczos filters.
Returns FilterSample with position and weight = f[sampled_point] / pdf.
"""
@inline function filter_sample_tabulated(
    data::GPUFilterSamplerData,
    u::Point2f
)::FilterSample
    # Sample y from marginal distribution
    py, pdf_y, iy = sample_piecewise_1d(
        data.marginal_cdf,
        data.marginal_func,
        data.func_integral,
        u[2],
        data.ny,
        data.domain_min[2],
        data.domain_max[2]
    )

    # Sample x from conditional distribution given y
    row_integral = data.marginal_func[iy]
    px, pdf_x, ix = sample_conditional_1d(
        data.conditional_cdf,
        data.func,
        row_integral,
        iy,
        u[1],
        data.nx,
        data.domain_min[1],
        data.domain_max[1]
    )

    # Combined PDF
    pdf = pdf_x * pdf_y

    # Weight = f[ix, iy] / pdf (pbrt-v4 style)
    f_val = data.func[iy, ix]
    weight = pdf > 0f0 ? f_val / pdf : 0f0

    FilterSample(Point2f(px, py), weight)
end

"""
Sample filter - uses analytical sampling where possible,
tabulated importance sampling for Gaussian/Mitchell/Lanczos.
"""
@inline function filter_sample(params::GPUFilterParams, u::Point2f)::FilterSample
    filter_type = params.filter_type

    if filter_type == Int32(1)
        # Box filter - uniform sampling, weight = 1
        p = Point2f(
            lerp(-params.radius[1], params.radius[1], u[1]),
            lerp(-params.radius[2], params.radius[2], u[2])
        )
        return FilterSample(p, 1f0)

    elseif filter_type == Int32(2)
        # Triangle filter - analytical sampling, weight = 1
        p = Point2f(
            sample_tent(u[1], params.radius[1]),
            sample_tent(u[2], params.radius[2])
        )
        return FilterSample(p, 1f0)

    else
        # Gaussian, Mitchell, Lanczos - need tabulated sampling
        # This path should not be called directly; use filter_sample_tabulated instead
        # Fallback to uniform sampling (high variance) if no sampler data available
        p = Point2f(
            lerp(-params.radius[1], params.radius[1], u[1]),
            lerp(-params.radius[2], params.radius[2], u[2])
        )
        weight = filter_evaluate(params, p)
        area = 4f0 * params.radius[1] * params.radius[2]
        return FilterSample(p, weight * area)
    end
end

"""
Sample filter with tabulated data for importance sampling.
Overload for when sampler_data is nothing (Box/Triangle filters).
"""
@inline function filter_sample(
    params::GPUFilterParams,
    sampler_data::Nothing,
    u::Point2f
)::FilterSample
    # Box and Triangle filters use analytical sampling
    filter_sample(params, u)
end

"""
Sample filter with tabulated data for importance sampling.
Overload for when sampler_data is GPUFilterSamplerData (Gaussian/Mitchell/Lanczos).
"""
@inline function filter_sample(
    params::GPUFilterParams,
    sampler_data::GPUFilterSamplerData,
    u::Point2f
)::FilterSample
    filter_type = params.filter_type

    if filter_type == Int32(1)
        # Box filter - uniform sampling, weight = 1
        p = Point2f(
            lerp(-params.radius[1], params.radius[1], u[1]),
            lerp(-params.radius[2], params.radius[2], u[2])
        )
        return FilterSample(p, 1f0)

    elseif filter_type == Int32(2)
        # Triangle filter - analytical sampling, weight = 1
        p = Point2f(
            sample_tent(u[1], params.radius[1]),
            sample_tent(u[2], params.radius[2])
        )
        return FilterSample(p, 1f0)

    else
        # Gaussian, Mitchell, Lanczos - use tabulated importance sampling
        return filter_sample_tabulated(sampler_data, u)
    end
end

"""
Evaluate filter at point p.
"""
@inline function filter_evaluate(params::GPUFilterParams, p::Point2f)::Float32
    filter_type = params.filter_type

    if filter_type == Int32(1)
        # Box
        return (abs(p[1]) <= params.radius[1] && abs(p[2]) <= params.radius[2]) ? 1f0 : 0f0

    elseif filter_type == Int32(2)
        # Triangle
        return max(0f0, params.radius[1] - abs(p[1])) * max(0f0, params.radius[2] - abs(p[2]))

    elseif filter_type == Int32(3)
        # Gaussian
        sigma = params.param1
        gx = max(0f0, gaussian_1d(p[1], sigma) - params.exp_x)
        gy = max(0f0, gaussian_1d(p[2], sigma) - params.exp_y)
        return gx * gy

    elseif filter_type == Int32(4)
        # Mitchell
        B, C = params.param1, params.param2
        return mitchell_1d(2f0 * p[1] / params.radius[1], B, C) *
               mitchell_1d(2f0 * p[2] / params.radius[2], B, C)

    elseif filter_type == Int32(5)
        # Lanczos
        τ = params.param1
        return windowed_sinc(p[1], params.radius[1], τ) *
               windowed_sinc(p[2], params.radius[2], τ)
    end

    return 0f0
end

# ============================================================================
# Convenience constructor for default filter
# ============================================================================

"""
    Filter(type::Symbol; kwargs...)

Create a filter by type name.

# Arguments
- `type`: One of :box, :triangle, :gaussian, :mitchell, :lanczos
- `kwargs`: Filter-specific parameters

# Examples
```julia
Filter(:box)                          # Default box filter
Filter(:triangle, radius=Point2f(2))  # Triangle with custom radius
Filter(:gaussian, sigma=0.5)          # Gaussian filter
Filter(:mitchell, B=1/3, C=1/3)       # Mitchell filter
Filter(:lanczos, tau=3)               # Lanczos sinc filter
```
"""
function Filter(type::Symbol; kwargs...)
    if type == :box
        radius = get(kwargs, :radius, Point2f(0.5f0, 0.5f0))
        return BoxFilter(radius)
    elseif type == :triangle
        radius = get(kwargs, :radius, Point2f(2f0, 2f0))
        return TriangleFilter(radius)
    elseif type == :gaussian
        radius = get(kwargs, :radius, Point2f(1.5f0, 1.5f0))
        sigma = Float32(get(kwargs, :sigma, 0.5f0))
        return GaussianFilter(radius, sigma)
    elseif type == :mitchell
        radius = get(kwargs, :radius, Point2f(2f0, 2f0))
        B = Float32(get(kwargs, :B, 1f0/3f0))
        C = Float32(get(kwargs, :C, 1f0/3f0))
        return MitchellFilter(radius, B, C)
    elseif type == :lanczos
        radius = get(kwargs, :radius, Point2f(4f0, 4f0))
        tau = Float32(get(kwargs, :tau, 3f0))
        return LanczosSincFilter(radius, tau)
    else
        error("Unknown filter type: $type. Use :box, :triangle, :gaussian, :mitchell, or :lanczos")
    end
end

# ============================================================================
# Adapt.jl Integration for GPU Kernels
# ============================================================================

"""
    Adapt.adapt_structure(to, data::GPUFilterSamplerData)

Adapt GPUFilterSamplerData for use inside GPU kernels. This converts the
GPU arrays (e.g., ROCArray) to device-compatible representations
(e.g., ROCDeviceArray) that can be used inside kernels.
"""
function Adapt.adapt_structure(to, data::GPUFilterSamplerData)
    GPUFilterSamplerData(
        Adapt.adapt(to, data.func),
        Adapt.adapt(to, data.marginal_cdf),
        Adapt.adapt(to, data.marginal_func),
        Adapt.adapt(to, data.conditional_cdf),
        data.domain_min,
        data.domain_max,
        data.nx,
        data.ny,
        data.func_integral
    )
end
