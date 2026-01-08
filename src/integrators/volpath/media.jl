# Core media types for volumetric path tracing
# Based on pbrt-v4's participating media implementation

# ============================================================================
# Phase Function (Henyey-Greenstein)
# ============================================================================

"""
    HGPhaseFunction

Henyey-Greenstein phase function for anisotropic scattering.
- g > 0: Forward scattering (clouds typically g ≈ 0.85)
- g = 0: Isotropic scattering
- g < 0: Backward scattering
"""
struct HGPhaseFunction
    g::Float32
end

HGPhaseFunction() = HGPhaseFunction(0f0)

"""
    hg_p(g, cos_θ) -> Float32

Evaluate Henyey-Greenstein phase function.
p(cos θ) = (1 - g²) / [4π(1 + g² - 2g cos θ)^(3/2)]
"""
@inline function hg_p(g::Float32, cos_θ::Float32)::Float32
    g2 = g * g
    denom = 1f0 + g2 - 2f0 * g * cos_θ
    return (1f0 - g2) / (4f0 * Float32(π) * denom * sqrt(denom))
end

@inline hg_p(phase::HGPhaseFunction, cos_θ::Float32) = hg_p(phase.g, cos_θ)

"""
    sample_hg(g, wo, u) -> (wi, pdf)

Importance sample the Henyey-Greenstein phase function.
Returns sampled direction and PDF.
"""
@inline function sample_hg(g::Float32, wo::Vec3f, u::Point2f)
    # Sample cos_θ from HG distribution
    cos_θ = if abs(g) < 1f-3
        # Isotropic case
        1f0 - 2f0 * u[1]
    else
        # Anisotropic case
        g2 = g * g
        sqr_term = (1f0 - g2) / (1f0 - g + 2f0 * g * u[1])
        cos_θ = (1f0 + g2 - sqr_term * sqr_term) / (2f0 * g)
        clamp(cos_θ, -1f0, 1f0)
    end

    # Compute sin_θ
    sin_θ = sqrt(max(0f0, 1f0 - cos_θ * cos_θ))

    # Sample azimuthal angle
    ϕ = 2f0 * Float32(π) * u[2]

    # Build local coordinate system around wo
    # We want wi such that dot(wo, wi) = cos_θ
    # So we build frame where wo is the z-axis
    t1, t2 = coordinate_system(-wo)

    # Compute wi in world space
    wi = sin_θ * cos(ϕ) * t1 + sin_θ * sin(ϕ) * t2 + cos_θ * (-wo)
    wi = normalize(wi)

    # PDF equals phase function value
    pdf = hg_p(g, cos_θ)

    return (wi, pdf)
end

@inline sample_hg(phase::HGPhaseFunction, wo::Vec3f, u::Point2f) = sample_hg(phase.g, wo, u)

# ============================================================================
# Medium Properties (returned at sample points)
# ============================================================================

"""
    MediumProperties

Properties of a participating medium at a specific point.
Returned by medium sampling functions.
"""
struct MediumProperties
    σ_a::SpectralRadiance      # Absorption coefficient
    σ_s::SpectralRadiance      # Scattering coefficient
    Le::SpectralRadiance       # Emission (for emissive media)
    g::Float32                 # HG asymmetry parameter
end

MediumProperties() = MediumProperties(
    SpectralRadiance(0f0),
    SpectralRadiance(0f0),
    SpectralRadiance(0f0),
    0f0
)

@inline σ_t(mp::MediumProperties) = mp.σ_a + mp.σ_s

# ============================================================================
# Ray Majorant Segment
# ============================================================================

"""
    RayMajorantSegment

A segment along a ray with a majorant (upper bound) extinction coefficient.
Used for delta tracking in heterogeneous media.
"""
struct RayMajorantSegment
    t_min::Float32
    t_max::Float32
    σ_maj::SpectralRadiance    # Majorant extinction coefficient
end

RayMajorantSegment() = RayMajorantSegment(0f0, 0f0, SpectralRadiance(0f0))

# ============================================================================
# Medium Interaction
# ============================================================================

"""
    MediumInteraction

Represents an interaction point within a participating medium.
Created when a real scattering event occurs during delta tracking.
"""
struct MediumInteraction
    p::Point3f                 # Position
    wo::Vec3f                  # Outgoing direction (toward camera/previous vertex)
    time::Float32
    g::Float32                 # HG parameter at this point
end

MediumInteraction() = MediumInteraction(
    Point3f(0f0, 0f0, 0f0),
    Vec3f(0f0, 0f0, 1f0),
    0f0,
    0f0
)

# ============================================================================
# Medium Index Type
# ============================================================================

# MediumIndex and MediumInterface are defined in materials/medium-interface.jl
# (included earlier to support spectral-eval.jl BSDF forwarding)

# ============================================================================
# Abstract Medium Type
# ============================================================================

abstract type Medium end

# ============================================================================
# Homogeneous Medium
# ============================================================================

"""
    HomogeneousMedium

A participating medium with constant properties throughout.
Simplest medium type - majorant equals actual extinction everywhere.
"""
struct HomogeneousMedium <: Medium
    σ_a::RGBSpectrum           # Absorption coefficient (RGB, uplifted at sample time)
    σ_s::RGBSpectrum           # Scattering coefficient (RGB)
    Le::RGBSpectrum            # Emission (RGB)
    g::Float32                 # HG asymmetry parameter
end

function HomogeneousMedium(;
    σ_a::RGBSpectrum = RGBSpectrum(0.01f0),
    σ_s::RGBSpectrum = RGBSpectrum(1f0),
    Le::RGBSpectrum = RGBSpectrum(0f0),
    g::Float32 = 0f0
)
    HomogeneousMedium(σ_a, σ_s, Le, g)
end

"""Check if medium has emission"""
@inline is_emissive(m::HomogeneousMedium) = !is_black(m.Le)

"""
    sample_point(table, medium::HomogeneousMedium, p, λ) -> MediumProperties

Sample medium properties at a point. For homogeneous media, this is constant.
"""
@inline function sample_point(
    table::RGBToSpectrumTable,
    medium::HomogeneousMedium,
    p::Point3f,
    λ::Wavelengths
)::MediumProperties
    # Use unbounded uplift for scattering/absorption coefficients (can be > 1.0)
    σ_a = uplift_rgb_unbounded(table, medium.σ_a, λ)
    σ_s = uplift_rgb_unbounded(table, medium.σ_s, λ)
    Le = uplift_rgb_unbounded(table, medium.Le, λ)
    return MediumProperties(σ_a, σ_s, Le, medium.g)
end

"""
    get_majorant(table, medium::HomogeneousMedium, ray, t_min, t_max, λ) -> RayMajorantSegment

Get majorant for ray segment. For homogeneous media, majorant = σ_t everywhere.
"""
@inline function get_majorant(
    table::RGBToSpectrumTable,
    medium::HomogeneousMedium,
    ray::Raycore.Ray,
    t_min::Float32,
    t_max::Float32,
    λ::Wavelengths
)::RayMajorantSegment
    # Use unbounded uplift for scattering/absorption coefficients (can be > 1.0)
    σ_a = uplift_rgb_unbounded(table, medium.σ_a, λ)
    σ_s = uplift_rgb_unbounded(table, medium.σ_s, λ)
    σ_maj = σ_a + σ_s
    return RayMajorantSegment(t_min, t_max, σ_maj)
end

# ============================================================================
# Grid Medium (Heterogeneous)
# ============================================================================

"""
    MajorantGrid

Coarse grid storing maximum extinction coefficients for DDA traversal.
Used to provide tight majorant bounds in heterogeneous media.
"""
struct MajorantGrid{T<:AbstractVector{Float32}}
    voxels::T                  # Flat array of max density multipliers
    res::Vec3i                 # Grid resolution
end

function MajorantGrid(res::Vec3i, alloc=Vector{Float32})
    n = res[1] * res[2] * res[3]
    voxels = alloc(undef, n)
    fill!(voxels, 0f0)
    MajorantGrid(voxels, res)
end

@inline function majorant_lookup(grid::MajorantGrid, x::Int, y::Int, z::Int)::Float32
    @inbounds grid.voxels[x + grid.res[1] * (y + grid.res[2] * z) + 1]
end

@inline function majorant_set!(grid::MajorantGrid, x::Int, y::Int, z::Int, v::Float32)
    @inbounds grid.voxels[x + grid.res[1] * (y + grid.res[2] * z) + 1] = v
end

"""
    GridMedium

A participating medium with spatially varying density.
Uses a 3D grid for density and a coarser majorant grid for efficient sampling.
"""
struct GridMedium{T<:AbstractArray{Float32,3}, M<:MajorantGrid} <: Medium
    # Volume bounds (in medium space, typically unit cube)
    bounds::Bounds3

    # Transform from render space to medium space
    render_to_medium::Mat4f
    medium_to_render::Mat4f

    # Base optical properties (scaled by density)
    σ_a::RGBSpectrum
    σ_s::RGBSpectrum

    # 3D density field (values typically 0-1)
    density::T

    # Phase function asymmetry
    g::Float32

    # Majorant grid for efficient sampling
    majorant_grid::M

    # Precomputed max density for quick majorant
    max_density::Float32
end

function GridMedium(
    density::AbstractArray{Float32,3};
    σ_a::RGBSpectrum = RGBSpectrum(0.01f0),
    σ_s::RGBSpectrum = RGBSpectrum(1f0),
    g::Float32 = 0f0,
    bounds::Bounds3 = Bounds3(Point3f(0f0, 0f0, 0f0), Point3f(1f0, 1f0, 1f0)),
    transform::Mat4f = Mat4f(I),
    majorant_res::Vec3i = Vec3i(16, 16, 16)
)
    # Compute inverse transform
    inv_transform = inv(transform)

    # Compute max density
    max_density = Float32(maximum(density))

    # Build majorant grid
    majorant_grid = build_majorant_grid(density, majorant_res)

    GridMedium(
        bounds,
        inv_transform,
        transform,
        σ_a,
        σ_s,
        density,
        g,
        majorant_grid,
        max_density
    )
end

"""Build a coarse majorant grid from the density field"""
function build_majorant_grid(density::AbstractArray{Float32,3}, res::Vec3i)
    nx, ny, nz = size(density)
    grid = MajorantGrid(res, Vector{Float32})

    # For each majorant voxel, find max density in corresponding region
    for iz in 0:res[3]-1
        z_start = 1 + (iz * nz) ÷ res[3]
        z_end = ((iz + 1) * nz) ÷ res[3]

        for iy in 0:res[2]-1
            y_start = 1 + (iy * ny) ÷ res[2]
            y_end = ((iy + 1) * ny) ÷ res[2]

            for ix in 0:res[1]-1
                x_start = 1 + (ix * nx) ÷ res[1]
                x_end = ((ix + 1) * nx) ÷ res[1]

                # Find max in this region
                max_val = 0f0
                for z in z_start:z_end, y in y_start:y_end, x in x_start:x_end
                    @inbounds max_val = max(max_val, density[x, y, z])
                end

                majorant_set!(grid, ix, iy, iz, max_val)
            end
        end
    end

    return grid
end

@inline is_emissive(::GridMedium) = false

"""Sample density at a point using trilinear interpolation"""
@inline function sample_density(medium::GridMedium, p_medium::Point3f)::Float32
    # Normalize to [0,1] within bounds
    p_norm = (p_medium - medium.bounds.p_min) ./ (medium.bounds.p_max - medium.bounds.p_min)

    # Check bounds
    if any(p_norm .< 0f0) || any(p_norm .> 1f0)
        return 0f0
    end

    # Grid coordinates
    nx, ny, nz = size(medium.density)
    gx = p_norm[1] * (nx - 1) + 1
    gy = p_norm[2] * (ny - 1) + 1
    gz = p_norm[3] * (nz - 1) + 1

    # Integer indices
    ix = clamp(floor(Int, gx), 1, nx - 1)
    iy = clamp(floor(Int, gy), 1, ny - 1)
    iz = clamp(floor(Int, gz), 1, nz - 1)

    # Fractional parts
    fx = Float32(gx - ix)
    fy = Float32(gy - iy)
    fz = Float32(gz - iz)

    # Trilinear interpolation
    @inbounds begin
        d000 = medium.density[ix, iy, iz]
        d100 = medium.density[ix+1, iy, iz]
        d010 = medium.density[ix, iy+1, iz]
        d110 = medium.density[ix+1, iy+1, iz]
        d001 = medium.density[ix, iy, iz+1]
        d101 = medium.density[ix+1, iy, iz+1]
        d011 = medium.density[ix, iy+1, iz+1]
        d111 = medium.density[ix+1, iy+1, iz+1]
    end

    fx1 = 1f0 - fx
    d00 = d000 * fx1 + d100 * fx
    d10 = d010 * fx1 + d110 * fx
    d01 = d001 * fx1 + d101 * fx
    d11 = d011 * fx1 + d111 * fx

    fy1 = 1f0 - fy
    d0 = d00 * fy1 + d10 * fy
    d1 = d01 * fy1 + d11 * fy

    return d0 * (1f0 - fz) + d1 * fz
end

"""Sample medium properties at a point"""
@inline function sample_point(
    table::RGBToSpectrumTable,
    medium::GridMedium,
    p::Point3f,
    λ::Wavelengths
)::MediumProperties
    # Transform to medium space
    p_medium = Point3f((medium.render_to_medium * Vec4f(p..., 1f0))[1:3])

    # Sample density
    d = sample_density(medium, p_medium)

    # Scale coefficients by density
    # Use unbounded uplift for scattering/absorption coefficients (can be > 1.0)
    σ_a = uplift_rgb_unbounded(table, medium.σ_a, λ) * d
    σ_s = uplift_rgb_unbounded(table, medium.σ_s, λ) * d

    return MediumProperties(σ_a, σ_s, SpectralRadiance(0f0), medium.g)
end

"""Get majorant for ray segment (conservative, uses max density)"""
@inline function get_majorant(
    table::RGBToSpectrumTable,
    medium::GridMedium,
    ray::Raycore.Ray,
    t_min::Float32,
    t_max::Float32,
    λ::Wavelengths
)::RayMajorantSegment
    # Use global max density for simplicity
    # TODO: Implement DDA traversal for tighter bounds
    # Use unbounded uplift for scattering/absorption coefficients (can be > 1.0)
    σ_a = uplift_rgb_unbounded(table, medium.σ_a, λ)
    σ_s = uplift_rgb_unbounded(table, medium.σ_s, λ)
    σ_maj = (σ_a + σ_s) * medium.max_density

    return RayMajorantSegment(t_min, t_max, σ_maj)
end

