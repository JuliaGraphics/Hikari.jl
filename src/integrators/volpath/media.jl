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
@propagate_inbounds function hg_p(g::Float32, cos_θ::Float32)::Float32
    g2 = g * g
    denom = 1f0 + g2 - 2f0 * g * cos_θ
    return (1f0 - g2) / (4f0 * Float32(π) * denom * sqrt(denom))
end

@propagate_inbounds hg_p(phase::HGPhaseFunction, cos_θ::Float32) = hg_p(phase.g, cos_θ)

"""
    sample_hg(g, wo, u) -> (wi, pdf)

Importance sample the Henyey-Greenstein phase function.
Returns sampled direction and PDF.
"""
@propagate_inbounds function sample_hg(g::Float32, wo::Vec3f, u::Point2f)
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

@propagate_inbounds sample_hg(phase::HGPhaseFunction, wo::Vec3f, u::Point2f) = sample_hg(phase.g, wo, u)

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

@propagate_inbounds σ_t(mp::MediumProperties) = mp.σ_a + mp.σ_s

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
# Homogeneous Majorant Iterator (trivial single-segment iterator)
# ============================================================================

"""
    HomogeneousMajorantIterator

Simple iterator that returns a single majorant segment for homogeneous media.
Provides consistent iterator interface with DDAMajorantIterator.
"""
struct HomogeneousMajorantIterator
    t_min::Float32
    t_max::Float32
    σ_maj::SpectralRadiance
    called::Bool  # Has the segment been returned?
end

HomogeneousMajorantIterator() = HomogeneousMajorantIterator(Inf32, -Inf32, SpectralRadiance(0f0), true)

function HomogeneousMajorantIterator(t_min::Float32, t_max::Float32, σ_maj::SpectralRadiance)
    HomogeneousMajorantIterator(t_min, t_max, σ_maj, false)
end

"""
    homogeneous_next(iter::HomogeneousMajorantIterator) -> (RayMajorantSegment, HomogeneousMajorantIterator) or nothing

Return the single majorant segment for homogeneous media.
Returns `nothing` on second call (iterator exhausted).
"""
@propagate_inbounds function homogeneous_next(iter::HomogeneousMajorantIterator)
    if iter.called || iter.t_min >= iter.t_max
        return nothing
    end

    seg = RayMajorantSegment(iter.t_min, iter.t_max, iter.σ_maj)
    new_iter = HomogeneousMajorantIterator(iter.t_min, iter.t_max, iter.σ_maj, true)
    return (seg, new_iter)
end

# ============================================================================
# Majorant Grid (forward declaration for DDA iterator)
# ============================================================================

"""
    MajorantGrid

Coarse grid storing maximum extinction coefficients for DDA traversal.
Used to provide tight majorant bounds in heterogeneous media.

Note: Uses Vec{3, Int32} instead of Vec3i (Int64) for GPU compatibility.
"""
struct MajorantGrid{T<:AbstractVector{Float32}}
    voxels::T                  # Flat array of max density multipliers
    res::Vec{3, Int32}         # Grid resolution (Int32 for GPU)
end

function MajorantGrid(res::Vec{3, Int32}, alloc=Vector{Float32})
    n = Int(res[1]) * Int(res[2]) * Int(res[3])
    voxels = alloc(undef, n)
    fill!(voxels, 0f0)
    MajorantGrid(voxels, res)
end

# Convenience constructor from Vec3i (Int64)
function MajorantGrid(res::Vec3i, alloc=Vector{Float32})
    MajorantGrid(Vec{3, Int32}(Int32(res[1]), Int32(res[2]), Int32(res[3])), alloc)
end

@propagate_inbounds function majorant_lookup(grid::MajorantGrid, x::Integer, y::Integer, z::Integer)::Float32
     # Use Int32 arithmetic for GPU compatibility
     idx = Int32(x) + grid.res[1] * (Int32(y) + grid.res[2] * Int32(z)) + Int32(1)
     grid.voxels[idx]
end

@propagate_inbounds function majorant_set!(grid::MajorantGrid, x::Int, y::Int, z::Int, v::Float32)
     grid.voxels[x + grid.res[1] * (y + grid.res[2] * z) + 1] = v
end

# ============================================================================
# DDA Majorant Iterator (for heterogeneous media)
# ============================================================================

"""
    DDAMajorantIterator

3D DDA iterator for traversing voxels along a ray through a majorant grid.
Returns per-voxel majorant bounds for tight delta tracking.

Following PBRT-v4's DDAMajorantIterator implementation.
The iterator state is immutable - dda_next returns a new iterator with updated state.

Note: Uses NTuple instead of Vec types for GPU compatibility.
"""
struct DDAMajorantIterator{M<:MajorantGrid}
    # Base extinction coefficient (σ_a + σ_s at full density)
    σ_t::SpectralRadiance
    # Current position and bounds along ray
    t_min::Float32
    t_max::Float32
    # Majorant grid reference
    grid::M
    grid_res::NTuple{3, Int32}
    # DDA state: next crossing t for each axis
    next_crossing_t::NTuple{3, Float32}
    # DDA state: step increment per voxel
    delta_t::NTuple{3, Float32}
    # DDA state: step direction per axis (-1 or +1)
    step::NTuple{3, Int32}
    # DDA state: limit voxel (exclusive) per axis
    voxel_limit::NTuple{3, Int32}
    # DDA state: current voxel indices
    voxel::NTuple{3, Int32}
end

"""Create an empty/invalid DDA iterator (for rays that miss the grid)"""
@inline function DDAMajorantIterator(grid::M) where {M<:MajorantGrid}
    res = grid.res
    DDAMajorantIterator{M}(
        SpectralRadiance(0f0),
        Inf32, -Inf32,  # t_min > t_max means empty
        grid,
        (Int32(res[1]), Int32(res[2]), Int32(res[3])),
        (0f0, 0f0, 0f0),
        (0f0, 0f0, 0f0),
        (Int32(0), Int32(0), Int32(0)),
        (Int32(0), Int32(0), Int32(0)),
        (Int32(0), Int32(0), Int32(0))
    )
end

"""
    create_dda_iterator(grid, bounds, ray_o, ray_d, t_min, t_max, σ_t) -> DDAMajorantIterator

Initialize a DDA iterator for traversing the majorant grid along a ray.
The ray should already be transformed to medium space.
t_min and t_max are the ray segment bounds (in medium space).

Following PBRT-v4's DDAMajorantIterator constructor.
"""
@inline @propagate_inbounds function create_dda_iterator(
    grid::M,
    bounds::Bounds3,
    ray_o::Point3f,
    ray_d::Vec3f,
    t_min::Float32,
    t_max::Float32,
    σ_t::SpectralRadiance
) where {M<:MajorantGrid}
    res = grid.res
    res_x = Int32(res[1])
    res_y = Int32(res[2])
    res_z = Int32(res[3])

    # Transform ray to normalized grid space [0,1]³
    # In PBRT: rayGrid has origin at bounds.Offset(ray.o) and direction scaled by diagonal
    diag = bounds.p_max - bounds.p_min

    # Offset: maps world point to [0,1]³ within bounds
    grid_o_x = (ray_o[1] - bounds.p_min[1]) / diag[1]
    grid_o_y = (ray_o[2] - bounds.p_min[2]) / diag[2]
    grid_o_z = (ray_o[3] - bounds.p_min[3]) / diag[3]

    # Scale direction by inverse diagonal (so stepping 1 unit in grid space = crossing bounds)
    # Avoid division by zero for flat bounds
    inv_diag_x = abs(diag[1]) > 1f-10 ? 1f0 / diag[1] : 0f0
    inv_diag_y = abs(diag[2]) > 1f-10 ? 1f0 / diag[2] : 0f0
    inv_diag_z = abs(diag[3]) > 1f-10 ? 1f0 / diag[3] : 0f0

    grid_d_x = ray_d[1] * inv_diag_x
    grid_d_y = ray_d[2] * inv_diag_y
    grid_d_z = ray_d[3] * inv_diag_z

    # Compute grid intersection point at t_min
    grid_intersect_x = grid_o_x + grid_d_x * t_min
    grid_intersect_y = grid_o_y + grid_d_y * t_min
    grid_intersect_z = grid_o_z + grid_d_z * t_min

    # Initialize per-axis DDA state (use floor_int32 for GPU compatibility)
    voxel_x = clamp(floor_int32(grid_intersect_x * res_x), Int32(0), res_x - Int32(1))
    voxel_y = clamp(floor_int32(grid_intersect_y * res_y), Int32(0), res_y - Int32(1))
    voxel_z = clamp(floor_int32(grid_intersect_z * res_z), Int32(0), res_z - Int32(1))

    # Per-axis: deltaT = 1 / (|d| * res) = distance along ray to cross one voxel
    delta_t_x = abs(grid_d_x) > 1f-10 ? 1f0 / (abs(grid_d_x) * res_x) : Inf32
    delta_t_y = abs(grid_d_y) > 1f-10 ? 1f0 / (abs(grid_d_y) * res_y) : Inf32
    delta_t_z = abs(grid_d_z) > 1f-10 ? 1f0 / (abs(grid_d_z) * res_z) : Inf32

    # Per-axis: step direction and voxel limit
    # Also compute nextCrossingT: t value when we exit current voxel

    # X axis
    next_crossing_t_x::Float32 = 0f0
    step_x::Int32 = Int32(0)
    voxel_limit_x::Int32 = Int32(0)
    if grid_d_x >= 0f0
        next_voxel_pos_x = Float32(voxel_x + Int32(1)) / Float32(res_x)
        next_crossing_t_x = grid_d_x > 1f-10 ?
            t_min + (next_voxel_pos_x - grid_intersect_x) / grid_d_x : Inf32
        step_x = Int32(1)
        voxel_limit_x = res_x
    else
        next_voxel_pos_x = Float32(voxel_x) / Float32(res_x)
        next_crossing_t_x = grid_d_x < -1f-10 ?
            t_min + (next_voxel_pos_x - grid_intersect_x) / grid_d_x : Inf32
        step_x = Int32(-1)
        voxel_limit_x = Int32(-1)
    end

    # Y axis
    next_crossing_t_y::Float32 = 0f0
    step_y::Int32 = Int32(0)
    voxel_limit_y::Int32 = Int32(0)
    if grid_d_y >= 0f0
        next_voxel_pos_y = Float32(voxel_y + Int32(1)) / Float32(res_y)
        next_crossing_t_y = grid_d_y > 1f-10 ?
            t_min + (next_voxel_pos_y - grid_intersect_y) / grid_d_y : Inf32
        step_y = Int32(1)
        voxel_limit_y = res_y
    else
        next_voxel_pos_y = Float32(voxel_y) / Float32(res_y)
        next_crossing_t_y = grid_d_y < -1f-10 ?
            t_min + (next_voxel_pos_y - grid_intersect_y) / grid_d_y : Inf32
        step_y = Int32(-1)
        voxel_limit_y = Int32(-1)
    end

    # Z axis
    next_crossing_t_z::Float32 = 0f0
    step_z::Int32 = Int32(0)
    voxel_limit_z::Int32 = Int32(0)
    if grid_d_z >= 0f0
        next_voxel_pos_z = Float32(voxel_z + Int32(1)) / Float32(res_z)
        next_crossing_t_z = grid_d_z > 1f-10 ?
            t_min + (next_voxel_pos_z - grid_intersect_z) / grid_d_z : Inf32
        step_z = Int32(1)
        voxel_limit_z = res_z
    else
        next_voxel_pos_z = Float32(voxel_z) / Float32(res_z)
        next_crossing_t_z = grid_d_z < -1f-10 ?
            t_min + (next_voxel_pos_z - grid_intersect_z) / grid_d_z : Inf32
        step_z = Int32(-1)
        voxel_limit_z = Int32(-1)
    end

    DDAMajorantIterator{M}(
        σ_t,
        t_min, t_max,
        grid,
        (res_x, res_y, res_z),
        (next_crossing_t_x, next_crossing_t_y, next_crossing_t_z),
        (delta_t_x, delta_t_y, delta_t_z),
        (step_x, step_y, step_z),
        (voxel_limit_x, voxel_limit_y, voxel_limit_z),
        (voxel_x, voxel_y, voxel_z)
    )
end

"""
    dda_next(iter::DDAMajorantIterator) -> (RayMajorantSegment, DDAMajorantIterator) or nothing

Advance the DDA iterator and return the next majorant segment.
Returns `nothing` when iteration is complete (t_min >= t_max).

Since structs are immutable in Julia, this returns a new iterator with updated state.
Following PBRT-v4's DDAMajorantIterator::Next().
"""
@inline @propagate_inbounds function dda_next(iter::DDAMajorantIterator{M}) where M
    t_min = iter.t_min
    t_max = iter.t_max

    # Check if done
    if t_min >= t_max
        return nothing
    end

    next_crossing_t = iter.next_crossing_t
    voxel = iter.voxel

    # Find step axis: axis with smallest nextCrossingT
    # Using bit manipulation like PBRT for branchless selection
    # bits = (t_x < t_y) << 2 + (t_x < t_z) << 1 + (t_y < t_z)
    t_x, t_y, t_z = next_crossing_t[1], next_crossing_t[2], next_crossing_t[3]
    bits = ((t_x < t_y) ? 4 : 0) + ((t_x < t_z) ? 2 : 0) + ((t_y < t_z) ? 1 : 0)

    # Lookup table: cmpToAxis[bits] gives the axis to step
    # cmpToAxis = [2, 1, 2, 1, 2, 2, 0, 0]  (0-indexed in C++)
    # Julia 1-indexed: axis 1=X, 2=Y, 3=Z
    step_axis = if bits == 0
        Int32(3)  # Z axis
    elseif bits == 1
        Int32(2)  # Y axis
    elseif bits == 2
        Int32(3)  # Z axis
    elseif bits == 3
        Int32(2)  # Y axis
    elseif bits == 4
        Int32(3)  # Z axis
    elseif bits == 5
        Int32(3)  # Z axis
    elseif bits == 6
        Int32(1)  # X axis
    else  # bits == 7
        Int32(1)  # X axis
    end

    # Exit t for this voxel
    t_voxel_exit = min(t_max, next_crossing_t[step_axis])

    # Lookup majorant density for current voxel
    max_density = majorant_lookup(iter.grid, voxel[1], voxel[2], voxel[3])
    σ_maj = iter.σ_t * max_density

    # Create segment
    seg = RayMajorantSegment(t_min, t_voxel_exit, σ_maj)

    # Advance state for next iteration
    new_t_min = t_voxel_exit

    # Check if we're exiting the grid on this axis - compute new voxel
    new_voxel_x = step_axis == Int32(1) ? voxel[1] + iter.step[1] : voxel[1]
    new_voxel_y = step_axis == Int32(2) ? voxel[2] + iter.step[2] : voxel[2]
    new_voxel_z = step_axis == Int32(3) ? voxel[3] + iter.step[3] : voxel[3]

    # If stepped voxel hits limit, we're done
    new_voxel_axis = step_axis == Int32(1) ? new_voxel_x : (step_axis == Int32(2) ? new_voxel_y : new_voxel_z)
    voxel_limit_axis = iter.voxel_limit[step_axis]
    if new_voxel_axis == voxel_limit_axis
        new_t_min = t_max
    end

    # Also terminate if next crossing exceeds t_max
    if next_crossing_t[step_axis] > t_max
        new_t_min = t_max
    end

    # Update next crossing t
    new_next_crossing_t = (
        step_axis == Int32(1) ? next_crossing_t[1] + iter.delta_t[1] : next_crossing_t[1],
        step_axis == Int32(2) ? next_crossing_t[2] + iter.delta_t[2] : next_crossing_t[2],
        step_axis == Int32(3) ? next_crossing_t[3] + iter.delta_t[3] : next_crossing_t[3]
    )

    # Create new iterator with updated state
    new_iter = DDAMajorantIterator{M}(
        iter.σ_t,
        new_t_min, t_max,
        iter.grid,
        iter.grid_res,
        new_next_crossing_t,
        iter.delta_t,
        iter.step,
        iter.voxel_limit,
        (new_voxel_x, new_voxel_y, new_voxel_z)
    )

    return (seg, new_iter)
end

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
@propagate_inbounds is_emissive(m::HomogeneousMedium) = !is_black(m.Le)

"""
    sample_point(table, medium::HomogeneousMedium, p, λ) -> MediumProperties

Sample medium properties at a point. For homogeneous media, this is constant.
"""
@propagate_inbounds function sample_point(
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
@propagate_inbounds function get_majorant(
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

"""
    create_majorant_iterator(table, medium::HomogeneousMedium, ray, t_max, λ) -> HomogeneousMajorantIterator

Create a majorant iterator for homogeneous medium (single segment).
"""
@propagate_inbounds function create_majorant_iterator(
    table::RGBToSpectrumTable,
    medium::HomogeneousMedium,
    ray::Raycore.Ray,
    t_max::Float32,
    λ::Wavelengths
)::HomogeneousMajorantIterator
    σ_a = uplift_rgb_unbounded(table, medium.σ_a, λ)
    σ_s = uplift_rgb_unbounded(table, medium.σ_s, λ)
    σ_maj = σ_a + σ_s
    return HomogeneousMajorantIterator(0f0, t_max, σ_maj)
end

# ============================================================================
# Grid Medium (Heterogeneous)
# ============================================================================

"""
    GridMedium

A participating medium with spatially varying density.
Uses a 3D grid for density and a coarser majorant grid for efficient sampling.

Note: Uses Vec{3, Int32} for density_res for GPU compatibility.
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

    # Density grid resolution (stored to avoid size() call on GPU, Int32 for GPU compatibility)
    density_res::Vec{3, Int32}

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

    # Store density resolution to avoid size() call on GPU (Int32 for GPU compatibility)
    nx, ny, nz = size(density)
    density_res = Vec{3, Int32}(Int32(nx), Int32(ny), Int32(nz))

    GridMedium(
        bounds,
        inv_transform,
        transform,
        σ_a,
        σ_s,
        density,
        density_res,
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
                     max_val = max(max_val, density[x, y, z])
                end

                majorant_set!(grid, ix, iy, iz, max_val)
            end
        end
    end

    return grid
end

@propagate_inbounds is_emissive(::GridMedium) = false

"""Sample density at a point using trilinear interpolation"""
@propagate_inbounds function sample_density(medium::GridMedium, p_medium::Point3f)::Float32
    # Normalize to [0,1] within bounds
    p_norm = (p_medium - medium.bounds.p_min) ./ (medium.bounds.p_max - medium.bounds.p_min)

    # Check bounds (explicit element-wise check for GPU compatibility - avoid any() which creates arrays)
    if p_norm[1] < 0f0 || p_norm[2] < 0f0 || p_norm[3] < 0f0 ||
       p_norm[1] > 1f0 || p_norm[2] > 1f0 || p_norm[3] > 1f0
        return 0f0
    end

    # Grid coordinates (use stored resolution to avoid size() on GPU)
    # Use Int32 literals to avoid promotion to Int64
    nx, ny, nz = medium.density_res[1], medium.density_res[2], medium.density_res[3]
    gx = p_norm[1] * (nx - Int32(1)) + Int32(1)
    gy = p_norm[2] * (ny - Int32(1)) + Int32(1)
    gz = p_norm[3] * (nz - Int32(1)) + Int32(1)

    # Integer indices (use floor_int32 for GPU compatibility)
    ix = clamp(floor_int32(gx), Int32(1), nx - Int32(1))
    iy = clamp(floor_int32(gy), Int32(1), ny - Int32(1))
    iz = clamp(floor_int32(gz), Int32(1), nz - Int32(1))

    # Fractional parts
    fx = Float32(gx - ix)
    fy = Float32(gy - iy)
    fz = Float32(gz - iz)

    # Trilinear interpolation (use Int32(1) to avoid promotion to Int64)
     begin
        d000 = medium.density[ix, iy, iz]
        d100 = medium.density[ix+Int32(1), iy, iz]
        d010 = medium.density[ix, iy+Int32(1), iz]
        d110 = medium.density[ix+Int32(1), iy+Int32(1), iz]
        d001 = medium.density[ix, iy, iz+Int32(1)]
        d101 = medium.density[ix+Int32(1), iy, iz+Int32(1)]
        d011 = medium.density[ix, iy+Int32(1), iz+Int32(1)]
        d111 = medium.density[ix+Int32(1), iy+Int32(1), iz+Int32(1)]
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
@propagate_inbounds function sample_point(
    table::RGBToSpectrumTable,
    medium::GridMedium,
    p::Point3f,
    λ::Wavelengths
)::MediumProperties
    # Transform to medium space using scalar operations (GPU-compatible)
    M = medium.render_to_medium
    p_x = M[1,1] * p[1] + M[1,2] * p[2] + M[1,3] * p[3] + M[1,4]
    p_y = M[2,1] * p[1] + M[2,2] * p[2] + M[2,3] * p[3] + M[2,4]
    p_z = M[3,1] * p[1] + M[3,2] * p[2] + M[3,3] * p[3] + M[3,4]
    p_medium = Point3f(p_x, p_y, p_z)

    # Sample density
    d = sample_density(medium, p_medium)

    # Scale coefficients by density
    # Use unbounded uplift for scattering/absorption coefficients (can be > 1.0)
    σ_a = uplift_rgb_unbounded(table, medium.σ_a, λ) * d
    σ_s = uplift_rgb_unbounded(table, medium.σ_s, λ) * d

    return MediumProperties(σ_a, σ_s, SpectralRadiance(0f0), medium.g)
end

"""
    create_majorant_iterator(table, medium::GridMedium, ray, t_max, λ) -> DDAMajorantIterator

Create a DDA majorant iterator for traversing the medium along a ray.
Following PBRT-v4's GridMedium::SampleRay pattern.

The ray is transformed to medium space and intersected with the bounds
to determine the valid segment for DDA traversal.
"""
@propagate_inbounds function create_majorant_iterator(
    table::RGBToSpectrumTable,
    medium::GridMedium,
    ray::Raycore.Ray,
    t_max::Float32,
    λ::Wavelengths
)
    # Transform ray to medium space using scalar operations (GPU-compatible)
    M = medium.render_to_medium

    # Transform origin (point) - M * [o, 1]
    ray_o_x = M[1,1] * ray.o[1] + M[1,2] * ray.o[2] + M[1,3] * ray.o[3] + M[1,4]
    ray_o_y = M[2,1] * ray.o[1] + M[2,2] * ray.o[2] + M[2,3] * ray.o[3] + M[2,4]
    ray_o_z = M[3,1] * ray.o[1] + M[3,2] * ray.o[2] + M[3,3] * ray.o[3] + M[3,4]
    ray_o = Point3f(ray_o_x, ray_o_y, ray_o_z)

    # Transform direction (vector, no translation) - M * [d, 0]
    ray_d_x = M[1,1] * ray.d[1] + M[1,2] * ray.d[2] + M[1,3] * ray.d[3]
    ray_d_y = M[2,1] * ray.d[1] + M[2,2] * ray.d[2] + M[2,3] * ray.d[3]
    ray_d_z = M[3,1] * ray.d[1] + M[3,2] * ray.d[2] + M[3,3] * ray.d[3]
    ray_d = Vec3f(ray_d_x, ray_d_y, ray_d_z)

    # Compute ray-bounds intersection in medium space
    # This gives us [t_enter, t_exit] where the ray is inside the medium
    t_enter, t_exit = ray_bounds_intersect(ray_o, ray_d, medium.bounds)

    # Clamp to requested range and check validity
    t_enter = max(t_enter, 0f0)
    t_exit = min(t_exit, t_max)

    if t_enter >= t_exit
        # Ray misses bounds or segment is empty - return empty iterator
        return DDAMajorantIterator(medium.majorant_grid)
    end

    # Compute base extinction coefficient (at full density)
    σ_a = uplift_rgb_unbounded(table, medium.σ_a, λ)
    σ_s = uplift_rgb_unbounded(table, medium.σ_s, λ)
    σ_t = σ_a + σ_s

    # Create the DDA iterator
    return create_dda_iterator(
        medium.majorant_grid,
        medium.bounds,
        ray_o,
        ray_d,
        t_enter,
        t_exit,
        σ_t
    )
end

"""
    ray_bounds_intersect(ray_o, ray_d, bounds) -> (t_min, t_max)

Compute ray-AABB intersection. Returns (Inf, -Inf) if no intersection.
Uses scalar operations for GPU compatibility.
"""
@inline @propagate_inbounds function ray_bounds_intersect(
    ray_o::Point3f,
    ray_d::Vec3f,
    bounds::Bounds3
)::Tuple{Float32, Float32}
    # Compute inverse direction with each slab (scalar operations for GPU)
    inv_d_x = abs(ray_d[1]) > 1f-10 ? 1f0 / ray_d[1] : (ray_d[1] >= 0 ? Inf32 : -Inf32)
    inv_d_y = abs(ray_d[2]) > 1f-10 ? 1f0 / ray_d[2] : (ray_d[2] >= 0 ? Inf32 : -Inf32)
    inv_d_z = abs(ray_d[3]) > 1f-10 ? 1f0 / ray_d[3] : (ray_d[3] >= 0 ? Inf32 : -Inf32)

    # X slab
    t0x = (bounds.p_min[1] - ray_o[1]) * inv_d_x
    t1x = (bounds.p_max[1] - ray_o[1]) * inv_d_x
    if t0x > t1x
        t0x, t1x = t1x, t0x
    end

    # Y slab
    t0y = (bounds.p_min[2] - ray_o[2]) * inv_d_y
    t1y = (bounds.p_max[2] - ray_o[2]) * inv_d_y
    if t0y > t1y
        t0y, t1y = t1y, t0y
    end

    # Z slab
    t0z = (bounds.p_min[3] - ray_o[3]) * inv_d_z
    t1z = (bounds.p_max[3] - ray_o[3]) * inv_d_z
    if t0z > t1z
        t0z, t1z = t1z, t0z
    end

    # Find overlap
    t_enter = max(t0x, t0y, t0z)
    t_exit = min(t1x, t1y, t1z)

    return (t_enter, t_exit)
end

"""Get majorant for ray segment (conservative, uses max density) - DEPRECATED for GridMedium"""
@propagate_inbounds function get_majorant(
    table::RGBToSpectrumTable,
    medium::GridMedium,
    ray::Raycore.Ray,
    t_min::Float32,
    t_max::Float32,
    λ::Wavelengths
)::RayMajorantSegment
    # Fallback: Use global max density for simplicity
    # NOTE: Use create_majorant_iterator for proper DDA-based majorant traversal
    σ_a = uplift_rgb_unbounded(table, medium.σ_a, λ)
    σ_s = uplift_rgb_unbounded(table, medium.σ_s, λ)
    σ_maj = (σ_a + σ_s) * medium.max_density

    return RayMajorantSegment(t_min, t_max, σ_maj)
end
