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
    homogeneous_next(iter::HomogeneousMajorantIterator) -> (RayMajorantSegment, HomogeneousMajorantIterator, Bool)

Return the single majorant segment for homogeneous media.
Returns (seg, new_iter, true) if valid, (invalid_seg, exhausted_iter, false) if exhausted.

The Bool indicates validity: true = has segment, false = exhausted.
"""
@propagate_inbounds function homogeneous_next(iter::HomogeneousMajorantIterator)
    if iter.called || iter.t_min >= iter.t_max
        # Return invalid/exhausted state
        invalid_seg = RayMajorantSegment()
        exhausted_iter = HomogeneousMajorantIterator()
        return (invalid_seg, exhausted_iter, false)
    end

    seg = RayMajorantSegment(iter.t_min, iter.t_max, iter.σ_maj)
    new_iter = HomogeneousMajorantIterator(iter.t_min, iter.t_max, iter.σ_maj, true)
    return (seg, new_iter, true)
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
# Template Grid Extraction (for mixed media type consistency)
# ============================================================================

# NOTE: get_template_grid and get_template_grid_from_tuple are defined after
# HomogeneousMedium and GridMedium structs (see end of file)

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
    dda_next(iter::DDAMajorantIterator) -> (RayMajorantSegment, DDAMajorantIterator, Bool)

Advance the DDA iterator and return the next majorant segment.
Returns (seg, new_iter, true) if valid, (invalid_seg, exhausted_iter, false) if exhausted.

The Bool indicates validity: true = has segment, false = exhausted.

Since structs are immutable in Julia, this returns a new iterator with updated state.
Following PBRT-v4's DDAMajorantIterator::Next().
"""
@inline @propagate_inbounds function dda_next(iter::DDAMajorantIterator{M}) where M
    t_min = iter.t_min
    t_max = iter.t_max

    # Check if done
    if t_min >= t_max
        invalid_seg = RayMajorantSegment()
        exhausted_iter = DDAMajorantIterator(iter.grid)
        return (invalid_seg, exhausted_iter, false)
    end

    next_crossing_t = iter.next_crossing_t
    voxel = iter.voxel

    # Find step axis: axis with smallest nextCrossingT
    # Using bit manipulation like PBRT for branchless selection
    # bits = (t_x < t_y) << 2 + (t_x < t_z) << 1 + (t_y < t_z)
    t_x, t_y, t_z = next_crossing_t[1], next_crossing_t[2], next_crossing_t[3]
    # IMPORTANT: Use Int32 literals to avoid Int64 which causes SPIR-V "Unsupported integer width" error
    # See spirv-int64-ifelse-mwe.jl for minimal reproduction case
    bits = ((t_x < t_y) ? Int32(4) : Int32(0)) + ((t_x < t_z) ? Int32(2) : Int32(0)) + ((t_y < t_z) ? Int32(1) : Int32(0))

    # Lookup table: cmpToAxis[bits] gives the axis to step
    # cmpToAxis = [2, 1, 2, 1, 2, 2, 0, 0]  (0-indexed in C++)
    # Julia 1-indexed: axis 1=X, 2=Y, 3=Z
    step_axis = if bits == Int32(0)
        Int32(3)  # Z axis
    elseif bits == Int32(1)
        Int32(2)  # Y axis
    elseif bits == Int32(2)
        Int32(3)  # Z axis
    elseif bits == Int32(3)
        Int32(2)  # Y axis
    elseif bits == Int32(4)
        Int32(3)  # Z axis
    elseif bits == Int32(5)
        Int32(3)  # Z axis
    elseif bits == Int32(6)
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

    return (seg, new_iter, true)
end

# ============================================================================
# Unified Ray Majorant Iterator (GPU-compatible variant type)
# ============================================================================

"""
    RayMajorantIterator

Unified majorant iterator that can represent either homogeneous or DDA iteration.
This avoids Union types which cause GPU compilation issues.

Following pbrt-v4's RayMajorantIterator which is a TaggedPointer variant.

Mode:
- mode = 0: Invalid/exhausted iterator
- mode = 1: Homogeneous mode (single segment)
- mode = 2: DDA mode (voxel traversal)
"""
struct RayMajorantIterator{M<:MajorantGrid}
    # Mode tag (0=invalid, 1=homogeneous, 2=DDA)
    mode::Int32

    # Shared fields
    σ_t::SpectralRadiance       # Majorant extinction
    t_min::Float32
    t_max::Float32

    # Homogeneous mode state
    hom_called::Bool            # Has single segment been returned?

    # DDA mode state
    grid::M
    grid_res::NTuple{3, Int32}
    next_crossing_t::NTuple{3, Float32}
    delta_t::NTuple{3, Float32}
    step::NTuple{3, Int32}
    voxel_limit::NTuple{3, Int32}
    voxel::NTuple{3, Int32}
end

# GPU-compatible empty grid constructor for homogeneous media (type placeholder)
# Uses SVector{1,Float32} which is stored inline (not a heap pointer), so it works on GPU.
# This is a function rather than a const to avoid GPU kernels referencing global memory.
@inline EmptyMajorantGrid() = MajorantGrid(SVector{1, Float32}(0f0), Vec{3, Int32}(Int32(1), Int32(1), Int32(1)))

"""Create an invalid/exhausted iterator"""
@inline function RayMajorantIterator(grid::M) where {M<:MajorantGrid}
    res = grid.res
    RayMajorantIterator{M}(
        Int32(0),  # mode = invalid
        SpectralRadiance(0f0),
        Inf32, -Inf32,
        true,
        grid,
        (Int32(res[1]), Int32(res[2]), Int32(res[3])),
        (0f0, 0f0, 0f0),
        (0f0, 0f0, 0f0),
        (Int32(0), Int32(0), Int32(0)),
        (Int32(0), Int32(0), Int32(0)),
        (Int32(0), Int32(0), Int32(0))
    )
end

"""Create a homogeneous mode iterator directly (for HomogeneousMedium)"""
@inline function RayMajorantIterator_homogeneous(t_min::Float32, t_max::Float32, σ_maj::SpectralRadiance)
    grid = EmptyMajorantGrid()
    mode = (t_min >= t_max) ? Int32(0) : Int32(1)
    RayMajorantIterator{typeof(grid)}(
        mode,
        σ_maj,
        t_min, t_max,
        false,  # not called yet
        grid,
        (Int32(1), Int32(1), Int32(1)),
        (0f0, 0f0, 0f0),
        (0f0, 0f0, 0f0),
        (Int32(0), Int32(0), Int32(0)),
        (Int32(0), Int32(0), Int32(0)),
        (Int32(0), Int32(0), Int32(0))
    )
end

"""Create a homogeneous mode iterator (from HomogeneousMajorantIterator)"""
@inline function RayMajorantIterator(hom::HomogeneousMajorantIterator, grid::M) where {M<:MajorantGrid}
    res = grid.res
    mode = (hom.called || hom.t_min >= hom.t_max) ? Int32(0) : Int32(1)
    RayMajorantIterator{M}(
        mode,
        hom.σ_maj,
        hom.t_min, hom.t_max,
        hom.called,
        grid,
        (Int32(res[1]), Int32(res[2]), Int32(res[3])),
        (0f0, 0f0, 0f0),
        (0f0, 0f0, 0f0),
        (Int32(0), Int32(0), Int32(0)),
        (Int32(0), Int32(0), Int32(0)),
        (Int32(0), Int32(0), Int32(0))
    )
end

"""Create a DDA mode iterator (from DDAMajorantIterator)"""
@inline function RayMajorantIterator(dda::DDAMajorantIterator{M}) where {M<:MajorantGrid}
    mode = (dda.t_min >= dda.t_max) ? Int32(0) : Int32(2)
    RayMajorantIterator{M}(
        mode,
        dda.σ_t,
        dda.t_min, dda.t_max,
        false,
        dda.grid,
        dda.grid_res,
        dda.next_crossing_t,
        dda.delta_t,
        dda.step,
        dda.voxel_limit,
        dda.voxel
    )
end

"""
    ray_majorant_next(iter::RayMajorantIterator) -> (RayMajorantSegment, RayMajorantIterator, Bool)

Advance the unified iterator and return the next majorant segment.
Dispatches internally based on mode (homogeneous vs DDA).
Returns (segment, new_iter, valid) where valid=false means exhausted.
"""
@inline @propagate_inbounds function ray_majorant_next(iter::RayMajorantIterator{M}) where {M}
    if iter.mode == Int32(0)
        # Invalid/exhausted
        return (RayMajorantSegment(), iter, false)

    elseif iter.mode == Int32(1)
        # Homogeneous mode - return single segment
        if iter.hom_called || iter.t_min >= iter.t_max
            new_iter = RayMajorantIterator{M}(
                Int32(0), iter.σ_t, iter.t_min, iter.t_max, true,
                iter.grid, iter.grid_res, iter.next_crossing_t, iter.delta_t,
                iter.step, iter.voxel_limit, iter.voxel
            )
            return (RayMajorantSegment(), new_iter, false)
        end
        seg = RayMajorantSegment(iter.t_min, iter.t_max, iter.σ_t)
        new_iter = RayMajorantIterator{M}(
            Int32(1), iter.σ_t, iter.t_min, iter.t_max, true,
            iter.grid, iter.grid_res, iter.next_crossing_t, iter.delta_t,
            iter.step, iter.voxel_limit, iter.voxel
        )
        return (seg, new_iter, true)

    else
        # DDA mode - voxel traversal
        return _ray_majorant_next_dda(iter)
    end
end

"""DDA next implementation (separated for clarity)"""
@inline @propagate_inbounds function _ray_majorant_next_dda(iter::RayMajorantIterator{M}) where {M}
    t_min = iter.t_min
    t_max = iter.t_max

    if t_min >= t_max
        new_iter = RayMajorantIterator{M}(
            Int32(0), iter.σ_t, Inf32, -Inf32, true,
            iter.grid, iter.grid_res, iter.next_crossing_t, iter.delta_t,
            iter.step, iter.voxel_limit, iter.voxel
        )
        return (RayMajorantSegment(), new_iter, false)
    end

    voxel_x, voxel_y, voxel_z = iter.voxel
    next_t_x, next_t_y, next_t_z = iter.next_crossing_t
    step_x, step_y, step_z = iter.step
    delta_x, delta_y, delta_z = iter.delta_t
    limit_x, limit_y, limit_z = iter.voxel_limit

    # Find which axis we exit first
    step_axis = (next_t_x < next_t_y) ?
        ((next_t_x < next_t_z) ? Int32(0) : Int32(2)) :
        ((next_t_y < next_t_z) ? Int32(1) : Int32(2))

    # Compute segment end (clamped to t_max)
    seg_t_max = if step_axis == Int32(0)
        min(next_t_x, t_max)
    elseif step_axis == Int32(1)
        min(next_t_y, t_max)
    else
        min(next_t_z, t_max)
    end

    # Get majorant for current voxel
    ρ = majorant_lookup(iter.grid, voxel_x, voxel_y, voxel_z)
    σ_maj = iter.σ_t * ρ
    seg = RayMajorantSegment(t_min, seg_t_max, σ_maj)

    # Update state for next iteration
    new_t_min = seg_t_max
    new_voxel_x = voxel_x
    new_voxel_y = voxel_y
    new_voxel_z = voxel_z
    new_next_t_x = next_t_x
    new_next_t_y = next_t_y
    new_next_t_z = next_t_z

    if step_axis == Int32(0)
        new_voxel_x = voxel_x + step_x
        new_next_t_x = next_t_x + delta_x
    elseif step_axis == Int32(1)
        new_voxel_y = voxel_y + step_y
        new_next_t_y = next_t_y + delta_y
    else
        new_voxel_z = voxel_z + step_z
        new_next_t_z = next_t_z + delta_z
    end

    # Check if we've exited the grid
    new_mode = Int32(2)
    if new_voxel_x == limit_x || new_voxel_y == limit_y || new_voxel_z == limit_z
        new_mode = Int32(0)  # Mark as exhausted
        new_t_min = t_max    # No more segments
    end

    new_iter = RayMajorantIterator{M}(
        new_mode, iter.σ_t, new_t_min, t_max, false,
        iter.grid, iter.grid_res,
        (new_next_t_x, new_next_t_y, new_next_t_z),
        iter.delta_t, iter.step, iter.voxel_limit,
        (new_voxel_x, new_voxel_y, new_voxel_z)
    )

    return (seg, new_iter, true)
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
    create_majorant_iterator(table, medium::HomogeneousMedium, ray, t_max, λ) -> RayMajorantIterator

Create a majorant iterator for homogeneous medium (single segment).
Returns unified RayMajorantIterator for GPU compatibility with mixed media.

NOTE: This version uses EMPTY_MAJORANT_GRID which has Vector{Float32} storage.
For mixed scenes with GridMedium on GPU, use the template_grid version instead.
"""
@propagate_inbounds function create_majorant_iterator(
    table::RGBToSpectrumTable,
    medium::HomogeneousMedium,
    ray::Raycore.Ray,
    t_max::Float32,
    λ::Wavelengths
)
    σ_a = uplift_rgb_unbounded(table, medium.σ_a, λ)
    σ_s = uplift_rgb_unbounded(table, medium.σ_s, λ)
    σ_maj = σ_a + σ_s
    return RayMajorantIterator_homogeneous(0f0, t_max, σ_maj)
end

"""
    create_majorant_iterator(table, medium::HomogeneousMedium, ray, t_max, λ, template_grid) -> RayMajorantIterator

Create a majorant iterator for homogeneous medium using the provided template grid for type consistency.
This version should be used in mixed media scenes to ensure all iterators have the same type.
"""
@propagate_inbounds function create_majorant_iterator(
    table::RGBToSpectrumTable,
    medium::HomogeneousMedium,
    ray::Raycore.Ray,
    t_max::Float32,
    λ::Wavelengths,
    template_grid::M
) where {M<:MajorantGrid}
    σ_a = uplift_rgb_unbounded(table, medium.σ_a, λ)
    σ_s = uplift_rgb_unbounded(table, medium.σ_s, λ)
    σ_maj = σ_a + σ_s

    # Create iterator with template grid type for GPU type consistency
    mode = (0f0 >= t_max) ? Int32(0) : Int32(1)
    res = template_grid.res
    return RayMajorantIterator{M}(
        mode,
        σ_maj,
        0f0, t_max,
        false,  # not called yet
        template_grid,  # Use template grid (never accessed in homogeneous mode)
        (Int32(res[1]), Int32(res[2]), Int32(res[3])),
        (0f0, 0f0, 0f0),
        (0f0, 0f0, 0f0),
        (Int32(0), Int32(0), Int32(0)),
        (Int32(0), Int32(0), Int32(0)),
        (Int32(0), Int32(0), Int32(0))
    )
end

"""
    get_template_grid(medium) -> MajorantGrid

Extract a template grid from a medium for type consistency in mixed media scenes.
HomogeneousMedium returns EmptyMajorantGrid(), GridMedium returns its majorant_grid.
"""
@inline get_template_grid(::HomogeneousMedium) = EmptyMajorantGrid()

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

# Template grid extraction for GridMedium (defined here after GridMedium struct)
@inline get_template_grid(medium::GridMedium) = medium.majorant_grid

"""
    get_template_grid_from_tuple(media::Tuple) -> MajorantGrid

Extract a template grid from the first GridMedium or RGBGridMedium in the tuple, or EmptyMajorantGrid() if none.
This is used to ensure all majorant iterators have consistent types for GPU compilation.
"""
@generated function get_template_grid_from_tuple(media::M) where {M <: Tuple}
    N = length(M.parameters)

    # Find first GridMedium or RGBGridMedium in the tuple
    for i in 1:N
        T = M.parameters[i]
        if T <: GridMedium || T <: RGBGridMedium
            return :(@inbounds get_template_grid(media[$i]))
        end
    end

    # No GridMedium/RGBGridMedium found - return empty grid (all homogeneous)
    # Use function call instead of global const to be GPU-compatible
    return :(EmptyMajorantGrid())
end

# Single medium case
@inline get_template_grid_from_tuple(medium::Medium) = get_template_grid(medium)

# ============================================================================
# RGB Grid Medium (Heterogeneous with per-voxel RGB coefficients)
# Following pbrt-v4's RGBGridMedium implementation
# ============================================================================

"""
    RGBGridMedium

A participating medium with spatially varying per-voxel RGB absorption and scattering
coefficients. Following pbrt-v4's RGBGridMedium implementation exactly.

Key design (matching pbrt-v4):
- Optional σ_a grid: 3D array of RGBSpectrum absorption coefficients (if absent, defaults to 1.0)
- Optional σ_s grid: 3D array of RGBSpectrum scattering coefficients (if absent, defaults to 1.0)
- Optional Le grid: 3D array of RGBSpectrum emission coefficients (for emissive volumes)
- sigma_scale: Global multiplier for σ_a and σ_s grids
- Le_scale: Global multiplier for Le grid
- The majorant grid stores `sigma_scale * max(σ_a + σ_s)` per coarse voxel
- In SampleRay, uses unit σ_t since scaling is already in the majorant grid
"""
struct RGBGridMedium{A<:Union{Nothing, AbstractArray{RGBSpectrum,3}},
                     S<:Union{Nothing, AbstractArray{RGBSpectrum,3}},
                     L<:Union{Nothing, AbstractArray{RGBSpectrum,3}},
                     M<:MajorantGrid} <: Medium
    # Volume bounds (in medium space, typically unit cube)
    bounds::Bounds3

    # Transform from render space to medium space
    render_to_medium::Mat4f
    medium_to_render::Mat4f

    # Optional 3D σ_a grid (absorption coefficients per voxel)
    # If nothing, defaults to RGBSpectrum(1.0) everywhere
    σ_a_grid::A

    # Optional 3D σ_s grid (scattering coefficients per voxel)
    # If nothing, defaults to RGBSpectrum(1.0) everywhere
    σ_s_grid::S

    # Global scale factor for σ_a and σ_s (like pbrt-v4's sigmaScale)
    sigma_scale::Float32

    # Optional 3D Le grid (emission coefficients per voxel)
    # Following pbrt-v4: if LeGrid is present, σ_a_grid must also be present
    Le_grid::L

    # Global scale factor for Le (like pbrt-v4's LeScale)
    Le_scale::Float32

    # Grid resolution (stored for GPU compatibility, Int32)
    grid_res::Vec{3, Int32}

    # Phase function asymmetry (Henyey-Greenstein g parameter)
    g::Float32

    # Majorant grid for efficient delta tracking
    # Stores sigma_scale * max(σ_a.MaxValue + σ_s.MaxValue) per coarse voxel
    majorant_grid::M
end

"""
    RGBGridMedium(; σ_a_grid, σ_s_grid, Le_grid, sigma_scale, Le_scale, g, bounds, transform, majorant_res)

Create an RGBGridMedium following pbrt-v4's design exactly.

At least one of σ_a_grid or σ_s_grid must be provided. The grids contain per-voxel
RGBSpectrum values that are multiplied by sigma_scale (or Le_scale for emission).

# Arguments
- `σ_a_grid`: Optional 3D array of RGBSpectrum absorption coefficients
- `σ_s_grid`: Optional 3D array of RGBSpectrum scattering coefficients
- `Le_grid`: Optional 3D array of RGBSpectrum emission coefficients (requires σ_a_grid)
- `sigma_scale`: Global multiplier for σ_a and σ_s grids (default: 1.0)
- `Le_scale`: Global multiplier for Le grid (default: 0.0)
- `g`: Henyey-Greenstein asymmetry parameter (default: 0.0)
- `bounds`: Volume bounds in medium space (default: unit cube)
- `transform`: Transform from medium to render space (default: identity)
- `majorant_res`: Resolution of the majorant grid (default: 16³)
"""
function RGBGridMedium(;
    σ_a_grid::Union{Nothing, AbstractArray{RGBSpectrum,3}} = nothing,
    σ_s_grid::Union{Nothing, AbstractArray{RGBSpectrum,3}} = nothing,
    Le_grid::Union{Nothing, AbstractArray{RGBSpectrum,3}} = nothing,
    sigma_scale::Float32 = 1f0,
    Le_scale::Float32 = 0f0,
    g::Float32 = 0f0,
    bounds::Bounds3 = Bounds3(Point3f(0f0, 0f0, 0f0), Point3f(1f0, 1f0, 1f0)),
    transform::Mat4f = Mat4f(I),
    majorant_res::Vec3i = Vec3i(16, 16, 16)
)
    # At least one grid must be provided
    @assert !isnothing(σ_a_grid) || !isnothing(σ_s_grid) "At least one of σ_a_grid or σ_s_grid must be provided"

    # Following pbrt-v4: if LeGrid is present, σ_a_grid must also be present
    if !isnothing(Le_grid)
        @assert !isnothing(σ_a_grid) "Le_grid requires σ_a_grid to be provided (following pbrt-v4)"
    end

    # If multiple grids provided, they must have same dimensions
    if !isnothing(σ_a_grid) && !isnothing(σ_s_grid)
        @assert size(σ_a_grid) == size(σ_s_grid) "σ_a_grid and σ_s_grid must have the same dimensions"
    end
    if !isnothing(Le_grid) && !isnothing(σ_a_grid)
        @assert size(Le_grid) == size(σ_a_grid) "Le_grid and σ_a_grid must have the same dimensions"
    end

    # Get grid resolution from whichever grid is provided
    grid_size = !isnothing(σ_a_grid) ? size(σ_a_grid) : size(σ_s_grid)
    nx, ny, nz = grid_size
    grid_res = Vec{3, Int32}(Int32(nx), Int32(ny), Int32(nz))

    # Compute inverse transform (render_to_medium)
    inv_transform = inv(transform)

    # Build majorant grid following pbrt-v4:
    # For each majorant voxel, compute sigma_scale * max(σ_a.MaxValue + σ_s.MaxValue)
    majorant_grid = build_rgb_majorant_grid(σ_a_grid, σ_s_grid, sigma_scale, grid_size, majorant_res)

    RGBGridMedium(
        bounds,
        inv_transform,
        transform,
        σ_a_grid,
        σ_s_grid,
        sigma_scale,
        Le_grid,
        Le_scale,
        grid_res,
        g,
        majorant_grid
    )
end

"""
Build majorant grid for RGBGridMedium following pbrt-v4.

For each majorant voxel, computes:
  sigma_scale * (max(σ_a.MaxValue) + max(σ_s.MaxValue))

where MaxValue returns the maximum RGB component of each spectrum.
"""
function build_rgb_majorant_grid(
    σ_a_grid::Union{Nothing, AbstractArray{RGBSpectrum,3}},
    σ_s_grid::Union{Nothing, AbstractArray{RGBSpectrum,3}},
    sigma_scale::Float32,
    grid_size::Tuple{Int,Int,Int},
    majorant_res::Vec3i
)
    nx, ny, nz = grid_size
    grid = MajorantGrid(majorant_res, Vector{Float32})

    for iz in 0:majorant_res[3]-1
        # Map majorant voxel to density grid range
        z_start_f = iz * nz / majorant_res[3]
        z_end_f = (iz + 1) * nz / majorant_res[3]
        z_start = max(1, floor(Int, z_start_f) + 1)
        z_end = min(nz, ceil(Int, z_end_f))

        for iy in 0:majorant_res[2]-1
            y_start_f = iy * ny / majorant_res[2]
            y_end_f = (iy + 1) * ny / majorant_res[2]
            y_start = max(1, floor(Int, y_start_f) + 1)
            y_end = min(ny, ceil(Int, y_end_f))

            for ix in 0:majorant_res[1]-1
                x_start_f = ix * nx / majorant_res[1]
                x_end_f = (ix + 1) * nx / majorant_res[1]
                x_start = max(1, floor(Int, x_start_f) + 1)
                x_end = min(nx, ceil(Int, x_end_f))

                # Find max σ_t = max(σ_a) + max(σ_s) in this region
                # Following pbrt-v4: use MaxValue (max RGB component) of each spectrum
                max_σ_a = 0f0
                max_σ_s = 0f0

                for z in z_start:z_end, y in y_start:y_end, x in x_start:x_end
                    if !isnothing(σ_a_grid)
                        rgb = σ_a_grid[x, y, z]
                        max_σ_a = max(max_σ_a, max(rgb.c[1], rgb.c[2], rgb.c[3]))
                    end
                    if !isnothing(σ_s_grid)
                        rgb = σ_s_grid[x, y, z]
                        max_σ_s = max(max_σ_s, max(rgb.c[1], rgb.c[2], rgb.c[3]))
                    end
                end

                # If grid is absent, default value is 1.0
                if isnothing(σ_a_grid)
                    max_σ_a = 1f0
                end
                if isnothing(σ_s_grid)
                    max_σ_s = 1f0
                end

                # Store sigma_scale * max_σ_t
                majorant_set!(grid, ix, iy, iz, sigma_scale * (max_σ_a + max_σ_s))
            end
        end
    end

    return grid
end

# Template grid extraction for RGBGridMedium
@inline get_template_grid(medium::RGBGridMedium) = medium.majorant_grid

# Following pbrt-v4: bool IsEmissive() const { return LeGrid && LeScale > 0; }
@propagate_inbounds is_emissive(medium::RGBGridMedium) = !isnothing(medium.Le_grid) && medium.Le_scale > 0f0

"""
Sample σ_a at a point using trilinear interpolation.
Returns default RGBSpectrum(1.0) if σ_a_grid is nothing.
"""
@propagate_inbounds function sample_σ_a(medium::RGBGridMedium, p_norm::Point3f)::RGBSpectrum
    isnothing(medium.σ_a_grid) && return RGBSpectrum(1f0)
    return _sample_rgb_grid(medium.σ_a_grid, medium.grid_res, p_norm)
end

"""
Sample σ_s at a point using trilinear interpolation.
Returns default RGBSpectrum(1.0) if σ_s_grid is nothing.
"""
@propagate_inbounds function sample_σ_s(medium::RGBGridMedium, p_norm::Point3f)::RGBSpectrum
    isnothing(medium.σ_s_grid) && return RGBSpectrum(1f0)
    return _sample_rgb_grid(medium.σ_s_grid, medium.grid_res, p_norm)
end

"""
Sample Le at a point using trilinear interpolation.
Returns RGBSpectrum(0.0) if Le_grid is nothing.
"""
@propagate_inbounds function sample_Le(medium::RGBGridMedium, p_norm::Point3f)::RGBSpectrum
    isnothing(medium.Le_grid) && return RGBSpectrum(0f0)
    return _sample_rgb_grid(medium.Le_grid, medium.grid_res, p_norm)
end

"""
Trilinear interpolation for RGB grid.
p_norm is in [0,1]³ normalized coordinates within bounds.
"""
@propagate_inbounds function _sample_rgb_grid(
    grid::AbstractArray{RGBSpectrum,3},
    grid_res::Vec{3, Int32},
    p_norm::Point3f
)::RGBSpectrum
    # Check bounds
    if p_norm[1] < 0f0 || p_norm[2] < 0f0 || p_norm[3] < 0f0 ||
       p_norm[1] > 1f0 || p_norm[2] > 1f0 || p_norm[3] > 1f0
        return RGBSpectrum(0f0)
    end

    nx, ny, nz = grid_res[1], grid_res[2], grid_res[3]

    # Cell-centered interpretation (like pbrt-v4)
    gx = p_norm[1] * Float32(nx) + 0.5f0
    gy = p_norm[2] * Float32(ny) + 0.5f0
    gz = p_norm[3] * Float32(nz) + 0.5f0

    ix = clamp(floor_int32(gx), Int32(1), nx - Int32(1))
    iy = clamp(floor_int32(gy), Int32(1), ny - Int32(1))
    iz = clamp(floor_int32(gz), Int32(1), nz - Int32(1))

    fx = clamp(gx - Float32(ix), 0f0, 1f0)
    fy = clamp(gy - Float32(iy), 0f0, 1f0)
    fz = clamp(gz - Float32(iz), 0f0, 1f0)

    # Trilinear interpolation
    c000 = grid[ix, iy, iz]
    c100 = grid[ix+Int32(1), iy, iz]
    c010 = grid[ix, iy+Int32(1), iz]
    c110 = grid[ix+Int32(1), iy+Int32(1), iz]
    c001 = grid[ix, iy, iz+Int32(1)]
    c101 = grid[ix+Int32(1), iy, iz+Int32(1)]
    c011 = grid[ix, iy+Int32(1), iz+Int32(1)]
    c111 = grid[ix+Int32(1), iy+Int32(1), iz+Int32(1)]

    fx1 = 1f0 - fx
    c00 = c000 * fx1 + c100 * fx
    c10 = c010 * fx1 + c110 * fx
    c01 = c001 * fx1 + c101 * fx
    c11 = c011 * fx1 + c111 * fx

    fy1 = 1f0 - fy
    c0 = c00 * fy1 + c10 * fy
    c1 = c01 * fy1 + c11 * fy

    return c0 * (1f0 - fz) + c1 * fz
end

"""
Sample medium properties at a point for RGBGridMedium.
Following pbrt-v4's RGBGridMedium::SamplePoint exactly.
"""
@propagate_inbounds function sample_point(
    table::RGBToSpectrumTable,
    medium::RGBGridMedium,
    p::Point3f,
    λ::Wavelengths
)::MediumProperties
    # Transform to medium space (following pbrt-v4: renderFromMedium.ApplyInverse)
    M = medium.render_to_medium
    p_x = M[1,1] * p[1] + M[1,2] * p[2] + M[1,3] * p[3] + M[1,4]
    p_y = M[2,1] * p[1] + M[2,2] * p[2] + M[2,3] * p[3] + M[2,4]
    p_z = M[3,1] * p[1] + M[3,2] * p[2] + M[3,3] * p[3] + M[3,4]
    p_medium = Point3f(p_x, p_y, p_z)

    # Normalize to [0,1] within bounds (following pbrt-v4: bounds.Offset(p))
    p_norm = (p_medium - medium.bounds.p_min) ./ (medium.bounds.p_max - medium.bounds.p_min)

    # Compute σ_a and σ_s for RGBGridMedium (following pbrt-v4)
    σ_a_rgb = sample_σ_a(medium, p_norm)
    σ_s_rgb = sample_σ_s(medium, p_norm)

    # Convert to spectral and apply sigma_scale
    # Following pbrt-v4: sigma_a = sigmaScale * (sigma_aGrid ? sigma_aGrid->Lookup(...) : 1.0)
    σ_a = uplift_rgb_unbounded(table, σ_a_rgb, λ) * medium.sigma_scale
    σ_s = uplift_rgb_unbounded(table, σ_s_rgb, λ) * medium.sigma_scale

    # Find emitted radiance Le for RGBGridMedium (following pbrt-v4)
    # SampledSpectrum Le(0.f);
    # if (LeGrid && LeScale > 0) {
    #     Le = LeScale * LeGrid->Lookup(p, convert);
    # }
    Le = SpectralRadiance(0f0)
    if !isnothing(medium.Le_grid) && medium.Le_scale > 0f0
        Le_rgb = sample_Le(medium, p_norm)
        # Note: pbrt-v4 uses RGBIlluminantSpectrum for Le, we use unbounded for simplicity
        Le = uplift_rgb_unbounded(table, Le_rgb, λ) * medium.Le_scale
    end

    return MediumProperties(σ_a, σ_s, Le, medium.g)
end

"""
Create majorant iterator for RGBGridMedium.
Following pbrt-v4's RGBGridMedium::SampleRay.
"""
@propagate_inbounds function create_majorant_iterator(
    table::RGBToSpectrumTable,
    medium::RGBGridMedium,
    ray::Raycore.Ray,
    t_max::Float32,
    λ::Wavelengths
)
    # Transform ray to medium space
    M = medium.render_to_medium

    ray_o_x = M[1,1] * ray.o[1] + M[1,2] * ray.o[2] + M[1,3] * ray.o[3] + M[1,4]
    ray_o_y = M[2,1] * ray.o[1] + M[2,2] * ray.o[2] + M[2,3] * ray.o[3] + M[2,4]
    ray_o_z = M[3,1] * ray.o[1] + M[3,2] * ray.o[2] + M[3,3] * ray.o[3] + M[3,4]
    ray_o = Point3f(ray_o_x, ray_o_y, ray_o_z)

    ray_d_x = M[1,1] * ray.d[1] + M[1,2] * ray.d[2] + M[1,3] * ray.d[3]
    ray_d_y = M[2,1] * ray.d[1] + M[2,2] * ray.d[2] + M[2,3] * ray.d[3]
    ray_d_z = M[3,1] * ray.d[1] + M[3,2] * ray.d[2] + M[3,3] * ray.d[3]
    ray_d = Vec3f(ray_d_x, ray_d_y, ray_d_z)

    # Check for degenerate direction
    dir_len_sq = ray_d_x * ray_d_x + ray_d_y * ray_d_y + ray_d_z * ray_d_z
    if dir_len_sq < 1f-20
        return RayMajorantIterator(medium.majorant_grid)
    end

    # Intersect with bounds
    t_enter, t_exit = ray_bounds_intersect(ray_o, ray_d, medium.bounds)
    t_enter = max(t_enter, 0f0)
    t_exit = min(t_exit, t_max)

    if t_enter >= t_exit
        return RayMajorantIterator(medium.majorant_grid)
    end

    # Following pbrt-v4: use unit sigma_t since scaling is baked into majorant grid
    # SampledSpectrum sigma_t(1);
    σ_t = SpectralRadiance(1f0)

    dda_iter = create_dda_iterator(
        medium.majorant_grid,
        medium.bounds,
        ray_o,
        ray_d,
        t_enter,
        t_exit,
        σ_t
    )
    return RayMajorantIterator(dda_iter)
end

@propagate_inbounds function create_majorant_iterator(
    table::RGBToSpectrumTable,
    medium::RGBGridMedium,
    ray::Raycore.Ray,
    t_max::Float32,
    λ::Wavelengths,
    ::MajorantGrid
)
    return create_majorant_iterator(table, medium, ray, t_max, λ)
end

"""
    get_majorant(table, medium::RGBGridMedium, ray, t_min, t_max, λ) -> RayMajorantSegment

Get majorant for ray segment using global maximum from majorant grid.
For better performance, use create_majorant_iterator for DDA-based traversal.
"""
@propagate_inbounds function get_majorant(
    table::RGBToSpectrumTable,
    medium::RGBGridMedium,
    ray::Raycore.Ray,
    t_min::Float32,
    t_max::Float32,
    λ::Wavelengths
)::RayMajorantSegment
    # Find global max from majorant grid (conservative bound)
    # The majorant grid already stores sigma_scale * max(σ_a + σ_s) per voxel
    max_majorant = 0f0
    for v in medium.majorant_grid.voxels
        max_majorant = max(max_majorant, v)
    end

    # Following pbrt-v4: majorant grid stores scalar max, use unit spectrum
    # since the actual σ_t values are wavelength-dependent but bounded by this max
    σ_maj = SpectralRadiance(max_majorant)

    return RayMajorantSegment(t_min, t_max, σ_maj)
end

"""Build a coarse majorant grid from the density field"""
function build_majorant_grid(density::AbstractArray{Float32,3}, res::Vec3i)
    nx, ny, nz = size(density)
    grid = MajorantGrid(res, Vector{Float32})

    # For each majorant voxel, find max density in corresponding region
    # Use floating point mapping to handle cases where majorant res > density res
    for iz in 0:res[3]-1
        # Map majorant voxel [iz, iz+1)/res to density range [0, nz)
        z_start_f = iz * nz / res[3]
        z_end_f = (iz + 1) * nz / res[3]
        z_start = max(1, floor(Int, z_start_f) + 1)
        z_end = min(nz, ceil(Int, z_end_f))

        for iy in 0:res[2]-1
            y_start_f = iy * ny / res[2]
            y_end_f = (iy + 1) * ny / res[2]
            y_start = max(1, floor(Int, y_start_f) + 1)
            y_end = min(ny, ceil(Int, y_end_f))

            for ix in 0:res[1]-1
                x_start_f = ix * nx / res[1]
                x_end_f = (ix + 1) * nx / res[1]
                x_start = max(1, floor(Int, x_start_f) + 1)
                x_end = min(nx, ceil(Int, x_end_f))

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

"""
Sample density at a point using trilinear interpolation.

Uses pbrt-v4's cell-centered interpretation:
- p ∈ [0,1]³ is normalized position within bounds
- Grid has nx×ny×nz voxels (1-indexed in Julia)
- Voxel i is centered at (i - 0.5) / n in normalized space
- Interpolation uses 8 neighboring voxels with proper clamping at boundaries
"""
@propagate_inbounds function sample_density(medium::GridMedium, p_medium::Point3f)::Float32
    # Normalize to [0,1] within bounds
    p_norm = (p_medium - medium.bounds.p_min) ./ (medium.bounds.p_max - medium.bounds.p_min)

    # Check bounds (explicit element-wise check for GPU compatibility - avoid any() which creates arrays)
    if p_norm[1] < 0f0 || p_norm[2] < 0f0 || p_norm[3] < 0f0 ||
       p_norm[1] > 1f0 || p_norm[2] > 1f0 || p_norm[3] > 1f0
        return 0f0
    end

    # Grid coordinates following pbrt-v4's SampledGrid::Lookup:
    # pSamples = p * n - 0.5 (in 0-indexed C++)
    # For Julia 1-indexing: pSamples = p * n + 0.5 (so p=0 → 0.5, p=1 → n+0.5)
    # This places voxel i's center at p = (i - 0.5) / n
    nx, ny, nz = medium.density_res[1], medium.density_res[2], medium.density_res[3]
    gx = p_norm[1] * Float32(nx) + 0.5f0
    gy = p_norm[2] * Float32(ny) + 0.5f0
    gz = p_norm[3] * Float32(nz) + 0.5f0

    # Integer indices (use floor_int32 for GPU compatibility)
    # Clamp to [1, nx-1] so that ix+1 stays within [2, nx]
    ix = clamp(floor_int32(gx), Int32(1), nx - Int32(1))
    iy = clamp(floor_int32(gy), Int32(1), ny - Int32(1))
    iz = clamp(floor_int32(gz), Int32(1), nz - Int32(1))

    # Fractional parts (clamp to [0,1] for edge cases)
    fx = clamp(gx - Float32(ix), 0f0, 1f0)
    fy = clamp(gy - Float32(iy), 0f0, 1f0)
    fz = clamp(gz - Float32(iz), 0f0, 1f0)

    # Trilinear interpolation (use Int32(1) to avoid promotion to Int64)
    d000 = medium.density[ix, iy, iz]
    d100 = medium.density[ix+Int32(1), iy, iz]
    d010 = medium.density[ix, iy+Int32(1), iz]
    d110 = medium.density[ix+Int32(1), iy+Int32(1), iz]
    d001 = medium.density[ix, iy, iz+Int32(1)]
    d101 = medium.density[ix+Int32(1), iy, iz+Int32(1)]
    d011 = medium.density[ix, iy+Int32(1), iz+Int32(1)]
    d111 = medium.density[ix+Int32(1), iy+Int32(1), iz+Int32(1)]

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
    create_majorant_iterator(table, medium::GridMedium, ray, t_max, λ) -> RayMajorantIterator

Create a DDA majorant iterator for traversing the medium along a ray.
Following PBRT-v4's GridMedium::SampleRay pattern.
Returns unified RayMajorantIterator for GPU compatibility with mixed media.

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
    # NOTE: Do NOT normalize - t-parameterization must be preserved so that
    # t values from DDA work with the original world-space ray
    ray_d_x = M[1,1] * ray.d[1] + M[1,2] * ray.d[2] + M[1,3] * ray.d[3]
    ray_d_y = M[2,1] * ray.d[1] + M[2,2] * ray.d[2] + M[2,3] * ray.d[3]
    ray_d_z = M[3,1] * ray.d[1] + M[3,2] * ray.d[2] + M[3,3] * ray.d[3]
    ray_d = Vec3f(ray_d_x, ray_d_y, ray_d_z)

    # Check for degenerate direction
    dir_len_sq = ray_d_x * ray_d_x + ray_d_y * ray_d_y + ray_d_z * ray_d_z
    if dir_len_sq < 1f-20
        return RayMajorantIterator(medium.majorant_grid)
    end

    # Compute ray-bounds intersection in medium space
    # The t values from this are valid for both world-space and medium-space rays
    # because the transform preserves t-parameterization (linear transform)
    t_enter, t_exit = ray_bounds_intersect(ray_o, ray_d, medium.bounds)

    # Clamp to requested range and check validity
    t_enter = max(t_enter, 0f0)
    t_exit = min(t_exit, t_max)

    if t_enter >= t_exit
        # Ray misses bounds or segment is empty - return empty iterator
        return RayMajorantIterator(medium.majorant_grid)
    end

    # Compute base extinction coefficient (at full density)
    σ_a = uplift_rgb_unbounded(table, medium.σ_a, λ)
    σ_s = uplift_rgb_unbounded(table, medium.σ_s, λ)
    σ_t = σ_a + σ_s

    # Create the DDA iterator and wrap in unified iterator
    dda_iter = create_dda_iterator(
        medium.majorant_grid,
        medium.bounds,
        ray_o,
        ray_d,
        t_enter,
        t_exit,
        σ_t
    )
    return RayMajorantIterator(dda_iter)
end

"""
    create_majorant_iterator(table, medium::GridMedium, ray, t_max, λ, template_grid) -> RayMajorantIterator

Version with template_grid parameter for API consistency with HomogeneousMedium.
GridMedium ignores the template and uses its own majorant_grid.
"""
@propagate_inbounds function create_majorant_iterator(
    table::RGBToSpectrumTable,
    medium::GridMedium,
    ray::Raycore.Ray,
    t_max::Float32,
    λ::Wavelengths,
    ::MajorantGrid  # Ignored - GridMedium uses its own grid
)
    # Just call the non-template version
    return create_majorant_iterator(table, medium, ray, t_max, λ)
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

# ============================================================================
# Preset Medium Constructors
# Based on measured data from pbrt-v4 (SIGGRAPH 2001 & 2006 papers)
# Values are σ_s (scattering) and σ_a (absorption) in mm⁻¹
# ============================================================================

# Subsurface scattering presets from:
# - "A Practical Model for Subsurface Light Transport" (Jensen et al., SIGGRAPH 2001)
# - "Acquiring Scattering Properties of Participating Media by Dilution" (SIGGRAPH 2006)

const _MEDIUM_PRESETS = Dict{String, NamedTuple{(:σ_s, :σ_a), Tuple{NTuple{3,Float32}, NTuple{3,Float32}}}}(
    # === Milk and dairy products ===
    "Wholemilk" => (σ_s=(2.55f0, 3.21f0, 3.77f0), σ_a=(0.0011f0, 0.0024f0, 0.014f0)),
    "Skimmilk" => (σ_s=(0.70f0, 1.22f0, 1.90f0), σ_a=(0.0014f0, 0.0025f0, 0.0142f0)),
    "LowfatMilk" => (σ_s=(0.89f0, 1.51f0, 2.53f0), σ_a=(0.0029f0, 0.0058f0, 0.0115f0)),
    "ReducedMilk" => (σ_s=(2.49f0, 3.17f0, 4.52f0), σ_a=(0.0026f0, 0.0051f0, 0.0128f0)),
    "RegularMilk" => (σ_s=(4.55f0, 5.83f0, 7.14f0), σ_a=(0.0015f0, 0.0046f0, 0.0199f0)),
    "Cream" => (σ_s=(7.38f0, 5.47f0, 3.15f0), σ_a=(0.0002f0, 0.0028f0, 0.0163f0)),
    "LowfatChocolateMilk" => (σ_s=(0.65f0, 0.84f0, 1.11f0), σ_a=(0.0115f0, 0.0368f0, 0.1564f0)),
    "RegularChocolateMilk" => (σ_s=(1.46f0, 2.13f0, 2.95f0), σ_a=(0.0101f0, 0.0431f0, 0.1438f0)),
    "LowfatSoyMilk" => (σ_s=(0.31f0, 0.34f0, 0.62f0), σ_a=(0.0014f0, 0.0072f0, 0.0359f0)),
    "RegularSoyMilk" => (σ_s=(0.59f0, 0.74f0, 1.47f0), σ_a=(0.0019f0, 0.0096f0, 0.0652f0)),

    # === Coffee and beverages ===
    "Espresso" => (σ_s=(0.72f0, 0.85f0, 1.02f0), σ_a=(4.80f0, 6.58f0, 8.85f0)),
    "MintMochaCoffee" => (σ_s=(0.32f0, 0.39f0, 0.48f0), σ_a=(3.77f0, 5.82f0, 7.82f0)),

    # === Alcoholic beverages ===
    "Chardonnay" => (σ_s=(1.8f-5, 1.4f-5, 1.2f-5), σ_a=(0.0108f0, 0.0119f0, 0.0240f0)),
    "WhiteZinfandel" => (σ_s=(1.8f-5, 1.9f-5, 1.3f-5), σ_a=(0.0121f0, 0.0162f0, 0.0198f0)),
    "Merlot" => (σ_s=(2.1f-5, 0f0, 0f0), σ_a=(0.116f0, 0.252f0, 0.294f0)),
    "BudweiserBeer" => (σ_s=(2.4f-5, 2.4f-5, 1.1f-5), σ_a=(0.0115f0, 0.0249f0, 0.0578f0)),
    "CoorsLightBeer" => (σ_s=(5.1f-5, 4.3f-5, 0f0), σ_a=(0.0062f0, 0.0140f0, 0.0350f0)),

    # === Fruit juices ===
    "AppleJuice" => (σ_s=(1.4f-4, 1.6f-4, 2.3f-4), σ_a=(0.0130f0, 0.0237f0, 0.0522f0)),
    "CranberryJuice" => (σ_s=(1.0f-4, 1.2f-4, 7.8f-5), σ_a=(0.0394f0, 0.0942f0, 0.1243f0)),
    "GrapeJuice" => (σ_s=(5.4f-5, 0f0, 0f0), σ_a=(0.1040f0, 0.2396f0, 0.2933f0)),
    "RubyGrapefruitJuice" => (σ_s=(0.011f0, 0.011f0, 0.011f0), σ_a=(0.0859f0, 0.1831f0, 0.2526f0)),

    # === Sodas (nearly transparent) ===
    "Sprite" => (σ_s=(6.0f-6, 6.4f-6, 6.6f-6), σ_a=(0.00189f0, 0.00183f0, 0.00200f0)),
    "Coke" => (σ_s=(8.9f-5, 8.4f-5, 0f0), σ_a=(0.1001f0, 0.1650f0, 0.2468f0)),
    "Pepsi" => (σ_s=(6.2f-5, 4.3f-5, 0f0), σ_a=(0.0916f0, 0.1416f0, 0.2073f0)),

    # === Foods and organics ===
    "Apple" => (σ_s=(2.29f0, 2.39f0, 1.97f0), σ_a=(0.0030f0, 0.0034f0, 0.046f0)),
    "Potato" => (σ_s=(0.68f0, 0.70f0, 0.55f0), σ_a=(0.0024f0, 0.0090f0, 0.12f0)),
    "Chicken1" => (σ_s=(0.15f0, 0.21f0, 0.38f0), σ_a=(0.015f0, 0.077f0, 0.19f0)),
    "Chicken2" => (σ_s=(0.19f0, 0.25f0, 0.32f0), σ_a=(0.018f0, 0.088f0, 0.20f0)),
    "Ketchup" => (σ_s=(0.18f0, 0.07f0, 0.03f0), σ_a=(0.061f0, 0.97f0, 1.45f0)),

    # === Skin ===
    "Skin1" => (σ_s=(0.74f0, 0.88f0, 1.01f0), σ_a=(0.032f0, 0.17f0, 0.48f0)),
    "Skin2" => (σ_s=(1.09f0, 1.59f0, 1.79f0), σ_a=(0.013f0, 0.070f0, 0.145f0)),

    # === Other materials ===
    "Marble" => (σ_s=(2.19f0, 2.62f0, 3.00f0), σ_a=(0.0021f0, 0.0041f0, 0.0071f0)),
    "Spectralon" => (σ_s=(11.6f0, 20.4f0, 14.9f0), σ_a=(0f0, 0f0, 0f0)),
    "Shampoo" => (σ_s=(0.0007f0, 0.0008f0, 0.0009f0), σ_a=(0.0141f0, 0.0457f0, 0.0617f0)),
    "HeadShouldersShampoo" => (σ_s=(0.0238f0, 0.0288f0, 0.0343f0), σ_a=(0.0846f0, 0.1569f0, 0.2037f0)),
    "Clorox" => (σ_s=(0.0024f0, 0.0031f0, 0.0040f0), σ_a=(0.0034f0, 0.0149f0, 0.0263f0)),

    # === Powders ===
    "CappuccinoPowder" => (σ_s=(1.84f0, 2.59f0, 2.17f0), σ_a=(35.84f0, 49.55f0, 61.08f0)),
    "SaltPowder" => (σ_s=(0.0273f0, 0.0325f0, 0.0320f0), σ_a=(0.284f0, 0.326f0, 0.341f0)),
    "SugarPowder" => (σ_s=(2.2f-4, 2.6f-4, 2.7f-4), σ_a=(0.0126f0, 0.0311f0, 0.0501f0)),

    # === Water ===
    "PacificOceanSurfaceWater" => (σ_s=(1.8f-4, 3.2f-4, 2.0f-4), σ_a=(0.0318f0, 0.0313f0, 0.0301f0)),
)

"""
    get_medium_preset(name::String) -> NamedTuple{(:σ_s, :σ_a), ...}

Get the scattering properties for a named medium preset.
Returns a NamedTuple with σ_s (scattering) and σ_a (absorption) coefficients in mm⁻¹.

Available presets include:
- Milk: "Wholemilk", "Skimmilk", "LowfatMilk", "ReducedMilk", "RegularMilk", "Cream"
- Chocolate milk: "LowfatChocolateMilk", "RegularChocolateMilk"
- Soy milk: "LowfatSoyMilk", "RegularSoyMilk"
- Coffee: "Espresso", "MintMochaCoffee"
- Wine/Beer: "Chardonnay", "WhiteZinfandel", "Merlot", "BudweiserBeer", "CoorsLightBeer"
- Juices: "AppleJuice", "CranberryJuice", "GrapeJuice", "RubyGrapefruitJuice"
- Sodas: "Sprite", "Coke", "Pepsi"
- Foods: "Apple", "Potato", "Chicken1", "Chicken2", "Ketchup"
- Skin: "Skin1", "Skin2"
- Materials: "Marble", "Spectralon", "Shampoo", "HeadShouldersShampoo", "Clorox"
- Powders: "CappuccinoPowder", "SaltPowder", "SugarPowder"
- Water: "PacificOceanSurfaceWater"
"""
function get_medium_preset(name::String)
    haskey(_MEDIUM_PRESETS, name) || error("Unknown medium preset: $name. Available: $(keys(_MEDIUM_PRESETS))")
    return _MEDIUM_PRESETS[name]
end

# ============================================================================
# Convenient Medium Constructors
# ============================================================================

"""
    Milk(; scale=1.0, g=0.0) -> HomogeneousMedium

Create a whole milk medium with realistic scattering properties.
The `scale` parameter allows adjusting the density (for diluted milk use scale < 1).

# Examples
```julia
Milk()                    # Standard whole milk
Milk(scale=0.5)          # Diluted milk
Milk(g=0.8)              # Forward-scattering milk
```
"""
function Milk(; scale::Real=1f0, g::Real=0f0)
    props = _MEDIUM_PRESETS["Wholemilk"]
    σ_s = RGBSpectrum(props.σ_s...) * Float32(scale)
    σ_a = RGBSpectrum(props.σ_a...) * Float32(scale)
    HomogeneousMedium(σ_a=σ_a, σ_s=σ_s, g=Float32(g))
end

"""
    Smoke(; density=0.5, albedo=0.9, g=0.0) -> HomogeneousMedium

Create a smoke/fog medium. Smoke is characterized by high scattering
and low absorption (high albedo = scattering / extinction).

# Arguments
- `density`: Overall density multiplier (higher = thicker smoke)
- `albedo`: Single-scattering albedo (0-1, higher = more scattering, whiter smoke)
- `g`: Henyey-Greenstein asymmetry parameter (-1 to 1, 0 = isotropic)

# Examples
```julia
Smoke()                   # Light gray smoke
Smoke(density=2.0)        # Dense smoke
Smoke(albedo=0.5)         # Darker, more absorbing smoke
Smoke(g=0.6)              # Forward-scattering (typical for smoke)
```
"""
function Smoke(; density::Real=0.5f0, albedo::Real=0.9f0, g::Real=0f0)
    # Smoke is typically gray, so use equal RGB values
    σ_t = Float32(density)  # Total extinction
    σ_s_val = σ_t * Float32(albedo)
    σ_a_val = σ_t * (1f0 - Float32(albedo))
    σ_s = RGBSpectrum(σ_s_val)
    σ_a = RGBSpectrum(σ_a_val)
    HomogeneousMedium(σ_a=σ_a, σ_s=σ_s, g=Float32(g))
end

"""
    Fog(; density=0.1, g=0.0) -> HomogeneousMedium

Create a fog medium. Fog is very light, highly scattering, and nearly
non-absorbing (appears white).

# Arguments
- `density`: Fog density (lower = more transparent)
- `g`: Henyey-Greenstein asymmetry (0 = isotropic, typical for fog)

# Examples
```julia
Fog()                     # Light fog
Fog(density=0.3)          # Dense fog
```
"""
function Fog(; density::Real=0.1f0, g::Real=0f0)
    # Fog has very high albedo (nearly pure scattering)
    σ_s = RGBSpectrum(Float32(density))
    σ_a = RGBSpectrum(Float32(density) * 0.001f0)  # Very low absorption
    HomogeneousMedium(σ_a=σ_a, σ_s=σ_s, g=Float32(g))
end

"""
    Juice(name::Symbol; scale=1.0, g=0.0) -> HomogeneousMedium

Create a juice medium from presets.

# Available names
- `:apple`, `:cranberry`, `:grape`, `:grapefruit`

# Examples
```julia
Juice(:apple)             # Apple juice
Juice(:grape, scale=0.5)  # Diluted grape juice
```
"""
function Juice(name::Symbol; scale::Real=1f0, g::Real=0f0)
    preset_name = if name == :apple
        "AppleJuice"
    elseif name == :cranberry
        "CranberryJuice"
    elseif name == :grape
        "GrapeJuice"
    elseif name == :grapefruit
        "RubyGrapefruitJuice"
    else
        error("Unknown juice type: $name. Available: :apple, :cranberry, :grape, :grapefruit")
    end
    props = _MEDIUM_PRESETS[preset_name]
    σ_s = RGBSpectrum(props.σ_s...) * Float32(scale)
    σ_a = RGBSpectrum(props.σ_a...) * Float32(scale)
    HomogeneousMedium(σ_a=σ_a, σ_s=σ_s, g=Float32(g))
end

"""
    Wine(name::Symbol; scale=1.0, g=0.0) -> HomogeneousMedium

Create a wine medium from presets.

# Available names
- `:chardonnay`, `:zinfandel`, `:merlot`

# Examples
```julia
Wine(:merlot)             # Red merlot wine
Wine(:chardonnay)         # White wine
```
"""
function Wine(name::Symbol; scale::Real=1f0, g::Real=0f0)
    preset_name = if name == :chardonnay
        "Chardonnay"
    elseif name == :zinfandel
        "WhiteZinfandel"
    elseif name == :merlot
        "Merlot"
    else
        error("Unknown wine type: $name. Available: :chardonnay, :zinfandel, :merlot")
    end
    props = _MEDIUM_PRESETS[preset_name]
    σ_s = RGBSpectrum(props.σ_s...) * Float32(scale)
    σ_a = RGBSpectrum(props.σ_a...) * Float32(scale)
    HomogeneousMedium(σ_a=σ_a, σ_s=σ_s, g=Float32(g))
end

"""
    Coffee(; scale=1.0, g=0.0) -> HomogeneousMedium

Create an espresso coffee medium with realistic scattering properties.

# Examples
```julia
Coffee()                  # Espresso
Coffee(scale=0.3)         # Diluted coffee (americano-like)
```
"""
function Coffee(; scale::Real=1f0, g::Real=0f0)
    props = _MEDIUM_PRESETS["Espresso"]
    σ_s = RGBSpectrum(props.σ_s...) * Float32(scale)
    σ_a = RGBSpectrum(props.σ_a...) * Float32(scale)
    HomogeneousMedium(σ_a=σ_a, σ_s=σ_s, g=Float32(g))
end

"""
    SubsurfaceMedium(name::String; scale=1.0, g=0.0) -> HomogeneousMedium

Create a medium from any of the available presets by name.

# Examples
```julia
SubsurfaceMedium("Marble")
SubsurfaceMedium("Skin1", scale=2.0)
SubsurfaceMedium("Ketchup")
```

See `get_medium_preset` for available preset names.
"""
function SubsurfaceMedium(name::String; scale::Real=1f0, g::Real=0f0)
    props = get_medium_preset(name)
    σ_s = RGBSpectrum(props.σ_s...) * Float32(scale)
    σ_a = RGBSpectrum(props.σ_a...) * Float32(scale)
    HomogeneousMedium(σ_a=σ_a, σ_s=σ_s, g=Float32(g))
end
