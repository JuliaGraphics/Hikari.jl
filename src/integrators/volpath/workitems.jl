# Work items for VolPath wavefront integrator
# Extends PhysicalWavefront work items with medium support

# ============================================================================
# Ray Samples (pre-computed low-discrepancy samples per bounce)
# ============================================================================

"""
    VPRaySamples

Pre-computed Sobol samples for a single bounce, matching pbrt-v4's RaySamples.
These are generated once per bounce and stored in PixelSampleState.
"""
struct VPRaySamples
    # Direct lighting samples
    direct_uc::Float32      # Light source selection [0,1)
    direct_u::Point2f       # Light position sample

    # Indirect ray samples
    indirect_uc::Float32    # BSDF component selection [0,1)
    indirect_u::Point2f     # BSDF direction sample
    indirect_rr::Float32    # Russian roulette decision [0,1)
end

# Default constructor with zero samples
VPRaySamples() = VPRaySamples(0f0, Point2f(0f0), 0f0, Point2f(0f0), 0f0)

# ============================================================================
# Ray Work Item with Medium
# ============================================================================

"""
    VPRayWorkItem

Ray work item for volumetric path tracing.
Includes medium index to track which medium the ray is currently in.
"""
struct VPRayWorkItem
    ray::Raycore.Ray
    depth::Int32
    lambda::Wavelengths
    pixel_index::Int32
    beta::SpectralRadiance          # Path throughput
    r_u::SpectralRadiance           # Rescaled unidirectional PDF
    r_l::SpectralRadiance           # Rescaled light PDF
    prev_intr_p::Point3f            # Previous interaction point (for MIS)
    prev_intr_n::Vec3f              # Previous interaction normal
    eta_scale::Float32              # Accumulated IOR ratio
    specular_bounce::Bool           # Was last bounce specular?
    any_non_specular_bounces::Bool  # Any non-specular bounces in path?
    medium_idx::SetKey         # Current medium (0 = vacuum)
end

function VPRayWorkItem(
    ray::Raycore.Ray,
    lambda::Wavelengths,
    pixel_index::Int32;
    medium_idx::SetKey = SetKey()
)
    VPRayWorkItem(
        ray,
        Int32(0),
        lambda,
        pixel_index,
        SpectralRadiance(1f0),
        SpectralRadiance(1f0),
        SpectralRadiance(1f0),
        Point3f(0f0, 0f0, 0f0),
        Vec3f(0f0, 0f0, 1f0),
        1f0,
        false,
        false,
        medium_idx
    )
end

# ============================================================================
# Medium Sample Work Item (for delta tracking with bounded t_max)
# ============================================================================

"""
    VPMediumSampleWorkItem

Work item for rays in medium that need delta tracking.
Follows pbrt-v4's approach: intersection is found FIRST, then delta tracking
runs with the known t_max distance.

If the ray reaches t_max without scattering/absorption, it processes the
surface hit normally. This allows proper bounded medium traversal.
"""
struct VPMediumSampleWorkItem
    # Ray info
    ray::Raycore.Ray
    depth::Int32
    t_max::Float32                  # Distance to surface (bounds delta tracking)

    # Path state
    lambda::Wavelengths
    beta::SpectralRadiance
    r_u::SpectralRadiance
    r_l::SpectralRadiance
    pixel_index::Int32
    eta_scale::Float32
    specular_bounce::Bool
    any_non_specular_bounces::Bool
    prev_intr_p::Point3f
    prev_intr_n::Vec3f

    # Current medium
    medium_idx::SetKey

    # Surface hit info (used if ray reaches t_max without medium event)
    # If t_max == Inf, these are invalid (ray escaped)
    has_surface_hit::Bool
    hit_pi::Point3f
    hit_n::Vec3f
    hit_ns::Vec3f
    hit_uv::Point2f
    hit_material_idx::SetKey
    hit_interface::MediumInterfaceIdx  # For medium transitions at surface
end

# Constructor from VPRayWorkItem with surface hit
function VPMediumSampleWorkItem(
    work::VPRayWorkItem,
    t_max::Float32,
    pi::Point3f, n::Vec3f, ns::Vec3f, uv::Point2f, mat_idx::SetKey,
    interface::MediumInterfaceIdx
)
    VPMediumSampleWorkItem(
        work.ray, work.depth, t_max,
        work.lambda, work.beta, work.r_u, work.r_l,
        work.pixel_index, work.eta_scale,
        work.specular_bounce, work.any_non_specular_bounces,
        work.prev_intr_p, work.prev_intr_n,
        work.medium_idx,
        true, pi, n, ns, uv, mat_idx, interface
    )
end

# Constructor from VPRayWorkItem for escaped rays (no surface hit)
function VPMediumSampleWorkItem(work::VPRayWorkItem)
    VPMediumSampleWorkItem(
        work.ray, work.depth, Inf32,
        work.lambda, work.beta, work.r_u, work.r_l,
        work.pixel_index, work.eta_scale,
        work.specular_bounce, work.any_non_specular_bounces,
        work.prev_intr_p, work.prev_intr_n,
        work.medium_idx,
        false, Point3f(0f0), Vec3f(0f0, 0f0, 1f0), Vec3f(0f0, 0f0, 1f0),
        Point2f(0f0, 0f0), SetKey(), MediumInterfaceIdx(SetKey(), SetKey(), SetKey())
    )
end

# ============================================================================
# Medium Scatter Work Item
# ============================================================================

"""
    VPMediumScatterWorkItem

Work item for a real scattering event in a participating medium.
Created when delta tracking samples a real scatter (not null scatter).
"""
struct VPMediumScatterWorkItem
    p::Point3f                      # Scatter position
    wo::Vec3f                       # Outgoing direction (toward camera)
    time::Float32
    lambda::Wavelengths
    pixel_index::Int32
    beta::SpectralRadiance          # Path throughput at scatter point
    r_u::SpectralRadiance           # Rescaled PDF
    depth::Int32
    medium_idx::SetKey         # Which medium we're in
    g::Float32                      # HG asymmetry at this point
end

# ============================================================================
# Material Evaluation Work Item (with medium)
# ============================================================================

"""
    VPMaterialEvalWorkItem

Surface interaction work item for VolPath.
Similar to PWMaterialEvalWorkItem but includes medium transition info.
"""
struct VPMaterialEvalWorkItem
    # Surface interaction data
    pi::Point3f                     # Intersection point
    n::Vec3f                        # Geometric normal
    ns::Vec3f                       # Shading normal
    wo::Vec3f                       # Outgoing direction
    uv::Point2f                     # Texture coordinates
    material_idx::SetKey            # Material index
    interface::MediumInterfaceIdx   # For medium transitions

    # Path state
    lambda::Wavelengths
    pixel_index::Int32
    beta::SpectralRadiance
    r_u::SpectralRadiance
    r_l::SpectralRadiance
    depth::Int32
    eta_scale::Float32
    specular_bounce::Bool
    any_non_specular_bounces::Bool

    # Previous interaction (for MIS)
    prev_intr_p::Point3f
    prev_intr_n::Vec3f

    # Medium transition info
    current_medium::SetKey     # Medium ray was traveling through
end

# Note: Constructor from VPHitSurfaceWorkItem is defined after VPHitSurfaceWorkItem struct

# ============================================================================
# Shadow Ray Work Item
# ============================================================================

"""
    VPShadowRayWorkItem

Shadow ray for direct lighting (works for both surface and volume scattering).
"""
struct VPShadowRayWorkItem
    ray::Raycore.Ray
    t_max::Float32
    lambda::Wavelengths
    Ld::SpectralRadiance            # Direct lighting contribution (if unoccluded)
    r_u::SpectralRadiance           # MIS weight numerator
    r_l::SpectralRadiance           # MIS weight denominator
    pixel_index::Int32
    medium_idx::SetKey         # Medium the shadow ray travels through
end

# ============================================================================
# Escaped Ray Work Item
# ============================================================================

"""
    VPEscapedRayWorkItem

Ray that escaped the scene (for environment lighting).
"""
struct VPEscapedRayWorkItem
    ray_d::Vec3f                    # Ray direction
    lambda::Wavelengths
    pixel_index::Int32
    beta::SpectralRadiance
    r_u::SpectralRadiance
    r_l::SpectralRadiance
    depth::Int32
    specular_bounce::Bool
    prev_intr_p::Point3f
    prev_intr_n::Vec3f
end

# Constructor from VPRayWorkItem
function VPEscapedRayWorkItem(work::VPRayWorkItem)
    VPEscapedRayWorkItem(
        work.ray.d, work.lambda, work.pixel_index,
        work.beta, work.r_u, work.r_l,
        work.depth, work.specular_bounce,
        work.prev_intr_p, work.prev_intr_n
    )
end

# Constructor from VPMediumSampleWorkItem (with updated beta/r_u/r_l from delta tracking)
function VPEscapedRayWorkItem(
    work::VPMediumSampleWorkItem,
    beta::SpectralRadiance, r_u::SpectralRadiance, r_l::SpectralRadiance
)
    VPEscapedRayWorkItem(
        work.ray.d, work.lambda, work.pixel_index,
        beta, r_u, r_l,
        work.depth, work.specular_bounce,
        work.prev_intr_p, work.prev_intr_n
    )
end

# ============================================================================
# Hit Surface Work Item (intermediate, before material eval)
# ============================================================================

"""
    VPHitSurfaceWorkItem

Intermediate work item when ray hits a surface.
Used to separate intersection from material evaluation.
"""
struct VPHitSurfaceWorkItem
    # Ray that hit
    ray::Raycore.Ray

    # Hit info
    pi::Point3f
    n::Vec3f
    ns::Vec3f
    uv::Point2f
    material_idx::SetKey
    interface::MediumInterfaceIdx  # For medium transitions

    # Path state
    lambda::Wavelengths
    pixel_index::Int32
    beta::SpectralRadiance
    r_u::SpectralRadiance
    r_l::SpectralRadiance
    depth::Int32
    eta_scale::Float32
    specular_bounce::Bool
    any_non_specular_bounces::Bool
    prev_intr_p::Point3f
    prev_intr_n::Vec3f

    # Medium info
    current_medium::SetKey

    # Distance traveled through medium (for transmittance)
    t_hit::Float32
end

# Constructor from VPRayWorkItem with hit geometry
function VPHitSurfaceWorkItem(
    work::VPRayWorkItem,
    pi::Point3f, n::Vec3f, ns::Vec3f, uv::Point2f, mat_idx::SetKey,
    interface::MediumInterfaceIdx,
    t_hit::Float32
)
    VPHitSurfaceWorkItem(
        work.ray,
        pi, n, ns, uv, mat_idx, interface,
        work.lambda, work.pixel_index,
        work.beta, work.r_u, work.r_l,
        work.depth, work.eta_scale,
        work.specular_bounce, work.any_non_specular_bounces,
        work.prev_intr_p, work.prev_intr_n,
        work.medium_idx, t_hit
    )
end

# Constructor from VPMediumSampleWorkItem (with updated beta/r_u/r_l from delta tracking)
function VPHitSurfaceWorkItem(
    work::VPMediumSampleWorkItem,
    beta::SpectralRadiance, r_u::SpectralRadiance, r_l::SpectralRadiance
)
    VPHitSurfaceWorkItem(
        work.ray,
        work.hit_pi, work.hit_n, work.hit_ns, work.hit_uv, work.hit_material_idx, work.hit_interface,
        work.lambda, work.pixel_index,
        beta, r_u, r_l,
        work.depth, work.eta_scale,
        work.specular_bounce, work.any_non_specular_bounces,
        work.prev_intr_p, work.prev_intr_n,
        work.medium_idx, work.t_max
    )
end

# Constructor for VPMaterialEvalWorkItem from VPHitSurfaceWorkItem
# (wo and material_idx are computed externally, e.g. after MixMaterial resolution)
function VPMaterialEvalWorkItem(work::VPHitSurfaceWorkItem, wo::Vec3f, material_idx::SetKey)
    VPMaterialEvalWorkItem(
        work.pi, work.n, work.ns, wo, work.uv, material_idx, work.interface,
        work.lambda, work.pixel_index,
        work.beta, work.r_u, work.r_l,
        work.depth, work.eta_scale,
        work.specular_bounce, work.any_non_specular_bounces,
        work.prev_intr_p, work.prev_intr_n,
        work.current_medium
    )
end
