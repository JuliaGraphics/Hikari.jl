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
    hit_dpdu::Vec3f                 # Geometric position derivative w.r.t. u
    hit_dpdv::Vec3f                 # Geometric position derivative w.r.t. v
    hit_ns::Vec3f
    hit_dpdus::Vec3f                # Shading position derivative w.r.t. u
    hit_dpdvs::Vec3f                # Shading position derivative w.r.t. v
    hit_uv::Point2f
    hit_material_idx::SetKey
    hit_interface::MediumInterfaceIdx  # For medium transitions at surface
    hit_face_idx::UInt32              # Triangle face index (for vertex colors)
    hit_bary::SVector{3, Float32}     # Barycentric coordinates (for vertex colors)
end

# Constructor from VPRayWorkItem with surface hit
function VPMediumSampleWorkItem(
    work::VPRayWorkItem,
    t_max::Float32,
    pi::Point3f, n::Vec3f, dpdu::Vec3f, dpdv::Vec3f,
    ns::Vec3f, dpdus::Vec3f, dpdvs::Vec3f,
    uv::Point2f, mat_idx::SetKey,
    interface::MediumInterfaceIdx,
    face_idx::UInt32, bary::SVector{3, Float32}
)
    VPMediumSampleWorkItem(
        work.ray, work.depth, t_max,
        work.lambda, work.beta, work.r_u, work.r_l,
        work.pixel_index, work.eta_scale,
        work.specular_bounce, work.any_non_specular_bounces,
        work.prev_intr_p, work.prev_intr_n,
        work.medium_idx,
        true, pi, n, dpdu, dpdv, ns, dpdus, dpdvs, uv, mat_idx, interface,
        face_idx, bary
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
        false, Point3f(0f0), Vec3f(0f0, 0f0, 1f0), Vec3f(0f0), Vec3f(0f0),
        Vec3f(0f0, 0f0, 1f0), Vec3f(0f0), Vec3f(0f0),
        Point2f(0f0, 0f0), SetKey(), MediumInterfaceIdx(SetKey(), SetKey(), SetKey()),
        UInt32(0), SVector{3, Float32}(0f0, 0f0, 0f0)
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

Following pbrt-v4's MaterialEvalWorkItem, stores surface partial derivatives
for texture filtering. Screen-space derivatives (dudx, dudy, dvdx, dvdy) are
computed on-the-fly during material evaluation using camera.Approximate_dp_dxy().
"""
struct VPMaterialEvalWorkItem
    # Surface interaction data (matching pbrt-v4 MaterialEvalWorkItem order)
    pi::Point3f                     # Intersection point
    n::Vec3f                        # Geometric normal
    dpdu::Vec3f                     # Geometric position derivative w.r.t. u
    dpdv::Vec3f                     # Geometric position derivative w.r.t. v
    time::Float32                   # Intersection time (for motion blur)
    ns::Vec3f                       # Shading normal
    dpdus::Vec3f                    # Shading position derivative w.r.t. u
    dpdvs::Vec3f                    # Shading position derivative w.r.t. v
    uv::Point2f                     # Texture coordinates

    # Per-vertex color support
    face_idx::UInt32                # Triangle face index (for vertex colors)
    bary::SVector{3, Float32}       # Barycentric coordinates (for vertex colors)

    wo::Vec3f                       # Outgoing direction
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
Stores surface partial derivatives for texture filtering (pbrt-v4 style).
"""
struct VPHitSurfaceWorkItem
    # Ray that hit
    ray::Raycore.Ray

    # Hit geometry (matching pbrt-v4 SurfaceInteraction decomposition)
    pi::Point3f                     # Intersection point
    n::Vec3f                        # Geometric normal
    dpdu::Vec3f                     # Geometric position derivative w.r.t. u
    dpdv::Vec3f                     # Geometric position derivative w.r.t. v
    ns::Vec3f                       # Shading normal
    dpdus::Vec3f                    # Shading position derivative w.r.t. u
    dpdvs::Vec3f                    # Shading position derivative w.r.t. v
    uv::Point2f                     # Texture coordinates
    material_idx::SetKey
    interface::MediumInterfaceIdx   # For medium transitions

    # Per-vertex color support
    face_idx::UInt32                # Triangle face index (for vertex colors)
    bary::SVector{3, Float32}       # Barycentric coordinates (for vertex colors)

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
    pi::Point3f, n::Vec3f, dpdu::Vec3f, dpdv::Vec3f,
    ns::Vec3f, dpdus::Vec3f, dpdvs::Vec3f,
    uv::Point2f, mat_idx::SetKey,
    interface::MediumInterfaceIdx,
    face_idx::UInt32, bary::SVector{3, Float32},
    t_hit::Float32
)
    VPHitSurfaceWorkItem(
        work.ray,
        pi, n, dpdu, dpdv, ns, dpdus, dpdvs, uv, mat_idx, interface,
        face_idx, bary,
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
        work.hit_pi, work.hit_n, work.hit_dpdu, work.hit_dpdv,
        work.hit_ns, work.hit_dpdus, work.hit_dpdvs,
        work.hit_uv, work.hit_material_idx, work.hit_interface,
        work.hit_face_idx, work.hit_bary,
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
        work.pi, work.n, work.dpdu, work.dpdv, work.ray.time,
        work.ns, work.dpdus, work.dpdvs, work.uv,
        work.face_idx, work.bary,
        wo, material_idx, work.interface,
        work.lambda, work.pixel_index,
        work.beta, work.r_u, work.r_l,
        work.depth, work.eta_scale,
        work.specular_bounce, work.any_non_specular_bounces,
        work.prev_intr_p, work.prev_intr_n,
        work.current_medium
    )
end
