# Work items for VolPath wavefront integrator
# Extends PhysicalWavefront work items with medium support

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
    medium_idx::MediumIndex         # Current medium (0 = vacuum)
end

function VPRayWorkItem(
    ray::Raycore.Ray,
    lambda::Wavelengths,
    pixel_index::Int32;
    medium_idx::MediumIndex = MediumIndex()
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
    medium_idx::MediumIndex

    # Surface hit info (used if ray reaches t_max without medium event)
    # If t_max == Inf, these are invalid (ray escaped)
    has_surface_hit::Bool
    hit_pi::Point3f
    hit_n::Vec3f
    hit_ns::Vec3f
    hit_uv::Point2f
    hit_material_idx::MaterialIndex
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
    medium_idx::MediumIndex         # Which medium we're in
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
    material_idx::MaterialIndex     # Material index

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
    current_medium::MediumIndex     # Medium ray was traveling through
    # Note: After surface interaction, new medium depends on refraction/reflection
end

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
    medium_idx::MediumIndex         # Medium the shadow ray travels through
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
    material_idx::MaterialIndex

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
    current_medium::MediumIndex

    # Distance traveled through medium (for transmittance)
    t_hit::Float32
end
