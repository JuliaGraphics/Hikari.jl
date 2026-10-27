# Work items for VolPath wavefront integrator
# Extends PhysicalWavefront work items with medium support

# ============================================================================
# Ray Samples: generated inline in each consumer
# ============================================================================

# pbrt-v4 caches Sobol samples in a per-pixel `PixelSampleState.RaySamples`
# buffer filled by `WavefrontPathIntegrator::GenerateRaySamples` once per
# bounce. There is no such struct here: mirroring it with a per-pixel SOA
# buffer and a kernel to fill it measures 58 % of total GPU time on killeroo
# (RTX 4000 Ada), against ~0.1 % in the actual path-tracing kernels, so each
# consumer (`surface_direct_lighting_inner!`, `evaluate_material_inner!`,
# `medium_direct_lighting_inner!`, `medium_scatter_inner!`) generates the
# dimensions it needs inline.

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

"Prepare a curved chord before either hardware or software intersection."
@propagate_inbounds function prepare_medium_work(work::VPRayWorkItem, media)
    has_medium(work.medium_idx) || return work
    ray=work.ray
    h=Raycore.with_index(medium_step_limit,media,work.medium_idx,ray.o,ray.d)
    isfinite(h) || return work
    d=Raycore.with_index(medium_start_direction,media,work.medium_idx,ray.o,ray.d,h)
    # A curved transport step is valid only over this chord. Querying geometry
    # beyond it is unnecessary (and feeds an unbounded ray to HW traversal
    # despite having a finite integration interval). A miss still goes through
    # the medium continuation, which re-intersects from the next endpoint.
    chord=Raycore.Ray(o=ray.o,d=d,t_min=ray.t_min,t_max=min(ray.t_max,h),time=ray.time)
    VPRayWorkItem(chord,work.depth,work.lambda,work.pixel_index,
        work.beta,work.r_u,work.r_l,work.prev_intr_p,work.prev_intr_n,
        work.eta_scale,work.specular_bounce,work.any_non_specular_bounces,work.medium_idx)
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

If the ray reaches t_max without scattering/absorption, its surface hit goes to
the material's shading kernel as any other does: `hit` is the record the trace
found, which that kernel expands. A miss record when the ray escapes.
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

    # The surface the ray reaches if no medium event stops it first.
    hit::Raycore.HitRecord
end

"""Whether the medium ray reaches a surface at `t_max`, rather than escaping."""
@inline has_surface_hit(work::VPMediumSampleWorkItem) = work.hit.kind != Raycore.HIT_MISS

# Delta tracking is bounded by the hit, or by the ray's own extent on a miss.
function VPMediumSampleWorkItem(work::VPRayWorkItem, hit::Raycore.HitRecord)
    t_max = hit.kind == Raycore.HIT_MISS ? work.ray.t_max : hit.t
    VPMediumSampleWorkItem(
        work.ray, work.depth, t_max,
        work.lambda, work.beta, work.r_u, work.r_l,
        work.pixel_index, work.eta_scale,
        work.specular_bounce, work.any_non_specular_bounces,
        work.prev_intr_p, work.prev_intr_n,
        work.medium_idx,
        hit,
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
    eta_scale::Float32              # Accumulated IOR ratio (carried from path state)
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
    ps::Point3f                     # Where shadow rays leave (see `vp_shadow_point`)
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
    refract::Bool              # Through Snell's window: bends at surfaces, `Ld` waits for the sky (see `snells_window`)
end

# ============================================================================
# Surface Hit
# ============================================================================

"""
    VPSurfaceHit

A ray's surface hit as the trace leaves it for the material's shading kernel:
the hit record and where the ray's path state is. `path > 0` is its index in the
round's ray queue; `path < 0` is `-index` into the arrivals, the paths a medium
delivered to their surface with its transmittance applied. The shading kernel
expands the record, so the surface is computed once, where it is shaded.
"""
struct VPSurfaceHit
    path::Int32
    hit::Raycore.HitRecord
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

A surface hit expanded for shading: the path state with the surface's geometry,
shading frame and partial derivatives for texture filtering (pbrt-v4 style).
Built in registers by the shading kernel from a [`VPSurfaceHit`](@ref).
"""
struct VPHitSurfaceWorkItem
    # Ray that hit
    ray::Raycore.Ray

    # Hit geometry (matching pbrt-v4 SurfaceInteraction decomposition)
    pi::Point3f                     # Intersection point
    ps::Point3f                     # Where shadow rays leave (see `vp_shadow_point`)
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
    # The previous interaction, for the MIS weight of emission the BSDF ray
    # reaches from it.
    prev_intr_p::Point3f
    prev_intr_n::Vec3f

    # Medium info
    current_medium::SetKey
end

# Constructor from VPRayWorkItem with hit geometry.
function VPHitSurfaceWorkItem(
    work::VPRayWorkItem,
    pi::Point3f, ps::Point3f, n::Vec3f, dpdu::Vec3f, dpdv::Vec3f,
    ns::Vec3f, dpdus::Vec3f, dpdvs::Vec3f,
    uv::Point2f, mat_idx::SetKey,
    interface::MediumInterfaceIdx,
    face_idx::UInt32, bary::SVector{3, Float32},
)
    VPHitSurfaceWorkItem(
        work.ray,
        pi, ps, n, dpdu, dpdv, ns, dpdus, dpdvs, uv, mat_idx, interface,
        face_idx, bary,
        work.lambda, work.pixel_index,
        work.beta, work.r_u, work.r_l,
        work.depth, work.eta_scale,
        work.specular_bounce, work.any_non_specular_bounces,
        work.prev_intr_p, work.prev_intr_n,
        work.medium_idx,
    )
end

# Constructor for VPMaterialEvalWorkItem from VPHitSurfaceWorkItem
# (wo and material_idx are computed externally, e.g. after Mix resolution).
function VPMaterialEvalWorkItem(work::VPHitSurfaceWorkItem, wo::Vec3f, material_idx::SetKey)
    VPMaterialEvalWorkItem(
        work.pi, work.ps, work.n, work.dpdu, work.dpdv, work.ray.time,
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
