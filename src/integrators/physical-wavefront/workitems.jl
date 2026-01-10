# Work item structures for PhysicalWavefront path tracing
# These are the data structures that flow between GPU kernels
# Adapted from PbrtWavefront to use Hikari's MaterialIndex

# ============================================================================
# Supporting Structures
# ============================================================================

"""
    PWRaySamples

Random samples used for direct and indirect lighting at each path vertex.
"""
struct PWRaySamples
    # Direct lighting samples
    direct_u::Point2f        # 2D sample for light sampling
    direct_uc::Float32       # 1D sample for light selection

    # Indirect lighting samples
    indirect_u::Point2f      # 2D sample for BSDF sampling
    indirect_uc::Float32     # 1D sample for BSDF component selection
    indirect_rr::Float32     # Russian roulette sample
end

@propagate_inbounds function PWRaySamples()
    return PWRaySamples(
        Point2f(0f0, 0f0), 0f0,
        Point2f(0f0, 0f0), 0f0, 0f0
    )
end

"""
    PWLightSampleContext

Context for sampling lights - position, geometric normal, shading normal.
Used for MIS weight computation.
"""
struct PWLightSampleContext
    p::Point3f      # Position
    n::Vec3f        # Geometric normal
    ns::Vec3f       # Shading normal
end

@propagate_inbounds PWLightSampleContext() = PWLightSampleContext(
    Point3f(0f0, 0f0, 0f0),
    Vec3f(0f0, 0f0, 1f0),
    Vec3f(0f0, 0f0, 1f0)
)

# ============================================================================
# Main Work Items
# ============================================================================

"""
    PWRayWorkItem

A ray to be traced, along with full path state.
This is the main work item flowing through the ray queue.
"""
struct PWRayWorkItem
    ray::Raycore.Ray
    depth::Int32
    lambda::Wavelengths
    pixel_index::Int32
    beta::SpectralRadiance       # Path throughput (spectral)
    r_u::SpectralRadiance        # Unidirectional path PDF for MIS
    r_l::SpectralRadiance        # Light path PDF for MIS
    prev_intr_ctx::PWLightSampleContext
    eta_scale::Float32
    specular_bounce::Bool
    any_non_specular_bounces::Bool
end

"""
    PWRayWorkItem(ray, lambda, pixel_index)

Create a new camera ray work item with default path state.
"""
@propagate_inbounds function PWRayWorkItem(ray::Raycore.Ray, lambda::Wavelengths, pixel_index::Int32)
    return PWRayWorkItem(
        ray,
        Int32(0),
        lambda,
        pixel_index,
        SpectralRadiance(1f0),
        SpectralRadiance(1f0),
        SpectralRadiance(1f0),
        PWLightSampleContext(),
        1f0,
        false,
        false
    )
end

"""
    PWEscapedRayWorkItem

A ray that escaped the scene (missed all geometry).
Used to compute contribution from environment/infinite lights.
"""
struct PWEscapedRayWorkItem
    ray_o::Point3f
    ray_d::Vec3f
    depth::Int32
    lambda::Wavelengths
    pixel_index::Int32
    beta::SpectralRadiance
    specular_bounce::Bool
    r_u::SpectralRadiance
    r_l::SpectralRadiance
    prev_intr_ctx::PWLightSampleContext
end

@propagate_inbounds function PWEscapedRayWorkItem(w::PWRayWorkItem)
    return PWEscapedRayWorkItem(
        w.ray.o,
        w.ray.d,
        w.depth,
        w.lambda,
        w.pixel_index,
        w.beta,
        w.specular_bounce,
        w.r_u,
        w.r_l,
        w.prev_intr_ctx
    )
end

"""
    PWHitAreaLightWorkItem

A ray that hit an emissive surface (area light).
Contains info needed to compute emission contribution with MIS.
"""
struct PWHitAreaLightWorkItem
    # Hit geometry
    p::Point3f              # Hit point
    n::Vec3f                # Geometric normal at hit point
    uv::Point2f             # UV coordinates for texture lookup
    wo::Vec3f               # Outgoing direction (toward camera/previous vertex)

    # Path state
    lambda::Wavelengths
    depth::Int32
    beta::SpectralRadiance
    r_u::SpectralRadiance
    r_l::SpectralRadiance
    prev_intr_ctx::PWLightSampleContext
    specular_bounce::Bool
    pixel_index::Int32

    # Material reference (for EmissiveMaterial lookup)
    material_idx::MaterialIndex
end

"""
    PWShadowRayWorkItem

A shadow ray for direct lighting visibility test.
"""
struct PWShadowRayWorkItem
    ray::Raycore.Ray
    t_max::Float32
    lambda::Wavelengths
    Ld::SpectralRadiance    # Tentative radiance contribution if unoccluded
    r_u::SpectralRadiance
    r_l::SpectralRadiance
    pixel_index::Int32
end

"""
    PWMaterialEvalWorkItem

Work item for material/BSDF evaluation after ray intersection.
Contains all intersection data needed for shading.
"""
struct PWMaterialEvalWorkItem
    # Intersection geometry
    pi::Point3f             # Intersection point
    n::Vec3f                # Geometric normal
    dpdu::Vec3f             # Partial derivative of p wrt u
    dpdv::Vec3f             # Partial derivative of p wrt v

    # Shading geometry
    ns::Vec3f               # Shading normal
    dpdus::Vec3f            # Shading dpdu
    dpdvs::Vec3f            # Shading dpdv
    uv::Point2f             # Texture coordinates

    # Path state
    depth::Int32
    lambda::Wavelengths
    pixel_index::Int32
    any_non_specular_bounces::Bool
    wo::Vec3f               # Outgoing direction (toward camera)
    beta::SpectralRadiance
    r_u::SpectralRadiance
    eta_scale::Float32

    # Material reference - uses Hikari's MaterialIndex for type-stable dispatch
    material_idx::MaterialIndex
end

# ============================================================================
# Pixel State
# ============================================================================

"""
    PWPixelSampleState

Per-pixel state accumulated during spectral path tracing.
"""
struct PWPixelSampleState
    p_pixel::Point2i             # Pixel coordinates
    L::SpectralRadiance          # Accumulated spectral radiance
    lambda::Wavelengths          # Sampled wavelengths for this pixel
    filter_weight::Float32       # Filter weight for this sample
    camera_ray_weight::SpectralRadiance  # Weight from camera ray generation
    samples::PWRaySamples        # Current ray samples
end

@propagate_inbounds function PWPixelSampleState(p_pixel::Point2i, lambda::Wavelengths)
    return PWPixelSampleState(
        p_pixel,
        SpectralRadiance(0f0),
        lambda,
        1f0,
        SpectralRadiance(1f0),
        PWRaySamples()
    )
end

# ============================================================================
# MIS (Multiple Importance Sampling) Helper
# ============================================================================

"""
    PWMISContext

Context needed for computing MIS weights between light and BSDF sampling.
"""
struct PWMISContext
    r_u::SpectralRadiance    # BSDF sampling PDF contribution
    r_l::SpectralRadiance    # Light sampling PDF contribution
end

@propagate_inbounds PWMISContext() = PWMISContext(SpectralRadiance(1f0), SpectralRadiance(1f0))

"""
    balance_heuristic(pdf_f::Float32, pdf_g::Float32) -> Float32

Balance heuristic for MIS: w_f = pdf_f / (pdf_f + pdf_g)
"""
@propagate_inbounds function balance_heuristic(pdf_f::Float32, pdf_g::Float32)
    return pdf_f / (pdf_f + pdf_g)
end

"""
    power_heuristic(pdf_f::Float32, pdf_g::Float32, beta::Float32=2f0) -> Float32

Power heuristic for MIS: w_f = pdf_f^beta / (pdf_f^beta + pdf_g^beta)
Default beta=2 (squared terms).
"""
@propagate_inbounds function power_heuristic(pdf_f::Float32, pdf_g::Float32, beta::Float32=2f0)
    f = pdf_f^beta
    g = pdf_g^beta
    return f / (f + g)
end
