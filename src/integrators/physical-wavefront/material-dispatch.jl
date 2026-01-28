# Material dispatch for PhysicalWavefront path tracing
# Uses with_index for type-stable dispatch over StaticMultiTypeVec
# Each material type implements sample_bsdf_spectral() from spectral-eval.jl
#
# NOTE: `materials` is a StaticMultiTypeVec containing both materials and textures.
# It's passed as the `textures` argument to spectral functions for eval_tex calls.

# ============================================================================
# Helper functions for with_index dispatch
# ============================================================================

# sample_bsdf_spectral helper
@propagate_inbounds function _sample_spectral_impl(mat, table, ctx, wo, ns, uv, lambda, u, rng, regularize)
    return sample_bsdf_spectral(table, mat, ctx, wo, ns, uv, lambda, u, rng, regularize)
end

# evaluate_bsdf_spectral helper
@propagate_inbounds function _evaluate_spectral_impl(mat, table, ctx, wo, wi, ns, uv, lambda)
    return evaluate_bsdf_spectral(table, mat, ctx, wo, wi, ns, uv, lambda)
end

# get_emission_spectral helper (with direction)
@propagate_inbounds function _emission_spectral_impl(mat, table, ctx, wo, n, uv, lambda)
    return get_emission_spectral(table, mat, ctx, wo, n, uv, lambda)
end

# get_emission_spectral helper (UV only)
@propagate_inbounds function _emission_spectral_uv_impl(mat, table, ctx, uv, lambda)
    return get_emission_spectral(table, mat, ctx, uv, lambda)
end


# get_albedo_spectral helper
@propagate_inbounds function _albedo_spectral_impl(mat, table, ctx, uv, lambda)
    return get_albedo_spectral(table, mat, ctx, uv, lambda)
end

# ============================================================================
# Spectral Material Sampling Dispatch
# ============================================================================

"""
    sample_spectral_material(table, materials::StaticMultiTypeVec, idx, wo, ns, uv, lambda, u, rng, regularize=false)

Type-stable dispatch for spectral BSDF sampling.
Returns SpectralBSDFSample from the appropriate material type.

When `regularize=true`, near-specular BSDFs will be roughened to reduce fireflies.
"""
@propagate_inbounds function sample_spectral_material(
    table::RGBToSpectrumTable, materials::StaticMultiTypeVec,
    idx::MaterialIndex,
    wo::Vec3f, ns::Vec3f, uv::Point2f,
    lambda::Wavelengths, u::Point2f, rng::Float32,
    regularize::Bool = false
)
    return with_index(_sample_spectral_impl, materials, idx, table, materials, wo, ns, uv, lambda, u, rng, regularize)
end

# ============================================================================
# Spectral BSDF Evaluation Dispatch
# ============================================================================

"""
    evaluate_spectral_material(table, materials::StaticMultiTypeVec, idx, wo, wi, ns, uv, lambda)

Type-stable dispatch for spectral BSDF evaluation.
Returns (f::SpectralRadiance, pdf::Float32).
"""
@propagate_inbounds function evaluate_spectral_material(
    table::RGBToSpectrumTable, materials::StaticMultiTypeVec,
    idx::MaterialIndex,
    wo::Vec3f, wi::Vec3f, ns::Vec3f, uv::Point2f,
    lambda::Wavelengths
)
    return with_index(_evaluate_spectral_impl, materials, idx, table, materials, wo, wi, ns, uv, lambda)
end

# ============================================================================
# Emission Dispatch
# ============================================================================

"""
    get_emission_spectral_dispatch(table, materials::StaticMultiTypeVec, idx, wo, n, uv, lambda)

Type-stable dispatch for getting spectral emission from materials.
Returns SpectralRadiance (zero for non-emissive materials).
"""
@propagate_inbounds function get_emission_spectral_dispatch(
    table::RGBToSpectrumTable, materials::StaticMultiTypeVec,
    idx::MaterialIndex,
    wo::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
)
    return with_index(_emission_spectral_impl, materials, idx, table, materials, wo, n, uv, lambda)
end

"""
    get_emission_spectral_uv_dispatch(table, materials::StaticMultiTypeVec, idx, uv, lambda)

Type-stable dispatch for getting spectral emission without directional check.
Returns SpectralRadiance (zero for non-emissive materials).
"""
@propagate_inbounds function get_emission_spectral_uv_dispatch(
    table::RGBToSpectrumTable, materials::StaticMultiTypeVec,
    idx::MaterialIndex,
    uv::Point2f, lambda::Wavelengths
)
    return with_index(_emission_spectral_uv_impl, materials, idx, table, materials, uv, lambda)
end

# ============================================================================
# Emissive Check Dispatch
# ============================================================================

"""
    is_emissive(materials::StaticMultiTypeVec, idx::HeteroVecIndex)

Type-stable dispatch for checking if a material/medium is emissive.
Returns Bool. Works for both materials and media via element-level dispatch.
"""
@propagate_inbounds function is_emissive(
    collection::StaticMultiTypeVec, idx::HeteroVecIndex
)::Bool
    return with_index(is_emissive, collection, idx)
end

"""
    is_pure_emissive(materials::StaticMultiTypeVec, idx::HeteroVecIndex)

Type-stable dispatch for checking if a material is purely emissive (no BSDF).
Returns Bool.
"""
@propagate_inbounds function is_pure_emissive(
    collection::StaticMultiTypeVec, idx::HeteroVecIndex
)::Bool
    return with_index(is_pure_emissive, collection, idx)
end

# ============================================================================
# Albedo Dispatch (for denoising auxiliary buffers)
# ============================================================================

"""
    get_albedo_spectral_dispatch(table, materials::StaticMultiTypeVec, idx, uv, lambda)

Type-stable dispatch for getting material albedo for denoising.
Returns SpectralRadiance.
"""
@propagate_inbounds function get_albedo_spectral_dispatch(
    table::RGBToSpectrumTable, materials::StaticMultiTypeVec,
    idx::MaterialIndex,
    uv::Point2f, lambda::Wavelengths
)
    return with_index(_albedo_spectral_impl, materials, idx, table, materials, uv, lambda)
end

# ============================================================================
# Combined Material Evaluation for Wavefront Pipeline
# ============================================================================

"""
    PWMaterialEvalResult

Result of evaluating a material for the wavefront pipeline.
"""
struct PWMaterialEvalResult
    # BSDF sample result
    sample::SpectralBSDFSample

    # Emission (if material is emissive)
    Le::SpectralRadiance

    # Material properties for path decisions
    is_emissive::Bool
end

@propagate_inbounds function PWMaterialEvalResult()
    return PWMaterialEvalResult(
        SpectralBSDFSample(),
        SpectralRadiance(0f0),
        false
    )
end

# Helper for evaluate_material_complete
@propagate_inbounds function _eval_complete_impl(mat, table, ctx, wo, ns, n, uv, lambda, u, rng, regularize)
    sample = sample_bsdf_spectral(table, mat, ctx, wo, ns, uv, lambda, u, rng, regularize)
    Le = get_emission_spectral(table, mat, ctx, wo, n, uv, lambda)
    is_em = is_emissive(mat)
    return PWMaterialEvalResult(sample, Le, is_em)
end

"""
    evaluate_material_complete(table, materials::StaticMultiTypeVec, idx, wo, ns, n, uv, lambda, u, rng, regularize=false)

Complete material evaluation for wavefront pipeline.
Returns PWMaterialEvalResult with BSDF sample and emission.

When `regularize=true`, near-specular BSDFs will be roughened to reduce fireflies.
"""
@propagate_inbounds function evaluate_material_complete(
    table::RGBToSpectrumTable, materials::StaticMultiTypeVec,
    idx::MaterialIndex,
    wo::Vec3f, ns::Vec3f, n::Vec3f, uv::Point2f,
    lambda::Wavelengths, u::Point2f, rng::Float32,
    regularize::Bool = false
)
    return with_index(_eval_complete_impl, materials, idx, table, materials, wo, ns, n, uv, lambda, u, rng, regularize)
end

# ============================================================================
# Spawn Indirect Ray
# ============================================================================

"""
    spawn_spectral_ray(item::PWMaterialEvalWorkItem, wi::Vec3f, new_beta::SpectralRadiance,
                       is_specular::Bool, pdf::Float32, eta_scale::Float32) -> PWRayWorkItem

Create a new ray work item for indirect lighting from a material evaluation.
"""
@propagate_inbounds function spawn_spectral_ray(
    item::PWMaterialEvalWorkItem,
    wi::Vec3f,
    new_beta::SpectralRadiance,
    is_specular::Bool,
    pdf::Float32,
    eta_scale::Float32
)
    # Offset origin slightly along normal to avoid self-intersection
    offset = 1f-4 * item.n

    # Determine offset direction based on wi direction relative to normal
    ray_origin = if dot(wi, item.n) > 0f0
        Point3f((item.pi + offset)...)
    else
        Point3f((item.pi - offset)...)
    end

    ray = Raycore.Ray(o=ray_origin, d=wi)

    # Update path state
    new_depth = item.depth + Int32(1)
    any_non_specular = item.any_non_specular_bounces || !is_specular

    # Update r_u (unidirectional PDF) for non-specular bounces
    new_r_u = if is_specular
        item.r_u
    else
        item.r_u * SpectralRadiance(pdf)
    end

    # Create light sample context for MIS
    prev_ctx = PWLightSampleContext(item.pi, item.n, item.ns)

    return PWRayWorkItem(
        ray,
        new_depth,
        item.lambda,
        item.pixel_index,
        new_beta,
        new_r_u,
        SpectralRadiance(1f0),  # r_l reset for new path segment
        prev_ctx,
        eta_scale,
        is_specular,
        any_non_specular
    )
end

# ============================================================================
# Russian Roulette
# ============================================================================

"""
    russian_roulette_spectral(beta::SpectralRadiance, depth::Int32, rr_sample::Float32, min_depth::Int32=3)

Apply Russian roulette for path termination.
Returns (should_continue::Bool, new_beta::SpectralRadiance).
"""
@propagate_inbounds function russian_roulette_spectral(
    beta::SpectralRadiance,
    depth::Int32,
    rr_sample::Float32,
    min_depth::Int32=Int32(3)
)
    # Don't apply RR for first few bounces
    if depth <= min_depth
        return (true, beta)
    end

    # Use max component of throughput for continuation probability
    # Higher throughput = higher probability of continuation
    max_comp = max_component(beta)
    q = max(0.05f0, 1f0 - max_comp)

    if rr_sample < q
        # Terminate path
        return (false, beta)
    else
        # Continue with boosted beta
        scale = 1f0 / (1f0 - q)
        return (true, beta * scale)
    end
end

# ============================================================================
# MediumInterface Dispatch Functions
# ============================================================================

# These functions are used by VolPath to handle medium transitions at surfaces.
# MediumInterface is defined in integrators/volpath/media.jl

"""
    has_medium_interface(mat) -> Bool

Check if a material is a MediumInterface (defines medium boundary).
"""
@propagate_inbounds has_medium_interface(::Material) = false
@propagate_inbounds has_medium_interface(::MediumInterface) = true
@propagate_inbounds has_medium_interface(::MediumInterfaceIdx) = true

# Helper for has_medium_interface_dispatch
@propagate_inbounds _has_medium_interface_impl(mat) = has_medium_interface(mat)

"""
    has_medium_interface_dispatch(materials::StaticMultiTypeVec, idx::MaterialIndex) -> Bool

Type-stable dispatch for checking if a material defines a medium boundary.
"""
@propagate_inbounds function has_medium_interface_dispatch(
    materials::StaticMultiTypeVec, idx::MaterialIndex
)::Bool
    return with_index(_has_medium_interface_impl, materials, idx)
end

"""
    get_medium_index_for_direction(mat::MediumInterface, wi::Vec3f, n::Vec3f) -> MediumIndex

Get the medium index a ray enters when crossing a MediumInterface surface.
- If dot(wi, n) > 0: ray going in direction of normal -> outside medium
- If dot(wi, n) < 0: ray going against normal -> inside medium
"""
@propagate_inbounds function get_medium_index_for_direction(mi::MediumInterface, wi::Vec3f, n::Vec3f)
    return get_medium_index(mi, wi, n)
end

# MediumInterfaceIdx version (after scene building, MediumInterface is converted to this)
@propagate_inbounds function get_medium_index_for_direction(mi::MediumInterfaceIdx, wi::Vec3f, n::Vec3f)
    return get_medium_index(mi, wi, n)
end

# Fallback for non-MediumInterface materials (returns vacuum)
@propagate_inbounds get_medium_index_for_direction(::Material, ::Vec3f, ::Vec3f) = MediumIndex()

# Helper for get_medium_index_for_direction_dispatch
@propagate_inbounds function _get_medium_idx_impl(mat, wi, n)
    return get_medium_index_for_direction(mat, wi, n)
end

"""
    get_medium_index_for_direction_dispatch(materials::StaticMultiTypeVec, idx::MaterialIndex, wi::Vec3f, n::Vec3f)

Type-stable dispatch for getting the new medium index after crossing a surface.
Returns the MediumIndex from MediumInterface, or MediumIndex() (vacuum) for regular materials.
"""
@propagate_inbounds function get_medium_index_for_direction_dispatch(
    materials::StaticMultiTypeVec, idx::MaterialIndex, wi::Vec3f, n::Vec3f
)
    return with_index(_get_medium_idx_impl, materials, idx, wi, n)
end
