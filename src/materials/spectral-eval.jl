# Spectral BSDF Evaluation Interface for PhysicalWavefront
# Enables spectral path tracing while keeping materials as RGB containers
#
# NOTE: All functions take a context parameter (historically named `textures`)
# which is a StaticMultiTypeVec containing both materials and textures.
# Use eval_tex(ctx, field, uv) to sample textures via TextureRef.

# ============================================================================
# Spectral BSDF Sample Result
# ============================================================================

"""
    SpectralBSDFSample

Result of sampling a BSDF with spectral wavelengths.
Used by PhysicalWavefront for spectral path tracing.
"""
struct SpectralBSDFSample
    wi::Vec3f                    # Sampled incident direction
    f::SpectralRadiance          # Spectral BSDF value
    pdf::Float32                 # Probability density
    is_specular::Bool            # True if delta distribution (no MIS needed)
    eta_scale::Float32           # Scale factor for refraction (1.0 for reflection)
end

# Default invalid sample
@propagate_inbounds SpectralBSDFSample() = SpectralBSDFSample(Vec3f(0, 0, 1), SpectralRadiance(), 0f0, false, 1f0)

# ============================================================================
# Spectral BSDF Evaluation for Each Material Type
# ============================================================================

# These functions extract material properties and evaluate them spectrally.
# They use the RGB-to-spectral uplift to convert material colors to wavelength-dependent values.

"""
    sample_bsdf_spectral(table, mat::MatteMaterial, textures, wo, n, uv, lambda, sample_u, rng) -> SpectralBSDFSample

Sample diffuse BSDF with spectral evaluation.
Uses pbrt-v4 convention: work in local shading space where n = (0,0,1).
"""
@propagate_inbounds function sample_bsdf_spectral(
    table::RGBToSpectrumTable, mat::MatteMaterial, textures,
    wo::Vec3f, n::Vec3f, uv::Point2f,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32,
    regularize::Bool = false
)
    # Check for grazing angle (wo perpendicular to shading normal)
    # This matches pbrt-v4's wo.z == 0 check in BSDF::Sample_f
    wo_dot_n = dot(wo, n)
    if abs(wo_dot_n) < 1f-6
        return SpectralBSDFSample()
    end

    # Get material properties (rng unused for diffuse)
    kd_rgb = eval_tex(textures, mat.Kd, uv)
    σ = eval_tex(textures, mat.σ, uv)

    # Clamp reflectance to [0,1] as per pbrt-v4
    kd_rgb = clamp(kd_rgb)

    # Uplift to spectral
    kd_spectral = uplift_rgb(table, kd_rgb, lambda)

    # Build local coordinate system from shading normal
    tangent, bitangent = coordinate_system(n)

    # Cosine-weighted hemisphere sampling (in local space, normal = +z)
    local_wi = cosine_sample_hemisphere(sample_u)
    cos_theta = local_wi[3]

    if cos_theta < 1f-6
        return SpectralBSDFSample()
    end

    # If wo is on the backside of the shading normal, flip wi to same hemisphere
    # This matches pbrt-v4's: if (wo.z < 0) wi.z *= -1;
    if wo_dot_n < 0f0
        local_wi = Vec3f(local_wi[1], local_wi[2], -local_wi[3])
    end

    # Transform to world space
    wi = local_to_world(local_wi, n, tangent, bitangent)
    wi = normalize(wi)

    # f = Kd / π (Lambertian), pdf = cos_theta / π
    # For Oren-Nayar with σ > 0, use the full model
    if σ > 0f0
        # Simplified Oren-Nayar: f ≈ Kd/π * (A + B * max(0, cos(φi-φo)) * sin(α) * tan(β))
        # For simplicity, we use a roughness-scaled Lambertian approximation
        roughness_factor = 1f0 - 0.5f0 * σ / (σ + 0.33f0)
        f = kd_spectral * (roughness_factor / Float32(π))
    else
        f = kd_spectral * (1f0 / Float32(π))
    end

    pdf = cos_theta / Float32(π)

    return SpectralBSDFSample(wi, f, pdf, false, 1f0)
end

"""
    sample_bsdf_spectral(table, mat::MirrorMaterial, textures, wo, n, uv, lambda, sample_u, rng) -> SpectralBSDFSample

Sample perfect specular reflection with spectral evaluation.
"""
@propagate_inbounds function sample_bsdf_spectral(
    table::RGBToSpectrumTable, mat::MirrorMaterial, textures,
    wo::Vec3f, n::Vec3f, uv::Point2f,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32,
    regularize::Bool = false
)
    # Check for grazing angle
    wo_dot_n = dot(wo, n)
    if abs(wo_dot_n) < 1f-6
        return SpectralBSDFSample()
    end

    # Get reflectance (rng and sample_u unused for perfect specular)
    kr_rgb = eval_tex(textures, mat.Kr, uv)
    kr_spectral = uplift_rgb(table, kr_rgb, lambda)

    # Orient normal to face wo for reflection
    n_oriented = wo_dot_n < 0f0 ? -n : n

    # Perfect reflection
    wi = reflect(wo, n_oriented)

    # Delta distribution: f = Kr, pdf = 1 (conceptually infinite, but we handle it specially)
    return SpectralBSDFSample(wi, kr_spectral, 1f0, true, 1f0)
end

"""
    sample_bsdf_spectral(table, mat::GlassMaterial, textures, wo, n, uv, lambda, sample_u, rng) -> SpectralBSDFSample

Sample glass BSDF with reflection or refraction.
Uses Fresnel to choose between reflection and transmission.
"""
@propagate_inbounds function sample_bsdf_spectral(
    table::RGBToSpectrumTable, mat::GlassMaterial, textures,
    wo::Vec3f, n::Vec3f, uv::Point2f,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32,
    regularize::Bool = false
)
    # Get material properties
    kr_rgb = eval_tex(textures, mat.Kr, uv)
    kt_rgb = eval_tex(textures, mat.Kt, uv)
    ior = eval_tex(textures, mat.index, uv)

    # Handle edge case where IOR is 0 (matches pbrt-v4 DielectricMaterial)
    ior == 0f0 && (ior = 1f0)

    kr_spectral = uplift_rgb(table, kr_rgb, lambda)
    kt_spectral = uplift_rgb(table, kt_rgb, lambda)

    # Determine if entering or exiting
    cos_theta_o = dot(wo, n)
    entering = cos_theta_o > 0f0

    n_oriented = entering ? n : -n
    cos_theta_o = abs(cos_theta_o)

    # pbrt-v4 convention: eta = n_t / n_i
    # When entering glass (air->glass): eta = ior (e.g., 1.5)
    # When exiting glass (glass->air): eta = 1/ior (e.g., 0.67)
    eta = entering ? ior : (1f0 / ior)

    # Compute Fresnel reflectance using pbrt-v4 convention
    F = fresnel_dielectric(cos_theta_o, eta)

    # Choose reflection or refraction based on Fresnel
    if rng < F
        # Reflection
        wi = reflect(wo, n_oriented)
        # f = F * Kr, probability = F, result = Kr
        return SpectralBSDFSample(wi, kr_spectral, 1f0, true, 1f0)
    else
        # Refraction using pbrt-v4 formula
        sin2_theta_i = max(0f0, 1f0 - cos_theta_o * cos_theta_o)
        sin2_theta_t = sin2_theta_i / (eta * eta)

        if sin2_theta_t >= 1f0
            # Total internal reflection
            wi = reflect(wo, n_oriented)
            return SpectralBSDFSample(wi, kr_spectral, 1f0, true, 1f0)
        end

        cos_theta_t = sqrt(1f0 - sin2_theta_t)
        # pbrt-v4 refracted direction formula: -wi/eta + (cos_i/eta - cos_t) * n
        wi = normalize(-wo / eta + (cos_theta_o / eta - cos_theta_t) * n_oriented)

        # f = (1-F) * Kt, probability = (1-F), result = Kt
        # Include 1/eta² correction for radiance transport (matches pbrt-v4)
        eta_scale = 1f0 / (eta * eta)
        return SpectralBSDFSample(wi, kt_spectral, 1f0, true, eta_scale)
    end
end

# PlasticMaterial spectral functions removed - PlasticMaterial is now an alias for CoatedDiffuseMaterial
# which uses the full LayeredBxDF implementation (see sample_bsdf_spectral for CoatedDiffuseMaterial below)

"""
    sample_bsdf_spectral(table, mat::MetalMaterial, textures, wo, n, uv, lambda, sample_u, rng, regularize=false) -> SpectralBSDFSample

Sample metal BSDF with conductor Fresnel.
Matches pbrt-v4's ConductorBxDF::Sample_f exactly.

The implementation works in local shading coordinates where n = (0,0,1), then transforms back.

When `regularize=true`, the microfacet alpha is increased to reduce fireflies
from near-specular paths (matches pbrt-v4 BSDF::Regularize).
"""
@propagate_inbounds function sample_bsdf_spectral(
    table::RGBToSpectrumTable, mat::MetalMaterial, textures,
    wo_world::Vec3f, n::Vec3f, uv::Point2f,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32,
    regularize::Bool = false
)
    # Build local coordinate frame (matches pbrt-v4's BSDF shading frame)
    tangent, bitangent = coordinate_system(n)

    # Transform wo to local coordinates (matches pbrt-v4's RenderToLocal)
    wo = world_to_local(wo_world, n, tangent, bitangent)

    # Check for grazing angle (matches pbrt-v4's wo.z == 0 check)
    if wo[3] == 0f0
        return SpectralBSDFSample()
    end

    # Get material properties
    eta_rgb = eval_tex(textures, mat.eta, uv)
    k_rgb = eval_tex(textures, mat.k, uv)
    roughness = eval_tex(textures, mat.roughness, uv)

    # Compute alpha values (matches pbrt-v4's roughness remapping)
    alpha_x = mat.remap_roughness ? roughness_to_α(roughness) : roughness
    alpha_y = alpha_x  # Isotropic for now

    # Apply regularization if requested (pbrt-v4: doubles alpha if < 0.3, clamps to [0.1, 0.3])
    if regularize
        alpha_x = regularize_alpha(alpha_x)
        alpha_y = regularize_alpha(alpha_y)
    end

    # Clamp alpha to minimum value if not smooth (matches pbrt-v4 TrowbridgeReitzDistribution constructor)
    if !trowbridge_reitz_effectively_smooth(alpha_x, alpha_y)
        alpha_x = max(alpha_x, 1f-4)
        alpha_y = max(alpha_y, 1f-4)
    end

    # Uplift eta and k to spectral
    eta_spectral = uplift_rgb_unbounded(table, eta_rgb, lambda)
    k_spectral = uplift_rgb_unbounded(table, k_rgb, lambda)

    if trowbridge_reitz_effectively_smooth(alpha_x, alpha_y)
        # Sample perfect specular conductor BRDF (matches pbrt-v4 line 301-305)
        # wi = (-wo.x, -wo.y, wo.z) in local coordinates
        wi = Vec3f(-wo[1], -wo[2], wo[3])

        # f = FrComplex(AbsCosTheta(wi), eta, k) / AbsCosTheta(wi)
        cos_theta_i = abs_cos_theta(wi)
        F = fr_complex_spectral(cos_theta_i, eta_spectral, k_spectral)
        f = F / cos_theta_i

        # Transform wi back to world coordinates
        wi_world = local_to_world(wi, n, tangent, bitangent)

        return SpectralBSDFSample(wi_world, f, 1f0, true, 1f0)
    else
        # Sample rough conductor BRDF (matches pbrt-v4 line 307-327)

        # Sample microfacet normal wm (matches pbrt-v4 line 311)
        wm = trowbridge_reitz_sample_wm(wo, sample_u, alpha_x, alpha_y)

        # Compute reflected direction (matches pbrt-v4 line 312)
        # Reflect(wo, wm) = -wo + 2 * dot(wo, wm) * wm
        wi = -wo + 2f0 * dot(wo, wm) * wm

        # Reject if not in same hemisphere (matches pbrt-v4 line 313-314)
        if !same_hemisphere(wo, wi)
            return SpectralBSDFSample()
        end

        # Compute PDF of wi for microfacet reflection (matches pbrt-v4 line 317)
        # pdf = mfDistrib.PDF(wo, wm) / (4 * AbsDot(wo, wm))
        pdf = trowbridge_reitz_pdf(wo, wm, alpha_x, alpha_y) / (4f0 * abs(dot(wo, wm)))

        # Get cos values (matches pbrt-v4 line 319-321)
        cos_theta_o = abs_cos_theta(wo)
        cos_theta_i = abs_cos_theta(wi)
        if cos_theta_i == 0f0 || cos_theta_o == 0f0
            return SpectralBSDFSample()
        end

        # Evaluate Fresnel factor F for conductor BRDF (matches pbrt-v4 line 323)
        # FrComplex uses AbsDot(wo, wm), not AbsCosTheta
        F = fr_complex_spectral(abs(dot(wo, wm)), eta_spectral, k_spectral)

        # Compute BSDF value (matches pbrt-v4 line 325-326)
        # f = D(wm) * F * G(wo, wi) / (4 * cosTheta_i * cosTheta_o)
        D = trowbridge_reitz_d(wm, alpha_x, alpha_y)
        G = trowbridge_reitz_g(wo, wi, alpha_x, alpha_y)
        f = D * F * G / (4f0 * cos_theta_i * cos_theta_o)

        # Transform wi back to world coordinates
        wi_world = local_to_world(wi, n, tangent, bitangent)

        return SpectralBSDFSample(wi_world, f, pdf, false, 1f0)
    end
end

"""
    sample_bsdf_spectral(table, mat::EmissiveMaterial, textures, wo, n, uv, lambda, sample_u, rng) -> SpectralBSDFSample

Emissive materials don't scatter - return invalid sample.
"""
@propagate_inbounds function sample_bsdf_spectral(
    table::RGBToSpectrumTable, mat::EmissiveMaterial, textures,
    wo::Vec3f, n::Vec3f, uv::Point2f,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32,
    regularize::Bool = false
)
    return SpectralBSDFSample()
end

# Fallback for unknown materials
@propagate_inbounds function sample_bsdf_spectral(
    table::RGBToSpectrumTable, mat::Material, textures,
    wo::Vec3f, n::Vec3f, uv::Point2f,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32,
    regularize::Bool = false
)
    # Check for grazing angle
    wo_dot_n = dot(wo, n)
    if abs(wo_dot_n) < 1f-6
        return SpectralBSDFSample()
    end

    # Default to Lambertian with gray albedo
    kd_spectral = SpectralRadiance(0.5f0)

    # Build coordinate system from shading normal
    tangent, bitangent = coordinate_system(n)

    # Cosine-weighted hemisphere sampling
    local_wi = cosine_sample_hemisphere(sample_u)
    cos_theta = local_wi[3]

    if cos_theta < 1f-6
        return SpectralBSDFSample()
    end

    # Flip wi to same hemisphere as wo (like pbrt-v4)
    if wo_dot_n < 0f0
        local_wi = Vec3f(local_wi[1], local_wi[2], -local_wi[3])
    end

    wi = normalize(local_to_world(local_wi, n, tangent, bitangent))

    f = kd_spectral * (1f0 / Float32(π))
    pdf = cos_theta / Float32(π)

    return SpectralBSDFSample(wi, f, pdf, false, 1f0)
end

# ============================================================================
# BSDF Evaluation (for MIS)
# ============================================================================

"""
    evaluate_bsdf_spectral(table, mat, textures, wo, wi, n, uv, lambda) -> (f::SpectralRadiance, pdf::Float32)

Evaluate BSDF for a given pair of directions. Used for MIS in direct lighting.
Returns the BSDF value and PDF.
"""
@propagate_inbounds function evaluate_bsdf_spectral(
    table::RGBToSpectrumTable, mat::MatteMaterial, textures,
    wo::Vec3f, wi::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
)
    # Check if wi is in the correct hemisphere
    cos_theta_i = dot(wi, n)
    cos_theta_o = dot(wo, n)
    if cos_theta_i * cos_theta_o < 0f0
        return (SpectralRadiance(), 0f0)
    end

    cos_theta = abs(cos_theta_i)
    if cos_theta < 1f-6
        return (SpectralRadiance(), 0f0)
    end

    kd_rgb = eval_tex(textures, mat.Kd, uv)
    # Clamp reflectance to [0,1] as per pbrt-v4
    kd_rgb = clamp(kd_rgb)
    kd_spectral = uplift_rgb(table, kd_rgb, lambda)
    f = kd_spectral * (1f0 / Float32(π))
    pdf = cos_theta / Float32(π)

    return (f, pdf)
end

@propagate_inbounds function evaluate_bsdf_spectral(
    table::RGBToSpectrumTable, mat::MirrorMaterial, textures,
    wo::Vec3f, wi::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
)
    # Perfect specular has zero PDF for non-delta directions
    return (SpectralRadiance(), 0f0)
end

@propagate_inbounds function evaluate_bsdf_spectral(
    table::RGBToSpectrumTable, mat::GlassMaterial, textures,
    wo::Vec3f, wi::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
)
    # Specular has zero PDF for non-delta directions
    return (SpectralRadiance(), 0f0)
end

"""
    evaluate_bsdf_spectral(table, mat::MetalMaterial, ...) -> (f, pdf)

Evaluate metal BSDF for given directions.
Matches pbrt-v4's ConductorBxDF::f and ConductorBxDF::PDF exactly.
"""
@propagate_inbounds function evaluate_bsdf_spectral(
    table::RGBToSpectrumTable, mat::MetalMaterial, textures,
    wo_world::Vec3f, wi_world::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
)
    # Build local coordinate frame
    tangent, bitangent = coordinate_system(n)

    # Transform to local coordinates
    wo = world_to_local(wo_world, n, tangent, bitangent)
    wi = world_to_local(wi_world, n, tangent, bitangent)

    # Must be in same hemisphere (matches pbrt-v4 line 332-333)
    if !same_hemisphere(wo, wi)
        return (SpectralRadiance(), 0f0)
    end

    # Get material properties
    eta_rgb = eval_tex(textures, mat.eta, uv)
    k_rgb = eval_tex(textures, mat.k, uv)
    roughness = eval_tex(textures, mat.roughness, uv)

    # Compute alpha values
    alpha_x = mat.remap_roughness ? roughness_to_α(roughness) : roughness
    alpha_y = alpha_x

    # Clamp alpha if not smooth
    if !trowbridge_reitz_effectively_smooth(alpha_x, alpha_y)
        alpha_x = max(alpha_x, 1f-4)
        alpha_y = max(alpha_y, 1f-4)
    end

    # Specular returns zero for evaluation (matches pbrt-v4 line 334-335)
    if trowbridge_reitz_effectively_smooth(alpha_x, alpha_y)
        return (SpectralRadiance(), 0f0)
    end

    # Evaluate rough conductor BRDF (matches pbrt-v4 line 336-350)
    cos_theta_o = abs_cos_theta(wo)
    cos_theta_i = abs_cos_theta(wi)
    if cos_theta_i == 0f0 || cos_theta_o == 0f0
        return (SpectralRadiance(), 0f0)
    end

    # Compute half-vector wm (matches pbrt-v4 line 341-344)
    wm = wi + wo
    if dot(wm, wm) == 0f0
        return (SpectralRadiance(), 0f0)
    end
    wm = normalize(wm)

    # Uplift eta and k to spectral
    eta_spectral = uplift_rgb_unbounded(table, eta_rgb, lambda)
    k_spectral = uplift_rgb_unbounded(table, k_rgb, lambda)

    # Evaluate Fresnel factor F (matches pbrt-v4 line 347)
    F = fr_complex_spectral(abs(dot(wo, wm)), eta_spectral, k_spectral)

    # Compute BSDF value (matches pbrt-v4 line 349)
    # f = D(wm) * F * G(wo, wi) / (4 * cosTheta_i * cosTheta_o)
    D = trowbridge_reitz_d(wm, alpha_x, alpha_y)
    G = trowbridge_reitz_g(wo, wi, alpha_x, alpha_y)
    f = D * F * G / (4f0 * cos_theta_i * cos_theta_o)

    # Compute PDF (matches pbrt-v4 line 361-367)
    # wm needs to face forward for PDF
    wm_pdf = face_forward(wm, Vec3f(0f0, 0f0, 1f0))
    pdf = trowbridge_reitz_pdf(wo, wm_pdf, alpha_x, alpha_y) / (4f0 * abs(dot(wo, wm_pdf)))

    return (f, pdf)
end

@propagate_inbounds function evaluate_bsdf_spectral(
    table::RGBToSpectrumTable, mat::EmissiveMaterial, textures,
    wo::Vec3f, wi::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
)
    # Emissive materials don't scatter light (black body)
    return (SpectralRadiance(), 0f0)
end

# Fallback
@propagate_inbounds function evaluate_bsdf_spectral(
    table::RGBToSpectrumTable, mat::Material, textures,
    wo::Vec3f, wi::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
)
    cos_theta_i = dot(wi, n)
    cos_theta_o = dot(wo, n)
    if cos_theta_i * cos_theta_o < 0f0
        return (SpectralRadiance(), 0f0)
    end

    cos_theta = abs(cos_theta_i)
    if cos_theta < 1f-6
        return (SpectralRadiance(), 0f0)
    end

    # Default gray Lambertian
    f = SpectralRadiance(0.5f0 / Float32(π))
    pdf = cos_theta / Float32(π)

    return (f, pdf)
end

# ============================================================================
# Spectral Emission Evaluation
# ============================================================================

"""
    get_emission_spectral(table::RGBToSpectrumTable, mat::EmissiveMaterial, textures, wo, n, uv, lambda) -> SpectralRadiance

Get spectral emission from an emissive material.
"""
@propagate_inbounds function get_emission_spectral(
    table::RGBToSpectrumTable, mat::EmissiveMaterial, textures,
    wo::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
)
    cos_theta = dot(wo, n)
    if !mat.two_sided && cos_theta < 0f0
        return SpectralRadiance()
    end

    Le_rgb = eval_tex(textures, mat.Le, uv) * mat.scale
    return uplift_rgb(table, Le_rgb, lambda)
end

# Non-emissive materials return zero
@propagate_inbounds function get_emission_spectral(
    table::RGBToSpectrumTable, mat::Material, textures,
    wo::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
)
    return SpectralRadiance()
end

# ============================================================================
# Spectral Albedo Extraction (for denoising aux buffers)
# ============================================================================

"""
    get_albedo_spectral(table::RGBToSpectrumTable, mat, textures, uv, lambda) -> SpectralRadiance

Extract material albedo as spectral value for denoising auxiliary buffers.
"""
@propagate_inbounds function get_albedo_spectral(table::RGBToSpectrumTable, mat::MatteMaterial, textures, uv::Point2f, lambda::Wavelengths)
    return uplift_rgb(table, eval_tex(textures, mat.Kd, uv), lambda)
end

@propagate_inbounds function get_albedo_spectral(table::RGBToSpectrumTable, mat::MirrorMaterial, textures, uv::Point2f, lambda::Wavelengths)
    return uplift_rgb(table, eval_tex(textures, mat.Kr, uv), lambda)
end

@propagate_inbounds function get_albedo_spectral(table::RGBToSpectrumTable, mat::GlassMaterial, textures, uv::Point2f, lambda::Wavelengths)
    # For glass, use average of reflection and transmission
    kr = eval_tex(textures, mat.Kr, uv)
    kt = eval_tex(textures, mat.Kt, uv)
    avg = RGBSpectrum((kr.c[1] + kt.c[1]) * 0.5f0,
                      (kr.c[2] + kt.c[2]) * 0.5f0,
                      (kr.c[3] + kt.c[3]) * 0.5f0)
    return uplift_rgb(table, avg, lambda)
end

@propagate_inbounds function get_albedo_spectral(table::RGBToSpectrumTable, mat::MetalMaterial, textures, uv::Point2f, lambda::Wavelengths)
    return uplift_rgb(table, eval_tex(textures, mat.reflectance, uv), lambda)
end

@propagate_inbounds function get_albedo_spectral(table::RGBToSpectrumTable, mat::EmissiveMaterial, textures, uv::Point2f, lambda::Wavelengths)
    # Normalized emission color
    Le = eval_tex(textures, mat.Le, uv)
    lum = to_Y(Le)
    if lum > 0f0
        normalized = Le / lum
        return uplift_rgb(table, normalized, lambda)
    end
    return SpectralRadiance()
end

@propagate_inbounds function get_albedo_spectral(table::RGBToSpectrumTable, mat::Material, textures, uv::Point2f, lambda::Wavelengths)
    return SpectralRadiance(0.5f0)
end

# ============================================================================
# pbrt-v4 Compatible RNG and Hash Functions for LayeredBxDF
# ============================================================================
# These functions exactly match pbrt-v4's implementation for deterministic
# random number generation within LayeredBxDF. This is critical for
# reproducibility and correct firefly-free rendering.
#
# Reference: pbrt-v4 src/pbrt/util/hash.h, src/pbrt/util/rng.h

"""
    murmur_hash_64a(data::NTuple{N,UInt8}, seed::UInt64) -> UInt64

MurmurHash2 64-bit hash function, exactly matching pbrt-v4's MurmurHash64A.
Reference: https://github.com/explosion/murmurhash/blob/master/murmurhash/MurmurHash2.cpp
"""
@inline function murmur_hash_64a(data::NTuple{N,UInt8}, seed::UInt64=UInt64(0))::UInt64 where N
    m = 0xc6a4a7935bd1e995
    r = 47

    h = seed ⊻ (UInt64(N) * m)

    # Process 8-byte chunks
    n_chunks = N ÷ 8
    @inbounds for i in 0:(n_chunks-1)
        # Read 8 bytes as UInt64 (little-endian)
        k = UInt64(data[8*i + 1]) |
            (UInt64(data[8*i + 2]) << 8) |
            (UInt64(data[8*i + 3]) << 16) |
            (UInt64(data[8*i + 4]) << 24) |
            (UInt64(data[8*i + 5]) << 32) |
            (UInt64(data[8*i + 6]) << 40) |
            (UInt64(data[8*i + 7]) << 48) |
            (UInt64(data[8*i + 8]) << 56)

        k *= m
        k ⊻= k >> r
        k *= m

        h ⊻= k
        h *= m
    end

    # Handle remaining bytes (switch fallthrough in C++)
    remaining = N & 7
    offset = 8 * n_chunks
    @inbounds if remaining >= 7
        h ⊻= UInt64(data[offset + 7]) << 48
    end
    @inbounds if remaining >= 6
        h ⊻= UInt64(data[offset + 6]) << 40
    end
    @inbounds if remaining >= 5
        h ⊻= UInt64(data[offset + 5]) << 32
    end
    @inbounds if remaining >= 4
        h ⊻= UInt64(data[offset + 4]) << 24
    end
    @inbounds if remaining >= 3
        h ⊻= UInt64(data[offset + 3]) << 16
    end
    @inbounds if remaining >= 2
        h ⊻= UInt64(data[offset + 2]) << 8
    end
    @inbounds if remaining >= 1
        h ⊻= UInt64(data[offset + 1])
        h *= m
    end

    h ⊻= h >> r
    h *= m
    h ⊻= h >> r

    return h
end

"""
    mix_bits(v::UInt64) -> UInt64

Bit mixing function from pbrt-v4's hash.h.
Reference: http://zimbry.blogspot.ch/2011/09/better-bit-mixing-improving-on.html
"""
@inline function mix_bits(v::UInt64)::UInt64
    v ⊻= v >> 31
    v *= 0x7fb5d329728ea185
    v ⊻= v >> 27
    v *= 0x81dadef4bc2dd44d
    v ⊻= v >> 33
    return v
end

"""
    float32_to_bytes(v::Float32) -> NTuple{4,UInt8}

GPU-compatible conversion of Float32 to bytes using Core.bitcast.
"""
@inline function float32_to_bytes(v::Float32)::NTuple{4,UInt8}
    bits = Core.bitcast(UInt32, v)
    return (
        UInt8(bits & 0xff),
        UInt8((bits >> 8) & 0xff),
        UInt8((bits >> 16) & 0xff),
        UInt8((bits >> 24) & 0xff)
    )
end

"""
    uint64_to_bytes(v::UInt64) -> NTuple{8,UInt8}

GPU-compatible conversion of UInt64 to bytes.
"""
@inline function uint64_to_bytes(v::UInt64)::NTuple{8,UInt8}
    return (
        UInt8(v & 0xff),
        UInt8((v >> 8) & 0xff),
        UInt8((v >> 16) & 0xff),
        UInt8((v >> 24) & 0xff),
        UInt8((v >> 32) & 0xff),
        UInt8((v >> 40) & 0xff),
        UInt8((v >> 48) & 0xff),
        UInt8((v >> 56) & 0xff)
    )
end

"""
    pbrt_hash(args...) -> UInt64

Hash function matching pbrt-v4's variadic Hash() template.
Packs arguments into a byte buffer and applies MurmurHash64A.
Uses GPU-compatible bit manipulation instead of reinterpret.
"""
@inline function pbrt_hash(v::Float32)::UInt64
    bytes = float32_to_bytes(v)
    murmur_hash_64a(bytes, UInt64(0))
end

@inline function pbrt_hash(v::UInt64)::UInt64
    bytes = uint64_to_bytes(v)
    murmur_hash_64a(bytes, UInt64(0))
end

@inline function pbrt_hash(v::Vec3f)::UInt64
    # Vec3f is 12 bytes (3 x Float32)
    x_bytes = float32_to_bytes(v[1])
    y_bytes = float32_to_bytes(v[2])
    z_bytes = float32_to_bytes(v[3])
    bytes = (x_bytes..., y_bytes..., z_bytes...)
    murmur_hash_64a(bytes, UInt64(0))
end

@inline function pbrt_hash(p::Point3f)::UInt64
    # Point3f is 12 bytes (3 x Float32)
    x_bytes = float32_to_bytes(p[1])
    y_bytes = float32_to_bytes(p[2])
    z_bytes = float32_to_bytes(p[3])
    bytes = (x_bytes..., y_bytes..., z_bytes...)
    murmur_hash_64a(bytes, UInt64(0))
end

@inline function pbrt_hash(seed::UInt64, v::Vec3f)::UInt64
    # Hash(seed, wo) - combine seed with vector
    seed_bytes = uint64_to_bytes(seed)
    x_bytes = float32_to_bytes(v[1])
    y_bytes = float32_to_bytes(v[2])
    z_bytes = float32_to_bytes(v[3])
    bytes = (seed_bytes..., x_bytes..., y_bytes..., z_bytes...)
    murmur_hash_64a(bytes, UInt64(0))
end

@inline function pbrt_hash(a::UInt64, b::Float32)::UInt64
    a_bytes = uint64_to_bytes(a)
    b_bytes = float32_to_bytes(b)
    bytes = (a_bytes..., b_bytes...)
    murmur_hash_64a(bytes, UInt64(0))
end

@inline function pbrt_hash(a::Float32, b::Point2f)::UInt64
    a_bytes = float32_to_bytes(a)
    bx_bytes = float32_to_bytes(b[1])
    by_bytes = float32_to_bytes(b[2])
    bytes = (a_bytes..., bx_bytes..., by_bytes...)
    murmur_hash_64a(bytes, UInt64(0))
end

# PCG32 Random Number Generator - GPU-compatible functional implementation
# Exact port of pbrt-v4's RNG class but using immutable state
const PCG32_DEFAULT_STATE = 0x853c49e6748fea9b
const PCG32_DEFAULT_STREAM = 0xda3e39cb94b95bdb
const PCG32_MULT = 0x5851f42d4c957f2d

"""
    PCG32State

GPU-compatible immutable PCG32 random number generator state.
Uses tuple-like struct for stack allocation on GPU.
"""
struct PCG32State
    state::UInt64
    inc::UInt64
end

"""
    pcg32_init(seq_index::UInt64, seed::UInt64) -> PCG32State

Initialize PCG32 with sequence index and seed, matching pbrt-v4's SetSequence.
Returns initialized state.
"""
@inline function pcg32_init(seq_index::UInt64, seed::UInt64)::PCG32State
    inc = (seq_index << 1) | UInt64(1)
    state = UInt64(0)

    # First advance
    state = state * PCG32_MULT + inc

    # Add seed
    state += seed

    # Second advance
    state = state * PCG32_MULT + inc

    return PCG32State(state, inc)
end

@inline function pcg32_init(seq_index::UInt64)::PCG32State
    pcg32_init(seq_index, mix_bits(seq_index))
end

"""
    pcg32_uniform_u32(rng::PCG32State) -> (UInt32, PCG32State)

Generate uniform random UInt32 and return new state.
Matching pbrt-v4's Uniform<uint32_t>().
"""
@inline function pcg32_uniform_u32(rng::PCG32State)::Tuple{UInt32, PCG32State}
    oldstate = rng.state
    newstate = oldstate * PCG32_MULT + rng.inc
    # Keep intermediate values as UInt64, then mask to UInt32
    xorshifted = ((oldstate >> 18) ⊻ oldstate) >> 27
    rot = oldstate >> 59
    xorshifted32 = UInt32(xorshifted & 0xFFFFFFFF)
    rot32 = UInt32(rot & 0x1F)
    result = (xorshifted32 >> rot32) | (xorshifted32 << ((32 - rot32) & 31))
    return (result, PCG32State(newstate, rng.inc))
end

"""
    pcg32_uniform_f32(rng::PCG32State) -> (Float32, PCG32State)

Generate uniform random Float32 in [0, 1) and return new state.
Matching pbrt-v4's Uniform<float>().
"""
@inline function pcg32_uniform_f32(rng::PCG32State)::Tuple{Float32, PCG32State}
    u32, new_rng = pcg32_uniform_u32(rng)
    # 0x1p-32f = 2^-32 as Float32 ≈ 2.3283064e-10
    f = min(Float32(1) - eps(Float32), Float32(u32) * 2.3283064f-10)
    return (f, new_rng)
end

"""
    sample_exponential(u::Float32, a::Float32) -> Float32

Sample from exponential distribution with rate parameter a.
Returns -log(1-u)/a, matching pbrt-v4's SampleExponential.
"""
@inline function sample_exponential(u::Float32, a::Float32)::Float32
    return -log(1f0 - u) / a
end

# ============================================================================
# CoatedDiffuseMaterial - LayeredBxDF Implementation (pbrt-v4 port)
# ============================================================================

"""
    Tr(thickness, w) -> Float32

Transmittance through a layer of given thickness along direction w.
Used in LayeredBxDF random walk.
"""
@propagate_inbounds function layer_transmittance(thickness::Float32, w::Vec3f)::Float32
    abs(thickness) <= eps(Float32) && return 1f0
    exp(-abs(thickness / w[3]))
end

"""
    sample_hg_phase_spectral(g, wo, u) -> (wi, phase_pdf)

Sample direction from Henyey-Greenstein phase function.
"""
@propagate_inbounds function sample_hg_phase_spectral(g::Float32, wo::Vec3f, u::Point2f)
    # Sample cos_θ from HG distribution
    cos_θ = if abs(g) < 1f-3
        # Isotropic case
        1f0 - 2f0 * u[1]
    else
        g2 = g * g
        sqr_term = (1f0 - g2) / (1f0 - g + 2f0 * g * u[1])
        clamp((1f0 + g2 - sqr_term * sqr_term) / (2f0 * g), -1f0, 1f0)
    end

    sin_θ = sqrt(max(0f0, 1f0 - cos_θ * cos_θ))
    ϕ = 2f0 * Float32(π) * u[2]

    # Build local frame around -wo
    t1, t2 = coordinate_system(-wo)
    wi = sin_θ * cos(ϕ) * t1 + sin_θ * sin(ϕ) * t2 + cos_θ * (-wo)
    wi = normalize(wi)

    # HG phase function value (equals PDF)
    g2 = g * g
    denom = 1f0 + g2 - 2f0 * g * cos_θ
    p = (1f0 - g2) / (4f0 * Float32(π) * denom * sqrt(max(1f-10, denom)))

    return (wi, p)
end

"""
    hg_phase_pdf(g, cos_θ) -> Float32

Evaluate Henyey-Greenstein phase function PDF.
"""
@propagate_inbounds function hg_phase_pdf(g::Float32, cos_θ::Float32)::Float32
    g2 = g * g
    denom = 1f0 + g2 - 2f0 * g * cos_θ
    (1f0 - g2) / (4f0 * Float32(π) * denom * sqrt(max(1f-10, denom)))
end

"""
    sample_dielectric_transmission_spectral(eta, wo, uc) -> (wi, T, valid)

Sample transmission through a dielectric interface.
Returns transmitted direction, transmittance, and validity flag.

Uses pbrt-v4 convention: eta = n_t / n_i (transmitted IOR / incident IOR).
The wo direction is in local shading space where z is the surface normal.
"""
@propagate_inbounds function sample_dielectric_transmission_spectral(
    eta::Float32, wo::Vec3f, uc::Float32
)
    cos_θo = abs(wo[3])
    F = fresnel_dielectric(cos_θo, eta)

    # Use uc to decide reflection vs transmission
    if uc < F
        # Reflection - not transmission
        return (Vec3f(0), 0f0, false)
    end

    # Compute transmitted direction using pbrt-v4 Snell's law formula
    # For pbrt-v4 convention: sin²θt = sin²θi / eta²
    sin2_θi = max(0f0, 1f0 - cos_θo^2)
    sin2_θt = sin2_θi / (eta^2)

    # Check for total internal reflection
    if sin2_θt >= 1f0
        return (Vec3f(0), 0f0, false)
    end

    cos_θt = sqrt(1f0 - sin2_θt)

    # pbrt-v4 refracted direction: -wo/eta + (cos_i/eta - cos_t) * n
    # In local coords, n = (0, 0, sign(wo.z))
    entering = wo[3] > 0f0
    n_sign = entering ? 1f0 : -1f0

    wi = Vec3f(
        -wo[1] / eta,
        -wo[2] / eta,
        (cos_θo / eta - cos_θt) * n_sign
    )
    wi = normalize(wi)

    T = 1f0 - F
    return (wi, T, true)
end

# ============================================================================
# LayeredBxDF Helper Functions - Dielectric Interface Sampling
# ============================================================================
# These functions implement the top (dielectric) and bottom (diffuse) interfaces
# for the LayeredBxDF random walk algorithm, matching pbrt-v4 exactly.

"""
    BxDFReflTransFlags - Flags for controlling reflection/transmission sampling
"""
const BXDF_REFLECTION = UInt8(1)
const BXDF_TRANSMISSION = UInt8(2)
const BXDF_ALL = UInt8(3)

"""
    LayeredBSDFSample - Internal sample result for LayeredBxDF interfaces

Contains the sampled direction, BSDF value, pdf, and flags indicating
whether the sample is reflection/transmission and specular/glossy.
"""
struct LayeredBSDFSample
    f::SpectralRadiance      # BSDF value (spectral)
    wi::Vec3f                # Sampled direction
    pdf::Float32             # Probability density
    is_reflection::Bool      # True if reflection, false if transmission
    is_specular::Bool        # True if delta distribution
    eta::Float32             # Relative IOR (for transmission)
    valid::Bool              # Whether sample is valid
end

LayeredBSDFSample() = LayeredBSDFSample(SpectralRadiance(), Vec3f(0), 0f0, false, false, 1f0, false)

"""
    sample_dielectric_interface(wo, uc, u, alpha_x, alpha_y, eta, refl_trans_flags) -> LayeredBSDFSample

Sample the dielectric coating interface (top layer in CoatedDiffuse).
Handles both smooth (specular) and rough (microfacet) dielectric surfaces.

This matches pbrt-v4's DielectricBxDF::Sample_f exactly.
"""
@propagate_inbounds function sample_dielectric_interface(
    wo::Vec3f, uc::Float32, u::Point2f,
    alpha_x::Float32, alpha_y::Float32, eta::Float32,
    refl_trans_flags::UInt8
)
    is_smooth = trowbridge_reitz_effectively_smooth(alpha_x, alpha_y)

    if is_smooth || eta == 1f0
        # Sample perfect specular dielectric BSDF
        cos_θo = wo[3]
        R = fresnel_dielectric(cos_θo, eta)
        T = 1f0 - R

        # Compute probabilities for sampling reflection vs transmission
        pr = (refl_trans_flags & BXDF_REFLECTION) != 0 ? R : 0f0
        pt = (refl_trans_flags & BXDF_TRANSMISSION) != 0 ? T : 0f0

        if pr == 0f0 && pt == 0f0
            return LayeredBSDFSample()
        end

        if uc < pr / (pr + pt)
            # Sample perfect specular reflection
            wi = Vec3f(-wo[1], -wo[2], wo[3])
            f_val = R / abs(wi[3])
            pdf = pr / (pr + pt)
            return LayeredBSDFSample(SpectralRadiance(f_val), wi, pdf, true, true, 1f0, true)
        else
            # Sample perfect specular transmission
            # Compute refracted direction
            valid, wi, etap = refract_pbrt(wo, eta)
            if !valid
                return LayeredBSDFSample()
            end

            f_val = T / abs(wi[3])
            pdf = pt / (pr + pt)
            return LayeredBSDFSample(SpectralRadiance(f_val), wi, pdf, false, true, etap, true)
        end
    else
        # Sample rough dielectric BSDF using microfacet distribution
        wm = trowbridge_reitz_sample_wm(wo, u, alpha_x, alpha_y)
        cos_θo_m = dot(wo, wm)

        R = fresnel_dielectric(cos_θo_m, eta)
        T = 1f0 - R

        pr = (refl_trans_flags & BXDF_REFLECTION) != 0 ? R : 0f0
        pt = (refl_trans_flags & BXDF_TRANSMISSION) != 0 ? T : 0f0

        if pr == 0f0 && pt == 0f0
            return LayeredBSDFSample()
        end

        if uc < pr / (pr + pt)
            # Sample reflection at rough dielectric interface
            wi = reflect(wo, wm)
            if !same_hemisphere(wo, wi)
                return LayeredBSDFSample()
            end

            # Compute PDF of rough dielectric reflection
            pdf_m = trowbridge_reitz_pdf(wo, wm, alpha_x, alpha_y)
            pdf = pdf_m / (4f0 * abs(cos_θo_m)) * pr / (pr + pt)

            D = trowbridge_reitz_d(wm, alpha_x, alpha_y)
            G = trowbridge_reitz_g(wo, wi, alpha_x, alpha_y)
            f_val = D * G * R / (4f0 * wo[3] * wi[3])

            return LayeredBSDFSample(SpectralRadiance(f_val), wi, pdf, true, false, 1f0, true)
        else
            # Sample transmission at rough dielectric interface
            valid, wi, etap = refract_microfacet(wo, wm, eta)
            if !valid || same_hemisphere(wo, wi) || wi[3] == 0f0
                return LayeredBSDFSample()
            end

            # Compute PDF of rough dielectric transmission
            denom = (dot(wi, wm) + dot(wo, wm) / etap)^2
            dwm_dwi = abs(dot(wi, wm)) / denom
            pdf_m = trowbridge_reitz_pdf(wo, wm, alpha_x, alpha_y)
            pdf = pdf_m * dwm_dwi * pt / (pr + pt)

            D = trowbridge_reitz_d(wm, alpha_x, alpha_y)
            G = trowbridge_reitz_g(wo, wi, alpha_x, alpha_y)
            f_val = T * D * G * abs(dot(wi, wm) * dot(wo, wm) / (wi[3] * wo[3] * denom))

            return LayeredBSDFSample(SpectralRadiance(f_val), wi, pdf, false, false, etap, true)
        end
    end
end

"""
    refract_pbrt(wo, eta) -> (valid, wi, etap)

Compute refracted direction using pbrt-v4 convention.
eta = n_transmitted / n_incident
Returns (valid, wi, effective_eta).
"""
@inline function refract_pbrt(wo::Vec3f, eta::Float32)
    cos_θi = wo[3]
    # Flip eta if entering from below (wo.z < 0)
    etap = cos_θi > 0f0 ? eta : (1f0 / eta)

    sin2_θi = max(0f0, 1f0 - cos_θi^2)
    sin2_θt = sin2_θi / (etap^2)

    # Total internal reflection check
    if sin2_θt >= 1f0
        return (false, Vec3f(0), 1f0)
    end

    cos_θt = sqrt(1f0 - sin2_θt)
    # Flip cos_θt sign to match wo hemisphere convention
    cos_θt_signed = cos_θi > 0f0 ? -cos_θt : cos_θt

    wi = Vec3f(-wo[1] / etap, -wo[2] / etap, cos_θt_signed)
    wi = normalize(wi)

    return (true, wi, etap)
end

"""
    refract_microfacet(wo, wm, eta) -> (valid, wi, etap)

Compute refracted direction through a microfacet with normal wm.
"""
@inline function refract_microfacet(wo::Vec3f, wm::Vec3f, eta::Float32)
    cos_θi = dot(wo, wm)
    etap = cos_θi > 0f0 ? eta : (1f0 / eta)

    sin2_θi = max(0f0, 1f0 - cos_θi^2)
    sin2_θt = sin2_θi / (etap^2)

    if sin2_θt >= 1f0
        return (false, Vec3f(0), 1f0)
    end

    cos_θt = sqrt(1f0 - sin2_θt)
    # Sign convention: transmitted ray goes to opposite side of microfacet
    cos_θt_signed = cos_θi > 0f0 ? -cos_θt : cos_θt

    # Refracted direction formula: wi = -wo/etap + (cos_θi/etap + cos_θt) * wm
    wi = -wo / etap + (cos_θi / etap + cos_θt_signed) * wm
    wi = normalize(wi)

    return (true, wi, etap)
end

"""
    reflect(wo, n) -> wi

Compute reflected direction: wi = -wo + 2*dot(wo,n)*n
"""
@inline function reflect(wo::Vec3f, n::Vec3f)::Vec3f
    return -wo + 2f0 * dot(wo, n) * n
end

"""
    same_hemisphere(w1, w2) -> Bool

Check if two directions are in the same hemisphere (both have same sign of z).
"""
@inline same_hemisphere(w1::Vec3f, w2::Vec3f)::Bool = w1[3] * w2[3] > 0f0

"""
    sample_diffuse_interface(wo, u, reflectance) -> LayeredBSDFSample

Sample the diffuse base layer (bottom in CoatedDiffuse).
This is a simple cosine-weighted hemisphere sampler.
"""
@propagate_inbounds function sample_diffuse_interface(
    wo::Vec3f, u::Point2f, reflectance::SpectralRadiance,
    refl_trans_flags::UInt8
)
    # Diffuse only reflects, never transmits
    if (refl_trans_flags & BXDF_REFLECTION) == 0
        return LayeredBSDFSample()
    end

    # Cosine-weighted hemisphere sampling
    wi = cosine_sample_hemisphere(u)

    # Ensure wi is in same hemisphere as wo
    if wo[3] < 0f0
        wi = Vec3f(wi[1], wi[2], -wi[3])
    end

    cos_θi = abs(wi[3])
    if cos_θi < 1f-6
        return LayeredBSDFSample()
    end

    # Lambertian: f = R/π, pdf = cos_θ/π
    f = reflectance * (1f0 / Float32(π))
    pdf = cos_θi / Float32(π)

    return LayeredBSDFSample(f, wi, pdf, true, false, 1f0, true)
end

"""
    eval_diffuse_interface(wo, wi, reflectance) -> (f, pdf)

Evaluate diffuse BSDF for given directions.
"""
@propagate_inbounds function eval_diffuse_interface(
    wo::Vec3f, wi::Vec3f, reflectance::SpectralRadiance
)
    if !same_hemisphere(wo, wi)
        return (SpectralRadiance(), 0f0)
    end
    f = reflectance * (1f0 / Float32(π))
    pdf = abs(wi[3]) / Float32(π)
    return (f, pdf)
end

"""
    pdf_diffuse_interface(wo, wi) -> Float32

Compute PDF of diffuse sampling.
"""
@inline function pdf_diffuse_interface(wo::Vec3f, wi::Vec3f)::Float32
    if !same_hemisphere(wo, wi)
        return 0f0
    end
    return abs(wi[3]) / Float32(π)
end

"""
    power_heuristic(nf, fPdf, ng, gPdf) -> Float32

Balance heuristic for MIS with power=2.
"""
@inline function power_heuristic(nf::Int, fPdf::Float32, ng::Int, gPdf::Float32)::Float32
    f = nf * fPdf
    g = ng * gPdf
    f_sq = f * f
    g_sq = g * g
    if f_sq + g_sq == 0f0
        return 0f0
    end
    return f_sq / (f_sq + g_sq)
end

"""
    sample_bsdf_spectral(table, mat::CoatedDiffuseMaterial, textures, wo, n, uv, lambda, sample_u, rng, regularize=false) -> SpectralBSDFSample

Sample CoatedDiffuse BSDF using pbrt-v4's LayeredBxDF random walk algorithm.

This is a 100% port of pbrt-v4's LayeredBxDF::Sample_f. The algorithm:
1. Sample entrance interface (top dielectric for wo.z > 0)
2. If reflection at entrance: return immediately with pdfIsProportional=true
3. If transmission: start random walk through layers
4. At each depth: possibly scatter in medium, then sample interface
5. When ray exits through transmission: return the accumulated sample
6. Russian roulette for path termination

When `regularize=true`, the coating's microfacet alpha is increased to reduce fireflies.
"""
@propagate_inbounds function sample_bsdf_spectral(
    table::RGBToSpectrumTable, mat::CoatedDiffuseMaterial, textures,
    wo::Vec3f, n::Vec3f, uv::Point2f,
    lambda::Wavelengths, sample_u::Point2f, rng_in::Float32,
    regularize::Bool = false
)
    # Check for grazing angle
    wo_dot_n = dot(wo, n)
    if abs(wo_dot_n) < 1f-6
        return SpectralBSDFSample()
    end

    # Get material properties
    refl_rgb = eval_tex(textures, mat.reflectance, uv)
    eta = mat.eta
    thickness = max(eval_tex(textures, mat.thickness, uv), eps(Float32))
    albedo_rgb = eval_tex(textures, mat.albedo, uv)
    g_val = clamp(eval_tex(textures, mat.g, uv), -0.99f0, 0.99f0)

    # Get roughness parameters
    u_roughness = eval_tex(textures, mat.u_roughness, uv)
    v_roughness = eval_tex(textures, mat.v_roughness, uv)

    # Remap roughness if needed
    alpha_x = mat.remap_roughness ? roughness_to_α(u_roughness) : u_roughness
    alpha_y = mat.remap_roughness ? roughness_to_α(v_roughness) : v_roughness

    # Apply regularization if requested
    if regularize
        alpha_x = regularize_alpha(alpha_x)
        alpha_y = regularize_alpha(alpha_y)
    end

    refl_spectral = uplift_rgb(table, refl_rgb, lambda)
    albedo_spectral = uplift_rgb(table, albedo_rgb, lambda)
    has_medium = !is_black(albedo_rgb)

    max_depth = Int(mat.max_depth)

    # Build coordinate system from shading normal
    tangent, bitangent = coordinate_system(n)

    # Transform wo to local space
    wo_local = Vec3f(dot(wo, tangent), dot(wo, bitangent), wo_dot_n)

    # Two-sided handling: flip if entering from below
    flip_wi = wo_local[3] < 0f0
    if flip_wi
        wo_local = -wo_local
    end

    # Determine entrance interface (top for wo.z > 0, which is always true after flip)
    entered_top = true  # After flip, wo_local.z > 0, so we enter from top

    # === Sample entrance interface ===
    # Use rng_in and sample_u for entrance sampling
    bs = sample_dielectric_interface(wo_local, rng_in, sample_u, alpha_x, alpha_y, eta, BXDF_ALL)

    if !bs.valid || bs.pdf == 0f0 || bs.wi[3] == 0f0
        return SpectralBSDFSample()
    end

    # If reflection at entrance: return immediately (pdfIsProportional case)
    if bs.is_reflection
        wi_local = bs.wi
        if flip_wi
            wi_local = -wi_local
        end
        wi = tangent * wi_local[1] + bitangent * wi_local[2] + n * wi_local[3]
        wi = normalize(wi)

        # pdfIsProportional=true means the pdf is only proportional, not exact
        # For specular, f already includes proper weighting
        return SpectralBSDFSample(wi, bs.f, bs.pdf, bs.is_specular, 1f0)
    end

    # === Begin random walk through layers ===
    w = bs.wi
    specular_path = bs.is_specular
    f = bs.f * abs(w[3])  # f * AbsCosTheta(wi)
    pdf = bs.pdf
    z = entered_top ? thickness : 0f0  # Start at bottom of coating after transmission

    # Initialize deterministic RNG for random walk (GPU-compatible functional style)
    # Use Hash(seed, wo) combined with Hash(uc, u) for reproducibility
    seed = UInt64(0)  # pbrt uses GetOptions().seed, we use 0
    rng = pcg32_init(pbrt_hash(seed, wo_local), pbrt_hash(rng_in, sample_u))

    for depth in 0:(max_depth-1)
        # Possibly terminate with Russian Roulette
        rr_beta = max_component(f) / pdf
        if depth > 3 && rr_beta < 0.25f0
            q = max(0f0, 1f0 - rr_beta)
            rr_val, rng = pcg32_uniform_f32(rng)
            if rr_val < q
                return SpectralBSDFSample()
            end
            pdf *= 1f0 - q
        end

        if w[3] == 0f0
            return SpectralBSDFSample()
        end

        if has_medium
            # Sample potential scattering in medium
            sigma_t = 1f0
            exp_u, rng = pcg32_uniform_f32(rng)
            dz = sample_exponential(exp_u, sigma_t / abs(w[3]))
            zp = w[3] > 0f0 ? (z + dz) : (z - dz)

            if zp == z
                return SpectralBSDFSample()
            end

            if 0f0 < zp && zp < thickness
                # Scattering event within the medium
                phase_u1, rng = pcg32_uniform_f32(rng)
                phase_u2, rng = pcg32_uniform_f32(rng)
                wi_phase, phase_p = sample_hg_phase_spectral(g_val, -w, Point2f(phase_u1, phase_u2))
                if phase_p == 0f0 || wi_phase[3] == 0f0
                    return SpectralBSDFSample()
                end
                f = f * albedo_spectral * phase_p
                pdf *= phase_p
                specular_path = false
                w = wi_phase
                z = zp
                continue
            end

            # Clamp to layer boundary
            z = clamp(zp, 0f0, thickness)
        else
            # No medium: advance directly to other interface
            z = (z == thickness) ? 0f0 : thickness
            f = f * layer_transmittance(thickness, w)
        end

        # Determine which interface we're at
        at_bottom = z == 0f0

        # Sample interface BSDF
        uc, rng = pcg32_uniform_f32(rng)
        u1, rng = pcg32_uniform_f32(rng)
        u2, rng = pcg32_uniform_f32(rng)
        u = Point2f(u1, u2)

        if at_bottom
            # Sample diffuse base (reflection only)
            bs_interface = sample_diffuse_interface(-w, u, refl_spectral, BXDF_ALL)
        else
            # Sample dielectric top (can reflect or transmit)
            bs_interface = sample_dielectric_interface(-w, uc, u, alpha_x, alpha_y, eta, BXDF_ALL)
        end

        if !bs_interface.valid || bs_interface.pdf == 0f0 || bs_interface.wi[3] == 0f0
            return SpectralBSDFSample()
        end

        f = f * bs_interface.f
        pdf *= bs_interface.pdf
        specular_path = specular_path && bs_interface.is_specular
        w = bs_interface.wi

        # Check if path has exited the layers (transmission through an interface)
        if !bs_interface.is_reflection
            # Ray has exited - determine final direction flags
            wi_local = w
            if flip_wi
                wi_local = -wi_local
            end

            wi = tangent * wi_local[1] + bitangent * wi_local[2] + n * wi_local[3]
            wi = normalize(wi)

            # pdfIsProportional=true for LayeredBxDF
            return SpectralBSDFSample(wi, f, pdf, specular_path, bs_interface.eta)
        end

        # Continuing random walk: multiply by AbsCosTheta for next segment
        f = f * abs(bs_interface.wi[3])
    end

    # Max depth reached without exiting
    return SpectralBSDFSample()
end

"""
    eval_dielectric_interface(wo, wi, alpha_x, alpha_y, eta) -> (f, pdf)

Evaluate the dielectric interface BSDF for given directions.
Returns (f_value, pdf) for the given wo/wi pair.
"""
@propagate_inbounds function eval_dielectric_interface(
    wo::Vec3f, wi::Vec3f, alpha_x::Float32, alpha_y::Float32, eta::Float32
)
    is_smooth = trowbridge_reitz_effectively_smooth(alpha_x, alpha_y)

    if is_smooth || eta == 1f0
        # Specular dielectric: f = 0 for non-delta directions
        return (SpectralRadiance(), 0f0)
    end

    # Rough dielectric evaluation
    if same_hemisphere(wo, wi)
        # Reflection
        wh = normalize(wo + wi)
        if wh[3] < 0f0
            wh = -wh
        end

        cos_θo_h = dot(wo, wh)
        R = fresnel_dielectric(cos_θo_h, eta)

        D = trowbridge_reitz_d(wh, alpha_x, alpha_y)
        G = trowbridge_reitz_g(wo, wi, alpha_x, alpha_y)

        f_val = D * G * R / (4f0 * wo[3] * wi[3])
        pdf = trowbridge_reitz_pdf(wo, wh, alpha_x, alpha_y) / (4f0 * abs(cos_θo_h))

        return (SpectralRadiance(f_val), pdf)
    else
        # Transmission
        etap = wo[3] > 0f0 ? eta : (1f0 / eta)

        # Compute half vector for transmission
        wh = normalize(wo + wi * etap)
        if wh[3] < 0f0
            wh = -wh
        end

        cos_θo_h = dot(wo, wh)
        cos_θi_h = dot(wi, wh)

        # Check for same side condition
        if cos_θo_h * cos_θi_h > 0f0
            return (SpectralRadiance(), 0f0)
        end

        R = fresnel_dielectric(cos_θo_h, eta)
        T = 1f0 - R

        denom = (cos_θi_h + cos_θo_h / etap)^2
        D = trowbridge_reitz_d(wh, alpha_x, alpha_y)
        G = trowbridge_reitz_g(wo, wi, alpha_x, alpha_y)

        f_val = T * D * G * abs(cos_θi_h * cos_θo_h / (wo[3] * wi[3] * denom))

        dwm_dwi = abs(cos_θi_h) / denom
        pdf = trowbridge_reitz_pdf(wo, wh, alpha_x, alpha_y) * dwm_dwi

        return (SpectralRadiance(f_val), pdf)
    end
end

"""
    pdf_dielectric_interface(wo, wi, alpha_x, alpha_y, eta, refl_trans_flags) -> Float32

Compute PDF of dielectric interface sampling.
"""
@propagate_inbounds function pdf_dielectric_interface(
    wo::Vec3f, wi::Vec3f, alpha_x::Float32, alpha_y::Float32, eta::Float32,
    refl_trans_flags::UInt8 = BXDF_ALL
)
    is_smooth = trowbridge_reitz_effectively_smooth(alpha_x, alpha_y)

    if is_smooth || eta == 1f0
        return 0f0  # Specular has delta PDF
    end

    if same_hemisphere(wo, wi)
        # Reflection
        if (refl_trans_flags & BXDF_REFLECTION) == 0
            return 0f0
        end

        wh = normalize(wo + wi)
        if wh[3] < 0f0
            wh = -wh
        end

        cos_θo_h = abs(dot(wo, wh))
        R = fresnel_dielectric(cos_θo_h, eta)
        T = 1f0 - R

        pr = (refl_trans_flags & BXDF_REFLECTION) != 0 ? R : 0f0
        pt = (refl_trans_flags & BXDF_TRANSMISSION) != 0 ? T : 0f0

        pdf = trowbridge_reitz_pdf(wo, wh, alpha_x, alpha_y) / (4f0 * cos_θo_h)
        return pdf * pr / (pr + pt)
    else
        # Transmission
        if (refl_trans_flags & BXDF_TRANSMISSION) == 0
            return 0f0
        end

        etap = wo[3] > 0f0 ? eta : (1f0 / eta)
        wh = normalize(wo + wi * etap)
        if wh[3] < 0f0
            wh = -wh
        end

        cos_θo_h = dot(wo, wh)
        cos_θi_h = dot(wi, wh)

        if cos_θo_h * cos_θi_h > 0f0
            return 0f0
        end

        R = fresnel_dielectric(abs(cos_θo_h), eta)
        T = 1f0 - R

        pr = (refl_trans_flags & BXDF_REFLECTION) != 0 ? R : 0f0
        pt = (refl_trans_flags & BXDF_TRANSMISSION) != 0 ? T : 0f0

        denom = (cos_θi_h + cos_θo_h / etap)^2
        dwm_dwi = abs(cos_θi_h) / denom
        pdf = trowbridge_reitz_pdf(wo, wh, alpha_x, alpha_y) * dwm_dwi

        return pdf * pt / (pr + pt)
    end
end

"""
    evaluate_bsdf_spectral(table, mat::CoatedDiffuseMaterial, textures, wo, wi, n, uv, lambda) -> (f, pdf)

Evaluate CoatedDiffuse BSDF using pbrt-v4's LayeredBxDF::f random walk algorithm.

This is a 100% port of pbrt-v4's LayeredBxDF::f. The algorithm uses nSamples
random walks to estimate the BSDF value using Monte Carlo integration with MIS.
"""
@propagate_inbounds function evaluate_bsdf_spectral(
    table::RGBToSpectrumTable, mat::CoatedDiffuseMaterial, textures,
    wo::Vec3f, wi::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
)
    # Get material properties
    refl_rgb = eval_tex(textures, mat.reflectance, uv)
    eta = mat.eta
    thickness = max(eval_tex(textures, mat.thickness, uv), eps(Float32))
    albedo_rgb = eval_tex(textures, mat.albedo, uv)
    g_val = clamp(eval_tex(textures, mat.g, uv), -0.99f0, 0.99f0)

    # Get roughness parameters
    u_roughness = eval_tex(textures, mat.u_roughness, uv)
    v_roughness = eval_tex(textures, mat.v_roughness, uv)

    alpha_x = mat.remap_roughness ? roughness_to_α(u_roughness) : u_roughness
    alpha_y = mat.remap_roughness ? roughness_to_α(v_roughness) : v_roughness

    refl_spectral = uplift_rgb(table, refl_rgb, lambda)
    albedo_spectral = uplift_rgb(table, albedo_rgb, lambda)
    has_medium = !is_black(albedo_rgb)

    n_samples = Int(mat.n_samples)
    max_depth = Int(mat.max_depth)

    # Build local frame
    tangent, bitangent = coordinate_system(n)
    cos_θo = dot(wo, n)
    cos_θi = dot(wi, n)

    wo_local = Vec3f(dot(wo, tangent), dot(wo, bitangent), cos_θo)
    wi_local = Vec3f(dot(wi, tangent), dot(wi, bitangent), cos_θi)

    # Two-sided handling: flip if entering from below
    if wo_local[3] < 0f0
        wo_local = -wo_local
        wi_local = -wi_local
    end

    # Check for grazing angles
    if abs(wo_local[3]) < 1f-6 || abs(wi_local[3]) < 1f-6
        return (SpectralRadiance(), 0f0)
    end

    # Determine entrance interface (always top after flip)
    entered_top = true

    # Determine exit interface based on wo/wi hemisphere relationship
    same_hemi = same_hemisphere(wo_local, wi_local)
    # For CoatedDiffuse (twoSided=true), exit_z logic:
    # SameHemisphere(wo,wi) ^ enteredTop -> if true, exit at bottom (z=0)
    exit_at_bottom = same_hemi ⊻ entered_top
    exit_z = exit_at_bottom ? 0f0 : thickness

    # Initialize result
    f_result = SpectralRadiance()

    # Account for reflection at entrance interface (same hemisphere case)
    if same_hemi
        enter_f, _ = eval_dielectric_interface(wo_local, wi_local, alpha_x, alpha_y, eta)
        f_result = f_result + enter_f * Float32(n_samples)
    end

    # Initialize RNG for evaluation (GPU-compatible functional style)
    seed = UInt64(0)
    rng = pcg32_init(pbrt_hash(seed, wo_local), pbrt_hash(wi_local))

    is_smooth = trowbridge_reitz_effectively_smooth(alpha_x, alpha_y)

    for s in 1:n_samples
        # Sample transmission through entrance interface (wo direction)
        uc, rng = pcg32_uniform_f32(rng)
        u1, rng = pcg32_uniform_f32(rng)
        u2, rng = pcg32_uniform_f32(rng)
        wos = sample_dielectric_interface(wo_local, uc, Point2f(u1, u2), alpha_x, alpha_y, eta, BXDF_TRANSMISSION)
        if !wos.valid || wos.pdf == 0f0 || wos.wi[3] == 0f0
            continue
        end

        # Sample "virtual light" from exit interface (wi direction)
        uc, rng = pcg32_uniform_f32(rng)
        u1, rng = pcg32_uniform_f32(rng)
        u2, rng = pcg32_uniform_f32(rng)
        if exit_at_bottom
            # Exit through diffuse base
            wis = sample_diffuse_interface(wi_local, Point2f(u1, u2), refl_spectral, BXDF_TRANSMISSION)
        else
            # Exit through dielectric top
            wis = sample_dielectric_interface(wi_local, uc, Point2f(u1, u2), alpha_x, alpha_y, eta, BXDF_TRANSMISSION)
        end
        if !wis.valid || wis.pdf == 0f0 || wis.wi[3] == 0f0
            continue
        end

        # Initialize random walk state
        beta = wos.f * abs(wos.wi[3]) / wos.pdf
        z = entered_top ? thickness : 0f0
        w = wos.wi

        for depth in 0:(max_depth-1)
            # Russian Roulette termination
            if depth > 3 && max_component(beta) < 0.25f0
                q = max(0f0, 1f0 - max_component(beta))
                rr_val, rng = pcg32_uniform_f32(rng)
                if rr_val < q
                    break
                end
                beta = beta / (1f0 - q)
            end

            if has_medium
                # Sample medium scattering
                sigma_t = 1f0
                exp_u, rng = pcg32_uniform_f32(rng)
                dz = sample_exponential(exp_u, sigma_t / abs(w[3]))
                zp = w[3] > 0f0 ? (z + dz) : (z - dz)

                if zp == z
                    continue
                end

                if 0f0 < zp && zp < thickness
                    # Scattering within medium - NEE contribution through exit
                    if exit_at_bottom
                        # Exit interface is diffuse - always non-specular
                        wt = power_heuristic(1, wis.pdf, 1, hg_phase_pdf(g_val, dot(-w, -wis.wi)))
                        f_exit, _ = eval_diffuse_interface(-w, -wis.wi, refl_spectral)
                    else
                        if !is_smooth
                            wt = power_heuristic(1, wis.pdf, 1, hg_phase_pdf(g_val, dot(-w, -wis.wi)))
                        else
                            wt = 1f0
                        end
                        f_exit, _ = eval_dielectric_interface(-w, -wis.wi, alpha_x, alpha_y, eta)
                    end

                    phase_val = hg_phase_pdf(g_val, dot(-w, -wis.wi))
                    f_result = f_result + beta * albedo_spectral * phase_val * wt *
                               layer_transmittance(zp - exit_z, wis.wi) * wis.f / wis.pdf

                    # Sample phase function for next segment
                    phase_u1, rng = pcg32_uniform_f32(rng)
                    phase_u2, rng = pcg32_uniform_f32(rng)
                    wi_phase, phase_p = sample_hg_phase_spectral(g_val, -w, Point2f(phase_u1, phase_u2))
                    if phase_p == 0f0 || wi_phase[3] == 0f0
                        break
                    end

                    beta = beta * albedo_spectral * phase_p / phase_p
                    w = wi_phase
                    z = zp

                    # NEE through exit interface after phase scattering
                    if ((z < exit_z && w[3] > 0f0) || (z > exit_z && w[3] < 0f0))
                        if exit_at_bottom
                            f_exit2, exit_pdf = eval_diffuse_interface(-w, wi_local, refl_spectral)
                        else
                            if !is_smooth
                                f_exit2, _ = eval_dielectric_interface(-w, wi_local, alpha_x, alpha_y, eta)
                                exit_pdf = pdf_dielectric_interface(-w, wi_local, alpha_x, alpha_y, eta, BXDF_TRANSMISSION)
                            else
                                continue  # Specular exit - no NEE
                            end
                        end
                        if max_component(f_exit2) > 0f0
                            wt2 = power_heuristic(1, phase_p, 1, exit_pdf)
                            f_result = f_result + beta * layer_transmittance(zp - exit_z, wi_phase) * f_exit2 * wt2
                        end
                    end

                    continue
                end

                z = clamp(zp, 0f0, thickness)
            else
                # No medium: advance to other interface
                z = (z == thickness) ? 0f0 : thickness
                beta = beta * layer_transmittance(thickness, w)
            end

            # Scattering at interface
            at_exit = z == exit_z

            if at_exit
                # At exit interface - sample reflection to continue walk
                uc, rng = pcg32_uniform_f32(rng)
                u1, rng = pcg32_uniform_f32(rng)
                u2, rng = pcg32_uniform_f32(rng)
                if exit_at_bottom
                    bs = sample_diffuse_interface(-w, Point2f(u1, u2), refl_spectral, BXDF_REFLECTION)
                else
                    bs = sample_dielectric_interface(-w, uc, Point2f(u1, u2), alpha_x, alpha_y, eta, BXDF_REFLECTION)
                end
                if !bs.valid || bs.pdf == 0f0 || bs.wi[3] == 0f0
                    break
                end
                beta = beta * bs.f * abs(bs.wi[3]) / bs.pdf
                w = bs.wi
            else
                # At non-exit interface - add NEE contribution
                non_exit_is_specular = (z == thickness) ? is_smooth : false  # top is dielectric, bottom is diffuse

                if !non_exit_is_specular
                    # Add NEE to exit direction
                    if z == thickness
                        # At top (dielectric)
                        f_nee, _ = eval_dielectric_interface(-w, -wis.wi, alpha_x, alpha_y, eta)
                    else
                        # At bottom (diffuse)
                        f_nee, _ = eval_diffuse_interface(-w, -wis.wi, refl_spectral)
                    end

                    if max_component(f_nee) > 0f0
                        wt = 1f0
                        if !exit_at_bottom || !is_smooth
                            # MIS weight
                            if z == thickness
                                nee_pdf = pdf_dielectric_interface(-w, -wis.wi, alpha_x, alpha_y, eta)
                            else
                                nee_pdf = pdf_diffuse_interface(-w, -wis.wi)
                            end
                            wt = power_heuristic(1, wis.pdf, 1, nee_pdf)
                        end
                        f_result = f_result + beta * f_nee * abs(wis.wi[3]) * wt *
                                   layer_transmittance(thickness, wis.wi) * wis.f / wis.pdf
                    end
                end

                # Sample new direction
                uc, rng = pcg32_uniform_f32(rng)
                u1, rng = pcg32_uniform_f32(rng)
                u2, rng = pcg32_uniform_f32(rng)
                if z == thickness
                    bs = sample_dielectric_interface(-w, uc, Point2f(u1, u2), alpha_x, alpha_y, eta, BXDF_REFLECTION)
                else
                    bs = sample_diffuse_interface(-w, Point2f(u1, u2), refl_spectral, BXDF_REFLECTION)
                end
                if !bs.valid || bs.pdf == 0f0 || bs.wi[3] == 0f0
                    break
                end

                beta = beta * bs.f * abs(bs.wi[3]) / bs.pdf
                w = bs.wi

                # NEE through exit after scattering
                if !is_smooth || exit_at_bottom
                    if exit_at_bottom
                        f_exit3, _ = eval_diffuse_interface(-w, wi_local, refl_spectral)
                    else
                        f_exit3, _ = eval_dielectric_interface(-w, wi_local, alpha_x, alpha_y, eta)
                    end

                    if max_component(f_exit3) > 0f0
                        wt3 = 1f0
                        if !non_exit_is_specular
                            if exit_at_bottom
                                exit_pdf3 = pdf_diffuse_interface(-w, wi_local)
                            else
                                exit_pdf3 = pdf_dielectric_interface(-w, wi_local, alpha_x, alpha_y, eta, BXDF_TRANSMISSION)
                            end
                            wt3 = power_heuristic(1, bs.pdf, 1, exit_pdf3)
                        end
                        f_result = f_result + beta * layer_transmittance(thickness, bs.wi) * f_exit3 * wt3
                    end
                end
            end
        end
    end

    f_result = f_result / Float32(n_samples)

    # Compute PDF using a simplified estimate
    # The full LayeredBxDF::PDF is complex; we use a reasonable approximation
    pdf = pdf_layered_bsdf(wo_local, wi_local, alpha_x, alpha_y, eta, n_samples, max_depth, refl_spectral, has_medium, g_val, thickness)

    return (f_result, pdf)
end

"""
    pdf_layered_bsdf(...) -> Float32

Compute PDF for LayeredBxDF using Monte Carlo estimation.
This is a simplified version of pbrt-v4's LayeredBxDF::PDF.
"""
@propagate_inbounds function pdf_layered_bsdf(
    wo::Vec3f, wi::Vec3f,
    alpha_x::Float32, alpha_y::Float32, eta::Float32,
    n_samples::Int, max_depth::Int,
    refl_spectral::SpectralRadiance,
    has_medium::Bool, g_val::Float32, thickness::Float32
)
    # Initialize RNG (GPU-compatible functional style)
    seed = UInt64(0)
    rng = pcg32_init(pbrt_hash(seed, wi), pbrt_hash(wo))

    entered_top = true
    same_hemi = same_hemisphere(wo, wi)
    is_smooth = trowbridge_reitz_effectively_smooth(alpha_x, alpha_y)

    pdf_sum = 0f0

    # Add entrance reflection PDF (same hemisphere)
    if same_hemi
        if is_smooth
            # Specular - delta PDF contribution
            pdf_sum += Float32(n_samples) * 0f0  # Delta has no continuous PDF
        else
            pdf_sum += Float32(n_samples) * pdf_dielectric_interface(wo, wi, alpha_x, alpha_y, eta, BXDF_REFLECTION)
        end
    end

    for s in 1:n_samples
        if same_hemi
            # TRT term
            # Sample transmission through top
            uc1, rng = pcg32_uniform_f32(rng)
            u1, rng = pcg32_uniform_f32(rng)
            u2, rng = pcg32_uniform_f32(rng)
            wos = sample_dielectric_interface(wo, uc1, Point2f(u1, u2), alpha_x, alpha_y, eta, BXDF_TRANSMISSION)

            uc2, rng = pcg32_uniform_f32(rng)
            u3, rng = pcg32_uniform_f32(rng)
            u4, rng = pcg32_uniform_f32(rng)
            wis = sample_dielectric_interface(wi, uc2, Point2f(u3, u4), alpha_x, alpha_y, eta, BXDF_TRANSMISSION)

            if wos.valid && wos.pdf > 0f0 && wis.valid && wis.pdf > 0f0
                if is_smooth
                    # Specular top - just use bottom PDF
                    pdf_sum += pdf_diffuse_interface(-wos.wi, -wis.wi)
                else
                    # MIS between paths
                    u5, rng = pcg32_uniform_f32(rng)
                    u6, rng = pcg32_uniform_f32(rng)
                    rs = sample_diffuse_interface(-wos.wi, Point2f(u5, u6), refl_spectral, BXDF_ALL)
                    if rs.valid && rs.pdf > 0f0
                        r_pdf = pdf_diffuse_interface(-wos.wi, -wis.wi)
                        wt = power_heuristic(1, wis.pdf, 1, r_pdf)
                        pdf_sum += wt * r_pdf

                        t_pdf = pdf_dielectric_interface(-rs.wi, wi, alpha_x, alpha_y, eta)
                        wt2 = power_heuristic(1, rs.pdf, 1, t_pdf)
                        pdf_sum += wt2 * t_pdf
                    end
                end
            end
        else
            # TT term
            uc1, rng = pcg32_uniform_f32(rng)
            u1, rng = pcg32_uniform_f32(rng)
            u2, rng = pcg32_uniform_f32(rng)
            wos = sample_dielectric_interface(wo, uc1, Point2f(u1, u2), alpha_x, alpha_y, eta, BXDF_TRANSMISSION)
            if !wos.valid || wos.pdf == 0f0 || wos.is_reflection
                continue
            end

            u3, rng = pcg32_uniform_f32(rng)
            u4, rng = pcg32_uniform_f32(rng)
            wis = sample_diffuse_interface(wi, Point2f(u3, u4), refl_spectral, BXDF_TRANSMISSION)
            if !wis.valid || wis.pdf == 0f0 || wis.is_reflection
                continue
            end

            if is_smooth
                pdf_sum += pdf_diffuse_interface(-wos.wi, wi)
            else
                pdf_sum += (pdf_dielectric_interface(wo, -wis.wi, alpha_x, alpha_y, eta) +
                           pdf_diffuse_interface(-wos.wi, wi)) / 2f0
            end
        end
    end

    # Return mixture of PDF estimate and constant PDF (as per pbrt-v4)
    return lerp(0.9f0, 1f0 / (4f0 * Float32(π)), pdf_sum / Float32(n_samples))
end

"""
    get_emission_spectral for CoatedDiffuseMaterial - returns zero (non-emissive).
"""
@propagate_inbounds function get_emission_spectral(
    table::RGBToSpectrumTable, mat::CoatedDiffuseMaterial, textures,
    wo::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
)
    return SpectralRadiance()
end

"""
    get_albedo_spectral for CoatedDiffuseMaterial.
"""
@propagate_inbounds function get_albedo_spectral(table::RGBToSpectrumTable, mat::CoatedDiffuseMaterial, textures, uv::Point2f, lambda::Wavelengths)
    return uplift_rgb(table, eval_tex(textures, mat.reflectance, uv), lambda)
end

# ============================================================================
# ThinDielectricMaterial - Thin Dielectric Surface (pbrt-v4 port)
# ============================================================================

"""
    sample_bsdf_spectral(table, mat::ThinDielectricMaterial, textures, wo, n, uv, lambda, sample_u, rng, regularize=false) -> SpectralBSDFSample

Sample thin dielectric BSDF matching pbrt-v4's ThinDielectricBxDF::Sample_f.

Thin dielectric surfaces model materials like window glass where light can
either reflect or transmit straight through (no refraction bend).

Key physics (pbrt-v4 lines 225-230):
- R₀ = FrDielectric(|cos_θ|, eta)
- R = R₀ + T₀²R₀/(1 - R₀²)  where T₀ = 1 - R₀
- T = 1 - R
- Transmitted direction: wi = -wo (straight through)
- Reflected direction: wi = (-wo.x, -wo.y, wo.z) (mirror reflection in local coords)
"""
@propagate_inbounds function sample_bsdf_spectral(
    table::RGBToSpectrumTable, mat::ThinDielectricMaterial, textures,
    wo::Vec3f, n::Vec3f, uv::Point2f,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32,
    regularize::Bool = false
)
    # Check for grazing angle
    wo_dot_n = dot(wo, n)
    if abs(wo_dot_n) < 1f-6
        return SpectralBSDFSample()
    end

    eta = mat.eta

    # Build local coordinate frame
    tangent, bitangent = coordinate_system(n)
    wo_local = Vec3f(dot(wo, tangent), dot(wo, bitangent), wo_dot_n)

    # Compute single-interface Fresnel reflectance
    cos_θo = abs(wo_local[3])
    R0 = fresnel_dielectric(cos_θo, eta)
    T0 = 1f0 - R0

    # Account for multiple internal bounces (pbrt-v4 lines 227-230)
    # R = R0 + T0² * R0 / (1 - R0²)
    R = R0
    if R0 < 1f0
        R = R0 + T0 * T0 * R0 / (1f0 - R0 * R0)
    end
    T = 1f0 - R

    # Choose reflection or transmission based on rng
    pr = R
    pt = T

    if pr + pt < 1f-10
        return SpectralBSDFSample()
    end

    prob_reflect = pr / (pr + pt)

    if rng < prob_reflect
        # Sample perfect specular reflection
        # wi = (-wo.x, -wo.y, wo.z) in local coords
        wi_local = Vec3f(-wo_local[1], -wo_local[2], wo_local[3])

        # Transform back to world space
        wi = tangent * wi_local[1] + bitangent * wi_local[2] + n * wi_local[3]
        wi = normalize(wi)

        # f = R / |cos_θ|
        f_val = R / abs(wi_local[3])
        return SpectralBSDFSample(wi, SpectralRadiance(f_val), prob_reflect, true, 1f0)
    else
        # Sample perfect specular transmission
        # wi = -wo (straight through, no refraction)
        wi = -wo

        # f = T / |cos_θ|
        f_val = T / cos_θo
        return SpectralBSDFSample(wi, SpectralRadiance(f_val), 1f0 - prob_reflect, true, 1f0)
    end
end

"""
    evaluate_bsdf_spectral(table, mat::ThinDielectricMaterial, textures, wo, wi, n, uv, lambda) -> (f, pdf)

Evaluate thin dielectric BSDF - returns zero for non-delta directions.
ThinDielectric is purely specular, so f() and PDF() both return 0.
"""
@propagate_inbounds function evaluate_bsdf_spectral(
    table::RGBToSpectrumTable, mat::ThinDielectricMaterial, textures,
    wo::Vec3f, wi::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
)
    # ThinDielectric is purely specular - f() returns 0 for all non-delta directions
    return (SpectralRadiance(), 0f0)
end

"""
    get_emission_spectral for ThinDielectricMaterial - returns zero (non-emissive).
"""
@propagate_inbounds function get_emission_spectral(
    table::RGBToSpectrumTable, mat::ThinDielectricMaterial, textures,
    wo::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
)
    return SpectralRadiance()
end

"""
    get_albedo_spectral for ThinDielectricMaterial - returns white (transparent).
"""
@propagate_inbounds function get_albedo_spectral(table::RGBToSpectrumTable, mat::ThinDielectricMaterial, textures, uv::Point2f, lambda::Wavelengths)
    # For thin dielectric, albedo is effectively white (transparent material)
    return SpectralRadiance(1f0)
end

# ============================================================================
# DiffuseTransmissionMaterial - Diffuse Reflection/Transmission (pbrt-v4 port)
# ============================================================================

"""
    sample_bsdf_spectral(table, mat::DiffuseTransmissionMaterial, textures, wo, n, uv, lambda, sample_u, rng, regularize=false) -> SpectralBSDFSample

Sample diffuse transmission BSDF matching pbrt-v4's DiffuseTransmissionBxDF::Sample_f.

This material diffusely scatters light in both reflection (same hemisphere)
and transmission (opposite hemisphere). Sampling is proportional to max(R) and max(T).
"""
@propagate_inbounds function sample_bsdf_spectral(
    table::RGBToSpectrumTable, mat::DiffuseTransmissionMaterial, textures,
    wo::Vec3f, n::Vec3f, uv::Point2f,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32,
    regularize::Bool = false
)
    # Check for grazing angle
    wo_dot_n = dot(wo, n)
    if abs(wo_dot_n) < 1f-6
        return SpectralBSDFSample()
    end

    # Get material properties and apply scale
    r_rgb = eval_tex(textures, mat.reflectance, uv) * mat.scale
    t_rgb = eval_tex(textures, mat.transmittance, uv) * mat.scale

    # Clamp to [0, 1]
    r_rgb = RGBSpectrum(clamp(r_rgb.c[1], 0f0, 1f0), clamp(r_rgb.c[2], 0f0, 1f0), clamp(r_rgb.c[3], 0f0, 1f0))
    t_rgb = RGBSpectrum(clamp(t_rgb.c[1], 0f0, 1f0), clamp(t_rgb.c[2], 0f0, 1f0), clamp(t_rgb.c[3], 0f0, 1f0))

    r_spectral = uplift_rgb(table, r_rgb, lambda)
    t_spectral = uplift_rgb(table, t_rgb, lambda)

    # Compute probabilities based on max component (pbrt-v4 lines 102-108)
    pr = max(r_rgb.c[1], r_rgb.c[2], r_rgb.c[3])
    pt = max(t_rgb.c[1], t_rgb.c[2], t_rgb.c[3])

    if pr + pt < 1f-10
        return SpectralBSDFSample()
    end

    # Build local coordinate frame
    tangent, bitangent = coordinate_system(n)
    wo_local = Vec3f(dot(wo, tangent), dot(wo, bitangent), wo_dot_n)

    prob_reflect = pr / (pr + pt)

    if rng < prob_reflect
        # Sample diffuse reflection (same hemisphere as wo)
        local_wi = cosine_sample_hemisphere(sample_u)

        # Flip to same hemisphere as wo
        if wo_local[3] < 0f0
            local_wi = Vec3f(local_wi[1], local_wi[2], -local_wi[3])
        end

        cos_theta = abs(local_wi[3])
        if cos_theta < 1f-6
            return SpectralBSDFSample()
        end

        wi = tangent * local_wi[1] + bitangent * local_wi[2] + n * local_wi[3]
        wi = normalize(wi)

        # f = R / π
        f_spectral = r_spectral * (1f0 / Float32(π))
        pdf = prob_reflect * cos_theta / Float32(π)

        return SpectralBSDFSample(wi, f_spectral, pdf, false, 1f0)
    else
        # Sample diffuse transmission (opposite hemisphere from wo)
        local_wi = cosine_sample_hemisphere(sample_u)

        # Flip to opposite hemisphere from wo
        if wo_local[3] > 0f0
            local_wi = Vec3f(local_wi[1], local_wi[2], -local_wi[3])
        end

        cos_theta = abs(local_wi[3])
        if cos_theta < 1f-6
            return SpectralBSDFSample()
        end

        wi = tangent * local_wi[1] + bitangent * local_wi[2] + n * local_wi[3]
        wi = normalize(wi)

        # f = T / π
        f_spectral = t_spectral * (1f0 / Float32(π))
        pdf = (1f0 - prob_reflect) * cos_theta / Float32(π)

        return SpectralBSDFSample(wi, f_spectral, pdf, false, 1f0)
    end
end

"""
    evaluate_bsdf_spectral(table, mat::DiffuseTransmissionMaterial, textures, wo, wi, n, uv, lambda) -> (f, pdf)

Evaluate diffuse transmission BSDF matching pbrt-v4's DiffuseTransmissionBxDF::f and PDF.
"""
@propagate_inbounds function evaluate_bsdf_spectral(
    table::RGBToSpectrumTable, mat::DiffuseTransmissionMaterial, textures,
    wo::Vec3f, wi::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
)
    cos_θi = dot(wi, n)
    cos_θo = dot(wo, n)

    abs_cos_θi = abs(cos_θi)
    if abs_cos_θi < 1f-6
        return (SpectralRadiance(), 0f0)
    end

    # Get material properties and apply scale
    r_rgb = eval_tex(textures, mat.reflectance, uv) * mat.scale
    t_rgb = eval_tex(textures, mat.transmittance, uv) * mat.scale

    # Clamp to [0, 1]
    r_rgb = RGBSpectrum(clamp(r_rgb.c[1], 0f0, 1f0), clamp(r_rgb.c[2], 0f0, 1f0), clamp(r_rgb.c[3], 0f0, 1f0))
    t_rgb = RGBSpectrum(clamp(t_rgb.c[1], 0f0, 1f0), clamp(t_rgb.c[2], 0f0, 1f0), clamp(t_rgb.c[3], 0f0, 1f0))

    r_spectral = uplift_rgb(table, r_rgb, lambda)
    t_spectral = uplift_rgb(table, t_rgb, lambda)

    # Probabilities
    pr = max(r_rgb.c[1], r_rgb.c[2], r_rgb.c[3])
    pt = max(t_rgb.c[1], t_rgb.c[2], t_rgb.c[3])

    if pr + pt < 1f-10
        return (SpectralRadiance(), 0f0)
    end

    same_hemisphere = (cos_θi * cos_θo) > 0f0

    if same_hemisphere
        # Reflection: f = R / π
        f_spectral = r_spectral * (1f0 / Float32(π))
        prob_reflect = pr / (pr + pt)
        pdf = prob_reflect * abs_cos_θi / Float32(π)
        return (f_spectral, pdf)
    else
        # Transmission: f = T / π
        f_spectral = t_spectral * (1f0 / Float32(π))
        prob_transmit = pt / (pr + pt)
        pdf = prob_transmit * abs_cos_θi / Float32(π)
        return (f_spectral, pdf)
    end
end

"""
    get_emission_spectral for DiffuseTransmissionMaterial - returns zero (non-emissive).
"""
@propagate_inbounds function get_emission_spectral(
    table::RGBToSpectrumTable, mat::DiffuseTransmissionMaterial, textures,
    wo::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
)
    return SpectralRadiance()
end

"""
    get_albedo_spectral for DiffuseTransmissionMaterial.
"""
@propagate_inbounds function get_albedo_spectral(table::RGBToSpectrumTable, mat::DiffuseTransmissionMaterial, textures, uv::Point2f, lambda::Wavelengths)
    # Return average of reflectance and transmittance
    r_rgb = eval_tex(textures, mat.reflectance, uv) * mat.scale
    t_rgb = eval_tex(textures, mat.transmittance, uv) * mat.scale
    avg = RGBSpectrum((r_rgb.c[1] + t_rgb.c[1]) * 0.5f0,
                      (r_rgb.c[2] + t_rgb.c[2]) * 0.5f0,
                      (r_rgb.c[3] + t_rgb.c[3]) * 0.5f0)
    return uplift_rgb(table, avg, lambda)
end

# ============================================================================
# CoatedConductorMaterial - Layered dielectric over conductor (pbrt-v4 port)
# ============================================================================

"""
    sample_bsdf_spectral(table, mat::CoatedConductorMaterial, textures, wo, n, uv, lambda, sample_u, rng, regularize=false) -> SpectralBSDFSample

Sample CoatedConductor BSDF using pbrt-v4's LayeredBxDF approach.

This is a layered material with:
- Top layer: Dielectric coating (can be rough or smooth)
- Bottom layer: Conductor (metal) with complex Fresnel

Key pbrt-v4 details (materials.cpp lines 345-392):
- Conductor eta/k are scaled by interface IOR: ce /= ieta, ck /= ieta
- If reflectance mode: k = 2 * sqrt(r) / sqrt(1 - r), eta = 1

When `regularize=true`, both interface and conductor microfacet alphas are increased
to reduce fireflies from near-specular paths (matches pbrt-v4 BSDF::Regularize).
"""
@propagate_inbounds function sample_bsdf_spectral(
    table::RGBToSpectrumTable, mat::CoatedConductorMaterial, textures,
    wo::Vec3f, n::Vec3f, uv::Point2f,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32,
    regularize::Bool = false
)
    # Check for grazing angle
    wo_dot_n = dot(wo, n)
    if abs(wo_dot_n) < 1f-6
        return SpectralBSDFSample()
    end

    # Get interface (coating) parameters
    ieta = mat.interface_eta
    if ieta == 0f0
        ieta = 1f0
    end

    iu_roughness = eval_tex(textures, mat.interface_u_roughness, uv)
    iv_roughness = eval_tex(textures, mat.interface_v_roughness, uv)
    i_alpha_x = mat.remap_roughness ? roughness_to_α(iu_roughness) : iu_roughness
    i_alpha_y = mat.remap_roughness ? roughness_to_α(iv_roughness) : iv_roughness

    # Get conductor parameters
    cu_roughness = eval_tex(textures, mat.conductor_u_roughness, uv)
    cv_roughness = eval_tex(textures, mat.conductor_v_roughness, uv)
    c_alpha_x = mat.remap_roughness ? roughness_to_α(cu_roughness) : cu_roughness
    c_alpha_y = mat.remap_roughness ? roughness_to_α(cv_roughness) : cv_roughness

    # Apply regularization if requested (pbrt-v4: doubles alpha if < 0.3, clamps to [0.1, 0.3])
    if regularize
        i_alpha_x = regularize_alpha(i_alpha_x)
        i_alpha_y = regularize_alpha(i_alpha_y)
        c_alpha_x = regularize_alpha(c_alpha_x)
        c_alpha_y = regularize_alpha(c_alpha_y)
    end

    # Get conductor eta/k - either from eta/k textures or derived from reflectance
    local ce_spectral::SpectralRadiance
    local ck_spectral::SpectralRadiance

    if mat.use_eta_k
        ce_rgb = eval_tex(textures, mat.conductor_eta, uv)
        ck_rgb = eval_tex(textures, mat.conductor_k, uv)
        ce_spectral = uplift_rgb_unbounded(table, ce_rgb, lambda)
        ck_spectral = uplift_rgb_unbounded(table, ck_rgb, lambda)
    else
        # Reflectance mode: eta = 1, k = 2 * sqrt(r) / sqrt(1 - r)
        refl_rgb = eval_tex(textures, mat.reflectance, uv)
        # Clamp to avoid r==1 NaN (pbrt-v4 line 371)
        refl_rgb = RGBSpectrum(
            clamp(refl_rgb.c[1], 0f0, 0.9999f0),
            clamp(refl_rgb.c[2], 0f0, 0.9999f0),
            clamp(refl_rgb.c[3], 0f0, 0.9999f0)
        )
        r_spectral = uplift_rgb(table, refl_rgb, lambda)
        ce_spectral = SpectralRadiance(1f0)
        # k = 2 * sqrt(r) / sqrt(1 - r) (pbrt-v4 line 373)
        ck_spectral = 2f0 * sqrt(r_spectral) / sqrt(clamp_zero(SpectralRadiance(1f0) - r_spectral) + SpectralRadiance(1f-6))
    end

    # Critical: scale conductor eta/k by interface IOR (pbrt-v4 lines 375-376)
    ce_spectral = ce_spectral / ieta
    ck_spectral = ck_spectral / ieta

    # Volumetric parameters
    thickness = max(eval_tex(textures, mat.thickness, uv), eps(Float32))
    albedo_rgb = eval_tex(textures, mat.albedo, uv)
    albedo_spectral = uplift_rgb(table, albedo_rgb, lambda)
    g_val = clamp(eval_tex(textures, mat.g, uv), -0.99f0, 0.99f0)
    has_medium = !is_black(albedo_rgb)

    # Build coordinate system from shading normal
    tangent, bitangent = coordinate_system(n)

    # Transform wo to local space
    wo_local = Vec3f(dot(wo, tangent), dot(wo, bitangent), wo_dot_n)

    # Two-sided: flip if entering from below
    flip = wo_local[3] < 0f0
    if flip
        wo_local = -wo_local
    end

    cos_θo = abs(wo_local[3])

    # Check if interface coating is effectively smooth
    i_is_smooth = trowbridge_reitz_effectively_smooth(i_alpha_x, i_alpha_y)
    c_is_smooth = trowbridge_reitz_effectively_smooth(c_alpha_x, c_alpha_y)

    if i_is_smooth
        # === Smooth interface coating ===
        F_interface = fresnel_dielectric(cos_θo, ieta)

        if rng < F_interface
            # Specular reflection at coating surface
            wi_local = Vec3f(-wo_local[1], -wo_local[2], wo_local[3])
            if flip
                wi_local = -wi_local
            end

            wi = tangent * wi_local[1] + bitangent * wi_local[2] + n * wi_local[3]
            wi = normalize(wi)

            f_spectral = SpectralRadiance(1f0)
            return SpectralBSDFSample(wi, f_spectral, 1f0, true, 1f0)
        end

        # Transmitted through interface - now sample conductor base
        # Refract direction into coating
        sin2_θt = max(0f0, 1f0 - cos_θo^2) / (ieta^2)
        if sin2_θt >= 1f0
            return SpectralBSDFSample()  # TIR at interface
        end
        cos_θt_in = sqrt(1f0 - sin2_θt)

        if c_is_smooth
            # Smooth conductor: perfect reflection at base
            # wi in coating space points straight up after reflection
            wi_base = Vec3f(-wo_local[1] / ieta, -wo_local[2] / ieta, cos_θt_in)
            wi_base = normalize(wi_base)

            # Conductor Fresnel (using scaled eta/k)
            F_conductor = fr_complex_spectral(cos_θt_in, ce_spectral, ck_spectral)

            # Refract back out through interface
            sin2_θ_out = max(0f0, 1f0 - wi_base[3]^2) * (ieta^2)
            if sin2_θ_out >= 1f0
                return SpectralBSDFSample()  # TIR on way out
            end
            cos_θ_out = sqrt(1f0 - sin2_θ_out)

            F_out = fresnel_dielectric(cos_θ_out, ieta)
            T_in = 1f0 - F_interface
            T_out = 1f0 - F_out

            # Layer transmittance through medium
            layer_tr = if has_medium
                tr = layer_transmittance(thickness, Vec3f(0, 0, cos_θt_in))
                tr * tr * albedo_spectral
            else
                SpectralRadiance(1f0)
            end

            # Final direction is reflection
            wi_local = Vec3f(-wo_local[1], -wo_local[2], wo_local[3])

            if flip
                wi_local = -wi_local
            end

            wi = tangent * wi_local[1] + bitangent * wi_local[2] + n * wi_local[3]
            wi = normalize(wi)

            f_spectral = F_conductor * T_in * T_out * layer_tr / cos_θo
            return SpectralBSDFSample(wi, f_spectral, 1f0 - F_interface, true, 1f0)
        else
            # Rough conductor: sample microfacet
            # Transform wo to conductor local frame (inside coating)
            wo_conductor = Vec3f(wo_local[1] / ieta, wo_local[2] / ieta, cos_θt_in)
            wo_conductor = normalize(wo_conductor)

            # Clamp conductor alpha
            c_alpha_x = max(c_alpha_x, 1f-4)
            c_alpha_y = max(c_alpha_y, 1f-4)

            # Sample conductor microfacet
            wm = trowbridge_reitz_sample_wm(wo_conductor, sample_u, c_alpha_x, c_alpha_y)
            cos_θo_m = dot(wo_conductor, wm)
            if cos_θo_m < 0f0
                return SpectralBSDFSample()
            end

            # Reflect off conductor microfacet
            wi_conductor = -wo_conductor + 2f0 * cos_θo_m * wm
            if wi_conductor[3] < 0f0
                return SpectralBSDFSample()
            end

            # Conductor Fresnel at microfacet
            F_conductor = fr_complex_spectral(abs(cos_θo_m), ce_spectral, ck_spectral)

            # Conductor microfacet BRDF
            D = trowbridge_reitz_d(wm, c_alpha_x, c_alpha_y)
            G = trowbridge_reitz_g(wo_conductor, wi_conductor, c_alpha_x, c_alpha_y)
            f_conductor = D * F_conductor * G / (4f0 * abs(wo_conductor[3]) * abs(wi_conductor[3]))

            # Refract outgoing direction back through interface
            sin2_θ_out = (wi_conductor[1]^2 + wi_conductor[2]^2) * (ieta^2)
            if sin2_θ_out >= 1f0
                return SpectralBSDFSample()  # TIR
            end
            cos_θ_out = sqrt(1f0 - sin2_θ_out)

            F_out = fresnel_dielectric(cos_θ_out, ieta)
            T_in = 1f0 - F_interface
            T_out = 1f0 - F_out

            layer_tr = if has_medium
                tr_in = layer_transmittance(thickness, Vec3f(0, 0, cos_θt_in))
                tr_out = layer_transmittance(thickness, Vec3f(0, 0, wi_conductor[3]))
                tr_in * tr_out * albedo_spectral
            else
                SpectralRadiance(1f0)
            end

            # Transform wi back to world
            wi_local = Vec3f(wi_conductor[1] * ieta, wi_conductor[2] * ieta, cos_θ_out)
            wi_local = normalize(wi_local)

            if flip
                wi_local = -wi_local
            end

            wi = tangent * wi_local[1] + bitangent * wi_local[2] + n * wi_local[3]
            wi = normalize(wi)

            f_spectral = f_conductor * T_in * T_out * layer_tr

            # PDF for conductor sampling
            pdf_m = trowbridge_reitz_pdf(wo_conductor, wm, c_alpha_x, c_alpha_y)
            pdf_conductor = pdf_m / (4f0 * abs(cos_θo_m))
            pdf = (1f0 - F_interface) * pdf_conductor

            return SpectralBSDFSample(wi, f_spectral, pdf, false, 1f0)
        end
    else
        # === Rough interface coating ===
        # Clamp interface alpha
        i_alpha_x = max(i_alpha_x, 1f-4)
        i_alpha_y = max(i_alpha_y, 1f-4)

        # Sample interface microfacet normal
        wm = trowbridge_reitz_sample_wm(wo_local, sample_u, i_alpha_x, i_alpha_y)
        cos_θo_m = dot(wo_local, wm)
        if cos_θo_m < 0f0
            return SpectralBSDFSample()
        end

        F_interface = fresnel_dielectric(cos_θo_m, ieta)

        if rng < F_interface
            # Reflect off interface microfacet
            wi_local = -wo_local + 2f0 * cos_θo_m * wm

            if wi_local[3] * wo_local[3] < 0f0
                return SpectralBSDFSample()
            end

            if flip
                wi_local = -wi_local
            end

            wi = tangent * wi_local[1] + bitangent * wi_local[2] + n * wi_local[3]
            wi = normalize(wi)

            # Interface microfacet BRDF (dielectric)
            D = trowbridge_reitz_d(wm, i_alpha_x, i_alpha_y)
            G = trowbridge_reitz_g(wo_local, wi_local, i_alpha_x, i_alpha_y)

            cos_i = abs(wi_local[3])
            cos_o = abs(wo_local[3])

            pdf_m = trowbridge_reitz_pdf(wo_local, wm, i_alpha_x, i_alpha_y)
            pdf = F_interface * pdf_m / (4f0 * abs(cos_θo_m))

            f = D * G / (4f0 * cos_i * cos_o)

            return SpectralBSDFSample(wi, SpectralRadiance(f), pdf, false, 1f0)
        else
            # Transmit through rough interface to conductor
            # For rough interface, approximate with average transmission
            T_in = 1f0 - F_interface

            # Sample conductor - simplified for rough interface case
            # Use a cosine-weighted sample centered around specular direction
            local_conductor_wi = Vec3f(-wo_local[1], -wo_local[2], wo_local[3])

            if c_is_smooth
                # Smooth conductor with rough interface
                cos_θ_base = abs(local_conductor_wi[3])
                F_conductor = fr_complex_spectral(cos_θ_base, ce_spectral, ck_spectral)

                F_out = fresnel_dielectric(cos_θ_base, ieta)
                T_out = 1f0 - F_out

                layer_tr = if has_medium
                    tr = layer_transmittance(thickness, local_conductor_wi)
                    tr * tr * albedo_spectral
                else
                    SpectralRadiance(1f0)
                end

                if flip
                    local_conductor_wi = -local_conductor_wi
                end

                wi = tangent * local_conductor_wi[1] + bitangent * local_conductor_wi[2] + n * local_conductor_wi[3]
                wi = normalize(wi)

                f_spectral = F_conductor * T_in * T_out * layer_tr / cos_θo

                pdf_m = trowbridge_reitz_pdf(wo_local, wm, i_alpha_x, i_alpha_y)
                pdf = (1f0 - F_interface) * pdf_m / (4f0 * abs(cos_θo_m))

                return SpectralBSDFSample(wi, f_spectral, pdf, false, 1f0)
            else
                # Both interface and conductor rough
                # Use the interface-sampled direction for conductor evaluation
                c_alpha_x = max(c_alpha_x, 1f-4)
                c_alpha_y = max(c_alpha_y, 1f-4)

                # Sample conductor from the transmitted direction
                wm_c = trowbridge_reitz_sample_wm(wo_local, sample_u, c_alpha_x, c_alpha_y)
                cos_θo_mc = dot(wo_local, wm_c)
                if cos_θo_mc < 0f0
                    return SpectralBSDFSample()
                end

                wi_local = -wo_local + 2f0 * cos_θo_mc * wm_c
                if wi_local[3] * wo_local[3] < 0f0
                    return SpectralBSDFSample()
                end

                F_conductor = fr_complex_spectral(abs(cos_θo_mc), ce_spectral, ck_spectral)

                D = trowbridge_reitz_d(wm_c, c_alpha_x, c_alpha_y)
                G = trowbridge_reitz_g(wo_local, wi_local, c_alpha_x, c_alpha_y)

                cos_i = abs(wi_local[3])
                cos_o = abs(wo_local[3])

                f_conductor = D * F_conductor * G / (4f0 * cos_i * cos_o)

                F_out = fresnel_dielectric(cos_i, ieta)
                T_out = 1f0 - F_out

                layer_tr = if has_medium
                    tr_in = layer_transmittance(thickness, Vec3f(0, 0, cos_o))
                    tr_out = layer_transmittance(thickness, wi_local)
                    tr_in * tr_out * albedo_spectral
                else
                    SpectralRadiance(1f0)
                end

                if flip
                    wi_local = -wi_local
                end

                wi = tangent * wi_local[1] + bitangent * wi_local[2] + n * wi_local[3]
                wi = normalize(wi)

                f_spectral = f_conductor * T_in * T_out * layer_tr

                pdf_m = trowbridge_reitz_pdf(wo_local, wm_c, c_alpha_x, c_alpha_y)
                pdf = (1f0 - F_interface) * pdf_m / (4f0 * abs(cos_θo_mc))

                return SpectralBSDFSample(wi, f_spectral, pdf, false, 1f0)
            end
        end
    end
end

"""
    evaluate_bsdf_spectral(table, mat::CoatedConductorMaterial, textures, wo, wi, n, uv, lambda) -> (f, pdf)

Evaluate CoatedConductor BSDF for given directions.
"""
@propagate_inbounds function evaluate_bsdf_spectral(
    table::RGBToSpectrumTable, mat::CoatedConductorMaterial, textures,
    wo::Vec3f, wi::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
)
    # Check hemisphere - coated conductor only reflects
    cos_θi = dot(wi, n)
    cos_θo = dot(wo, n)
    if cos_θi * cos_θo < 0f0
        return (SpectralRadiance(), 0f0)
    end

    abs_cos_θi = abs(cos_θi)
    abs_cos_θo = abs(cos_θo)

    if abs_cos_θi < 1f-6 || abs_cos_θo < 1f-6
        return (SpectralRadiance(), 0f0)
    end

    # Get interface parameters
    ieta = mat.interface_eta
    if ieta == 0f0
        ieta = 1f0
    end

    iu_roughness = eval_tex(textures, mat.interface_u_roughness, uv)
    iv_roughness = eval_tex(textures, mat.interface_v_roughness, uv)
    i_alpha_x = mat.remap_roughness ? roughness_to_α(iu_roughness) : iu_roughness
    i_alpha_y = mat.remap_roughness ? roughness_to_α(iv_roughness) : iv_roughness

    # Get conductor parameters
    cu_roughness = eval_tex(textures, mat.conductor_u_roughness, uv)
    cv_roughness = eval_tex(textures, mat.conductor_v_roughness, uv)
    c_alpha_x = mat.remap_roughness ? roughness_to_α(cu_roughness) : cu_roughness
    c_alpha_y = mat.remap_roughness ? roughness_to_α(cv_roughness) : cv_roughness

    # Get conductor eta/k
    local ce_spectral::SpectralRadiance
    local ck_spectral::SpectralRadiance

    if mat.use_eta_k
        ce_rgb = eval_tex(textures, mat.conductor_eta, uv)
        ck_rgb = eval_tex(textures, mat.conductor_k, uv)
        ce_spectral = uplift_rgb_unbounded(table, ce_rgb, lambda)
        ck_spectral = uplift_rgb_unbounded(table, ck_rgb, lambda)
    else
        refl_rgb = eval_tex(textures, mat.reflectance, uv)
        refl_rgb = RGBSpectrum(
            clamp(refl_rgb.c[1], 0f0, 0.9999f0),
            clamp(refl_rgb.c[2], 0f0, 0.9999f0),
            clamp(refl_rgb.c[3], 0f0, 0.9999f0)
        )
        r_spectral = uplift_rgb(table, refl_rgb, lambda)
        ce_spectral = SpectralRadiance(1f0)
        ck_spectral = 2f0 * sqrt(r_spectral) / sqrt(clamp_zero(SpectralRadiance(1f0) - r_spectral) + SpectralRadiance(1f-6))
    end

    # Scale by interface IOR
    ce_spectral = ce_spectral / ieta
    ck_spectral = ck_spectral / ieta

    # Volumetric parameters
    thickness = max(eval_tex(textures, mat.thickness, uv), eps(Float32))
    albedo_rgb = eval_tex(textures, mat.albedo, uv)
    albedo_spectral = uplift_rgb(table, albedo_rgb, lambda)
    has_medium = !is_black(albedo_rgb)

    # Build local frame
    tangent, bitangent = coordinate_system(n)
    wo_local = Vec3f(dot(wo, tangent), dot(wo, bitangent), cos_θo)
    wi_local = Vec3f(dot(wi, tangent), dot(wi, bitangent), cos_θi)

    flip = wo_local[3] < 0f0
    if flip
        wo_local = -wo_local
        wi_local = -wi_local
    end

    i_is_smooth = trowbridge_reitz_effectively_smooth(i_alpha_x, i_alpha_y)
    c_is_smooth = trowbridge_reitz_effectively_smooth(c_alpha_x, c_alpha_y)

    if i_is_smooth && c_is_smooth
        # Both smooth - delta functions, return zero for non-delta evaluation
        return (SpectralRadiance(), 0f0)
    end

    # Compute half-vector
    wh = normalize(wo_local + wi_local)
    if wh[3] < 0f0
        wh = -wh
    end
    cos_θo_h = dot(wo_local, wh)

    # Interface Fresnel
    F_interface_wh = fresnel_dielectric(abs(cos_θo_h), ieta)
    F_interface_o = fresnel_dielectric(abs(wo_local[3]), ieta)
    F_interface_i = fresnel_dielectric(abs(wi_local[3]), ieta)

    if i_is_smooth
        # Smooth interface, rough conductor
        # Only conductor contribution (interface specular is delta)
        T_o = 1f0 - F_interface_o
        T_i = 1f0 - F_interface_i

        c_alpha_x = max(c_alpha_x, 1f-4)
        c_alpha_y = max(c_alpha_y, 1f-4)

        D = trowbridge_reitz_d(wh, c_alpha_x, c_alpha_y)
        G = trowbridge_reitz_g(wo_local, wi_local, c_alpha_x, c_alpha_y)
        F_conductor = fr_complex_spectral(abs(cos_θo_h), ce_spectral, ck_spectral)

        f_conductor = D * F_conductor * G / (4f0 * abs(wi_local[3]) * abs(wo_local[3]))

        layer_tr = if has_medium
            tr = layer_transmittance(thickness, wi_local)
            tr * tr * albedo_spectral
        else
            SpectralRadiance(1f0)
        end

        f_spectral = f_conductor * T_o * T_i * layer_tr

        pdf_m = trowbridge_reitz_pdf(wo_local, wh, c_alpha_x, c_alpha_y)
        pdf = T_o * pdf_m / (4f0 * abs(cos_θo_h))

        return (f_spectral, pdf)
    else
        # Rough interface (and possibly rough conductor)
        i_alpha_x = max(i_alpha_x, 1f-4)
        i_alpha_y = max(i_alpha_y, 1f-4)

        # Interface specular contribution
        D_i = trowbridge_reitz_d(wh, i_alpha_x, i_alpha_y)
        G_i = trowbridge_reitz_g(wo_local, wi_local, i_alpha_x, i_alpha_y)
        f_interface = D_i * F_interface_wh * G_i / (4f0 * abs(wi_local[3]) * abs(wo_local[3]))

        # Conductor contribution
        T_o = 1f0 - F_interface_o
        T_i = 1f0 - F_interface_i

        local f_conductor::SpectralRadiance
        local pdf_conductor::Float32

        if c_is_smooth
            # Smooth conductor under rough interface
            F_conductor = fr_complex_spectral(abs(wo_local[3]), ce_spectral, ck_spectral)
            f_conductor = F_conductor / abs(wo_local[3])
            pdf_conductor = 1f0
        else
            c_alpha_x = max(c_alpha_x, 1f-4)
            c_alpha_y = max(c_alpha_y, 1f-4)

            D_c = trowbridge_reitz_d(wh, c_alpha_x, c_alpha_y)
            G_c = trowbridge_reitz_g(wo_local, wi_local, c_alpha_x, c_alpha_y)
            F_conductor = fr_complex_spectral(abs(cos_θo_h), ce_spectral, ck_spectral)

            f_conductor = D_c * F_conductor * G_c / (4f0 * abs(wi_local[3]) * abs(wo_local[3]))
            pdf_m_c = trowbridge_reitz_pdf(wo_local, wh, c_alpha_x, c_alpha_y)
            pdf_conductor = pdf_m_c / (4f0 * abs(cos_θo_h))
        end

        layer_tr = if has_medium
            tr = layer_transmittance(thickness, wi_local)
            tr * tr * albedo_spectral
        else
            SpectralRadiance(1f0)
        end

        f_conductor_contrib = f_conductor * T_o * T_i * layer_tr

        # Combined BSDF
        f_spectral = SpectralRadiance(f_interface) + f_conductor_contrib

        # Combined PDF
        pdf_m_i = trowbridge_reitz_pdf(wo_local, wh, i_alpha_x, i_alpha_y)
        pdf_interface = F_interface_o * pdf_m_i / (4f0 * abs(cos_θo_h))
        pdf = pdf_interface + T_o * pdf_conductor

        return (f_spectral, pdf)
    end
end

"""
    get_emission_spectral for CoatedConductorMaterial - returns zero (non-emissive).
"""
@propagate_inbounds function get_emission_spectral(
    table::RGBToSpectrumTable, mat::CoatedConductorMaterial, textures,
    wo::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
)
    return SpectralRadiance()
end

"""
    get_albedo_spectral for CoatedConductorMaterial.
"""
@propagate_inbounds function get_albedo_spectral(table::RGBToSpectrumTable, mat::CoatedConductorMaterial, textures, uv::Point2f, lambda::Wavelengths)
    if mat.use_eta_k
        # For eta/k mode, compute approximate reflectance
        ce_rgb = eval_tex(textures, mat.conductor_eta, uv)
        ck_rgb = eval_tex(textures, mat.conductor_k, uv)
        # Approximate normal incidence reflectance: ((n-1)² + k²) / ((n+1)² + k²)
        n = ce_rgb.c[1]
        k = ck_rgb.c[1]
        r = ((n - 1f0)^2 + k^2) / ((n + 1f0)^2 + k^2)
        return SpectralRadiance(r)
    else
        return uplift_rgb(table, eval_tex(textures, mat.reflectance, uv), lambda)
    end
end

# ============================================================================
# MediumInterface Forwarding
# ============================================================================

# MediumInterface is defined in integrators/volpath/media.jl
# These forwarding functions delegate BSDF operations to the wrapped material

"""
    sample_bsdf_spectral for MediumInterface - forwards to wrapped material.
"""
@propagate_inbounds function sample_bsdf_spectral(
    table::RGBToSpectrumTable, mi::MediumInterface, textures,
    wo::Vec3f, n::Vec3f, uv::Point2f,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32,
    regularize::Bool = false
)
    return sample_bsdf_spectral(table, mi.material, textures, wo, n, uv, lambda, sample_u, rng, regularize)
end

"""
    evaluate_bsdf_spectral for MediumInterface - forwards to wrapped material.
"""
@propagate_inbounds function evaluate_bsdf_spectral(
    table::RGBToSpectrumTable, mi::MediumInterface, textures,
    wo::Vec3f, wi::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
)
    return evaluate_bsdf_spectral(table, mi.material, textures, wo, wi, n, uv, lambda)
end

"""
    get_emission_spectral for MediumInterface - forwards to wrapped material.
"""
@propagate_inbounds function get_emission_spectral(
    table::RGBToSpectrumTable, mi::MediumInterface, textures,
    wo::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
)
    return get_emission_spectral(table, mi.material, textures, wo, n, uv, lambda)
end

"""
    is_emissive for MediumInterface - forwards to wrapped material.
"""
@propagate_inbounds function is_emissive(mi::MediumInterface)
    return is_emissive(mi.material)
end

"""
    get_albedo_spectral for MediumInterface - forwards to wrapped material.
"""
@propagate_inbounds function get_albedo_spectral(table::RGBToSpectrumTable, mi::MediumInterface, textures, uv::Point2f, lambda::Wavelengths)
    return get_albedo_spectral(table, mi.material, textures, uv, lambda)
end

# ============================================================================
# MediumInterfaceIdx forwarding functions
# After scene building, MediumInterface is converted to MediumInterfaceIdx
# These forwarding functions delegate BSDF operations to the wrapped material
# ============================================================================

"""
    sample_bsdf_spectral for MediumInterfaceIdx - forwards to wrapped material.
"""
@propagate_inbounds function sample_bsdf_spectral(
    table::RGBToSpectrumTable, mi::MediumInterfaceIdx, textures,
    wo::Vec3f, n::Vec3f, uv::Point2f,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32,
    regularize::Bool = false
)
    return sample_bsdf_spectral(table, mi.material, textures, wo, n, uv, lambda, sample_u, rng, regularize)
end

"""
    evaluate_bsdf_spectral for MediumInterfaceIdx - forwards to wrapped material.
"""
@propagate_inbounds function evaluate_bsdf_spectral(
    table::RGBToSpectrumTable, mi::MediumInterfaceIdx, textures,
    wo::Vec3f, wi::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
)
    return evaluate_bsdf_spectral(table, mi.material, textures, wo, wi, n, uv, lambda)
end

"""
    get_emission_spectral for MediumInterfaceIdx - forwards to wrapped material.
"""
@propagate_inbounds function get_emission_spectral(
    table::RGBToSpectrumTable, mi::MediumInterfaceIdx, textures,
    wo::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
)
    return get_emission_spectral(table, mi.material, textures, wo, n, uv, lambda)
end

"""
    is_emissive for MediumInterfaceIdx - forwards to wrapped material.
"""
@propagate_inbounds function is_emissive(mi::MediumInterfaceIdx)
    return is_emissive(mi.material)
end

"""
    get_albedo_spectral for MediumInterfaceIdx - forwards to wrapped material.
"""
@propagate_inbounds function get_albedo_spectral(table::RGBToSpectrumTable, mi::MediumInterfaceIdx, textures, uv::Point2f, lambda::Wavelengths)
    return get_albedo_spectral(table, mi.material, textures, uv, lambda)
end

# ============================================================================
# Helper Functions
# ============================================================================

# cosine_sample_hemisphere is defined in Hikari.jl

"""
    coordinate_system(n::Vec3f) -> (tangent, bitangent)

Build orthonormal basis from a normal vector.
"""
@propagate_inbounds function coordinate_system(n::Vec3f)
    if abs(n[1]) > abs(n[2])
        inv_len = 1f0 / sqrt(n[1] * n[1] + n[3] * n[3])
        tangent = Vec3f(n[3] * inv_len, 0f0, -n[1] * inv_len)
    else
        inv_len = 1f0 / sqrt(n[2] * n[2] + n[3] * n[3])
        tangent = Vec3f(0f0, n[3] * inv_len, -n[2] * inv_len)
    end
    bitangent = cross(n, tangent)
    return (tangent, bitangent)
end

"""
    local_to_world(local_dir, n, tangent, bitangent) -> Vec3f

Transform direction from local (shading) space to world space.
"""
@propagate_inbounds function local_to_world(local_dir::Vec3f, n::Vec3f, tangent::Vec3f, bitangent::Vec3f)::Vec3f
    return tangent * local_dir[1] + bitangent * local_dir[2] + n * local_dir[3]
end

# Note: reflect is defined earlier in this file
# Note: fresnel_dielectric is defined in reflection/bxdf.jl - do not duplicate here

"""
    sample_ggx_vndf(wo, alpha_x, alpha_y, u) -> Vec3f

Sample visible normal from GGX distribution.
"""
@propagate_inbounds function sample_ggx_vndf(wo::Vec3f, alpha_x::Float32, alpha_y::Float32, u::Point2f)::Vec3f
    # Transform to hemisphere configuration
    wh = normalize(Vec3f(alpha_x * wo[1], alpha_y * wo[2], wo[3]))

    if wh[3] < 0f0
        wh = -wh
    end

    # Sample projected area
    t1 = wh[3] < 0.9999f0 ? normalize(cross(Vec3f(0, 0, 1), wh)) : Vec3f(1, 0, 0)
    t2 = cross(wh, t1)

    r = sqrt(u[1])
    phi = 2f0 * Float32(π) * u[2]
    p1 = r * cos(phi)
    p2 = r * sin(phi)
    s = 0.5f0 * (1f0 + wh[3])
    p2 = (1f0 - s) * sqrt(max(0f0, 1f0 - p1 * p1)) + s * p2

    # Compute normal
    n = p1 * t1 + p2 * t2 + sqrt(max(0f0, 1f0 - p1 * p1 - p2 * p2)) * wh

    # Transform back
    return normalize(Vec3f(alpha_x * n[1], alpha_y * n[2], max(1f-6, n[3])))
end

# ============================================================================
# pbrt-v4 Compatible Functions for ConductorBxDF
# ============================================================================

"""
    world_to_local(v, n, tangent, bitangent) -> Vec3f

Transform direction from world space to local (shading) space.
In local space, the normal is (0, 0, 1).
"""
@propagate_inbounds function world_to_local(v::Vec3f, n::Vec3f, tangent::Vec3f, bitangent::Vec3f)::Vec3f
    return Vec3f(dot(v, tangent), dot(v, bitangent), dot(v, n))
end

"""
    cos_theta(w) -> Float32

Get cos(θ) of a direction in local coordinates (matches pbrt-v4's CosTheta).
In local space, z is the surface normal.
"""
@inline cos_theta(w::Vec3f)::Float32 = w[3]

"""
    cos2_theta(w) -> Float32

Get cos²(θ) of a direction in local coordinates.
"""
@inline cos2_theta(w::Vec3f)::Float32 = w[3] * w[3]

"""
    abs_cos_theta(w) -> Float32

Get |cos(θ)| of a direction in local coordinates (matches pbrt-v4's AbsCosTheta).
"""
@inline abs_cos_theta(w::Vec3f)::Float32 = abs(w[3])

"""
    sin2_theta(w) -> Float32

Get sin²(θ) of a direction in local coordinates.
"""
@inline sin2_theta(w::Vec3f)::Float32 = max(0f0, 1f0 - cos2_theta(w))

"""
    sin_theta(w) -> Float32

Get sin(θ) of a direction in local coordinates.
"""
@inline sin_theta(w::Vec3f)::Float32 = sqrt(sin2_theta(w))

"""
    tan2_theta(w) -> Float32

Get tan²(θ) of a direction in local coordinates.
"""
@inline function tan2_theta(w::Vec3f)::Float32
    c2 = cos2_theta(w)
    return sin2_theta(w) / c2
end

"""
    cos_phi(w) -> Float32

Get cos(φ) of a direction in local coordinates (matches pbrt-v4's CosPhi).
"""
@inline function cos_phi(w::Vec3f)::Float32
    sin_θ = sin_theta(w)
    return sin_θ == 0f0 ? 1f0 : clamp(w[1] / sin_θ, -1f0, 1f0)
end

"""
    sin_phi(w) -> Float32

Get sin(φ) of a direction in local coordinates (matches pbrt-v4's SinPhi).
"""
@inline function sin_phi(w::Vec3f)::Float32
    sin_θ = sin_theta(w)
    return sin_θ == 0f0 ? 0f0 : clamp(w[2] / sin_θ, -1f0, 1f0)
end

# Note: same_hemisphere is defined earlier in this file

"""
    fr_complex(cos_theta_i, eta, k) -> Float32

Compute Fresnel reflectance for a conductor using complex IOR (matches pbrt-v4's FrComplex).

Arguments:
- `cos_theta_i`: Cosine of incident angle (clamped to [0, 1])
- `eta`: Real part of complex IOR (n)
- `k`: Imaginary part of complex IOR (extinction coefficient)

This uses the exact same formula as pbrt-v4 with complex arithmetic.
"""
@propagate_inbounds function fr_complex(cos_theta_i::Float32, eta::Float32, k::Float32)::Float32
    cos_theta_i = clamp(cos_theta_i, 0f0, 1f0)

    # Compute complex cos(θt) for Fresnel equations using Snell's law
    sin2_theta_i = 1f0 - cos_theta_i * cos_theta_i

    # Complex eta: eta_c = eta + i*k
    # sin²θt = sin²θi / eta_c²
    # For complex division: (a+bi)² = a² - b² + 2abi
    # eta_c² = eta² - k² + 2*eta*k*i
    eta2 = eta * eta
    k2 = k * k
    eta_c2_re = eta2 - k2
    eta_c2_im = 2f0 * eta * k

    # sin²θt (complex) = sin²θi / eta_c²
    # For complex division: (a) / (c + di) = a*(c - di) / (c² + d²)
    denom = eta_c2_re * eta_c2_re + eta_c2_im * eta_c2_im
    sin2_theta_t_re = sin2_theta_i * eta_c2_re / denom
    sin2_theta_t_im = -sin2_theta_i * eta_c2_im / denom

    # cos²θt (complex) = 1 - sin²θt
    cos2_theta_t_re = 1f0 - sin2_theta_t_re
    cos2_theta_t_im = -sin2_theta_t_im

    # cosθt (complex) = sqrt(cos²θt)
    # For complex sqrt: sqrt(a + bi) where result has positive real part
    mag = sqrt(cos2_theta_t_re * cos2_theta_t_re + cos2_theta_t_im * cos2_theta_t_im)
    cos_theta_t_re = sqrt(0.5f0 * (mag + cos2_theta_t_re))
    cos_theta_t_im = cos2_theta_t_im / (2f0 * cos_theta_t_re)
    # Handle edge case where cos_theta_t_re would be 0
    if cos_theta_t_re == 0f0
        cos_theta_t_im = sqrt(0.5f0 * mag)
    end

    # r_parl = (eta_c * cos_theta_i - cos_theta_t) / (eta_c * cos_theta_i + cos_theta_t)
    # eta_c * cos_theta_i = (eta + i*k) * cos_theta_i = eta*cos_i + i*k*cos_i
    eta_cos_i_re = eta * cos_theta_i
    eta_cos_i_im = k * cos_theta_i

    # numerator: eta_c * cos_theta_i - cos_theta_t
    num_parl_re = eta_cos_i_re - cos_theta_t_re
    num_parl_im = eta_cos_i_im - cos_theta_t_im
    # denominator: eta_c * cos_theta_i + cos_theta_t
    den_parl_re = eta_cos_i_re + cos_theta_t_re
    den_parl_im = eta_cos_i_im + cos_theta_t_im

    # r_parl = num / den (complex division)
    den_parl_mag2 = den_parl_re * den_parl_re + den_parl_im * den_parl_im
    r_parl_re = (num_parl_re * den_parl_re + num_parl_im * den_parl_im) / den_parl_mag2
    r_parl_im = (num_parl_im * den_parl_re - num_parl_re * den_parl_im) / den_parl_mag2

    # r_perp = (cos_theta_i - eta_c * cos_theta_t) / (cos_theta_i + eta_c * cos_theta_t)
    # eta_c * cos_theta_t = (eta + i*k) * (cos_t_re + i*cos_t_im)
    #                     = eta*cos_t_re - k*cos_t_im + i*(eta*cos_t_im + k*cos_t_re)
    eta_cos_t_re = eta * cos_theta_t_re - k * cos_theta_t_im
    eta_cos_t_im = eta * cos_theta_t_im + k * cos_theta_t_re

    # numerator: cos_theta_i - eta_c * cos_theta_t (cos_theta_i is real)
    num_perp_re = cos_theta_i - eta_cos_t_re
    num_perp_im = -eta_cos_t_im
    # denominator: cos_theta_i + eta_c * cos_theta_t
    den_perp_re = cos_theta_i + eta_cos_t_re
    den_perp_im = eta_cos_t_im

    # r_perp = num / den (complex division)
    den_perp_mag2 = den_perp_re * den_perp_re + den_perp_im * den_perp_im
    r_perp_re = (num_perp_re * den_perp_re + num_perp_im * den_perp_im) / den_perp_mag2
    r_perp_im = (num_perp_im * den_perp_re - num_perp_re * den_perp_im) / den_perp_mag2

    # Return (|r_parl|² + |r_perp|²) / 2
    # |r|² = r_re² + r_im² (norm of complex number)
    norm_parl = r_parl_re * r_parl_re + r_parl_im * r_parl_im
    norm_perp = r_perp_re * r_perp_re + r_perp_im * r_perp_im

    return (norm_parl + norm_perp) * 0.5f0
end

"""
    fr_complex_spectral(cos_theta_i, eta, k) -> SpectralRadiance

Compute spectral Fresnel reflectance for a conductor (matches pbrt-v4's FrComplex for SampledSpectrum).
Evaluates fr_complex for each wavelength channel.
"""
@propagate_inbounds function fr_complex_spectral(cos_theta_i::Float32, eta::SpectralRadiance, k::SpectralRadiance)::SpectralRadiance
    return SpectralRadiance(
        fr_complex(cos_theta_i, eta[1], k[1]),
        fr_complex(cos_theta_i, eta[2], k[2]),
        fr_complex(cos_theta_i, eta[3], k[3]),
        fr_complex(cos_theta_i, eta[4], k[4])
    )
end

# ============================================================================
# TrowbridgeReitz Distribution Functions (matching pbrt-v4 exactly)
# ============================================================================

"""
    trowbridge_reitz_effectively_smooth(alpha_x, alpha_y) -> Bool

Check if the distribution is effectively smooth (matches pbrt-v4's EffectivelySmooth).
"""
@inline trowbridge_reitz_effectively_smooth(alpha_x::Float32, alpha_y::Float32)::Bool =
    max(alpha_x, alpha_y) < 1f-3

"""
    trowbridge_reitz_d(wm, alpha_x, alpha_y) -> Float32

Evaluate the TrowbridgeReitz D (normal distribution function) at microfacet normal wm.
Matches pbrt-v4's TrowbridgeReitzDistribution::D(wm).
"""
@propagate_inbounds function trowbridge_reitz_d(wm::Vec3f, alpha_x::Float32, alpha_y::Float32)::Float32
    tan2_θ = tan2_theta(wm)
    isinf(tan2_θ) && return 0f0

    cos4_θ = cos2_theta(wm) * cos2_theta(wm)
    cos4_θ < 1f-16 && return 0f0

    e = tan2_θ * ((cos_phi(wm) / alpha_x)^2 + (sin_phi(wm) / alpha_y)^2)
    return 1f0 / (Float32(π) * alpha_x * alpha_y * cos4_θ * (1f0 + e)^2)
end

"""
    trowbridge_reitz_lambda(w, alpha_x, alpha_y) -> Float32

Compute Lambda(w) for Smith masking-shadowing (matches pbrt-v4's Lambda).
"""
@propagate_inbounds function trowbridge_reitz_lambda(w::Vec3f, alpha_x::Float32, alpha_y::Float32)::Float32
    tan2_θ = tan2_theta(w)
    isinf(tan2_θ) && return 0f0

    alpha2 = (cos_phi(w) * alpha_x)^2 + (sin_phi(w) * alpha_y)^2
    return (sqrt(1f0 + alpha2 * tan2_θ) - 1f0) * 0.5f0
end

"""
    trowbridge_reitz_g1(w, alpha_x, alpha_y) -> Float32

Compute G1(w) Smith masking function (matches pbrt-v4's G1).
"""
@inline trowbridge_reitz_g1(w::Vec3f, alpha_x::Float32, alpha_y::Float32)::Float32 =
    1f0 / (1f0 + trowbridge_reitz_lambda(w, alpha_x, alpha_y))

"""
    trowbridge_reitz_g(wo, wi, alpha_x, alpha_y) -> Float32

Compute G(wo, wi) Smith masking-shadowing function (matches pbrt-v4's G).
"""
@inline trowbridge_reitz_g(wo::Vec3f, wi::Vec3f, alpha_x::Float32, alpha_y::Float32)::Float32 =
    1f0 / (1f0 + trowbridge_reitz_lambda(wo, alpha_x, alpha_y) + trowbridge_reitz_lambda(wi, alpha_x, alpha_y))

"""
    trowbridge_reitz_d_pdf(w, wm, alpha_x, alpha_y) -> Float32

Evaluate the visible normal distribution D(w, wm) for PDF computation.
Matches pbrt-v4's D(w, wm) = G1(w) / AbsCosTheta(w) * D(wm) * AbsDot(w, wm).
"""
@propagate_inbounds function trowbridge_reitz_d_pdf(w::Vec3f, wm::Vec3f, alpha_x::Float32, alpha_y::Float32)::Float32
    return trowbridge_reitz_g1(w, alpha_x, alpha_y) / abs_cos_theta(w) *
           trowbridge_reitz_d(wm, alpha_x, alpha_y) * abs(dot(w, wm))
end

"""
    trowbridge_reitz_pdf(w, wm, alpha_x, alpha_y) -> Float32

Compute PDF for visible normal sampling (matches pbrt-v4's PDF).
"""
@inline trowbridge_reitz_pdf(w::Vec3f, wm::Vec3f, alpha_x::Float32, alpha_y::Float32)::Float32 =
    trowbridge_reitz_d_pdf(w, wm, alpha_x, alpha_y)

"""
    trowbridge_reitz_sample_wm(w, u, alpha_x, alpha_y) -> Vec3f

Sample visible normal from TrowbridgeReitz distribution (matches pbrt-v4's Sample_wm).
"""
@propagate_inbounds function trowbridge_reitz_sample_wm(w::Vec3f, u::Point2f, alpha_x::Float32, alpha_y::Float32)::Vec3f
    # Transform w to hemispherical configuration
    wh = normalize(Vec3f(alpha_x * w[1], alpha_y * w[2], w[3]))
    if wh[3] < 0f0
        wh = -wh
    end

    # Find orthonormal basis for visible normal sampling
    t1 = wh[3] < 0.99999f0 ? normalize(cross(Vec3f(0f0, 0f0, 1f0), wh)) : Vec3f(1f0, 0f0, 0f0)
    t2 = cross(wh, t1)

    # Generate uniformly distributed points on the unit disk (polar sampling)
    r = sqrt(u[1])
    phi = 2f0 * Float32(π) * u[2]
    p_x = r * cos(phi)
    p_y = r * sin(phi)

    # Warp hemispherical projection for visible normal sampling
    h = sqrt(1f0 - p_x * p_x)
    p_y = lerp(h, p_y, 0.5f0 * (1f0 + wh[3]))

    # Reproject to hemisphere and transform normal to ellipsoid configuration
    pz = sqrt(max(0f0, 1f0 - p_x * p_x - p_y * p_y))
    nh = p_x * t1 + p_y * t2 + pz * wh

    return normalize(Vec3f(alpha_x * nh[1], alpha_y * nh[2], max(1f-6, nh[3])))
end

# Note: lerp is defined in spectrum.jl - do not duplicate here

"""
    face_forward(v, n) -> Vec3f

Flip v to be in the same hemisphere as n (matches pbrt-v4's FaceForward).
"""
@inline face_forward(v::Vec3f, n::Vec3f)::Vec3f = dot(v, n) < 0f0 ? -v : v
