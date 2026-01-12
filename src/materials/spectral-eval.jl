# Spectral BSDF Evaluation Interface for PhysicalWavefront
# Enables spectral path tracing while keeping materials as RGB containers
#
# NOTE: All functions take a `textures` parameter for GPU compatibility.
# On CPU, textures is ignored (Texture structs contain their data).
# On GPU, textures is a tuple of CLDeviceArrays, and materials contain TextureRef.

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
    lambda::Wavelengths, sample_u::Point2f, rng::Float32
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
    lambda::Wavelengths, sample_u::Point2f, rng::Float32
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
    lambda::Wavelengths, sample_u::Point2f, rng::Float32
)
    # Get material properties
    kr_rgb = eval_tex(textures, mat.Kr, uv)
    kt_rgb = eval_tex(textures, mat.Kt, uv)
    ior = eval_tex(textures, mat.index, uv)

    kr_spectral = uplift_rgb(table, kr_rgb, lambda)
    kt_spectral = uplift_rgb(table, kt_rgb, lambda)

    # Determine if entering or exiting
    cos_theta_o = dot(wo, n)
    entering = cos_theta_o > 0f0

    n_oriented = entering ? n : -n
    cos_theta_o = abs(cos_theta_o)

    # Compute eta ratio
    eta = entering ? (1f0 / ior) : ior

    # Compute Fresnel reflectance
    F = fresnel_dielectric(cos_theta_o, entering ? ior : (1f0 / ior))

    # Choose reflection or refraction based on Fresnel
    if rng < F
        # Reflection
        wi = reflect(wo, n_oriented)
        # f = F * Kr, probability = F, result = Kr
        return SpectralBSDFSample(wi, kr_spectral, 1f0, true, 1f0)
    else
        # Refraction
        sin2_theta_i = max(0f0, 1f0 - cos_theta_o * cos_theta_o)
        sin2_theta_t = eta * eta * sin2_theta_i

        if sin2_theta_t >= 1f0
            # Total internal reflection
            wi = reflect(wo, n_oriented)
            return SpectralBSDFSample(wi, kr_spectral, 1f0, true, 1f0)
        end

        cos_theta_t = sqrt(1f0 - sin2_theta_t)
        wi = normalize(eta * (-wo) + (eta * cos_theta_o - cos_theta_t) * n_oriented)

        # f = (1-F) * Kt, probability = (1-F), result = Kt
        # Include eta^2 correction for radiance (non-symmetric due to refraction)
        eta_scale = eta * eta
        return SpectralBSDFSample(wi, kt_spectral, 1f0, true, eta_scale)
    end
end

"""
    sample_bsdf_spectral(table, mat::PlasticMaterial, textures, wo, n, uv, lambda, sample_u, rng) -> SpectralBSDFSample

Sample plastic BSDF (diffuse + glossy specular).
"""
@propagate_inbounds function sample_bsdf_spectral(
    table::RGBToSpectrumTable, mat::PlasticMaterial, textures,
    wo::Vec3f, n::Vec3f, uv::Point2f,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32
)
    # Check for grazing angle
    wo_dot_n = dot(wo, n)
    if abs(wo_dot_n) < 1f-6
        return SpectralBSDFSample()
    end

    # Get material properties
    kd_rgb = eval_tex(textures, mat.Kd, uv)
    ks_rgb = eval_tex(textures, mat.Ks, uv)
    roughness = eval_tex(textures, mat.roughness, uv)

    kd_spectral = uplift_rgb(table, kd_rgb, lambda)
    ks_spectral = uplift_rgb(table, ks_rgb, lambda)

    # Compute diffuse and specular weights for sampling
    kd_lum = average(kd_spectral)
    ks_lum = average(ks_spectral)
    total_lum = kd_lum + ks_lum

    if total_lum < 1f-6
        return SpectralBSDFSample()
    end

    # Probability of sampling diffuse vs specular
    p_diffuse = kd_lum / total_lum

    # Build coordinate system from shading normal
    tangent, bitangent = coordinate_system(n)

    if rng < p_diffuse
        # Sample diffuse component
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

        # Combined BSDF: Kd/π + specular term
        # For simplicity, just return diffuse contribution scaled by 1/p_diffuse
        f = kd_spectral * (1f0 / (Float32(π) * p_diffuse))
        pdf = p_diffuse * cos_theta / Float32(π)

        return SpectralBSDFSample(wi, f, pdf, false, 1f0)
    else
        # Sample specular component - orient normal to face wo
        n_oriented = wo_dot_n < 0f0 ? -n : n
        wi = reflect(wo, n_oriented)

        if roughness > 0.001f0
            # Add roughness perturbation
            offset = (Vec3f(sample_u[1], sample_u[2], 0f0) * 2f0 .- 1f0) * roughness
            wi = normalize(wi + offset)
        end

        # Fresnel for plastic (IOR ≈ 1.5)
        cos_theta_i = abs(dot(wi, n_oriented))
        F = fresnel_dielectric(cos_theta_i, 1.5f0)

        f = ks_spectral * F * (1f0 / (1f0 - p_diffuse))
        pdf = 1f0 - p_diffuse

        return SpectralBSDFSample(wi, f, pdf, roughness < 0.001f0, 1f0)
    end
end

"""
    sample_bsdf_spectral(table, mat::MetalMaterial, textures, wo, n, uv, lambda, sample_u, rng) -> SpectralBSDFSample

Sample metal BSDF with conductor Fresnel.
"""
@propagate_inbounds function sample_bsdf_spectral(
    table::RGBToSpectrumTable, mat::MetalMaterial, textures,
    wo::Vec3f, n::Vec3f, uv::Point2f,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32
)
    # Check for grazing angle
    wo_dot_n = dot(wo, n)
    if abs(wo_dot_n) < 1f-6
        return SpectralBSDFSample()
    end

    # Get material properties (rng unused)
    eta_rgb = eval_tex(textures, mat.eta, uv)
    k_rgb = eval_tex(textures, mat.k, uv)
    roughness = eval_tex(textures, mat.roughness, uv)
    reflectance_rgb = eval_tex(textures, mat.reflectance, uv)

    reflectance_spectral = uplift_rgb(table, reflectance_rgb, lambda)

    # Orient normal to face wo for reflection
    n_oriented = wo_dot_n < 0f0 ? -n : n

    # For smooth metals, use perfect reflection
    if roughness < 0.01f0
        wi = reflect(wo, n_oriented)
        cos_theta = abs(wo_dot_n)

        # Fresnel for conductor (using RGB approximation for now)
        # TODO: Full spectral Fresnel for metals
        F_rgb = fresnel_conductor(cos_theta, RGBSpectrum(1f0), eta_rgb, k_rgb)
        F_spectral = uplift_rgb(table, F_rgb, lambda)

        f = reflectance_spectral * F_spectral
        return SpectralBSDFSample(wi, f, 1f0, true, 1f0)
    else
        # Rough metal: use GGX microfacet sampling
        # Simplified: sample visible normal distribution
        alpha = mat.remap_roughness ? roughness_to_α(roughness) : roughness

        # Sample microfacet normal
        wm = sample_ggx_vndf(wo, alpha, alpha, sample_u)
        wi = reflect(wo, wm)

        # Reject samples that go through the surface
        if dot(wi, n_oriented) < 0f0
            return SpectralBSDFSample()
        end

        cos_theta = abs(wo_dot_n)
        F_rgb = fresnel_conductor(cos_theta, RGBSpectrum(1f0), eta_rgb, k_rgb)
        F_spectral = uplift_rgb(table, F_rgb, lambda)

        # Compute PDF and BSDF value
        # Simplified: approximate as specular for now
        f = reflectance_spectral * F_spectral

        return SpectralBSDFSample(wi, f, 1f0, false, 1f0)
    end
end

"""
    sample_bsdf_spectral(table, mat::EmissiveMaterial, textures, wo, n, uv, lambda, sample_u, rng) -> SpectralBSDFSample

Emissive materials don't scatter - return invalid sample.
"""
@propagate_inbounds function sample_bsdf_spectral(
    table::RGBToSpectrumTable, mat::EmissiveMaterial, textures,
    wo::Vec3f, n::Vec3f, uv::Point2f,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32
)
    return SpectralBSDFSample()
end

# Fallback for unknown materials
@propagate_inbounds function sample_bsdf_spectral(
    table::RGBToSpectrumTable, mat::Material, textures,
    wo::Vec3f, n::Vec3f, uv::Point2f,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32
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

@propagate_inbounds function evaluate_bsdf_spectral(
    table::RGBToSpectrumTable, mat::PlasticMaterial, textures,
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

    # Diffuse component (simplified - ignoring specular for MIS evaluation)
    kd_rgb = eval_tex(textures, mat.Kd, uv)
    kd_spectral = uplift_rgb(table, kd_rgb, lambda)
    f = kd_spectral * (1f0 / Float32(π))
    pdf = cos_theta / Float32(π)

    return (f, pdf)
end

@propagate_inbounds function evaluate_bsdf_spectral(
    table::RGBToSpectrumTable, mat::MetalMaterial, textures,
    wo::Vec3f, wi::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
)
    # For rough metals, we'd need microfacet evaluation
    # For now, treat as specular (zero contribution)
    return (SpectralRadiance(), 0f0)
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

@propagate_inbounds function get_albedo_spectral(table::RGBToSpectrumTable, mat::PlasticMaterial, textures, uv::Point2f, lambda::Wavelengths)
    return uplift_rgb(table, eval_tex(textures, mat.Kd, uv), lambda)
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

    # Compute transmitted direction using Snell's law
    entering = wo[3] > 0f0
    η_ratio = entering ? (1f0 / eta) : eta

    sin2_θi = max(0f0, 1f0 - cos_θo^2)
    sin2_θt = η_ratio^2 * sin2_θi

    # Check for total internal reflection
    if sin2_θt >= 1f0
        return (Vec3f(0), 0f0, false)
    end

    cos_θt = sqrt(1f0 - sin2_θt)
    # Flip sign based on which side we're entering
    cos_θt = entering ? -cos_θt : cos_θt

    wi = Vec3f(-η_ratio * wo[1], -η_ratio * wo[2], cos_θt)
    wi = normalize(wi)

    T = 1f0 - F
    return (wi, T, true)
end

"""
    sample_bsdf_spectral(table, mat::CoatedDiffuseMaterial, textures, wo, n, uv, lambda, sample_u, rng) -> SpectralBSDFSample

Sample CoatedDiffuse BSDF using pbrt-v4's LayeredBxDF random walk algorithm.
This performs a Monte Carlo estimate of the layered material response.
"""
@propagate_inbounds function sample_bsdf_spectral(
    table::RGBToSpectrumTable, mat::CoatedDiffuseMaterial, textures,
    wo::Vec3f, n::Vec3f, uv::Point2f,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32
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

    refl_spectral = uplift_rgb(table, refl_rgb, lambda)
    albedo_spectral = uplift_rgb(table, albedo_rgb, lambda)
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

    # === LayeredBxDF sampling strategy ===
    # For smooth coating (roughness ≈ 0), we use a simplified approach:
    # 1. Compute Fresnel at top interface
    # 2. Probabilistically choose reflection or transmission
    # 3. If transmission, sample diffuse base and transmit back out

    cos_θo = abs(wo_local[3])
    F = fresnel_dielectric(cos_θo, eta)

    # Use rng to decide: reflect at top interface or transmit through
    if rng < F
        # Specular reflection at coating surface
        wi_local = Vec3f(-wo_local[1], -wo_local[2], wo_local[3])
        if flip
            wi_local = -wi_local
        end

        # Transform back to world space
        wi = tangent * wi_local[1] + bitangent * wi_local[2] + n * wi_local[3]
        wi = normalize(wi)

        # For specular reflection, f = 1 (Fresnel already accounted in probability)
        f_spectral = SpectralRadiance(1f0)
        return SpectralBSDFSample(wi, f_spectral, 1f0, true, 1f0)
    end

    # Transmitted through top interface - sample the diffuse base
    # Compute refracted direction into the coating
    entering = true  # wo_local[3] > 0 after flip
    η_into = 1f0 / eta

    sin2_θi = max(0f0, 1f0 - cos_θo^2)
    sin2_θt = η_into^2 * sin2_θi

    if sin2_θt >= 1f0
        # Total internal reflection (shouldn't happen for typical coating IOR)
        return SpectralBSDFSample()
    end

    cos_θt_in = sqrt(1f0 - sin2_θt)

    # Sample diffuse base layer with cosine-weighted hemisphere
    local_diffuse_wi = cosine_sample_hemisphere(sample_u)
    cos_θ_diffuse = local_diffuse_wi[3]

    if cos_θ_diffuse < 1f-6
        return SpectralBSDFSample()
    end

    # Compute Fresnel for light exiting the coating (from inside to outside)
    # Note: when light travels from medium with IOR=eta to air (IOR=1), use eta as the ratio
    F_out = fresnel_dielectric(cos_θ_diffuse, eta)
    T_in = 1f0 - F
    T_out = 1f0 - F_out

    # Account for layer transmittance if there's absorption
    layer_tr = if has_medium
        # Simple absorption: Tr = exp(-thickness / |cos_θ|)
        # Average over entry and exit paths
        tr_in = layer_transmittance(thickness, Vec3f(0, 0, cos_θt_in))
        tr_out = layer_transmittance(thickness, local_diffuse_wi)
        tr_in * tr_out * albedo_spectral
    else
        SpectralRadiance(1f0)
    end

    # Transform diffuse sample to world space (it's in local shading frame)
    if flip
        local_diffuse_wi = Vec3f(local_diffuse_wi[1], local_diffuse_wi[2], -local_diffuse_wi[3])
    end

    # Refract the diffuse direction back out through the coating
    # Simplified: for smooth coating, use the diffuse direction directly
    # (accurate for thin coatings where refraction deviation is small)
    wi = tangent * local_diffuse_wi[1] + bitangent * local_diffuse_wi[2] + n * local_diffuse_wi[3]
    wi = normalize(wi)

    # Total BSDF value:
    # f = T_in * (Kd/π) * T_out * layer_transmittance
    # pdf = (1-F) * cos_θ/π
    diffuse_f = refl_spectral * (1f0 / Float32(π))
    f_spectral = diffuse_f * T_in * T_out * layer_tr

    pdf = (1f0 - F) * cos_θ_diffuse / Float32(π)

    return SpectralBSDFSample(wi, f_spectral, pdf, false, 1f0)
end

"""
    evaluate_bsdf_spectral(table, mat::CoatedDiffuseMaterial, textures, wo, wi, n, uv, lambda) -> (f, pdf)

Evaluate CoatedDiffuse BSDF for given directions.
"""
@propagate_inbounds function evaluate_bsdf_spectral(
    table::RGBToSpectrumTable, mat::CoatedDiffuseMaterial, textures,
    wo::Vec3f, wi::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
)
    # Check hemisphere
    cos_θi = dot(wi, n)
    cos_θo = dot(wo, n)
    if cos_θi * cos_θo < 0f0
        # Opposite hemispheres - no contribution for reflection-only material
        return (SpectralRadiance(), 0f0)
    end

    cos_θ = abs(cos_θi)
    if cos_θ < 1f-6
        return (SpectralRadiance(), 0f0)
    end

    # Get material properties
    refl_rgb = eval_tex(textures, mat.reflectance, uv)
    eta = mat.eta
    thickness = max(eval_tex(textures, mat.thickness, uv), eps(Float32))
    albedo_rgb = eval_tex(textures, mat.albedo, uv)

    refl_spectral = uplift_rgb(table, refl_rgb, lambda)
    albedo_spectral = uplift_rgb(table, albedo_rgb, lambda)
    has_medium = !is_black(albedo_rgb)

    # Compute Fresnel terms for both directions
    F_o = fresnel_dielectric(abs(cos_θo), eta)
    F_i = fresnel_dielectric(cos_θ, eta)
    T_o = 1f0 - F_o
    T_i = 1f0 - F_i

    # Layer transmittance
    layer_tr = if has_medium
        # Simplified: constant absorption factor
        tr = layer_transmittance(thickness, wi)
        tr * tr * albedo_spectral
    else
        SpectralRadiance(1f0)
    end

    # Diffuse contribution through coating
    diffuse_f = refl_spectral * (1f0 / Float32(π))
    f_spectral = diffuse_f * T_o * T_i * layer_tr

    # PDF is cosine-weighted for diffuse, weighted by transmission probability
    pdf = T_o * cos_θ / Float32(π)

    return (f_spectral, pdf)
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
    lambda::Wavelengths, sample_u::Point2f, rng::Float32
)
    return sample_bsdf_spectral(table, mi.material, textures, wo, n, uv, lambda, sample_u, rng)
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
    lambda::Wavelengths, sample_u::Point2f, rng::Float32
)
    return sample_bsdf_spectral(table, mi.material, textures, wo, n, uv, lambda, sample_u, rng)
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

"""
    reflect(wo, n) -> Vec3f

Compute reflection direction.
"""
@propagate_inbounds function reflect(wo::Vec3f, n::Vec3f)::Vec3f
    return 2f0 * dot(wo, n) * n - wo
end

"""
    fresnel_dielectric(cos_theta_i, eta) -> Float32

Compute Fresnel reflectance for dielectric.
eta = n_t / n_i (ratio of indices of refraction)
"""
@propagate_inbounds function fresnel_dielectric(cos_theta_i::Float32, eta::Float32)::Float32
    cos_theta_i = clamp(cos_theta_i, -1f0, 1f0)

    if cos_theta_i < 0f0
        eta = 1f0 / eta
        cos_theta_i = -cos_theta_i
    end

    sin2_theta_i = 1f0 - cos_theta_i * cos_theta_i
    sin2_theta_t = sin2_theta_i / (eta * eta)

    if sin2_theta_t >= 1f0
        return 1f0  # Total internal reflection
    end

    cos_theta_t = sqrt(max(0f0, 1f0 - sin2_theta_t))

    r_parl = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t)
    r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t)

    return (r_parl * r_parl + r_perp * r_perp) * 0.5f0
end

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
