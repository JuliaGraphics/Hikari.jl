# Spectral BSDF Evaluation Interface for PhysicalWavefront
# Enables spectral path tracing while keeping materials as RGB containers

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
@inline SpectralBSDFSample() = SpectralBSDFSample(Vec3f(0, 0, 1), SpectralRadiance(), 0f0, false, 1f0)

# ============================================================================
# Spectral BSDF Evaluation for Each Material Type
# ============================================================================

# These functions extract material properties and evaluate them spectrally.
# They use the RGB-to-spectral uplift to convert material colors to wavelength-dependent values.

"""
    sample_bsdf_spectral(mat::MatteMaterial, wo, n, uv, lambda, sample_u, rng) -> SpectralBSDFSample

Sample diffuse BSDF with spectral evaluation.
Uses pbrt-v4 convention: work in local shading space where n = (0,0,1).
"""
@inline function sample_bsdf_spectral(
    mat::MatteMaterial, wo::Vec3f, n::Vec3f, uv::Point2f,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32
)
    # Check for grazing angle (wo perpendicular to shading normal)
    # This matches pbrt-v4's wo.z == 0 check in BSDF::Sample_f
    wo_dot_n = dot(wo, n)
    if abs(wo_dot_n) < 1f-6
        return SpectralBSDFSample()
    end

    # Get material properties (rng unused for diffuse)
    kd_rgb = evaluate_texture(mat.Kd, uv)
    σ = evaluate_texture(mat.σ, uv)

    # Uplift to spectral
    kd_spectral = uplift_rgb(kd_rgb, lambda)

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
    sample_bsdf_spectral(mat::MirrorMaterial, wo, n, uv, lambda, sample_u, rng) -> SpectralBSDFSample

Sample perfect specular reflection with spectral evaluation.
"""
@inline function sample_bsdf_spectral(
    mat::MirrorMaterial, wo::Vec3f, n::Vec3f, uv::Point2f,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32
)
    # Check for grazing angle
    wo_dot_n = dot(wo, n)
    if abs(wo_dot_n) < 1f-6
        return SpectralBSDFSample()
    end

    # Get reflectance (rng and sample_u unused for perfect specular)
    kr_rgb = evaluate_texture(mat.Kr, uv)
    kr_spectral = uplift_rgb(kr_rgb, lambda)

    # Orient normal to face wo for reflection
    n_oriented = wo_dot_n < 0f0 ? -n : n

    # Perfect reflection
    wi = reflect(wo, n_oriented)

    # Delta distribution: f = Kr, pdf = 1 (conceptually infinite, but we handle it specially)
    return SpectralBSDFSample(wi, kr_spectral, 1f0, true, 1f0)
end

"""
    sample_bsdf_spectral(mat::GlassMaterial, wo, n, uv, lambda, sample_u, rng) -> SpectralBSDFSample

Sample glass BSDF with reflection or refraction.
Uses Fresnel to choose between reflection and transmission.
"""
@inline function sample_bsdf_spectral(
    mat::GlassMaterial, wo::Vec3f, n::Vec3f, uv::Point2f,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32
)
    # Get material properties
    kr_rgb = evaluate_texture(mat.Kr, uv)
    kt_rgb = evaluate_texture(mat.Kt, uv)
    ior = evaluate_texture(mat.index, uv)

    kr_spectral = uplift_rgb(kr_rgb, lambda)
    kt_spectral = uplift_rgb(kt_rgb, lambda)

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
    sample_bsdf_spectral(mat::PlasticMaterial, wo, n, uv, lambda, sample_u, rng) -> SpectralBSDFSample

Sample plastic BSDF (diffuse + glossy specular).
"""
@inline function sample_bsdf_spectral(
    mat::PlasticMaterial, wo::Vec3f, n::Vec3f, uv::Point2f,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32
)
    # Check for grazing angle
    wo_dot_n = dot(wo, n)
    if abs(wo_dot_n) < 1f-6
        return SpectralBSDFSample()
    end

    # Get material properties
    kd_rgb = evaluate_texture(mat.Kd, uv)
    ks_rgb = evaluate_texture(mat.Ks, uv)
    roughness = evaluate_texture(mat.roughness, uv)

    kd_spectral = uplift_rgb(kd_rgb, lambda)
    ks_spectral = uplift_rgb(ks_rgb, lambda)

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
    sample_bsdf_spectral(mat::MetalMaterial, wo, n, uv, lambda, sample_u, rng) -> SpectralBSDFSample

Sample metal BSDF with conductor Fresnel.
"""
@inline function sample_bsdf_spectral(
    mat::MetalMaterial, wo::Vec3f, n::Vec3f, uv::Point2f,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32
)
    # Check for grazing angle
    wo_dot_n = dot(wo, n)
    if abs(wo_dot_n) < 1f-6
        return SpectralBSDFSample()
    end

    # Get material properties (rng unused)
    eta_rgb = evaluate_texture(mat.eta, uv)
    k_rgb = evaluate_texture(mat.k, uv)
    roughness = evaluate_texture(mat.roughness, uv)
    reflectance_rgb = evaluate_texture(mat.reflectance, uv)

    reflectance_spectral = uplift_rgb(reflectance_rgb, lambda)

    # Orient normal to face wo for reflection
    n_oriented = wo_dot_n < 0f0 ? -n : n

    # For smooth metals, use perfect reflection
    if roughness < 0.01f0
        wi = reflect(wo, n_oriented)
        cos_theta = abs(wo_dot_n)

        # Fresnel for conductor (using RGB approximation for now)
        # TODO: Full spectral Fresnel for metals
        F_rgb = fresnel_conductor(cos_theta, RGBSpectrum(1f0), eta_rgb, k_rgb)
        F_spectral = uplift_rgb(F_rgb, lambda)

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
        F_spectral = uplift_rgb(F_rgb, lambda)

        # Compute PDF and BSDF value
        # Simplified: approximate as specular for now
        f = reflectance_spectral * F_spectral

        return SpectralBSDFSample(wi, f, 1f0, false, 1f0)
    end
end

"""
    sample_bsdf_spectral(mat::EmissiveMaterial, wo, n, uv, lambda, sample_u, rng) -> SpectralBSDFSample

Emissive materials don't scatter - return invalid sample.
"""
@inline function sample_bsdf_spectral(
    mat::EmissiveMaterial, wo::Vec3f, n::Vec3f, uv::Point2f,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32
)
    return SpectralBSDFSample()
end

# Fallback for unknown materials
@inline function sample_bsdf_spectral(
    mat::Material, wo::Vec3f, n::Vec3f, uv::Point2f,
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
    evaluate_bsdf_spectral(mat, wo, wi, n, uv, lambda) -> (f::SpectralRadiance, pdf::Float32)

Evaluate BSDF for a given pair of directions. Used for MIS in direct lighting.
Returns the BSDF value and PDF.
"""
@inline function evaluate_bsdf_spectral(
    mat::MatteMaterial, wo::Vec3f, wi::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
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

    kd_rgb = evaluate_texture(mat.Kd, uv)
    kd_spectral = uplift_rgb(kd_rgb, lambda)
    f = kd_spectral * (1f0 / Float32(π))
    pdf = cos_theta / Float32(π)

    return (f, pdf)
end

@inline function evaluate_bsdf_spectral(
    mat::MirrorMaterial, wo::Vec3f, wi::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
)
    # Perfect specular has zero PDF for non-delta directions
    return (SpectralRadiance(), 0f0)
end

@inline function evaluate_bsdf_spectral(
    mat::GlassMaterial, wo::Vec3f, wi::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
)
    # Specular has zero PDF for non-delta directions
    return (SpectralRadiance(), 0f0)
end

@inline function evaluate_bsdf_spectral(
    mat::PlasticMaterial, wo::Vec3f, wi::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
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
    kd_rgb = evaluate_texture(mat.Kd, uv)
    kd_spectral = uplift_rgb(kd_rgb, lambda)
    f = kd_spectral * (1f0 / Float32(π))
    pdf = cos_theta / Float32(π)

    return (f, pdf)
end

@inline function evaluate_bsdf_spectral(
    mat::MetalMaterial, wo::Vec3f, wi::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
)
    # For rough metals, we'd need microfacet evaluation
    # For now, treat as specular (zero contribution)
    return (SpectralRadiance(), 0f0)
end

@inline function evaluate_bsdf_spectral(
    mat::EmissiveMaterial, wo::Vec3f, wi::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
)
    # Emissive materials don't scatter light (black body)
    return (SpectralRadiance(), 0f0)
end

# Fallback
@inline function evaluate_bsdf_spectral(
    mat::Material, wo::Vec3f, wi::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
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
    get_emission_spectral(mat::EmissiveMaterial, wo, n, uv, lambda) -> SpectralRadiance

Get spectral emission from an emissive material.
"""
@inline function get_emission_spectral(
    mat::EmissiveMaterial, wo::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
)
    cos_theta = dot(wo, n)
    if !mat.two_sided && cos_theta < 0f0
        return SpectralRadiance()
    end

    Le_rgb = evaluate_texture(mat.Le, uv) * mat.scale
    return uplift_rgb(Le_rgb, lambda)
end

# Non-emissive materials return zero
@inline function get_emission_spectral(
    mat::Material, wo::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
)
    return SpectralRadiance()
end

# ============================================================================
# Spectral Albedo Extraction (for denoising aux buffers)
# ============================================================================

"""
    get_albedo_spectral(mat, uv, lambda) -> SpectralRadiance

Extract material albedo as spectral value for denoising auxiliary buffers.
"""
@inline function get_albedo_spectral(mat::MatteMaterial, uv::Point2f, lambda::Wavelengths)
    return uplift_rgb(evaluate_texture(mat.Kd, uv), lambda)
end

@inline function get_albedo_spectral(mat::MirrorMaterial, uv::Point2f, lambda::Wavelengths)
    return uplift_rgb(evaluate_texture(mat.Kr, uv), lambda)
end

@inline function get_albedo_spectral(mat::GlassMaterial, uv::Point2f, lambda::Wavelengths)
    # For glass, use average of reflection and transmission
    kr = evaluate_texture(mat.Kr, uv)
    kt = evaluate_texture(mat.Kt, uv)
    avg = RGBSpectrum((kr.c[1] + kt.c[1]) * 0.5f0,
                      (kr.c[2] + kt.c[2]) * 0.5f0,
                      (kr.c[3] + kt.c[3]) * 0.5f0)
    return uplift_rgb(avg, lambda)
end

@inline function get_albedo_spectral(mat::PlasticMaterial, uv::Point2f, lambda::Wavelengths)
    return uplift_rgb(evaluate_texture(mat.Kd, uv), lambda)
end

@inline function get_albedo_spectral(mat::MetalMaterial, uv::Point2f, lambda::Wavelengths)
    return uplift_rgb(evaluate_texture(mat.reflectance, uv), lambda)
end

@inline function get_albedo_spectral(mat::EmissiveMaterial, uv::Point2f, lambda::Wavelengths)
    # Normalized emission color
    Le = evaluate_texture(mat.Le, uv)
    lum = to_Y(Le)
    if lum > 0f0
        normalized = Le / lum
        return uplift_rgb(normalized, lambda)
    end
    return SpectralRadiance()
end

@inline function get_albedo_spectral(mat::Material, uv::Point2f, lambda::Wavelengths)
    return SpectralRadiance(0.5f0)
end

# ============================================================================
# Helper Functions
# ============================================================================

# cosine_sample_hemisphere is defined in Hikari.jl

"""
    coordinate_system(n::Vec3f) -> (tangent, bitangent)

Build orthonormal basis from a normal vector.
"""
@inline function coordinate_system(n::Vec3f)
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
@inline function local_to_world(local_dir::Vec3f, n::Vec3f, tangent::Vec3f, bitangent::Vec3f)::Vec3f
    return tangent * local_dir[1] + bitangent * local_dir[2] + n * local_dir[3]
end

"""
    reflect(wo, n) -> Vec3f

Compute reflection direction.
"""
@inline function reflect(wo::Vec3f, n::Vec3f)::Vec3f
    return 2f0 * dot(wo, n) * n - wo
end

"""
    fresnel_dielectric(cos_theta_i, eta) -> Float32

Compute Fresnel reflectance for dielectric.
eta = n_t / n_i (ratio of indices of refraction)
"""
@inline function fresnel_dielectric(cos_theta_i::Float32, eta::Float32)::Float32
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
@inline function sample_ggx_vndf(wo::Vec3f, alpha_x::Float32, alpha_y::Float32, u::Point2f)::Vec3f
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
