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

"""
    sample_bsdf_spectral(table, mat::PlasticMaterial, textures, wo, n, uv, lambda, sample_u, rng) -> SpectralBSDFSample

Sample plastic BSDF using CoatedDiffuse model (as per pbrt-v4).

In pbrt-v4, "plastic" material is converted to "coateddiffuse" with:
- Kd → reflectance
- roughness → roughness (for coating)
- eta = 1.5 (typical plastic IOR)
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

    # Get material properties - Kd is the diffuse reflectance, Ks influences nothing in pbrt-v4
    kd_rgb = eval_tex(textures, mat.Kd, uv)
    roughness = eval_tex(textures, mat.roughness, uv)

    # Remap roughness to microfacet alpha if needed
    alpha = if mat.remap_roughness
        roughness_to_α(roughness)
    else
        roughness
    end

    refl_spectral = uplift_rgb(table, kd_rgb, lambda)

    # Plastic IOR (typical dielectric)
    eta = 1.5f0

    # Build coordinate system from shading normal
    tangent, bitangent = coordinate_system(n)

    # Transform wo to local space
    wo_local = Vec3f(dot(wo, tangent), dot(wo, bitangent), wo_dot_n)

    # Two-sided: flip if entering from below
    flip = wo_local[3] < 0f0
    if flip
        wo_local = -wo_local
    end

    # === CoatedDiffuse LayeredBxDF strategy ===
    # For rough coating, we need to use microfacet sampling for the interface
    # For smooth coating, use simplified Fresnel-based sampling

    cos_θo = abs(wo_local[3])

    if alpha < 0.001f0
        # Smooth coating - use Fresnel-weighted sampling
        F = fresnel_dielectric(cos_θo, eta)

        if rng < F
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

        # Transmitted through top interface - sample diffuse base
        sin2_θi = max(0f0, 1f0 - cos_θo^2)
        sin2_θt = sin2_θi / (eta^2)

        if sin2_θt >= 1f0
            return SpectralBSDFSample()
        end

        cos_θt_in = sqrt(1f0 - sin2_θt)

        # Sample diffuse base
        local_diffuse_wi = cosine_sample_hemisphere(sample_u)
        cos_θ_diffuse = local_diffuse_wi[3]

        if cos_θ_diffuse < 1f-6
            return SpectralBSDFSample()
        end

        # Fresnel for exiting the coating
        F_out = fresnel_dielectric(cos_θ_diffuse, eta)
        T_in = 1f0 - F
        T_out = 1f0 - F_out

        if flip
            local_diffuse_wi = Vec3f(local_diffuse_wi[1], local_diffuse_wi[2], -local_diffuse_wi[3])
        end

        wi = tangent * local_diffuse_wi[1] + bitangent * local_diffuse_wi[2] + n * local_diffuse_wi[3]
        wi = normalize(wi)

        diffuse_f = refl_spectral * (1f0 / Float32(π))
        f_spectral = diffuse_f * T_in * T_out

        pdf = (1f0 - F) * cos_θ_diffuse / Float32(π)

        return SpectralBSDFSample(wi, f_spectral, pdf, false, 1f0)
    else
        # Rough coating - use microfacet distribution for interface
        # Sample the microfacet normal using TrowbridgeReitz VNDF
        wm = trowbridge_reitz_sample_wm(wo_local, sample_u, alpha, alpha)

        cos_θo_m = dot(wo_local, wm)
        if cos_θo_m < 0f0
            return SpectralBSDFSample()
        end

        F = fresnel_dielectric(cos_θo_m, eta)

        if rng < F
            # Reflect off microfacet
            wi_local = -wo_local + 2f0 * cos_θo_m * wm

            if wi_local[3] * wo_local[3] < 0f0
                return SpectralBSDFSample()
            end

            if flip
                wi_local = -wi_local
            end

            wi = tangent * wi_local[1] + bitangent * wi_local[2] + n * wi_local[3]
            wi = normalize(wi)

            # Microfacet BRDF: D * F * G / (4 * cos_i * cos_o)
            # But F is already in the probability, so f = D * G / (4 * cos_i * cos_o) / pdf
            D = trowbridge_reitz_d(wm, alpha, alpha)
            G = trowbridge_reitz_g(wo_local, wi_local, alpha, alpha)

            cos_i = abs(wi_local[3])
            cos_o = abs(wo_local[3])

            pdf_m = trowbridge_reitz_pdf(wo_local, wm, alpha, alpha)
            pdf = F * pdf_m / (4f0 * abs(cos_θo_m))

            f = D * G / (4f0 * cos_i * cos_o)

            return SpectralBSDFSample(wi, SpectralRadiance(f), pdf, false, 1f0)
        else
            # Transmit through coating to diffuse base
            # For rough coating, we still sample the diffuse simply
            # (full LayeredBxDF would do random walk, but this is a reasonable approximation)

            local_diffuse_wi = cosine_sample_hemisphere(sample_u)
            cos_θ_diffuse = local_diffuse_wi[3]

            if cos_θ_diffuse < 1f-6
                return SpectralBSDFSample()
            end

            # Average Fresnel transmission through rough interface
            T_in = 1f0 - F
            F_out = fresnel_dielectric(cos_θ_diffuse, eta)
            T_out = 1f0 - F_out

            if flip
                local_diffuse_wi = Vec3f(local_diffuse_wi[1], local_diffuse_wi[2], -local_diffuse_wi[3])
            end

            wi = tangent * local_diffuse_wi[1] + bitangent * local_diffuse_wi[2] + n * local_diffuse_wi[3]
            wi = normalize(wi)

            diffuse_f = refl_spectral * (1f0 / Float32(π))
            f_spectral = diffuse_f * T_in * T_out

            pdf = (1f0 - F) * cos_θ_diffuse / Float32(π)

            return SpectralBSDFSample(wi, f_spectral, pdf, false, 1f0)
        end
    end
end

"""
    evaluate_bsdf_spectral(table, mat::PlasticMaterial, textures, wo, wi, n, uv, lambda) -> (f, pdf)

Evaluate plastic BSDF using CoatedDiffuse model (as per pbrt-v4).
"""
@propagate_inbounds function evaluate_bsdf_spectral(
    table::RGBToSpectrumTable, mat::PlasticMaterial, textures,
    wo::Vec3f, wi::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
)
    # Check hemispheres
    cos_θi = dot(wi, n)
    cos_θo = dot(wo, n)
    if cos_θi * cos_θo < 0f0
        # Opposite hemispheres - no contribution for reflection-only
        return (SpectralRadiance(), 0f0)
    end

    # Get material properties
    kd_rgb = eval_tex(textures, mat.Kd, uv)
    roughness = eval_tex(textures, mat.roughness, uv)

    alpha = if mat.remap_roughness
        roughness_to_α(roughness)
    else
        roughness
    end

    refl_spectral = uplift_rgb(table, kd_rgb, lambda)
    eta = 1.5f0

    # Build local frame
    tangent, bitangent = coordinate_system(n)
    wo_local = Vec3f(dot(wo, tangent), dot(wo, bitangent), cos_θo)
    wi_local = Vec3f(dot(wi, tangent), dot(wi, bitangent), cos_θi)

    # Two-sided handling
    flip = wo_local[3] < 0f0
    if flip
        wo_local = -wo_local
        wi_local = -wi_local
    end

    abs_cos_θi = abs(wi_local[3])
    abs_cos_θo = abs(wo_local[3])

    if abs_cos_θi < 1f-6 || abs_cos_θo < 1f-6
        return (SpectralRadiance(), 0f0)
    end

    if alpha < 0.001f0
        # Smooth coating - only diffuse contribution (specular is delta)
        F_in = fresnel_dielectric(abs_cos_θo, eta)
        F_out = fresnel_dielectric(abs_cos_θi, eta)
        T_in = 1f0 - F_in
        T_out = 1f0 - F_out

        diffuse_f = refl_spectral * (1f0 / Float32(π))
        f_spectral = diffuse_f * T_in * T_out

        pdf = (1f0 - F_in) * abs_cos_θi / Float32(π)

        return (f_spectral, pdf)
    else
        # Rough coating - both specular and diffuse contributions
        # Compute half-vector for reflection
        wh = normalize(wo_local + wi_local)
        if wh[3] < 0f0
            wh = -wh
        end

        cos_θo_h = dot(wo_local, wh)
        F_spec = fresnel_dielectric(abs(cos_θo_h), eta)

        # Specular contribution: D * F * G / (4 * cos_i * cos_o)
        D = trowbridge_reitz_d(wh, alpha, alpha)
        G = trowbridge_reitz_g(wo_local, wi_local, alpha, alpha)

        f_spec = D * F_spec * G / (4f0 * abs_cos_θi * abs_cos_θo)

        # Diffuse contribution with Fresnel transmission
        F_in = fresnel_dielectric(abs_cos_θo, eta)
        F_out = fresnel_dielectric(abs_cos_θi, eta)
        T_in = 1f0 - F_in
        T_out = 1f0 - F_out

        diffuse_f = refl_spectral * (1f0 / Float32(π))
        f_diff = diffuse_f * T_in * T_out

        # Combined BSDF
        f_spectral = SpectralRadiance(f_spec) + f_diff

        # Combined PDF
        pdf_spec = F_in * trowbridge_reitz_pdf(wo_local, wh, alpha, alpha) / (4f0 * abs(cos_θo_h))
        pdf_diff = (1f0 - F_in) * abs_cos_θi / Float32(π)
        pdf = pdf_spec + pdf_diff

        return (f_spectral, pdf)
    end
end

"""
    sample_bsdf_spectral(table, mat::MetalMaterial, textures, wo, n, uv, lambda, sample_u, rng) -> SpectralBSDFSample

Sample metal BSDF with conductor Fresnel.
Matches pbrt-v4's ConductorBxDF::Sample_f exactly.

The implementation works in local shading coordinates where n = (0,0,1), then transforms back.
"""
@propagate_inbounds function sample_bsdf_spectral(
    table::RGBToSpectrumTable, mat::MetalMaterial, textures,
    wo_world::Vec3f, n::Vec3f, uv::Point2f,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32
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

"""
    sample_bsdf_spectral(table, mat::CoatedDiffuseMaterial, textures, wo, n, uv, lambda, sample_u, rng) -> SpectralBSDFSample

Sample CoatedDiffuse BSDF using pbrt-v4's LayeredBxDF approach.

For smooth coatings: Fresnel-weighted specular/diffuse sampling
For rough coatings: Microfacet interface with diffuse base
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

    # Get roughness parameters
    u_roughness = eval_tex(textures, mat.u_roughness, uv)
    v_roughness = eval_tex(textures, mat.v_roughness, uv)

    # Remap roughness if needed
    alpha_x = mat.remap_roughness ? roughness_to_α(u_roughness) : u_roughness
    alpha_y = mat.remap_roughness ? roughness_to_α(v_roughness) : v_roughness

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

    cos_θo = abs(wo_local[3])

    # Check if coating is effectively smooth
    is_smooth = trowbridge_reitz_effectively_smooth(alpha_x, alpha_y)

    if is_smooth
        # === Smooth coating: Fresnel-weighted sampling ===
        F = fresnel_dielectric(cos_θo, eta)

        if rng < F
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

        # Transmitted through - sample diffuse base
        sin2_θi = max(0f0, 1f0 - cos_θo^2)
        sin2_θt = sin2_θi / (eta^2)

        if sin2_θt >= 1f0
            return SpectralBSDFSample()
        end

        cos_θt_in = sqrt(1f0 - sin2_θt)

        local_diffuse_wi = cosine_sample_hemisphere(sample_u)
        cos_θ_diffuse = local_diffuse_wi[3]

        if cos_θ_diffuse < 1f-6
            return SpectralBSDFSample()
        end

        F_out = fresnel_dielectric(cos_θ_diffuse, eta)
        T_in = 1f0 - F
        T_out = 1f0 - F_out

        layer_tr = if has_medium
            tr_in = layer_transmittance(thickness, Vec3f(0, 0, cos_θt_in))
            tr_out = layer_transmittance(thickness, local_diffuse_wi)
            tr_in * tr_out * albedo_spectral
        else
            SpectralRadiance(1f0)
        end

        if flip
            local_diffuse_wi = Vec3f(local_diffuse_wi[1], local_diffuse_wi[2], -local_diffuse_wi[3])
        end

        wi = tangent * local_diffuse_wi[1] + bitangent * local_diffuse_wi[2] + n * local_diffuse_wi[3]
        wi = normalize(wi)

        diffuse_f = refl_spectral * (1f0 / Float32(π))
        f_spectral = diffuse_f * T_in * T_out * layer_tr

        pdf = (1f0 - F) * cos_θ_diffuse / Float32(π)

        return SpectralBSDFSample(wi, f_spectral, pdf, false, 1f0)
    else
        # === Rough coating: Microfacet interface ===
        # Sample microfacet normal using TrowbridgeReitz VNDF
        wm = trowbridge_reitz_sample_wm(wo_local, sample_u, alpha_x, alpha_y)

        cos_θo_m = dot(wo_local, wm)
        if cos_θo_m < 0f0
            return SpectralBSDFSample()
        end

        F = fresnel_dielectric(cos_θo_m, eta)

        if rng < F
            # Reflect off microfacet
            wi_local = -wo_local + 2f0 * cos_θo_m * wm

            if wi_local[3] * wo_local[3] < 0f0
                return SpectralBSDFSample()
            end

            if flip
                wi_local = -wi_local
            end

            wi = tangent * wi_local[1] + bitangent * wi_local[2] + n * wi_local[3]
            wi = normalize(wi)

            # Microfacet BRDF for dielectric reflection
            D = trowbridge_reitz_d(wm, alpha_x, alpha_y)
            G = trowbridge_reitz_g(wo_local, wi_local, alpha_x, alpha_y)

            cos_i = abs(wi_local[3])
            cos_o = abs(wo_local[3])

            pdf_m = trowbridge_reitz_pdf(wo_local, wm, alpha_x, alpha_y)
            pdf = F * pdf_m / (4f0 * abs(cos_θo_m))

            f = D * G / (4f0 * cos_i * cos_o)

            return SpectralBSDFSample(wi, SpectralRadiance(f), pdf, false, 1f0)
        else
            # Transmit through coating to diffuse base
            local_diffuse_wi = cosine_sample_hemisphere(sample_u)
            cos_θ_diffuse = local_diffuse_wi[3]

            if cos_θ_diffuse < 1f-6
                return SpectralBSDFSample()
            end

            # Average Fresnel transmission through rough interface
            T_in = 1f0 - F
            F_out = fresnel_dielectric(cos_θ_diffuse, eta)
            T_out = 1f0 - F_out

            layer_tr = if has_medium
                tr_in = layer_transmittance(thickness, Vec3f(0, 0, cos_θo))
                tr_out = layer_transmittance(thickness, local_diffuse_wi)
                tr_in * tr_out * albedo_spectral
            else
                SpectralRadiance(1f0)
            end

            if flip
                local_diffuse_wi = Vec3f(local_diffuse_wi[1], local_diffuse_wi[2], -local_diffuse_wi[3])
            end

            wi = tangent * local_diffuse_wi[1] + bitangent * local_diffuse_wi[2] + n * local_diffuse_wi[3]
            wi = normalize(wi)

            diffuse_f = refl_spectral * (1f0 / Float32(π))
            f_spectral = diffuse_f * T_in * T_out * layer_tr

            pdf = (1f0 - F) * cos_θ_diffuse / Float32(π)

            return SpectralBSDFSample(wi, f_spectral, pdf, false, 1f0)
        end
    end
end

"""
    evaluate_bsdf_spectral(table, mat::CoatedDiffuseMaterial, textures, wo, wi, n, uv, lambda) -> (f, pdf)

Evaluate CoatedDiffuse BSDF for given directions.
Supports both smooth and rough coatings.
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

    abs_cos_θi = abs(cos_θi)
    abs_cos_θo = abs(cos_θo)

    if abs_cos_θi < 1f-6 || abs_cos_θo < 1f-6
        return (SpectralRadiance(), 0f0)
    end

    # Get material properties
    refl_rgb = eval_tex(textures, mat.reflectance, uv)
    eta = mat.eta
    thickness = max(eval_tex(textures, mat.thickness, uv), eps(Float32))
    albedo_rgb = eval_tex(textures, mat.albedo, uv)

    # Get roughness parameters
    u_roughness = eval_tex(textures, mat.u_roughness, uv)
    v_roughness = eval_tex(textures, mat.v_roughness, uv)

    alpha_x = mat.remap_roughness ? roughness_to_α(u_roughness) : u_roughness
    alpha_y = mat.remap_roughness ? roughness_to_α(v_roughness) : v_roughness

    refl_spectral = uplift_rgb(table, refl_rgb, lambda)
    albedo_spectral = uplift_rgb(table, albedo_rgb, lambda)
    has_medium = !is_black(albedo_rgb)

    # Build local frame
    tangent, bitangent = coordinate_system(n)
    wo_local = Vec3f(dot(wo, tangent), dot(wo, bitangent), cos_θo)
    wi_local = Vec3f(dot(wi, tangent), dot(wi, bitangent), cos_θi)

    # Two-sided handling
    flip = wo_local[3] < 0f0
    if flip
        wo_local = -wo_local
        wi_local = -wi_local
    end

    is_smooth = trowbridge_reitz_effectively_smooth(alpha_x, alpha_y)

    if is_smooth
        # Smooth coating - only diffuse contribution (specular is delta)
        F_o = fresnel_dielectric(abs(wo_local[3]), eta)
        F_i = fresnel_dielectric(abs(wi_local[3]), eta)
        T_o = 1f0 - F_o
        T_i = 1f0 - F_i

        layer_tr = if has_medium
            tr = layer_transmittance(thickness, wi_local)
            tr * tr * albedo_spectral
        else
            SpectralRadiance(1f0)
        end

        diffuse_f = refl_spectral * (1f0 / Float32(π))
        f_spectral = diffuse_f * T_o * T_i * layer_tr

        pdf = T_o * abs(wi_local[3]) / Float32(π)

        return (f_spectral, pdf)
    else
        # Rough coating - both specular and diffuse contributions
        # Compute half-vector for reflection
        wh = normalize(wo_local + wi_local)
        if wh[3] < 0f0
            wh = -wh
        end

        cos_θo_h = dot(wo_local, wh)
        F_spec = fresnel_dielectric(abs(cos_θo_h), eta)

        # Specular contribution: D * F * G / (4 * cos_i * cos_o)
        D = trowbridge_reitz_d(wh, alpha_x, alpha_y)
        G = trowbridge_reitz_g(wo_local, wi_local, alpha_x, alpha_y)

        f_spec = D * F_spec * G / (4f0 * abs(wi_local[3]) * abs(wo_local[3]))

        # Diffuse contribution with Fresnel transmission
        F_o = fresnel_dielectric(abs(wo_local[3]), eta)
        F_i = fresnel_dielectric(abs(wi_local[3]), eta)
        T_o = 1f0 - F_o
        T_i = 1f0 - F_i

        layer_tr = if has_medium
            tr = layer_transmittance(thickness, wi_local)
            tr * tr * albedo_spectral
        else
            SpectralRadiance(1f0)
        end

        diffuse_f = refl_spectral * (1f0 / Float32(π))
        f_diff = diffuse_f * T_o * T_i * layer_tr

        # Combined BSDF
        f_spectral = SpectralRadiance(f_spec) + f_diff

        # Combined PDF
        pdf_spec = F_o * trowbridge_reitz_pdf(wo_local, wh, alpha_x, alpha_y) / (4f0 * abs(cos_θo_h))
        pdf_diff = T_o * abs(wi_local[3]) / Float32(π)
        pdf = pdf_spec + pdf_diff

        return (f_spectral, pdf)
    end
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
    sample_bsdf_spectral(table, mat::ThinDielectricMaterial, textures, wo, n, uv, lambda, sample_u, rng) -> SpectralBSDFSample

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
    lambda::Wavelengths, sample_u::Point2f, rng::Float32
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
    sample_bsdf_spectral(table, mat::DiffuseTransmissionMaterial, textures, wo, n, uv, lambda, sample_u, rng) -> SpectralBSDFSample

Sample diffuse transmission BSDF matching pbrt-v4's DiffuseTransmissionBxDF::Sample_f.

This material diffusely scatters light in both reflection (same hemisphere)
and transmission (opposite hemisphere). Sampling is proportional to max(R) and max(T).
"""
@propagate_inbounds function sample_bsdf_spectral(
    table::RGBToSpectrumTable, mat::DiffuseTransmissionMaterial, textures,
    wo::Vec3f, n::Vec3f, uv::Point2f,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32
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
    sample_bsdf_spectral(table, mat::CoatedConductorMaterial, textures, wo, n, uv, lambda, sample_u, rng) -> SpectralBSDFSample

Sample CoatedConductor BSDF using pbrt-v4's LayeredBxDF approach.

This is a layered material with:
- Top layer: Dielectric coating (can be rough or smooth)
- Bottom layer: Conductor (metal) with complex Fresnel

Key pbrt-v4 details (materials.cpp lines 345-392):
- Conductor eta/k are scaled by interface IOR: ce /= ieta, ck /= ieta
- If reflectance mode: k = 2 * sqrt(r) / sqrt(1 - r), eta = 1
"""
@propagate_inbounds function sample_bsdf_spectral(
    table::RGBToSpectrumTable, mat::CoatedConductorMaterial, textures,
    wo::Vec3f, n::Vec3f, uv::Point2f,
    lambda::Wavelengths, sample_u::Point2f, rng::Float32
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

"""
    same_hemisphere(w, wp) -> Bool

Check if two directions are in the same hemisphere (matches pbrt-v4's SameHemisphere).
"""
@inline same_hemisphere(w::Vec3f, wp::Vec3f)::Bool = w[3] * wp[3] > 0f0

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
