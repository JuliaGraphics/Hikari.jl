# Spectral light sampling interface for PhysicalWavefront
# Wraps Hikari's existing light types and converts to spectral domain

# ============================================================================
# Spectral Light Sample Result
# ============================================================================

"""
    PWLightSample

Result of sampling a light source with spectral radiance.
"""
struct PWLightSample
    Li::SpectralRadiance   # Spectral incident radiance
    wi::Vec3f              # Direction to light
    pdf::Float32           # Probability density
    p_light::Point3f       # Point on light (for shadow ray)
    is_delta::Bool         # True for point/distant lights (no MIS needed)
end

@propagate_inbounds PWLightSample() = PWLightSample(
    SpectralRadiance(0f0),
    Vec3f(0f0, 0f0, 1f0),
    0f0,
    Point3f(0f0, 0f0, 0f0),
    false
)

# ============================================================================
# Light Sampling for Each Light Type
# All functions take rgb2spec_table for GPU-compatible spectral conversion
# ============================================================================

"""
    sample_light_spectral(table, lights, light::PointLight, p::Point3f, lambda::Wavelengths, u::Point2f)

Sample a point light spectrally.
"""
@propagate_inbounds function sample_light_spectral(
    table::RGBToSpectrumTable, lights, light::PointLight, p::Point3f, lambda::Wavelengths, ::Point2f
)::PWLightSample
    # Direction and distance to light
    to_light = light.position - p
    dist_sq = dot(to_light, to_light)
    dist = sqrt(dist_sq)

    if dist < 1f-6
        return PWLightSample()
    end

    wi = to_light / dist

    # Point lights have delta distribution
    # Li = scale * I->Sample(lambda) / r^2  (matching pbrt-v4's PointLight::SampleLi)
    # Sample() handles the D65 illuminant multiplication for RGBIlluminantSpectrum
    Li = light.scale * Sample(table, light.i, lambda) / dist_sq

    return PWLightSample(Li, wi, 1f0, light.position, true)
end

"""
    sample_light_spectral(table, lights, light::SpotLight, p::Point3f, lambda::Wavelengths, u::Point2f)

Sample a spotlight spectrally.
"""
@propagate_inbounds function sample_light_spectral(
    table::RGBToSpectrumTable, lights, light::SpotLight, p::Point3f, lambda::Wavelengths, ::Point2f
)::PWLightSample
    # Direction and distance to light
    to_light = light.position - p
    dist_sq = dot(to_light, to_light)
    dist = sqrt(dist_sq)

    if dist < 1f-6
        return PWLightSample()
    end

    wi = to_light / dist

    # Compute spotlight falloff
    # Transform -wi to light's local space and check z-component
    # (spotlight points in +Z direction in local space)
    wi_local = normalize(light.world_to_light(-wi))
    cos_theta = wi_local[3]

    if cos_theta < light.cos_total_width
        return PWLightSample()
    end

    # Compute falloff
    spot_falloff = if cos_theta >= light.cos_falloff_start
        1f0
    else
        # Smooth falloff
        delta = (cos_theta - light.cos_total_width) /
                (light.cos_falloff_start - light.cos_total_width)
        delta * delta * delta * delta
    end

    # Li = scale * I->Sample(lambda) * falloff / r^2  (matching pbrt-v4's SpotLight::I)
    # Sample() handles the D65 illuminant multiplication for RGBIlluminantSpectrum
    Li = light.scale * Sample(table, light.i, lambda) * spot_falloff / dist_sq

    return PWLightSample(Li, wi, 1f0, light.position, true)
end

"""
    sample_light_spectral(table, lights, light::DirectionalLight, p::Point3f, lambda::Wavelengths, u::Point2f)

Sample a directional light spectrally.
"""
@propagate_inbounds function sample_light_spectral(
    table::RGBToSpectrumTable, lights, light::DirectionalLight, p::Point3f, lambda::Wavelengths, ::Point2f
)::PWLightSample
    # Direction is opposite to light's travel direction
    wi = -light.direction

    # Distant lights have delta distribution
    # p_light is at "infinity" - use large distance for shadow ray
    p_light = Point3f(p + 1f6 * wi)

    # Use illuminant uplift (matches pbrt's RGBIlluminantSpectrum)
    # This multiplies by D65 illuminant spectrum - critical for correct white reproduction
    # Apply light.scale for photometric normalization
    Li = light.scale * uplift_rgb_illuminant(table, light.i, lambda)

    return PWLightSample(Li, wi, 1f0, p_light, true)
end

"""
    sample_light_spectral(table, lights, light::SunLight, p::Point3f, lambda::Wavelengths, u::Point2f)

Sample a sun light spectrally.
"""
@propagate_inbounds function sample_light_spectral(
    table::RGBToSpectrumTable, lights, light::SunLight, p::Point3f, lambda::Wavelengths, ::Point2f
)::PWLightSample
    # Direction is opposite to light's travel direction
    wi = -light.direction

    # Distant lights have delta distribution
    p_light = Point3f(p + 1f6 * wi)

    # Use illuminant uplift (matches pbrt's RGBIlluminantSpectrum)
    # This multiplies by D65 illuminant spectrum - critical for correct white reproduction
    # Apply light.scale for photometric normalization (1/D65_PHOTOMETRIC for RGB-constructed lights)
    Li = light.scale * uplift_rgb_illuminant(table, light.i, lambda)

    return PWLightSample(Li, wi, 1f0, p_light, true)
end

"""
    sample_light_spectral(table, lights, light::SunSkyLight, p::Point3f, lambda::Wavelengths, u::Point2f)

Importance-sample the sky hemisphere from SunSkyLight spectrally.
Uses the pre-computed Distribution2D to sample directions proportional to sky+sun radiance,
matching the SunSkyLight's sample_li method.
"""
@propagate_inbounds function sample_light_spectral(
    table::RGBToSpectrumTable, lights, light::SunSkyLight, p::Point3f, lambda::Wavelengths, u::Point2f
)::PWLightSample
    # Importance sample direction from pre-computed distribution (matches sample_li)
    uv, map_pdf = sample_continuous(light.distribution, u, lights)
    wi = hemisphere_uv_to_direction(uv)

    # Convert PDF from image space to solid angle
    θ = uv[2] * Float32(π) / 2f0
    sin_θ = sin(θ)
    pdf_val = sin_θ > 0f0 ? map_pdf / (Float32(π) * Float32(π) * sin_θ) : 0f0

    if pdf_val <= 0f0
        return PWLightSample(SpectralRadiance(0f0), wi, 0f0, Point3f(0f0), false)
    end

    # Get sky + sun radiance for this direction
    Le_rgb = sky_radiance(light, wi) + sun_disk_radiance(light, wi)

    # Unbounded uplift: sky model outputs RGB radiance (Preetham with 0.04 scale),
    # NOT illuminant quantities. D65 illuminant uplift would multiply by ~100x.
    Li = uplift_rgb_unbounded(table, Le_rgb, lambda)

    # Light at infinity
    p_light = Point3f(p + 1f6 * wi)

    return PWLightSample(Li, wi, pdf_val, p_light, false)
end

"""
    sample_light_spectral(table, lights, light::EnvironmentLight, p::Point3f, lambda::Wavelengths, u::Point2f)

Sample environment light spectrally with importance sampling.
"""
@propagate_inbounds function sample_light_spectral(
    table::RGBToSpectrumTable, lights, light::EnvironmentLight, p::Point3f, lambda::Wavelengths, u::Point2f
)::PWLightSample
    # Importance sample the environment map based on luminance
    # Pass lights for deref of TextureRef fields in Distribution2D
    uv, map_pdf = sample_continuous(light.env_map.distribution, u, lights)

    # Convert UV to direction using equal-area mapping
    wi = uv_to_direction(uv, light.env_map.rotation)

    # Convert PDF from image space to solid angle
    # For equal-area mapping: pdf_solidangle = pdf_image / (4π)
    # This matches pbrt-v4's ImageInfiniteLight::PDF_Li: return pdf / (4 * Pi);
    pdf = map_pdf / (4f0 * Float32(π))

    if pdf <= 0f0
        return PWLightSample()
    end

    # Sample environment map color at UV (using lookup_uv like pbrt-v4's ImageLe)
    # Pass lights for deref of TextureRef fields in EnvironmentMap
    Li_rgb = lookup_uv(light.env_map, uv, lights) * light.scale

    # p_light at infinity
    p_light = Point3f(p + 1f6 * wi)

    # Use illuminant uplift (matches pbrt's RGBIlluminantSpectrum)
    # This multiplies by D65 illuminant spectrum - critical for correct white reproduction
    Li = uplift_rgb_illuminant(table, Li_rgb, lambda)

    return PWLightSample(Li, wi, pdf, p_light, false)
end

"""
    sample_light_spectral(table, lights, light::AmbientLight, p::Point3f, lambda::Wavelengths, u::Point2f)

Sample ambient light spectrally (uniform sphere).
NOTE: Ambient light represents uniform illumination from all directions.
We sample the full sphere uniformly - the BSDF evaluation will naturally
give zero for directions below the surface, and cos_theta weighting handles the rest.
"""
@propagate_inbounds function sample_light_spectral(
    table::RGBToSpectrumTable, lights, light::AmbientLight, p::Point3f, lambda::Wavelengths, u::Point2f
)::PWLightSample
    # Ambient lights provide constant illumination from all directions
    # Sample uniform sphere - BSDF evaluation filters invalid directions

    # Uniform sphere sampling
    z = 1f0 - 2f0 * u[1]  # z in [-1, 1]
    r = sqrt(max(0f0, 1f0 - z * z))
    phi = 2f0 * Float32(π) * u[2]
    wi = Vec3f(r * cos(phi), r * sin(phi), z)

    # PDF for uniform sphere = 1 / (4π)
    pdf = 1f0 / (4f0 * Float32(π))

    p_light = Point3f(p + 1f6 * wi)

    # Matches pbrt-v4's UniformInfiniteLight::SampleLi:
    # Li = scale * Lemit->Sample(lambda)
    Li = light.scale * Sample(table, light.i, lambda)

    return PWLightSample(Li, wi, pdf, p_light, false)
end

# Fallback for unknown light types
@propagate_inbounds function sample_light_spectral(
    ::RGBToSpectrumTable, lights, ::Light, ::Point3f, ::Wavelengths, ::Point2f
)::PWLightSample
    return PWLightSample()
end

# ============================================================================
# Light PDF Evaluation (for MIS)
# ============================================================================

"""
    pdf_li_spectral(light::PointLight, p::Point3f, wi::Vec3f)

PDF for sampling direction wi from point light (always delta).
"""
@propagate_inbounds pdf_li_spectral(::PointLight, ::Point3f, ::Vec3f) = 0f0

"""
    pdf_li_spectral(light::DirectionalLight, p::Point3f, wi::Vec3f)

PDF for sampling direction wi from directional light (always delta).
"""
@propagate_inbounds pdf_li_spectral(::DirectionalLight, ::Point3f, ::Vec3f) = 0f0

"""
    pdf_li_spectral(light::SunLight, p::Point3f, wi::Vec3f)

PDF for sampling direction wi from sun light (always delta).
"""
@propagate_inbounds pdf_li_spectral(::SunLight, ::Point3f, ::Vec3f) = 0f0

"""
    pdf_li_spectral(light::SpotLight, p::Point3f, wi::Vec3f)

PDF for sampling direction wi from spotlight (always delta).
"""
@propagate_inbounds pdf_li_spectral(::SpotLight, ::Point3f, ::Vec3f) = 0f0

"""
    pdf_li_spectral(lights, light::EnvironmentLight, p::Point3f, wi::Vec3f)

PDF for sampling direction wi from environment light.
"""
@propagate_inbounds function pdf_li_spectral(lights, light::EnvironmentLight, ::Point3f, wi::Vec3f)
    # Convert direction to UV using equal-area mapping
    uv = direction_to_uv(wi, light.env_map.rotation)

    # Get PDF from distribution (pass lights for deref of TextureRef fields)
    map_pdf = pdf(light.env_map.distribution, uv, lights)

    # Convert from image space to solid angle
    # For equal-area mapping: pdf_solidangle = pdf_image / (4π)
    # This matches pbrt-v4's ImageInfiniteLight::PDF_Li: return pdf / (4 * Pi);
    return map_pdf / (4f0 * Float32(π))
end

# Fallback
@propagate_inbounds pdf_li_spectral(::Light, ::Point3f, ::Vec3f) = 0f0

# ============================================================================
# Light StaticMultiTypeSet Dispatch (uses with_index for type-stable dispatch)
# ============================================================================

# flat_to_light_index is defined in lights/light-sampler.jl (included earlier)

# Helper for argument reordering: with_index calls f(element, args...)
# but sample_light_spectral expects (table, lights, light, p, lambda, u)
# We pass lights through so EnvironmentLight can deref TextureRef fields
@propagate_inbounds _sample_light_spectral(light, lights, table, p, lambda, u) =
    sample_light_spectral(table, lights, light, p, lambda, u)

"""
    sample_light_spectral(table, lights::StaticMultiTypeSet, idx::SetKey, p, lambda, u)

Sample a light from a StaticMultiTypeSet using type-stable dispatch via with_index.
"""
@propagate_inbounds function sample_light_spectral(
    table::RGBToSpectrumTable,
    lights::Raycore.StaticMultiTypeSet,
    idx::SetKey,
    p::Point3f,
    lambda::Wavelengths,
    u::Point2f
)
    return with_index(_sample_light_spectral, lights, idx, lights, table, p, lambda, u)
end

"""
    sample_light_spectral(table, lights::StaticMultiTypeSet, flat_idx::Int32, p, lambda, u)

Sample a light from a StaticMultiTypeSet using a flat 1-based index.
Converts flat index to SetKey internally.
"""
@propagate_inbounds function sample_light_spectral(
    table::RGBToSpectrumTable,
    lights::Raycore.StaticMultiTypeSet,
    flat_idx::Int32,
    p::Point3f,
    lambda::Wavelengths,
    u::Point2f
)
    idx = flat_to_light_index(lights, flat_idx)
    return with_index(_sample_light_spectral, lights, idx, lights, table, p, lambda, u)
end
# ============================================================================
# Environment Light Evaluation (for escaped rays)
# All functions take rgb2spec_table for GPU-compatible spectral conversion
# ============================================================================

"""
    evaluate_environment_spectral(light::EnvironmentLight, lights, table, ray_d::Vec3f, lambda::Wavelengths)

Evaluate environment light for an escaped ray direction.
The `lights` parameter is used to deref TextureRef fields in EnvironmentMap.
"""
@propagate_inbounds function evaluate_environment_spectral(
    light::EnvironmentLight, lights, table::RGBToSpectrumTable, ray_d::Vec3f, lambda::Wavelengths
)::SpectralRadiance
    # Sample environment map by direction (env_map handles direction->UV internally)
    # Following pbrt-v4 ImageInfiniteLight::Le which passes direction directly
    # Pass lights for deref of TextureRef fields in EnvironmentMap
    Le_rgb = light.env_map(ray_d, lights) * light.scale

    # Use illuminant uplift for environment lights (matches pbrt's RGBIlluminantSpectrum)
    # This multiplies by D65 illuminant spectrum - critical for correct white reproduction
    return uplift_rgb_illuminant(table, Le_rgb, lambda)
end

"""
    evaluate_environment_spectral(light::SunSkyLight, lights, table, ray_d::Vec3f, lambda::Wavelengths)

Evaluate sun/sky light for an escaped ray direction.
"""
@propagate_inbounds function evaluate_environment_spectral(
    light::SunSkyLight, lights, table::RGBToSpectrumTable, ray_d::Vec3f, lambda::Wavelengths
)::SpectralRadiance
    # Get sky + sun radiance for direction (same as le() function)
    Le_rgb = sky_radiance(light, ray_d) + sun_disk_radiance(light, ray_d)
    # Unbounded uplift: sky model outputs RGB radiance (Preetham with scaling),
    # NOT illuminant quantities that need D65 normalization.
    return uplift_rgb_unbounded(table, Le_rgb, lambda)
end

"""
    evaluate_environment_spectral(light::AmbientLight, lights, table, ray_d::Vec3f, lambda::Wavelengths)

Evaluate ambient light for an escaped ray - provides constant radiance regardless of direction.
"""
@propagate_inbounds function evaluate_environment_spectral(
    light::AmbientLight, lights, table::RGBToSpectrumTable, ray_d::Vec3f, lambda::Wavelengths
)::SpectralRadiance
    # Matches pbrt-v4's UniformInfiniteLight::Le:
    # return scale * Lemit->Sample(lambda)
    return light.scale * Sample(table, light.i, lambda)
end

# Fallback - non-environment lights contribute nothing for escaped rays
@propagate_inbounds evaluate_environment_spectral(::Light, lights, ::RGBToSpectrumTable, ::Vec3f, ::Wavelengths) = SpectralRadiance(0f0)

"""
    evaluate_escaped_ray_spectral(table, lights::StaticMultiTypeSet, ray_d, lambda)

Evaluate all environment-type lights for an escaped ray using StaticMultiTypeSet.
"""
@propagate_inbounds function evaluate_escaped_ray_spectral(
    table::RGBToSpectrumTable, lights::Raycore.StaticMultiTypeSet, ray_d::Vec3f, lambda::Wavelengths
)::SpectralRadiance
    # Pass lights twice: first as collection to iterate, second as arg for deref
    return mapreduce(evaluate_environment_spectral, +, lights, lights, table, ray_d, lambda; init=SpectralRadiance(0f0))
end

# Helper to compute PDF from a single light (only EnvironmentLight has non-zero PDF)
# Takes lights container for deref of TextureRef fields in Distribution2D
@propagate_inbounds function _env_light_pdf_single(light::EnvironmentLight, lights, wi::Vec3f)::Float32
    return pdf_li_spectral(lights, light, Point3f(0f0, 0f0, 0f0), wi)
end

# SunSkyLight contributes to environment PDF via hemisphere importance sampling
@propagate_inbounds function _env_light_pdf_single(light::SunSkyLight, lights, wi::Vec3f)::Float32
    # Below horizon has zero probability
    if wi[3] <= 0f0
        return 0f0
    end
    uv = hemisphere_direction_to_uv(wi)
    map_pdf = pdf(light.distribution, uv, lights)
    θ = uv[2] * Float32(π) / 2f0
    sin_θ = sin(θ)
    return sin_θ > 0f0 ? map_pdf / (Float32(π) * Float32(π) * sin_θ) : 0f0
end

# Other light types don't contribute to environment PDF
@propagate_inbounds _env_light_pdf_single(::Light, lights, ::Vec3f)::Float32 = 0f0

"""
    compute_env_light_pdf(lights::StaticMultiTypeSet, ray_d::Vec3f)

Compute PDF for sampling direction from environment-type lights using StaticMultiTypeSet.
"""
@propagate_inbounds function compute_env_light_pdf(lights::Raycore.StaticMultiTypeSet, ray_d::Vec3f)::Float32
    return mapreduce(_env_light_pdf_single, +, lights, lights, ray_d; init=0f0)
end

# ============================================================================
# Power Heuristic for MIS
# ============================================================================

"""
    mis_weight_spectral(pdf_f::Float32, pdf_g::Float32) -> Float32

Compute MIS weight using power heuristic (beta=2).
Returns weight for strategy f: w_f = pdf_f^2 / (pdf_f^2 + pdf_g^2)
"""
@propagate_inbounds function mis_weight_spectral(pdf_f::Float32, pdf_g::Float32)
    if pdf_f <= 0f0
        return 0f0
    end
    f2 = pdf_f * pdf_f
    g2 = pdf_g * pdf_g
    return f2 / (f2 + g2 + 1f-10)
end

# ============================================================================
# Direct Lighting Helper
# ============================================================================

"""
    PWDirectLightingResult

Result of direct lighting calculation for one light sample.
"""
struct PWDirectLightingResult
    # Shadow ray info
    ray_origin::Point3f
    ray_direction::Vec3f
    t_max::Float32

    # Contribution (if unoccluded)
    Ld::SpectralRadiance

    # For MIS
    r_u::SpectralRadiance
    r_l::SpectralRadiance

    # Valid flag
    valid::Bool
end

@propagate_inbounds PWDirectLightingResult() = PWDirectLightingResult(
    Point3f(0f0, 0f0, 0f0),
    Vec3f(0f0, 0f0, 1f0),
    0f0,
    SpectralRadiance(0f0),
    SpectralRadiance(1f0),
    SpectralRadiance(1f0),
    false
)

"""
    compute_direct_lighting_spectral(p, n, wo, beta, r_u, lambda, light_sample, bsdf_f, bsdf_pdf)

Compute direct lighting contribution from a light sample with MIS.

Following pbrt-v4 (surfscatter.cpp lines 288-316):
- Ld = beta * f * Li * cos_theta (NO MIS weight or PDF division here)
- r_u = r_u * bsdfPDF (0 for delta lights)
- r_l = r_u * lightPDF
- MIS weighting happens at shadow ray resolution: Ld * T_ray / (r_u * tr_r_u + r_l * tr_r_l).Average()
"""
@propagate_inbounds function compute_direct_lighting_spectral(
    p::Point3f,
    n::Vec3f,
    wo::Vec3f,
    beta::SpectralRadiance,
    r_u::SpectralRadiance,
    lambda::Wavelengths,
    ls::PWLightSample,
    bsdf_f::SpectralRadiance,
    bsdf_pdf::Float32
)::PWDirectLightingResult
    # Check for valid light sample
    if ls.pdf <= 0f0 || is_black(ls.Li)
        return PWDirectLightingResult()
    end

    # Check for valid BSDF
    if is_black(bsdf_f)
        return PWDirectLightingResult()
    end

    # Compute cosine term
    cos_theta = abs(dot(ls.wi, n))

    # Following pbrt-v4: Ld = beta * f * Li * cos_theta
    # NO MIS weight or PDF division - that happens at shadow ray resolution
    Ld = beta * bsdf_f * ls.Li * cos_theta

    if is_black(Ld)
        return PWDirectLightingResult()
    end

    # Create shadow ray origin (offset to avoid self-intersection)
    offset = 1f-4 * n
    ray_origin = if dot(ls.wi, n) > 0f0
        Point3f((p + offset)...)
    else
        Point3f((p - offset)...)
    end

    # Distance to light (for t_max)
    to_light = ls.p_light - ray_origin
    t_max = sqrt(dot(to_light, to_light)) - 1f-3

    # MIS weights following pbrt-v4 (surfscatter.cpp lines 299-305):
    # bsdfPDF = 0 for delta lights (causes r_u to be zero, disabling BSDF MIS)
    # r_u = w.r_u * bsdfPDF
    # r_l = w.r_u * lightPDF
    new_bsdf_pdf = if ls.is_delta
        0f0  # Delta lights have no MIS with BSDF sampling
    else
        bsdf_pdf
    end
    new_r_u = r_u * new_bsdf_pdf
    new_r_l = r_u * ls.pdf

    return PWDirectLightingResult(
        ray_origin,
        ls.wi,
        t_max,
        Ld,
        new_r_u,
        new_r_l,
        true
    )
end
