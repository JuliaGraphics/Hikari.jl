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
    sample_light_spectral(table, light::PointLight, p::Point3f, lambda::Wavelengths, u::Point2f)

Sample a point light spectrally.
"""
@propagate_inbounds function sample_light_spectral(
    table::RGBToSpectrumTable, light::PointLight, p::Point3f, lambda::Wavelengths, ::Point2f
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
    # Li = I / r^2
    # Convert RGB intensity to spectral using illuminant uplift (matches pbrt's RGBIlluminantSpectrum)
    # This multiplies by D65 illuminant spectrum - critical for correct white reproduction
    Li_rgb = light.i / dist_sq
    Li = uplift_rgb_illuminant(table, Li_rgb, lambda)

    return PWLightSample(Li, wi, 1f0, light.position, true)
end

"""
    sample_light_spectral(table, light::SpotLight, p::Point3f, lambda::Wavelengths, u::Point2f)

Sample a spotlight spectrally.
"""
@propagate_inbounds function sample_light_spectral(
    table::RGBToSpectrumTable, light::SpotLight, p::Point3f, lambda::Wavelengths, ::Point2f
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

    # Li = I * falloff / r^2
    # Use illuminant uplift (matches pbrt's RGBIlluminantSpectrum)
    # This multiplies by D65 illuminant spectrum - critical for correct white reproduction
    Li_rgb = light.i * spot_falloff / dist_sq
    Li = uplift_rgb_illuminant(table, Li_rgb, lambda)

    return PWLightSample(Li, wi, 1f0, light.position, true)
end

"""
    sample_light_spectral(table, light::DirectionalLight, p::Point3f, lambda::Wavelengths, u::Point2f)

Sample a directional light spectrally.
"""
@propagate_inbounds function sample_light_spectral(
    table::RGBToSpectrumTable, light::DirectionalLight, p::Point3f, lambda::Wavelengths, ::Point2f
)::PWLightSample
    # Direction is opposite to light's travel direction
    wi = -light.direction

    # Distant lights have delta distribution
    # p_light is at "infinity" - use large distance for shadow ray
    p_light = Point3f(p + 1f6 * wi)

    # Use illuminant uplift (matches pbrt's RGBIlluminantSpectrum)
    # This multiplies by D65 illuminant spectrum - critical for correct white reproduction
    Li = uplift_rgb_illuminant(table, light.i, lambda)

    return PWLightSample(Li, wi, 1f0, p_light, true)
end

"""
    sample_light_spectral(table, light::SunLight, p::Point3f, lambda::Wavelengths, u::Point2f)

Sample a sun light spectrally.
"""
@propagate_inbounds function sample_light_spectral(
    table::RGBToSpectrumTable, light::SunLight, p::Point3f, lambda::Wavelengths, ::Point2f
)::PWLightSample
    # Direction is opposite to light's travel direction
    wi = -light.direction

    # Distant lights have delta distribution
    p_light = Point3f(p + 1f6 * wi)

    # Use illuminant uplift (matches pbrt's RGBIlluminantSpectrum)
    # This multiplies by D65 illuminant spectrum - critical for correct white reproduction
    Li = uplift_rgb_illuminant(table, light.l, lambda)

    return PWLightSample(Li, wi, 1f0, p_light, true)
end

"""
    sample_light_spectral(table, light::SunSkyLight, p::Point3f, lambda::Wavelengths, u::Point2f)

Sample sun direction from SunSkyLight spectrally.
"""
@propagate_inbounds function sample_light_spectral(
    table::RGBToSpectrumTable, light::SunSkyLight, p::Point3f, lambda::Wavelengths, ::Point2f
)::PWLightSample
    # Direction TO the sun
    wi = light.sun_direction

    # Delta distribution for sun disk
    p_light = Point3f(p + 1f6 * wi)

    # Sun radiance (field is sun_intensity)
    # Use illuminant uplift (matches pbrt's RGBIlluminantSpectrum)
    # This multiplies by D65 illuminant spectrum - critical for correct white reproduction
    Li = uplift_rgb_illuminant(table, light.sun_intensity, lambda)

    return PWLightSample(Li, wi, 1f0, p_light, true)
end

"""
    sample_light_spectral(table, light::EnvironmentLight, p::Point3f, lambda::Wavelengths, u::Point2f)

Sample environment light spectrally with importance sampling.
"""
@propagate_inbounds function sample_light_spectral(
    table::RGBToSpectrumTable, light::EnvironmentLight, p::Point3f, lambda::Wavelengths, u::Point2f
)::PWLightSample
    # Importance sample the environment map based on luminance
    uv, map_pdf = sample_continuous(light.env_map.distribution, u)

    # Convert UV to direction
    wi = uv_to_direction(uv, light.env_map.rotation)

    # PDF in solid angle
    sin_theta = sin(Float32(π) * uv[2])
    if sin_theta < 1f-6
        return PWLightSample()
    end
    pdf = map_pdf / (2f0 * Float32(π) * Float32(π) * sin_theta)

    if pdf <= 0f0
        return PWLightSample()
    end

    # Sample environment map color at UV (using lookup_uv like pbrt-v4's ImageLe)
    Li_rgb = lookup_uv(light.env_map, uv) * light.scale

    # p_light at infinity
    p_light = Point3f(p + 1f6 * wi)

    # Use illuminant uplift (matches pbrt's RGBIlluminantSpectrum)
    # This multiplies by D65 illuminant spectrum - critical for correct white reproduction
    Li = uplift_rgb_illuminant(table, Li_rgb, lambda)

    return PWLightSample(Li, wi, pdf, p_light, false)
end

"""
    sample_light_spectral(table, light::AmbientLight, p::Point3f, lambda::Wavelengths, u::Point2f)

Sample ambient light spectrally (uniform sphere).
NOTE: Ambient light represents uniform illumination from all directions.
We sample the full sphere uniformly - the BSDF evaluation will naturally
give zero for directions below the surface, and cos_theta weighting handles the rest.
"""
@propagate_inbounds function sample_light_spectral(
    table::RGBToSpectrumTable, light::AmbientLight, p::Point3f, lambda::Wavelengths, u::Point2f
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

    # Use illuminant uplift (matches pbrt's RGBIlluminantSpectrum)
    # This multiplies by D65 illuminant spectrum - critical for correct white reproduction
    Li = uplift_rgb_illuminant(table, light.i, lambda)

    return PWLightSample(Li, wi, pdf, p_light, false)
end

# Fallback for unknown light types
@propagate_inbounds function sample_light_spectral(
    ::RGBToSpectrumTable, ::Light, ::Point3f, ::Wavelengths, ::Point2f
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
    pdf_li_spectral(light::EnvironmentLight, p::Point3f, wi::Vec3f)

PDF for sampling direction wi from environment light.
"""
@propagate_inbounds function pdf_li_spectral(light::EnvironmentLight, ::Point3f, wi::Vec3f)
    # Convert direction to UV
    uv = direction_to_uv(wi, light.env_map.rotation)

    # Get PDF from distribution
    map_pdf = pdf(light.env_map.distribution, uv)

    # Convert to solid angle measure
    sin_theta = sin(Float32(π) * uv[2])
    if sin_theta < 1f-6
        return 0f0
    end

    return map_pdf / (2f0 * Float32(π) * Float32(π) * sin_theta)
end

# Fallback
@propagate_inbounds pdf_li_spectral(::Light, ::Point3f, ::Vec3f) = 0f0

# ============================================================================
# Light Tuple Dispatch (GPU-Safe with Unrolled If-Branches)
# ============================================================================

@propagate_inbounds @generated function sample_light_from_tuple(
    table::RGBToSpectrumTable, lights::L, idx::Int32, p::Point3f, lambda::Wavelengths, u::Point2f
) where {L <: Tuple}
    N = length(L.parameters)

    if N == 0
        return :(PWLightSample())
    end

    # Build unrolled if-else chain - each branch calls sample_light_spectral directly
    expr = :(@inbounds sample_light_spectral(table, lights[$N], p, lambda, u))

    for i in (N-1):-1:1
        expr = quote
            if idx == Int32($i)
                @inbounds sample_light_spectral(table, lights[$i], p, lambda, u)
            else
                $expr
            end
        end
    end

    return expr
end

"""
    count_lights(lights::Tuple) -> Int32

Count total number of lights in a tuple (recursively for nested structures).
"""
@propagate_inbounds count_lights(::NTuple{N, Any}) where {N} = Int32(N)

# ============================================================================
# Environment Light Evaluation (for escaped rays)
# All functions take rgb2spec_table for GPU-compatible spectral conversion
# ============================================================================

"""
    evaluate_environment_spectral(table, light::EnvironmentLight, ray_d::Vec3f, lambda::Wavelengths)

Evaluate environment light for an escaped ray direction.
"""
@propagate_inbounds function evaluate_environment_spectral(
    table::RGBToSpectrumTable, light::EnvironmentLight, ray_d::Vec3f, lambda::Wavelengths
)::SpectralRadiance
    # Sample environment map by direction (env_map handles direction->UV internally)
    # Following pbrt-v4 ImageInfiniteLight::Le which passes direction directly
    Le_rgb = light.env_map(ray_d) * light.scale

    # Use illuminant uplift for environment lights (matches pbrt's RGBIlluminantSpectrum)
    # This multiplies by D65 illuminant spectrum - critical for correct white reproduction
    return uplift_rgb_illuminant(table, Le_rgb, lambda)
end

"""
    evaluate_environment_spectral(table, light::SunSkyLight, ray_d::Vec3f, lambda::Wavelengths)

Evaluate sun/sky light for an escaped ray direction.
"""
@propagate_inbounds function evaluate_environment_spectral(
    table::RGBToSpectrumTable, light::SunSkyLight, ray_d::Vec3f, lambda::Wavelengths
)::SpectralRadiance
    # Get sky + sun radiance for direction (same as le() function)
    Le_rgb = sky_radiance(light, ray_d) + sun_disk_radiance(light, ray_d)

    # Use illuminant uplift (matches pbrt's RGBIlluminantSpectrum)
    # This multiplies by D65 illuminant spectrum - critical for correct white reproduction
    return uplift_rgb_illuminant(table, Le_rgb, lambda)
end

"""
    evaluate_environment_spectral(table, light::AmbientLight, ray_d::Vec3f, lambda::Wavelengths)

Evaluate ambient light for an escaped ray - provides constant radiance regardless of direction.
"""
@propagate_inbounds function evaluate_environment_spectral(
    table::RGBToSpectrumTable, light::AmbientLight, ray_d::Vec3f, lambda::Wavelengths
)::SpectralRadiance
    # Use illuminant uplift (matches pbrt's RGBIlluminantSpectrum)
    # This multiplies by D65 illuminant spectrum - critical for correct white reproduction
    return uplift_rgb_illuminant(table, light.i, lambda)
end

# Fallback - non-environment lights contribute nothing for escaped rays
@propagate_inbounds evaluate_environment_spectral(::RGBToSpectrumTable, ::Light, ::Vec3f, ::Wavelengths) = SpectralRadiance(0f0)

"""
    evaluate_escaped_ray_spectral(table, lights::Tuple, ray_d::Vec3f, lambda::Wavelengths)

Evaluate all environment-type lights for an escaped ray.
Returns total spectral radiance from infinite lights.
"""
@propagate_inbounds evaluate_escaped_ray_spectral(::RGBToSpectrumTable, ::Tuple{}, ::Vec3f, ::Wavelengths) = SpectralRadiance(0f0)

@propagate_inbounds function evaluate_escaped_ray_spectral(
    table::RGBToSpectrumTable, lights::Tuple, ray_d::Vec3f, lambda::Wavelengths
)::SpectralRadiance
    first_Le = evaluate_environment_spectral(table, first(lights), ray_d, lambda)
    rest_Le = evaluate_escaped_ray_spectral(table, Base.tail(lights), ray_d, lambda)
    return first_Le + rest_Le
end

"""
    compute_env_light_pdf(lights::Tuple, ray_d::Vec3f)

Compute PDF for sampling direction ray_d from environment-type lights.
Used for MIS weighting in escaped ray handling.
Following pbrt-v4: only environment/infinite lights contribute.
"""
@propagate_inbounds compute_env_light_pdf(::Tuple{}, ::Vec3f)::Float32 = 0f0

@propagate_inbounds function compute_env_light_pdf(lights::Tuple, ray_d::Vec3f)::Float32
    # PDF from first light (only environment lights contribute)
    first_pdf = _env_light_pdf_single(first(lights), ray_d)
    rest_pdf = compute_env_light_pdf(Base.tail(lights), ray_d)
    # Sum PDFs - for MIS we typically have one dominant env light
    return first_pdf + rest_pdf
end

# Helper to compute PDF from a single light (only EnvironmentLight has non-zero PDF)
@propagate_inbounds function _env_light_pdf_single(light::EnvironmentLight, wi::Vec3f)::Float32
    return pdf_li_spectral(light, Point3f(0f0, 0f0, 0f0), wi)
end

# Other light types don't contribute to environment PDF
@propagate_inbounds _env_light_pdf_single(::Light, ::Vec3f)::Float32 = 0f0

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
