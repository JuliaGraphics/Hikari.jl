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

@inline PWLightSample() = PWLightSample(
    SpectralRadiance(0f0),
    Vec3f(0f0, 0f0, 1f0),
    0f0,
    Point3f(0f0, 0f0, 0f0),
    false
)

# ============================================================================
# Light Sampling for Each Light Type
# ============================================================================

"""
    sample_light_spectral(light::PointLight, p::Point3f, lambda::Wavelengths, u::Point2f)

Sample a point light spectrally.
"""
@inline function sample_light_spectral(
    light::PointLight, p::Point3f, lambda::Wavelengths, ::Point2f
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
    # Convert RGB intensity to spectral
    Li_rgb = light.i / dist_sq
    Li = uplift_rgb(Li_rgb, lambda)

    return PWLightSample(Li, wi, 1f0, light.position, true)
end

"""
    sample_light_spectral(light::SpotLight, p::Point3f, lambda::Wavelengths, u::Point2f)

Sample a spotlight spectrally.
"""
@inline function sample_light_spectral(
    light::SpotLight, p::Point3f, lambda::Wavelengths, ::Point2f
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
    Li_rgb = light.i * spot_falloff / dist_sq
    Li = uplift_rgb(Li_rgb, lambda)

    return PWLightSample(Li, wi, 1f0, light.position, true)
end

"""
    sample_light_spectral(light::DirectionalLight, p::Point3f, lambda::Wavelengths, u::Point2f)

Sample a directional light spectrally.
"""
@inline function sample_light_spectral(
    light::DirectionalLight, p::Point3f, lambda::Wavelengths, ::Point2f
)::PWLightSample
    # Direction is opposite to light's travel direction
    wi = -light.direction

    # Distant lights have delta distribution
    # p_light is at "infinity" - use large distance for shadow ray
    p_light = Point3f(p + 1f6 * wi)

    Li = uplift_rgb(light.l, lambda)

    return PWLightSample(Li, wi, 1f0, p_light, true)
end

"""
    sample_light_spectral(light::SunLight, p::Point3f, lambda::Wavelengths, u::Point2f)

Sample a sun light spectrally.
"""
@inline function sample_light_spectral(
    light::SunLight, p::Point3f, lambda::Wavelengths, ::Point2f
)::PWLightSample
    # Direction is opposite to light's travel direction
    wi = -light.direction

    # Distant lights have delta distribution
    p_light = Point3f(p + 1f6 * wi)

    Li = uplift_rgb(light.l, lambda)

    return PWLightSample(Li, wi, 1f0, p_light, true)
end

"""
    sample_light_spectral(light::SunSkyLight, p::Point3f, lambda::Wavelengths, u::Point2f)

Sample sun direction from SunSkyLight spectrally.
"""
@inline function sample_light_spectral(
    light::SunSkyLight, p::Point3f, lambda::Wavelengths, ::Point2f
)::PWLightSample
    # Direction TO the sun
    wi = light.sun_direction

    # Delta distribution for sun disk
    p_light = Point3f(p + 1f6 * wi)

    # Sun radiance
    Li = uplift_rgb(light.sun_radiance, lambda)

    return PWLightSample(Li, wi, 1f0, p_light, true)
end

"""
    sample_light_spectral(light::EnvironmentLight, p::Point3f, lambda::Wavelengths, u::Point2f)

Sample environment light spectrally with importance sampling.
"""
@inline function sample_light_spectral(
    light::EnvironmentLight, p::Point3f, lambda::Wavelengths, u::Point2f
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

    # Sample environment map color at UV
    Li_rgb = light.env_map(uv) * light.scale

    # p_light at infinity
    p_light = Point3f(p + 1f6 * wi)

    Li = uplift_rgb(Li_rgb, lambda)

    return PWLightSample(Li, wi, pdf, p_light, false)
end

"""
    sample_light_spectral(light::AmbientLight, p::Point3f, lambda::Wavelengths, u::Point2f)

Sample ambient light spectrally (uniform sphere).
NOTE: Ambient light represents uniform illumination from all directions.
We sample the full sphere uniformly - the BSDF evaluation will naturally
give zero for directions below the surface, and cos_theta weighting handles the rest.
"""
@inline function sample_light_spectral(
    light::AmbientLight, p::Point3f, lambda::Wavelengths, u::Point2f
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

    Li = uplift_rgb(light.i, lambda)

    return PWLightSample(Li, wi, pdf, p_light, false)
end

# Fallback for unknown light types
@inline function sample_light_spectral(
    ::Light, ::Point3f, ::Wavelengths, ::Point2f
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
@inline pdf_li_spectral(::PointLight, ::Point3f, ::Vec3f) = 0f0

"""
    pdf_li_spectral(light::DirectionalLight, p::Point3f, wi::Vec3f)

PDF for sampling direction wi from directional light (always delta).
"""
@inline pdf_li_spectral(::DirectionalLight, ::Point3f, ::Vec3f) = 0f0

"""
    pdf_li_spectral(light::SunLight, p::Point3f, wi::Vec3f)

PDF for sampling direction wi from sun light (always delta).
"""
@inline pdf_li_spectral(::SunLight, ::Point3f, ::Vec3f) = 0f0

"""
    pdf_li_spectral(light::SpotLight, p::Point3f, wi::Vec3f)

PDF for sampling direction wi from spotlight (always delta).
"""
@inline pdf_li_spectral(::SpotLight, ::Point3f, ::Vec3f) = 0f0

"""
    pdf_li_spectral(light::EnvironmentLight, p::Point3f, wi::Vec3f)

PDF for sampling direction wi from environment light.
"""
@inline function pdf_li_spectral(light::EnvironmentLight, ::Point3f, wi::Vec3f)
    # Convert direction to UV
    uv = direction_to_uv(wi, light.env_map.rotation)

    # Get PDF from distribution
    map_pdf = pdf_continuous(light.env_map.distribution, uv)

    # Convert to solid angle measure
    sin_theta = sin(Float32(π) * uv[2])
    if sin_theta < 1f-6
        return 0f0
    end

    return map_pdf / (2f0 * Float32(π) * Float32(π) * sin_theta)
end

# Fallback
@inline pdf_li_spectral(::Light, ::Point3f, ::Vec3f) = 0f0

# ============================================================================
# Recursive Light Tuple Dispatch (Type-Stable)
# ============================================================================

"""
    sample_light_from_tuple(lights::Tuple, idx::Int32, p::Point3f, lambda::Wavelengths, u::Point2f)

Recursively sample from the appropriate light in a heterogeneous tuple.
"""
@inline sample_light_from_tuple(::Tuple{}, ::Int32, ::Point3f, ::Wavelengths, ::Point2f) =
    PWLightSample()

@inline function sample_light_from_tuple(
    lights::Tuple, idx::Int32, p::Point3f, lambda::Wavelengths, u::Point2f
)
    if idx == Int32(1)
        return sample_light_spectral(first(lights), p, lambda, u)
    else
        return sample_light_from_tuple(Base.tail(lights), idx - Int32(1), p, lambda, u)
    end
end

"""
    count_lights(lights::Tuple) -> Int32

Count total number of lights in a tuple (recursively for nested structures).
"""
@inline count_lights(::Tuple{}) = Int32(0)
@inline count_lights(lights::Tuple) = Int32(1) + count_lights(Base.tail(lights))

# ============================================================================
# Environment Light Evaluation (for escaped rays)
# ============================================================================

"""
    evaluate_environment_spectral(light::EnvironmentLight, ray_d::Vec3f, lambda::Wavelengths)

Evaluate environment light for an escaped ray direction.
"""
@inline function evaluate_environment_spectral(
    light::EnvironmentLight, ray_d::Vec3f, lambda::Wavelengths
)::SpectralRadiance
    # Convert direction to UV
    uv = direction_to_uv(ray_d, light.env_map.rotation)

    # Sample environment map
    Le_rgb = light.env_map(uv) * light.scale

    return uplift_rgb(Le_rgb, lambda)
end

"""
    evaluate_environment_spectral(light::SunSkyLight, ray_d::Vec3f, lambda::Wavelengths)

Evaluate sun/sky light for an escaped ray direction.
"""
@inline function evaluate_environment_spectral(
    light::SunSkyLight, ray_d::Vec3f, lambda::Wavelengths
)::SpectralRadiance
    # Get sky color for direction
    Le_rgb = sky_color(light, ray_d)

    return uplift_rgb(Le_rgb, lambda)
end

"""
    evaluate_environment_spectral(light::AmbientLight, ray_d::Vec3f, lambda::Wavelengths)

Evaluate ambient light for an escaped ray - provides constant radiance regardless of direction.
"""
@inline function evaluate_environment_spectral(
    light::AmbientLight, ray_d::Vec3f, lambda::Wavelengths
)::SpectralRadiance
    return uplift_rgb(light.i, lambda)
end

# Fallback - non-environment lights contribute nothing for escaped rays
@inline evaluate_environment_spectral(::Light, ::Vec3f, ::Wavelengths) = SpectralRadiance(0f0)

"""
    evaluate_escaped_ray_spectral(lights::Tuple, ray_d::Vec3f, lambda::Wavelengths)

Evaluate all environment-type lights for an escaped ray.
Returns total spectral radiance from infinite lights.
"""
@inline evaluate_escaped_ray_spectral(::Tuple{}, ::Vec3f, ::Wavelengths) = SpectralRadiance(0f0)

@inline function evaluate_escaped_ray_spectral(
    lights::Tuple, ray_d::Vec3f, lambda::Wavelengths
)::SpectralRadiance
    first_Le = evaluate_environment_spectral(first(lights), ray_d, lambda)
    rest_Le = evaluate_escaped_ray_spectral(Base.tail(lights), ray_d, lambda)
    return first_Le + rest_Le
end

# ============================================================================
# Power Heuristic for MIS
# ============================================================================

"""
    mis_weight_spectral(pdf_f::Float32, pdf_g::Float32) -> Float32

Compute MIS weight using power heuristic (beta=2).
Returns weight for strategy f: w_f = pdf_f^2 / (pdf_f^2 + pdf_g^2)
"""
@inline function mis_weight_spectral(pdf_f::Float32, pdf_g::Float32)
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

@inline PWDirectLightingResult() = PWDirectLightingResult(
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
"""
@inline function compute_direct_lighting_spectral(
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

    # MIS weight (power heuristic)
    # For delta lights, skip MIS (bsdf_pdf doesn't matter)
    weight = if ls.is_delta
        1f0
    else
        mis_weight_spectral(ls.pdf, bsdf_pdf)
    end

    # Ld = beta * f * Li * cos_theta * weight / pdf_light
    Ld = beta * bsdf_f * ls.Li * (cos_theta * weight / ls.pdf)

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

    # Update r_l for MIS tracking
    new_r_l = if ls.is_delta
        SpectralRadiance(1f0)
    else
        SpectralRadiance(ls.pdf)
    end

    return PWDirectLightingResult(
        ray_origin,
        ls.wi,
        t_max,
        Ld,
        r_u,
        new_r_l,
        true
    )
end
