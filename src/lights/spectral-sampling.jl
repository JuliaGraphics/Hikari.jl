# Spectral light sampling interface
# Wraps Hikari's existing light types and converts to spectral domain

# ============================================================================
# Spectral Light Sample Result
# ============================================================================

"""
    LightSampleSpectral

Result of sampling a light source with spectral radiance.
"""
struct LightSampleSpectral
    Li::SpectralRadiance   # Spectral incident radiance
    wi::Vec3f              # Direction to light
    pdf::Float32           # Probability density
    p_light::Point3f       # Point on light (for shadow ray)
    is_delta::Bool         # True for point/distant lights (no MIS needed)
end

@propagate_inbounds LightSampleSpectral() = LightSampleSpectral(
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

@propagate_inbounds function sample_light_spectral(
    light::PointLight, table::RGBToSpectrumTable, lights, p::Point3f, lambda::Wavelengths, ::Point2f
)::LightSampleSpectral
    to_light = light.position - p
    dist_sq = dot(to_light, to_light)
    dist = sqrt(dist_sq)
    if dist < 1f-6
        return LightSampleSpectral()
    end
    wi = to_light / dist
    Li = light.scale * Sample(table, light.i, lambda) / dist_sq
    return LightSampleSpectral(Li, wi, 1f0, light.position, true)
end

@propagate_inbounds function sample_light_spectral(
    light::SpotLight, table::RGBToSpectrumTable, lights, p::Point3f, lambda::Wavelengths, ::Point2f
)::LightSampleSpectral
    to_light = Vec3f(light.position - p)
    dist_sq = dot(to_light, to_light)
    dist = sqrt(dist_sq)
    if dist < 1f-6
        return LightSampleSpectral()
    end
    wi = to_light / dist
    wi_local = normalize(light.world_to_light(Vec3f(-wi)))
    cos_theta = wi_local[3]
    if cos_theta < light.cos_total_width
        return LightSampleSpectral()
    end
    spot_falloff = if cos_theta >= light.cos_falloff_start
        1f0
    else
        delta = (cos_theta - light.cos_total_width) /
                (light.cos_falloff_start - light.cos_total_width)
        delta * delta * (3f0 - 2f0 * delta)  # SmoothStep (pbrt-v4 math.h:273)
    end
    Li = light.scale * Sample(table, light.i, lambda) * spot_falloff / dist_sq
    return LightSampleSpectral(Li, wi, 1f0, light.position, true)
end

@propagate_inbounds function sample_light_spectral(
    light::DirectionalLight, table::RGBToSpectrumTable, lights, p::Point3f, lambda::Wavelengths, ::Point2f
)::LightSampleSpectral
    wi = -light.direction
    p_light = Point3f(p + 1f6 * wi)
    Li = light.scale * uplift_rgb_illuminant(table, light.i, lambda)
    return LightSampleSpectral(Li, wi, 1f0, p_light, true)
end

@propagate_inbounds function sample_light_spectral(
    light::SunLight, table::RGBToSpectrumTable, lights, p::Point3f, lambda::Wavelengths, u::Point2f
)::LightSampleSpectral
    wi = sun_disc_direction(light, u)
    p_light = Point3f(p + 1f6 * wi)
    Li = light.scale * uplift_rgb_illuminant(table, light.i, lambda)
    return LightSampleSpectral(Li, wi, 1f0, p_light, true)
end

"""
    sun_disc_direction(light::SunLight, u::Point2f) -> Vec3f

A direction toward a uniformly chosen point of the sun's disc, `light.angular_diameter`
across. The sun stays a delta light carrying its whole irradiance in each sample; only
its direction spreads, which is what turns hard shadow edges into penumbrae.
"""
@propagate_inbounds function sun_disc_direction(light::SunLight, u::Point2f)
    centre = -light.direction
    light.angular_diameter > 0f0 || return centre
    x, y = coordinate_system(centre)
    # Sampled in angle, not via cos θ: across the sun, cos θ spans only ~200 Float32 steps.
    θ = light.angular_diameter / 2f0 * sqrt(u[1])
    ϕ = 2f0 * Float32(π) * u[2]
    return normalize(centre * cos(θ) + (x * cos(ϕ) + y * sin(ϕ)) * sin(θ))
end

@propagate_inbounds function sample_light_spectral(
    light::EnvironmentLight, table::RGBToSpectrumTable, lights, p::Point3f, lambda::Wavelengths, u::Point2f
)::LightSampleSpectral
    uv, map_pdf = sample_continuous(light.env_map.distribution, u, lights)
    wi = uv_to_direction(uv, light.env_map.rotation)
    pdf = map_pdf / (4f0 * Float32(π))
    if pdf <= 0f0
        return LightSampleSpectral()
    end
    Li_rgb = lookup_uv(light.env_map, uv, lights)
    p_light = Point3f(p + 1f6 * wi)
    # Scale AFTER spectral uplift — sigmoid is nonlinear, so scale must be outside
    Li = light.scale.c[1] * uplift_rgb_illuminant(table, Li_rgb, lambda)
    return LightSampleSpectral(Li, wi, pdf, p_light, false)
end

@propagate_inbounds function sample_light_spectral(
    light::AmbientLight, table::RGBToSpectrumTable, lights, p::Point3f, lambda::Wavelengths, u::Point2f
)::LightSampleSpectral
    z = 1f0 - 2f0 * u[1]
    r = sqrt(max(0f0, 1f0 - z * z))
    phi = 2f0 * Float32(π) * u[2]
    wi = Vec3f(r * cos(phi), r * sin(phi), z)
    pdf = 1f0 / (4f0 * Float32(π))
    p_light = Point3f(p + 1f6 * wi)
    Li = light.scale * Sample(table, light.i, lambda)
    return LightSampleSpectral(Li, wi, pdf, p_light, false)
end

@propagate_inbounds function sample_light_spectral(
    light::DiffuseAreaLight, table::RGBToSpectrumTable, lights, p::Point3f, lambda::Wavelengths, u::Point2f
)::LightSampleSpectral
    b0, b1 = if u[1] < u[2]
        _b0 = u[1] / 2f0
        _b1 = u[2] - _b0
        (_b0, _b1)
    else
        _b1 = u[2] / 2f0
        _b0 = u[1] - _b1
        (_b0, _b1)
    end
    b2 = 1f0 - b0 - b1
    p_light = Point3f(
        b0 * Vec3f(light.vertices[1]) + b1 * Vec3f(light.vertices[2]) + b2 * Vec3f(light.vertices[3])
    )
    to_light = Vec3f(p_light - p)
    dist_sq = dot(to_light, to_light)
    if dist_sq < 1f-12
        return LightSampleSpectral()
    end
    dist = sqrt(dist_sq)
    wi = Vec3f(to_light / dist)
    cos_theta = abs(dot(Vec3f(light.normal), -wi))
    if cos_theta < 1f-6
        return LightSampleSpectral()
    end
    pdf = dist_sq / (cos_theta * light.area)
    uv_sample = Point2f(
        b0 * Vec2f(light.uv[1]) + b1 * Vec2f(light.uv[2]) + b2 * Vec2f(light.uv[3])
    )
    wo = Vec3f(-wi[1], -wi[2], -wi[3])
    Le = arealight_Le(light, lights, table, wo, Vec3f(light.normal), uv_sample, lambda)
    if is_black(Le)
        return LightSampleSpectral()
    end
    return LightSampleSpectral(Le, wi, pdf, p_light, false)
end

# Fallback for unknown light types
@propagate_inbounds function sample_light_spectral(
    ::Light, ::RGBToSpectrumTable, lights, ::Point3f, ::Wavelengths, ::Point2f
)::LightSampleSpectral
    return LightSampleSpectral()
end

# ============================================================================
# Light PDF Evaluation (for MIS)
# ============================================================================

@propagate_inbounds pdf_li_spectral(::PointLight, ::Point3f, ::Vec3f) = 0f0
@propagate_inbounds pdf_li_spectral(::DirectionalLight, ::Point3f, ::Vec3f) = 0f0
@propagate_inbounds pdf_li_spectral(::SunLight, ::Point3f, ::Vec3f) = 0f0
@propagate_inbounds pdf_li_spectral(::SpotLight, ::Point3f, ::Vec3f) = 0f0

@propagate_inbounds function pdf_li_spectral(lights, light::EnvironmentLight, ::Point3f, wi::Vec3f)
    uv = direction_to_uv(wi, light.env_map.rotation)
    map_pdf = pdf(light.env_map.distribution, uv, lights)
    return map_pdf / (4f0 * Float32(π))
end

@propagate_inbounds pdf_li_spectral(::AmbientLight, ::Point3f, ::Vec3f) = 1f0 / (4f0 * Float32(π))

@propagate_inbounds pdf_li_spectral(::Light, ::Point3f, ::Vec3f) = 0f0

# ============================================================================
# StaticMultiTypeSet Dispatch
# ============================================================================

@propagate_inbounds function sample_light_spectral(
    table::RGBToSpectrumTable,
    lights::Raycore.StaticMultiTypeSet,
    idx::SetKey,
    p::Point3f,
    lambda::Wavelengths,
    u::Point2f
)
    return with_index(sample_light_spectral, lights, idx, table, lights, p, lambda, u)
end

@propagate_inbounds function sample_light_spectral(
    table::RGBToSpectrumTable,
    lights::Raycore.StaticMultiTypeSet,
    flat_idx::Int32,
    p::Point3f,
    lambda::Wavelengths,
    u::Point2f
)
    idx = flat_to_light_index(lights, flat_idx)
    return with_index(sample_light_spectral, lights, idx, table, lights, p, lambda, u)
end

# ============================================================================
# Environment Light Evaluation (for escaped rays)
# ============================================================================

@propagate_inbounds function evaluate_environment_spectral(
    light::EnvironmentLight, lights, table::RGBToSpectrumTable, ray_d::Vec3f, lambda::Wavelengths
)::SpectralRadiance
    Le_rgb = light.env_map(ray_d, lights)
    return light.scale.c[1] * uplift_rgb_illuminant(table, Le_rgb, lambda)
end

@propagate_inbounds function evaluate_environment_spectral(
    light::AmbientLight, lights, table::RGBToSpectrumTable, ray_d::Vec3f, lambda::Wavelengths
)::SpectralRadiance
    return light.scale * Sample(table, light.i, lambda)
end

@propagate_inbounds evaluate_environment_spectral(::Light, lights, ::RGBToSpectrumTable, ::Vec3f, ::Wavelengths) = SpectralRadiance(0f0)

@propagate_inbounds function evaluate_escaped_ray_spectral(
    table::RGBToSpectrumTable, lights::Raycore.StaticMultiTypeSet, ray_d::Vec3f, lambda::Wavelengths
)::SpectralRadiance
    return mapreduce(evaluate_environment_spectral, +, lights, lights, table, ray_d, lambda; init=SpectralRadiance(0f0))
end

@propagate_inbounds function env_light_pdf_single(light::EnvironmentLight, lights, wi::Vec3f)::Float32
    return pdf_li_spectral(lights, light, Point3f(0f0, 0f0, 0f0), wi)
end

@propagate_inbounds env_light_pdf_single(::AmbientLight, lights, ::Vec3f)::Float32 = 1f0 / (4f0 * Float32(π))

@propagate_inbounds env_light_pdf_single(::Light, lights, ::Vec3f)::Float32 = 0f0

@propagate_inbounds function compute_env_light_pdf(lights::Raycore.StaticMultiTypeSet, ray_d::Vec3f)::Float32
    return mapreduce(env_light_pdf_single, +, lights, lights, ray_d; init=0f0)
end

# ============================================================================
# MIS and Direct Lighting
# ============================================================================

@propagate_inbounds function mis_weight_spectral(pdf_f::Float32, pdf_g::Float32)
    if pdf_f <= 0f0
        return 0f0
    end
    f2 = pdf_f * pdf_f
    g2 = pdf_g * pdf_g
    return f2 / (f2 + g2 + 1f-10)
end

"""
    DirectLightingResult

Result of direct lighting calculation for one light sample.
"""
struct DirectLightingResult
    ray_origin::Point3f
    ray_direction::Vec3f
    t_max::Float32
    Ld::SpectralRadiance
    r_u::SpectralRadiance
    r_l::SpectralRadiance
    valid::Bool
end

@propagate_inbounds DirectLightingResult() = DirectLightingResult(
    Point3f(0f0, 0f0, 0f0),
    Vec3f(0f0, 0f0, 1f0),
    0f0,
    SpectralRadiance(0f0),
    SpectralRadiance(1f0),
    SpectralRadiance(1f0),
    false
)

"""
    compute_direct_lighting_spectral(p, ng, ns, wo, beta, r_u, lambda, light_sample, bsdf_f, bsdf_pdf)

Compute direct lighting contribution from a light sample with MIS.
Following pbrt-v4 surfscatter.cpp lines 288-316 + interaction.h:
  - cos_theta = AbsDot(wi, shading.n)  — uses SHADING normal (bumped)
  - OffsetRayOrigin uses GEOMETRIC normal  — to dodge the real triangle plane

`ng` is the geometric (face) normal; `ns` is the (possibly bumped) shading normal.
"""
@propagate_inbounds function compute_direct_lighting_spectral(
    p::Point3f,
    ng::Vec3f,
    ns::Vec3f,
    wo::Vec3f,
    beta::SpectralRadiance,
    r_u::SpectralRadiance,
    lambda::Wavelengths,
    ls::LightSampleSpectral,
    bsdf_f::SpectralRadiance,
    bsdf_pdf::Float32
)::DirectLightingResult
    if ls.pdf <= 0f0 || is_black(ls.Li)
        return DirectLightingResult()
    end
    if is_black(bsdf_f)
        return DirectLightingResult()
    end
    cos_theta = abs(dot(ls.wi, ns))
    Ld = beta * bsdf_f * ls.Li * cos_theta
    if is_black(Ld)
        return DirectLightingResult()
    end
    offset = 1f-4 * ng
    ray_origin = if dot(ls.wi, ng) > 0f0
        Point3f((p + offset)...)
    else
        Point3f((p - offset)...)
    end
    to_light = ls.p_light - ray_origin
    t_max = sqrt(dot(to_light, to_light)) - 1f-3
    new_bsdf_pdf = if ls.is_delta
        0f0
    else
        bsdf_pdf
    end
    new_r_u = r_u * new_bsdf_pdf
    new_r_l = r_u * ls.pdf
    return DirectLightingResult(
        ray_origin,
        ls.wi,
        t_max,
        Ld,
        new_r_u,
        new_r_l,
        true
    )
end
