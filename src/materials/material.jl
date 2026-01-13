# Material Interface Implementation
# Provides compute_bsdf(), shade(), and sample_bounce() for each material type

# ============================================================================
# BSDF computation for each material type
# ============================================================================

"""
Compute BSDF for MatteMaterial.
"""
@propagate_inbounds function compute_bsdf(m::MatteMaterial, textures, si::SurfaceInteraction, ::Bool, transport)
    r = clamp(eval_tex(textures, m.Kd, si.uv))
    is_black(r) && return BSDF(si)
    σ = clamp(eval_tex(textures, m.σ, si.uv), 0f0, 90f0)
    lambertian = (σ ≈ 0.0f0)
    return BSDF(si, LambertianReflection(lambertian, r), OrenNayar(!lambertian, r, σ))
end

"""
Compute BSDF for MirrorMaterial.
"""
@propagate_inbounds function compute_bsdf(m::MirrorMaterial, textures, si::SurfaceInteraction, ::Bool, transport)
    r = clamp(eval_tex(textures, m.Kr, si.uv))
    return BSDF(si, SpecularReflection(!is_black(r), r, FresnelNoOp()))
end

"""
Compute BSDF for GlassMaterial.
"""
@propagate_inbounds function compute_bsdf(g::GlassMaterial, textures, si::SurfaceInteraction, allow_multiple_lobes::Bool, transport)
    η = eval_tex(textures, g.index, si.uv)
    u_roughness = eval_tex(textures, g.u_roughness, si.uv)
    v_roughness = eval_tex(textures, g.v_roughness, si.uv)

    r = clamp(eval_tex(textures, g.Kr, si.uv))
    t = clamp(eval_tex(textures, g.Kt, si.uv))
    r_black = is_black(r)
    t_black = is_black(t)
    r_black && t_black && return BSDF(si, η)

    is_specular = u_roughness ≈ 0f0 && v_roughness ≈ 0f0

    # For specular glass with multiple lobes, use FresnelSpecular
    if is_specular && allow_multiple_lobes
        return BSDF(si, η, FresnelSpecular(true, r, t, 1.0f0, η, transport))
    end

    if g.remap_roughness
        u_roughness = roughness_to_α(u_roughness)
        v_roughness = roughness_to_α(v_roughness)
    end

    # For rough glass, use FresnelMicrofacet which combines reflection and transmission
    if !is_specular && allow_multiple_lobes
        distribution = TrowbridgeReitzDistribution(u_roughness, v_roughness)
        return BSDF(si, η, FresnelMicrofacet(true, r, t, distribution, 1.0f0, η, transport))
    end

    # Fallback: separate BxDFs for specular-only or when multiple lobes not allowed
    distribution = is_specular ? TrowbridgeReitzDistribution() : TrowbridgeReitzDistribution(
        u_roughness, v_roughness,
    )
    fresnel = FresnelDielectric(1f0, η)
    return BSDF(
        si, η,
        SpecularReflection(!r_black && is_specular, r, fresnel),
        MicrofacetReflection(!r_black && !is_specular, r, distribution, fresnel, transport),
        SpecularTransmission(!t_black && is_specular, t, 1.0f0, η, transport),
        MicrofacetTransmission(!t_black && !is_specular, t, distribution, 1.0f0, η, transport)
    )
end

"""
Compute BSDF for PlasticMaterial.
"""
@propagate_inbounds function compute_bsdf(p::PlasticMaterial, textures, si::SurfaceInteraction, ::Bool, transport)
    # Initialize diffuse component
    kd = clamp(eval_tex(textures, p.Kd, si.uv))
    bsdf_1 = LambertianReflection(!is_black(kd), kd)
    # Initialize specular component
    ks = clamp(eval_tex(textures, p.Ks, si.uv))
    is_black(ks) && return BSDF(si, bsdf_1)
    # Create microfacet distribution for plastic material
    fresnel = FresnelDielectric(1.5f0, 1f0)
    rough = eval_tex(textures, p.roughness, si.uv)
    p.remap_roughness && (rough = roughness_to_α(rough))
    distribution = TrowbridgeReitzDistribution(rough, rough)
    bsdf_2 = MicrofacetReflection(true, ks, distribution, fresnel, transport)
    return BSDF(si, bsdf_1, bsdf_2)
end

"""
Compute BSDF for MetalMaterial - conductor with Fresnel reflectance.
"""
@propagate_inbounds function compute_bsdf(m::MetalMaterial, textures, si::SurfaceInteraction, ::Bool, transport)
    # Get material parameters
    eta = clamp(eval_tex(textures, m.eta, si.uv))
    k_val = clamp(eval_tex(textures, m.k, si.uv))
    rough = eval_tex(textures, m.roughness, si.uv)
    # Reflectance is a color tint that multiplies the Fresnel result
    r = clamp(eval_tex(textures, m.reflectance, si.uv))

    # Create Fresnel conductor (ni=1 for air, nt=eta, k=k_val)
    fresnel = FresnelConductor(RGBSpectrum(1f0), eta, k_val)

    # Check if effectively smooth BEFORE remapping (like pbrt's EffectivelySmooth)
    # User roughness < 0.01 is considered a perfect mirror
    is_smooth = rough < 0.01f0
    if is_smooth
        bsdf = SpecularReflection(true, r, fresnel)
        return BSDF(si, bsdf)
    else
        # Remap roughness to alpha for microfacet distribution
        m.remap_roughness && (rough = roughness_to_α(rough))
        distribution = TrowbridgeReitzDistribution(rough, rough)
        bsdf = MicrofacetReflection(true, r, distribution, fresnel, transport)
        return BSDF(si, bsdf)
    end
end

# ============================================================================
# Material Interface: shade() for each material type
# ============================================================================

# Non-allocating sum of le() over lights tuple (recursive for type stability)
@propagate_inbounds sum_light_le(lights::Tuple{}, ray) = RGBSpectrum(0f0)
@propagate_inbounds function sum_light_le(lights::Tuple, ray)
    return le(first(lights), ray) + sum_light_le(Base.tail(lights), ray)
end

@propagate_inbounds function shade_light(light, interaction, scene, hit_wo, bsdf, shading_n, beta)
    u_light = rand(Point2f)
    Li, wi, pdf, visibility = sample_li(light, interaction, u_light, scene)
    (is_black(Li) || pdf ≈ 0f0) && return RGBSpectrum(0f0)
    f = bsdf(hit_wo, wi)
    is_black(f) && return RGBSpectrum(0f0)
    !unoccluded(visibility, scene) && return RGBSpectrum(0f0)
    cos_theta = abs(wi ⋅ shading_n)
    return beta * f * Li * cos_theta / pdf
end

# Type-stable recursive light shading (handles heterogeneous tuple)
@propagate_inbounds shade_lights(::Tuple{}, interaction, scene, hit_wo, bsdf, shading_n, beta) = RGBSpectrum(0f0)
@propagate_inbounds function shade_lights(lights::Tuple, interaction, scene, hit_wo, bsdf, shading_n, beta)
    first_contrib = shade_light(first(lights), interaction, scene, hit_wo, bsdf, shading_n, beta)
    rest_contrib = shade_lights(Base.tail(lights), interaction, scene, hit_wo, bsdf, shading_n, beta)
    return first_contrib + rest_contrib
end

@propagate_inbounds specular_type(::Reflect) = BSDF_REFLECTION | BSDF_SPECULAR
@propagate_inbounds specular_type(::Transmit) = BSDF_TRANSMISSION | BSDF_SPECULAR

"""
    specular_bounce(type, bsdf, ray, si, scene, beta, depth, max_depth) -> RGBSpectrum

Compute specular reflection or transmission contribution by tracing a bounce ray.
"""
@propagate_inbounds function specular_bounce(type, bsdf::BSDF, ray::RayDifferentials, si::SurfaceInteraction,
                                  scene::S, beta::RGBSpectrum, depth::Int32, max_depth::Int32) where {S<:AbstractScene}
    wo = si.core.wo
    ns = si.shading.n

    # Sample specular direction from BSDF
    u = rand(Point2f)
    wi, f, pdf, sampled_type = sample_f(bsdf, wo, u, specular_type(type))

    # Check for valid sample
    if !(pdf > 0f0 && !is_black(f) && abs(wi ⋅ ns) != 0f0)
        return RGBSpectrum(0f0)
    end

    # Compute throughput for this bounce
    bounce_beta = beta * f * abs(wi ⋅ ns) / pdf

    # Spawn bounce ray
    bounce_ray = RayDifferentials(spawn_ray(si, wi))

    # Add ray differentials if available - use sampled_type to determine reflect vs transmit
    if ray.has_differentials
        is_reflection = (sampled_type & BSDF_REFLECTION) != 0
        diff_type = is_reflection ? Reflect() : Transmit()
        bounce_ray = specular_differentials(diff_type, bounce_ray, bsdf, si, ray, wo, wi)
    end

    # Trace bounce ray through scene
    hit, primitive, bounce_si = intersect!(scene, bounce_ray)

    if !hit
        result = sum_light_le(scene.lights, bounce_ray)
        return bounce_beta * result
    end

    # Recursively shade the hit point
    return shade_material(
        scene.aggregate.materials, primitive.metadata,
        bounce_ray, bounce_si, scene, bounce_beta, depth + Int32(1), max_depth
    )
end

"""
Compute ray differentials for specular reflection.
"""
@propagate_inbounds function specular_differentials(::Reflect, rd, bsdf, si, ray, wo, wi)
    ns = si.shading.n
    rx_origin = si.core.p + si.∂p∂x
    ry_origin = si.core.p + si.∂p∂y
    # Compute differential reflected directions
    ∂n∂x = si.shading.∂n∂u * si.∂u∂x + si.shading.∂n∂v * si.∂v∂x
    ∂n∂y = si.shading.∂n∂u * si.∂u∂y + si.shading.∂n∂v * si.∂v∂y
    ∂wo∂x = -ray.rx_direction - wo
    ∂wo∂y = -ray.ry_direction - wo
    ∂dn∂x = ∂wo∂x ⋅ ns + wo ⋅ ∂n∂x
    ∂dn∂y = ∂wo∂y ⋅ ns + wo ⋅ ∂n∂y
    rx_direction = wi - ∂wo∂x + 2f0 * (wo ⋅ ns) * ∂n∂x + ∂dn∂x * ns
    ry_direction = wi - ∂wo∂y + 2f0 * (wo ⋅ ns) * ∂n∂y + ∂dn∂y * ns
    return RayDifferentials(rd, rx_origin=rx_origin, ry_origin=ry_origin, rx_direction=rx_direction, ry_direction=ry_direction)
end

"""
Compute ray differentials for specular transmission.
"""
@propagate_inbounds function specular_differentials(::Transmit, rd, bsdf, si, ray, wo, wi)
    ns = si.shading.n
    rx_origin = si.core.p + si.∂p∂x
    ry_origin = si.core.p + si.∂p∂y
    # Compute differential transmitted directions
    ∂n∂x = si.shading.∂n∂u * si.∂u∂x + si.shading.∂n∂v * si.∂v∂x
    ∂n∂y = si.shading.∂n∂u * si.∂u∂y + si.shading.∂n∂v * si.∂v∂y
    # The BSDF stores the IOR of the interior of the object
    η = 1f0 / bsdf.η
    # Check if ray is exiting the object (wo on opposite side of normal)
    if (wo ⋅ ns) < 0f0
        η = 1f0 / η
        ∂n∂x, ∂n∂y, ns = -∂n∂x, -∂n∂y, -ns
    end
    ∂wo∂x = -ray.rx_direction - wo
    ∂wo∂y = -ray.ry_direction - wo
    ∂dn∂x = ∂wo∂x ⋅ ns + wo ⋅ ∂n∂x
    ∂dn∂y = ∂wo∂y ⋅ ns + wo ⋅ ∂n∂y
    μ = η * (wo ⋅ ns) - abs(wi ⋅ ns)
    ν = η - (η * η * (wo ⋅ ns)) / abs(wi ⋅ ns)
    ∂μ∂x = ν * ∂dn∂x
    ∂μ∂y = ν * ∂dn∂y
    rx_direction = wi - η * ∂wo∂x + μ * ∂n∂x + ∂μ∂x * ns
    ry_direction = wi - η * ∂wo∂y + μ * ∂n∂y + ∂μ∂y * ns
    return RayDifferentials(rd, rx_origin=rx_origin, ry_origin=ry_origin, rx_direction=rx_direction, ry_direction=ry_direction)
end

"""
    shade(material::Material, ray, si, scene, beta, depth, max_depth) -> RGBSpectrum

Compute direct lighting and specular bounces for a surface hit.
This is the generic implementation that works for all material types.
"""
@propagate_inbounds function shade(material::Material, ray::RayDifferentials, si::SurfaceInteraction,
                       scene::S, beta::RGBSpectrum, depth::Int32, max_depth::Int32) where {S<:AbstractScene}
    # Compute BSDF from material
    bsdf = compute_bsdf(material, si, false, Radiance)

    # Get hit info from SurfaceInteraction
    hit_p = si.core.p
    hit_wo = si.core.wo
    hit_n = si.core.n
    shading_n = si.shading.n

    # Compute direct lighting from all lights
    interaction = Interaction(hit_p, 0f0, hit_wo, hit_n)
    total = shade_lights(scene.lights, interaction, scene, hit_wo, bsdf, shading_n, beta)

    # Handle specular bounces if we haven't exceeded max depth
    if depth < max_depth
        # Specular reflection
        total += specular_bounce(Reflect(), bsdf, ray, si, scene, beta, depth, max_depth)
        # Specular transmission
        total += specular_bounce(Transmit(), bsdf, ray, si, scene, beta, depth, max_depth)
    end

    return total
end

# ============================================================================
# sample_bounce for path tracing
# ============================================================================

"""
    sample_bounce(material::Material, ray, si, scene, beta, depth) -> (should_bounce, new_ray, new_beta, new_depth)

Sample BSDF to generate a bounce ray for path continuation.
"""
@propagate_inbounds function sample_bounce(material::Material, ray::RayDifferentials, si::SurfaceInteraction,
                               scene::Scene, beta::RGBSpectrum, depth::Int32)
    # Compute BSDF
    bsdf = compute_bsdf(material, si, true, Radiance)

    hit_wo = si.core.wo
    shading_n = si.shading.n

    # Sample BSDF for bounce direction
    u_bsdf = rand(Point2f)
    wi, f, pdf, sampled_type = sample_f(bsdf, hit_wo, u_bsdf, BSDF_ALL)

    # Check if valid sample
    if pdf > 0f0 && !is_black(f)
        # Update throughput
        cos_theta = abs(wi ⋅ shading_n)
        new_beta = beta * f * cos_theta / pdf

        # Russian roulette for path termination (after first few bounces)
        terminate = false
        if depth > Int32(2)
            q = max(0.05f0, 1f0 - to_Y(new_beta))
            if rand(Float32) < q
                terminate = true
            else
                new_beta = new_beta / (1f0 - q)
            end
        end

        if !terminate
            # Spawn bounce ray
            bounce_ray = RayDifferentials(spawn_ray(si, wi))
            return (true, bounce_ray, new_beta, depth + Int32(1))
        end
    end

    # No valid bounce
    dummy_ray = RayDifferentials(Ray(Point3f(0), Vec3f(0, 0, 1), Inf32, 0f0))
    return (false, dummy_ray, RGBSpectrum(0f0), Int32(0))
end
