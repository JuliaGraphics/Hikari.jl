const MATTE_MATERIAL = Radiance


"""
    MatteMaterial(Kd::Texture, σ::Texture)

* `Kd:` Spectral diffuse reflection value.
* `σ:` Scalar roughness.
"""
function MatteMaterial(
        Kd::Texture, σ::Texture,
    )
    return UberMaterial(MATTE_MATERIAL; Kd=Kd, σ=σ)
end


"""
Compute scattering function.
"""
Base.Base.@propagate_inbounds function matte_material(m::UberMaterial, si::SurfaceInteraction, ::Bool, transport)
    # TODO perform bump mapping
    # Evaluate textures and create BSDF.
    r = clamp(m.Kd(si))
    is_black(r) && return BSDF(si)
    σ = clamp(m.σ(si), 0f0, 90f0)
    lambertian = (σ ≈ 0.0f0)
    return BSDF(si, LambertianReflection(lambertian, r), OrenNayar(!lambertian, r, σ))
end

const MIRROR_MATERIAL = UInt8(2)

function MirrorMaterial(Kr::Texture)
    return UberMaterial(MIRROR_MATERIAL; Kr=Kr)
end

Base.Base.@propagate_inbounds function mirror_material(m::UberMaterial, si::SurfaceInteraction, ::Bool, transport)
    r = clamp(m.Kr(si))
    return BSDF(si, SpecularReflection(!is_black(r), r, FresnelNoOp()))
end

const GLASS_MATERIAL = UInt8(3)

function GlassMaterial(
        Kr::Texture, Kt::Texture, u_roughness::Texture, v_roughness::Texture, index::Texture,
        remap_roughness::Bool,
    )
    return UberMaterial(GLASS_MATERIAL; Kr=Kr, Kt=Kt, u_roughness=u_roughness, v_roughness=v_roughness, index=index, remap_roughness=remap_roughness)
end

Base.Base.@propagate_inbounds function glass_material(g::UberMaterial, si::SurfaceInteraction, allow_multiple_lobes::Bool, transport)

    η = g.index(si)
    u_roughness = g.u_roughness(si)
    v_roughness = g.v_roughness(si)

    r = clamp(g.Kr(si))
    t = clamp(g.Kt(si))
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
    # with Fresnel-weighted sampling (avoids uniform 50/50 split that wastes samples)
    if !is_specular && allow_multiple_lobes
        distribution = TrowbridgeReitzDistribution(u_roughness, v_roughness)
        # Use combined color for FresnelMicrofacet (r for reflection, t for transmission)
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

const PLASTIC_MATERIAL = UInt8(4)

function PlasticMaterial(
        Kd::Texture, Ks::Texture, roughness::Texture, remap_roughness::Bool,
    )
    return UberMaterial(PLASTIC_MATERIAL; Kd=Kd, Ks=Ks, roughness=roughness, remap_roughness=remap_roughness)
end

Base.Base.@propagate_inbounds function plastic_material(p::UberMaterial,
        si::SurfaceInteraction, ::Bool, transport,
    )
    # Initialize diffuse componen of plastic material.
    kd = clamp(p.Kd(si))
    bsdf_1 = LambertianReflection(!is_black(kd), kd)
    # Initialize specular component.
    ks = clamp(p.Ks(si))
    is_black(ks) && return BSDF(si, bsdf_1)
    # Create microfacet distribution for plastic material.
    fresnel = FresnelDielectric(1.5f0, 1f0)
    rough = p.roughness(si)
    p.remap_roughness && (rough = roughness_to_α(rough))
    distribution = TrowbridgeReitzDistribution(rough, rough)
    bsdf_2 = MicrofacetReflection(true, ks, distribution, fresnel, transport)
    return BSDF(si, bsdf_1, bsdf_2)
end

# ============================================================================
# Material Interface: shade() and sample_bounce() for UberMaterial
# ============================================================================


function shade_light(light, interaction, scene, hit_wo, bsdf, shading_n, beta)
    u_light = rand(Point2f)
    Li, wi, pdf, visibility = @inline Hikari.sample_li(light, interaction, u_light, scene)
    (is_black(Li) || pdf ≈ 0f0) && return RGBSpectrum(0f0)
    f = @inline bsdf(hit_wo, wi)
    is_black(f) && return RGBSpectrum(0f0)
    !unoccluded(visibility, scene) && return RGBSpectrum(0f0)
    cos_theta = abs(wi ⋅ shading_n)
    return beta * f * Li * cos_theta / pdf
end

"""
    shade(material::UberMaterial, ray, si, scene, beta, depth, max_depth) -> RGBSpectrum

Compute direct lighting and specular bounces for a surface hit with UberMaterial.
"""
function shade(material::UberMaterial, ray::RayDifferentials, si::SurfaceInteraction,
                       scene::Scene, beta::RGBSpectrum, depth::Int32, max_depth::Int32)
    # Compute BSDF from material
    bsdf = material(si, true, Radiance)

    # Get hit info from SurfaceInteraction
    hit_p = si.core.p
    hit_wo = si.core.wo
    hit_n = si.core.n
    shading_n = si.shading.n

    # Compute direct lighting from all lights
    interaction = Interaction(hit_p, 0f0, hit_wo, hit_n)
    total = shade_light(scene.lights[1], interaction, scene, hit_wo, bsdf, shading_n, beta)
    # Handle specular bounces if we haven't exceeded max depth
    if depth < max_depth
        # Specular reflection
        total += specular_bounce(Reflect(), bsdf, ray, si, scene, beta, depth, max_depth)
        # Specular transmission
        total += specular_bounce(Transmit(), bsdf, ray, si, scene, beta, depth, max_depth)
    end

    return total
end

@inline specular_type(::Reflect) = BSDF_REFLECTION | BSDF_SPECULAR
@inline specular_type(::Transmit) = BSDF_TRANSMISSION | BSDF_SPECULAR

"""
    specular_bounce(type, bsdf, ray, si, scene, beta, depth, max_depth) -> RGBSpectrum

Compute specular reflection or transmission contribution by tracing a bounce ray.
"""
@inline function specular_bounce(type, bsdf, ray::RayDifferentials, si::SurfaceInteraction,
                                  scene::Scene, beta::RGBSpectrum, depth::Int32, max_depth::Int32)
    wo = si.core.wo
    ns = si.shading.n

    # Sample specular direction from BSDF
    u = rand(Point2f)
    wi, f, pdf, sampled_type = @inline sample_f(bsdf, wo, u, specular_type(type))

    # Check for valid sample
    if !(pdf > 0f0 && !is_black(f) && abs(wi ⋅ ns) != 0f0)
        return RGBSpectrum(0f0)
    end

    # Compute throughput for this bounce
    bounce_beta = beta * f * abs(wi ⋅ ns) / pdf

    # Spawn bounce ray
    bounce_ray = RayDifferentials(spawn_ray(si, wi))

    # Add ray differentials if available
    if ray.has_differentials
        bounce_ray = specular_differentials(type, bounce_ray, bsdf, si, ray, wo, wi)
    end

    # Trace bounce ray through scene
    hit, primitive, bounce_si = intersect!(scene, bounce_ray)

    if !hit
        # No hit - get background from lights
        br = bounce_ray
        result = sum(map(l -> le(l, br), scene.lights))
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
@inline function specular_differentials(::Reflect, rd, bsdf, si, ray, wo, wi)
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
@inline function specular_differentials(::Transmit, rd, bsdf, si, ray, wo, wi)
    ns = si.shading.n
    rx_origin = si.core.p + si.∂p∂x
    ry_origin = si.core.p + si.∂p∂y
    # Compute differential transmitted directions
    ∂n∂x = si.shading.∂n∂u * si.∂u∂x + si.shading.∂n∂v * si.∂v∂x
    ∂n∂y = si.shading.∂n∂u * si.∂u∂y + si.shading.∂n∂v * si.∂v∂y
    # The BSDF stores the IOR of the interior of the object
    η = 1f0 / bsdf.η
    if (ns ⋅ ns) < 0f0
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
    sample_bounce(material::UberMaterial, ray, si, scene, beta, depth) -> (should_bounce, new_ray, new_beta, new_depth)

Sample BSDF to generate a bounce ray for path continuation.
"""
@inline function sample_bounce(material::UberMaterial, ray::RayDifferentials, si::SurfaceInteraction,
                               scene::Scene, beta::RGBSpectrum, depth::Int32)
    # Compute BSDF
    bsdf = material(si, true, Radiance)

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
