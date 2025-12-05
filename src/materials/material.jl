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
