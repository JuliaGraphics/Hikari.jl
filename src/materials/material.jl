struct MatteMaterial <: Material
    """
    Spectral diffuse reflection value.
    """
    Kd::Texture  # TODO check that texture is spectral
    """
    Scalar roughness.
    """
    σ::Texture  # TODO check that texture is scalar
    # TODO bump map
end

"""
Compute scattering function.
"""
function (m::MatteMaterial)(
    si::SurfaceInteraction, ::Bool, ::Type{T},
) where T<:TransportMode
    # TODO perform bump mapping
    # Evaluate textures and create BSDF.
    si.bsdf = BSDF(si)
    r = clamp(m.Kd(si))
    is_black(r) && return

    σ = clamp(m.σ(si), 0f0, 90f0)
    if σ ≈ 0f0
        add!(si.bsdf, LambertianReflection(r))
    else
        add!(si.bsdf, OrenNayar(r, σ))
    end
end


struct MirrorMaterial <: Material
    Kr::Texture
    # TODO bump map
end

function (m::MirrorMaterial)(
    si::SurfaceInteraction, ::Bool, ::Type{T},
) where T<:TransportMode
    si.bsdf = BSDF(si)
    r = clamp(m.Kr(si))
    is_black(r) && return
    add!(si.bsdf, SpecularReflection(r, FresnelNoOp()))
end


struct GlassMaterial <: Material
    """
    Spectrum texture.
    """
    Kr::Texture
    """
    Spectrum texture.
    """
    Kt::Texture
    """
    Float texture.
    """
    u_roughness::Texture
    """
    Float texture.
    """
    v_roughness::Texture
    """
    Float texture.
    """
    index::Texture

    remap_roughness::Bool
    # TODO bump mapping
end

function (g::GlassMaterial)(
    si::SurfaceInteraction, allow_multiple_lobes::Bool, ::Type{T},
) where T<:TransportMode
    η = g.index(si)
    u_roughness = g.u_roughness(si)
    v_roughness = g.v_roughness(si)

    si.bsdf = BSDF(si, η)
    r = clamp(g.Kr(si))
    t = clamp(g.Kt(si))
    is_black(r) && is_black(t) && return

    is_specular = u_roughness ≈ 0 && v_roughness ≈ 0
    if is_specular && allow_multiple_lobes
        add!(si.bsdf, FresnelSpecular(r, t, 1f0, η, T))
        return
    end

    if g.remap_roughness
        u_roughness = roughness_to_α(u_roughness)
        v_roughness = roughness_to_α(v_roughness)
    end
    distribution = is_specular ? nothing : TrowbridgeReitzDistribution(
        u_roughness, v_roughness,
    )

    if !is_black(r)
        fresnel = FresnelDielectric(1f0, η)
        if is_specular
            add!(si.bsdf, SpecularReflection(r, fresnel))
        else
            add!(si.bsdf, MicrofacetReflection(r, distribution, fresnel, T))
        end
    end
    if !is_black(t)
        if is_specular
            add!(si.bsdf, SpecularTransmission(t, 1f0, η, T))
        else
            add!(si.bsdf, MicrofacetTransmission(t, distribution, 1f0, η, T))
        end
    end
end


struct PlasticMaterial <: Material
    """
    Diffuse component. Spectrum texture.
    """
    Kd::Texture
    """
    Specular component. Spectrum texture.
    """
    Ks::Texture
    """
    Float texture
    """
    roughness::Texture
    remap_roughness::Bool
end

function (p::PlasticMaterial)(
    si::SurfaceInteraction, ::Bool, ::Type{T},
) where T<:TransportMode
    si.bsdf = BSDF(si)
    # Initialize diffuse componen of plastic material.
    kd = clamp(p.Kd(si))
    !is_black(kd) && add!(si.bsdf, LambertianReflection(kd))
    # Initialize specular component.
    ks = clamp(p.Ks(si))
    is_black(ks) && return
    # Create microfacet distribution for plastic material.
    fresnel = FresnelDielectric(1.5f0, 1f0)
    rough = p.roughness(si)
    p.remap_roughness && (rough = roughness_to_α(rough))
    distribution = TrowbridgeReitzDistribution(rough, rough)
    add!(si.bsdf, MicrofacetReflection(ks, distribution, fresnel, T))
end
