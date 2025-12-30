# BxDF Infrastructure - shared by all materials
# This file defines the low-level BxDF components used by all material types

abstract type MicrofacetDistribution end

"""
Microfacet distribution function based on Gaussian distribution of
microfacet slopes.
Distribution has higher tails, it falls off to zero more slowly for
directions far from the surface normal.
"""
struct TrowbridgeReitzDistribution <: MicrofacetDistribution
    α_x::Float32
    α_y::Float32
    sample_visible_area::Bool
    TrowbridgeReitzDistribution() = new(0f0, 0f0, false)
    function TrowbridgeReitzDistribution(
        α_x::Float32, α_y::Float32, sample_visible_area::Bool=true,
    )
        new(max(1.0f-3, α_x), max(1.0f-3, α_y), sample_visible_area)
    end
end


const FRESNEL_CONDUCTOR = UInt8(1)
const FRESNEL_DIELECTRIC = UInt8(2)
const FRESNEL_NO_OP = UInt8(3)

struct Fresnel
    ηi::RGBSpectrum
    ηt::RGBSpectrum
    k::RGBSpectrum
    type::UInt8
end

FresnelConductor(ni, nt, k) = Fresnel(ni, nt, k, FRESNEL_CONDUCTOR)
FresnelDielectric(ni::Float32, nt::Float32) = Fresnel(RGBSpectrum(ni), RGBSpectrum(nt), RGBSpectrum(0.0f0), FRESNEL_DIELECTRIC)
FresnelNoOp() = Fresnel(RGBSpectrum(0.0f0), RGBSpectrum(0.0f0), RGBSpectrum(0.0f0), FRESNEL_NO_OP)

function (f::Fresnel)(cos_θi::Float32)::Float32
    if f.type === FRESNEL_DIELECTRIC
        return fresnel_dielectric(cos_θi, f.ηi[1], f.ηt[1])
    end
    return 1f0
end


struct UberBxDF{S<:Spectrum}
    """
    Describes fresnel properties.
    """
    fresnel::Fresnel
    """
    Spectrum used to scale the reflected color.
    """
    r::S
    t::S

    a::Float32
    b::Float32
    """
    Index of refraction above the surface.
    Side the surface normal lies in is "above".
    """
    η_a::Float32
    """
    Index of refraction below the surface.
    Side the surface normal lies in is "above".
    """
    η_b::Float32

    distribution::TrowbridgeReitzDistribution

    transport::UInt8
    type::UInt8
    bxdf_type::UInt8
    active::Bool
end

function Base.:&(b::UberBxDF, type::UInt8)::Bool
    return b.active && ((b.type & type) == b.type)
end

UberBxDF{S}() where {S} = UberBxDF{S}(false, UInt8(0))

function UberBxDF{S}(active::Bool, bxdf_type::UInt8;
        r=RGBSpectrum(1f0), t=RGBSpectrum(1f0),
        a=0f0, b=0f0, η_a=0f0, η_b=0f0,
        distribution=TrowbridgeReitzDistribution(),
        fresnel=FresnelNoOp(),
        type=UInt8(0),
        transport=UInt8(0)
    ) where {S<:Spectrum}
    _distribution = distribution isa TrowbridgeReitzDistribution ? distribution : TrowbridgeReitzDistribution()
    return UberBxDF{S}(fresnel, r, t, a, b, η_a, η_b, _distribution, transport, type, bxdf_type, active)
end

@inline function sample_f(s::UberBxDF, wo::Vec3f, sample::Point2f)::Tuple{Vec3f,Float32,RGBSpectrum,UInt8}
    if s.bxdf_type === SPECULAR_REFLECTION
        return sample_specular_reflection(s, wo, sample)
    elseif s.bxdf_type === SPECULAR_TRANSMISSION
        return sample_specular_transmission(s, wo, sample)
    elseif s.bxdf_type === FRESNEL_SPECULAR
        return sample_fresnel_specular(s, wo, sample)
    elseif s.bxdf_type === LAMENTIAN_TRANSMISSION
        return sample_lambertian_transmission(s, wo, sample)
    elseif s.bxdf_type === MICROFACET_REFLECTION
        return sample_microfacet_reflection(s, wo, sample)
    elseif s.bxdf_type === MICROFACET_TRANSMISSION
        return sample_microfacet_transmission(s, wo, sample)
    elseif s.bxdf_type === FRESNEL_MICROFACET
        return sample_fresnel_microfacet(s, wo, sample)
    end
    wi::Vec3f = cosine_sample_hemisphere(sample)
    # Flipping the direction if necessary.
    wo[3] < 0 && (wi = Vec3f(wi[1], wi[2], -wi[3]))
    pdf::Float32 = compute_pdf(s, wo, wi)
    return wi, pdf, s(wo, wi), UInt8(0)
end

"""
Compute PDF value for the given directions.
In comparison, `sample_f` computes PDF value for the incident directions *it*
chooses given the outgoing direction, while this returns a value of PDF
for the given pair of directions.
"""
@inline function compute_pdf(s::UberBxDF, wo::Vec3f, wi::Vec3f)::Float32
    if s.bxdf_type === FRESNEL_SPECULAR
        return pdf_fresnel_specular(s, wo, wi)
    elseif s.bxdf_type === LAMENTIAN_TRANSMISSION
        return pdf_lambertian_transmission(s, wo, wi)
    elseif s.bxdf_type === MICROFACET_REFLECTION
        return pdf_microfacet_reflection(s, wo, wi)
    elseif s.bxdf_type === MICROFACET_TRANSMISSION
        return pdf_microfacet_transmission(s, wo, wi)
    elseif s.bxdf_type === FRESNEL_MICROFACET
        return pdf_fresnel_microfacet(s, wo, wi)
    end
    # Default fallback for Lambertian reflection (and other diffuse BxDFs)
    return same_hemisphere(wo, wi) ? abs(cos_θ(wi)) * (1.0f0 / π) : 0.0f0
end

@inline function (s::UberBxDF)(wo::Vec3f, wi::Vec3f)
    if s.bxdf_type === SPECULAR_REFLECTION
        return distribution_specular_reflection(s, wo, wi)
    elseif s.bxdf_type === SPECULAR_TRANSMISSION
        return distribution_specular_transmission(s, wo, wi)
    elseif s.bxdf_type === FRESNEL_SPECULAR
        return distribution_fresnel_specular(s, wo, wi)
    elseif s.bxdf_type === LAMBERTIAN_REFLECTION
        return distribution_lambertian_reflection(s, wo, wi)
    elseif s.bxdf_type === LAMENTIAN_TRANSMISSION
        return distribution_lambertian_transmission(s, wo, wi)
    elseif s.bxdf_type === OREN_NAYAR
        return distribution_orennayar(s, wo, wi)
    elseif s.bxdf_type === MICROFACET_REFLECTION
        return distribution_microfacet_reflection(s, wo, wi)
    elseif s.bxdf_type === MICROFACET_TRANSMISSION
        return distribution_microfacet_transmission(s, wo, wi)
    elseif s.bxdf_type === FRESNEL_MICROFACET
        return distribution_fresnel_microfacet(s, wo, wi)
    end
    return RGBSpectrum(0.0f0)
end

# ============================================================================
# Clean Material Types - each is a data container with only necessary parameters
# ============================================================================

"""
    MatteMaterial(Kd::Texture, σ::Texture)

Matte (diffuse) material with Lambertian or Oren-Nayar BRDF.

* `Kd`: Spectral diffuse reflection (color texture)
* `σ`: Scalar roughness for Oren-Nayar model (0 = Lambertian)
"""
struct MatteMaterial{KdType, σType} <: Material
    Kd::Texture{RGBSpectrum, 2, KdType}
    σ::Texture{Float32, 2, σType}
end

function MatteMaterial(Kd::Texture, σ::Texture)
    KdType = typeof(Kd.data)
    σType = typeof(σ.data)
    MatteMaterial{KdType, σType}(Kd, σ)
end

"""
    MirrorMaterial(Kr::Texture)

Perfect mirror (specular reflection) material.

* `Kr`: Spectral reflectance (color texture)
"""
struct MirrorMaterial{KrType} <: Material
    Kr::Texture{RGBSpectrum, 2, KrType}
end

function MirrorMaterial(Kr::Texture)
    KrType = typeof(Kr.data)
    MirrorMaterial{KrType}(Kr)
end

"""
    GlassMaterial(Kr, Kt, u_roughness, v_roughness, index, remap_roughness)

Glass/dielectric material with reflection and transmission.

* `Kr`: Spectral reflectance
* `Kt`: Spectral transmittance
* `u_roughness`: Roughness in u direction (0 = perfect specular)
* `v_roughness`: Roughness in v direction (0 = perfect specular)
* `index`: Index of refraction
* `remap_roughness`: Whether to remap roughness to alpha
"""
struct GlassMaterial{KrType, KtType, RoughType, IndexType} <: Material
    Kr::Texture{RGBSpectrum, 2, KrType}
    Kt::Texture{RGBSpectrum, 2, KtType}
    u_roughness::Texture{Float32, 2, RoughType}
    v_roughness::Texture{Float32, 2, RoughType}
    index::Texture{Float32, 2, IndexType}
    remap_roughness::Bool
end

function GlassMaterial(
    Kr::Texture, Kt::Texture,
    u_roughness::Texture, v_roughness::Texture,
    index::Texture, remap_roughness::Bool
)
    KrType = typeof(Kr.data)
    KtType = typeof(Kt.data)
    RoughType = typeof(u_roughness.data)
    IndexType = typeof(index.data)
    GlassMaterial{KrType, KtType, RoughType, IndexType}(
        Kr, Kt, u_roughness, v_roughness, index, remap_roughness
    )
end

"""
    PlasticMaterial(Kd, Ks, roughness, remap_roughness)

Plastic material with diffuse and glossy specular components.

* `Kd`: Diffuse reflectance
* `Ks`: Specular reflectance
* `roughness`: Surface roughness
* `remap_roughness`: Whether to remap roughness to alpha
"""
struct PlasticMaterial{KdType, KsType, RoughType} <: Material
    Kd::Texture{RGBSpectrum, 2, KdType}
    Ks::Texture{RGBSpectrum, 2, KsType}
    roughness::Texture{Float32, 2, RoughType}
    remap_roughness::Bool
end

function PlasticMaterial(
    Kd::Texture, Ks::Texture,
    roughness::Texture, remap_roughness::Bool
)
    KdType = typeof(Kd.data)
    KsType = typeof(Ks.data)
    RoughType = typeof(roughness.data)
    PlasticMaterial{KdType, KsType, RoughType}(Kd, Ks, roughness, remap_roughness)
end
