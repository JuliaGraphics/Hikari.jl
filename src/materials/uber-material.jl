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
    # if f.type === FRESNEL_CONDUCTOR
    #     return fresnel_conductor(cos_θi, f.ηi, f.ηt, f.k)
    # else
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
    # error("Unknown BxDF type $(s.bxdf_type)")
end

struct UberMaterial{STAType,FTAType} <: Material
    """
    Spectral diffuse reflection value.
    """
    Kd::Texture{RGBSpectrum, 2, STAType}
    """
    Specular component. Spectrum texture.
    """
    Ks::Texture{RGBSpectrum,2,STAType}
    """
    Spectrum texture.
    """
    Kr::Texture{RGBSpectrum,2,STAType}
    """
    Spectrum texture.
    """
    Kt::Texture{RGBSpectrum,2,STAType}

    """
    Scalar roughness.
    """
    σ::Texture{Float32,2,FTAType}

    """
    Float texture
    """
    roughness::Texture{Float32,2,FTAType}
    """
    Float texture.
    """
    u_roughness::Texture{Float32,2,FTAType}

    """
    Float texture.
    """
    v_roughness::Texture{Float32,2,FTAType}
    """
    Float texture.
    """
    index::Texture{Float32,2,FTAType}

    remap_roughness::Bool

    type::UInt8
end

function UberMaterial(type;
        remap_roughness=false,
        args...
    )
    fields = [
        :Kd,
        :Ks,
        :Kr,
        :Kt,
        :σ,
        :roughness,
        :u_roughness,
        :v_roughness,
        :index
    ]

    FType = Matrix{Float32}
    SType = Matrix{RGBSpectrum}

    values = map(fields) do field
        if haskey(args, field)
            arg = args[field]
            if eltype(arg) === Float32
                FType = typeof(arg)
            elseif eltype(arg) === RGBSpectrum
                SType = typeof(arg)
            end
            return arg
        else
            NoTexture()
        end
    end
    return UberMaterial{SType,FType}(values..., remap_roughness, type)
end

const NO_MATERIAL = UInt8(0)

# GPU-compatible NoMaterial constructor - avoids dynamic allocation
function NoMaterial()
    # Create a fully static NoMaterial without using map/collect
    no_tex_s = NoTexture()
    no_tex_f = NoTexture()
    return UberMaterial{Matrix{RGBSpectrum}, Matrix{Float32}}(
        no_tex_s, no_tex_s, no_tex_s, no_tex_s,  # Kd, Ks, Kr, Kt
        no_tex_f, no_tex_f, no_tex_f, no_tex_f, no_tex_f,  # σ, roughness, u_roughness, v_roughness, index
        false,  # remap_roughness
        NO_MATERIAL  # type
    )
end

const NO_MAT = NoMaterial()


Base.Base.@propagate_inbounds function (m::UberMaterial)(si::SurfaceInteraction, allow_multiple_lobes::Bool, transport)
    if m.type === MATTE_MATERIAL
        return matte_material(m, si, allow_multiple_lobes, transport)
    elseif m.type === MIRROR_MATERIAL
        return mirror_material(m, si, allow_multiple_lobes, transport)
    elseif m.type === GLASS_MATERIAL
        return glass_material(m, si, allow_multiple_lobes, transport)
    elseif m.type === PLASTIC_MATERIAL
        return plastic_material(m, si, allow_multiple_lobes, transport)
    else
        return BSDF()
    end
end
