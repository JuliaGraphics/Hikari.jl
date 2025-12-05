const LAMBERTIAN_REFLECTION = UInt8(4)

function LambertianReflection(active::Bool, r::S) where {S<:Spectrum}
    UberBxDF{S}(active, LAMBERTIAN_REFLECTION; r=r, type=BSDF_DIFFUSE | BSDF_REFLECTION)
end

const LAMENTIAN_TRANSMISSION = UInt8(5)

function LambertianTransmission(active::Bool, t::S) where {S<:Spectrum}
    UberBxDF{S}(active, LAMENTIAN_TRANSMISSION; t=t, type=BSDF_DIFFUSE | BSDF_TRANSMISSION)
end

"""
Reflection distribution is constant and divides reflectance spectrum
equally over the hemisphere.
"""
function distribution_lambertian_reflection(l::UberBxDF{S}, ::Vec3f, ::Vec3f)::S where {S<:Spectrum}
    l.r * (1f0 / π)
end

"""
Directional-hemisphirical reflectance value is constant.
"""
function ρ_lambertian_reflection(
        l::UberBxDF{S}, ::Vec3f, ::Int32, ::Vector{Point2f},
    ) where S<:Spectrum

    l.r
end

"""
Hemispherical-hemisphirical reflectance value is constant.
"""
function ρ_lambertian_reflection(
        l::UberBxDF{S}, ::Vector{Point2f}, ::Vector{Point2f},
    ) where S<:Spectrum

    l.r
end

function distribution_lambertian_transmission(t::UberBxDF{S}, ::Vec3f, ::Vec3f)::S where {S<:Spectrum}
    t.t * (1f0 / π)
end

function ρ_lambertian_transmission(
    t::UberBxDF{S}, ::Vec3f, ::Int32, ::Vector{Point2f},
) where S<:Spectrum

    t.t
end

function ρ_lambertian_transmission(
    t::UberBxDF{S}, ::Vector{Point2f}, ::Vector{Point2f},
) where S<:Spectrum

    t.t
end

@inline function sample_lambertian_transmission(
        b::UberBxDF{S}, wo::Vec3f, sample::Point2f,
    )::Tuple{Vec3f,Float32,RGBSpectrum,UInt8} where S<:Spectrum

    wi = cosine_sample_hemisphere(sample)
    # Flipping the direction if necessary.
    wo[3] > 0 && (wi = Vec3f(wi[1], wi[2], -wi[3]))
    pdf = pdf_lambertian_transmission(b, wo, wi)
    return wi, pdf, b(wo, wi), UInt8(0)
end

@inline function pdf_lambertian_transmission(
        ::UberBxDF, wo::Vec3f, wi::Vec3f,
    )::Float32

    !same_hemisphere(wo, wi) ? abs(cos_θ(wi)) * (1f0 / π) : 0f0
end
