const SPECULAR_REFLECTION = UInt8(1)

function SpecularReflection(active::Bool, r::S, fresnel::Fresnel, type=BSDF_SPECULAR | BSDF_REFLECTION) where {S}
    UberBxDF{S}(active, SPECULAR_REFLECTION; r=r, fresnel=fresnel, type=type)
end

const SPECULAR_TRANSMISSION = UInt8(2)

function SpecularTransmission(
        active::Bool, t::S, η_a::Float32, η_b::Float32, transport
    ) where {S<:Spectrum}

    UberBxDF{S}(
        active, SPECULAR_TRANSMISSION;
        t=t, η_a=η_a, η_b=η_b,
        fresnel=FresnelDielectric(η_a, η_b),
        type=BSDF_SPECULAR | BSDF_TRANSMISSION,
        transport=transport
    )
end

const FRESNEL_SPECULAR = UInt8(3)

function FresnelSpecular(
    active::Bool, r::S, t::S, η_a::Float32, η_b::Float32, transport
) where {S<:Spectrum}
    UberBxDF{S}(
        active, FRESNEL_SPECULAR;
        r=r, t=t, η_a=η_a, η_b=η_b,
        type=BSDF_SPECULAR | BSDF_TRANSMISSION | BSDF_REFLECTION,
        transport=transport
    )
end

"""
Return value of the distribution function for the given pair of directions.
For specular reflection, no scattering is returned, since
for arbitrary directions δ-funcion returns no scattering.
"""
function distribution_specular_reflection(::UberBxDF{S}, ::Vec3f, ::Vec3f)::S where {S<:Spectrum}
    S(0f0)
end

"""
Compute the direction of incident light wi, given an outgoing direction wo
and return the value of BxDF for the pair of directions.
`sample` parameter isn't needed for the δ-distribution.
"""
@inline function sample_specular_reflection(
        s::UberBxDF{S}, wo::Vec3f, ::Point2f,
    )::Tuple{Vec3f,Float32,S,UInt8} where {S<:Spectrum}
    wi = Vec3f(-wo[1], -wo[2], wo[3])
    wisp = s.fresnel(cos_θ(wi)) * s.r / abs(cos_θ(wi))
    return wi, 1.0f0, wisp, UInt8(0)
end


"""
Return value of the distribution function for the given pair of directions.
For specular transmission, no scattering is returned, since
for arbitrary directions δ-funcion returns no scattering.
"""
function distribution_specular_transmission(::UberBxDF{S}, ::Vec3f, ::Vec3f)::S where {S<:Spectrum}
    S(0f0)
end

"""
Compute the direction of incident light wi, given an outgoing direction wo
and return the value of BxDF for the pair of directions.
`sample` parameter isn't needed for the δ-distribution.
"""
@inline function sample_specular_transmission(s::UberBxDF{S}, wo::Vec3f, ::Point2f)::Tuple{Vec3f,Float32,S,UInt8} where {S}

    # Figure out which η is incident and which is transmitted.
    entering = cos_θ(wo) > 0
    η_i = entering ? s.η_a : s.η_b
    η_t = entering ? s.η_b : s.η_a
    # Compute ray direction for specular transmission.
    valid, wi = refract(
        wo, face_forward(Normal3f(0f0, 0f0, 1f0), wo), η_i / η_t,
    )
    # Total internal reflection.
    !valid && return Vec3f(0f0), 0f0, S(0f0), UInt8(0)
    pdf = 1f0

    cos_wi = cos_θ(wi)
    ft = s.t * (S(1.0f0) - s.fresnel(cos_wi))
    # Account for non-symmetry with transmission to different medium.
    s.transport === Radiance && (ft *= (η_i^2) / (η_t^2))
    return wi, pdf, ft / abs(cos_wi), UInt8(0)
end

@inline function distribution_fresnel_specular(::UberBxDF{S}, ::Vec3f, ::Vec3f)::S where {S<:Spectrum}
    S(0f0)
end

@inline pdf_fresnel_specular(f::UberBxDF, wo::Vec3f, wi::Vec3f)::Float32 = 0.0f0

"""
Compute the direction of incident light wi, given an outgoing direction wo
and return the value of BxDF for the pair of directions.
"""
@inline function sample_fresnel_specular(f::UberBxDF{S}, wo::Vec3f, u::Point2f)::Tuple{Vec3f,Float32,RGBSpectrum,UInt8} where {S<:Spectrum}

    fd = fresnel_dielectric(cos_θ(wo), f.η_a, f.η_b)
    if u[1] < fd # Compute perfect specular reflection direction.
        wi = Vec3f(-wo[1], -wo[2], wo[3])
        sampled_type = BSDF_SPECULAR | BSDF_REFLECTION
        return wi, fd, fd * f.r / abs(cos_θ(wi)), sampled_type
    end

    # Figure out which η is incident and which is transmitted.
    if cos_θ(wo) > 0
        η_i, η_t = f.η_a, f.η_b
    else
        η_i, η_t = f.η_b, f.η_a
    end
    # Compute ray direction for specular transmission.
    refracted, wi = refract(
        wo, face_forward(Normal3f(0f0, 0f0, 1f0), wo), η_i / η_t,
    )
    !refracted && return wi, fd, RGBSpectrum(0f0), UInt8(0)

    pdf = 1f0 - fd
    ft = f.t * pdf
    # Account for non-symmetry with transmission to different medium.
    f.transport === Radiance && (ft *= (η_i^2) / (η_t^2))
    sampled_type = BSDF_SPECULAR | BSDF_TRANSMISSION
    return wi, pdf, ft / abs(cos_θ(wi)), sampled_type
end
