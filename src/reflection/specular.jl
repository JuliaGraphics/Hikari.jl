struct SpecularReflection{S<:Spectrum,F<:Fresnel} <: BxDF
    """
    Spectrum used to scale the reflected color.
    """
    r::S
    """
    Describes fresnel properties.
    """
    fresnel::F

    type::UInt8

    function SpecularReflection(r::S, fresnel::F) where {S<:Spectrum,F<:Fresnel}
        new{S,F}(r, fresnel, BSDF_SPECULAR | BSDF_REFLECTION)
    end
end

"""
Return value of the distribution function for the given pair of directions.
For specular reflection, no scattering is returned, since
for arbitrary directions δ-funcion returns no scattering.
"""
function (s::SpecularReflection{S,F})(
    ::Vec3f, ::Vec3f,
)::RGBSpectrum where {S<:Spectrum,F<:Fresnel}
    S(0f0)
end

"""
Compute the direction of incident light wi, given an outgoing direction wo
and return the value of BxDF for the pair of directions.
`sample` parameter isn't needed for the δ-distribution.
"""
@inline function sample_f(
    s::SpecularReflection{S,F}, wo::Vec3f, ::Point2f,
)::Tuple{Vec3f,Float32,RGBSpectrum,Maybe{UInt8}} where {S<:Spectrum,F<:Fresnel}
    wi = Vec3f(-wo[1], -wo[2], wo[3])
    wi, 1f0, s.fresnel(cos_θ(wi)) * s.r / abs(cos_θ(wi)), nothing
end

struct SpecularTransmission{S<:Spectrum,T<:TransportMode} <: BxDF
    t::S
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
    fresnel::FresnelDielectric

    type::UInt8

    function SpecularTransmission(
        t::S, η_a::Float32, η_b::Float32, ::Type{T},
    ) where {S<:Spectrum,T<:TransportMode}
        new{S,T}(
            t, η_a, η_b,
            FresnelDielectric(η_a, η_b),
            BSDF_SPECULAR | BSDF_TRANSMISSION,
        )
    end
end

"""
Return value of the distribution function for the given pair of directions.
For specular transmission, no scattering is returned, since
for arbitrary directions δ-funcion returns no scattering.
"""
function (s::SpecularTransmission{S,T})(
    ::Vec3f, ::Vec3f,
)::RGBSpectrum where {S<:Spectrum,T<:TransportMode}
    S(0f0)
end

"""
Compute the direction of incident light wi, given an outgoing direction wo
and return the value of BxDF for the pair of directions.
`sample` parameter isn't needed for the δ-distribution.
"""
@inline function sample_f(
    s::SpecularTransmission{S,T}, wo::Vec3f, ::Point2f,
)::Tuple{Vec3f,Float32,RGBSpectrum,Maybe{UInt8}} where {S<:Spectrum,T<:TransportMode}
    # Figure out which η is incident and which is transmitted.
    entering = cos_θ(wo) > 0
    η_i = entering ? s.η_a : s.η_b
    η_t = entering ? s.η_b : s.η_a
    # Compute ray direction for specular transmission.
    valid, wi = refract(
        wo, face_forward(Normal3f(0f0, 0f0, 1f0), wo), η_i / η_t,
    )
    # Total internal reflection.
    !valid && return Vec3f(0f0), 0f0, S(0f0), nothing
    pdf = 1f0

    cos_wi = cos_θ(wi)
    ft = s.t * (S(1f0) - s.fresnel(cos_wi))
    # Account for non-symmetry with transmission to different medium.
    T isa Radiance && (ft *= (η_i^2) / (η_t^2))
    wi, pdf, ft / abs(cos_wi), nothing
end


struct FresnelSpecular{S<:Spectrum,T<:TransportMode} <: BxDF
    r::S
    t::S
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

    type::UInt8

    function FresnelSpecular(
        r::S, t::S, η_a::Float32, η_b::Float32, ::Type{T},
    ) where {S<:Spectrum,T<:TransportMode}
        new{S,T}(
            r, t, η_a, η_b,
            BSDF_SPECULAR | BSDF_TRANSMISSION | BSDF_REFLECTION,
        )
    end
end

@inline function (f::FresnelSpecular{S,T})(
    ::Vec3f, ::Vec3f,
)::RGBSpectrum where {S<:Spectrum,T<:TransportMode}
    S(0f0)
end

@inline compute_pdf(f::FresnelSpecular, wo::Vec3f, wi::Vec3f)::Float32 = 0f0

"""
Compute the direction of incident light wi, given an outgoing direction wo
and return the value of BxDF for the pair of directions.
"""
@inline function sample_f(
    f::FresnelSpecular{S,T}, wo::Vec3f, u::Point2f,
)::Tuple{Vec3f,Float32,RGBSpectrum,Maybe{UInt8}} where {S<:Spectrum,T<:TransportMode}
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
    !refracted && return wi, fd, RGBSpectrum(0f0), nothing

    pdf = 1f0 - fd
    ft = f.t * pdf
    # Account for non-symmetry with transmission to different medium.
    T isa Radiance && (ft *= (η_i^2) / (η_t^2))
    sampled_type = BSDF_SPECULAR | BSDF_TRANSMISSION
    wi, pdf, ft / abs(cos_θ(wi)), sampled_type
end
