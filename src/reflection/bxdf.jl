const BSDF_NONE = UInt8(0b00000)
const BSDF_REFLECTION = UInt8(0b00001)
const BSDF_TRANSMISSION = UInt8(0b00010)
const BSDF_DIFFUSE = UInt8(0b00100)
const BSDF_GLOSSY = UInt8(0b01000)
const BSDF_SPECULAR = UInt8(0b10000)
const BSDF_ALL = UInt8(0b11111)

function Base.:&(b::B, type::UInt8)::Bool where B<:BxDF
    (b.type & type) == b.type
end

@inline function same_hemisphere(w::Vec3f, wp::Union{Vec3f,Normal3f})::Bool
    w[3] * wp[3] > 0
end

"""
Compute PDF value for the given directions.
In comparison, `sample_f` computes PDF value for the incident directions *it*
chooses given the outgoing direction, while this returns a value of PDF
for the given pair of directions.
"""
@inline function compute_pdf(::B, wo::Vec3f, wi::Vec3f)::Float32 where B<:BxDF
    same_hemisphere(wo, wi) ? abs(cos_θ(wi)) * (1f0 / π) : 0f0
end

"""
Compute the direction of incident light wi, given an outgoing direction wo
and return the value of BxDF for the pair of directions.

**Note** all BxDFs that implement this method,
have to implement `compute_pdf` as well.
"""
@inline function sample_f(
    b::B, wo::Vec3f, sample::Point2f,
)::Tuple{Vec3f,Float32,RGBSpectrum,Maybe{UInt8}} where B<:BxDF
    wi::Vec3f = cosine_sample_hemisphere(sample)
    # Flipping the direction if necessary.
    wo[3] < 0 && (wi = Vec3f(wi[1], wi[2], -wi[3]))
    pdf::Float32 = compute_pdf(b, wo, wi)
    wi, pdf, b(wo, wi), nothing
end

"""
Compute refracted direction `wt` given an incident direction `wi`,
surface normal `n` in the same hemisphere as `wi` and `η`, the ratio
of indices of refraction in the incident transmitted media respectively.

Returned boolean indicates whether a valid refracted ray was returned
or is it the case of total internal reflection.
"""
function refract(wi::Vec3f, n::Normal3f, η::Float32)::Tuple{Bool,Vec3f}
    # Compute cosθt using Snell's law.
    cos_θi = n ⋅ wi
    sin2_θi = max(0f0, 1f0 - cos_θi^2)
    sin2_θt = (η^2) * sin2_θi
    # Handle total internal reflection for transmission.
    sin2_θt >= 1 && return false, Vec3f(0f0)
    cos_θt = √(1f0 - sin2_θt)
    wt = -η .* wi + (η * cos_θi - cos_θt) .* n
    true, wt
end


"""
Compute Fresnel reflection formula for dielectric materials
and unpolarized light.
Which describes the amount of light reflected from a surface.

- `cos_θi::Float32`: Cosine of the incident angle w.r.t. normal.
- `ηi::Float32`: index of refraction for the incident media.
- `ηt::Float32`: index of refraction for the transmitted media.
"""
function fresnel_dielectric(cos_θi::Float32, ηi::Float32, ηt::Float32)
    cos_θi = clamp(cos_θi, -1f0, 1f0)
    if cos_θi ≤ 0f0 # if not entering
        ηi, ηt = ηt, ηi
        cos_θi = abs(cos_θi)
    end
    # Compute cos_θt using Snell's law.
    sin_θi = √max(0f0, 1f0 - cos_θi^2)
    sin_θt = sin_θi * ηi / ηt
    sin_θt ≥ 1f0 && return 1f0 # Handle total internal reflection.
    cos_θt = √max(0f0, 1f0 - sin_θt^2)

    r_parallel = (
        (ηt * cos_θi - ηi * cos_θt) /
        (ηt * cos_θi + ηi * cos_θt)
    )
    r_perp = (
        (ηi * cos_θi - ηt * cos_θt) /
        (ηi * cos_θi + ηt * cos_θt)
    )
    0.5f0 * (r_parallel^2 + r_perp^2)
end

"""
General Fresnel reflection formula with complex index of refraction η^ = η + ik,
where some incident light is potentially absorbed by the material and turned into heat.
k - is referred to as the absorption coefficient.
"""
function fresnel_conductor(
    cos_θi::Float32, ηi::S, ηt::S, k::S,
) where S<:Spectrum
    cos_θi = clamp(cos_θi, -1f0, 1f0)
    η = ηt / ηi
    ηk = k / ηi

    cos_θi2 = cos_θi * cos_θi
    sin_θi2 = 1f0 - cos_θi2
    η2 = η * η
    ηk2 = ηk * ηk

    t0 = η2 - ηk2 - sin_θi2
    a2_plus_b2 = √(t0 * t0 + 4f0 * η2 * ηk2)
    t1 = a2_plus_b2 + cos_θi2
    a = √(0.5f0 * (a2_plus_b2 + t0))
    t2 = 2f0 * cos_θi * a
    r_perp = (t1 - t2) / (t1 + t2)

    t3 = cos_θi2 * a2_plus_b2 + sin_θi2 * sin_θi2
    t4 = t2 * sin_θi2
    r_parallel = r_perp * (t3 - t4) / (t3 + t4)
    0.5f0 * (r_parallel + r_perp)
end

abstract type Fresnel end
struct FresnelConductor{S<:Spectrum} <: Fresnel
    ηi::S
    ηt::S
    k::S
end
struct FresnelDielectric <: Fresnel
    ηi::Float32
    ηt::Float32
end
struct FresnelNoOp <: Fresnel end
(f::FresnelConductor)(cos_θi::Float32) = fresnel_conductor(cos_θi, f.ηi, f.ηt, f.k)
(f::FresnelDielectric)(cos_θi::Float32) = fresnel_dielectric(cos_θi, f.ηi, f.ηt)
(f::FresnelNoOp)(::Float32) = RGBSpectrum(1f0)
