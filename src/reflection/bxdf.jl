const BSDF_NONE = UInt8(0b00000)
const BSDF_REFLECTION = UInt8(0b00001)
const BSDF_TRANSMISSION = UInt8(0b00010)
const BSDF_DIFFUSE = UInt8(0b00100)
const BSDF_GLOSSY = UInt8(0b01000)
const BSDF_SPECULAR = UInt8(0b10000)
const BSDF_ALL = UInt8(0b11111)


@propagate_inbounds function same_hemisphere(w::Vec3f, wp::Union{Vec3f,Normal3f})::Bool
    w[3] * wp[3] > 0
end

"""
    refract(wi::Vec3f, n::Normal3f, eta::Float32) -> (valid::Bool, wt::Vec3f)

Compute refracted direction `wt` given an incident direction `wi`,
surface normal `n`, and `eta` = n_t / n_i (ratio of transmitted to incident IOR).

This matches pbrt-v4's Refract() function exactly:
- If wi comes from below the surface (cos_θi < 0), the interface is flipped
- Returns (false, zero) for total internal reflection
- Returns (true, wt) with the refracted direction otherwise

The convention is: eta = n_transmitted / n_incident
- For ray entering glass (n_i=1, n_t=1.5): eta = 1.5
- For ray exiting glass (n_i=1.5, n_t=1): eta = 1/1.5 ≈ 0.67
"""
function refract(wi::Vec3f, n::Normal3f, eta::Float32)::Tuple{Bool,Vec3f}
    cos_θi = n ⋅ wi

    # Potentially flip interface orientation for Snell's law (matches pbrt-v4)
    if cos_θi < 0f0
        eta = 1f0 / eta
        cos_θi = -cos_θi
        n = -n
    end

    # Compute cos_θt using Snell's law
    sin2_θi = max(0f0, 1f0 - cos_θi^2)
    sin2_θt = sin2_θi / (eta^2)

    # Handle total internal reflection
    sin2_θt >= 1f0 && return false, Vec3f(0f0)

    cos_θt = sqrt(1f0 - sin2_θt)

    # Compute refracted direction (matches pbrt-v4 exactly)
    wt = -wi / eta + (cos_θi / eta - cos_θt) * Vec3f(n)
    return true, wt
end


"""
    fresnel_dielectric(cos_θi::Float32, eta::Float32) -> Float32

Compute Fresnel reflection for dielectric materials (single-eta version).
This matches pbrt-v4's FrDielectric() exactly.

Arguments:
- `cos_θi`: Cosine of incident angle (can be negative if coming from inside)
- `eta`: Ratio n_t / n_i (transmitted IOR / incident IOR)

For a ray hitting glass from air: eta = 1.5 (glass IOR)
For a ray hitting air from inside glass: eta = 1/1.5
"""
function fresnel_dielectric(cos_θi::Float32, eta::Float32)::Float32
    cos_θi = clamp(cos_θi, -1f0, 1f0)

    # Potentially flip interface orientation for Fresnel equations (matches pbrt-v4)
    if cos_θi < 0f0
        eta = 1f0 / eta
        cos_θi = -cos_θi
    end

    # Compute cos_θt using Snell's law
    sin2_θi = 1f0 - cos_θi^2
    sin2_θt = sin2_θi / (eta^2)

    # Handle total internal reflection
    sin2_θt >= 1f0 && return 1f0

    cos_θt = sqrt(1f0 - sin2_θt)

    # Fresnel equations (matches pbrt-v4 exactly)
    r_parl = (eta * cos_θi - cos_θt) / (eta * cos_θi + cos_θt)
    r_perp = (cos_θi - eta * cos_θt) / (cos_θi + eta * cos_θt)

    return 0.5f0 * (r_parl^2 + r_perp^2)
end

"""
    fresnel_dielectric(cos_θi::Float32, ηi::Float32, ηt::Float32) -> Float32

Compute Fresnel reflection for dielectric materials (two-IOR version).
This is a convenience wrapper that computes eta = ηt / ηi.

Arguments:
- `cos_θi`: Cosine of incident angle w.r.t. normal
- `ηi`: Index of refraction for the incident media
- `ηt`: Index of refraction for the transmitted media
"""
function fresnel_dielectric(cos_θi::Float32, ηi::Float32, ηt::Float32)::Float32
    fresnel_dielectric(cos_θi, ηt / ηi)
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
    return 0.5f0 * (r_parallel + r_perp)
end
