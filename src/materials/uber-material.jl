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

function (f::Fresnel)(cos_θi::Float32)
    if f.type === FRESNEL_DIELECTRIC
        return fresnel_dielectric(cos_θi, f.ηi[1], f.ηt[1])
    elseif f.type === FRESNEL_CONDUCTOR
        return fresnel_conductor(cos_θi, f.ηi, f.ηt, f.k)
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

@propagate_inbounds function sample_f(s::UberBxDF, wo::Vec3f, sample::Point2f)::Tuple{Vec3f,Float32,RGBSpectrum,UInt8}
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
@propagate_inbounds function compute_pdf(s::UberBxDF, wo::Vec3f, wi::Vec3f)::Float32
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

@propagate_inbounds function (s::UberBxDF)(wo::Vec3f, wi::Vec3f)
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
# Hikari-specific _to_texture extensions (base overloads are in HikariBase)
# ============================================================================

import HikariBase: _to_texture

# Support Colors.jl RGB types (RGB, RGBA, etc.)
_to_texture(c::Colorant) = _to_texture(RGBSpectrum(Float32(red(c)), Float32(green(c)), Float32(blue(c))))
_to_texture(c::AbstractMatrix{<: RGB}) = Texture(map(c-> RGBSpectrum(Float32(red(c)), Float32(green(c)), Float32(blue(c))), c))

"""
    PlasticMaterial(; Kd=RGBSpectrum(0.5), Ks=RGBSpectrum(0.5), roughness=0.1, remap_roughness=true, eta=1.5)

Create a plastic material with diffuse base and dielectric coating.

This is an alias for `CoatedDiffuseMaterial` matching pbrt-v4's behavior where
"plastic" materials are implemented as coated diffuse with a dielectric coating.

# Arguments
- `Kd`: Diffuse color (reflectance of the base layer)
- `Ks`: Specular color (ignored - kept for API compatibility, Fresnel controls specular)
- `roughness`: Surface roughness of the coating (lower = sharper highlights)
- `remap_roughness`: Whether to remap roughness to microfacet alpha
- `eta`: Index of refraction of the coating (default 1.5 for typical plastic)

# Examples
```julia
PlasticMaterial(Kd=(0.8, 0.2, 0.6))           # Magenta plastic
PlasticMaterial(Kd=(0.1, 0.1, 0.8), roughness=0.05)  # Shiny blue
PlasticMaterial(Kd=wood_texture, roughness=0.3)      # Textured
```
"""
function PlasticMaterial(;
    Kd=RGBSpectrum(0.5f0),
    Ks=RGBSpectrum(0.5f0),  # Kept for API compatibility, ignored
    roughness=0.1f0,
    remap_roughness=true,
    eta=1.5f0
)
    # Convert to CoatedDiffuseMaterial (pbrt-v4's actual plastic implementation)
    CoatedDiffuseMaterial(
        reflectance=Kd,
        roughness=roughness,
        eta=Float32(eta),
        remap_roughness=remap_roughness
    )
end

# Identity passthrough for PiecewiseLinearSpectrum (not a texture, stored directly)
_to_texture(s::PiecewiseLinearSpectrum) = s

# Common metal presets as RGB (for custom metals via keyword constructors)
const METAL_COPPER = (eta=(0.27, 0.68, 1.22), k=(3.61, 2.63, 2.29))
const METAL_GOLD = (eta=(0.14, 0.38, 1.44), k=(3.98, 2.75, 1.95))
const METAL_SILVER = (eta=(0.16, 0.14, 0.13), k=(4.03, 3.59, 2.62))
const METAL_ALUMINUM = (eta=(1.35, 0.97, 0.60), k=(7.47, 6.40, 5.30))

# ============================================================================
# Type Aliases (Hikari-specific)
# ============================================================================

"""Type alias: `Plastic` is the same as `PlasticMaterial`"""
const Plastic = PlasticMaterial

"""Type alias: `MetalMaterial` is the same as `ConductorMaterial` (legacy alias)"""
const MetalMaterial = ConductorMaterial

# ============================================================================
# Conductor Preset Constructors - using measured spectral data from pbrt-v4
# ============================================================================

"""
    Gold(; roughness=0.0, reflectance=(1,1,1), remap_roughness=true)

Create a gold conductor material with measured spectral IOR data.

# Examples
```julia
Gold()                          # Polished gold
Gold(roughness=0.1)             # Brushed gold
Gold(roughness=0.3)             # Matte gold
```
"""
Gold(; roughness=0f0, reflectance=(1f0, 1f0, 1f0), remap_roughness=true) =
    ConductorMaterial(AU_ETA_SPECTRUM, AU_K_SPECTRUM, _to_texture(roughness), _to_texture(reflectance), remap_roughness)

"""
    Silver(; roughness=0.0, reflectance=(1,1,1), remap_roughness=true)

Create a silver conductor material with measured spectral IOR data.

# Examples
```julia
Silver()                        # Polished silver
Silver(roughness=0.05)          # Slightly brushed
```
"""
Silver(; roughness=0f0, reflectance=(1f0, 1f0, 1f0), remap_roughness=true) =
    ConductorMaterial(AG_ETA_SPECTRUM, AG_K_SPECTRUM, _to_texture(roughness), _to_texture(reflectance), remap_roughness)

"""
    Copper(; roughness=0.0, reflectance=(1,1,1), remap_roughness=true)

Create a copper conductor material with measured spectral IOR data.

# Examples
```julia
Copper()                        # Polished copper
Copper(roughness=0.2)           # Weathered copper
```
"""
Copper(; roughness=0f0, reflectance=(1f0, 1f0, 1f0), remap_roughness=true) =
    ConductorMaterial(CU_ETA_SPECTRUM, CU_K_SPECTRUM, _to_texture(roughness), _to_texture(reflectance), remap_roughness)

"""
    Aluminum(; roughness=0.0, reflectance=(1,1,1), remap_roughness=true)

Create an aluminum conductor material with measured spectral IOR data.

# Examples
```julia
Aluminum()                      # Polished aluminum
Aluminum(roughness=0.1)         # Brushed aluminum
```
"""
Aluminum(; roughness=0f0, reflectance=(1f0, 1f0, 1f0), remap_roughness=true) =
    ConductorMaterial(AL_ETA_SPECTRUM, AL_K_SPECTRUM, _to_texture(roughness), _to_texture(reflectance), remap_roughness)

"""
    Brass(; roughness=0.0, reflectance=(1,1,1), remap_roughness=true)

Create a brass (CuZn) conductor material with measured spectral IOR data.

# Examples
```julia
Brass()                         # Polished brass
Brass(roughness=0.15)           # Brushed brass
```
"""
Brass(; roughness=0f0, reflectance=(1f0, 1f0, 1f0), remap_roughness=true) =
    ConductorMaterial(CUZN_ETA_SPECTRUM, CUZN_K_SPECTRUM, _to_texture(roughness), _to_texture(reflectance), remap_roughness)
