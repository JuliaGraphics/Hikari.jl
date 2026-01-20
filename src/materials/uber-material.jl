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
# Clean Material Types - each is a data container with only necessary parameters
# ============================================================================

"""
    MatteMaterial(Kd::Texture, σ::Texture)

Matte (diffuse) material with Lambertian or Oren-Nayar BRDF.

* `Kd`: Spectral diffuse reflection (color texture or TextureRef)
* `σ`: Scalar roughness for Oren-Nayar model (0 = Lambertian)
"""
struct MatteMaterial{KdTex, σTex} <: Material
    Kd::KdTex   # Texture{RGBSpectrum} or TextureRef{RGBSpectrum}
    σ::σTex     # Texture{Float32} or TextureRef{Float32}
end

function MatteMaterial(Kd::Texture, σ::Texture)
    MatteMaterial{typeof(Kd), typeof(σ)}(Kd, σ)
end

# Constructor for TextureRef (GPU path)
function MatteMaterial(Kd::TextureRef{RGBSpectrum}, σ::TextureRef{Float32})
    MatteMaterial{typeof(Kd), typeof(σ)}(Kd, σ)
end

"""
    MirrorMaterial(Kr::Texture)

Perfect mirror (specular reflection) material.

* `Kr`: Spectral reflectance (color texture or TextureRef)
"""
struct MirrorMaterial{KrTex} <: Material
    Kr::KrTex   # Texture{RGBSpectrum} or TextureRef{RGBSpectrum}
end

function MirrorMaterial(Kr::Texture)
    MirrorMaterial{typeof(Kr)}(Kr)
end

# Constructor for TextureRef (GPU path)
function MirrorMaterial(Kr::TextureRef{RGBSpectrum})
    MirrorMaterial{typeof(Kr)}(Kr)
end

"""
    GlassMaterial(Kr, Kt, u_roughness, v_roughness, index, remap_roughness)

Glass/dielectric material with reflection and transmission.

* `Kr`: Spectral reflectance (Texture or TextureRef)
* `Kt`: Spectral transmittance (Texture or TextureRef)
* `u_roughness`: Roughness in u direction (0 = perfect specular)
* `v_roughness`: Roughness in v direction (0 = perfect specular)
* `index`: Index of refraction
* `remap_roughness`: Whether to remap roughness to alpha
"""
struct GlassMaterial{KrTex, KtTex, URoughTex, VRoughTex, IndexTex} <: Material
    Kr::KrTex           # Texture{RGBSpectrum} or TextureRef{RGBSpectrum}
    Kt::KtTex           # Texture{RGBSpectrum} or TextureRef{RGBSpectrum}
    u_roughness::URoughTex  # Texture{Float32} or TextureRef{Float32}
    v_roughness::VRoughTex  # Texture{Float32} or TextureRef{Float32}
    index::IndexTex     # Texture{Float32} or TextureRef{Float32}
    remap_roughness::Bool
end

function GlassMaterial(
    Kr::Texture, Kt::Texture,
    u_roughness::Texture, v_roughness::Texture,
    index::Texture, remap_roughness::Bool
)
    GlassMaterial{typeof(Kr), typeof(Kt), typeof(u_roughness), typeof(v_roughness), typeof(index)}(
        Kr, Kt, u_roughness, v_roughness, index, remap_roughness
    )
end

# Constructor for TextureRef (GPU path)
function GlassMaterial(
    Kr::TextureRef{RGBSpectrum}, Kt::TextureRef{RGBSpectrum},
    u_roughness::TextureRef{Float32}, v_roughness::TextureRef{Float32},
    index::TextureRef{Float32}, remap_roughness::Bool
)
    GlassMaterial{typeof(Kr), typeof(Kt), typeof(u_roughness), typeof(v_roughness), typeof(index)}(
        Kr, Kt, u_roughness, v_roughness, index, remap_roughness
    )
end

# ============================================================================
# PlasticMaterial - Now an alias for CoatedDiffuseMaterial
# ============================================================================
# The old PlasticMaterial struct has been removed.
# PlasticMaterial(; Kd=..., roughness=...) now returns a CoatedDiffuseMaterial,
# matching pbrt-v4's behavior where plastic is implemented as coated diffuse.

# ============================================================================
# User-friendly keyword constructors with auto texture wrapping
# ============================================================================

# Helper to wrap values in Texture if not already a Texture
_to_texture(t::Texture) = t
_to_texture(v::RGBSpectrum) = Texture(v)
_to_texture(v::Float32) = Texture(v)
_to_texture(v::Real) = Texture(Float32(v))
# For color tuples/vectors (use Tuple{Real,Real,Real} to handle mixed Int/Float)
_to_texture(v::Tuple{Real,Real,Real}) = Texture(RGBSpectrum(Float32(v[1]), Float32(v[2]), Float32(v[3])))
_to_texture(v::AbstractVector{<:Real}) = length(v) == 3 ? Texture(RGBSpectrum(Float32.(v)...)) : error("Expected 3-element color")
# Support Colors.jl RGB types (RGB, RGBA, etc.)
_to_texture(c::Colorant) = Texture(RGBSpectrum(Float32(red(c)), Float32(green(c)), Float32(blue(c))))

"""
    MatteMaterial(; Kd=RGBSpectrum(0.5), σ=0.0)

Create a matte (diffuse) material with optional Oren-Nayar roughness.

# Arguments
- `Kd`: Diffuse color - can be RGBSpectrum, (r,g,b) tuple, or Texture
- `σ`: Roughness angle in degrees (0 = Lambertian, >0 = Oren-Nayar)

# Examples
```julia
MatteMaterial(Kd=RGBSpectrum(0.8, 0.2, 0.2))  # Red matte
MatteMaterial(Kd=(0.8, 0.2, 0.2), σ=20)       # Red with roughness
MatteMaterial(Kd=my_texture)                   # Textured
```
"""
function MatteMaterial(; Kd=RGBSpectrum(0.5f0), σ=0f0)
    MatteMaterial(_to_texture(Kd), _to_texture(σ))
end

"""
    MirrorMaterial(; Kr=RGBSpectrum(0.9))

Create a perfect mirror (specular reflection) material.

# Arguments
- `Kr`: Reflectance color - can be RGBSpectrum, (r,g,b) tuple, or Texture

# Examples
```julia
MirrorMaterial()                               # Default silver mirror
MirrorMaterial(Kr=RGBSpectrum(0.95, 0.93, 0.88))  # Gold-tinted
MirrorMaterial(Kr=(0.9, 0.9, 0.9))            # Using tuple
```
"""
function MirrorMaterial(; Kr=RGBSpectrum(0.9f0))
    MirrorMaterial(_to_texture(Kr))
end

"""
    GlassMaterial(; Kr=RGBSpectrum(1), Kt=RGBSpectrum(1), roughness=0, index=1.5, remap_roughness=true)

Create a glass/dielectric material with reflection and transmission.

# Arguments
- `Kr`: Reflectance color
- `Kt`: Transmittance color
- `roughness`: Surface roughness (0 = perfect specular, can be single value or (u,v) tuple)
- `index`: Index of refraction (1.5 for glass, 1.33 for water, 2.4 for diamond)
- `remap_roughness`: Whether to remap roughness to microfacet alpha

# Examples
```julia
GlassMaterial()                                # Clear glass
GlassMaterial(Kt=(1, 0.9, 0.8), index=1.5)    # Amber tinted
GlassMaterial(roughness=0.1)                   # Frosted glass
GlassMaterial(roughness=(0.1, 0.05))          # Anisotropic roughness
```
"""
function GlassMaterial(;
    Kr=RGBSpectrum(1f0),
    Kt=RGBSpectrum(1f0),
    roughness=0f0,
    index=1.5f0,
    remap_roughness=true
)
    # Handle roughness - can be single value or (u, v) tuple
    if roughness isa Tuple
        u_rough, v_rough = roughness
    else
        u_rough = v_rough = roughness
    end
    GlassMaterial(
        _to_texture(Kr), _to_texture(Kt),
        _to_texture(u_rough), _to_texture(v_rough),
        _to_texture(index), remap_roughness
    )
end

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

# ============================================================================
# Metal Material - Conductor with Fresnel reflectance and microfacet roughness
# ============================================================================

"""
    MetalMaterial{EtaTex, KTex, RoughTex, ReflTex}

A metal/conductor material with wavelength-dependent complex index of refraction.

Metals reflect light based on Fresnel equations for conductors, characterized by:
- η (eta): Real part of complex IOR
- k: Imaginary part (extinction coefficient)
- roughness: Surface roughness for microfacet model

# Fields
* `eta`: Real part of complex index of refraction (wavelength-dependent)
* `k`: Extinction coefficient (wavelength-dependent)
* `roughness`: Surface roughness
* `reflectance`: Color multiplier for Fresnel reflectance (for tinting)
* `remap_roughness`: Whether to remap roughness to alpha
"""
struct MetalMaterial{EtaTex, KTex, RoughTex, ReflTex} <: Material
    eta::EtaTex             # Texture{RGBSpectrum} or TextureRef{RGBSpectrum}
    k::KTex                 # Texture{RGBSpectrum} or TextureRef{RGBSpectrum}
    roughness::RoughTex     # Texture{Float32} or TextureRef{Float32}
    reflectance::ReflTex    # Texture{RGBSpectrum} or TextureRef{RGBSpectrum}
    remap_roughness::Bool
end

function MetalMaterial(
    eta::Texture, k::Texture, roughness::Texture, reflectance::Texture, remap_roughness::Bool
)
    MetalMaterial{typeof(eta), typeof(k), typeof(roughness), typeof(reflectance)}(
        eta, k, roughness, reflectance, remap_roughness
    )
end

# Constructor for TextureRef (GPU path)
function MetalMaterial(
    eta::TextureRef{RGBSpectrum}, k::TextureRef{RGBSpectrum},
    roughness::TextureRef{Float32}, reflectance::TextureRef{RGBSpectrum}, remap_roughness::Bool
)
    MetalMaterial{typeof(eta), typeof(k), typeof(roughness), typeof(reflectance)}(
        eta, k, roughness, reflectance, remap_roughness
    )
end

# Backwards-compatible constructor without reflectance (defaults to white = no tint)
function MetalMaterial(
    eta::Texture, k::Texture, roughness::Texture, remap_roughness::Bool
)
    reflectance = ConstantTexture(RGBSpectrum(1f0))
    MetalMaterial(eta, k, roughness, reflectance, remap_roughness)
end

# Common metal presets (approximate values at 550nm)
const METAL_COPPER = (eta=(0.27, 0.68, 1.22), k=(3.61, 2.63, 2.29))
const METAL_GOLD = (eta=(0.14, 0.38, 1.44), k=(3.98, 2.75, 1.95))
const METAL_SILVER = (eta=(0.16, 0.14, 0.13), k=(4.03, 3.59, 2.62))
const METAL_ALUMINUM = (eta=(1.35, 0.97, 0.60), k=(7.47, 6.40, 5.30))
const METAL_IRON = (eta=(2.95, 2.93, 2.59), k=(3.10, 2.99, 2.74))

"""
    MetalMaterial(; eta=(0.2, 0.2, 0.2), k=(3.9, 3.9, 3.9), roughness=0.1, remap_roughness=true)

Create a metal/conductor material with Fresnel reflectance.

# Arguments
- `eta`: Real part of complex IOR - (r,g,b) tuple, RGBSpectrum, or Texture
- `k`: Extinction coefficient - (r,g,b) tuple, RGBSpectrum, or Texture
- `roughness`: Surface roughness (0 = mirror-like, higher = more diffuse)
- `reflectance`: Color multiplier for tinting the metal (default white = no tint)
- `remap_roughness`: Whether to remap roughness to microfacet alpha

# Presets
Use the provided metal constants for realistic materials:
- `METAL_COPPER`, `METAL_GOLD`, `METAL_SILVER`, `METAL_ALUMINUM`, `METAL_IRON`

# Examples
```julia
MetalMaterial()                                        # Generic metal
MetalMaterial(; METAL_COPPER..., roughness=0.05)      # Polished copper
MetalMaterial(; METAL_GOLD..., roughness=0.1)         # Brushed gold
MetalMaterial(eta=(0.2, 0.8, 0.2), k=(3, 3, 3))       # Custom green-tinted metal
MetalMaterial(; METAL_GOLD..., reflectance=(1, 0.5, 0.5))  # Gold tinted red
```
"""
function MetalMaterial(;
    eta=(0.2f0, 0.2f0, 0.2f0),
    k=(3.9f0, 3.9f0, 3.9f0),
    roughness=0.1f0,
    reflectance=(1f0, 1f0, 1f0),
    remap_roughness=true
)
    MetalMaterial(_to_texture(eta), _to_texture(k), _to_texture(roughness), _to_texture(reflectance), remap_roughness)
end

# ============================================================================
# Clean Type Aliases - shorter names without "Material" suffix
# ============================================================================

"""Type alias: `Diffuse` is the same as `MatteMaterial`"""
const Diffuse = MatteMaterial

"""Type alias: `Mirror` is the same as `MirrorMaterial`"""
const Mirror = MirrorMaterial

"""Type alias: `Dielectric` is the same as `GlassMaterial`"""
const Dielectric = GlassMaterial

"""Type alias: `Plastic` is the same as `PlasticMaterial`"""
const Plastic = PlasticMaterial

"""Type alias: `Conductor` is the same as `MetalMaterial`"""
const Conductor = MetalMaterial

# ============================================================================
# Metal Preset Constructors - convenient ways to create common metals
# ============================================================================

# Metal optical constants (approximate values at 550nm from pbrt-v4)
# These are the complex index of refraction values: n + ik
const _COPPER_ETA = (0.27f0, 0.68f0, 1.22f0)
const _COPPER_K = (3.61f0, 2.63f0, 2.29f0)

const _GOLD_ETA = (0.14f0, 0.38f0, 1.44f0)
const _GOLD_K = (3.98f0, 2.75f0, 1.95f0)

const _SILVER_ETA = (0.16f0, 0.14f0, 0.13f0)
const _SILVER_K = (4.03f0, 3.59f0, 2.62f0)

const _ALUMINUM_ETA = (1.35f0, 0.97f0, 0.60f0)
const _ALUMINUM_K = (7.47f0, 6.40f0, 5.30f0)

const _IRON_ETA = (2.95f0, 2.93f0, 2.59f0)
const _IRON_K = (3.10f0, 2.99f0, 2.74f0)

"""
    Gold(; roughness=0.0, reflectance=(1,1,1), remap_roughness=true)

Create a gold conductor material with realistic optical constants.

# Examples
```julia
Gold()                          # Polished gold
Gold(roughness=0.1)             # Brushed gold
Gold(roughness=0.3)             # Matte gold
```
"""
Gold(; roughness=0f0, reflectance=(1f0, 1f0, 1f0), remap_roughness=true) =
    Conductor(eta=_GOLD_ETA, k=_GOLD_K, roughness=roughness, reflectance=reflectance, remap_roughness=remap_roughness)

"""
    Silver(; roughness=0.0, reflectance=(1,1,1), remap_roughness=true)

Create a silver conductor material with realistic optical constants.

# Examples
```julia
Silver()                        # Polished silver
Silver(roughness=0.05)          # Slightly brushed
```
"""
Silver(; roughness=0f0, reflectance=(1f0, 1f0, 1f0), remap_roughness=true) =
    Conductor(eta=_SILVER_ETA, k=_SILVER_K, roughness=roughness, reflectance=reflectance, remap_roughness=remap_roughness)

"""
    Copper(; roughness=0.0, reflectance=(1,1,1), remap_roughness=true)

Create a copper conductor material with realistic optical constants.

# Examples
```julia
Copper()                        # Polished copper
Copper(roughness=0.2)           # Weathered copper
```
"""
Copper(; roughness=0f0, reflectance=(1f0, 1f0, 1f0), remap_roughness=true) =
    Conductor(eta=_COPPER_ETA, k=_COPPER_K, roughness=roughness, reflectance=reflectance, remap_roughness=remap_roughness)

"""
    Aluminum(; roughness=0.0, reflectance=(1,1,1), remap_roughness=true)

Create an aluminum conductor material with realistic optical constants.

# Examples
```julia
Aluminum()                      # Polished aluminum
Aluminum(roughness=0.1)         # Brushed aluminum
```
"""
Aluminum(; roughness=0f0, reflectance=(1f0, 1f0, 1f0), remap_roughness=true) =
    Conductor(eta=_ALUMINUM_ETA, k=_ALUMINUM_K, roughness=roughness, reflectance=reflectance, remap_roughness=remap_roughness)

"""
    Iron(; roughness=0.0, reflectance=(1,1,1), remap_roughness=true)

Create an iron conductor material with realistic optical constants.

# Examples
```julia
Iron()                          # Polished iron
Iron(roughness=0.3)             # Rough cast iron
```
"""
Iron(; roughness=0f0, reflectance=(1f0, 1f0, 1f0), remap_roughness=true) =
    Conductor(eta=_IRON_ETA, k=_IRON_K, roughness=roughness, reflectance=reflectance, remap_roughness=remap_roughness)
