# ============================================================================
# Conductor (Metal) and Mirror Materials
# Contains: Mirror struct, keyword constructor, Mirror alias,
#           Conductor struct, keyword constructor, Conductor/Metal aliases,
#           Gold/Silver/Copper/Aluminum/Brass preset constructors,
#           and the BSDFs of both as lobes
# ============================================================================

# ============================================================================
# Mirror
# ============================================================================

"""
    Mirror(Kr::Texture)

Perfect mirror (specular reflection) material.

* `Kr`: Spectral reflectance (color texture or TextureRef)
"""
# Non-parametric texture parameters: constant-vs-texture must not change the
# material's TYPE, or the per-material shading compiles a separate kernel per
# combination.
struct Mirror{KrT, DispT} <: Material
    Kr::KrT
    displacement::DispT       # pbrt-v4 `Material::displacement` height field (NONE = flat)
end

"""
    Mirror(; Kr=RGBSpectrum(0.9))

Create a perfect mirror (specular reflection) material.

# Arguments
- `Kr`: Reflectance color - can be RGBSpectrum, (r,g,b) tuple, or Texture

# Examples
```julia
Mirror()                               # Default silver mirror
Mirror(Kr=RGBSpectrum(0.95, 0.93, 0.88))  # Gold-tinted
Mirror(Kr=(0.9, 0.9, 0.9))            # Using tuple
```
"""
function Mirror(; Kr=RGBSpectrum(0.9f0), bump=nothing)
    Mirror(matparam(Kr), matparam(bump))
end


# ============================================================================
# Conductor
# ============================================================================

"""
    Conductor{EtaTex, KTex, RoughTex, ReflTex}

A metal/conductor material with wavelength-dependent complex index of refraction.

Metals reflect light based on Fresnel equations for conductors, characterized by:
- η (eta): Real part of complex IOR (PiecewiseLinearSpectrum for presets, or RGB texture)
- k: Imaginary part (extinction coefficient)
- roughness: Surface roughness for microfacet model, along the tangent and across it

# Fields
* `eta`: Real part of complex index of refraction (PiecewiseLinearSpectrum or texture)
* `k`: Extinction coefficient (PiecewiseLinearSpectrum or texture)
* `u_roughness`, `v_roughness`: Surface roughness along the shading tangent and across it
* `reflectance`: instead of `eta` and `k`, the metal's reflectance at normal incidence, as pbrt-v4's
  (`TexHandle()` where the metal is given by `eta` and `k`)
* `remap_roughness`: Whether to remap roughness to alpha
"""
# Roughness / `reflectance` are `TexHandle`, not type parameters: whether a
# parameter is a constant or a texture must not change the MATERIAL's type, or
# the per-material shading compiles a separate kernel for each combination.
# Crown alone had three `Conductor` types that differ only in this.
#
# `eta` / `k` stay parametric because they may be a `PiecewiseLinearSpectrum{56}`
# (448 bytes of measured metal data) which cannot live inline in a handle. In
# practice every conductor in a scene uses the same spectral representation, so
# they do not multiply types.
struct Conductor{EtaTex, KTex, URoughT, VRoughT, ReflT, DispT} <: Material
    eta::EtaTex             # PiecewiseLinearSpectrum or spectral value
    k::KTex                 # PiecewiseLinearSpectrum or spectral value
    u_roughness::URoughT
    v_roughness::VRoughT
    reflectance::ReflT
    remap_roughness::Bool
    displacement::DispT       # pbrt-v4 `Material::displacement` height field (NONE = flat)
end


"""
    Conductor(; eta=(0.2, 0.2, 0.2), k=(3.9, 3.9, 3.9), roughness=0.1, remap_roughness=true)
    Conductor(; reflectance, roughness=0.1, remap_roughness=true)

Create a metal/conductor material with Fresnel reflectance: of the complex IOR `eta` + i`k`, or of
the metal whose reflectance at normal incidence is `reflectance`, as pbrt-v4's (IOR 1 + i 2√r/√(1 - r)
at each wavelength). Not both.

# Arguments
- `eta`: Real part of complex IOR - PiecewiseLinearSpectrum, (r,g,b) tuple, RGBSpectrum, or Texture
- `k`: Extinction coefficient - PiecewiseLinearSpectrum, (r,g,b) tuple, RGBSpectrum, or Texture
- `reflectance`: The metal's colour, its reflectance at normal incidence - (r,g,b) tuple, RGBSpectrum, or Texture
- `roughness`: Surface roughness (0 = mirror-like, higher = more diffuse), or a `(u, v)` pair for an anisotropic metal
- `remap_roughness`: Whether to remap roughness to microfacet alpha

# Presets
Use the spectral preset constructors: `Gold()`, `Silver()`, `Copper()`, `Aluminum()`, `Brass()`

# Examples
```julia
Conductor()                                        # Generic metal
Gold(roughness=0.05)                               # Polished gold
Conductor(eta=(0.2, 0.8, 0.2), k=(3, 3, 3))       # Custom green-tinted metal
Conductor(reflectance=(0.9, 0.6, 0.3))             # A metal of this colour
```
"""
function Conductor(;
    eta=nothing,
    k=nothing,
    roughness=0.1f0,
    reflectance=nothing,
    remap_roughness=true,
    bump=nothing,
)
    if !isnothing(reflectance)
        (isnothing(eta) && isnothing(k)) ||
            throw(ArgumentError("a Conductor takes either `eta` and `k` or a `reflectance`, not both"))
        return Conductor(TexHandle(), TexHandle(), matparam.(roughness_pair(roughness))..., matparam(reflectance),
                         remap_roughness, matparam(bump))
    end
    # eta/k stay whatever they are (a `PiecewiseLinearSpectrum` for the measured
    # presets); RGB ones become inline handles so `Conductor(eta=rgb)` and
    # `Conductor(eta=other_rgb)` share one concrete type.
    Conductor(matparam(something(eta, (0.2f0, 0.2f0, 0.2f0))), matparam(something(k, (3.9f0, 3.9f0, 3.9f0))),
              matparam.(roughness_pair(roughness))..., TexHandle(), remap_roughness, matparam(bump))
end

# ============================================================================
# Conductor Preset Constructors - using measured spectral data from pbrt-v4
# ============================================================================

"""
    Gold(; roughness=0.0, remap_roughness=true)

Create a gold conductor material with measured spectral IOR data.

# Examples
```julia
Gold()                          # Polished gold
Gold(roughness=0.1)             # Brushed gold
Gold(roughness=0.3)             # Matte gold
```
"""
Gold(; roughness=0f0, remap_roughness=true, bump=nothing) =
    Conductor(AU_ETA_SPECTRUM, AU_K_SPECTRUM, matparam.(roughness_pair(roughness))..., TexHandle(),
              remap_roughness, matparam(bump))

"""
    Silver(; roughness=0.0, remap_roughness=true)

Create a silver conductor material with measured spectral IOR data.

# Examples
```julia
Silver()                        # Polished silver
Silver(roughness=0.05)          # Slightly brushed
```
"""
Silver(; roughness=0f0, remap_roughness=true, bump=nothing) =
    Conductor(AG_ETA_SPECTRUM, AG_K_SPECTRUM, matparam.(roughness_pair(roughness))..., TexHandle(),
              remap_roughness, matparam(bump))

"""
    Copper(; roughness=0.0, remap_roughness=true)

Create a copper conductor material with measured spectral IOR data.

# Examples
```julia
Copper()                        # Polished copper
Copper(roughness=0.2)           # Weathered copper
```
"""
Copper(; roughness=0f0, remap_roughness=true, bump=nothing) =
    Conductor(CU_ETA_SPECTRUM, CU_K_SPECTRUM, matparam.(roughness_pair(roughness))..., TexHandle(),
              remap_roughness, matparam(bump))

"""
    Aluminum(; roughness=0.0, remap_roughness=true)

Create an aluminum conductor material with measured spectral IOR data.

# Examples
```julia
Aluminum()                      # Polished aluminum
Aluminum(roughness=0.1)         # Brushed aluminum
```
"""
Aluminum(; roughness=0f0, remap_roughness=true, bump=nothing) =
    Conductor(AL_ETA_SPECTRUM, AL_K_SPECTRUM, matparam.(roughness_pair(roughness))..., TexHandle(),
              remap_roughness, matparam(bump))

"""
    Brass(; roughness=0.0, remap_roughness=true)

Create a brass (CuZn) conductor material with measured spectral IOR data.

# Examples
```julia
Brass()                         # Polished brass
Brass(roughness=0.15)           # Brushed brass
```
"""
Brass(; roughness=0f0, remap_roughness=true, bump=nothing) =
    Conductor(CUZN_ETA_SPECTRUM, CUZN_K_SPECTRUM, matparam.(roughness_pair(roughness))..., TexHandle(),
              remap_roughness, matparam(bump))

# ============================================================================
# Helpers for evaluating IOR values (PiecewiseLinearSpectrum or RGB textures)
# ============================================================================

"""
A conductor's BSDF: pbrt-v4's `ConductorBxDF`, one GGX reflection lobe with the Fresnel of its
complex IOR (`ConductorFresnel`), single scattering, in pbrt's convention.
"""
@propagate_inbounds function get_bxdf(mat::Conductor, table, textures, tfc::TextureFilterContext, lambda::Wavelengths,
                                      regularize::Bool, wo::Vec3f, ns::Vec3f, dpdus::Vec3f)
    u_roughness = eval_handle(textures, mat.u_roughness, tfc)
    v_roughness = eval_handle(textures, mat.v_roughness, tfc)
    alpha_x = mat.remap_roughness ? roughness_to_α(u_roughness) : u_roughness
    alpha_y = mat.remap_roughness ? roughness_to_α(v_roughness) : v_roughness
    if regularize
        alpha_x, alpha_y = regularize_alpha(alpha_x), regularize_alpha(alpha_y)
    end
    lobe = pbrt_conductor(conductor_ior(table, textures, mat.eta, mat.k, mat.reflectance, tfc, lambda)..., alpha_x, alpha_y)
    n, t = pbrt_frame(wo, ns, dpdus)
    return pbrt_bsdf((Weighted(lobe, SpectralRadiance(1f0), 1f0, n, t),), lambda)
end

"""A mirror's BSDF: a singular reflection of `Kr`."""
@propagate_inbounds function get_bxdf(mat::Mirror, table, textures, tfc::TextureFilterContext, lambda::Wavelengths,
                                      regularize::Bool, wo::Vec3f, ns::Vec3f, dpdus::Vec3f)
    lobe = MicrofacetLobe{:reflection}(NoFresnel(SpectralRadiance(1f0)); alpha_x = 0f0, convention = PbrtRadiance())
    n, t = pbrt_frame(wo, ns, dpdus)
    kr = uplift_rgb(table, eval_handle_spectrum(textures, mat.Kr, tfc), lambda)
    return pbrt_bsdf((Weighted(lobe, kr, 1f0, n, t),), lambda)
end

"""
    get_bxdf(mat, table, textures, tfc, lambda, regularize, wo, ns, dpdus) -> bxdf

Hikari's analogue of pbrt-v4's `Material::GetBxDF`: the BSDF of `mat` at a hit, with its textures
read through `tfc`, seen from `wo` in the shading frame of `ns` and `dpdus`, its spectra at the
path's wavelengths. A pbrt material's is a `LobeBSDF` (`pbrt_bsdf`).
"""
function get_bxdf end

# The layered materials, not yet lobes, resolve without the frame.
@inline get_bxdf(mat, table, textures, tfc, lambda, regularize::Bool, wo::Vec3f, ns::Vec3f, dpdus::Vec3f) =
    get_bxdf(mat, table, textures, tfc, lambda, regularize)
@inline get_bxdf(mat, table, textures, tfc, lambda, regularize::Bool) = mat
