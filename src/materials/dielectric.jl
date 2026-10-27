# ============================================================================
# Dielectric (Glass) and ThinDielectric Materials
# Contains: Dielectric struct, keyword constructor, Dielectric alias,
#           ThinDielectric struct, keyword constructor, ThinDielectric alias,
#           eval_dielectric_ior helpers, and the BSDFs of both as lobes
# ============================================================================

"""
    Dielectric(Kr, Kt, u_roughness, v_roughness, index, remap_roughness)

Glass/dielectric material with reflection and transmission.

* `Kr`: Spectral reflectance (Texture or TextureRef)
* `Kt`: Spectral transmittance (Texture or TextureRef)
* `u_roughness`: Roughness in u direction (0 = perfect specular)
* `v_roughness`: Roughness in v direction (0 = perfect specular)
* `index`: Index of refraction
* `remap_roughness`: Whether to remap roughness to alpha
"""
# Non-parametric texture parameters: constant-vs-texture must not change the
# material's TYPE. Spectral IOR fields stay parametric because a
# PiecewiseLinearSpectrum cannot live inline in a handle.
struct Dielectric{KrT, KtT, URoughT, VRoughT, IndexT, DispT} <: Material
    Kr::KrT
    Kt::KtT
    u_roughness::URoughT
    v_roughness::VRoughT
    index::IndexT
    remap_roughness::Bool
    displacement::DispT       # pbrt-v4 `Material::displacement` height field (NONE = flat)
end

"""
    Dielectric(; Kr=RGBSpectrum(1), Kt=RGBSpectrum(1), roughness=0, index=1.5, remap_roughness=true)

Create a glass/dielectric material with reflection and transmission.

# Arguments
- `Kr`: Reflectance color
- `Kt`: Transmittance color
- `roughness`: Surface roughness (0 = perfect specular, can be single value or (u,v) tuple)
- `index`: Index of refraction (1.5 for glass, 1.33 for water, 2.4 for diamond)
- `remap_roughness`: Whether to remap roughness to microfacet alpha

# Examples
```julia
Dielectric()                                # Clear glass
Dielectric(Kt=(1, 0.9, 0.8), index=1.5)    # Amber tinted
Dielectric(roughness=0.1)                   # Frosted glass
Dielectric(roughness=(0.1, 0.05))          # Anisotropic roughness
```
"""
function Dielectric(;
    Kr=RGBSpectrum(1f0),
    Kt=RGBSpectrum(1f0),
    roughness=0f0,
    index=1.5f0,
    remap_roughness=true,
    bump=nothing,
)
    u_rough, v_rough = roughness_pair(roughness)
    Dielectric(
        matparam(Kr), matparam(Kt),
        matparam(u_rough), matparam(v_rough),
        matparam(index), remap_roughness, matparam(bump)
    )
end


# ============================================================================
# ThinDielectric - Thin dielectric surface (e.g., window glass)
# ============================================================================
# Port of pbrt-v4's ThinDielectric and ThinDielectricBxDF
#
# Models a thin dielectric surface where light can pass through without
# the usual refraction bend. This is appropriate for thin surfaces like
# window glass where internal bounces contribute to the overall reflection.
#
# Key difference from regular Dielectric:
# - Transmitted direction is -wo (straight through, no refraction)
# - Reflectance accounts for multiple internal bounces: R' = R + T²R/(1-R²)
# - Always specular (no roughness parameter)
#
# Reference: pbrt-v4 src/pbrt/bxdfs.h ThinDielectricBxDF (lines 209-277)

"""
    ThinDielectric

A thin dielectric material for surfaces like window glass.

Unlike regular dielectric materials which refract light according to Snell's law,
thin dielectric materials transmit light straight through (wi = -wo) while
accounting for multiple internal reflections within the thin layer.

# Fields
- `eta`: Index of refraction of the dielectric

# Physics
For a thin dielectric layer:
- Single-surface Fresnel: R₀ = FrDielectric(cos_θ, eta)
- Multiple-bounce reflectance: R = R₀ + T₀²R₀/(1 - R₀²) where T₀ = 1 - R₀
- Transmittance: T = 1 - R
- Transmitted direction: wi = -wo (straight through, no bend)

# Usage
```julia
# Thin glass window
window = ThinDielectric(eta=1.5)

# Thin plastic film
film = ThinDielectric(eta=1.4)
```
"""
struct ThinDielectric{E, DispT} <: Material
    eta::E  # Float32 or PiecewiseLinearSpectrum
    displacement::DispT       # pbrt-v4 `Material::displacement` height field (NONE = flat)
end

# Keyword constructor
function ThinDielectric(; eta=1.5f0, bump=nothing)
    ThinDielectric(matparam(eta), matparam(bump))
end

# Mark as non-emissive
is_emissive(::ThinDielectric) = false


# ============================================================================
# Helpers for dielectric IOR evaluation
# ============================================================================

# Evaluate dielectric IOR: returns (ior::Float32, is_dispersive::Bool)
# PiecewiseLinearSpectrum → sample at hero wavelength only, dispersive
@inline eval_dielectric_ior(textures, idx::PiecewiseLinearSpectrum, tfc, lambda) =
    (sample(idx, lambda.lambda[1]), true)
# Scalar/texture IOR → not dispersive
@propagate_inbounds eval_dielectric_ior(textures, idx::TexHandle, tfc, lambda) =
    (eval_handle(textures, idx, tfc), false)
@inline function eval_dielectric_ior(textures, idx, tfc, lambda)
    ior = eval_tex(textures, idx, tfc)
    return (ior, false)
end

"""
A dielectric's BSDF: pbrt-v4's `DielectricBxDF`, one GGX glass lobe with the
dielectric Fresnel, tinted by `Kr` and `Kt`, single scattering, in pbrt's convention. An IOR that is
a spectrum is taken at the path's first wavelength, and ends the others (`dispersive`).
"""
@propagate_inbounds function get_bxdf(mat::Dielectric, table, textures, tfc::TextureFilterContext, lambda::Wavelengths,
                                      regularize::Bool, wo::Vec3f, ns::Vec3f, dpdus::Vec3f)
    ior, dispersive = eval_dielectric_ior(textures, mat.index, tfc, lambda)
    ior == 0f0 && (ior = 1f0)
    u_roughness = eval_handle(textures, mat.u_roughness, tfc)
    v_roughness = eval_handle(textures, mat.v_roughness, tfc)
    alpha_x = mat.remap_roughness ? roughness_to_α(u_roughness) : u_roughness
    alpha_y = mat.remap_roughness ? roughness_to_α(v_roughness) : v_roughness
    if regularize
        alpha_x = regularize_alpha(alpha_x)
        alpha_y = regularize_alpha(alpha_y)
    end
    # The IOR across the surface from the side `wo` is on, as pbrt's `etap`.
    eta = dot(wo, ns) < 0f0 ? 1f0 / ior : ior
    lobe = pbrt_dielectric(eta, alpha_x, alpha_y;
                           kr = uplift_rgb(table, eval_handle_spectrum(textures, mat.Kr, tfc), lambda),
                           kt = uplift_rgb(table, eval_handle_spectrum(textures, mat.Kt, tfc), lambda))
    n, t = pbrt_frame(wo, ns, dpdus)
    return pbrt_bsdf((Weighted(lobe, SpectralRadiance(1f0), 1f0, n, t),), lambda; dispersive)
end

"""
A thin dielectric's BSDF: pbrt-v4's `ThinDielectricBxDF` (`ThinDielectricLobe`).
An IOR that is a spectrum is taken at the path's first wavelength, and ends the others.
"""
@propagate_inbounds function get_bxdf(mat::ThinDielectric, table, textures, tfc::TextureFilterContext, lambda::Wavelengths,
                                      regularize::Bool, wo::Vec3f, ns::Vec3f, dpdus::Vec3f)
    eta, dispersive = eval_dielectric_ior(textures, mat.eta, tfc, lambda)
    n, t = pbrt_frame(wo, ns, dpdus)
    return pbrt_bsdf((Weighted(ThinDielectricLobe(eta), SpectralRadiance(1f0), 1f0, n, t),), lambda; dispersive)
end
