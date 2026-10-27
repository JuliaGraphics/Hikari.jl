# ============================================================================
# DiffuseTransmission - Diffuse reflection and transmission
# ============================================================================
# Port of pbrt-v4's DiffuseTransmission and DiffuseTransmissionBxDF
#
# This material models surfaces that scatter light diffusely in both
# reflection and transmission, like thin cloth, paper, or leaves.
#
# Reference: pbrt-v4 src/pbrt/bxdfs.h DiffuseTransmissionBxDF (lines 84-164)

"""
    DiffuseTransmission{RTex, TTex}

A material that diffusely reflects and transmits light.

Models surfaces like paper, thin fabric, or leaves where light scatters
diffusely on both sides. The reflection and transmission are independent
Lambertian distributions.

# Fields
- `reflectance`: Diffuse reflectance color (same hemisphere as incident)
- `transmittance`: Diffuse transmittance color (opposite hemisphere)
- `scale`: Intensity multiplier applied to both R and T

# Physics
- Reflection: f = R/π (same hemisphere)
- Transmission: f = T/π (opposite hemisphere)
- Sampling: probability proportional to max(R) and max(T)

# Usage
```julia
# Thin white paper (equal reflection and transmission)
paper = DiffuseTransmission(reflectance=(0.8, 0.8, 0.8), transmittance=(0.5, 0.5, 0.5))

# Green leaf (green transmission, less reflection)
leaf = DiffuseTransmission(reflectance=(0.2, 0.3, 0.1), transmittance=(0.1, 0.5, 0.1))
```
"""
# Non-parametric texture parameters: constant-vs-texture must not change the
# material's TYPE, or the per-material shading compiles a separate kernel per
# combination.
struct DiffuseTransmission{ReflT, TransT, DispT} <: Material
    reflectance::ReflT
    transmittance::TransT
    scale::Float32       # Intensity scale
    displacement::DispT       # pbrt-v4 `Material::displacement` height field (NONE = flat)
end

# No explicit positional constructor: the synthesized
# `DiffuseTransmission(reflectance, transmittance, scale)` already accepts
# any texture-like value (Texture, CheckerboardTexture, raw constants,
# TextureRef). Re-declaring it with the identical signature is a method
# overwrite, which breaks precompilation.

"""
    DiffuseTransmission(; reflectance, transmittance, scale=1.0)

Create a diffuse transmission material with keyword arguments.

# Arguments
- `reflectance`: Diffuse reflection color (RGBSpectrum, tuple, or Texture)
- `transmittance`: Diffuse transmission color (RGBSpectrum, tuple, or Texture)
- `scale`: Intensity multiplier (default 1.0)

# Examples
```julia
# Thin translucent material
DiffuseTransmission(reflectance=(0.5, 0.5, 0.5), transmittance=(0.3, 0.3, 0.3))

# Pure transmission (no reflection)
DiffuseTransmission(reflectance=(0, 0, 0), transmittance=(1, 1, 1))
```
"""
function DiffuseTransmission(;
    reflectance = RGBSpectrum(0.5f0),
    transmittance = RGBSpectrum(0.5f0),
    scale::Real = 1f0,
    bump = nothing,
)
    DiffuseTransmission(
        matparam(reflectance),
        matparam(transmittance),
        Float32(scale),
        matparam(bump),
    )
end

# Mark as non-emissive
is_emissive(::DiffuseTransmission) = false


"""
The BSDF of a diffuse transmission material: pbrt-v4's
`DiffuseTransmissionBxDF`, Lambert reflection of `R` and transmission of `T`, both scaled and
clamped to [0, 1], sampled in proportion to their largest values at the path's wavelengths.
"""
@propagate_inbounds function get_bxdf(mat::DiffuseTransmission, table, textures, tfc::TextureFilterContext, lambda::Wavelengths,
                                      regularize::Bool, wo::Vec3f, ns::Vec3f, dpdus::Vec3f)
    r = uplift_rgb(table, clamp(eval_handle_spectrum(textures, mat.reflectance, tfc) * mat.scale), lambda)
    t_ = uplift_rgb(table, clamp(eval_handle_spectrum(textures, mat.transmittance, tfc) * mat.scale), lambda)
    n, t = pbrt_frame(wo, ns, dpdus)
    return pbrt_bsdf((Weighted(LambertLobe(SpectralRadiance), r, maximum(r.data), n, t),
                      Weighted(TranslucentLobe(), t_, maximum(t_.data), n, t)), lambda)
end
