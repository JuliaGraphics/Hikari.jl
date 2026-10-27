# ============================================================================
# Diffuse (Matte) Material
# Contains: Diffuse struct, keyword constructor, Diffuse alias,
#           and its BSDF as a lobe
# ============================================================================

"""
    Diffuse(Kd::Texture, σ::Texture)

Matte (diffuse) material: pbrt-v4's Lambertian `DiffuseBxDF`, or with roughness the
energy-preserving Oren-Nayar lobe of Blender's Diffuse BSDF (OpenPBR's).

* `Kd`: Spectral diffuse reflection (color texture or TextureRef)
* `σ`: Roughness in degrees, 0 to 90: `σ/90` is the Oren-Nayar roughness (0 = Lambertian)
"""
# Non-parametric: constant-vs-texture must not change the material's TYPE, or
# the per-material shading compiles a separate kernel for each combination.
struct Diffuse{KdT, SigmaT, DispT} <: Material
    Kd::KdT
    σ::SigmaT
    displacement::DispT       # pbrt-v4 `Material::displacement` height field (NONE = flat)
end


"""
    Diffuse(; Kd=RGBSpectrum(0.5), σ=0.0)

Create a matte (diffuse) material with optional Oren-Nayar roughness.

# Arguments
- `Kd`: Diffuse color - can be RGBSpectrum, (r,g,b) tuple, or Texture
- `σ`: Roughness in degrees, 0 to 90 (0 = Lambertian, >0 = energy-preserving Oren-Nayar of roughness `σ/90`)

# Examples
```julia
Diffuse(Kd=RGBSpectrum(0.8, 0.2, 0.2))  # Red matte
Diffuse(Kd=(0.8, 0.2, 0.2), σ=20)       # Red with roughness
Diffuse(Kd=my_texture)                   # Textured
```
"""
function Diffuse(; Kd=RGBSpectrum(0.5f0), σ=0f0, bump=nothing)
    Diffuse(matparam(Kd), matparam(σ), matparam(bump))
end


"""
A diffuse material's BSDF: Lambert of `Kd` clamped to [0, 1], as pbrt-v4's
`DiffuseMaterial`, or the energy-preserving Oren-Nayar lobe with roughness.
"""
@propagate_inbounds function get_bxdf(mat::Diffuse, table, textures, tfc::TextureFilterContext, lambda::Wavelengths,
                                      regularize::Bool, wo::Vec3f, ns::Vec3f, dpdus::Vec3f)
    kd = clamp(eval_handle_spectrum(textures, mat.Kd, tfc))
    roughness = clamp(eval_handle(textures, mat.σ, tfc) / 90f0, 0f0, 1f0)
    n, t = pbrt_frame(wo, ns, dpdus)
    lobe = lift(DiffuseLobe(roughness, Vec3f(kd.c[1], kd.c[2], kd.c[3]), n, wo), table, lambda, regularize)
    return pbrt_bsdf((Weighted(lobe, uplift_rgb(table, kd, lambda), 1f0, n, t),), lambda)
end

# ============================================================================
# Surface Alpha Evaluation
# ============================================================================

@propagate_inbounds function get_surface_alpha(mat::Diffuse, textures, tfc::TextureFilterContext)
    return eval_handle_alpha(textures, mat.Kd, tfc)
end
