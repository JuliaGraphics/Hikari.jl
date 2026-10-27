# ============================================================================
# CoatedDiffuse - Layered material with dielectric coating over diffuse
# ============================================================================
# Port of pbrt-v4's CoatedDiffuse using LayeredBxDF
#
# The material consists of:
# - Top layer: Dielectric interface (can be rough or smooth)
# - Bottom layer: Diffuse reflector
# - Optional absorbing medium between layers
#
# Reference: pbrt-v4 src/pbrt/bxdfs.h LayeredBxDF, CoatedDiffuseBxDF

"""
    CoatedDiffuse

A layered material with a dielectric coating over a diffuse base.
This implements pbrt-v4's coateddiffuse material using random walk
sampling between the layers (LayeredBxDF algorithm).

# Fields
- `reflectance`: Diffuse reflectance of the base layer (RGB color)
- `u_roughness`: Roughness in U direction for the dielectric coating
- `v_roughness`: Roughness in V direction for the dielectric coating
- `thickness`: Thickness of the coating layer (affects absorption)
- `eta`: Index of refraction of the dielectric coating
- `albedo`: Single-scattering albedo of medium between layers (0 = no absorption)
- `g`: Henyey-Greenstein asymmetry parameter for medium scattering
- `max_depth`: Maximum random walk depth
- `n_samples`: Number of samples for estimating the BSDF
- `remap_roughness`: Whether to remap roughness to microfacet alpha
"""
# Non-parametric texture parameters: constant-vs-texture must not change the
# material's TYPE, or the per-material shading compiles a separate kernel per
# combination.
struct CoatedDiffuse{ReflT, URoughT, VRoughT, ThickT, EtaT, AlbedoT, GT, DispT} <: Material
    reflectance::ReflT
    u_roughness::URoughT
    v_roughness::VRoughT
    thickness::ThickT
    eta::EtaT               # Index of refraction: a number, or a spectrum that disperses
    albedo::AlbedoT
    g::GT
    max_depth::Int32
    n_samples::Int32
    remap_roughness::Bool
    displacement::DispT       # pbrt-v4 `Material::displacement` height field (NONE = flat)
end

# Positional constructor taking Ints for the two counters; the struct's own
# synthesized constructor converts every other argument through `TexHandle`.
function CoatedDiffuse(
    reflectance, u_roughness, v_roughness, thickness, eta,
    albedo, g, max_depth::Int, n_samples::Int, remap_roughness::Bool,
    displacement = TexHandle(),
)
    CoatedDiffuse(
        matparam(reflectance), matparam(u_roughness), matparam(v_roughness),
        matparam(thickness), matparam(eta), matparam(albedo), matparam(g),
        Int32(max_depth), Int32(n_samples), remap_roughness, matparam(displacement),
    )
end

"""
    CoatedDiffuse(; reflectance, roughness=0, thickness=0.01, eta=1.5, ...)

Create a coated diffuse material with keyword arguments.

# Arguments
- `reflectance`: Diffuse color (RGBSpectrum, tuple, or Texture)
- `roughness`: Surface roughness (scalar or (u,v) tuple)
- `thickness`: Coating thickness (default 0.01)
- `eta`: Index of refraction (default 1.5 for typical dielectric), or a spectrum (`GLASS_BK7_ETA_SPECTRUM`)
- `albedo`: Medium albedo for absorption (default 0 = no medium)
- `g`: HG asymmetry parameter (default 0 = isotropic)
- `max_depth`: Max random walk depth (default 10)
- `n_samples`: Number of samples (default 1)
- `remap_roughness`: Remap roughness to alpha (default true)

# Examples
```julia
# Simple coated diffuse (glossy plastic-like)
CoatedDiffuse(reflectance=(0.4, 0.45, 0.35), roughness=0)

# Rough coating
CoatedDiffuse(reflectance=(0.8, 0.2, 0.2), roughness=0.3)

# With absorbing medium
CoatedDiffuse(reflectance=(0.9, 0.9, 0.9), albedo=(0.8, 0.4, 0.2))
```
"""
function CoatedDiffuse(;
    reflectance = RGBSpectrum(0.5f0),
    roughness = 0f0,
    thickness = 0.01f0,
    eta = 1.5f0,
    albedo = RGBSpectrum(0f0),
    g = 0f0,
    max_depth::Int = 10,
    n_samples::Int = 1,
    remap_roughness::Bool = true,
    bump = nothing,
)
    u_rough, v_rough = roughness_pair(roughness)

    CoatedDiffuse(
        matparam(reflectance),
        matparam(u_rough),
        matparam(v_rough),
        matparam(thickness),
        eta,
        matparam(albedo),
        matparam(g),
        max_depth,
        n_samples,
        remap_roughness,
        matparam(bump),
    )
end

# Mark as non-emissive
is_emissive(::CoatedDiffuse) = false


"""
A coated diffuse material's BSDF: pbrt-v4's `CoatedDiffuseBxDF`, a `LayeredBSDF` of its dielectric
coat over Lambert of its reflectance clamped to [0, 1].
"""
@propagate_inbounds function get_bxdf(mat::CoatedDiffuse, table, textures, tfc::TextureFilterContext, lambda::Wavelengths,
                                      regularize::Bool, wo::Vec3f, ns::Vec3f, dpdus::Vec3f)
    u_roughness = eval_handle(textures, mat.u_roughness, tfc)
    v_roughness = eval_handle(textures, mat.v_roughness, tfc)
    alpha_x = mat.remap_roughness ? roughness_to_α(u_roughness) : u_roughness
    alpha_y = mat.remap_roughness ? roughness_to_α(v_roughness) : v_roughness
    if regularize
        alpha_x = regularize_alpha(alpha_x)
        alpha_y = regularize_alpha(alpha_y)
    end
    thickness = max(eval_handle(textures, mat.thickness, tfc), eps(Float32))
    albedo = uplift_rgb(table, clamp(eval_handle_spectrum(textures, mat.albedo, tfc)), lambda)
    g = clamp(eval_handle(textures, mat.g, tfc), -0.99f0, 0.99f0)
    eta, dispersive = eval_dielectric_ior(textures, mat.eta, tfc, lambda)
    eta == 0f0 && (eta = 1f0)
    reflectance = uplift_rgb(table, clamp(eval_handle_spectrum(textures, mat.reflectance, tfc)), lambda)
    top = (local_lobe(pbrt_dielectric(eta, alpha_x, alpha_y), SpectralRadiance(1f0)),)
    bottom = (local_lobe(LambertLobe(SpectralRadiance), reflectance),)
    return LayeredBSDF{true}(top, bottom, ns, dpdus, thickness, albedo, g, mat.max_depth, mat.n_samples, lambda;
                             dispersive)
end

# ============================================================================
# Plastic — convenience preset for CoatedDiffuse (like Gold() for Conductor)
# ============================================================================

"""
    Plastic(; color=(0.5, 0.5, 0.5), roughness=0.1, eta=1.5)

Create a plastic-like material. Returns a `CoatedDiffuse` with a dielectric
coating (eta ≈ 1.5) over a diffuse base — the standard way to represent
plastic in pbrt-v4.

# Examples
```julia
Plastic()                                    # grey plastic
Plastic(color=(0.8, 0.1, 0.1))              # red plastic
Plastic(color=(0.9, 0.9, 0.9), roughness=0.01)  # glossy white
```
"""
function Plastic(; color=(0.5f0, 0.5f0, 0.5f0), roughness=0.1f0, eta=1.5f0)
    CoatedDiffuse(
        reflectance=color,
        roughness=Float32(roughness),
        eta=eta,
    )
end

# ============================================================================
# Surface Alpha Evaluation
# ============================================================================

"""
    get_surface_alpha(mat::CoatedDiffuse, textures, tfc) -> Float32

The cutout alpha of the coat's base texture, so alpha-masked geometry is cut the
same way it is under [`Diffuse`](@ref).

Without this the generic `Material` fallback returns 1 and the surface is fully
opaque. That is a silent loss on any asset whose silhouette lives in its alpha
channel: the raven demo model paints its feather shapes there — 38% of its map is
fully transparent and 30% of its triangles are partly cut — so giving the bird a
coat turned every feather card back into a solid quad.
"""
@propagate_inbounds function get_surface_alpha(mat::CoatedDiffuse, textures, tfc::TextureFilterContext)
    return eval_handle_alpha(textures, mat.reflectance, tfc)
end
