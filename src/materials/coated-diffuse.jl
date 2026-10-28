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
    CoatedDiffuse(; reflectance = (0.5, 0.5, 0.5), roughness = 0, thickness = 0.01, ior = 1.5,
                  albedo = (0, 0, 0), g = 0, max_depth = 10, n_samples = 1, remap_roughness = true,
                  displacement = nothing)

pbrt-v4's coated diffuse: a dielectric coat over a diffuse base, with a slab of medium between them,
its BSDF estimated by a random walk between the two (pbrt's `LayeredBxDF`).

* `reflectance`: the base's albedo; its alpha cuts the surface out.
* `roughness`: the coat's, one value or `(u, v)`; with `remap_roughness` the GGX alpha is pbrt's of it.
* `thickness`, `albedo`, `g`: the slab's thickness, and its medium's single-scattering albedo
  (0 is none) and Henyey-Greenstein asymmetry.
* `ior`: the coat's, a number or a spectrum that disperses the path's wavelengths.
* `max_depth`, `n_samples`: the random walk's length and how many it averages.
"""
struct CoatedDiffuse{ReflT, URoughT, VRoughT, ThickT, IorT, AlbedoT, GT, DispT} <: Material
    reflectance::ReflT
    u_roughness::URoughT
    v_roughness::VRoughT
    thickness::ThickT
    ior::IorT
    albedo::AlbedoT
    g::GT
    max_depth::Int32
    n_samples::Int32
    remap_roughness::Bool
    displacement::DispT
end

CoatedDiffuse(; reflectance = (0.5f0, 0.5f0, 0.5f0), roughness = 0f0, thickness = 0.01f0, ior = 1.5f0,
              albedo = (0f0, 0f0, 0f0), g = 0f0, max_depth::Integer = 10, n_samples::Integer = 1,
              remap_roughness::Bool = true, displacement = nothing) =
    CoatedDiffuse(reflectance, roughness_pair(roughness)..., thickness, ior, albedo, g, Int32(max_depth),
                  Int32(n_samples), remap_roughness, displacement)



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
    eta, dispersive = eval_dielectric_ior(textures, mat.ior, tfc, lambda)
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
    Plastic(; color = (0.5, 0.5, 0.5), roughness = 0.1, ior = 1.5)

Plastic: a [`CoatedDiffuse`](@ref) of base `color`, as pbrt-v4 represents it.
"""
Plastic(; color = (0.5f0, 0.5f0, 0.5f0), roughness = 0.1f0, ior = 1.5f0) =
    CoatedDiffuse(; reflectance = color, roughness, ior)

# ============================================================================
# Surface Alpha Evaluation
# ============================================================================

"""
    surface_transparency(mat::CoatedDiffuse, textures, tfc) -> Vec3f

What the cutout alpha of the coat's base texture lets through, so alpha-masked geometry is cut
the same way it is under [`Diffuse`](@ref).

Without this the generic `Material` fallback returns 1 and the surface is fully
opaque. That is a silent loss on any asset whose silhouette lives in its alpha
channel: the raven demo model paints its feather shapes there — 38% of its map is
fully transparent and 30% of its triangles are partly cut — so giving the bird a
coat turned every feather card back into a solid quad.
"""
@propagate_inbounds surface_transparency(mat::CoatedDiffuse, textures, tfc::TextureFilterContext) =
    Vec3f(1f0 - eval_handle_alpha(textures, mat.reflectance, tfc))
