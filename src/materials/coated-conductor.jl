# ============================================================================
# CoatedConductor - Layered material with dielectric coating over conductor
# ============================================================================
# Port of pbrt-v4's CoatedConductor using LayeredBxDF
#
# The material consists of:
# - Top layer: Dielectric interface (can be rough or smooth)
# - Bottom layer: Conductor (metal)
# - Optional absorbing medium between layers
#
# Reference: pbrt-v4 src/pbrt/materials.cpp CoatedConductor::GetBxDF (lines 345-392)
# Reference: pbrt-v4 src/pbrt/bxdfs.h CoatedConductorBxDF (lines 911-918)

"""
    CoatedConductor(; interface_roughness = 0, interface_ior = 1.5, conductor_ior, conductor_extinction,
                    reflectance, conductor_roughness = 0.01, thickness = 0.01, albedo = (0, 0, 0), g = 0,
                    max_depth = 10, n_samples = 1, remap_roughness = true, displacement = nothing)

pbrt-v4's coated conductor: a dielectric coat over a [`Conductor`](@ref), with a slab of medium
between them, its BSDF estimated by a random walk between the two (pbrt's `LayeredBxDF`).

* `interface_roughness`, `interface_ior`: the coat's roughness (one value or `(u, v)`) and IOR,
  a number or a spectrum that disperses the path's wavelengths.
* `conductor_ior`, `conductor_extinction`: the metal's complex IOR, colours or spectra; or instead
  its `reflectance`, its colour at normal incidence. Neither is pbrt's default, copper.
* `conductor_roughness`: the metal's, one value or `(u, v)`.
* `thickness`, `albedo`, `g`, `max_depth`, `n_samples`, `remap_roughness`: as [`CoatedDiffuse`](@ref)'s.

The conductor's IOR is taken relative to the coat's, as pbrt-v4 does.
"""
struct CoatedConductor{IRoughT, IIorT, CIorT, CExtT, CReflT, CRoughT, ThickT, AlbedoT, GT, DispT} <: Material
    interface_roughness::IRoughT
    interface_ior::IIorT
    conductor_ior::CIorT
    conductor_extinction::CExtT
    reflectance::CReflT
    conductor_roughness::CRoughT
    thickness::ThickT
    albedo::AlbedoT
    g::GT
    max_depth::Int32
    n_samples::Int32
    remap_roughness::Bool
    displacement::DispT
end

function CoatedConductor(; interface_roughness = 0f0, interface_ior = 1.5f0, conductor_ior = nothing,
                         conductor_extinction = nothing, reflectance = nothing, conductor_roughness = 0.01f0,
                         thickness = 0.01f0, albedo = (0f0, 0f0, 0f0), g = 0f0, max_depth::Integer = 10,
                         n_samples::Integer = 1, remap_roughness::Bool = true, displacement = nothing)
    if !isnothing(reflectance)
        (isnothing(conductor_ior) && isnothing(conductor_extinction)) ||
            throw(ArgumentError("a CoatedConductor takes either `conductor_ior` and `conductor_extinction` or a `reflectance`, not both"))
    elseif !isnothing(conductor_ior)
        isnothing(conductor_extinction) &&
            throw(ArgumentError("`conductor_extinction` must be given with `conductor_ior`"))
    else
        conductor_ior, conductor_extinction = CU_ETA_SPECTRUM, CU_K_SPECTRUM
    end
    return CoatedConductor(interface_roughness, interface_ior, conductor_ior, conductor_extinction,
                           reflectance, conductor_roughness, thickness, albedo, g,
                           Int32(max_depth), Int32(n_samples), remap_roughness, displacement)
end



"""
A coated conductor's BSDF: pbrt-v4's `CoatedConductorBxDF`, a `LayeredBSDF` of its dielectric coat
over its conductor, whose IOR is relative to the coat's.
"""
@propagate_inbounds function get_bxdf(mat::CoatedConductor, table, textures, tfc::TextureFilterContext,
                                      lambda::Wavelengths, regularize::Bool, wo::Vec3f, ns::Vec3f, dpdus::Vec3f)
    ieta, dispersive = eval_dielectric_ior(textures, mat.interface_ior, tfc, lambda)
    ieta == 0f0 && (ieta = 1f0)
    iu, iv = roughness_pair(mat.interface_roughness)
    iu_roughness = eval_handle(textures, iu, tfc)
    iv_roughness = eval_handle(textures, iv, tfc)
    i_alpha_x = mat.remap_roughness ? roughness_to_α(iu_roughness) : iu_roughness
    i_alpha_y = mat.remap_roughness ? roughness_to_α(iv_roughness) : iv_roughness
    cu, cv = roughness_pair(mat.conductor_roughness)
    cu_roughness = eval_handle(textures, cu, tfc)
    cv_roughness = eval_handle(textures, cv, tfc)
    c_alpha_x = mat.remap_roughness ? roughness_to_α(cu_roughness) : cu_roughness
    c_alpha_y = mat.remap_roughness ? roughness_to_α(cv_roughness) : cv_roughness
    if regularize
        i_alpha_x = regularize_alpha(i_alpha_x)
        i_alpha_y = regularize_alpha(i_alpha_y)
        c_alpha_x = regularize_alpha(c_alpha_x)
        c_alpha_y = regularize_alpha(c_alpha_y)
    end
    thickness = max(eval_handle(textures, mat.thickness, tfc), eps(Float32))
    albedo = uplift_rgb(table, clamp(eval_handle_spectrum(textures, mat.albedo, tfc)), lambda)
    g = clamp(eval_handle(textures, mat.g, tfc), -0.99f0, 0.99f0)
    # pbrt-v4 materials.cpp: the conductor's IOR relative to the coat's.
    ce, ck = conductor_ior(table, textures, mat.conductor_ior, mat.conductor_extinction, mat.reflectance, tfc, lambda)
    ce, ck = ce / ieta, ck / ieta
    top = (local_lobe(pbrt_dielectric(ieta, i_alpha_x, i_alpha_y), SpectralRadiance(1f0)),)
    bottom = (local_lobe(pbrt_conductor(ce, ck, c_alpha_x, c_alpha_y), SpectralRadiance(1f0)),)
    return LayeredBSDF{true}(top, bottom, ns, dpdus, thickness, albedo, g, mat.max_depth, mat.n_samples, lambda;
                             dispersive)
end
