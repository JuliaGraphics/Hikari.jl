# ============================================================================
# CoatedDiffuseTransmission - Layered: dielectric coating + diffuse transmission
# ============================================================================
# Extension of CoatedDiffuse that uses DiffuseTransmissionBxDF as the
# bottom layer instead of DiffuseBxDF. This creates a material with a glossy
# dielectric coating over a diffuse substrate that both reflects and transmits.
#
# Ideal for: leaves, thin fabric, paper with a glossy surface coating.
#
# In pbrt-v4 terms: LayeredBxDF<DielectricBxDF, DiffuseTransmissionBxDF, true>

"""
    CoatedDiffuseTransmission(; reflectance = (0.5, 0.5, 0.5), transmittance = (0.25, 0.25, 0.25), roughness = 0,
                              thickness = 0.01, ior = 1.5, albedo = (0, 0, 0), g = 0, max_depth = 10,
                              n_samples = 1, remap_roughness = true, displacement = nothing)

A dielectric coat over a [`DiffuseTransmission`](@ref) base, as [`CoatedDiffuse`](@ref) coats a
diffuse one (pbrt-v4's `LayeredBxDF<DielectricBxDF, DiffuseTransmissionBxDF, true>`): leaves, thin
fabric, paper with a glossy surface.
"""
struct CoatedDiffuseTransmission{ReflT, TransT, RoughT, ThickT, IorT, AlbedoT, GT, DispT} <: Material
    reflectance::ReflT
    transmittance::TransT
    roughness::RoughT
    thickness::ThickT
    ior::IorT
    albedo::AlbedoT
    g::GT
    max_depth::Int32
    n_samples::Int32
    remap_roughness::Bool
    displacement::DispT
end

CoatedDiffuseTransmission(; reflectance = (0.5f0, 0.5f0, 0.5f0), transmittance = (0.25f0, 0.25f0, 0.25f0),
                          roughness = 0f0, thickness = 0.01f0, ior = 1.5f0, albedo = (0f0, 0f0, 0f0), g = 0f0,
                          max_depth::Integer = 10, n_samples::Integer = 1, remap_roughness::Bool = true,
                          displacement = nothing) =
    CoatedDiffuseTransmission(reflectance, transmittance, roughness, thickness, ior, albedo, g,
                              Int32(max_depth), Int32(n_samples), remap_roughness, displacement)



"""
A coated diffuse transmission material's BSDF: pbrt-v4's
`LayeredBxDF<DielectricBxDF, DiffuseTransmissionBxDF, true>`, a `LayeredBSDF` of its dielectric coat
over Lambert reflection and transmission of its reflectance and transmittance, clamped to [0, 1].
"""
@propagate_inbounds function get_bxdf(mat::CoatedDiffuseTransmission, table, textures, tfc::TextureFilterContext,
                                      lambda::Wavelengths, regularize::Bool, wo::Vec3f, ns::Vec3f, dpdus::Vec3f)
    u, v = roughness_pair(mat.roughness)
    u_roughness = eval_handle(textures, u, tfc)
    v_roughness = eval_handle(textures, v, tfc)
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
    r = uplift_rgb(table, clamp(eval_handle_spectrum(textures, mat.reflectance, tfc)), lambda)
    t = uplift_rgb(table, clamp(eval_handle_spectrum(textures, mat.transmittance, tfc)), lambda)
    top = (local_lobe(pbrt_dielectric(eta, alpha_x, alpha_y), SpectralRadiance(1f0)),)
    bottom = (local_lobe(LambertLobe(SpectralRadiance), r, maximum(r.data)), local_lobe(TranslucentLobe(), t, maximum(t.data)))
    return LayeredBSDF{true}(top, bottom, ns, dpdus, thickness, albedo, g, mat.max_depth, mat.n_samples, lambda;
                             dispersive)
end
