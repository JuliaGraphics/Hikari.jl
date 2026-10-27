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

# Non-parametric texture parameters: constant-vs-texture must not change the
# material's TYPE, or the per-material shading compiles a separate kernel per
# combination.
struct CoatedDiffuseTransmission{ReflT, TransT, URoughT, VRoughT, ThickT, EtaT, AlbedoT, GT, DispT} <: Material
    reflectance::ReflT
    transmittance::TransT
    u_roughness::URoughT
    v_roughness::VRoughT
    thickness::ThickT
    eta::EtaT               # IOR of the coating: a number, or a spectrum that disperses
    albedo::AlbedoT
    g::GT
    max_depth::Int32
    n_samples::Int32
    remap_roughness::Bool
    displacement::DispT       # pbrt-v4 `Material::displacement` height field (NONE = flat)
end

# Full constructor. Texture args are unannotated on purpose: fields accept any
# texture-like value (Texture, CheckerboardTexture, raw constants, TextureRef).
function CoatedDiffuseTransmission(
    reflectance, transmittance, u_roughness, v_roughness, thickness,
    eta, albedo, g, max_depth::Int, n_samples::Int,
    remap_roughness::Bool, displacement = TexHandle(),
)
    CoatedDiffuseTransmission(
        matparam(reflectance), matparam(transmittance),
        matparam(u_roughness), matparam(v_roughness), matparam(thickness),
        matparam(eta), matparam(albedo), matparam(g),
        Int32(max_depth), Int32(n_samples), remap_roughness, matparam(displacement),
    )
end

function CoatedDiffuseTransmission(;
    reflectance = RGBSpectrum(0.5f0),
    transmittance = RGBSpectrum(0.25f0),
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

    CoatedDiffuseTransmission(
        matparam(reflectance),
        matparam(transmittance),
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

is_emissive(::CoatedDiffuseTransmission) = false


"""
A coated diffuse transmission material's BSDF: pbrt-v4's
`LayeredBxDF<DielectricBxDF, DiffuseTransmissionBxDF, true>`, a `LayeredBSDF` of its dielectric coat
over Lambert reflection and transmission of its reflectance and transmittance, clamped to [0, 1].
"""
@propagate_inbounds function get_bxdf(mat::CoatedDiffuseTransmission, table, textures, tfc::TextureFilterContext,
                                      lambda::Wavelengths, regularize::Bool, wo::Vec3f, ns::Vec3f, dpdus::Vec3f)
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
    r = uplift_rgb(table, clamp(eval_handle_spectrum(textures, mat.reflectance, tfc)), lambda)
    t = uplift_rgb(table, clamp(eval_handle_spectrum(textures, mat.transmittance, tfc)), lambda)
    top = (local_lobe(pbrt_dielectric(eta, alpha_x, alpha_y), SpectralRadiance(1f0)),)
    bottom = (local_lobe(LambertLobe(SpectralRadiance), r, maximum(r.data)), local_lobe(TranslucentLobe(), t, maximum(t.data)))
    return LayeredBSDF{true}(top, bottom, ns, dpdus, thickness, albedo, g, mat.max_depth, mat.n_samples, lambda;
                             dispersive)
end
