# ============================================================================
# pbrt-v4's dielectric and thin dielectric
# ============================================================================
#
# The glass models of pbrt-v4, for the pbrt importer, whose renders must match pbrt's. A user's
# `Glass` is the canonical Principled transmission preset in Cycles' convention.

"""
pbrt-v4's `DielectricMaterial`: one GGX glass lobe with the dielectric Fresnel, single scattering,
in pbrt's convention. `ior` is a number, a texture, or a spectrum that disperses the path's
wavelengths; the roughness along and across the tangent is remapped to the GGX alpha with
`remap_roughness`, as pbrt does.
"""
struct Dielectric{IorT, RoughT, DispT} <: Material
    ior::IorT
    roughness::RoughT
    remap_roughness::Bool
    displacement::DispT
end

Dielectric(; ior = 1.5f0, roughness = 0f0, remap_roughness = true, displacement = nothing) =
    Dielectric(ior, roughness, remap_roughness, displacement)

"""
pbrt-v4's `ThinDielectricMaterial`: a slab of no thickness that light passes straight through,
reflecting what its two interfaces and the light between them reflect (`ThinDielectricLobe`).
"""
struct ThinDielectric{IorT, DispT} <: Material
    ior::IorT
    displacement::DispT
end

ThinDielectric(; ior = 1.5f0, displacement = nothing) = ThinDielectric(ior, displacement)

# The IOR at the path's first wavelength, and whether it disperses the others: a spectrum does.
@inline eval_dielectric_ior(textures, idx::PiecewiseLinearSpectrum, tfc, lambda) =
    (sample(idx, lambda.lambda[1]), true)
@propagate_inbounds eval_dielectric_ior(textures, idx::TexHandle, tfc, lambda) =
    (eval_handle(textures, idx, tfc), false)

"""
A dielectric's BSDF: pbrt-v4's `DielectricBxDF`. An IOR that is a spectrum is taken at the path's
first wavelength, and ends the others (`dispersive`).
"""
@propagate_inbounds function get_bxdf(mat::Dielectric, table, textures, tfc::TextureFilterContext, lambda::Wavelengths,
                                      regularize::Bool, wo::Vec3f, ns::Vec3f, dpdus::Vec3f)
    ior, dispersive = eval_dielectric_ior(textures, mat.ior, tfc, lambda)
    ior == 0f0 && (ior = 1f0)
    u, v = roughness_pair(mat.roughness)
    u_roughness = eval_handle(textures, u, tfc)
    v_roughness = eval_handle(textures, v, tfc)
    alpha_x = mat.remap_roughness ? roughness_to_α(u_roughness) : u_roughness
    alpha_y = mat.remap_roughness ? roughness_to_α(v_roughness) : v_roughness
    if regularize
        alpha_x = regularize_alpha(alpha_x)
        alpha_y = regularize_alpha(alpha_y)
    end
    # The IOR across the surface from the side `wo` is on, as pbrt's `etap`.
    eta = dot(wo, ns) < 0f0 ? 1f0 / ior : ior
    lobe = pbrt_dielectric(eta, alpha_x, alpha_y)
    n, t = pbrt_frame(wo, ns, dpdus)
    return pbrt_bsdf((Weighted(lobe, SpectralRadiance(1f0), 1f0, n, t),), lambda; dispersive)
end

"""
A thin dielectric's BSDF: pbrt-v4's `ThinDielectricBxDF` (`ThinDielectricLobe`).
An IOR that is a spectrum is taken at the path's first wavelength, and ends the others.
"""
@propagate_inbounds function get_bxdf(mat::ThinDielectric, table, textures, tfc::TextureFilterContext, lambda::Wavelengths,
                                      regularize::Bool, wo::Vec3f, ns::Vec3f, dpdus::Vec3f)
    eta, dispersive = eval_dielectric_ior(textures, mat.ior, tfc, lambda)
    n, t = pbrt_frame(wo, ns, dpdus)
    return pbrt_bsdf((Weighted(ThinDielectricLobe(eta), SpectralRadiance(1f0), 1f0, n, t),), lambda; dispersive)
end
