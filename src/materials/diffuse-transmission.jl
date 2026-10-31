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
    DiffuseTransmission(; reflectance = (0.5, 0.5, 0.5), transmittance = (0.5, 0.5, 0.5), scale = 1,
                        displacement = nothing)

pbrt-v4's diffuse transmission: Lambert reflection of `reflectance` and Lambert transmission of
`transmittance`, both times `scale`, for paper, thin fabric or leaves. Blender's equivalent is an
`Add` of a `Diffuse` and a `Translucent`.
"""
struct DiffuseTransmission{ReflT, TransT, DispT} <: Material
    reflectance::ReflT
    transmittance::TransT
    scale::Float32
    displacement::DispT
end

DiffuseTransmission(; reflectance = (0.5f0, 0.5f0, 0.5f0), transmittance = (0.5f0, 0.5f0, 0.5f0), scale::Real = 1f0,
                    displacement = nothing) =
    DiffuseTransmission(reflectance, transmittance, Float32(scale), displacement)



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
