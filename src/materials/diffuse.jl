# ============================================================================
# Diffuse
# ============================================================================

"""
    Diffuse(; color = (0.8, 0.8, 0.8), roughness = 0, normal = nothing, displacement = nothing)

Blender's Diffuse BSDF: Lambert, or with `roughness` (0 to 1) the energy-preserving Oren-Nayar lobe
Cycles and OpenPBR use. Any field takes a number, a colour, an image or a node graph.

* `color`: the albedo; its alpha cuts the surface out.
* `normal`: the shading normal, for a `bump` or `normal_map` node; `nothing` is the surface's own.
* `displacement`: a height the shading normal is bumped by, as pbrt's `displacement`.

A `Diffuse` with a `normal` is shaded as a shader graph; without one by its own kernel, which
renders the same lobe faster.
"""
@description struct Diffuse <: Material
    color = (0.8, 0.8, 0.8)
    roughness = 0
    normal = nothing
    displacement = nothing
end

# A normal changes Diffuse to a ShaderMaterial before conversion. Its own
# kernel never reads this field, so absence stays a zero-size Nothing instead
# of another 32-byte handle in every material record.
to_device_material(store, mat::Diffuse{C, R, Nothing, D}) where {C, R, D} =
    Diffuse(device_param(store, mat.color), device_param(store, mat.roughness), nothing,
            device_param(store, mat.displacement))

"""
A diffuse material's BSDF: Lambert of `color` clamped to [0, 1], as pbrt-v4's `DiffuseMaterial`, or
the energy-preserving Oren-Nayar lobe with roughness.
"""
@propagate_inbounds function get_bxdf(mat::Diffuse, table, textures, tfc::TextureFilterContext, lambda::Wavelengths,
                                      regularize::Bool, wo::Vec3f, ns::Vec3f, dpdus::Vec3f)
    kd = clamp(eval_handle_spectrum(textures, mat.color, tfc))
    roughness = clamp(eval_handle(textures, mat.roughness, tfc), 0f0, 1f0)
    n, t = pbrt_frame(wo, ns, dpdus)
    lobe = lift(DiffuseLobe(roughness, Vec3f(kd.c[1], kd.c[2], kd.c[3]), n, wo), table, lambda)
    return pbrt_bsdf((Weighted(lobe, uplift_rgb(table, kd, lambda), 1f0, n, t),), lambda)
end

@propagate_inbounds surface_transparency(mat::Diffuse, textures, tfc::TextureFilterContext) =
    Vec3f(1f0 - eval_handle_alpha(textures, mat.color, tfc))
