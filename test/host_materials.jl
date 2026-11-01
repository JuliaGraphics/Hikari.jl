# Shared host reference fixture for material dispatch and texture evaluation.
# The production store keeps device handles; the tests need owned host arrays.
if !@isdefined(host_materials)
    # Download texture owners, not the device pointer table: Array(static.textures)
    # copies handles whose payloads still name device memory.
    function host_materials(mats)
        data = Tuple(copy(mats.data_vectors[T]) for T in mats.data_order)
        textures = Tuple([Array(a) for a in mats.texture_gpu_arrays if typeof(a) === T]
                         for T in mats.texture_order)
        return Raycore.StaticMultiTypeSet(data, textures)
    end
end

if !@isdefined(host_bxdf)
# Resolve the stored graph through the same BSDF construction used by the renderer.
# Non-graph carriers (including BSDF-less materials) still use get_bxdf directly.
function host_bxdf(mat::Hikari.ShaderMaterial, table, textures, tfc, lambda, regularize, wo, ns, tangent)
    sc = Hikari.ShaderContext(tfc, wo, Hikari.Vec3f(0), Hikari.Vec3f(0), tangent, ns,
                             1f0, Hikari.Vec3f(0), Hikari.Point2f(0))
    program = Hikari.shader_programs(textures, mat.program)[mat.program.idx]
    return Hikari.shader_bsdf(Hikari.bsdf_nodes(typeof(mat)), program, textures, sc, regularize, table, lambda)
end
host_bxdf(mat, args...) = Hikari.get_bxdf(mat, args...)
end
