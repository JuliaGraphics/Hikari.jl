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
