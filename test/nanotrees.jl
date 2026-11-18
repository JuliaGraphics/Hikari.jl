# NanoTrees for host references: built by the device kernels, their words read back.
if !@isdefined(hosttree)
    function hosttree(origins, values, lo, voxel; background = 0f0, device = Mantle.Device())
        t = Hikari.NanoTree(Mantle.devicearray(device, origins), Mantle.devicearray(device, values), lo, voxel;
                            background, backend = device)
        return Hikari.NanoTree(Array(t.words), t.upper, t.lower, t.leaf, t.uppers, t.lowers, t.leaves, t.roottiles,
                               t.lo, t.voxel, t.background, t.imin, t.imax)
    end
end
