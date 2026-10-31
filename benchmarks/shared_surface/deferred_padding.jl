# Optional counted-tuple experiment: initialize padding after live setup.
Base.@constprop :none @noinline function surface_setup_slots(::Val{N}, s, prog, i, sc, lut, table, lambda, scale, mode::Val{Looped}, mask::Val{Mask}) where {N, Looped, Mask}
    storage = MVector{N, SurfaceStoredWeight{Looped, Mask}}(undef)
    at = Int32(1); emission = Vec3f(0)
    @inbounds while prog[i].op != NodeOp.RESULT
        i, prefix = surface_zero_prefix(s, prog, i, sc)
        emission += prefix
        prog[i].op == NodeOp.RESULT && break
        i, at, e = surface_store_dispatch!(storage, at, s, prog, i, sc, lut, table, lambda, scale, mode, mask)
        emission += e
    end
    # Only initialize unused slots. Every live slot was written in full above;
    # all padding is initialized before the immutable tuple is constructed.
    empty = first(surface_empty(Val(1), mode, mask))
    @inbounds for j in at:N
        storage[j] = empty
    end
    return Tuple(storage), emission, at - Int32(1)
end
