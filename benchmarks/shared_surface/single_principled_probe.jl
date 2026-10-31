# Diagnostic only: fixtures must contain one Principled node. This removes the
# program traversal while retaining the same node setup, lifting and storage.
Base.@constprop :none @noinline function surface_setup_slots(::Val{N}, s, prog, i, sc, lut, table, lambda, scale, mode::Val{Looped}, mask::Val{Mask}) where {N, Looped, Mask}
    storage = MVector{N, SurfaceStoredWeight{Looped, Mask}}(surface_empty(Val(N), mode, mask))
    _, at, emission = surface_principled_store!(storage, Int32(1), s, prog, i, sc, lut, table, lambda, scale, mode, mask)
    return Tuple(storage), emission, at - Int32(1)
end
