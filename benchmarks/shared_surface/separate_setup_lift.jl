# Isolation variant: finish the RGB node setup before lifting any of its terms.
# This keeps exactly the original physics and one setup per node, while changing
# the lifetimes of setup temporaries relative to the spectral lookup temporaries.
@inline function surface_principled_mode!(thin::Val, storage, at, p, sc, lut, table, lambda, scale, mode, mask, multi, convention)
    terms, _, emission = principled_lobes(thin, Val(surface_has_walk(mask)), p, sc, lut, multi, convention)
    return surface_append!(storage, terms, at, table, lambda, scale, mode, mask), emission
end
@inline function surface_principled_store!(storage, at, s, prog, i, sc, lut, table, lambda, scale, mode, mask::Val{Mask}) where {Mask}
    p = principled_inputs(s, prog, i, sc)
    @inbounds distribution = prog[i].mode[2]
    if !iszero(Mask & (UInt32(1) << 18)) && distribution == 0x02
        if p.thin_wall > 0.5f0
            after, emission = surface_principled_mode!(Val(true), storage, at, p, sc, lut, table, lambda, scale, mode, mask, false, CyclesBeckmann())
        else
            after, emission = surface_principled_mode!(Val(false), storage, at, p, sc, lut, table, lambda, scale, mode, mask, false, CyclesBeckmann())
        end
    elseif p.thin_wall > 0.5f0
        after, emission = surface_principled_mode!(Val(true), storage, at, p, sc, lut, table, lambda, scale, mode, mask, distribution == 0x00, CyclesGGX())
    else
        after, emission = surface_principled_mode!(Val(false), storage, at, p, sc, lut, table, lambda, scale, mode, mask, distribution == 0x00, CyclesGGX())
    end
    return next_node(prog, i, instruction_count(PrincipledBSDF)), after, emission
end
