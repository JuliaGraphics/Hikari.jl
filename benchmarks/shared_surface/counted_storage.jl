# Optional experiment: only lift/store live terms and carry their count across
# setup and evaluation. The capacity and capability mask still bound kernel types.
struct SurfaceLobeSpan{L <: Tuple} <: AbstractLobeStorage
    data::L
    count::Int32
end
@inline surface_ready(lobes) = lobes
@inline Base.getindex(s::SurfaceLobeSpan, i) = s.data[i]
@inline Base.length(s::SurfaceLobeSpan) = Int(s.count)
@inline Base.map(f, s::SurfaceLobeSpan) = SurfaceLobeSpan(map(f, s.data), s.count)
@inline function share(s::SurfaceLobeSpan, restriction::Scattering)
    total = 0.0f0
    @inbounds for i in 1:s.count
        total += share(s.data[i], restriction)
    end
    return total
end
@inline mixture(s::SurfaceLobeSpan, wo::Vec3f, wi::Vec3f, ns::Vec3f, lambda::Wavelengths, restriction::Scattering = AnyScattering()) =
    surface_mixture(s.data, s.count, wo, wi, ns, lambda, restriction)
@inline pick_sample(s::SurfaceLobeSpan, target::Float32, before::Float32, wo::Vec3f, u::Point2f, lambda::Wavelengths, restriction::Scattering = AnyScattering()) =
    surface_pick_sample(s.data, s.count, target, before, wo, u, lambda, restriction)
@inline terminates_at_vertex(::LobeBSDF{<:SurfaceLobeSpan}) = false
subsurface_slot(::Type{SurfaceLobeSpan{L}}) where {L} = subsurface_slot(L) == 0 ? 0 : 1
@inline subsurface_weight(s::SurfaceLobeSpan) = subsurface_weight(s.data)
@inline subsurface_choice(s::SurfaceLobeSpan, target::Float32) = subsurface_choice(s.data, target)
@inline picked_walk_params(s::SurfaceLobeSpan, selected::Int32, k::Val) = picked_walk_params(s.data, selected, k)
@inline with_lobe(::Type{SurfaceLobeSpan{L}}, k::Val{K}, w) where {L, K} = SurfaceLobeSpan(with_lobe(L, k, w), Int32(K))
@inline no_samples(::Type{S}, lambda::Wavelengths) where {S <: SurfaceLobeSpan} =
    LobeBSDF(blank(S), 0.0f0, lambda, SpectralRadiance(0.0f0), false)

@inline function surface_append!(storage, terms::Tuple, at, table, lambda, scale, mode, mask)
    term = first(terms)
    if term.sample_weight > 0.0f0
        @inbounds storage[at] = surface_store(lift(term, scale, table, lambda), mode, mask)
        at += Int32(1)
    end
    return surface_append!(storage, Base.tail(terms), at, table, lambda, scale, mode, mask)
end
Base.@constprop :none @noinline function surface_setup_slots(::Val{N}, s, prog, i, sc, lut, table, lambda, scale, mode::Val{Looped}, mask::Val{Mask}) where {N, Looped, Mask}
    storage = MVector{N, SurfaceStoredWeight{Looped, Mask}}(surface_empty(Val(N), mode, mask))
    at = Int32(1); emission = Vec3f(0)
    @inbounds while prog[i].op != NodeOp.RESULT
        i, prefix = surface_zero_prefix(s, prog, i, sc)
        emission += prefix
        prog[i].op == NodeOp.RESULT && break
        i, at, e = surface_store_dispatch!(storage, at, s, prog, i, sc, lut, table, lambda, scale, mode, mask)
        emission += e
    end
    return Tuple(storage), emission, at - Int32(1)
end
Base.@constprop :none @noinline function surface_setup_slots(::Val{0}, s, prog, i, sc, lut, table, lambda, scale, mode::Val, mask::Val)
    _, emission = surface_zero_prefix(s, prog, i, sc)
    return (), emission, Int32(0)
end

# Dispersion is BSDF metadata even when its transmission lobe has no weight.
# Preserve it before compaction. Only solid Principled transmission carries it
# in the accepted Cycles program registry.
@inline function surface_program_metadata(s, prog, sc, ::Val{Mask}) where {Mask}
    transparency_sum = Vec3f(0); dispersive = false; i = Int32(1)
    @inbounds while prog[i].op != NodeOp.RESULT
        ins = prog[i]
        if ins.op == NodeOp.PRINCIPLED
            c = instruction_count(PrincipledBSDF)
            transparency_sum += Vec3f(node_f(s, prog, i, 1, c, sc) * (1.0f0 - clamp(node_f(s, prog, i, 6, c, sc), 0.0f0, 1.0f0)))
            if !iszero(Mask & (UInt32(1) << 7))
                abbe = max(node_f(s, prog, i, 21, c, sc), 0.0f0)
                dispersion = clamp(node_f(s, prog, i, 20, c, sc), 0.0f0, 1.0f0)
                thin = node_f(s, prog, i, 34, c, sc) > 0.5f0
                dispersive |= !thin && abbe > 0.0f0 && dispersion > 0.0f0
            end
        elseif ins.mode[1] == ClosureKind.DIFFUSE
            transparency_sum += Vec3f(node_f(s, prog, i, 1, 1, sc) * (1.0f0 - clamp(node_f(s, prog, i, 5, 1, sc), 0.0f0, 1.0f0)))
        elseif ins.mode[1] == ClosureKind.TRANSPARENT
            transparency_sum += transparency(s, prog, i, sc)
        end
        i = next_node(prog, i, surface_chunks(ins))
    end
    return transparency_sum, dispersive
end
@inline function surface_program_bsdf(::Type{SurfaceProgramNodes{N, Looped, Mask, E}}, s, prog, sc, lut, table, lambda) where {N, Looped, Mask, E}
    transparency_sum, dispersive = surface_program_metadata(s, prog, sc, Val(Mask))
    alpha = 1.0f0 - _avg(transparency_sum)
    scale = alpha > 1.0f-6 ? 1.0f0 / alpha : 0.0f0
    slots, emission, count = surface_setup_slots(Val(N), s, prog, Int32(1), sc, lut, table, lambda, scale, Val(Looped), Val(Mask))
    lobes = SurfaceLobeSpan(slots, count)
    total = reduce(+, map(w -> w.sample_weight, slots); init = 0.0f0)
    L = E && _max(emission) > 0.0f0 ? uplift_rgb_illuminant(table, RGBSpectrum(emission...), lambda) / D65_PHOTOMETRIC : SpectralRadiance(0.0f0)
    return LobeBSDF(surface_ready(lobes), total, lambda, L, dispersive)
end
