# Experimental caller-local storage. Keep one mutable bounded array throughout
# setup/evaluation instead of copying it to a tuple and selecting tuple fields.
struct SurfaceMutableSpan{N, Looped, Mask} <: AbstractLobeStorage
    data::MVector{N, SurfaceStoredWeight{Looped, Mask}}
    count::Int32
end
@inline Base.getindex(s::SurfaceMutableSpan, i) = s.data[i]
@inline Base.length(s::SurfaceMutableSpan) = Int(s.count)
@inline Base.map(f, s::SurfaceMutableSpan) = SurfaceMutableSpan(map(f, s.data), s.count)
@inline function share(s::SurfaceMutableSpan, restriction::Scattering)
    total = 0.0f0
    @inbounds for i in 1:s.count
        total += share(s.data[i], restriction)
    end
    return total
end
@inline mixture(s::SurfaceMutableSpan, wo::Vec3f, wi::Vec3f, ns::Vec3f, lambda::Wavelengths, restriction::Scattering = AnyScattering()) =
    surface_mixture(s.data, s.count, wo, wi, ns, lambda, restriction)
@inline pick_sample(s::SurfaceMutableSpan, target::Float32, before::Float32, wo::Vec3f, u::Point2f, lambda::Wavelengths, restriction::Scattering = AnyScattering()) =
    surface_pick_sample(s.data, s.count, target, before, wo, u, lambda, restriction)
@inline terminates_at_vertex(::LobeBSDF{<:SurfaceMutableSpan}) = false
subsurface_slot(::Type{SurfaceMutableSpan{N, L, M}}) where {N, L, M} = surface_has_walk(Val(M)) ? 1 : 0
@inline subsurface_weight(s::SurfaceMutableSpan) = subsurface_weight(Tuple(s.data))
@inline subsurface_choice(s::SurfaceMutableSpan, target::Float32) = subsurface_choice(Tuple(s.data), target)
@inline picked_walk_params(s::SurfaceMutableSpan, selected::Int32, k::Val) = picked_walk_params(Tuple(s.data), selected, k)
@inline function blank(::Type{SurfaceMutableSpan{N, L, M}}) where {N, L, M}
    return SurfaceMutableSpan(MVector{N, SurfaceStoredWeight{L, M}}(surface_empty(Val(N), Val(L), Val(M))), Int32(0))
end
@inline function with_lobe(::Type{S}, k::Val{K}, w) where {S <: SurfaceMutableSpan, K}
    span = blank(S)
    @inbounds span.data[K] = w
    return SurfaceMutableSpan(span.data, Int32(K))
end
@inline no_samples(::Type{S}, lambda::Wavelengths) where {S <: SurfaceMutableSpan} =
    LobeBSDF(blank(S), 0.0f0, lambda, SpectralRadiance(0.0f0), false)

@inline function surface_setup_mutable(::Val{N}, s, prog, i, sc, lut, table, lambda, scale, mode::Val{Looped}, mask::Val{Mask}) where {N, Looped, Mask}
    storage = MVector{N, SurfaceStoredWeight{Looped, Mask}}(surface_empty(Val(N), mode, mask))
    at = Int32(1); emission = Vec3f(0)
    @inbounds while prog[i].op != NodeOp.RESULT
        i, prefix = surface_zero_prefix(s, prog, i, sc)
        emission += prefix
        prog[i].op == NodeOp.RESULT && break
        i, at, e = surface_store_dispatch!(storage, at, s, prog, i, sc, lut, table, lambda, scale, mode, mask)
        emission += e
    end
    return SurfaceMutableSpan(storage, at - Int32(1)), emission
end
@inline function surface_program_bsdf(::Type{SurfaceProgramNodes{N, Looped, Mask, E}}, s, prog, sc, lut, table, lambda) where {N, Looped, Mask, E}
    transparency_sum, dispersive = surface_program_metadata(s, prog, sc, Val(Mask))
    alpha = 1.0f0 - _avg(transparency_sum)
    scale = alpha > 1.0f-6 ? 1.0f0 / alpha : 0.0f0
    lobes, emission = surface_setup_mutable(Val(N), s, prog, Int32(1), sc, lut, table, lambda, scale, Val(Looped), Val(Mask))
    total = share(lobes, AnyScattering())
    light = E && _max(emission) > 0.0f0 ? uplift_rgb_illuminant(table, RGBSpectrum(emission...), lambda) / D65_PHOTOMETRIC : SpectralRadiance(0.0f0)
    return LobeBSDF(lobes, total, lambda, light, dispersive)
end
