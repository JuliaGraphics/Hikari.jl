# Ported from Blender Cycles v5.2.1: intern/cycles/kernel/closure/bsdf.h, bsdf_sheen.h,
# bsdf_oren_nayar.h, copyright 2011-2022 Blender Foundation, SPDX-License-Identifier: Apache-2.0;
# and bsdf_microfacet.h, adapted by Cycles from Open Shading Language, copyright 2009-2010 Sony
# Pictures Imageworks Inc., et al. All Rights Reserved, copyright 2011-2022 Blender Foundation,
# SPDX-License-Identifier: BSD-3-Clause. Modified: rewritten in Julia, spectral. See
# LICENSES/Apache-2.0.txt and LICENSES/BSD-3-Clause.txt.

# Bounded local BSDF storage shared by material graphs and node orderings.
struct SurfaceStoredLobe{Looped, Walk} <: BSDFLobe
    kind::UInt32
    data::NTuple{15, Float32}
end
const SurfaceStoredWeight{Looped, Walk} = Weighted{SurfaceStoredLobe{Looped, Walk}, SpectralRadiance}
@inline surface_data(xs::Tuple) = (xs..., ntuple(_ -> 0.0f0, Val(15 - length(xs)))...)
@inline surface_spectrum(d, ::Val{K}) where {K} = SpectralRadiance(ntuple(j -> d[K + j - 1], Val(4)))
@inline surface_store(l, ::Val{Looped}, ::Val{Walk}) where {Looped, Walk} =
    SurfaceStoredLobe{Looped, Walk}(surface_kind(typeof(l)), surface_data(surface_fields(l)))
@inline surface_store(w::Weighted, mode::Val, walk::Val) =
    Weighted(surface_store(w.lobe, mode, walk), w.weight, w.sample_weight, w.n, w.t)

# Cycles terms inside a pbrt composition use the same packed scattering code.
# The surrounding component estimator still belongs to the pbrt model.
@inline function stored_shader_term(w::Weighted, scale, table, lambda)
    v = lift(w, scale, table, lambda)
    mask = Val(UInt32(1) << surface_kind(typeof(v.lobe)))
    return surface_store(v, Val(true), mask)
end

surface_kind(::Type{<:DiffuseLobe}) = UInt32(1)
surface_kind(::Type{SheenLobe}) = UInt32(2)
@inline surface_fields(l::DiffuseLobe) = (l.a, l.b, l.ms.data...)
@inline surface_fields(l::SheenLobe) = (l.a, l.b)
@inline surface_restore(l::SurfaceStoredLobe, ::Val{1}) = DiffuseLobe(l.data[1], l.data[2], surface_spectrum(l.data, Val(3)))
@inline surface_restore(l::SurfaceStoredLobe, ::Val{2}) = SheenLobe(l.data[1], l.data[2])

@inline surface_fresnel_fields(::NoFresnel) = ()
@inline surface_fresnel_fields(f::F82Tint) = (f.f0.data..., f.tint.data..., f.film_thickness, f.film_ior)
@inline surface_fresnel_fields(f::DielectricFresnel) = (f.reflection_tint.data..., f.transmission_tint.data...)
@inline surface_fresnel_fields(f::PhysicalConductorFresnel) = (f.eta.data..., f.k.data..., f.film_thickness, f.film_ior)
@inline surface_fresnel_fields(f, mode::Val) = surface_fresnel_fields(f)
# The registry accepts Schlick only from Principled: specular has unit tints,
# and glass has unit reflection tint. Preserve the coloured glass transmission.
@inline surface_fresnel_fields(f::SchlickFresnel, ::Val{:reflection}) = (f.f0.data..., f.film_thickness, f.film_ior)
@inline surface_fresnel_fields(f::SchlickFresnel, ::Val{:glass}) = (f.f0.data..., f.transmission_tint.data..., f.film_thickness, f.film_ior)
@inline surface_fields(l::MicrofacetLobe{Mode}) where {Mode} = (l.alpha_x, l.alpha_y, l.ior, l.energy, l.inv_abbe, surface_fresnel_fields(l.fresnel, Val(Mode))...)

# A closed registry independent of graph composition. The convention/fresnel/mode
# determines behavior; the exact node order never enters the stored record type.
const SURFACE_STORAGE_MODELS = (
    (3, :reflection, NoFresnel{SpectralRadiance}, CyclesGGX),
    (4, :reflection, F82Tint{SpectralRadiance}, CyclesGGX),
    (5, :reflection, DielectricFresnel{SpectralRadiance}, CyclesGGX),
    (6, :reflection, SchlickFresnel{SpectralRadiance}, CyclesGGX),
    (7, :glass, SchlickFresnel{SpectralRadiance}, CyclesGGX),
    (8, :reflection, PhysicalConductorFresnel{SpectralRadiance}, CyclesGGX),
    (9, :refraction, NoFresnel{SpectralRadiance}, CyclesGGX),
    (10, :thin, NoFresnel{SpectralRadiance}, CyclesGGX),
    (15, :reflection, NoFresnel{SpectralRadiance}, CyclesBeckmann),
    (16, :reflection, F82Tint{SpectralRadiance}, CyclesBeckmann),
    (17, :reflection, SchlickFresnel{SpectralRadiance}, CyclesBeckmann),
    (18, :glass, SchlickFresnel{SpectralRadiance}, CyclesBeckmann),
    (19, :reflection, PhysicalConductorFresnel{SpectralRadiance}, CyclesBeckmann),
    (20, :refraction, NoFresnel{SpectralRadiance}, CyclesBeckmann),
)
@inline surface_fresnel(::Type{NoFresnel{SpectralRadiance}}, d) = NoFresnel(SpectralRadiance(0.0f0))
@inline surface_fresnel(::Type{F82Tint{SpectralRadiance}}, d) = F82Tint(surface_spectrum(d, Val(6)), surface_spectrum(d, Val(10)), d[14], d[15])
@inline surface_fresnel(::Type{DielectricFresnel{SpectralRadiance}}, d) = DielectricFresnel(surface_spectrum(d, Val(6)), surface_spectrum(d, Val(10)))
@inline surface_fresnel(::Type{PhysicalConductorFresnel{SpectralRadiance}}, d) = PhysicalConductorFresnel(surface_spectrum(d, Val(6)), surface_spectrum(d, Val(10)), d[14], d[15])
@inline surface_fresnel(T, d, mode::Val) = surface_fresnel(T, d)
@inline surface_fresnel(::Type{SchlickFresnel{SpectralRadiance}}, d, ::Val{:reflection}) =
    SchlickFresnel(surface_spectrum(d, Val(6)), SpectralRadiance(1.0f0), SpectralRadiance(1.0f0), d[10], d[11])
@inline surface_fresnel(::Type{SchlickFresnel{SpectralRadiance}}, d, ::Val{:glass}) =
    SchlickFresnel(surface_spectrum(d, Val(6)), SpectralRadiance(1.0f0), surface_spectrum(d, Val(10)), d[14], d[15])
for (kind, mode, fresnel, convention) in SURFACE_STORAGE_MODELS
    T = MicrofacetLobe{mode, fresnel, convention}
    @eval begin
        surface_kind(::Type{$T}) = UInt32($kind)
        @inline surface_restore(l::SurfaceStoredLobe, ::Val{$kind}) =
            $T(l.data[1], l.data[2], l.data[3], l.data[4], l.data[5], surface_fresnel($fresnel, l.data, Val($(QuoteNode(mode)))))
    end
end
surface_kind(::Type{TranslucentLobe}) = UInt32(11)
@inline surface_fields(::TranslucentLobe) = ()
@inline surface_restore(l::SurfaceStoredLobe, ::Val{11}) = TranslucentLobe()
surface_kind(::Type{ToonLobe{false}}) = UInt32(12)
surface_kind(::Type{ToonLobe{true}}) = UInt32(13)
@inline surface_fields(l::ToonLobe) = (l.size, l.smooth, l.sample_angle, l.one_minus_cos_angle)
@inline surface_restore(l::SurfaceStoredLobe, ::Val{12}) = ToonLobe{false}(l.data[1:4]...)
@inline surface_restore(l::SurfaceStoredLobe, ::Val{13}) = ToonLobe{true}(l.data[1:4]...)
surface_kind(::Type{SubsurfaceLobe{SpectralRadiance}}) = UInt32(14)
@inline surface_fields(l::SubsurfaceLobe) = (l.albedo.data..., l.radius.data..., l.alpha, l.ior, l.anisotropy, Float32(l.exited))
@inline surface_restore(l::SurfaceStoredLobe, ::Val{14}) = SubsurfaceLobe(surface_spectrum(l.data, Val(1)), surface_spectrum(l.data, Val(5)), l.data[9], l.data[10], l.data[11], l.data[12] > 0.0f0)

# GGX and Beckmann share the existing microfacet formulas and runtime geometry.
# Cloning those formulas per distribution/Fresnel pair made NVIDIA pipeline
# compilation exhaust memory for a shared Principled/Beckmann class. Only the
# distribution-specific D, lambda and visible-normal sampler branch here.
struct SurfaceCyclesDistribution <: MicrofacetConvention end
struct SurfaceMicrofacetFresnel{Mask, Mode} <: FresnelModel
    kind::UInt32
    data::NTuple{15, Float32}
end
@inline function fresnel_rt(f::SurfaceMicrofacetFresnel{Mask, Mode}, mu::Float32, ior::Float32, lambda::Wavelengths) where {Mask, Mode}
    d = f.data
    if Mode === :glass
        return fresnel_rt(surface_fresnel(SchlickFresnel{SpectralRadiance}, d, Val(:glass)), mu, ior, lambda)
    elseif Mode === :refraction
        return fresnel_rt(NoFresnel(SpectralRadiance(0.0f0)), mu, ior, lambda)
    end
    kind, d = f.kind, f.data
    capabilities = Mask isa Bool ? typemax(UInt32) : Mask
    if !iszero(capabilities & ((UInt32(1) << 4) | (UInt32(1) << 16)))
        kind in (4, 16) && return fresnel_rt(surface_fresnel(F82Tint{SpectralRadiance}, d), mu, ior, lambda)
    end
    if !iszero(capabilities & (UInt32(1) << 5))
        kind == 5 && return fresnel_rt(surface_fresnel(DielectricFresnel{SpectralRadiance}, d), mu, ior, lambda)
    end
    if !iszero(capabilities & ((UInt32(1) << 6) | (UInt32(1) << 17)))
        kind in (6, 17) && return fresnel_rt(surface_fresnel(SchlickFresnel{SpectralRadiance}, d, Val(:reflection)), mu, ior, lambda)
    end
    if !iszero(capabilities & ((UInt32(1) << 8) | (UInt32(1) << 19)))
        kind in (8, 19) && return fresnel_rt(surface_fresnel(PhysicalConductorFresnel{SpectralRadiance}, d), mu, ior, lambda)
    end
    return fresnel_rt(NoFresnel(SpectralRadiance(0.0f0)), mu, ior, lambda)
end
@inline function surface_microfacet(l::SurfaceStoredLobe{Looped, Mask}, ::Val{Mode}) where {Looped, Mask, Mode}
    d = l.data
    return MicrofacetLobe{Mode, SurfaceMicrofacetFresnel{Mask, Mode}, SurfaceCyclesDistribution}(
        d[1], d[2], d[3], d[4], d[5], SurfaceMicrofacetFresnel{Mask, Mode}(l.kind, d)
    )
end
@inline surface_is_beckmann(l::MicrofacetLobe{Mode, SurfaceMicrofacetFresnel{Mask, Mode}, SurfaceCyclesDistribution}) where {Mode, Mask} =
    !iszero((Mask isa Bool ? typemax(UInt32) : Mask) & UInt32(0x001f8000)) && l.fresnel.kind >= 15
@inline singular(l::MicrofacetLobe{<:Any, <:FresnelModel, SurfaceCyclesDistribution}) = l.alpha_x * l.alpha_y <= 2.0f-10
@inline valid_refraction_pair(::MicrofacetLobe{<:Any, <:FresnelModel, SurfaceCyclesDistribution}, ::Float32, ::Float32) = true
@inline microfacet_d(l::MicrofacetLobe{<:Any, <:FresnelModel, SurfaceCyclesDistribution}, h::Vec3f) =
    surface_is_beckmann(l) ? beckmann_d(h, l.alpha_x, l.alpha_y) : trowbridge_reitz_d(h, l.alpha_x, l.alpha_y)
@inline microfacet_lambda(l::MicrofacetLobe{<:Any, <:FresnelModel, SurfaceCyclesDistribution}, w::Vec3f) =
    surface_is_beckmann(l) ? beckmann_lambda(w, l.alpha_x, l.alpha_y) : trowbridge_reitz_lambda(w, l.alpha_x, l.alpha_y)
@inline microfacet_sample_wm(l::MicrofacetLobe{<:Any, <:FresnelModel, SurfaceCyclesDistribution}, wo::Vec3f, u::Point2f) =
    surface_is_beckmann(l) ? beckmann_sample_wm(wo, u, l.alpha_x, l.alpha_y) : trowbridge_reitz_sample_wm(wo, u, l.alpha_x, l.alpha_y)

@inline lobe_f(l::SurfaceStoredLobe, wo::Vec3f, wi::Vec3f, lambda::Wavelengths, s::Scattering) = surface_apply(lobe_f, l, wo, wi, lambda, s)
@inline lobe_sample(l::SurfaceStoredLobe, wo::Vec3f, u::Point2f, uc::Float32, lambda::Wavelengths, s::Scattering) = surface_apply(lobe_sample, l, wo, u, uc, lambda, s)
@inline is_diffuse_lobe(l::SurfaceStoredLobe) = l.kind in (1, 2, 11, 12, 14)
@inline can_scatter(l::SurfaceStoredLobe, s::Scattering) =
    l.kind in (7, 18) ? true : l.kind in (9, 10, 11, 20) ? allows_transmission(s) : allows_reflection(s)
@inline disperses(l::SurfaceStoredLobe) = l.kind in (7, 18) && l.data[5] > 0.0f0
@inline lobe_frame(l::SurfaceStoredLobe, n::Vec3f, t::Vec3f, wo::Vec3f) =
    l.kind == 2 ? lobe_frame(surface_restore(l, Val(2)), n, t, wo) :
    (t == Vec3f(0) ? (n, coordinate_system(n)...) : (n, t, cross(n, t)))
# Retain the cosine-only fast path and its numerical ordering for diffuse nodes.
@inline function weighted_f(w::SurfaceStoredWeight, wo::Vec3f, wi::Vec3f, ns::Vec3f, lambda::Wavelengths, s::Scattering = AnyScattering())
    if w.lobe.kind == 1
        rw = Weighted(surface_restore(w.lobe, Val(1)), w.weight, w.sample_weight, w.n, w.t)
        return weighted_f(rw, wo, wi, ns, lambda, s)
    elseif w.lobe.kind == 11
        rw = Weighted(TranslucentLobe(), w.weight, w.sample_weight, w.n, w.t)
        return weighted_f(rw, wo, wi, ns, lambda, s)
    end
    n, t, b = lobe_frame(w.lobe, w.n, w.t, wo)
    shadowing = bump_shadowing(is_diffuse_lobe(w.lobe), n, ns, wi)
    cos_s = abs(dot(wi, ns))
    (shadowing > 0.0f0 && cos_s > 1.0f-7) || return (SpectralRadiance(0.0f0), 0.0f0)
    f, pdf = lobe_f(w.lobe, world_to_local(wo, n, t, b), world_to_local(wi, n, t, b), lambda, s)
    return (w.weight * f * (shadowing * abs(dot(wi, n)) / cos_s), pdf)
end

# The same traversal supports immutable tuples and counted local collections.
@inline function mixture(
        lobes::Tuple{SurfaceStoredWeight{true}, Vararg{SurfaceStoredWeight{true}}},
        wo::Vec3f, wi::Vec3f, ns::Vec3f, lambda::Wavelengths, s::Scattering = AnyScattering()
    )
    return surface_mixture(lobes, length(lobes), wo, wi, ns, lambda, s)
end
@inline function surface_mixture(lobes, count, wo, wi, ns, lambda, s)
    f, pdf = SpectralRadiance(0.0f0), 0.0f0
    @inbounds for j in count:-1:1
        w = lobes[j]
        if w.sample_weight > 0.0f0
            fi, pi = weighted_f(w, wo, wi, ns, lambda, s)
            f += fi;pdf += pi * w.sample_weight
        end
    end
    return f, pdf
end
@inline function pick_sample(
        lobes::Tuple{SurfaceStoredWeight{true}, Vararg{SurfaceStoredWeight{true}}},
        target::Float32, before::Float32, wo::Vec3f, u::Point2f, lambda::Wavelengths,
        s::Scattering = AnyScattering()
    )
    return surface_pick_sample(lobes, length(lobes), target, before, wo, u, lambda, s)
end
@inline function surface_pick_sample(lobes, count, target, before, wo, u, lambda, s)
    @inbounds for j in 1:count
        w = lobes[j];sw = share(w, s);after = before + sw
        if sw > 0.0f0 && (target < after || j == count)
            uc = clamp((target - before) / max(sw, 1.0f-12), 0.0f0, 0.99999994f0)
            n, t, b = lobe_frame(w.lobe, w.n, w.t, wo)
            wi, f, pdf, flags, eta = lobe_sample(w.lobe, world_to_local(wo, n, t, b), u, uc, lambda, s)
            return (
                normalize(local_to_world(wi, n, t, b)), w.weight * f, pdf, flags, eta, sw, n,
                terminates_secondary(w.lobe, flags),
            )
        end
        before = after
    end
    return pick_sample((), target, before, wo, u, lambda, s)
end

subsurface_slot(::Type{L}) where {Looped, L <: Tuple{SurfaceStoredWeight{Looped, true}, Vararg{SurfaceStoredWeight{Looped, true}}}} = fieldcount(L)
@inline function subsurface_weight(lobes::Tuple{SurfaceStoredWeight, Vararg{SurfaceStoredWeight}})
    weight = 0.0f0
    for w in lobes
        w.lobe.kind == 14 && (weight += w.sample_weight)
    end
    return weight
end
@inline function subsurface_choice(lobes::Tuple{SurfaceStoredWeight, Vararg{SurfaceStoredWeight}}, target::Float32)
    @inbounds for j in 1:length(lobes)
        w = lobes[j]
        if w.lobe.kind == 14
            target < w.sample_weight && return Int32(j)
            target -= w.sample_weight
        end
    end
    return Int32(0)
end
@inline picked_walk_params(lobes::Tuple{SurfaceStoredWeight, Vararg{SurfaceStoredWeight}}, selected::Int32, ::Val) =
    (WalkParams(lobes[selected]), lobes[selected].sample_weight)
@inline without_subsurface_samples(w::SurfaceStoredWeight) = w.lobe.kind == 14 ? without_samples(w) : w
@inline WalkParams(w::SurfaceStoredWeight) = WalkParams(Weighted(surface_restore(w.lobe, Val(14)), w.weight, w.sample_weight, w.n, w.t))
@inline function exit_lobe(::Type{SurfaceStoredWeight{Looped, Walk}}, p::WalkParams, n::Vec3f) where {Looped, Walk}
    original = exit_lobe(Weighted{SubsurfaceLobe{SpectralRadiance}, SpectralRadiance}, p, n)
    return surface_store(original, Val(Looped), Val(Walk))
end

# Store only live terms; the capability mask and capacity bound kernel types.
struct SurfaceLobeSpan{L <: Tuple} <: AbstractLobeStorage
    data::L
    count::Int32
end
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

# Native compositions can contain packed Cycles lobes alongside native weighted
# lobes. Preserve subsurface intervals without changing their joint estimator.
stored_ss_type(::Type{T}) where {T} = T <: Weighted{<:SubsurfaceLobe}
stored_ss_type(::Type{<:SurfaceStoredWeight{L, M}}) where {L, M} =
    M isa Bool ? M : !iszero(M & (UInt32(1) << 14))
@generated function subsurface_slot(::Type{L}) where {L <: Tuple{Weighted, Vararg}}
    slots = findall(stored_ss_type, fieldtypes(L))
    return isempty(slots) ? 0 : last(slots)
end
@generated function subsurface_weight(lobes::L) where {L <: Tuple{Weighted, Vararg}}
    weights = map(findall(stored_ss_type, fieldtypes(L))) do k
        if fieldtype(L, k) <: SurfaceStoredWeight
            :(lobes[$k].lobe.kind == 14 ? lobes[$k].sample_weight : 0.0f0)
        else
            :(lobes[$k].sample_weight)
        end
    end
    return foldl((a, b) -> :($a + $b), weights; init = :(0.0f0))
end
@generated function subsurface_choice(lobes::L, target::Float32) where {L <: Tuple{Weighted, Vararg}}
    choices = map(findall(stored_ss_type, fieldtypes(L))) do k
        weight = fieldtype(L, k) <: SurfaceStoredWeight ?
            :(lobes[$k].lobe.kind == 14 ? lobes[$k].sample_weight : 0.0f0) :
            :(lobes[$k].sample_weight)
        quote
            amount = $weight
            target < amount && return Int32($k)
            target -= amount
        end
    end
    return quote
        $(Expr(:meta, :inline))
        $(choices...)
        return Int32(0)
    end
end
@generated function picked_walk_params(lobes::L, selected::Int32, ::Val{K}) where {L <: Tuple{Weighted, Vararg}, K}
    result = :((WalkParams(lobes[$K]), lobes[$K].sample_weight))
    for k in findall(stored_ss_type, fieldtypes(L))
        k == K && continue
        result = :(selected == $k ? (WalkParams(lobes[$k]), lobes[$k].sample_weight) : $result)
    end
    return result
end
