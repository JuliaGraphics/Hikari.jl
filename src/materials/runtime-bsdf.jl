# Ported from Blender Cycles v5.2.1: intern/cycles/kernel/closure/bsdf.h, bsdf_sheen.h,
# bsdf_oren_nayar.h, copyright 2011-2022 Blender Foundation, SPDX-License-Identifier: Apache-2.0;
# and bsdf_microfacet.h, adapted by Cycles from Open Shading Language, copyright 2009-2010 Sony
# Pictures Imageworks Inc., et al. All Rights Reserved, copyright 2011-2022 Blender Foundation,
# SPDX-License-Identifier: BSD-3-Clause. Modified: rewritten in Julia, spectral. See
# LICENSES/Apache-2.0.txt and LICENSES/BSD-3-Clause.txt.

# Local storage shared by permutations and combinations of supported BSDF nodes.
# All expensive node setup and spectral conversion happen once per hit.
# Six floats hold Oren-Nayar coefficients and its spectral multiple scattering.
# Glossy needs only its alphas and energy after setup; NoFresnel.color is unused
# by evaluation/sampling, and Glossy has no IOR contrast or dispersion.
struct RuntimeLobe{Looped} <: BSDFLobe
    kind::UInt32
    data::NTuple{6, Float32}
end
# Looped selects code generation, not scattering semantics. It is fixed when the
# host records a kernel for a backend; no device-side backend query is needed.
struct RuntimeBSDFNodes{N, Looped} end

# Extend this set only after packing, sampling and complete-render comparisons pass.
const RuntimeBSDFNode = Union{DiffuseBSDF, SheenBSDF, GlossyBSDF}

"""The BSDF kernel class: preserve elementary BSDFs; share bounded local storage for mixtures."""
function bsdf_kernel_nodes(::Type{M}, api) where {M <: Tuple}
    n = length(M.parameters)
    (2 <= n <= 8 && all(T -> T <: RuntimeBSDFNode, M.parameters)) || return M
    looped = api isa Mantle.MetalAPI
    capacity = looped && n == 3 ? 3 : n <= 4 ? 4 : 8
    return RuntimeBSDFNodes{capacity, looped}
end
emits(::Type{<:RuntimeBSDFNodes}) = false
const RuntimeWeighted{Looped} = Weighted{RuntimeLobe{Looped}, SpectralRadiance}

@inline runtime_pack(l::DiffuseLobe{SpectralRadiance}, ::Val{Looped}) where {Looped} =
    RuntimeLobe{Looped}(1, (l.a, l.b, l.ms.data...))
@inline runtime_pack(l::SheenLobe, ::Val{Looped}) where {Looped} = RuntimeLobe{Looped}(2, (l.a, l.b, 0f0, 0f0, 0f0, 0f0))
@inline runtime_pack(l::MicrofacetLobe{:reflection, NoFresnel{SpectralRadiance}, CyclesGGX}, ::Val{Looped}) where {Looped} =
    RuntimeLobe{Looped}(3, (l.alpha_x, l.alpha_y, l.energy, 0f0, 0f0, 0f0))
@inline runtime_pack(w::Weighted, mode::Val) = Weighted(runtime_pack(w.lobe, mode), w.weight, w.sample_weight, w.n, w.t)
@inline runtime_unpack(l::RuntimeLobe, ::Val{1}) =
    DiffuseLobe{SpectralRadiance}(l.data[1], l.data[2], SpectralRadiance(l.data[3:6]))
@inline runtime_unpack(l::RuntimeLobe, ::Val{2}) = SheenLobe(l.data[1], l.data[2])
@inline runtime_unpack(l::RuntimeLobe, ::Val{3}) =
    MicrofacetLobe{:reflection, NoFresnel{SpectralRadiance}, CyclesGGX}(
        l.data[1], l.data[2], 1f0, l.data[3], 0f0,
        NoFresnel(SpectralRadiance(0f0)))
@inline runtime_unpack(w::RuntimeWeighted, v) = Weighted(runtime_unpack(w.lobe, v), w.weight, w.sample_weight, w.n, w.t)
@inline runtime_empty_lobe(::Val{Looped}) where {Looped} = Weighted(RuntimeLobe{Looped}(0, ntuple(_ -> 0f0, 6)), SpectralRadiance(0f0),
                                    0f0, Vec3f(0, 0, 1), Vec3f(0))

# Dispatch before calling weighted_f so Diffuse retains its cosine-only fast path.
@inline function weighted_f(w::RuntimeWeighted, wo::Vec3f, wi::Vec3f, ns::Vec3f,
                                  lambda::Wavelengths, s::Scattering = AnyScattering())
    w.lobe.kind == 1 && return weighted_f(runtime_unpack(w, Val(1)), wo, wi, ns, lambda, s)
    n, t, b = runtime_frame(w, wo)
    shadowing = bump_shadowing(w.lobe.kind == 2, n, ns, wi)
    cos_s = abs(dot(wi, ns))
    (shadowing > 0f0 && cos_s > 1f-7) || return (SpectralRadiance(0f0), 0f0)
    local_wo = world_to_local(wo,n,t,b)
    local_wi = world_to_local(wi,n,t,b)
    f, pdf = runtime_f(w.lobe,local_wo,local_wi,lambda,s)
    return w.weight * f * (shadowing * abs(dot(wi,n)) / cos_s), pdf
end
@inline runtime_frame(w,wo) = w.lobe.kind == 2 ?
    lobe_frame(runtime_unpack(w.lobe,Val(2)),w.n,w.t,wo) :
    lobe_frame(w.lobe,w.n,w.t,wo)
@inline function runtime_f(l,wo,wi,lambda,s)
    l.kind == 1 && return lobe_f(runtime_unpack(l,Val(1)),wo,wi,lambda,s)
    l.kind == 2 && return lobe_f(runtime_unpack(l,Val(2)),wo,wi,lambda,s)
    return lobe_f(runtime_unpack(l,Val(3)),wo,wi,lambda,s)
end
@inline function runtime_sample(l,wo,u,uc,lambda,s)
    l.kind == 1 && return lobe_sample(runtime_unpack(l,Val(1)),wo,u,uc,lambda,s)
    l.kind == 2 && return lobe_sample(runtime_unpack(l,Val(2)),wo,u,uc,lambda,s)
    return lobe_sample(runtime_unpack(l,Val(3)),wo,u,uc,lambda,s)
end
@inline function runtime_pick(w::RuntimeWeighted, target, before, wo, u, lambda, s)
    sw = share(w,s)
    uc = clamp((target-before)/max(sw,1f-12),0f0,0.99999994f0)
    n,t,b = runtime_frame(w,wo)
    wi,f,pdf,flags,eta = runtime_sample(w.lobe,world_to_local(wo,n,t,b),u,uc,lambda,s)
    return (normalize(local_to_world(wi,n,t,b)), w.weight*f,pdf,flags,eta,sw,n,
            terminates_secondary(w.lobe,flags))
end
@inline function mixture(lobes::Tuple{RuntimeWeighted, Vararg{RuntimeWeighted}}, wo::Vec3f, wi::Vec3f, ns::Vec3f,
                                lambda::Wavelengths, s::Scattering = AnyScattering())
    f, pdf = mixture(Base.tail(lobes), wo, wi, ns, lambda, s)
    w = first(lobes)
    w.sample_weight > 0f0 || return f, pdf
    fi, pi = weighted_f(w, wo, wi, ns, lambda, s)
    return f + fi, pdf + pi * w.sample_weight
end
@inline function pick_sample(lobes::Tuple{RuntimeWeighted, Vararg{RuntimeWeighted}}, target::Float32, before::Float32,
                                    wo::Vec3f, u::Point2f, lambda::Wavelengths,
                                    s::Scattering = AnyScattering())
    w = first(lobes)
    sw = share(w, s)
    after = before + sw
    if sw > 0f0 && (target < after || length(lobes) == 1)
        return runtime_pick(w, target, before, wo, u, lambda, s)
    end
    return pick_sample(Base.tail(lobes), target, after, wo, u, lambda, s)
end


@inline function mixture(lobes::Tuple{RuntimeWeighted{true}, Vararg{RuntimeWeighted{true}}}, wo::Vec3f, wi::Vec3f, ns::Vec3f,
                                lambda::Wavelengths, s::Scattering = AnyScattering())
    f = SpectralRadiance(0f0)
    pdf = 0f0
    # Same summation order as the reference's tail-first recursion.
    @inbounds for j in length(lobes):-1:1
        w = lobes[j]
        if w.sample_weight > 0f0
            fi, pi = weighted_f(w, wo, wi, ns, lambda, s)
            f += fi
            pdf += pi * w.sample_weight
        end
    end
    return f, pdf
end
@inline function pick_sample(lobes::Tuple{RuntimeWeighted{true}, Vararg{RuntimeWeighted{true}}}, target::Float32, before::Float32,
                                    wo::Vec3f, u::Point2f, lambda::Wavelengths,
                                    s::Scattering = AnyScattering())
    @inbounds for j in 1:length(lobes)
        w = lobes[j]
        sw = share(w, s)
        after = before + sw
        if sw > 0f0 && (target < after || j == length(lobes))
            return runtime_pick(w, target, before, wo, u, lambda, s)
        end
        before = after
    end
    return pick_sample((), target, before, wo, u, lambda, s)
end


@inline function runtime_setup_one(s, prog, i, sc, lut, table, lambda, scale, mode::Val)
    kind = @inbounds prog[i].mode[1]
    if kind == ClosureKind.DIFFUSE
        lobes, _, _ = node_lobes(DiffuseBSDF, s, prog, i, sc, lut)
        return runtime_pack(lift(first(lobes), scale, table, lambda), mode)
    elseif kind == ClosureKind.SHEEN
        lobes, _, _ = node_lobes(SheenBSDF, s, prog, i, sc, lut)
        return runtime_pack(lift(first(lobes), scale, table, lambda), mode)
    else
        lobes, _, _ = node_lobes(GlossyBSDF, s, prog, i, sc, lut)
        return runtime_pack(lift(first(lobes), scale, table, lambda), mode)
    end
end

@inline runtime_setup_slots(::Val{0}, s, prog, i, sc, lut, table, lambda, scale, mode::Val) = ()
@inline function runtime_setup_slots(::Val{N}, s, prog, i, sc, lut, table, lambda, scale, mode::Val) where {N}
    @inbounds prog[i].op == NodeOp.RESULT && return ntuple(_ -> runtime_empty_lobe(mode), Val(N))
    w = runtime_setup_one(s, prog, i, sc, lut, table, lambda, scale, mode)
    return (w, runtime_setup_slots(Val(N-1), s, prog, next_node(prog,i,1), sc, lut, table, lambda, scale, mode)...)
end

@inline function runtime_bsdf(::Type{RuntimeBSDFNodes{N, Looped}}, s, prog, sc, lut, table, lambda) where {N, Looped}
    # Alpha is needed before lifting each weight. This walk reads scalars only; it does not
    # repeat node setup, table lookups or spectral conversion.
    transparency = 0f0
    i = Int32(1)
    @inbounds while prog[i].op != NodeOp.RESULT
        if prog[i].mode[1] == ClosureKind.DIFFUSE
            transparency += node_f(s, prog, i, 1, 1, sc) *
                (1f0 - clamp(node_f(s, prog, i, 5, 1, sc), 0f0, 1f0))
        end
        i = next_node(prog, i, 1)
    end
    alpha = 1f0 - transparency
    scale = alpha > 1f-6 ? 1f0 / alpha : 0f0
    lobes = runtime_setup_slots(Val(N), s, prog, Int32(1), sc, lut, table, lambda, scale, Val(Looped))
    total = reduce(+, map(w -> w.sample_weight, lobes); init = 0f0)
    return LobeBSDF(lobes, total, lambda, SpectralRadiance(0f0), false)
end

