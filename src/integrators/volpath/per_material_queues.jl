# Per-material-type queue infrastructure (pbrt-v4 wavefront pattern).
#
# Today's `vp_trace_and_shade_kernel!` shades each surface hit inline by
# going through `with_index(materials, idx, ...)` which expands to a switch
# over every concrete material type in the scene's `StaticMultiTypeSet`.
# That makes the kernel large (every material's BSDF code is inlined into
# the switch arms) and divergent (threads with different materials in the
# same warp follow different arms).
#
# pbrt-v4 sidesteps both by splitting the shading queue by concrete material
# type — `MaterialEvalQueue<ConcreteMaterial>` in wavefront/workitems.h —
# and running one shading kernel per type via `ForEachType` (per-type kernel
# launches in surfscatter.cpp:41). Each kernel is templated on a single
# concrete material, so the BSDF code is fully monomorphised; warps process
# threads that all run the same code path.
#
# The pieces below are the Julia analogue:
#
#   * `TypedHitRef{T}` — a 4-byte index into the shared `hit_surface_queue`
#     that attaches the concrete material type `T` at the type level. Its
#     only purpose is to let Julia's normal multiple dispatch pick a kernel
#     specialisation per `T`; the hit payload itself is stored ONCE in
#     `hit_surface_queue`. (The previous design stored a full 304-byte
#     `VPHitSurfaceWorkItem` copy per typed queue; with every queue sized
#     for the full pixel count — hits PARTITION across the queues, so the
#     worst case for any single queue is all of them — that was
#     `n_types × n_pixels × 304 B` = 4.76 GiB on Crown's 12-material
#     1.4 Mpx render. Indices bring that to 67 MiB.)
#
#   * `MultiTypeMaterialQueue` — a `MultiTypeWorkQueue` of `TypedHitRef{T}`
#     for each `T` in the scene's `StaticMultiTypeSet`. Built once at
#     integrator state init from the materials set's static type tuple.
#
#   * `push_typed_hit!(mtmq, materials, hit_surface_queue, hit)` — pushes
#     the hit into `hit_surface_queue` once, then uses `with_index` on the
#     materials set ONE time (per hit, at push site) to route the index into
#     the matching typed queue. The kernel that drains a typed queue never
#     touches `with_index`; it only sees one concrete `TypedHitRef{T}` type.
#
#   * `material_of_type(materials, T, vec_idx)` — a `@generated` lookup that
#     returns the concrete material instance for a known type `T` and the
#     `vec_idx` carried on the hit. Zero overhead — compiles down to one
#     array load on the data tuple at the slot where the type lives.

using Adapt
import KernelAbstractions as KA
import Raycore

# ─────────────────────────────────────────────────────────────────────────────
# TypedHitRef{T}: 4-byte index that lifts the material type into the type
# parameter of the work item.
# ─────────────────────────────────────────────────────────────────────────────

"""
    TypedHitRef{T}

4-byte index into the shared `hit_surface_queue`, tagged with the concrete
material type `T` at the type level. Per-material shading kernels dispatch
on `TypedHitRef{T}` — each `T` produces a separately compiled kernel via
Julia's normal multiple dispatch — and load the actual
`VPHitSurfaceWorkItem` from `hit_surface_queue.items[ref.idx]`.
"""
struct TypedHitRef{T}
    idx::Int32
end

# ─────────────────────────────────────────────────────────────────────────────
# material_of_type — compile-time-resolved per-type material lookup.
# ─────────────────────────────────────────────────────────────────────────────

"""
    material_of_type(materials::StaticMultiTypeSet, ::Type{T}, vec_idx::UInt32)

Return the concrete material instance of type `T` at position `vec_idx`
within the type-`T` slot of the materials set. The slot index is looked up
at compile time from the materials set's static type tuple, so this lowers
to a single indexed array load.

Used by per-material shading kernels: the kernel knows its material type
`T` (from the `TypedHitRef{T}` it processes) and `vec_idx` (from the hit's
`material_idx.vec_idx`), and reads the material directly — no `with_index`
switch.
"""
@inline @generated function material_of_type(
    materials::Raycore.StaticMultiTypeSet{Data},
    ::Type{T},
    vec_idx::UInt32,
) where {Data, T}
    # Find the slot index where eltype matches T
    slot = 0
    for (i, vec_type) in enumerate(Data.parameters)
        if eltype(vec_type) === T
            slot = i
            break
        end
    end
    if slot == 0
        return :(error($("material_of_type: type $T not present in materials set")))
    end
    return quote
        @inbounds materials.data[$slot][vec_idx]
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# MultiTypeMaterialQueue — MultiTypeWorkQueue of TypedHitRef{T} per material slot
# ─────────────────────────────────────────────────────────────────────────────

"""
    MultiTypeMaterialQueue

Alias for `MultiTypeWorkQueue` whose queues hold `TypedHitRef{T}` items, one
queue per concrete material type in the scene. Built from a materials
`StaticMultiTypeSet` via `build_per_material_queues`.
"""
const MultiTypeMaterialQueue{Qs} = MultiTypeWorkQueue{Qs}

"""
    build_per_material_queues(materials::StaticMultiTypeSet, capacity::Integer, backend)

Build one `WorkQueue{TypedHitRef{T}}` per concrete material type `T` in the
scene's `materials` set. Queue iteration order matches the materials set's
data-tuple order, which is also what `material_idx.type_idx` indexes into,
so a hit with `type_idx == i` belongs in queue `i`.
"""
function build_per_material_queues(
    materials::Raycore.StaticMultiTypeSet{Data},
    capacity::Integer,
    mem,
) where {Data}
    item_types = map(vec_type -> TypedHitRef{eltype(vec_type)}, Data.parameters)
    return MultiTypeWorkQueue(tuple(item_types...), capacity, mem)
end

# ─────────────────────────────────────────────────────────────────────────────
# push_typed_hit! — `with_index` once at push site, queue dispatch is then
# data-driven (just an indexed tuple access on the queue side).
# ─────────────────────────────────────────────────────────────────────────────

# Internal: pushes the hit's index into the correct typed queue. `with_index`
# picks the concrete material type `T`; this helper pushes `TypedHitRef{T}`
# into the matching queue slot.
@inline function _push_to_typed_queue!(
    ::M, materials, mtmq::MultiTypeMaterialQueue, idx::Int32,
) where {M}
    # Find the queue slot for material type M at compile time, push the index.
    _push_typed_hit_slot!(mtmq, M, idx)
    return nothing
end

# Compile-time slot lookup + push: identical to material_of_type but for queues.
@inline @generated function _push_typed_hit_slot!(
    mtmq::MultiTypeMaterialQueue{Qs},
    ::Type{M},
    idx::Int32,
) where {Qs, M}
    # `Qs` is a Tuple{WorkQueue{TypedHitRef{T1}, ...}, WorkQueue{TypedHitRef{T2}, ...}, ...}.
    # The WorkQueue's first type parameter IS the item type (T in `WorkQueue{T, V, S}`).
    slot = 0
    for (i, q_type) in enumerate(Qs.parameters)
        item_type = q_type.parameters[1]              # WorkQueue{T, V, S} → T
        if item_type <: TypedHitRef && item_type.parameters[1] === M
            slot = i
            break
        end
    end
    if slot == 0
        return :(error($("_push_typed_hit_slot!: no queue for material type $M")))
    end
    return quote
        @inbounds push!(mtmq.queues[$slot], TypedHitRef{M}(idx))
        nothing
    end
end

"""
    push_typed_hit!(mtmq, materials, hit_surface_queue, hit::VPHitSurfaceWorkItem)

Store `hit` once in the shared `hit_surface_queue` and route its index into
the matching per-material typed queue. Uses `with_index` on `materials`
*once* to pick the concrete material type, then a compile-time tuple lookup
to push into the right queue slot.
"""
@propagate_inbounds function push_typed_hit!(
    mtmq::MultiTypeMaterialQueue, materials,
    hit_surface_queue::WorkQueue{VPHitSurfaceWorkItem},
    hit::VPHitSurfaceWorkItem,
)
    idx = push!(hit_surface_queue, hit)
    # Dropped on overflow (same contract as WorkQueue.push! itself) — never
    # publish an index whose payload slot wasn't written.
    idx <= hit_surface_queue.capacity || return nothing
    Raycore.with_index(_push_to_typed_queue!, materials, hit.material_idx, materials, mtmq, Int32(idx))
    return nothing
end

"""
    enqueue_after_intersection!(per_material_queue, hit_area_light_queue,
                                materials, hit_surface_queue,
                                hit::VPHitSurfaceWorkItem,
                                arealight_id::UInt32,
                                triangle_area::Float32, t_hit::Float32)

Mirrors pbrt-v4's `EnqueueWorkAfterIntersection`
(wavefront/intersect.h:48-149). For surface intersections (no medium) we
need two parallel pushes:

* If the hit triangle is part of an area light
  (`arealight_id > 0`), push a `VPHitAreaLightWorkItem` so the
  dedicated emitter kernel can run MIS for the indirect-ray hit.
* Always push the hit into `hit_surface_queue` and its index into the
  per-material typed queue — the material kernel will sample direct
  lighting + continuation rays.

`arealight_id`, `triangle_area`, `t_hit` are passed as separate
arguments rather than stored on `VPHitSurfaceWorkItem` because they're
emission-MIS-only — pbrt-v4's `MaterialEvalWorkItem` doesn't carry them
either.
"""
@propagate_inbounds function enqueue_after_intersection!(
    per_material_queue::MultiTypeMaterialQueue,
    hit_area_light_queue::WorkQueue{VPHitAreaLightWorkItem},
    materials,
    hit_surface_queue::WorkQueue{VPHitSurfaceWorkItem},
    hit::VPHitSurfaceWorkItem,
    arealight_id::UInt32,
    triangle_area::Float32,
    t_hit::Float32,
)
    push_arealight_hit!(hit_area_light_queue, arealight_id,
        hit.pi, hit.n, hit.uv, -hit.ray.d, hit, hit.beta, hit.r_u, hit.r_l,
        triangle_area, t_hit)
    push_typed_hit!(per_material_queue, materials, hit_surface_queue, hit)
    return nothing
end

"""
    push_arealight_hit!(hit_area_light_queue, arealight_id, pi, n, uv, wo,
                        path, beta, r_u, r_l, triangle_area, t_hit)

Queue the emission of a hit on an area light (`arealight_id > 0`) for
`vp_handle_emitters_kernel!`. `path` is the work item the ray belongs to; its
wavelengths, depth, previous interaction and pixel are what the MIS needs.
Also called for emissive null-material surfaces, which queue their emission and
let the ray pass.
"""
@propagate_inbounds function push_arealight_hit!(
    hit_area_light_queue::WorkQueue{VPHitAreaLightWorkItem},
    arealight_id::UInt32, pi::Point3f, n::Vec3f, uv::Point2f, wo::Vec3f,
    path, beta::SpectralRadiance, r_u::SpectralRadiance, r_l::SpectralRadiance,
    triangle_area::Float32, t_hit::Float32,
)
    arealight_id > UInt32(0) || return nothing
    push!(hit_area_light_queue, VPHitAreaLightWorkItem(
        arealight_id,
        pi, n, uv, wo,
        path.lambda, path.depth,
        beta, r_u, r_l,
        path.prev_intr_p, path.prev_intr_n,
        path.specular_bounce,
        path.pixel_index,
        triangle_area, t_hit,
    ))
    return nothing
end

"""
The medium a ray is in after crossing a null-material surface. Only a surface
that separates two media swaps it: a null surface with the same medium on both
sides (an emissive sheet, say) leaves the ray in the medium it was in, where
`get_medium_index` would answer that shared medium, vacuum for a sheet in fog.
"""
@propagate_inbounds null_crossing_medium(mi, current::SetKey, d, n) =
    is_medium_transition(mi) ? get_medium_index(mi, d, n) : current
