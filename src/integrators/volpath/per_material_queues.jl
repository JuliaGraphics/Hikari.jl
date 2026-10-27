# Per-material-type queue infrastructure (pbrt-v4 wavefront pattern).
#
# The trace kernel leaves each surface hit as a `VPSurfaceHit` (a hit record and
# where the ray's path state is), and one shading kernel per concrete material
# type expands and shades it. pbrt-v4 splits its shading queue the same way —
# `MaterialEvalQueue<ConcreteMaterial>` in wavefront/workitems.h, one kernel per
# type via `ForEachType` (surfscatter.cpp:41) — so each kernel compiles exactly
# one material's code and the threads of a warp share it. That is the coherence
# NVIDIA's shader execution reordering gives a ray-tracing pipeline, on every
# backend.
#
#   * `TypedHitRef{T}` — a 4-byte index into the shared `surface_hits` store
#     that carries the hit's concrete material type `T` in its type, so Julia's
#     dispatch picks the shading kernel's specialisation. The hit itself is
#     stored ONCE; a typed queue per material sized for every path holds indices
#     only (with full items that was `n_types × n_paths × 304 B`, 4.76 GiB on
#     Crown's 12-material 1.4 Mpx render).
#
#   * `MultiTypeMaterialQueue` — a `MultiTypeWorkQueue` of `TypedHitRef{T}` for
#     each `T` in the scene's `StaticMultiTypeSet`.
#
#   * `enqueue_surface_hit!` — stores the hit and routes its index into the
#     queue of its material's type, with `with_index` once per hit.
#
#   * `material_of_type(materials, T, vec_idx)` — the concrete material of a
#     known type, a compile-time slot lookup and one array load.

using Adapt
import KernelAbstractions as KA
import Raycore

# ─────────────────────────────────────────────────────────────────────────────
# TypedHitRef{T}: 4-byte index that lifts the material type into the type
# parameter of the work item.
# ─────────────────────────────────────────────────────────────────────────────

"""
    TypedHitRef{T}

4-byte index into the shared `surface_hits` store, tagged with the concrete
material type `T` at the type level. Per-material shading kernels dispatch
on `TypedHitRef{T}` — each `T` produces a separately compiled kernel via
Julia's normal multiple dispatch — and load the `VPSurfaceHit` from
`surface_hits.items[ref.idx]`.
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
# enqueue_surface_hit! — `with_index` once at the push site; the shading kernel
# that drains a typed queue sees one concrete `TypedHitRef{T}`.
# ─────────────────────────────────────────────────────────────────────────────

# Pushes the hit's index into the typed queue of material type `M`, which
# `with_index` picked.
@inline function _push_to_typed_queue!(
    ::M, materials, mtmq::MultiTypeMaterialQueue, idx::Int32,
) where {M}
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
    wants_shading(depth, max_depth, arealight_id, materials, mat_idx) -> Bool

Whether a surface hit goes to its material's shading kernel: every hit below the
depth limit, and at the limit one that can add light there, on an area light or
with a material that emits (`vp_shade_resolved_hit!` shades nothing else there).
"""
@propagate_inbounds wants_shading(depth::Int32, max_depth::Int32, arealight_id::UInt32, materials, mat_idx::SetKey) =
    depth < max_depth || arealight_id > UInt32(0) || Raycore.with_index(emits, materials, mat_idx)

"""
    enqueue_surface_hit!(per_material_queue, materials, surface_hits, mat_idx, hit::VPSurfaceHit)

Store `hit` once in `surface_hits` and route its index into the queue of the
material type of `mat_idx`, the material of the mesh it hit. A `MixMaterial` hit
goes to the mix's queue, whose kernel picks the component per hit.
"""
@propagate_inbounds function enqueue_surface_hit!(
    mtmq::MultiTypeMaterialQueue, materials, surface_hits::WorkQueue{VPSurfaceHit},
    mat_idx::SetKey, hit::VPSurfaceHit,
)
    idx = push!(surface_hits, hit)
    # Dropped on overflow (same contract as WorkQueue.push! itself) — never
    # publish an index whose payload slot wasn't written.
    idx <= surface_hits.capacity || return nothing
    Raycore.with_index(_push_to_typed_queue!, materials, mat_idx, materials, mtmq, Int32(idx))
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
