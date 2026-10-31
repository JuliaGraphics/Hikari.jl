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
#   * `MaterialQueues` — the hits' indices into the shared `surface_hits` store,
#     a segment and a counter per slot of the scene's `StaticMultiTypeSet`. The
#     hit itself is stored ONCE; a segment per material sized for every path
#     holds indices only (with full items that was `n_types × n_paths × 304 B`,
#     4.76 GiB on Crown's 12-material 1.4 Mpx render). Its type names no
#     material, so neither do the kernels that fill it.
#
#   * `SlotQueue{T}` — the segment of the slot holding material type `T`, what
#     that type's shading kernel drains, and `TypedHitRef{T}` the index it reads
#     there, which carries `T` so Julia's dispatch picks the specialisation.
#
#   * `enqueue_surface_hit!` — stores the hit and routes its index into the
#     segment of its material's slot.
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
    materials::Union{Raycore.StaticMultiTypeSet{Data}, Raycore.SetView{<:Any, <:Any, Data}},
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
# MaterialQueues — a segment and a counter per material slot
# ─────────────────────────────────────────────────────────────────────────────

"""
    MaterialQueues

The surface hits waiting for their material's shading kernel: per slot `s` of the scene's
materials set, `sizes[s]` indices into the shared `surface_hits` store at
`items[(s - 1) * capacity + 1 : s * capacity]`, and `emits[s]`, whether the slot's type can emit
(`emits`), which decides whether a hit at the depth limit is shaded at all.

The kernels that fill it (the trace, the medium sampling) take it as it is, whatever the
materials. It was a tuple of queues typed per material type, so a new type changed the trace
kernel's type and compiled it again, the biggest kernel there is, with every node edit.
"""
struct MaterialQueues{V <: AbstractVector{Int32}, F <: AbstractVector{Bool}}
    items::V
    sizes::V
    emits::F
    capacity::Int32
end

Adapt.adapt_structure(to, q::MaterialQueues) =
    MaterialQueues(Adapt.adapt(to, q.items), Adapt.adapt(to, q.sizes), Adapt.adapt(to, q.emits), q.capacity)

"""
    MaterialQueues(materials::StaticMultiTypeSet, capacity, mem)

A segment of `capacity` hits for every slot of `materials`, in memory `mem` owns.
"""
function MaterialQueues(materials::Raycore.StaticMultiTypeSet{Data}, capacity::Integer,
                        mem::DeviceMemory) where {Data}
    types = map(eltype, Tuple(Data.parameters))
    n = length(types)
    items = alloc!(mem, Int32, max(n, 1) * Int(capacity))
    sizes = alloc!(mem, Int32, max(n, 1), Int32(0))
    emitting = alloc!(mem, Bool, max(n, 1), false)
    n > 0 && copyto!(emitting, collect(Bool, map(emits, types)))
    return MaterialQueues(items, sizes, emitting, Int32(capacity))
end

_collect_size_counters(acc::Tuple, q::MaterialQueues, rest...) =
    _collect_size_counters((acc..., q.sizes), rest...)

"""The material types of `materials`' slots, in slot order."""
slot_types(@nospecialize(materials::Raycore.StaticMultiTypeSet)) =
    map(eltype, Tuple(typeof(materials).parameters[1].parameters))

"""
    SlotQueue{T}(queues, slot)

The hits of slot `slot` of `queues`, whose material type is `T`: what that type's shading
kernel drains. The count is `sizes[slot]`, which the dispatch is sized by too
(`DeviceRange(sizes; at = slot)`).
"""
struct SlotQueue{T, V <: AbstractVector{Int32}}
    items::V
    sizes::V
    slot::Int32
    offset::Int32
end

SlotQueue{T}(q::MaterialQueues{V}, slot::Integer) where {T, V} =
    SlotQueue{T, V}(q.items, q.sizes, Int32(slot), Int32((slot - 1) * q.capacity))

function Adapt.adapt_structure(to, q::SlotQueue{T}) where {T}
    items, sizes = Adapt.adapt(to, q.items), Adapt.adapt(to, q.sizes)
    return SlotQueue{T, typeof(items)}(items, sizes, q.slot, q.offset)
end

"""How many hits the queue holds; read on the device."""
@propagate_inbounds queuelength(q::SlotQueue) = q.sizes[q.slot]

@propagate_inbounds Base.getindex(q::SlotQueue{T}, j::Integer) where {T} = TypedHitRef{T}(q.items[q.offset + j])

"""The material type whose hits the queue `q` holds."""
queued_material(::SlotQueue{T}) where {T} = T

# ─────────────────────────────────────────────────────────────────────────────
# shade_set — what the shading kernel of one material type reads of the set
# ─────────────────────────────────────────────────────────────────────────────

"""
    shade_set(materials::StaticMultiTypeSet, T) -> SetView

What the shading kernel of material type `T` reads of the scene's `materials`, as a view under the
set's own slot numbers: the slot of `T`, the slots of every type a shadow ray may be let through
(`may_cut`), and the texture slots their shading reads: every image and texture program
(`NodeInstr` with `NodeFeature.TEXTURE`), and of the shaders' programs only `T`'s own and the
transparency programs of the types that may be cut out.

The kernel's type names those slots only. A material of another type added to the scene, or a
shader of other node families, adds slots the view does not hold, so the kernel compiled for it
stays what it was: with the whole set, every shading kernel compiled again for each new material
type, and each of them grew by the programs of all the others (a texture read compiled every
interpreter in the scene), so a node added to one material cost minutes.

A mix (`MixMaterial`) picks one of any of the materials per hit, and its kernel takes the set.
"""
function shade_set(@nospecialize(materials::StaticMultiTypeSet), @nospecialize(T::Type))
    T <: MixMaterial && return materials
    D, X = typeof(materials).parameters
    types = map(eltype, D.parameters)
    data = Tuple(i for (i, M) in enumerate(types) if M === T || may_cut(M))
    families = (program_families(T)..., Iterators.flatten(alpha_families(types[i]) for i in data)...)
    textures = Tuple(i for (i, A) in enumerate(X.parameters) if read_by(eltype(eltype(A)), families))
    return Raycore.setview(materials, data, textures)
end

"""Whether a shading kernel reading the shader programs of `families` reads texture slots of element type `E`."""
read_by(::Type, families) = true
read_by(::Type{NodeInstr{F}}, families) where {F} = F & NodeFeature.TEXTURE != zero(UInt32) || F in families

"""
    cut_set(materials::StaticMultiTypeSet) -> SetView

What the kernels that trace through cut-out surfaces read of the scene's `materials`: the slots
of the types a ray may pass (`may_cut`) and the texture slots their transparency reads, none at
all when no type may. A material of another type, or a shader of other node families, leaves it
as it was, and so the trace kernel compiled for it.
"""
function cut_set(@nospecialize(materials::StaticMultiTypeSet))
    D, X = typeof(materials).parameters
    types = map(eltype, D.parameters)
    data = Tuple(i for (i, M) in enumerate(types) if may_cut(M))
    isempty(data) && return Raycore.setview(materials, (), ())
    families = Tuple(Iterators.flatten(alpha_families(types[i]) for i in data))
    textures = Tuple(i for (i, A) in enumerate(X.parameters) if read_by(eltype(eltype(A)), families))
    return Raycore.setview(materials, data, textures)
end

# ─────────────────────────────────────────────────────────────────────────────
# enqueue_surface_hit! — the hit's index into its material slot's segment
# ─────────────────────────────────────────────────────────────────────────────

"""
    wants_shading(depth, max_depth, arealight_id, queues, mat_idx) -> Bool

Whether a surface hit goes to its material's shading kernel: every hit below the
depth limit, and at the limit one that can add light there, on an area light or
with a material that emits (`vp_shade_resolved_hit!` shades nothing else there).
"""
@propagate_inbounds wants_shading(depth::Int32, max_depth::Int32, arealight_id::UInt32, queues::MaterialQueues,
                                  mat_idx::SetKey) =
    depth < max_depth || arealight_id > UInt32(0) || queues.emits[mat_idx.type_idx]

"""
    enqueue_surface_hit!(queues, surface_hits, mat_idx, hit::VPSurfaceHit)

Store `hit` once in `surface_hits` and route its index into the segment of the
slot of `mat_idx`, the material of the mesh it hit. A `MixMaterial` hit goes to
the mix's slot, whose kernel picks the component per hit.
"""
@propagate_inbounds function enqueue_surface_hit!(
    queues::MaterialQueues, surface_hits::WorkQueue{VPSurfaceHit}, mat_idx::SetKey, hit::VPSurfaceHit,
)
    idx = push!(surface_hits, hit)
    # Dropped on overflow (same contract as WorkQueue.push! itself) — never
    # publish an index whose payload slot wasn't written.
    idx <= surface_hits.capacity || return nothing
    slot = Int32(mat_idx.type_idx)
    i = @atomic :monotonic queues.sizes[slot] += Int32(1)
    i <= queues.capacity && (queues.items[(slot - Int32(1)) * queues.capacity + i] = Int32(idx))
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
