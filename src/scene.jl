# Scene type and operations

# Abstract scene type for dispatch - allows both mutable and immutable implementations
abstract type AbstractScene end

"""
    TriangleMeta

Per-triangle metadata baked into the TLAS.
"""
struct TriangleMeta
    medium_interface_idx::UInt32  # Index into scene.media_interfaces
    primitive_index::UInt32       # Face index within the mesh (1-based)
    arealight_id::UInt32          # Packed SetKey of its light in scene.lights, 0 = none (`pack_arealight`)
end

# Scene stores lights, accelerator, materials, and media directly.
# Materials and media use MultiTypeSet (or StaticMultiTypeSet for GPU/kernels).
# Textures are stored within the materials MultiTypeSet (materials have TextureRef fields).
# LightVec can be Tuple (for type-stable iteration) or AbstractVector.
struct Scene{Accel,LightVec,MatVec<:AbstractVector,MedVec<:AbstractVector, MI<:AbstractVector, Bounds} <: AbstractScene
    lights::LightVec
    accel::Accel  # BVH, TLAS, or any accelerator supporting closest_hit/any_hit
    materials::MatVec  # MultiTypeSet
    media::MedVec  # MultiTypeSet
    media_interfaces::MI  # MultiTypeSet
    bounds::Bounds  # RefValue{Tuple} - adapted to device RefValue for GPU
end

# Accessors for scene bounds (dereference the RefValue)
@inline world_bound(scene::Scene) = scene.bounds[][1]
@inline world_center(scene::Scene) = scene.bounds[][2].center
@inline world_radius(scene::Scene) = scene.bounds[][2].r

"""
    Scene(; backend=Mantle.defaultbackend())

Create an empty mutable Scene for incremental construction.

# Example
```julia
scene = Scene()
push!(scene, mesh, material)           # Add geometry
push!(scene, PointLight(...))   # Add lights
push!(scene, AmbientLight(...))
sync!(scene)  # Build acceleration structure
```
"""
# Typed over `Triangle{TriangleMeta}` while EMPTY, as the hardware default is by
# its type parameter: an untyped TLAS with no BLAS falls back to
# `Triangle{UInt32}`, whose metadata has no `medium_interface_idx`, so an empty
# scene compiled a kernel that could only throw, and Metal refused to compile it.
# That is every render before the first plot lands, and every one after the last
# is deleted.
default_accel(backend, ::Val{false}) = TLAS(backend; primtype = Raycore.Triangle{TriangleMeta})
# Val{true} defined in hw-rt.jl -> HWTLAS(backend) for any HW-capable backend

function Scene(; backend=Mantle.defaultbackend(), accel=nothing, hw_accel::Bool=false)
    tlas = accel !== nothing ? accel : default_accel(backend, Val(hw_accel))
    lights = MultiTypeSet(backend)
    materials = MultiTypeSet(backend)
    media = MultiTypeSet(backend)
    media_interfaces = Mantle.devicearray(backend, MediumInterfaceIdx, 0)
    Scene(lights, tlas, materials, media, media_interfaces, Ref((Bounds3(), Sphere(Point3f(0), 0f0))))
end

function Scene(mesh_material_pairs::Vector{<:Tuple}; lights = (), backend = Mantle.defaultbackend(), accel = nothing, hw_accel::Bool=false)
    scene = Scene(; backend=backend, accel=accel, hw_accel=hw_accel)
    for light in lights
        push!(scene, light)
    end
    foreach(mesh_material_pairs) do args
        return push!(scene, args...)
    end
    # Build the BVH structure
    sync!(scene)
    return scene
end

# ============================================================================
# Change notification
#
# A recorded plan is packed with the scene's addresses and pass structure once,
# and nothing is compared per render to find out whether that is still right.
# Instead the change announces itself: the mutators below and `sync!` call
# `notify_scene_changed`, which drops the recorded plans of every integrator
# listening to this scene. The next `render!` rebuilds.
# ============================================================================

"""
    invalidate_plans!(integrator)

Drop the plans `integrator` recorded against a scene that just changed.
Defined per integrator (VolPath's is in `volpath/graph.jl`); the scene side
only knows the verb.
"""
function invalidate_plans! end

# Weak on both ends: a scene keeps no integrator alive, an integrator no scene.
# Keyed by `objectid` rather than a WeakKeyDict, whose keys must be mutable and
# a Scene is not; the entry's own WeakRef is checked against the caller, so a
# collected scene's id being reused reads as "someone else's entry" and is
# replaced rather than followed.
#
# The WeakRef cannot wrap the scene itself: Scene is immutable, and a WeakRef
# to an immutable value may be cleared while the value is still alive (there
# is no object identity for the GC to track). That both spuriously replaced
# entries here (dropping already-registered listeners) and made
# `notify_scene_changed` bail out on a live scene. `_scene_token` returns a
# mutable field with real identity that lives exactly as long as the scene.
const SCENE_LISTENERS = IdDict{UInt,Tuple{WeakRef,Vector{WeakRef}}}()
const SCENE_LISTENERS_LOCK = ReentrantLock()

# The `bounds` RefValue is mutable, never reassigned, and dies with the scene.
_scene_token(scene::Scene) = scene.bounds
# Mutable scene implementations have real identity and can be wrapped directly.
_scene_token(scene::AbstractScene) = scene

"""Register `integrator` for plan invalidation when `scene` changes."""
function listen_scene!(scene::AbstractScene, integrator)
    lock(SCENE_LISTENERS_LOCK) do
        id = objectid(scene)
        tok = _scene_token(scene)
        e = get(SCENE_LISTENERS, id, nothing)
        if e === nothing || e[1].value !== tok
            e = (WeakRef(tok), WeakRef[])
            SCENE_LISTENERS[id] = e
        end
        v = e[2]
        any(w -> w.value === integrator, v) || push!(v, WeakRef(integrator))
    end
    return nothing
end

"""
    notify_scene_changed(scene)

Drop every listening integrator's plans. Called where a scene stops being what
it was: the `push!`/`update_material!` mutators, and `sync!` when it rebuilt.
"""
notify_scene_changed(scene::AbstractScene) = notify_listeners(invalidate_plans!, scene)

"""
    notify_lights_changed(scene)

Tell every listening integrator that a light changed its values in place
(`update_light!`): the same lights in the same slots, moved, recoloured,
dimmed. Their plans read the lights from the scene's buffers and stay; what an
integrator derived from where the lights are is its to refresh, see
[`refresh_lights!`](@ref).
"""
notify_lights_changed(scene::AbstractScene) = notify_listeners(i -> refresh_lights!(i, scene), scene)

"""
    refresh_lights!(integrator, scene)

Bring what `integrator` derived from the scene's lights up to date after an
in-place change. An integrator that derives nothing it could refresh drops
its plans.
"""
refresh_lights!(integrator, scene::AbstractScene) = invalidate_plans!(integrator)

function notify_listeners(f, scene::AbstractScene)
    lock(SCENE_LISTENERS_LOCK) do
        e = get(SCENE_LISTENERS, objectid(scene), nothing)
        (e === nothing || e[1].value !== _scene_token(scene)) && return nothing
        v = e[2]
        filter!(w -> w.value !== nothing, v)
        foreach(w -> f(w.value), v)
        nothing
    end
    return nothing
end

# ============================================================================
# Scene push! methods
# ============================================================================

function Base.push!(scene::Scene, light::Light)
    key = push!(scene.lights, light)
    notify_scene_changed(scene)
    return key
end

"""
    update_light!(scene, key, light)

Replace the light stored under `key` — what `push!(scene.lights, light)`
returned — in place. The same concrete type only, as every `MultiTypeSet`
update: a light that moves or dims is this, a different kind of light is not.

The slot is written where it lies, so recorded plans, which read the lights
from that buffer, stay valid: a light following the camera used to throw them
away and record them again every frame.
"""
function update_light!(scene::Scene, key::SetKey, light::Light)
    Raycore.update!(scene.lights, key, light)
    notify_lights_changed(scene)
    return nothing
end

function Base.push!(scene::Scene, material::Material)
    return push!(scene, MediumInterface(material))
end

function Base.push!(scene::Scene, medium::Medium)
    key = push!(scene.media, medium)
    notify_scene_changed(scene)
    return key
end

Base.push!(scene::Scene, ::Nothing) = SetKey()

"""
    interface_key(scene, medium::MediumInterface) -> MediumInterfaceIdx

Push `medium`'s material and both media and return the keys they got: what
`scene.media_interfaces` stores for it. `push!(scene, ::MediumInterface)` and
the batched `push_interfaces!` both build interfaces with this.
"""
function interface_key(scene::Scene, medium::MediumInterface)
    mat_idx = push!(scene.materials, resolve_material(scene, surface_material(medium.material)))
    return MediumInterfaceIdx(mat_idx, push!(scene, medium.inside), push!(scene, medium.outside), medium.shadows)
end

function Base.push!(scene::Scene, medium::MediumInterface)
    mi = interface_key(scene, medium)
    idx = findfirst(x -> mi === x, scene.media_interfaces)
    if idx === nothing
        @allowscalar push!(scene.media_interfaces, mi)
        idx = length(scene.media_interfaces)
    end
    notify_scene_changed(scene)
    return UInt32(idx)
end

"""
    update_material!(scene, idx::UInt32, new_medium::Medium)
    update_material!(scene, idx::UInt32, new_material::Material)

Store `new_material` (or `new_medium`, or a whole `MediumInterface`) in place of what the medium
interface `idx` holds. Does **not** synchronize with the GPU. The geometry using the interface
keeps the opacity it was pushed with: a material that lets shadow rays through where the old one
did not ([`opaque`](@ref)) needs the mesh's handle
(`update_material!(scene, handle::SceneHandle, material)`) or a new push.

A material of the stored one's type is written into its slot, so the scene keeps its recorded
plans. Another kind of material, or a shader of other node families, is another type: it takes a
slot of its own and the interface is pointed at it (`store_interface!`); every face and instance
drawn with the interface follows. The slot it leaves is vacated for the next material of its type.

Invariants:
- CPU-side bookkeeping and MultiTypeSet slot contents are updated.
- If a texture slot in the backing MultiTypeSet has to be reshaped,
  `copyto_texture!` handles the retirement sync internally (see its
  docstring) — callers don't need to pair this with anything.
- If the update rendered any transitively-owned resource unreachable,
  the caller is responsible for pairing the release with a `sync!` call:
  this function does not force one.
"""
function update_material!(scene::Scene, idx::UInt32, new)
    updating_in_place!(scene) do
        mi = @allowscalar scene.media_interfaces[idx]
        moved = store_interface!(scene, mi, new)
        moved === mi || @allowscalar scene.media_interfaces[idx] = moved
    end
end

"""
    updating_in_place!(f, scene)

Run the slot update `f` and keep the recorded plans unless it moved a buffer.
An update writes into the slot buffers the plans were packed with, so they stay
right; only one that had to store a new texture or move one (a reshape past
capacity) bumps a set's `layout`, and only that drops them. So does a shader whose
BSDF nodes read more computed values than the plans' hits have room for
(`shader_inputs_width`).
"""
function updating_in_place!(f, scene::Scene)
    before = (scene.materials.layout, scene.media.layout, shader_inputs_width(scene.materials))
    f()
    (scene.materials.layout, scene.media.layout, shader_inputs_width(scene.materials)) == before ||
        notify_scene_changed(scene)
    return nothing
end

"""
    store_interface!(scene, mi::MediumInterfaceIdx, new) -> MediumInterfaceIdx

Store `new` in place of what the interface `mi` points at and return the interface that points
at where it is now: a material replaces `mi`'s surface material, a medium its inside medium, a
`MediumInterface` all three and the shadow setting. Writes no interface; `update_material!` does.

`MediumInterface` is itself a `Material`, so it needs its own method: as a material it would be
stuffed whole into the surface slot. A side it leaves out (`inside = nothing`, a `NullMaterial`)
vacates the slot that side had.
"""
store_interface!(scene::Scene, mi::MediumInterfaceIdx, medium::Medium) =
    MediumInterfaceIdx(mi.material, store_medium!(scene, mi.inside, medium), mi.outside, mi.shadows)

store_interface!(scene::Scene, mi::MediumInterfaceIdx, material::Material) =
    MediumInterfaceIdx(store_surface!(scene, mi.material, material), mi.inside, mi.outside, mi.shadows)

store_interface!(scene::Scene, mi::MediumInterfaceIdx, new::MediumInterface) =
    MediumInterfaceIdx(store_surface!(scene, mi.material, new.material), store_medium!(scene, mi.inside, new.inside),
                       store_medium!(scene, mi.outside, new.outside), new.shadows)

"""`medium` stored in place of the medium under `key` (`Raycore.store!`); `nothing` vacates it."""
store_medium!(scene::Scene, key::SetKey, medium::Medium) = Raycore.store!(scene.media, key, medium)
store_medium!(scene::Scene, key::SetKey, ::Nothing) = (delete!(scene.media, key); SetKey())

"""
    store_surface!(scene, key, material) -> SetKey

Store what `material` is rendered with (`surface_material`) in place of the surface material under
`key` in `scene.materials`, and return the key it is stored under now (`Raycore.store!`): `key`
when it is of the stored material's type.

A mix that picks between its materials is stored by key (`MixMaterial`): each of them is stored
in place of the one the mix had, then the mix itself, whose factor texture is copied into the
slot it already has. A foam mask changed every frame is a copy, not a new texture.
"""
store_surface!(scene::Scene, key::SetKey, material::Material) =
    store_surface!(scene, key, stored_surface(scene, key), surface_material(material))

stored_surface(scene::Scene, key::SetKey) = Raycore.is_valid(key) ? scene.materials[key] : nothing

store_surface!(scene::Scene, key::SetKey, ::Any, material::Material) =
    Raycore.store!(scene.materials, key, resolve_material(scene, material))

"""
    vacate_surface!(scene, key)

Vacate the surface material under `key` and the materials it references (a `MixMaterial`'s).
"""
function vacate_surface!(scene::Scene, key::SetKey)
    Raycore.is_valid(key) || return nothing
    old = scene.materials[key]
    delete!(scene.materials, key)
    vacate_parts!(scene, old)
    return nothing
end
vacate_parts!(::Scene, ::Any) = nothing

struct SceneHandle
    scene::Scene
    interface::UInt32 # Index into  scene.media_interfaces
    geometry::TLASHandle # handle for geometry in TLAS
    area_lights::Vector{SetKey} # Stable emitter slots owned by this mesh
end

SceneHandle(scene, interface, geometry) = SceneHandle(scene, interface, geometry, SetKey[])

function Base.push!(scene::Scene, mesh::AbstractGeometry, materialidx::UInt32; arealight_indices=nothing)
    handle = push!(scene.accel, mesh, materialidx; arealight_indices=arealight_indices)
    notify_scene_changed(scene)
    return SceneHandle(scene, materialidx, handle)
end

function Base.push!(scene::Scene, mesh::AbstractGeometry, material::Material; arealight_indices=nothing)
    mat_idx = push!(scene, material)
    return push!(scene, mesh, mat_idx; arealight_indices=arealight_indices)
end


# NOTE: push!(scene, mesh::GB.Mesh, material) and area light helpers are in scene-mesh.jl
# (included after materials and lights are defined)

# ============================================================================
# Scene operations
# ============================================================================

"""
    sync!(scene::Scene)

Build/rebuild the acceleration structure, update scene bounds, and **wait
for the GPU to finish** all work queued on the scene's backend (via
`Raycore.sync!`).

After `sync!(scene)` returns:
- The acceleration structure reflects all prior `push!` / `delete!`.
- Scene bounds are up to date on the CPU.
- The GPU is idle for this scene's backend — it is safe to release
  resources that are no longer reachable (old materials / media / AS
  storage that a just-completed `update_material!` or `push!` rendered
  unreachable).

Invariants for callers (see the accel's `sync!` docstrings for detail):
- `push!`, `delete!`, and `update_material!` do NOT synchronize; they
  mutate CPU bookkeeping and mark state dirty. Pair any follow-up
  release of transitively-owned resources with a `sync!` call.
"""
function sync!(scene::Scene{<:TLAS})
    # Whether this sync! is about to change anything — the flag Raycore.sync!
    # consumes. Only a sync! that did work invalidates plans; the no-op kind is
    # meant to be called liberally. `transforms_dirty` is deliberately NOT here:
    # a refit writes the same buffers in place, which a recorded plan survives.
    changed = scene.accel.dirty
    sync!(scene.accel)   # Raycore.sync! runs KA.synchronize at the end.
    bound = Raycore.world_bound(scene.accel)
    scene.bounds[] = (bound, bounding_sphere(bound))
    changed && notify_scene_changed(scene)
    return scene
end

# Adapt Scene for GPU kernels - converts arrays to device arrays
function Adapt.adapt_structure(to, scene::Scene)
    Scene(
        Adapt.adapt(to, scene.lights),  # Convert MultiTypeSet → StaticMultiTypeSet if needed
        Adapt.adapt(to, scene.accel),
        Adapt.adapt(to, scene.materials),
        Adapt.adapt(to, scene.media),
        Adapt.adapt(to, scene.media_interfaces),
        Adapt.adapt(to, scene.bounds)  # RefValue → device RefValue
    )
end

# Common interface for both scene types
@propagate_inbounds function intersect!(scene::AbstractScene, ray::AbstractRay)
    intersect!(scene.accel, ray)
end

@propagate_inbounds function intersect_p(scene::AbstractScene, ray::AbstractRay)
    hit_found, _, _, _ = any_hit(scene.accel, ray)
    return hit_found
end

# ============================================================================
# Pretty printing
# ============================================================================

function Base.show(io::IO, ::MIME"text/plain", scene::Scene)
    n_lights = length(scene.lights)
    n_material_types = length(scene.materials)
    n_materials_total = sum(length, scene.materials; init=0)
    accel = scene.accel
    accel_type = nameof(typeof(accel))

    println(io, "Scene ($accel_type):")
    println(io, "  Lights:     ", n_lights)
    if n_lights > 0
        for (i, light) in enumerate(scene.lights)
            println(io, "    [", i, "] ", typeof(light).name.name)
        end
    end
    println(io, "  Materials:  ", n_materials_total, " ($n_material_types types)")
    if accel isa TLAS
        n_instances = length(accel.instances)
        n_geometries = length(accel.blas_array)
        println(io, "  Geometries: ", n_geometries)
        println(io, "  Instances:  ", n_instances)
    end
    bound = world_bound(scene)
    print(io,   "  Bounds:     ", bound.p_min, " to ", bound.p_max)
end

function Base.show(io::IO, scene::Scene)
    if get(io, :compact, false)
        n_lights = length(scene.lights)
        accel_type = nameof(typeof(scene.accel))
        n_material_types = length(scene.materials)
        print(io, "Scene($accel_type, lights=$n_lights, materials=$n_material_types)")
    else
        show(io, MIME("text/plain"), scene)
    end
end
