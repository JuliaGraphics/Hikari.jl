# ============================================================================
# GB.Mesh-based push! API — Hikari handles material resolution + area lights
# ============================================================================
# This file is included after all materials and lights are defined.

using LinearAlgebra: I, norm

# SBT slot for the most recently pushed material. The materials set converts
# texture wrappers to bare scalars at push! time (e.g. `Diffuse{Texture{RGB,
# 0, Array{RGB, 0}}, ...}` → `Diffuse{RGB, Float32}`), so the type that ends
# up in the chit-tuple slot order is the *converted* type — not the type the
# user pushed. `push!(::MediumInterface)` stashes the SetKey it got back from
# `MultiTypeSet.push!`, and we read the `type_idx` from that here.
function _last_pushed_sbt_offset()
    setkey = _LAST_MAT_SETKEY[]
    setkey.type_idx == UInt32(0) && return UInt32(0)
    return UInt32(setkey.type_idx - 1)
end

# Single material for entire mesh
# A material that carries its own geometry contributes ITS triangles, not the
# caller's. Both spellings, because the generic mesh method above is equally
# specific on its middle argument and Julia calls that a tie — the same reason
# `FEMMaterial` has two. A material that can be SOLVED overrides these with a
# method of its own and never reaches here; `fem.jl` is that case.
Base.push!(scene::Scene, ::GeometryBasics.Mesh, m::GeneratedGeometry; kw...) =
    push!(scene, tessellate(m), shadingmaterial(m); kw...)
Base.push!(scene::Scene, ::Any, m::GeneratedGeometry; kw...) =
    push!(scene, tessellate(m), shadingmaterial(m); kw...)

function Base.push!(scene::Scene, mesh::GeometryBasics.Mesh, material::Material;
                    transform::Mat4f=Mat4f(I))
    mat_idx = push!(scene, material)
    face_meta = build_face_meta(scene, mesh, mat_idx, material)
    mesh_with_meta = GeometryBasics.mesh(mesh; face_meta=GeometryBasics.per_face(face_meta, mesh))
    sbt_offset = _last_pushed_sbt_offset()
    handle = push!(scene.accel, mesh_with_meta, transform; sbt_offset=sbt_offset)
    return SceneHandle(scene, mat_idx, handle, mesh_area_lights(face_meta))
end

"""
    push!(scene::Scene, mesh::GeometryBasics.Mesh, mat_idx::UInt32, material::Material;
          transform=Mat4f(I))

Push geometry pointing at a **pre-existing** medium-interface slot.  Callers
are responsible for having already brought the slot's stored material up to
date via [`update_material!`](@ref) — this overload only builds the face
metadata / BLAS and registers the instance.  RayMakie's mesh-rebuild path
uses this to recycle a single material slot across many geometry rebuilds
instead of growing `scene.materials` and `scene.media_interfaces` on every
frame.
"""
function Base.push!(scene::Scene, mesh::GeometryBasics.Mesh, mat_idx::UInt32,
                    material::Material; transform::Mat4f=Mat4f(I), area_lights=SetKey[])
    face_meta = build_face_meta(scene, mesh, mat_idx, material; area_lights)
    mesh_with_meta = GeometryBasics.mesh(mesh; face_meta=GeometryBasics.per_face(face_meta, mesh))
    # The interface already names the stored material and therefore its SBT
    # type slot. Converting the raw material just to discover that type stores
    # all its textures again, leaking a full feather map on every mesh rebuild.
    mi = @allowscalar scene.media_interfaces[mat_idx]
    sbt_offset = Raycore.is_valid(mi.material) ? mi.material.type_idx - UInt32(1) : UInt32(0)
    handle = push!(scene.accel, mesh_with_meta, transform; sbt_offset=sbt_offset)
    return SceneHandle(scene, mat_idx, handle, mesh_area_lights(face_meta))
end

"""
    push!(scene::Scene, mesh::GeometryBasics.Mesh,
          materials::AbstractVector{<:Material},
          transforms::AbstractVector{Mat4f}) -> Vector{SceneHandle}

N-instance push: build **one** BLAS from `mesh` and append N
`InstanceDescriptor`s — one per (material, transform) pair.  Each
instance's `instance_id` carries its own `medium_interface_idx`, so the
hit shader resolves material per-instance via `resolve_mi_idx`.

This is the path `meshscatter` should use.  It avoids the "N BLASes with
identical geometry" explosion of calling the single-transform push!
per instance (~1 GB / frame memory growth in the dolphin demo).

`materials` and `transforms` must have equal length.  Emissive materials
are not yet supported here — an emitter would need per-instance area
lights and per-instance-transformed geometry, which is a different
feature.  Use the per-mesh `push!` for emitters.
"""
function Base.push!(scene::Scene, mesh::GeometryBasics.Mesh,
                    materials::AbstractVector{<:Material},
                    transforms::AbstractVector{Mat4f};
                    reuse_mi_indices::Union{Nothing, AbstractVector{UInt32}}=nothing)
    length(materials) == length(transforms) ||
        throw(ArgumentError("materials ($(length(materials))) and transforms ($(length(transforms))) must have same length"))

    for m in materials
        if get_emission_info(m) !== nothing
            throw(ArgumentError("per-instance emissive materials are not supported; use the single-transform push! for each emitter"))
        end
    end

    # Resolve one `mi_idx` per instance.  If the caller hands us
    # `reuse_mi_indices` (the indices returned by a prior push for the same
    # meshscatter/streamplot), update those slots in place via
    # `update_material!` — no growth of scene.materials at all.  Any excess
    # (`length(materials) > length(reuse_mi_indices)`) is pushed as new
    # MediumInterfaces, so the materials vector grows only up to the high
    # water mark of instance count.
    n = length(materials)
    if reuse_mi_indices === nothing
        mi_indices = push_interfaces!(scene, materials)
    else
        n_reuse = min(n, length(reuse_mi_indices))
        mi_indices = Vector{UInt32}(undef, n)
        mi_indices[1:n_reuse] .= view(reuse_mi_indices, 1:n_reuse)
        update_materials!(scene, view(mi_indices, 1:n_reuse), view(materials, 1:n_reuse))
        mi_indices[(n_reuse+1):n] .= push_interfaces!(scene, view(materials, (n_reuse+1):n))
    end

    # Bake a neutral per-face metadata: `medium_interface_idx = 0` marks
    # "inherit from instance override".  `arealight_id = 0` —
    # no per-face area lights (we already rejected emissive materials).
    gb_faces = GeometryBasics.faces(mesh)
    n_faces = length(gb_faces)
    face_meta = [TriangleMeta(UInt32(0), UInt32(i), UInt32(0)) for i in 1:n_faces]
    mesh_with_meta = GeometryBasics.mesh(mesh; face_meta=GeometryBasics.per_face(face_meta, mesh))

    accel_handle = push!(scene.accel, mesh_with_meta, collect(transforms);
                         instance_ids=mi_indices)
    # One SceneHandle per instance, all sharing the same accel handle.
    return [SceneHandle(scene, mi_indices[i], accel_handle) for i in eachindex(mi_indices)]
end

"""
    push_interfaces!(scene, materials) -> Vector{UInt32}

`push!(scene, MediumInterface(m))` for every `m` in `materials`, as one batch.

One material object is one interface, however many instances name it (a
meshscatter in one colour). The interfaces already in the scene are looked up
in one host copy of `scene.media_interfaces` and the new ones appended in one
`append!`. Done one at a time, each push ran a device `findfirst` over the
whole array and grew it by one element, which is quadratic in the instance
count: 20k instances held 2.7 GB of GPU memory, and 100k could not be built.
"""
function push_interfaces!(scene::Scene, materials::AbstractVector{<:Material})
    keys = IdDict{Material, MediumInterfaceIdx}()
    wanted = map(m -> get!(() -> interface_key(scene, MediumInterface(m)), keys, m), materials)
    # The first index of an equal interface, as `findfirst` gave.
    index = Dict{MediumInterfaceIdx, UInt32}()
    for (i, mi) in enumerate(Array(scene.media_interfaces))
        get!(index, mi, UInt32(i))
    end
    n0 = length(scene.media_interfaces)
    fresh = MediumInterfaceIdx[]
    idx = map(wanted) do mi
        get!(index, mi) do
            push!(fresh, mi)
            UInt32(n0 + length(fresh))
        end
    end
    isempty(fresh) || append!(scene.media_interfaces, fresh)
    notify_scene_changed(scene)
    return idx
end

"""
    update_materials!(scene, indices, materials)

`update_material!(scene, indices[i], materials[i])` for every `i`, reading
`scene.media_interfaces` once instead of once per instance.
"""
function update_materials!(scene::Scene, indices::AbstractVector{UInt32}, materials::AbstractVector{<:Material})
    isempty(indices) && return
    mis = Array(scene.media_interfaces)
    # Instances sharing one interface (one material for all) update it once.
    done = Dict{UInt32, Material}()
    for (idx, m) in zip(indices, materials)
        get(done, idx, nothing) === m && continue
        update_material!(scene, mis[idx], m)
        done[idx] = m
    end
    return
end

# Per-face materials (for MetaMesh with multiple materials)
function Base.push!(scene::Scene, mesh::GeometryBasics.Mesh, materials::AbstractVector{<:Material};
                    transform::Mat4f=Mat4f(I))
    # Deduplicate materials via cache (push! already deduplicates at scene level).
    # Each push! just marks its MultiTypeSet dirty; the next `get_static` read
    # collapses the whole batch into one rebuild.
    mat_cache = Dict{UInt64, UInt32}()
    mat_indices = map(materials) do m
        get!(mat_cache, objectid(m)) do
            push!(scene, m)
        end
    end
    face_meta = build_face_meta(scene, mesh, mat_indices, materials)
    mesh_with_meta = GeometryBasics.mesh(mesh; face_meta=GeometryBasics.per_face(face_meta, mesh))
    handle = push!(scene.accel, mesh_with_meta, transform)
    # Return SceneHandle with first material index (for compatibility)
    return SceneHandle(scene, first(mat_indices), handle, mesh_area_lights(face_meta))
end

mesh_area_lights(face_meta) = unique([arealight_key(m.arealight_id) for m in face_meta if m.arealight_id != 0])

"""
    update_material!(scene, handle::SceneHandle, material)

Update a retained mesh's surface/media and its existing face emitters. Sampled
emission reuses the lights' texture slots; shared textures are uploaded once.
The geometry and acceleration structures are unchanged. Adding an emitter to a
mesh without emitter slots requires rebuilding that mesh.
"""
function update_material!(scene::Scene, handle::SceneHandle, material::Material)
    emission = get_emission_info(material)
    if isempty(handle.area_lights)
        emission === nothing || error("adding mesh emission requires rebuilding its face emitters")
        return update_material!(scene, handle.interface, material)
    end
    lights = map(handle.area_lights) do key
        old = scene.lights[key]
        Le = emission === nothing ? old.Le :
            emission.Le isa Texture && !emission.Le.isconst ? emission.Le :
            face_emission!(IdDict(), scene.lights, emission.Le)
        DiffuseAreaLight(old.vertices, old.normal, old.area, old.uv, Le,
            emission === nothing ? 0f0 : emission.scale,
            emission === nothing ? old.two_sided : emission.two_sided)
    end
    update_face_area_lights!(scene, handle.area_lights, lights)
    update_material!(scene, handle.interface, material)
    notify_lights_changed(scene)
    return nothing
end

"""
    Raycore.set_visible!(scene, handles, visible::Bool, materials)

Hide or show pushed meshes: `handles` (a `SceneHandle` or several) with their
`materials`. No ray hits a hidden mesh and its face lights emit nothing; its
geometry, material slot and light slots stay where they are, so showing it again
rebuilds nothing. A mesh's material is what its face lights get their emission
back from when it is shown; only a mesh with face lights reads it.

Instances that share one geometry handle (a `meshscatter`) are hidden once.
"""
function Raycore.set_visible!(scene::Scene, handles::AbstractVector{SceneHandle}, visible::Bool, materials)
    for geometry in unique(h.geometry for h in handles)
        Raycore.set_visible!(scene.accel, geometry, visible) ||
            throw(ArgumentError("set_visible!: a mesh is not in the scene's acceleration structure"))
    end
    for (handle, material) in zip(handles, materials)
        isempty(handle.area_lights) && continue
        visible ? update_material!(scene, handle, material) :
                  disable_face_area_lights!(scene, handle.area_lights)
    end
    notify_scene_changed(scene)
    return nothing
end

Raycore.set_visible!(scene::Scene, handle::SceneHandle, visible::Bool, material) =
    Raycore.set_visible!(scene, [handle], visible, (material,))

"""
    delete!(scene, handle::SceneHandle) -> Bool

Take a pushed mesh out of the scene: its geometry, and the light of its emitting
faces. Deleting only the geometry left the lights in the scene's light set, so a
deleted lamp went on lighting the room. The light slots are switched off rather
than freed; a rebuild that is handed them back (`area_lights`) reuses them.
Returns whether the geometry was still there: instances sharing one (a
`meshscatter`) are taken out by the first of their handles, and each handle's
lights go off regardless.
"""
function Base.delete!(scene::Scene, handle::SceneHandle)
    deleted = delete!(scene.accel, handle.geometry)
    disable_face_area_lights!(scene, handle.area_lights)
    notify_scene_changed(scene)
    return deleted
end

# ============================================================================
# Emission dispatch — extract emission info from material
# ============================================================================

get_emission_info(::Material) = nothing
get_emission_info(m::Emissive) = (Le=m.Le, scale=m.scale, two_sided=m.two_sided)
function get_emission_info(m::MediumInterface)
    # Check emission field first (explicitly attached emission info)
    !isnothing(m.emission) && return m.emission
    # Fall through to inner material
    return get_emission_info(m.material)
end

# The emitted radiance a face's light stores.
#
# An image texture goes into the lights set ONCE, and every face's light holds
# the same `TextureRef`, so `arealight_Le` looks the image up at the hit's uv
# (pbrt-v4's image area light). It used to be sampled at each face centroid,
# which turned a two-triangle emissive quad into two flat colours. `refs` maps
# each texture to its stored ref, so the per-face path does not upload one copy
# per face either.
function face_emission!(refs::IdDict, lights, Le::Texture)
    Le.isconst && return Le.constval
    return get!(() -> Raycore.maybe_convert_field(lights, Le), refs, Le)
end
# Emissive stores Le as a handle; area lights are always specified as constants
# in pbrt (`AreaLightSource "diffuse" "rgb L"`), so `const_spectrum` errors
# loudly rather than silently registering a black light if that ever changes.
face_emission!(refs::IdDict, lights, Le::TexHandle) = const_spectrum(Le)
face_emission!(refs::IdDict, lights, Le) = Le

# A constant that emits nothing registers no light. An image might be dark at
# one face and bright at the next, so every face of it keeps its light.
emits_nothing(Le::RGBSpectrum) = luminance(Le) < 1f-4
emits_nothing(Le) = false

# ============================================================================
# build_face_meta — constructs TriangleMeta per face, registers area lights
# ============================================================================

# Single material
function build_face_meta(scene, mesh, mat_idx::UInt32, material::Material; area_lights=SetKey[])
    gb_faces = GeometryBasics.faces(mesh)
    n_faces = length(gb_faces)
    face_meta = Vector{TriangleMeta}(undef, n_faces)

    emission = get_emission_info(material)

    if isnothing(emission)
        disable_face_area_lights!(scene, area_lights)
        for i in 1:n_faces
            face_meta[i] = TriangleMeta(mat_idx, UInt32(i), UInt32(0))
        end
    else
        register_face_area_lights!(scene, mesh, face_meta, mat_idx, emission; area_lights)
    end
    return face_meta
end

# Per-face materials
function build_face_meta(scene, mesh, mat_indices::AbstractVector{UInt32},
                         materials::AbstractVector{<:Material})
    gb_faces = GeometryBasics.faces(mesh)
    n_faces = length(gb_faces)
    face_meta = Vector{TriangleMeta}(undef, n_faces)

    has_any_emission = any(m -> !isnothing(get_emission_info(m)), materials)

    if !has_any_emission
        for i in 1:n_faces
            face_meta[i] = TriangleMeta(mat_indices[i], UInt32(i), UInt32(0))
        end
    else
        register_face_area_lights!(scene, mesh, face_meta, mat_indices, materials)
    end
    return face_meta
end

# ============================================================================
# register_face_area_lights! — creates DiffuseAreaLight per emissive face
# ============================================================================
#
# The lights are collected on the host and appended to `scene.lights` in ONE
# call. `push!` on a `MultiTypeSet` resizes the GPU slot and writes one element
# per call, so pushing per face cost a `vkAllocateMemory`/`vkFreeMemory` pair
# and a host→device copy per face — ~150 s for a mesh whose emissive surface is
# a tessellated sphere (261 120 faces), which was ~95 % of the time to build
# `RayDemo/Materials/materials.pbrt`.

"""Append the collected `lights`, then stamp each emissive face's `TriangleMeta`
with its light's id (`pack_arealight` of the key `Raycore.append!` returns for
it). `emissive_faces[k]` is the face that produced the k-th light.

`face_material` is the material index for a face: one value shared by the whole
mesh, or one per face."""
face_material(mat_idx::UInt32, ::Int) = mat_idx
face_material(mat_indices::AbstractVector{UInt32}, face_i::Int) = mat_indices[face_i]

function update_face_area_lights!(scene, keys, lights)
    uploaded = Dict{Any, Any}()
    for (key, light) in zip(keys, lights)
        old = scene.lights[key]
        texturekey = old.Le isa Raycore.TextureRef ? (typeof(old.Le), old.Le.idx) : nothing
        Le = texturekey === nothing ? light.Le : get(uploaded, texturekey, light.Le)
        replacement = DiffuseAreaLight(light.vertices, light.normal, light.area, light.uv,
            Le, light.scale, light.two_sided)
        Raycore.update!(scene.lights, key, replacement)
        texturekey === nothing || (uploaded[texturekey] = scene.lights[key].Le)
    end
    return nothing
end

function disable_face_area_lights!(scene, keys)
    for key in keys
        old = scene.lights[key]
        Raycore.update!(scene.lights, key, DiffuseAreaLight(old.vertices, old.normal,
            old.area, old.uv, old.Le, 0f0, old.two_sided))
    end
    isempty(keys) || notify_lights_changed(scene)
    return nothing
end

function flush_face_area_lights!(scene, face_meta, lights, emissive_faces, mat; area_lights=SetKey[])
    if !isempty(area_lights)
        if length(area_lights) == length(lights)
            update_face_area_lights!(scene, area_lights, lights)
            for (k, face_i) in pairs(emissive_faces)
                face_meta[face_i] = TriangleMeta(face_material(mat, face_i),
                    UInt32(face_i), pack_arealight(area_lights[k]))
            end
            notify_lights_changed(scene)
            return face_meta
        end
        disable_face_area_lights!(scene, area_lights)
    end
    isempty(lights) && return face_meta
    keys = append!(scene.lights, lights)
    for (k, face_i) in pairs(emissive_faces)
        face_meta[face_i] = TriangleMeta(face_material(mat, face_i),
                                         UInt32(face_i), pack_arealight(keys[k]))
    end
    return face_meta
end

# Single material (all faces share one material + emission)
function register_face_area_lights!(scene, mesh, face_meta, mat_idx::UInt32, emission; area_lights=SetKey[])
    verts = GeometryBasics.coordinates(mesh)
    gb_faces = GeometryBasics.faces(mesh)
    has_uv = hasproperty(mesh, :uv)
    # Every face of this mesh shares `emission`, so the emitted-radiance type is
    # fixed and the staging vector can be concrete.
    Le = !isempty(area_lights) && emission.Le isa Texture && !emission.Le.isconst ?
        emission.Le : face_emission!(IdDict(), scene.lights, emission.Le)
    lights = DiffuseAreaLight{typeof(Le)}[]
    emissive_faces = Int[]

    for (i, face) in enumerate(gb_faces)
        vs = SVector(Point3f(verts[face[1]]), Point3f(verts[face[2]]), Point3f(verts[face[3]]))
        face_uv = if has_uv
            SVector(Point2f(mesh.uv[face[1]]), Point2f(mesh.uv[face[2]]), Point2f(mesh.uv[face[3]]))
        else
            SVector(Point2f(0), Point2f(1, 0), Point2f(1, 1))
        end

        if emits_nothing(Le)
            face_meta[i] = TriangleMeta(mat_idx, UInt32(i), UInt32(0))
            continue
        end

        e1 = Vec3f(vs[2] - vs[1]); e2 = Vec3f(vs[3] - vs[1])
        cross_product = e1 × e2
        twice_area = norm(cross_product)
        if twice_area < 1f-10
            face_meta[i] = TriangleMeta(mat_idx, UInt32(i), UInt32(0))
            continue
        end

        normal = Raycore.Normal3f(cross_product / twice_area)
        tri_area = 0.5f0 * twice_area
        push!(lights, DiffuseAreaLight(vs, normal, tri_area, face_uv, Le,
                                       emission.scale, emission.two_sided))
        push!(emissive_faces, i)
    end

    flush_face_area_lights!(scene, face_meta, lights, emissive_faces, mat_idx; area_lights)
end

# Per-face materials (different materials per face, some may be emissive)
function register_face_area_lights!(scene, mesh, face_meta,
                                    mat_indices::AbstractVector{UInt32},
                                    materials::AbstractVector{<:Material})
    verts = GeometryBasics.coordinates(mesh)
    gb_faces = GeometryBasics.faces(mesh)
    has_uv = hasproperty(mesh, :uv)
    # Faces may carry different materials here, so the emitted-radiance type is
    # not fixed across the mesh; `Raycore.append!` groups by stored type anyway.
    lights = DiffuseAreaLight[]
    emissive_faces = Int[]
    refs = IdDict()

    for (i, face) in enumerate(gb_faces)
        emission = get_emission_info(materials[i])
        if isnothing(emission)
            face_meta[i] = TriangleMeta(mat_indices[i], UInt32(i), UInt32(0))
            continue
        end

        vs = SVector(Point3f(verts[face[1]]), Point3f(verts[face[2]]), Point3f(verts[face[3]]))
        face_uv = if has_uv
            SVector(Point2f(mesh.uv[face[1]]), Point2f(mesh.uv[face[2]]), Point2f(mesh.uv[face[3]]))
        else
            SVector(Point2f(0), Point2f(1, 0), Point2f(1, 1))
        end

        Le = face_emission!(refs, scene.lights, emission.Le)
        if emits_nothing(Le)
            face_meta[i] = TriangleMeta(mat_indices[i], UInt32(i), UInt32(0))
            continue
        end

        e1 = Vec3f(vs[2] - vs[1]); e2 = Vec3f(vs[3] - vs[1])
        cross_product = e1 × e2
        twice_area = norm(cross_product)
        if twice_area < 1f-10
            face_meta[i] = TriangleMeta(mat_indices[i], UInt32(i), UInt32(0))
            continue
        end

        normal = Raycore.Normal3f(cross_product / twice_area)
        tri_area = 0.5f0 * twice_area
        push!(lights, DiffuseAreaLight(vs, normal, tri_area, face_uv, Le,
                                       emission.scale, emission.two_sided))
        push!(emissive_faces, i)
    end

    flush_face_area_lights!(scene, face_meta, lights, emissive_faces, mat_indices)
end
