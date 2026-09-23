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
    return SceneHandle(scene, mat_idx, handle)
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
                    material::Material; transform::Mat4f=Mat4f(I))
    face_meta = build_face_meta(scene, mesh, mat_idx, material)
    mesh_with_meta = GeometryBasics.mesh(mesh; face_meta=GeometryBasics.per_face(face_meta, mesh))
    # The interface already names the stored material and therefore its SBT
    # type slot. Converting the raw material just to discover that type stores
    # all its textures again, leaking a full feather map on every mesh rebuild.
    mi = @allowscalar scene.media_interfaces[mat_idx]
    sbt_offset = Raycore.is_valid(mi.material) ? mi.material.type_idx - UInt32(1) : UInt32(0)
    handle = push!(scene.accel, mesh_with_meta, transform; sbt_offset=sbt_offset)
    return SceneHandle(scene, mat_idx, handle)
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
        # Push all materials; MultiTypeSet's dirty flag absorbs the batch and
        # the next `get_static` read triggers a single rebuild.
        mi_indices = Vector{UInt32}(undef, n)
        for i in 1:n
            mi_indices[i] = push!(scene, MediumInterface(materials[i]))
        end
    else
        n_reuse = min(n, length(reuse_mi_indices))
        mi_indices = Vector{UInt32}(undef, n)
        for i in 1:n_reuse
            mi_indices[i] = reuse_mi_indices[i]
            Hikari.update_material!(scene, mi_indices[i], materials[i])
        end
        for i in (n_reuse+1):n
            mi_indices[i] = push!(scene, MediumInterface(materials[i]))
        end
    end

    # Bake a neutral per-face metadata: `medium_interface_idx = 0` marks
    # "inherit from instance override".  `arealight_flat_idx = 0` —
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
    return SceneHandle(scene, first(mat_indices), handle)
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

# Evaluate emission Le: textured or constant
evaluate_face_emission(Le::Texture, face_uv) = evaluate_texture(Le, Point2f((Vec2f(face_uv[1]) + Vec2f(face_uv[2]) + Vec2f(face_uv[3])) / 3f0))
# Emissive stores Le as a handle; area lights are always specified as constants
# in pbrt (`AreaLightSource "diffuse" "rgb L"`), so `const_spectrum` errors
# loudly rather than silently registering a black light if that ever changes.
evaluate_face_emission(Le::TexHandle, face_uv) = const_spectrum(Le)
evaluate_face_emission(Le, face_uv) = Le

# ============================================================================
# build_face_meta — constructs TriangleMeta per face, registers area lights
# ============================================================================

# Single material
function build_face_meta(scene, mesh, mat_idx::UInt32, material::Material)
    gb_faces = GeometryBasics.faces(mesh)
    n_faces = length(gb_faces)
    face_meta = Vector{TriangleMeta}(undef, n_faces)

    emission = get_emission_info(material)

    if isnothing(emission)
        for i in 1:n_faces
            face_meta[i] = TriangleMeta(mat_idx, UInt32(i), UInt32(0))
        end
    else
        register_face_area_lights!(scene, mesh, face_meta, mat_idx, emission)
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
with the light's flat index. `emissive_faces[k]` is the face that produced the
k-th light, and the flat index is the set's length before the append plus k —
`Raycore.append!` assigns exactly that order.

`face_material` is the material index for a face: one value shared by the whole
mesh, or one per face."""
face_material(mat_idx::UInt32, ::Int) = mat_idx
face_material(mat_indices::AbstractVector{UInt32}, face_i::Int) = mat_indices[face_i]

function flush_face_area_lights!(scene, face_meta, lights, emissive_faces, mat)
    isempty(lights) && return face_meta
    base = length(scene.lights)
    append!(scene.lights, lights)
    for (k, face_i) in pairs(emissive_faces)
        face_meta[face_i] = TriangleMeta(face_material(mat, face_i),
                                         UInt32(face_i), UInt32(base + k))
    end
    return face_meta
end

# Single material (all faces share one material + emission)
function register_face_area_lights!(scene, mesh, face_meta, mat_idx::UInt32, emission)
    verts = GeometryBasics.coordinates(mesh)
    gb_faces = GeometryBasics.faces(mesh)
    has_uv = hasproperty(mesh, :uv)
    # Every face of this mesh shares `emission`, so the emitted-radiance type is
    # fixed and the staging vector can be concrete.
    Le_type = typeof(evaluate_face_emission(emission.Le,
                                            SVector(Point2f(0f0), Point2f(1f0, 0f0), Point2f(1f0, 1f0))))
    lights = DiffuseAreaLight{Le_type}[]
    emissive_faces = Int[]

    for (i, face) in enumerate(gb_faces)
        vs = SVector(Point3f(verts[face[1]]), Point3f(verts[face[2]]), Point3f(verts[face[3]]))
        face_uv = if has_uv
            SVector(Point2f(mesh.uv[face[1]]), Point2f(mesh.uv[face[2]]), Point2f(mesh.uv[face[3]]))
        else
            SVector(Point2f(0), Point2f(1, 0), Point2f(1, 1))
        end

        Le = evaluate_face_emission(emission.Le, face_uv)
        if luminance(Le) < 1f-4
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

    flush_face_area_lights!(scene, face_meta, lights, emissive_faces, mat_idx)
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

        Le = evaluate_face_emission(emission.Le, face_uv)
        if luminance(Le) < 1f-4
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
