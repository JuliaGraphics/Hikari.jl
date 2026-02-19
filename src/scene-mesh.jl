# ============================================================================
# GB.Mesh-based push! API — Hikari handles material resolution + area lights
# ============================================================================
# This file is included after all materials and lights are defined.

using LinearAlgebra: I, norm

# Single material for entire mesh
function Base.push!(scene::Scene, mesh::GeometryBasics.Mesh, material::Material;
                    transform::Mat4f=Mat4f(I))
    mat_idx = push!(scene, material)
    face_meta = build_face_meta(scene, mesh, mat_idx, material)
    mesh_with_meta = GeometryBasics.mesh(mesh; face_meta=GeometryBasics.per_face(face_meta, mesh))
    handle = push!(scene.accel, mesh_with_meta, transform)
    return SceneHandle(scene, mat_idx, handle)
end

# Per-face materials (for MetaMesh with multiple materials)
function Base.push!(scene::Scene, mesh::GeometryBasics.Mesh, materials::AbstractVector{<:Material};
                    transform::Mat4f=Mat4f(I))
    # Deduplicate materials via cache (push! already deduplicates at scene level)
    mat_cache = Dict{UInt64, UInt32}()  # objectid → scene index
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

# Single material (all faces share one material + emission)
function register_face_area_lights!(scene, mesh, face_meta, mat_idx::UInt32, emission)
    verts = GeometryBasics.coordinates(mesh)
    gb_faces = GeometryBasics.faces(mesh)
    has_uv = hasproperty(mesh, :uv)

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
        light = DiffuseAreaLight(vs, normal, tri_area, face_uv, Le, emission.scale, emission.two_sided)
        push!(scene.lights, light; rebuild=false)
        face_meta[i] = TriangleMeta(mat_idx, UInt32(i), UInt32(length(scene.lights)))
    end

    Raycore.rebuild_static!(scene.lights)
end

# Per-face materials (different materials per face, some may be emissive)
function register_face_area_lights!(scene, mesh, face_meta,
                                    mat_indices::AbstractVector{UInt32},
                                    materials::AbstractVector{<:Material})
    verts = GeometryBasics.coordinates(mesh)
    gb_faces = GeometryBasics.faces(mesh)
    has_uv = hasproperty(mesh, :uv)

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
        light = DiffuseAreaLight(vs, normal, tri_area, face_uv, Le, emission.scale, emission.two_sided)
        push!(scene.lights, light; rebuild=false)
        face_meta[i] = TriangleMeta(mat_indices[i], UInt32(i), UInt32(length(scene.lights)))
    end

    Raycore.rebuild_static!(scene.lights)
end
