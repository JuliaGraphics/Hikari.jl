function load_triangle_mesh(
    model_file::String,
    core::Trace.ShapeCore = Trace.ShapeCore(Trace.Transformation(), false),
    assimp_flags::UInt32 = UInt32(0),
)
    scene::Assimp.Scene = Assimp.load(model_file, assimp_flags)
    triangle_meshes = Trace.TriangleMesh[]
    triangles = Trace.Triangle[]
    _node_to_triangle_mesh!(scene.node, core, triangle_meshes, triangles)
    triangle_meshes, triangles
end

"""
"Convert `Assimp.Node` recursively to `TriangleMesh` and its `Triangles`.
Write result in `triangle_meshes` & `triangles` arrays.
"""
function _node_to_triangle_mesh!(
    node::Assimp.Node, core::Trace.ShapeCore,
    triangle_meshes::Vector{Trace.TriangleMesh},
    triangles::Vector{Trace.Triangle},
)
    # TODO load tangents
    for mmesh in node.meshes
        mesh = mmesh.mesh
        m_vertices = mesh.position
        m_normals = convert(Vector{Trace.Normal3f}, mesh.normals)
        m_faces = faces(mesh)
        @real_assert length(eltype(m_faces)) == 3 "Only triangles supported."
        @real_assert length(m_vertices) == length(m_normals) "Number of normals is different from the number of vertices"

        indices = Vector{UInt32}(undef, length(m_faces) * 3)
        fi = 1
        @inbounds for face in m_faces, i in face
            @real_assert i <= length(m_vertices)
            indices[fi] = i + 1 # 1-based indexing
            fi += 1
        end

        triangle_mesh = Trace.TriangleMesh(
            core.object_to_world, length(m_faces), indices,
            length(m_vertices), m_vertices, m_normals,
        )
        append!(triangles, [
            Trace.Triangle(core, triangle_mesh, i)
            for i in UnitRange{UInt32}(0:length(m_faces)-1)
        ])
        push!(triangle_meshes, triangle_mesh)
    end

    for child in node.children
        _node_to_triangle_mesh!(child, core, triangle_meshes, triangles)
    end
end
