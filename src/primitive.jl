# GeometricPrimitive is no longer needed with new RayCaster API
# Materials are now stored separately in MaterialScene

# Helper function to build MaterialScene from TriangleMesh with materials
function build_material_scene(mesh_material_pairs::Vector{Tuple{TriangleMesh, M}}) where {M<:Material}
    materials = M[]
    meshes = TriangleMesh[]

    for (mesh, material) in mesh_material_pairs
        push!(materials, material)
        push!(meshes, mesh)
    end

    # Build BVH with material indices
    bvh = BVH(meshes)

    # MaterialScene expects a tuple of material vectors
    return MaterialScene(bvh, (materials,))
end

# Compatibility: keep GeometricPrimitive for backward compatibility in basic-scene.jl
struct GeometricPrimitive{M<:Material}
    shape::TriangleMesh
    material::M

    GeometricPrimitive(shape::TriangleMesh, material::M) where {M<:Material} = new{M}(shape, material)
end
