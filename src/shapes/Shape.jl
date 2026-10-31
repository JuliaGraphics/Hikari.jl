struct ShapeCore
    object_to_world::Transformation
    world_to_object::Transformation
    reverse_orientation::Bool
    transform_swaps_handedness::Bool

    function ShapeCore(
        object_to_world::Transformation, reverse_orientation::Bool,
    )
        new(
            object_to_world, inv(object_to_world), reverse_orientation,
            swaps_handedness(object_to_world),
        )
    end
end

function world_bound(s::AbstractShape)::Bounds3
    s.core.object_to_world(object_bound(s))
end

include("sphere.jl")
include("triangle_mesh.jl")
