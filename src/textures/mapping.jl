abstract type Mapping2D end
abstract type Mapping3D end

"""
Simplest mapping uses `(u, v)` coordinates from the `SurfaceInteraction`
to compute texture coordinates and can be offset and scaled with user-supplied
values in each dimension.
"""
struct UVMapping2D <: Mapping2D
    su::Float32
    sv::Float32
    du::Float32
    dv::Float32
end

"""
Given surface interaction `si` at the shading point,
return `(s, t)` texture coordinates and estimated changes in `(s, t)` w.r.t.
pixel `x` & `y` coordinates.

# Paramters:

- `m::UVMapping2D`: UVMapping with offset & scale parameters.
- `si::SurfaceInteraction`: SurfaceInteraction at the shading point.

# Returns:

    `Tuple{Point2f, Vec2f, Vec2f}`:
        Texture coordinates at the shading point, estimated changes
        in `(s, t)` w.r.t. pixel `x` & `y` coordinates.
"""
function Base.map(m::UVMapping2D, si::SurfaceInteraction)::Tuple{Point2f,Vec2f,Vec2f}
    ∂st∂x = Vec2f(m.su * si.∂u∂x, m.sv * si.∂v∂x)
    ∂st∂y = Vec2f(m.su * si.∂u∂y, m.sv * si.∂v∂y)
    t = Point2f(m.su * si.uv[1] + m.du, m.sv * si.uv[2] + m.dv)
    t, ∂st∂x, ∂st∂y
end


"""
3D mapping that takes world space coordinate of the point
and applies a linear transformation to it. This will often be a transformation
that takes the point back to the primitive's object space.

Because a linear mapping is used, the differential change in texture
coordinates can be found by applying the same transformation to the
partial derivatives of the position.
"""
struct TransformMapping3D <: Mapping3D
    world_to_texture::Transformation
end

function Base.map(m::TransformMapping3D, si::SurfaceInteraction)::Tuple{Point3f,Vec3f,Vec3f}
    ∂p∂x = m.world_to_texture(si.∂p∂x)
    ∂p∂y = m.world_to_texture(si.∂p∂y)
    t = m.world_to_texture(si.core.p)
    t, ∂p∂x, ∂p∂y
end
