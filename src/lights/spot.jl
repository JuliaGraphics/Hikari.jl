struct SpotLight{S<:Spectrum} <: Light
    flags::LightFlags
    light_to_world::Transformation
    world_to_light::Transformation
    position::Point3f
    i::S
    cos_total_width::Float32
    cos_falloff_start::Float32

    function SpotLight(
        light_to_world::Transformation, i::S,
        total_width::Float32, falloff_start::Float32,
    ) where S<:Spectrum
        new{S}(
            LightδPosition, light_to_world, inv(light_to_world),
            light_to_world(Point3f(0f0)), i,
            cos(deg2rad(total_width)), cos(deg2rad(falloff_start)),
        )
    end
end

"""
    SpotLight(position, target, intensity, cone_angle, falloff_angle)

Convenience constructor for SpotLight that takes position and target points.

# Arguments
- `position::Point3f`: World-space position of the spotlight
- `target::Point3f`: Point the spotlight is aimed at
- `i::Spectrum`: Light intensity/color
- `total_width::Float32`: Total cone angle in degrees
- `falloff_start::Float32`: Angle where intensity falloff begins (degrees)

# Example
```julia
# Spotlight at (0, 5, 0) pointing at origin with 30° cone
light = SpotLight(Point3f(0, 5, 0), Point3f(0, 0, 0), RGBSpectrum(100f0), 30f0, 25f0)
```
"""
function SpotLight(
    position::Point3f, target::Point3f, i::S,
    total_width::Float32, falloff_start::Float32,
) where S<:Spectrum
    light_to_world = _spotlight_transform(position, target)
    SpotLight(light_to_world, i, total_width, falloff_start)
end

"""
Create a transformation that positions a spotlight and orients it to point at a target.
The spotlight points in +Z direction in local space.
"""
function _spotlight_transform(position::Point3f, target::Point3f)
    dir = normalize(Vec3f(target - position))
    # Choose up vector that's not parallel to dir
    up = abs(dir[2]) < 0.99f0 ? Vec3f(0f0, 1f0, 0f0) : Vec3f(1f0, 0f0, 0f0)
    x_axis = normalize(up × dir)
    y_axis = dir × x_axis
    z_axis = dir

    # Rotation matrix: columns are where local axes map to in world space
    rot = Mat4f(
        x_axis[1], x_axis[2], x_axis[3], 0f0,
        y_axis[1], y_axis[2], y_axis[3], 0f0,
        z_axis[1], z_axis[2], z_axis[3], 0f0,
        0f0, 0f0, 0f0, 1f0
    )

    translate(Vec3f(position)) * Transformation(rot, inv(rot))
end

function sample_li(s::SpotLight, ref::Interaction, ::Point2f)
    wi = normalize(Vec3f(s.position - ref.p))
    pdf = 1f0
    visibility = VisibilityTester(
        ref, Interaction(s.position, ref.time, Vec3f(0f0), Normal3f(0f0)),
    )
    radiance = s.i * falloff(s, -wi) / distance_squared(s.position, ref.p)
    radiance, wi, pdf, visibility
end

function falloff(s::SpotLight, w::Vec3f)::Float32
    wl = normalize(s.world_to_light(w))
    cosθ = wl[3]
    cosθ < s.cos_total_width && return 0f0
    cosθ ≥ s.cos_falloff_start && return 1f0
    # Compute falloff inside spotlight cone.
    δ = (cosθ - s.cos_total_width) / (s.cos_falloff_start - s.cos_total_width)
    δ^4
end

@inline function power(s::SpotLight)
    s.i * 2f0 * π * (1f0 - 0.5f0 * (s.cos_falloff_start + s.cos_total_width))
end

function sample_le(
        s::SpotLight, u1::Point2f, ::Point2f, ::Float32,
    )::Tuple{RGBSpectrum,Ray,Normal3f,Float32,Float32}

    w = s.light_to_world(uniform_sample_cone(u1, s.cos_total_width))
    ray = Ray(o=s.position, d=w)
    light_normal = Normal3f(ray.d)
    pdf_pos = 1f0
    pdf_dir = uniform_cone_pdf(s.cos_total_width)
    s.i * falloff(s, ray.d), ray, light_normal, pdf_pos, pdf_dir
end
