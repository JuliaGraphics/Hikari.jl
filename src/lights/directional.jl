"""
Directional light does not take medium interface, since only reasonable
interface for it is vacuum, otherwise all the light would've been absorbed
by the medium, since the light is infinitely far away.
"""
struct DirectionalLight{S<:Spectrum} <: Light
    """
    Since directional lights represent singularities that emit light
    along a single direction, flag is set to `LightδDirection`.
    """
    flags::LightFlags
    """
    Light-source is positioned at the origin of its light space.
    """
    light_to_world::Transformation
    world_to_light::Transformation

    i::S
    direction::Vec3f

    function DirectionalLight(
        light_to_world::Transformation, l::S, direction::Vec3f,
    ) where S<:Spectrum
        new{S}(
            LightδDirection, light_to_world, inv(light_to_world),
            l, normalize(light_to_world(direction)),
        )
    end
end

function sample_li(
        d::DirectionalLight{S}, ref::Interaction, u::Point2f, scene::AbstractScene,
    )::Tuple{S,Vec3f,Float32,VisibilityTester} where S<:Spectrum

    # d.direction is the direction light TRAVELS (away from light source)
    # wi should be the direction TO the light source (opposite of d.direction)
    wi = -d.direction
    # outside_point should be in the direction the light COMES FROM
    outside_point = ref.p .+ wi .* (2 * scene.world_radius)
    tester = VisibilityTester(
        ref, Interaction(outside_point, ref.time, Vec3f(0f0), Normal3f(0f0)),
    )
    d.i, wi, 1f0, tester
end

"""
The total power emitted by the directional light is related to the
spatial extent of the scene and equals the amount of power arriving at the
inscribed by bounding sphere disk: `I * π * r^2`.
"""
@inline function power(d::DirectionalLight{S}, scene::AbstractScene)::S where S<:Spectrum
    d.i * π * scene.world_radius^2
end
