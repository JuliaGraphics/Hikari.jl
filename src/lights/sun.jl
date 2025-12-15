"""
SunLight - A directional light with additional parameters for volumetric cloud rendering.

Extends DirectionalLight with angular diameter and corona falloff parameters
useful for rendering sun disks and atmospheric scattering in clouds.
"""
struct SunLight{S<:Spectrum} <: Light
    """
    Since sun lights represent singularities that emit light
    along a single direction, flag is set to `LightδDirection`.
    """
    flags::LightFlags
    light_to_world::Transformation
    world_to_light::Transformation

    i::S
    direction::Vec3f

    # Cloud rendering parameters
    angular_diameter::Float32  # Angular size of sun disk in radians (~0.00933 for real sun)
    corona_falloff::Float32    # How quickly corona fades around sun disk

    function SunLight(
        light_to_world::Transformation, l::S, direction::Vec3f;
        angular_diameter::Float32 = 0.00933f0,
        corona_falloff::Float32 = 8.0f0,
    ) where S<:Spectrum
        new{S}(
            LightδDirection, light_to_world, inv(light_to_world),
            l, normalize(light_to_world(direction)),
            angular_diameter, corona_falloff,
        )
    end
end

# Convenience constructor without transformation
function SunLight(l::S, direction::Vec3f; kwargs...) where S<:Spectrum
    SunLight(Transformation(Mat4f(I)), l, direction; kwargs...)
end

function sample_li(
        s::SunLight{S}, ref::Interaction, u::Point2f, scene::Scene,
    )::Tuple{S,Vec3f,Float32,VisibilityTester} where S<:Spectrum

    # s.direction is the direction light TRAVELS (away from light source)
    # wi should be the direction TO the light source (opposite of s.direction)
    wi = -s.direction
    outside_point = ref.p .+ wi .* (2 * scene.world_radius)
    tester = VisibilityTester(
        ref, Interaction(outside_point, ref.time, Vec3f(0f0), Normal3f(0f0)),
    )
    s.i, wi, 1f0, tester
end

@inline function power(s::SunLight{S}, scene::Scene)::S where S<:Spectrum
    s.i * π * scene.world_radius^2
end
