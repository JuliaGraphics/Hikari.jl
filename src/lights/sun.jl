"""
SunLight - A directional light with additional parameters for volumetric cloud rendering.

Extends DirectionalLight with angular diameter and corona falloff parameters
useful for rendering sun disks and atmospheric scattering in clouds.
"""
struct SunLight{S<:Spectrum} <: Light
    light_to_world::Transformation
    world_to_light::Transformation

    i::S
    """
    Scale factor for light intensity (used for photometric normalization).
    In pbrt-v4, this is set to `1 / SpectrumToPhotometric(illuminant)`.
    """
    scale::Float32
    direction::Vec3f

    # Cloud rendering parameters
    angular_diameter::Float32  # Angular size of sun disk in radians (~0.00933 for real sun)
    corona_falloff::Float32    # How quickly corona fades around sun disk

    function SunLight(
        light_to_world::Transformation, l::S, direction::Vec3f, scale::Float32=1f0;
        angular_diameter::Float32 = 0.00933f0,
        corona_falloff::Float32 = 8.0f0,
    ) where S<:Spectrum
        new{S}(
            light_to_world, inv(light_to_world),
            l, scale, normalize(light_to_world(direction)),
            angular_diameter, corona_falloff,
        )
    end
end

# Sun lights are delta lights (emit along a single direction)
is_δ_light(::SunLight) = true
# Sun lights are infinite (at infinity)
is_infinite_light(::SunLight) = true

# Convenience constructor without transformation (for direct spectrum input)
function SunLight(l::S, direction::Vec3f, scale::Float32=1f0; kwargs...) where S<:Spectrum
    SunLight(Transformation(Mat4f(I)), l, direction, scale; kwargs...)
end

"""
    SunLight(rgb::RGB{Float32}, direction; angular_diameter=0.00933f0, corona_falloff=8f0)

Create a SunLight from RGB color with automatic spectral conversion and photometric
normalization, matching pbrt-v4's light creation pattern.

# Arguments
- `rgb`: RGB color (intensity encoded in color values)
- `direction`: Direction the light travels (away from source)
- `angular_diameter`: Angular size of sun disk in radians (default ~0.53°, real sun)
- `corona_falloff`: How quickly corona fades around sun disk

# Example
```julia
# Sun light traveling in -Y direction (sunlight from above)
light = SunLight(RGB{Float32}(1f0, 1f0, 0.9f0), Vec3f(0, -1, 0))
```
"""
function SunLight(rgb::RGB{Float32}, direction::Vec3f; kwargs...)
    table = get_srgb_table()
    spectrum = rgb_illuminant_spectrum(table, rgb)
    scale = 1f0 / spectrum_to_photometric(spectrum)
    SunLight(Transformation(Mat4f(I)), spectrum, direction, scale; kwargs...)
end

# Accept any RGB type (e.g., RGBf from Makie/Colors)
function SunLight(rgb::RGB, direction::Vec3f; kwargs...)
    SunLight(RGB{Float32}(rgb.r, rgb.g, rgb.b), direction; kwargs...)
end

function sample_li(
        s::SunLight{S}, ref::Interaction, u::Point2f, scene::AbstractScene,
    )::Tuple{S,Vec3f,Float32,VisibilityTester} where S<:Spectrum

    # s.direction is the direction light TRAVELS (away from light source)
    # wi should be the direction TO the light source (opposite of s.direction)
    wi = -s.direction
    outside_point = ref.p .+ wi .* (2 * world_radius(scene))
    tester = VisibilityTester(
        ref, Interaction(outside_point, ref.time, Vec3f(0f0), Normal3f(0f0)),
    )
    s.scale * s.i, wi, 1f0, tester
end

@propagate_inbounds function power(s::SunLight{S}, scene::AbstractScene)::S where S<:Spectrum
    s.scale * s.i * π * world_radius(scene)^2
end
