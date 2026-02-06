"""
Directional light does not take medium interface, since only reasonable
interface for it is vacuum, otherwise all the light would've been absorbed
by the medium, since the light is infinitely far away.
"""
struct DirectionalLight{S<:Spectrum} <: Light
    """Light-source is positioned at the origin of its light space."""
    light_to_world::Transformation
    world_to_light::Transformation

    i::S
    """
    Scale factor for light intensity (used for photometric normalization).
    In pbrt-v4, this is set to `1 / SpectrumToPhotometric(illuminant)`.
    """
    scale::Float32
    direction::Vec3f

    function DirectionalLight(
        light_to_world::Transformation, l::S, direction::Vec3f, scale::Float32=1f0,
    ) where S<:Spectrum
        new{S}(
            light_to_world, inv(light_to_world),
            l, scale, normalize(light_to_world(direction)),
        )
    end
end

# Directional lights are delta lights (emit along a single direction)
is_δ_light(::DirectionalLight) = true
# Directional lights are infinite (at infinity)
is_infinite_light(::DirectionalLight) = true

"""
    DirectionalLight(rgb::RGB{Float32}, direction; illuminance=nothing)

Create a DirectionalLight from RGB color with automatic spectral conversion and
photometric normalization, matching pbrt-v4's light creation pattern.

# Arguments
- `rgb`: RGB color (intensity encoded in color values)
- `direction`: Direction the light travels (away from source)
- `illuminance`: Optional target illuminance in lux. If specified, overrides the RGB intensity.

# Example
```julia
# Directional light traveling in -Y direction (sunlight from above)
light = DirectionalLight(RGB{Float32}(1f0, 1f0, 1f0), Vec3f(0, -1, 0))
```
"""
function DirectionalLight(rgb::RGB{Float32}, direction::Vec3f; illuminance::Union{Nothing,Float32}=nothing)
    table = get_srgb_table()
    spectrum = rgb_illuminant_spectrum(table, rgb)
    scale = 1f0 / spectrum_to_photometric(spectrum)
    if !isnothing(illuminance)
        scale *= illuminance
    end
    DirectionalLight(Transformation(Mat4f(I)), spectrum, direction, scale)
end

# Accept any RGB type (e.g., RGBf from Makie/Colors)
function DirectionalLight(rgb::RGB, direction::Vec3f; kwargs...)
    DirectionalLight(RGB{Float32}(rgb.r, rgb.g, rgb.b), direction; kwargs...)
end

@propagate_inbounds function sample_li(
        d::DirectionalLight{S}, ref::Interaction, u::Point2f, scene::AbstractScene,
    )::Tuple{S,Vec3f,Float32,VisibilityTester} where S<:Spectrum

    # d.direction is the direction light TRAVELS (away from light source)
    # wi should be the direction TO the light source (opposite of d.direction)
    wi = -d.direction
    # outside_point should be in the direction the light COMES FROM
    outside_point = ref.p .+ wi .* (2 * world_radius(scene))
    tester = VisibilityTester(
        ref, Interaction(outside_point, ref.time, Vec3f(0f0), Normal3f(0f0)),
    )
    d.scale * d.i, wi, 1f0, tester
end

"""
The total power emitted by the directional light is related to the
spatial extent of the scene and equals the amount of power arriving at the
inscribed by bounding sphere disk: `scale * I * π * r^2`.
"""
@propagate_inbounds function power(d::DirectionalLight{S}, scene::AbstractScene)::S where S<:Spectrum
    d.scale * d.i * π * world_radius(scene)^2
end
