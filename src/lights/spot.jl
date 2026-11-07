struct SpotLight{S<:Spectrum} <: Light
    light_to_world::Transformation
    world_to_light::Transformation
    position::Point3f
    i::S
    """
    Scale factor for light intensity (used for photometric normalization).
    In pbrt-v4, this is set to `1 / SpectrumToPhotometric(illuminant)`.
    """
    scale::Float32
    cos_total_width::Float32
    cos_falloff_start::Float32

    function SpotLight(
        light_to_world::Transformation, i::S,
        total_width::Float32, falloff_start::Float32,
        scale::Float32=1f0,
    ) where S<:Spectrum
        new{S}(
            light_to_world, inv(light_to_world),
            light_to_world(Point3f(0f0)), i, scale,
            cos(deg2rad(total_width)), cos(deg2rad(falloff_start)),
        )
    end
end

# Spot lights are delta lights (emit from a single point)
is_δ_light(::Type{<:SpotLight}) = true

"""
    SpotLight(position, target, intensity, cone_angle, falloff_angle; scale=1f0)

Convenience constructor for SpotLight that takes position and target points.

# Arguments
- `position::Point3f`: World-space position of the spotlight
- `target::Point3f`: Point the spotlight is aimed at
- `i::Spectrum`: Light intensity/color
- `total_width::Float32`: Total cone angle in degrees
- `falloff_start::Float32`: Angle where intensity falloff begins (degrees)
- `scale::Float32`: Photometric scale factor (default 1.0)

# Example
```julia
# Spotlight at (0, 5, 0) pointing at origin with 30° cone
light = SpotLight(Point3f(0, 5, 0), Point3f(0, 0, 0), RGBSpectrum(100f0), 30f0, 25f0)
```
"""
function SpotLight(
    position::Point3f, target::Point3f, i::S,
    total_width::Float32, falloff_start::Float32,
    scale::Float32=1f0,
) where S<:Spectrum
    light_to_world = spotlight_transform(position, target)
    SpotLight(light_to_world, i, total_width, falloff_start, scale)
end

# RGBSpectrum: apply photometric normalization (matching pbrt-v4)
function SpotLight(
    position::Point3f, target::Point3f, i::RGBSpectrum,
    total_width::Float32, falloff_start::Float32,
)
    scale = 1f0 / D65_PHOTOMETRIC
    light_to_world = spotlight_transform(position, target)
    SpotLight(light_to_world, i, total_width, falloff_start, scale)
end

"""
    SpotLight(rgb::RGB{Float32}, position, target, total_width, falloff_start; power=nothing)

Create a SpotLight from RGB color with automatic spectral conversion and photometric
normalization, matching pbrt-v4's light creation pattern.

# Arguments
- `rgb`: RGB color (intensity encoded in color values)
- `position::Point3f`: World-space position of the spotlight
- `target::Point3f`: Point the spotlight is aimed at
- `total_width::Float32`: Total cone angle in degrees
- `falloff_start::Float32`: Angle where intensity falloff begins (degrees)
- `power`: Optional radiant power in Watts. If specified, overrides the RGB intensity.

# Example
```julia
# Spotlight with RGB color
light = SpotLight(RGB{Float32}(100f0, 100f0, 100f0), Point3f(0, 5, 0), Point3f(0, 0, 0), 30f0, 25f0)
```
"""
function SpotLight(
    rgb::RGB{Float32}, position::Point3f, target::Point3f,
    total_width::Float32, falloff_start::Float32;
    power::Union{Nothing,Float32}=nothing
)
    table = get_srgb_table()
    spectrum = rgb_illuminant_spectrum(table, rgb)
    scale = 1f0 / spectrum_to_photometric(spectrum)
    if !isnothing(power)
        cos_falloff_end = cos(deg2rad(total_width))
        cos_falloff_start_val = cos(deg2rad(falloff_start))
        k_e = 2f0 * Float32(π) * ((1f0 - cos_falloff_start_val) + (cos_falloff_start_val - cos_falloff_end) / 2f0)
        scale *= power / k_e
    end
    light_to_world = spotlight_transform(position, target)
    SpotLight(light_to_world, spectrum, total_width, falloff_start, scale)
end

# Accept any RGB type (e.g., RGBf from Makie/Colors)
function SpotLight(rgb::RGB, position::Point3f, target::Point3f, total_width::Float32, falloff_start::Float32; kwargs...)
    SpotLight(RGB{Float32}(rgb.r, rgb.g, rgb.b), position, target, total_width, falloff_start; kwargs...)
end

"""
Create a transformation that positions a spotlight and orients it to point at a target.
The spotlight points in +Z direction in local space.
"""
function spotlight_transform(position::Point3f, target::Point3f)
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

function sample_li(s::SpotLight, ref::Interaction, ::Point2f, ::AbstractScene)
    wi = normalize(Vec3f(s.position - ref.p))
    pdf = 1f0
    visibility = VisibilityTester(
        ref, Interaction(s.position, ref.time, Vec3f(0f0), Normal3f(0f0)),
    )
    # Use scale * i (matching pbrt-v4's SpotLight::SampleLi)
    radiance = s.scale * s.i * falloff(s, -wi) / distance_squared(s.position, ref.p)
    radiance, wi, pdf, visibility
end

function falloff(s::SpotLight, w::Vec3f)::Float32
    wl = normalize(s.world_to_light(w))
    cosθ = wl[3]
    cosθ < s.cos_total_width && return 0f0
    cosθ ≥ s.cos_falloff_start && return 1f0
    # SmoothStep — matches pbrt-v4 util/math.h:268 SmoothStep(x, a, b) = t²(3 - 2t)
    δ = (cosθ - s.cos_total_width) / (s.cos_falloff_start - s.cos_total_width)
    δ * δ * (3f0 - 2f0 * δ)
end

