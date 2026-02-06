struct PointLight{S<:Spectrum} <: Light
    """Light-source is positioned at the origin of its light space."""
    light_to_world::Transformation
    world_to_light::Transformation

    i::S
    """
    Scale factor for light intensity (used for photometric normalization).
    In pbrt-v4, this is set to `1 / SpectrumToPhotometric(illuminant)`.
    """
    scale::Float32
    """Position in world space."""
    position::Point3f

    function PointLight(light_to_world::Transformation, i::S, scale::Float32=1f0) where S<:Spectrum
        new{S}(
            light_to_world, inv(light_to_world),
            i, scale, light_to_world(Point3f(0f0)),
        )
    end
end

# Point lights are delta lights (emit from a single point)
is_δ_light(::PointLight) = true

function PointLight(position, i::S, scale::Float32=1f0) where S<:Spectrum
    PointLight(translate(Vec3f(position)), i, scale)
end

"""
    PointLight(rgb::RGB{Float32}, position; power=nothing)
    PointLight(rgb::RGB, position; power=nothing)

Create a PointLight from RGB color with automatic spectral conversion and photometric
normalization, matching pbrt-v4's light creation pattern.

Converts RGB to RGBIlluminantSpectrum and applies photometric normalization:
`scale = 1 / SpectrumToPhotometric(spectrum)` where SpectrumToPhotometric extracts
the D65 illuminant component.

# Arguments
- `rgb`: RGB color (intensity encoded in color values, e.g., RGB(50,50,50) for bright white)
- `position`: World-space position of the light
- `power`: Optional radiant power in Watts. If specified, overrides the RGB intensity.

# Example
```julia
# Equivalent to pbrt-v4's: LightSource "point" "rgb I" [50 50 50]
light = PointLight(RGB{Float32}(50f0, 50f0, 50f0), Vec3f(10, 10, 10))

# Or with Makie's RGBf:
light = PointLight(RGBf(50, 50, 50), Vec3f(10, 10, 10))
```
"""
function PointLight(rgb::RGB{Float32}, position; power::Union{Nothing,Float32}=nothing)
    table = get_srgb_table()
    spectrum = rgb_illuminant_spectrum(table, rgb)
    scale = 1f0 / spectrum_to_photometric(spectrum)
    if !isnothing(power)
        k_e = 4f0 * Float32(π)  # Sphere solid angle
        scale *= power / k_e
    end
    PointLight(translate(Vec3f(position)), spectrum, scale)
end

# Accept any RGB type (e.g., RGBf from Makie/Colors)
function PointLight(rgb::RGB, position; kwargs...)
    PointLight(RGB{Float32}(rgb.r, rgb.g, rgb.b), position; kwargs...)
end

# Legacy: RGBSpectrum constructor (for direct spectral specification without conversion)
function PointLight(i::RGBSpectrum, position)
    scale = 1f0 / D65_PHOTOMETRIC
    PointLight(translate(Vec3f(position)), i, scale)
end

"""
Compute radiance arriving at `ref.p` interaction point at `ref.time` time
due to that light, assuming there are no occluding objects between them.

# Args

- `p::PointLight`: Light which illuminates the interaction point `ref`.
- `ref::Interaction`: Interaction point for which to compute radiance.
- `u::Point2f`: Sampling point that is ignored for `PointLight`,
    since it has no area.

# Returns

`Tuple{S, Vec3f, Float32, VisibilityTester} where S <: Spectrum`:

    - `S`: Computed radiance.
    - `Vec3f`: Incident direction to the light source `wi`.
    - `Float32`: Probability density for the light sample that was taken.
        For `PointLight` it is always `1`.
    - `VisibilityTester`: Initialized visibility tester that holds the
        shadow ray that must be traced to verify that
        there are no occluding objects between the light and reference point.
"""
function sample_li(p::PointLight, i::Interaction, ::Point2f, ::AbstractScene)
    wi = normalize(Vec3f(p.position - i.p))
    pdf = 1f0
    visibility = VisibilityTester(
        i, Interaction(p.position, i.time, Vec3f(0.0f0), Normal3f(0.0f0)),
    )
    radiance = p.scale * p.i / distance_squared(p.position, i.p)
    radiance, wi, pdf, visibility
end

function sample_le(
    p::PointLight, u1::Point2f, ::Point2f, ::Float32,
)::Tuple{RGBSpectrum,Ray,Normal3f,Float32,Float32}
    ray = Ray(o=p.position, d=uniform_sample_sphere(u1))
    @real_assert norm(ray.d) ≈ 1f0
    light_normal = Normal3f(ray.d)
    pdf_pos = 1f0
    pdf_dir = uniform_sphere_pdf()
    return p.scale * p.i, ray, light_normal, pdf_pos, pdf_dir
end

"""
Total power emitted by the light source over the entire sphere of directions.
"""
@propagate_inbounds function power(p::PointLight)
    4f0 * π * p.scale * p.i
end
