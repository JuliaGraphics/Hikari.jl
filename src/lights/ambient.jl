struct AmbientLight{S<:Spectrum} <: Light
    i::S
    scale::Float32
end

# Ambient lights are infinite (emit from all directions)
is_infinite_light(::AmbientLight) = true
is_infinite_light(::Type{<:AmbientLight}) = true
# …and it is VISIBLE along an escaped ray, unlike a directional light.
paints_escaped_rays(::AmbientLight) = true
paints_escaped_rays(::Type{<:AmbientLight}) = true

"""
    AmbientLight(rgb::RGB{Float32})

Create an AmbientLight from RGB color with automatic spectral conversion and
photometric normalization, matching pbrt-v4's UniformInfiniteLight creation:
`scale = 1 / SpectrumToPhotometric(spectrum)`.

# Example
```julia
light = AmbientLight(RGB{Float32}(0.1f0, 0.1f0, 0.1f0))
```
"""
function AmbientLight(rgb::RGB{Float32})
    table = get_srgb_table()
    spectrum = rgb_illuminant_spectrum(table, rgb)
    # Matches pbrt-v4: scale /= SpectrumToPhotometric(L[0])
    scale = 1f0 / spectrum_to_photometric(spectrum)
    AmbientLight(spectrum, scale)
end

# Accept any RGB type (e.g., RGBf from Makie/Colors)
AmbientLight(rgb::RGB) = AmbientLight(RGB{Float32}(rgb.r, rgb.g, rgb.b))

# RGBSpectrum constructor: apply photometric normalization (matching pbrt-v4)
AmbientLight(s::RGBSpectrum) = AmbientLight{RGBSpectrum}(s, 1f0 / D65_PHOTOMETRIC)

"""
Compute radiance arriving at `ref.p` interaction point at `ref.time` time
due to the ambient light.

# Args

- `a::AmbientLight`: Ambient light which illuminates the interaction point `ref`.
- `ref::Interaction`: Interaction point for which to compute radiance.
- `u::Point2f`: Sampling point that is ignored for `AmbientLight`,
    since it emits light uniformly.

# Returns

`Tuple{S, Vec3f, Float32, VisibilityTester} where S <: Spectrum`:

    - `S`: Computed radiance.
    - `Vec3f`: Incident direction to the light source `wi`.
    - `Float32`: Probability density for the light sample that was taken.
        For `AmbientLight` it is always `1`.
    - `VisibilityTester`: Initialized visibility tester that holds the
        shadow ray that must be traced to verify that
        there are no occluding objects between the light and reference point.
"""
function sample_li(a::AmbientLight, i::Interaction, ::Point2f, ::AbstractScene)
    pdf = 1.0f0
    radiance = a.scale * a.i
    inew = Interaction()
    radiance, Vec3f(normalize(i.p)), pdf, VisibilityTester(inew, inew)
end
