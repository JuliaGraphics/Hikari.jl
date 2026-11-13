struct AmbientLight{S<:Spectrum} <: Light
    i::S
    scale::Float32
    """Ray types its light is hidden from (see [`Visibility`](@ref))."""
    hidden::UInt8
end
AmbientLight{S}(i::S, scale::Float32) where S<:Spectrum = AmbientLight{S}(i, scale, 0x00)
AmbientLight(i::S, scale::Float32) where S<:Spectrum = AmbientLight{S}(i, scale)
AmbientLight(l::AmbientLight{S}, hidden::UInt8) where S<:Spectrum = AmbientLight{S}(l.i, l.scale, hidden)

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

