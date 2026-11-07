# ============================================================================
# Emission
# ============================================================================

"""
    Emission(; color = (1, 1, 1), strength = 1, two_sided = false)

Blender's Emission: a surface that emits radiance `strength` times `color`, which takes a colour or
an image. A mesh of it is a light, each triangle an area light sampled for direct lighting, as pbrt's
`AreaLightSource "diffuse"`; with `two_sided` it emits from its back as well. As a
`MediumInterface`'s `emission` it makes a lit surface of any material, and in a `Mix` or `Add` it
is Blender's Emission shader, which the paths that hit it find.

Its radiance is normalized as pbrt-v4 normalizes an area light's: `strength` 1 of a white `color`
is one nit.
"""
struct Emission{ColorT, StrengthT} <: Material
    color::ColorT
    strength::StrengthT
    two_sided::Bool
end

Emission(; color = (1f0, 1f0, 1f0), strength = 1f0, two_sided::Bool = false) = Emission(color, strength, two_sided)

is_emissive(::Emission) = true

"""
    get_emission_info(material) -> (; Le, scale, two_sided) or nothing

What a material emits, as its faces' area lights take it: the radiance `Le` (an image stays one),
the photometric `scale` pbrt-v4 gives an area light, and whether it emits from both sides.
"""
get_emission_info(::Material) = nothing
get_emission_info(e::Emission) =
    (Le = emitted(e.color), scale = Float32(e.strength) / D65_PHOTOMETRIC, two_sided = e.two_sided)
get_emission_info(m::MediumInterface) =
    isnothing(m.emission) ? get_emission_info(m.material) : get_emission_info(m.emission)

emitted(t::Texture) = t
emitted(x) = const_spectrum(TexHandle(x))
