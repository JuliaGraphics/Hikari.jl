# Emissive - Area light emission data
# Used inside MediumInterface.arealight to add emission to any surface

# ============================================================================
# Emissive Type
# ============================================================================

"""
    Emissive{LeTex}

Emission data for area lights. Always used inside `MediumInterface.arealight`
to add light emission to surfaces. A surface with an arealight both reflects
light (via the MediumInterface's BSDF material) AND emits light.

# Fields
* `Le`: Emitted radiance (color/intensity texture or TextureRef)
* `scale`: Intensity multiplier applied to Le
* `two_sided`: If true, emits from both sides of the surface

# Usage
```julia
# Area light panel (matching pbrt-v4's "rgb L" [3 3 3]):
Emissive(Le=(3, 3, 3), scale=1.0, two_sided=true)
```
"""
# Constants collapse into a `TexHandle`, so every `Emissive(Le=<colour>)` is
# ONE type however the colour was spelled — that is what pbrt area lights use
# (`AreaLightSource "diffuse" "rgb L"`), where a parametric `Le` gives an
# emissive scene one closest-hit shader per spelling.
#
# An image `Texture` stays as-is: `Raycore`'s push-time conversion turns it into
# a `TextureRef` in the scene's store, and per-face emission is baked from it in
# `register_face_area_lights!`. It cannot become a `TexHandle` at construction
# time because a handle needs the store slot, which only exists after the push.
struct Emissive{LeTex} <: Material
    Le::LeTex   # TexHandle for constants, Texture/TextureRef for image emission
    scale::Float32
    two_sided::Bool
end

@inline _emissive_le(t::Texture{T, N}) where {T, N} = t
@inline _emissive_le(t::Raycore.TextureRef) = t
@inline _emissive_le(x) = TexHandle(x)

# Value at a uv, for both representations.
@propagate_inbounds _emissive_value(Le::TexHandle, uv::Point2f) = const_spectrum(Le)
@propagate_inbounds _emissive_value(Le, uv::Point2f) = Le(uv)

# ============================================================================
# User-friendly keyword constructor
# ============================================================================

"""
    Emissive(; Le=RGBSpectrum(1), scale=1.0, two_sided=false)

Create an emissive material for area lights. Photometric normalization is applied
automatically (matching pbrt-v4). `Le` values correspond directly to pbrt's `"rgb L"`.

# Examples
```julia
# Equivalent to pbrt-v4: AreaLightSource "diffuse" "rgb L" [3 3 3]
Emissive(Le=(3, 3, 3), scale=1.0, two_sided=true)
```
"""
function Emissive(;
    Le=RGBSpectrum(1f0),
    scale::Real=1f0,
    two_sided::Bool=false
)
    # Apply photometric normalization matching pbrt-v4's area light creation:
    # scale /= SpectrumToPhotometric(Le_spectrum)
    normalized_scale = Float32(scale) / D65_PHOTOMETRIC
    Emissive(_emissive_le(Le), normalized_scale, two_sided)
end

# ============================================================================
# Material Interface Implementation
# ============================================================================

"""
    get_emission(mat::Emissive, si::SurfaceInteraction) -> RGBSpectrum

Get the emitted radiance at a surface point.
Returns zero if the surface is one-sided and we're on the back.
"""
@propagate_inbounds function get_emission(mat::Emissive, wo::Vec3f, n::Vec3f, uv::Point2f)
    # Check if we're on the emitting side
    cos_theta = dot(wo, n)
    if !mat.two_sided && cos_theta < 0f0
        return RGBSpectrum(0f0)
    end
    return _emissive_value(mat.Le, uv) * mat.scale
end

"""
    get_emission(mat::Emissive, uv::Point2f) -> RGBSpectrum

Get the emitted radiance at UV coordinates (without directional check).
"""
@propagate_inbounds function get_emission(mat::Emissive, uv::Point2f)
    return _emissive_value(mat.Le, uv) * mat.scale
end

# Base fallbacks for non-emissive materials are in material.jl (included before this file)

"""
    is_emissive(mat::Material) -> Bool

Check if a material emits light.
"""
@propagate_inbounds is_emissive(::Emissive) = true

"""
    is_pure_emissive(mat::Material) -> Bool

Check if a material is purely emissive (no BSDF, only emits light).
"""
@propagate_inbounds is_pure_emissive(::Emissive) = true

# ============================================================================
# GPU Support
# ============================================================================

"""
    to_gpu(ArrayType, mat::Emissive)

Convert Emissive to GPU-compatible form.
"""
function to_gpu(ArrayType, mat::Emissive)
    return Emissive(to_gpu(ArrayType, mat.Le), mat.scale, mat.two_sided)
end
to_gpu(ArrayType, h::TexHandle) = h

# ============================================================================
# Albedo extraction for denoising auxiliary buffers
# ============================================================================

"""
    get_albedo(mat::Emissive, uv::Point2f) -> RGBSpectrum

Get the "albedo" of an emissive material for denoising.
For emissive materials, we return the normalized emission color.
"""
@propagate_inbounds function get_albedo(mat::Emissive, uv::Point2f)
    Le = _emissive_value(mat.Le, uv)
    # Return normalized color (so it's in 0-1 range for denoising)
    luminance = to_Y(Le)
    if luminance > 0f0
        return Le / luminance
    else
        return RGBSpectrum(0f0)
    end
end

