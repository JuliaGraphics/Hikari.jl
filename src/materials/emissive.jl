# EmissiveMaterial - Area light emitter for PhysicalWavefront
# Enables geometry to act as light sources

# ============================================================================
# EmissiveMaterial Type
# ============================================================================

"""
    EmissiveMaterial{LeTex}

Material that emits light, enabling geometry to act as an area light source.

This is the primary way to create area lights in PhysicalWavefront rendering.
Triangles with this material will contribute direct illumination when hit
by shadow rays and indirect illumination when hit by camera/bounce rays.

# Fields
* `Le`: Emitted radiance (color/intensity texture or TextureRef)
* `scale`: Intensity multiplier applied to Le
* `two_sided`: If true, emits from both sides of the surface

# Notes
- EmissiveMaterial surfaces do not reflect light (they only emit)
- For surfaces that both emit AND reflect, layer EmissiveMaterial with another material
- The actual emitted radiance is `Le * scale`
"""
struct EmissiveMaterial{LeTex} <: Material
    Le::LeTex  # Texture{RGBSpectrum} or TextureRef{RGBSpectrum}
    scale::Float32
    two_sided::Bool
end

function EmissiveMaterial(Le::Texture, scale::Float32, two_sided::Bool)
    EmissiveMaterial{typeof(Le)}(Le, scale, two_sided)
end

# Constructor for TextureRef (GPU path)
function EmissiveMaterial(Le::TextureRef{RGBSpectrum}, scale::Float32, two_sided::Bool)
    EmissiveMaterial{typeof(Le)}(Le, scale, two_sided)
end

# ============================================================================
# User-friendly keyword constructor
# ============================================================================

"""
    EmissiveMaterial(; Le=RGBSpectrum(1), scale=1.0, two_sided=false)

Create an emissive (area light) material.

# Arguments
- `Le`: Emitted radiance color - can be RGBSpectrum, (r,g,b) tuple, or Texture
- `scale`: Intensity multiplier (useful for adjusting brightness without changing color)
- `two_sided`: Whether to emit from both sides of the surface

# Examples
```julia
# Simple white area light
EmissiveMaterial(Le=(10, 10, 10))

# Warm-colored light panel
EmissiveMaterial(Le=RGBSpectrum(15, 12, 8), two_sided=true)

# High-intensity spotlight
EmissiveMaterial(Le=(1, 1, 1), scale=100.0)

# Textured emission (e.g., neon sign)
EmissiveMaterial(Le=neon_texture, scale=5.0)
```
"""
function EmissiveMaterial(;
    Le=RGBSpectrum(1f0),
    scale::Real=1f0,
    two_sided::Bool=false
)
    EmissiveMaterial(_to_texture(Le), Float32(scale), two_sided)
end

# ============================================================================
# Material Interface Implementation
# ============================================================================

"""
    get_emission(mat::EmissiveMaterial, si::SurfaceInteraction) -> RGBSpectrum

Get the emitted radiance at a surface point.
Returns zero if the surface is one-sided and we're on the back.
"""
@propagate_inbounds function get_emission(mat::EmissiveMaterial, wo::Vec3f, n::Vec3f, uv::Point2f)
    # Check if we're on the emitting side
    cos_theta = dot(wo, n)
    if !mat.two_sided && cos_theta < 0f0
        return RGBSpectrum(0f0)
    end
    # Evaluate Le texture at UV
    Le = mat.Le(uv)
    return Le * mat.scale
end

"""
    get_emission(mat::EmissiveMaterial, uv::Point2f) -> RGBSpectrum

Get the emitted radiance at UV coordinates (without directional check).
"""
@propagate_inbounds function get_emission(mat::EmissiveMaterial, uv::Point2f)
    Le = mat.Le(uv)
    return Le * mat.scale
end

# For non-emissive materials, emission is zero
@propagate_inbounds get_emission(::Material, ::Vec3f, ::Vec3f, ::Point2f) = RGBSpectrum(0f0)
@propagate_inbounds get_emission(::Material, ::Point2f) = RGBSpectrum(0f0)

"""
    is_emissive(mat::Material) -> Bool

Check if a material emits light.
"""
@propagate_inbounds is_emissive(::EmissiveMaterial) = true
@propagate_inbounds is_emissive(::Material) = false

# ============================================================================
# BSDF Implementation for EmissiveMaterial
# ============================================================================

"""
    compute_bsdf(mat::EmissiveMaterial, si::SurfaceInteraction, ::Bool, transport)

EmissiveMaterial has no BSDF (pure emitter, doesn't scatter light).
Returns an empty BSDF.
"""
@propagate_inbounds function compute_bsdf(mat::EmissiveMaterial, textures, si::SurfaceInteraction, ::Bool, transport)
    # No scattering - just emission
    return BSDF(si)
end

"""
    shade(mat::EmissiveMaterial, ray, si, scene, beta, depth, max_depth) -> RGBSpectrum

Shading for emissive material returns the emission directly.
Emissive materials don't reflect light, they only emit.
"""
@propagate_inbounds function shade(mat::EmissiveMaterial, ray::RayDifferentials, si::SurfaceInteraction,
                       scene::Scene, beta::RGBSpectrum, depth::Int32, max_depth::Int32)
    wo = si.core.wo
    n = si.core.n
    uv = si.core.uv
    return beta * get_emission(mat, wo, n, uv)
end

# ============================================================================
# GPU Support
# ============================================================================

"""
    to_gpu(ArrayType, mat::EmissiveMaterial)

Convert EmissiveMaterial to GPU-compatible form.
"""
function to_gpu(ArrayType, mat::EmissiveMaterial)
    Le_gpu = to_gpu(ArrayType, mat.Le)
    return EmissiveMaterial(Le_gpu, mat.scale, mat.two_sided)
end

# ============================================================================
# Albedo extraction for denoising auxiliary buffers
# ============================================================================

"""
    get_albedo(mat::EmissiveMaterial, uv::Point2f) -> RGBSpectrum

Get the "albedo" of an emissive material for denoising.
For emissive materials, we return the normalized emission color.
"""
@propagate_inbounds function get_albedo(mat::EmissiveMaterial, uv::Point2f)
    Le = mat.Le(uv)
    # Return normalized color (so it's in 0-1 range for denoising)
    luminance = to_Y(Le)
    if luminance > 0f0
        return Le / luminance
    else
        return RGBSpectrum(0f0)
    end
end
