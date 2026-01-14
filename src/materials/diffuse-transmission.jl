# ============================================================================
# DiffuseTransmissionMaterial - Diffuse reflection and transmission
# ============================================================================
# Port of pbrt-v4's DiffuseTransmissionMaterial and DiffuseTransmissionBxDF
#
# This material models surfaces that scatter light diffusely in both
# reflection and transmission, like thin cloth, paper, or leaves.
#
# Reference: pbrt-v4 src/pbrt/bxdfs.h DiffuseTransmissionBxDF (lines 84-164)

"""
    DiffuseTransmissionMaterial{RTex, TTex}

A material that diffusely reflects and transmits light.

Models surfaces like paper, thin fabric, or leaves where light scatters
diffusely on both sides. The reflection and transmission are independent
Lambertian distributions.

# Fields
- `reflectance`: Diffuse reflectance color (same hemisphere as incident)
- `transmittance`: Diffuse transmittance color (opposite hemisphere)
- `scale`: Intensity multiplier applied to both R and T

# Physics
- Reflection: f = R/π (same hemisphere)
- Transmission: f = T/π (opposite hemisphere)
- Sampling: probability proportional to max(R) and max(T)

# Usage
```julia
# Thin white paper (equal reflection and transmission)
paper = DiffuseTransmission(reflectance=(0.8, 0.8, 0.8), transmittance=(0.5, 0.5, 0.5))

# Green leaf (green transmission, less reflection)
leaf = DiffuseTransmission(reflectance=(0.2, 0.3, 0.1), transmittance=(0.1, 0.5, 0.1))
```
"""
struct DiffuseTransmissionMaterial{RTex, TTex} <: Material
    reflectance::RTex    # Texture{RGBSpectrum} - diffuse reflection
    transmittance::TTex  # Texture{RGBSpectrum} - diffuse transmission
    scale::Float32       # Intensity scale
end

# Full constructor
function DiffuseTransmissionMaterial(
    reflectance::Texture,
    transmittance::Texture,
    scale::Float32
)
    DiffuseTransmissionMaterial{typeof(reflectance), typeof(transmittance)}(
        reflectance, transmittance, scale
    )
end

"""
    DiffuseTransmissionMaterial(; reflectance, transmittance, scale=1.0)

Create a diffuse transmission material with keyword arguments.

# Arguments
- `reflectance`: Diffuse reflection color (RGBSpectrum, tuple, or Texture)
- `transmittance`: Diffuse transmission color (RGBSpectrum, tuple, or Texture)
- `scale`: Intensity multiplier (default 1.0)

# Examples
```julia
# Thin translucent material
DiffuseTransmission(reflectance=(0.5, 0.5, 0.5), transmittance=(0.3, 0.3, 0.3))

# Pure transmission (no reflection)
DiffuseTransmission(reflectance=(0, 0, 0), transmittance=(1, 1, 1))
```
"""
function DiffuseTransmissionMaterial(;
    reflectance = RGBSpectrum(0.5f0),
    transmittance = RGBSpectrum(0.5f0),
    scale::Real = 1f0
)
    DiffuseTransmissionMaterial(
        _to_texture(reflectance),
        _to_texture(transmittance),
        Float32(scale)
    )
end

# Mark as non-emissive
is_emissive(::DiffuseTransmissionMaterial) = false

"""Type alias: `DiffuseTransmission` is the same as `DiffuseTransmissionMaterial`"""
const DiffuseTransmission = DiffuseTransmissionMaterial
