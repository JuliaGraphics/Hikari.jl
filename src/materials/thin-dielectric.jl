# ============================================================================
# ThinDielectricMaterial - Thin dielectric surface (e.g., window glass)
# ============================================================================
# Port of pbrt-v4's ThinDielectricMaterial and ThinDielectricBxDF
#
# Models a thin dielectric surface where light can pass through without
# the usual refraction bend. This is appropriate for thin surfaces like
# window glass where internal bounces contribute to the overall reflection.
#
# Key difference from regular Dielectric:
# - Transmitted direction is -wo (straight through, no refraction)
# - Reflectance accounts for multiple internal bounces: R' = R + T²R/(1-R²)
# - Always specular (no roughness parameter)
#
# Reference: pbrt-v4 src/pbrt/bxdfs.h ThinDielectricBxDF (lines 209-277)

"""
    ThinDielectricMaterial

A thin dielectric material for surfaces like window glass.

Unlike regular dielectric materials which refract light according to Snell's law,
thin dielectric materials transmit light straight through (wi = -wo) while
accounting for multiple internal reflections within the thin layer.

# Fields
- `eta`: Index of refraction of the dielectric

# Physics
For a thin dielectric layer:
- Single-surface Fresnel: R₀ = FrDielectric(cos_θ, eta)
- Multiple-bounce reflectance: R = R₀ + T₀²R₀/(1 - R₀²) where T₀ = 1 - R₀
- Transmittance: T = 1 - R
- Transmitted direction: wi = -wo (straight through, no bend)

# Usage
```julia
# Thin glass window
window = ThinDielectric(eta=1.5)

# Thin plastic film
film = ThinDielectric(eta=1.4)
```
"""
struct ThinDielectricMaterial <: Material
    eta::Float32
end

# Keyword constructor
function ThinDielectricMaterial(; eta::Real=1.5f0)
    ThinDielectricMaterial(Float32(eta))
end

# Mark as non-emissive
is_emissive(::ThinDielectricMaterial) = false

"""Type alias: `ThinDielectric` is the same as `ThinDielectricMaterial`"""
const ThinDielectric = ThinDielectricMaterial
