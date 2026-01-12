# ============================================================================
# CoatedDiffuseMaterial - Layered material with dielectric coating over diffuse
# ============================================================================
# Port of pbrt-v4's CoatedDiffuseMaterial using LayeredBxDF
#
# The material consists of:
# - Top layer: Dielectric interface (can be rough or smooth)
# - Bottom layer: Diffuse reflector
# - Optional absorbing medium between layers
#
# Reference: pbrt-v4 src/pbrt/bxdfs.h LayeredBxDF, CoatedDiffuseBxDF

"""
    CoatedDiffuseMaterial

A layered material with a dielectric coating over a diffuse base.
This implements pbrt-v4's coateddiffuse material using random walk
sampling between the layers (LayeredBxDF algorithm).

# Fields
- `reflectance`: Diffuse reflectance of the base layer (RGB color)
- `u_roughness`: Roughness in U direction for the dielectric coating
- `v_roughness`: Roughness in V direction for the dielectric coating
- `thickness`: Thickness of the coating layer (affects absorption)
- `eta`: Index of refraction of the dielectric coating
- `albedo`: Single-scattering albedo of medium between layers (0 = no absorption)
- `g`: Henyey-Greenstein asymmetry parameter for medium scattering
- `max_depth`: Maximum random walk depth
- `n_samples`: Number of samples for estimating the BSDF
- `remap_roughness`: Whether to remap roughness to microfacet alpha
"""
struct CoatedDiffuseMaterial{ReflTex, URoughTex, VRoughTex, ThickTex, AlbedoTex, GTex} <: Material
    reflectance::ReflTex    # Texture{RGBSpectrum} - diffuse color
    u_roughness::URoughTex  # Texture{Float32}
    v_roughness::VRoughTex  # Texture{Float32}
    thickness::ThickTex     # Texture{Float32}
    eta::Float32            # Index of refraction
    albedo::AlbedoTex       # Texture{RGBSpectrum} - medium albedo (0 = no medium)
    g::GTex                 # Texture{Float32} - HG asymmetry
    max_depth::Int32
    n_samples::Int32
    remap_roughness::Bool
end

# Full constructor
function CoatedDiffuseMaterial(
    reflectance::Texture,
    u_roughness::Texture,
    v_roughness::Texture,
    thickness::Texture,
    eta::Float32,
    albedo::Texture,
    g::Texture,
    max_depth::Int,
    n_samples::Int,
    remap_roughness::Bool
)
    CoatedDiffuseMaterial{
        typeof(reflectance), typeof(u_roughness), typeof(v_roughness),
        typeof(thickness), typeof(albedo), typeof(g)
    }(
        reflectance, u_roughness, v_roughness, thickness,
        eta, albedo, g, Int32(max_depth), Int32(n_samples), remap_roughness
    )
end

"""
    CoatedDiffuseMaterial(; reflectance, roughness=0, thickness=0.01, eta=1.5, ...)

Create a coated diffuse material with keyword arguments.

# Arguments
- `reflectance`: Diffuse color (RGBSpectrum, tuple, or Texture)
- `roughness`: Surface roughness (scalar or (u,v) tuple)
- `thickness`: Coating thickness (default 0.01)
- `eta`: Index of refraction (default 1.5 for typical dielectric)
- `albedo`: Medium albedo for absorption (default 0 = no medium)
- `g`: HG asymmetry parameter (default 0 = isotropic)
- `max_depth`: Max random walk depth (default 10)
- `n_samples`: Number of samples (default 1)
- `remap_roughness`: Remap roughness to alpha (default true)

# Examples
```julia
# Simple coated diffuse (glossy plastic-like)
CoatedDiffuseMaterial(reflectance=(0.4, 0.45, 0.35), roughness=0)

# Rough coating
CoatedDiffuseMaterial(reflectance=(0.8, 0.2, 0.2), roughness=0.3)

# With absorbing medium
CoatedDiffuseMaterial(reflectance=(0.9, 0.9, 0.9), albedo=(0.8, 0.4, 0.2))
```
"""
function CoatedDiffuseMaterial(;
    reflectance = RGBSpectrum(0.5f0),
    roughness = 0f0,
    thickness = 0.01f0,
    eta::Real = 1.5f0,
    albedo = RGBSpectrum(0f0),
    g = 0f0,
    max_depth::Int = 10,
    n_samples::Int = 1,
    remap_roughness::Bool = true
)
    # Handle roughness - can be scalar or (u,v) tuple
    u_rough, v_rough = if roughness isa Tuple
        Float32(roughness[1]), Float32(roughness[2])
    else
        Float32(roughness), Float32(roughness)
    end

    CoatedDiffuseMaterial(
        _to_texture(reflectance),
        _to_texture(u_rough),
        _to_texture(v_rough),
        _to_texture(Float32(thickness)),
        Float32(eta),
        _to_texture(albedo),
        _to_texture(Float32(g)),
        max_depth,
        n_samples,
        remap_roughness
    )
end

# Mark as non-emissive
is_emissive(::CoatedDiffuseMaterial) = false
