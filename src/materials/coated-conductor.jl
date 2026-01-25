# ============================================================================
# CoatedConductorMaterial - Layered material with dielectric coating over conductor
# ============================================================================
# Port of pbrt-v4's CoatedConductorMaterial using LayeredBxDF
#
# The material consists of:
# - Top layer: Dielectric interface (can be rough or smooth)
# - Bottom layer: Conductor (metal)
# - Optional absorbing medium between layers
#
# Reference: pbrt-v4 src/pbrt/materials.cpp CoatedConductorMaterial::GetBxDF (lines 345-392)
# Reference: pbrt-v4 src/pbrt/bxdfs.h CoatedConductorBxDF (lines 911-918)

"""
    CoatedConductorMaterial

A layered material with a dielectric coating over a conductor (metal) base.
This implements pbrt-v4's coatedconductor material using random walk
sampling between the layers (LayeredBxDF algorithm).

# Fields
## Interface (coating) layer
- `interface_u_roughness`: U roughness for the dielectric coating
- `interface_v_roughness`: V roughness for the dielectric coating
- `interface_eta`: Index of refraction of the dielectric coating

## Conductor (base) layer
- `conductor_eta`: Complex index of refraction (real part) - OR use reflectance
- `conductor_k`: Complex index of refraction (imaginary part)
- `reflectance`: Alternative to eta/k - artist-friendly reflectance color
- `conductor_u_roughness`: U roughness for the conductor
- `conductor_v_roughness`: V roughness for the conductor

## Volumetric scattering (between layers)
- `thickness`: Thickness of the coating layer (affects absorption)
- `albedo`: Single-scattering albedo of medium between layers (0 = no absorption)
- `g`: Henyey-Greenstein asymmetry parameter for medium scattering

## Random walk parameters
- `max_depth`: Maximum random walk depth
- `n_samples`: Number of samples for estimating the BSDF
- `remap_roughness`: Whether to remap roughness to microfacet alpha

# Notes
- **Critical:** Conductor eta/k are scaled by interface IOR: ce /= ieta, ck /= ieta
- If `conductor_eta` is nothing, uses reflectance-based approach
"""
struct CoatedConductorMaterial{
    IURoughTex, IVRoughTex,
    CETex, CKTex, CReflTex, CURoughTex, CVRoughTex,
    ThickTex, AlbedoTex, GTex
} <: Material
    # Interface parameters
    interface_u_roughness::IURoughTex  # Texture{Float32}
    interface_v_roughness::IVRoughTex  # Texture{Float32}
    interface_eta::Float32             # Scalar IOR for interface

    # Conductor parameters (either eta/k OR reflectance)
    conductor_eta::CETex               # Texture{RGBSpectrum} - complex IOR real part (or nothing)
    conductor_k::CKTex                 # Texture{RGBSpectrum} - complex IOR imaginary part
    reflectance::CReflTex              # Texture{RGBSpectrum} - alternative to eta/k
    conductor_u_roughness::CURoughTex  # Texture{Float32}
    conductor_v_roughness::CVRoughTex  # Texture{Float32}

    # Volumetric scattering
    thickness::ThickTex                # Texture{Float32}
    albedo::AlbedoTex                  # Texture{RGBSpectrum} - medium albedo
    g::GTex                            # Texture{Float32} - HG asymmetry

    # Algorithm parameters
    max_depth::Int32
    n_samples::Int32
    remap_roughness::Bool

    # Mode flag: true if using eta/k, false if using reflectance
    use_eta_k::Bool
end

# Full constructor with all textures
function CoatedConductorMaterial(
    interface_u_roughness::Texture,
    interface_v_roughness::Texture,
    interface_eta::Float32,
    conductor_eta::Union{Texture, Nothing},
    conductor_k::Union{Texture, Nothing},
    reflectance::Union{Texture, Nothing},
    conductor_u_roughness::Texture,
    conductor_v_roughness::Texture,
    thickness::Texture,
    albedo::Texture,
    g::Texture,
    max_depth::Int,
    n_samples::Int,
    remap_roughness::Bool
)
    use_eta_k = !isnothing(conductor_eta)

    # If not using eta/k, set them to dummy values (raw values for constants)
    ce = isnothing(conductor_eta) ? RGBSpectrum(1f0) : conductor_eta
    ck = isnothing(conductor_k) ? RGBSpectrum(0f0) : conductor_k
    refl = isnothing(reflectance) ? RGBSpectrum(1f0) : reflectance

    CoatedConductorMaterial{
        typeof(interface_u_roughness), typeof(interface_v_roughness),
        typeof(ce), typeof(ck), typeof(refl),
        typeof(conductor_u_roughness), typeof(conductor_v_roughness),
        typeof(thickness), typeof(albedo), typeof(g)
    }(
        interface_u_roughness, interface_v_roughness, interface_eta,
        ce, ck, refl,
        conductor_u_roughness, conductor_v_roughness,
        thickness, albedo, g,
        Int32(max_depth), Int32(n_samples), remap_roughness,
        use_eta_k
    )
end

"""
    CoatedConductorMaterial(; interface_roughness=0.0, interface_eta=1.5, ...)

Create a coated conductor material with keyword arguments.

# Arguments
## Interface (coating)
- `interface_roughness`: Coating roughness (scalar or (u,v) tuple), default 0
- `interface_eta`: Coating IOR (default 1.5)

## Conductor (base) - use EITHER eta/k OR reflectance
- `conductor_eta`: Complex IOR real part (RGBSpectrum, tuple, or Texture)
- `conductor_k`: Complex IOR imaginary part
- `reflectance`: Alternative artist-friendly color (if eta/k not specified)
- `conductor_roughness`: Conductor roughness (scalar or (u,v) tuple), default 0.01

## Volumetric
- `thickness`: Coating thickness (default 0.01)
- `albedo`: Medium albedo (default 0 = no medium)
- `g`: HG asymmetry (default 0 = isotropic)

## Algorithm
- `max_depth`: Max random walk depth (default 10)
- `n_samples`: Number of samples (default 1)
- `remap_roughness`: Remap roughness to alpha (default true)

# Examples
```julia
# Glossy coated gold
CoatedConductorMaterial(
    interface_roughness=0.05,
    conductor_eta=(0.143, 0.374, 1.442),  # Gold
    conductor_k=(3.983, 2.385, 1.603)
)

# Coated copper using reflectance
CoatedConductorMaterial(
    reflectance=(0.95, 0.64, 0.54),  # Copper-like
    conductor_roughness=0.1
)

# Car paint effect (rough coating over smooth metal)
CoatedConductorMaterial(
    interface_roughness=0.3,
    conductor_roughness=0.01,
    reflectance=(0.9, 0.1, 0.1)  # Red metallic
)
```
"""
function CoatedConductorMaterial(;
    # Interface parameters
    interface_roughness = 0f0,
    interface_eta::Real = 1.5f0,
    # Conductor parameters - eta/k mode
    conductor_eta = nothing,
    conductor_k = nothing,
    # Conductor parameters - reflectance mode
    reflectance = nothing,
    conductor_roughness = 0.01f0,
    # Volumetric
    thickness = 0.01f0,
    albedo = RGBSpectrum(0f0),
    g = 0f0,
    # Algorithm
    max_depth::Int = 10,
    n_samples::Int = 1,
    remap_roughness::Bool = true
)
    # Handle interface roughness - can be scalar or (u,v) tuple
    iu_rough, iv_rough = if interface_roughness isa Tuple
        Float32(interface_roughness[1]), Float32(interface_roughness[2])
    else
        Float32(interface_roughness), Float32(interface_roughness)
    end

    # Handle conductor roughness - can be scalar or (u,v) tuple
    cu_rough, cv_rough = if conductor_roughness isa Tuple
        Float32(conductor_roughness[1]), Float32(conductor_roughness[2])
    else
        Float32(conductor_roughness), Float32(conductor_roughness)
    end

    # Determine mode: eta/k or reflectance
    # If conductor_eta is provided, use eta/k mode; otherwise use reflectance mode
    if !isnothing(conductor_eta)
        # eta/k mode
        if isnothing(conductor_k)
            error("conductor_k must be provided when using conductor_eta")
        end
        CoatedConductorMaterial(
            _to_texture(iu_rough),
            _to_texture(iv_rough),
            Float32(interface_eta),
            _to_texture(conductor_eta),
            _to_texture(conductor_k),
            nothing,  # reflectance not used
            _to_texture(cu_rough),
            _to_texture(cv_rough),
            _to_texture(Float32(thickness)),
            _to_texture(albedo),
            _to_texture(Float32(g)),
            max_depth,
            n_samples,
            remap_roughness
        )
    else
        # reflectance mode
        refl = isnothing(reflectance) ? RGBSpectrum(1f0) : reflectance
        CoatedConductorMaterial(
            _to_texture(iu_rough),
            _to_texture(iv_rough),
            Float32(interface_eta),
            nothing,  # eta not used
            nothing,  # k not used
            _to_texture(refl),
            _to_texture(cu_rough),
            _to_texture(cv_rough),
            _to_texture(Float32(thickness)),
            _to_texture(albedo),
            _to_texture(Float32(g)),
            max_depth,
            n_samples,
            remap_roughness
        )
    end
end

# Mark as non-emissive
is_emissive(::CoatedConductorMaterial) = false

"""Type alias: `CoatedConductor` is the same as `CoatedConductorMaterial`"""
const CoatedConductor = CoatedConductorMaterial
