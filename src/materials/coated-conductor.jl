# ============================================================================
# CoatedConductor - Layered material with dielectric coating over conductor
# ============================================================================
# Port of pbrt-v4's CoatedConductor using LayeredBxDF
#
# The material consists of:
# - Top layer: Dielectric interface (can be rough or smooth)
# - Bottom layer: Conductor (metal)
# - Optional absorbing medium between layers
#
# Reference: pbrt-v4 src/pbrt/materials.cpp CoatedConductor::GetBxDF (lines 345-392)
# Reference: pbrt-v4 src/pbrt/bxdfs.h CoatedConductorBxDF (lines 911-918)

"""
    CoatedConductor

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
# Non-parametric texture parameters: constant-vs-texture must not change the
# material's TYPE. Spectral IOR fields stay parametric because a
# PiecewiseLinearSpectrum cannot live inline in a handle.
struct CoatedConductor{IURoughT, IVRoughT, IEtaT, CETex, CKTex, CReflT, CURoughT, CVRoughT, ThickT, AlbedoT, GT, DispT} <: Material
    # Interface parameters
    interface_u_roughness::IURoughT
    interface_v_roughness::IVRoughT
    interface_eta::IEtaT               # Interface IOR: a number, or a spectrum that disperses

    # Conductor parameters: eta/k, or a reflectance pbrt-v4 converts at each wavelength (`conductor_ior`)
    conductor_eta::CETex               # Spectral complex IOR real part
    conductor_k::CKTex                 # Spectral complex IOR imaginary part
    conductor_reflectance::CReflT      # Instead of eta/k (`TexHandle()` where they are given)
    conductor_u_roughness::CURoughT
    conductor_v_roughness::CVRoughT

    # Volumetric scattering
    thickness::ThickT
    albedo::AlbedoT
    g::GT

    # Algorithm parameters
    max_depth::Int32
    n_samples::Int32
    remap_roughness::Bool
    displacement::DispT       # pbrt-v4 `Material::displacement` height field (NONE = flat)
end

# Full constructor with all textures. Texture args are unannotated on purpose:
# fields accept any texture-like value (Texture, CheckerboardTexture, raw
# constants, TextureRef).
function CoatedConductor(
    interface_u_roughness, interface_v_roughness, interface_eta,
    conductor_eta,  # PiecewiseLinearSpectrum (named metal) or RGB-ish
    conductor_k,
    conductor_reflectance,
    conductor_u_roughness, conductor_v_roughness,
    thickness, albedo, g, max_depth::Int, n_samples::Int,
    remap_roughness::Bool, displacement = TexHandle(),
)
    CoatedConductor(
        matparam(interface_u_roughness), matparam(interface_v_roughness),
        matparam(interface_eta),
        matparam(conductor_eta), matparam(conductor_k), matparam(conductor_reflectance),
        matparam(conductor_u_roughness), matparam(conductor_v_roughness),
        matparam(thickness), matparam(albedo), matparam(g),
        Int32(max_depth), Int32(n_samples), remap_roughness, matparam(displacement),
    )
end

"""
    CoatedConductor(; interface_roughness=0.0, interface_eta=1.5, ...)

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
CoatedConductor(
    interface_roughness=0.05,
    conductor_eta=(0.143, 0.374, 1.442),  # Gold
    conductor_k=(3.983, 2.385, 1.603)
)

# Coated copper using reflectance
CoatedConductor(
    reflectance=(0.95, 0.64, 0.54),  # Copper-like
    conductor_roughness=0.1
)

# Car paint effect (rough coating over smooth metal)
CoatedConductor(
    interface_roughness=0.3,
    conductor_roughness=0.01,
    reflectance=(0.9, 0.1, 0.1)  # Red metallic
)
```
"""
function CoatedConductor(;
    # Interface parameters
    interface_roughness = 0f0,
    interface_eta = 1.5f0,
    # Conductor parameters — eta/k OR reflectance (converted to eta/k at construction)
    conductor_eta = nothing,
    conductor_k = nothing,
    reflectance = nothing,
    conductor_roughness = 0.01f0,
    # Volumetric
    thickness = 0.01f0,
    albedo = RGBSpectrum(0f0),
    g = 0f0,
    # Algorithm
    max_depth::Int = 10,
    n_samples::Int = 1,
    remap_roughness::Bool = true,
    bump = nothing,
)
    iu_rough, iv_rough = roughness_pair(interface_roughness)
    cu_rough, cv_rough = roughness_pair(conductor_roughness)

    # Either eta/k or a reflectance, as pbrt-v4's; neither is its copper.
    ce, ck, cr = if !isnothing(reflectance)
        (isnothing(conductor_eta) && isnothing(conductor_k)) ||
            throw(ArgumentError("a CoatedConductor takes either `conductor_eta` and `conductor_k` or a `reflectance`, not both"))
        TexHandle(), TexHandle(), reflectance
    elseif !isnothing(conductor_eta)
        isnothing(conductor_k) && throw(ArgumentError("`conductor_k` must be given with `conductor_eta`"))
        conductor_eta, conductor_k, TexHandle()
    else
        CU_ETA_SPECTRUM, CU_K_SPECTRUM, TexHandle()
    end

    CoatedConductor(
        matparam(iu_rough), matparam(iv_rough), interface_eta,
        ce, ck, cr,
        matparam(cu_rough), matparam(cv_rough),
        matparam(thickness), matparam(albedo), matparam(g),
        max_depth, n_samples, remap_roughness, matparam(bump),
    )
end

# Mark as non-emissive
is_emissive(::CoatedConductor) = false


"""
A coated conductor's BSDF: pbrt-v4's `CoatedConductorBxDF`, a `LayeredBSDF` of its dielectric coat
over its conductor, whose IOR is relative to the coat's.
"""
@propagate_inbounds function get_bxdf(mat::CoatedConductor, table, textures, tfc::TextureFilterContext,
                                      lambda::Wavelengths, regularize::Bool, wo::Vec3f, ns::Vec3f, dpdus::Vec3f)
    ieta, dispersive = eval_dielectric_ior(textures, mat.interface_eta, tfc, lambda)
    ieta == 0f0 && (ieta = 1f0)
    iu_roughness = eval_handle(textures, mat.interface_u_roughness, tfc)
    iv_roughness = eval_handle(textures, mat.interface_v_roughness, tfc)
    i_alpha_x = mat.remap_roughness ? roughness_to_α(iu_roughness) : iu_roughness
    i_alpha_y = mat.remap_roughness ? roughness_to_α(iv_roughness) : iv_roughness
    cu_roughness = eval_handle(textures, mat.conductor_u_roughness, tfc)
    cv_roughness = eval_handle(textures, mat.conductor_v_roughness, tfc)
    c_alpha_x = mat.remap_roughness ? roughness_to_α(cu_roughness) : cu_roughness
    c_alpha_y = mat.remap_roughness ? roughness_to_α(cv_roughness) : cv_roughness
    if regularize
        i_alpha_x = regularize_alpha(i_alpha_x)
        i_alpha_y = regularize_alpha(i_alpha_y)
        c_alpha_x = regularize_alpha(c_alpha_x)
        c_alpha_y = regularize_alpha(c_alpha_y)
    end
    thickness = max(eval_handle(textures, mat.thickness, tfc), eps(Float32))
    albedo = uplift_rgb(table, clamp(eval_handle_spectrum(textures, mat.albedo, tfc)), lambda)
    g = clamp(eval_handle(textures, mat.g, tfc), -0.99f0, 0.99f0)
    # pbrt-v4 materials.cpp: the conductor's IOR relative to the coat's.
    ce, ck = conductor_ior(table, textures, mat.conductor_eta, mat.conductor_k, mat.conductor_reflectance, tfc, lambda)
    ce, ck = ce / ieta, ck / ieta
    top = (local_lobe(pbrt_dielectric(ieta, i_alpha_x, i_alpha_y), SpectralRadiance(1f0)),)
    bottom = (local_lobe(pbrt_conductor(ce, ck, c_alpha_x, c_alpha_y), SpectralRadiance(1f0)),)
    return LayeredBSDF{true}(top, bottom, ns, dpdus, thickness, albedo, g, mat.max_depth, mat.n_samples, lambda;
                             dispersive)
end
