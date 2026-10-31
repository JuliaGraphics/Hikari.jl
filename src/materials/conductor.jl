# ============================================================================
# Conductor: pbrt-v4's metal, and the measured metals
# ============================================================================

"""
    Conductor(; ior = (0.2, 0.2, 0.2), extinction = (3.9, 3.9, 3.9), roughness = 0.1, remap_roughness = true,
              displacement = nothing)
    Conductor(; reflectance, roughness = 0.1, remap_roughness = true, displacement = nothing)

pbrt-v4's conductor: one GGX reflection lobe with the Fresnel of the complex IOR `ior` + i
`extinction` (Blender's Metallic BSDF with its physical conductor Fresnel), single scattering, as
pbrt renders it. Instead of its IOR a metal can be given its `reflectance`, its colour at normal
incidence, as pbrt converts it (IOR 1 + i 2√r/√(1 - r) at each wavelength). Not both.

* `ior`, `extinction`: a colour, a texture, or a spectrum (the measured metals `Gold()`, `Silver()`,
  `Copper()`, `Aluminum()` and `Brass()` are spectra).
* `roughness`: one value, or `(u, v)` along the tangent and across it; with `remap_roughness` the
  GGX alpha is pbrt's of it.

For Blender's artist-friendly metal with an edge tint, use [`Metallic`](@ref).
"""
struct Conductor{IorT, ExtinctionT, URoughT, VRoughT, ReflT, DispT} <: Material
    ior::IorT
    extinction::ExtinctionT
    u_roughness::URoughT
    v_roughness::VRoughT
    reflectance::ReflT
    remap_roughness::Bool
    displacement::DispT
end

function Conductor(; ior = nothing, extinction = nothing, reflectance = nothing, roughness = 0.1f0,
                   remap_roughness = true, displacement = nothing)
    if !isnothing(reflectance)
        (isnothing(ior) && isnothing(extinction)) ||
            throw(ArgumentError("a Conductor takes either `ior` and `extinction` or a `reflectance`, not both"))
    else
        ior = something(ior, (0.2f0, 0.2f0, 0.2f0))
        extinction = something(extinction, (3.9f0, 3.9f0, 3.9f0))
    end
    return Conductor(ior, extinction, roughness_pair(roughness)..., reflectance, remap_roughness, displacement)
end

"""    Gold(; roughness = 0, remap_roughness = true, displacement = nothing)

A gold `Conductor` of measured spectral IOR."""
Gold(; roughness = 0f0, kw...) = Conductor(; ior = AU_ETA_SPECTRUM, extinction = AU_K_SPECTRUM, roughness, kw...)
"""    Silver(; roughness = 0, remap_roughness = true, displacement = nothing)

A silver `Conductor` of measured spectral IOR."""
Silver(; roughness = 0f0, kw...) = Conductor(; ior = AG_ETA_SPECTRUM, extinction = AG_K_SPECTRUM, roughness, kw...)
"""    Copper(; roughness = 0, remap_roughness = true, displacement = nothing)

A copper `Conductor` of measured spectral IOR."""
Copper(; roughness = 0f0, kw...) = Conductor(; ior = CU_ETA_SPECTRUM, extinction = CU_K_SPECTRUM, roughness, kw...)
"""    Aluminum(; roughness = 0, remap_roughness = true, displacement = nothing)

An aluminium `Conductor` of measured spectral IOR."""
Aluminum(; roughness = 0f0, kw...) = Conductor(; ior = AL_ETA_SPECTRUM, extinction = AL_K_SPECTRUM, roughness, kw...)
"""    Brass(; roughness = 0, remap_roughness = true, displacement = nothing)

A brass (CuZn) `Conductor` of measured spectral IOR."""
Brass(; roughness = 0f0, kw...) = Conductor(; ior = CUZN_ETA_SPECTRUM, extinction = CUZN_K_SPECTRUM, roughness, kw...)

"""
A conductor's BSDF: pbrt-v4's `ConductorBxDF`, one GGX reflection lobe with the Fresnel of its
complex IOR (`ConductorFresnel`), single scattering, in pbrt's convention.
"""
@propagate_inbounds function get_bxdf(mat::Conductor, table, textures, tfc::TextureFilterContext, lambda::Wavelengths,
                                      regularize::Bool, wo::Vec3f, ns::Vec3f, dpdus::Vec3f)
    u_roughness = eval_handle(textures, mat.u_roughness, tfc)
    v_roughness = eval_handle(textures, mat.v_roughness, tfc)
    alpha_x = mat.remap_roughness ? roughness_to_α(u_roughness) : u_roughness
    alpha_y = mat.remap_roughness ? roughness_to_α(v_roughness) : v_roughness
    if regularize
        alpha_x, alpha_y = regularize_alpha(alpha_x), regularize_alpha(alpha_y)
    end
    eta, k = conductor_ior(table, textures, mat.ior, mat.extinction, mat.reflectance, tfc, lambda)
    lobe = pbrt_conductor(eta, k, alpha_x, alpha_y)
    n, t = pbrt_frame(wo, ns, dpdus)
    return pbrt_bsdf((Weighted(lobe, SpectralRadiance(1f0), 1f0, n, t),), lambda)
end

"""
    get_bxdf(mat, table, textures, tfc, lambda, regularize, wo, ns, dpdus) -> bxdf

Hikari's analogue of pbrt-v4's `Material::GetBxDF`: the BSDF of `mat` at a hit, with its textures
read through `tfc`, seen from `wo` in the shading frame of `ns` and `dpdus`, its spectra at the
path's wavelengths. A pbrt material's is a `LobeBSDF` (`pbrt_bsdf`).
"""
function get_bxdf end

# The layered materials, not yet lobes, resolve without the frame.
@inline get_bxdf(mat, table, textures, tfc, lambda, regularize::Bool, wo::Vec3f, ns::Vec3f, dpdus::Vec3f) =
    get_bxdf(mat, table, textures, tfc, lambda, regularize)
@inline get_bxdf(mat, table, textures, tfc, lambda, regularize::Bool) = mat
