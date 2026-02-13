# ============================================================================
# CoatedDiffuseTransmissionMaterial - Layered: dielectric coating + diffuse transmission
# ============================================================================
# Extension of CoatedDiffuseMaterial that uses DiffuseTransmissionBxDF as the
# bottom layer instead of DiffuseBxDF. This creates a material with a glossy
# dielectric coating over a diffuse substrate that both reflects and transmits.
#
# Ideal for: leaves, thin fabric, paper with a glossy surface coating.
#
# In pbrt-v4 terms: LayeredBxDF<DielectricBxDF, DiffuseTransmissionBxDF, true>

struct CoatedDiffuseTransmissionMaterial{ReflTex, TransTex, URoughTex, VRoughTex, ThickTex, AlbedoTex, GTex} <: Material
    reflectance::ReflTex    # Diffuse reflectance (same hemisphere)
    transmittance::TransTex # Diffuse transmittance (opposite hemisphere)
    u_roughness::URoughTex
    v_roughness::VRoughTex
    thickness::ThickTex     # Coating layer thickness
    eta::Float32            # IOR of dielectric coating
    albedo::AlbedoTex       # Medium albedo between layers (0 = no medium)
    g::GTex                 # HG asymmetry parameter
    max_depth::Int32
    n_samples::Int32
    remap_roughness::Bool
end

# Full constructor
function CoatedDiffuseTransmissionMaterial(
    reflectance::Texture,
    transmittance::Texture,
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
    CoatedDiffuseTransmissionMaterial{
        typeof(reflectance), typeof(transmittance),
        typeof(u_roughness), typeof(v_roughness),
        typeof(thickness), typeof(albedo), typeof(g)
    }(
        reflectance, transmittance, u_roughness, v_roughness, thickness,
        eta, albedo, g, Int32(max_depth), Int32(n_samples), remap_roughness
    )
end

function CoatedDiffuseTransmissionMaterial(;
    reflectance = RGBSpectrum(0.5f0),
    transmittance = RGBSpectrum(0.25f0),
    roughness = 0f0,
    thickness = 0.01f0,
    eta::Real = 1.5f0,
    albedo = RGBSpectrum(0f0),
    g = 0f0,
    max_depth::Int = 10,
    n_samples::Int = 1,
    remap_roughness::Bool = true
)
    u_rough, v_rough = if roughness isa Tuple
        Float32(roughness[1]), Float32(roughness[2])
    else
        Float32(roughness), Float32(roughness)
    end

    CoatedDiffuseTransmissionMaterial(
        _to_texture(reflectance),
        _to_texture(transmittance),
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

is_emissive(::CoatedDiffuseTransmissionMaterial) = false

const CoatedDiffuseTransmission = CoatedDiffuseTransmissionMaterial
