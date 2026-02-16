# DiffuseAreaLight - Per-triangle area light following pbrt-v4
#
# Each emissive triangle gets its own DiffuseAreaLight stored in scene.lights.
# Replaces EmissiveMaterial — emission now lives entirely on the light side.

"""
    DiffuseAreaLight{LeTex}

Per-triangle area light for emissive surfaces. Following pbrt-v4's DiffuseAreaLight,
stores triangle geometry for sampling and emission texture/color for evaluation.

Each emissive triangle creates one DiffuseAreaLight. The light sampler
(PowerLightSampler) includes these for importance sampling, enabling MIS
with BSDF sampling.

# Fields
- `vertices`: Triangle vertex positions (for uniform sampling)
- `normal`: Geometric normal of the triangle
- `area`: Triangle area (precomputed)
- `uv`: Vertex UV coordinates (for texture evaluation via barycentric interpolation)
- `Le`: Emitted radiance — `RGBSpectrum` for constant, `TextureRef` for textured emission
- `scale`: Intensity multiplier applied to Le
- `two_sided`: If true, emits from both sides of the surface
"""
struct DiffuseAreaLight{LeTex} <: Light
    vertices::SVector{3, Point3f}    # Triangle vertices (for sampling)
    normal::Normal3f                  # Geometric normal
    area::Float32                     # Triangle area (precomputed)
    uv::SVector{3, Point2f}          # Vertex UVs (for texture evaluation)
    Le::LeTex                         # Emission color/texture
    scale::Float32
    two_sided::Bool
end

is_δ_light(::DiffuseAreaLight) = false

# ============================================================================
# Emission Evaluation (pbrt-v4 DiffuseAreaLight::L)
# ============================================================================

"""
    arealight_Le(light::DiffuseAreaLight, lights_ctx, table, wo, n, uv, lambda) -> SpectralRadiance

Evaluate spectral emission at a surface point on this area light.
Following pbrt-v4 DiffuseAreaLight::L().

- `lights_ctx`: StaticMultiTypeSet for TextureRef dereference (lights container)
- `table`: RGBToSpectrumTable for spectral uplift
- `wo`: outgoing direction (toward camera)
- `n`: surface normal at hit point
- `uv`: texture coordinates at hit point
- `lambda`: sampled wavelengths
"""
@propagate_inbounds function arealight_Le(
    light::DiffuseAreaLight{RGBSpectrum}, lights_ctx,
    table::RGBToSpectrumTable, wo::Vec3f, n::Vec3f, ::Point2f, lambda::Wavelengths
)
    # One-sided check
    if !light.two_sided && dot(wo, n) < 0f0
        return SpectralRadiance()
    end
    Le_rgb = light.Le * light.scale
    return uplift_rgb(table, Le_rgb, lambda)
end

# TextureRef version — evaluates texture at hit UV
@propagate_inbounds function arealight_Le(
    light::DiffuseAreaLight, lights_ctx,
    table::RGBToSpectrumTable, wo::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
)
    # One-sided check
    if !light.two_sided && dot(wo, n) < 0f0
        return SpectralRadiance()
    end
    Le_rgb = eval_tex(lights_ctx, light.Le, uv) * light.scale
    return uplift_rgb(table, Le_rgb, lambda)
end

# Generic fallback for non-DiffuseAreaLight types (needed for GPU compilation:
# with_index generates code for ALL type slots, even unreachable ones)
@propagate_inbounds arealight_Le(::Light, lights_ctx, table, wo, n, uv, lambda) = SpectralRadiance()

# ============================================================================
# GPU Support
# ============================================================================

function to_gpu(ArrayType, light::DiffuseAreaLight)
    Le_gpu = to_gpu(ArrayType, light.Le)
    return DiffuseAreaLight(
        light.vertices, light.normal, light.area, light.uv,
        Le_gpu, light.scale, light.two_sided
    )
end

# RGBSpectrum doesn't need GPU conversion
to_gpu(::Any, le::RGBSpectrum) = le
