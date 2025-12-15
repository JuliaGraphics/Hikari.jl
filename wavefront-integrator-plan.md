# Wavefront Integrator for Hikari - Implementation Plan

## Overview

Port the wavefront path tracer from Raycore docs to Hikari as a new integrator that supports all existing Hikari materials and lights while being drop-in compatible with existing scenes.

## Architecture Decisions

### 1. Integrator Structure

Create `WavefrontIntegrator` struct inheriting from `Integrator`:

```julia
struct WavefrontIntegrator{C<:Camera} <: Integrator
    camera::C
    max_depth::Int32
    samples_per_pixel::Int32
end
```

### 2. Re-use vs. Convert Strategy

**Materials**: **Re-use Hikari materials directly**
- The wavefront renderer will call Hikari's material system via `bsdf = material(si, allow_multiple_lobes, transport)`
- Supports: MatteMaterial, MirrorMaterial, GlassMaterial, PlasticMaterial
- No conversion needed - evaluate BSDF at shading time

**Lights**: **Re-use Hikari lights directly**
- Use `sample_li(light, interaction, u)` to sample light contribution
- Use `unoccluded(visibility_tester, scene)` for shadow testing
- Supports: PointLight, SpotLight, DirectionalLight, AmbientLight, EnvironmentLight
- The wavefront renderer will iterate lights via the scene's light tuple

### 3. Key Differences from Original Wavefront Renderer

| Aspect | Original (Raycore docs) | Hikari Version |
|--------|------------------------|----------------|
| Scene type | Custom `ctx` with materials/lights | `Hikari.Scene` with `MaterialScene` |
| Materials | Simple struct with base_color, metallic, roughness | Full BSDF system (Lambertian, Microfacet, etc.) |
| Lights | Simple point lights | All Hikari light types via `sample_li` |
| Camera | Simple pinhole | Hikari `PerspectiveCamera` with ray differentials |
| Output | Direct to image array | Hikari `Film` with filtering |

### 4. Work Queue Structures (SoA)

Adapt the original work queue structures for Hikari's types:

```julia
# Primary ray work
struct WavefrontRayWork
    ray::RayDifferentials      # Use ray differentials for texture filtering
    pixel::Point2f             # Film sample point
    beta::RGBSpectrum          # Path throughput
    sample_idx::Int32
end

# Primary hit work
struct WavefrontHitWork
    hit_found::Bool
    si::SurfaceInteraction     # Hikari surface interaction
    material_idx::Int32        # Index into materials array
    ray::RayDifferentials
    pixel::Point2f
    beta::RGBSpectrum
    sample_idx::Int32
end

# Shadow ray work - one per hit × light
struct WavefrontShadowWork
    visibility::VisibilityTester
    hit_idx::Int32
    light_idx::Int32
    light_contribution::RGBSpectrum  # Pre-computed Li * f * cos / pdf
end

# Shaded result
struct WavefrontShadedResult
    color::RGBSpectrum
    pixel::Point2f
    sample_idx::Int32
end
```

## Wavefront Pipeline Stages

### Stage 1: Generate Primary Camera Rays
- Use `generate_ray_differential(camera, camera_sample)` from Hikari
- Apply pixel jittering via sampler
- Store ray, pixel coords, initial beta=1 in primary_ray_queue

### Stage 2: Intersect Primary Rays
- Use `intersect!(scene.aggregate, ray)` returning `(hit, material, si)`
- Store hit info, material index, surface interaction in hit_queue

### Stage 3: Generate Shadow Rays (One per Light)
- For each hit, iterate over `scene.lights` tuple
- Call `sample_li(light, si.core, sample)` to get light direction and contribution
- Create `VisibilityTester` for each light
- Store in shadow_ray_queue

### Stage 4: Test Shadow Rays
- Call `unoccluded(visibility_tester, scene)` for each shadow ray
- Store visibility results

### Stage 5: Shade Primary Hits
- For each hit with visible lights:
  - Get BSDF: `bsdf = material(si, true, Radiance)`
  - Evaluate: `f = bsdf(wo, wi)`
  - Accumulate: `L += beta * f * Li * |cos| / pdf` for visible lights
- Handle environment light for misses: `le(env_light, ray)`

### Stage 6: Generate Bounce Rays (Reflection/Transmission)
- For non-terminated paths:
  - Sample BSDF: `wi, f, pdf, type = sample_f(bsdf, wo, sample, BSDF_ALL)`
  - Update throughput: `beta *= f * |cos| / pdf`
  - Apply Russian roulette for path termination
  - Generate bounce ray via `spawn_ray(si, wi)`

### Stage 7: Intersect Bounce Rays
- Same as Stage 2 for secondary rays

### Stage 8: Shade Bounce Hits (Direct Lighting)
- Same as Stage 5 but with updated beta
- Blend/accumulate into primary color

### Stage 9: Write to Film
- Use Hikari's `add_sample!(film, pixel, color, weight)` or direct pixel accumulation
- Call `to_framebuffer!(film, scale)` after all samples

## File Structure

```
dev/Hikari/src/integrators/wavefront.jl
```

Single file containing:
1. SoA helper macros (copied from original)
2. Work queue struct definitions
3. GPU kernels for each stage
4. `WavefrontIntegrator` struct
5. `render!` function implementing the pipeline
6. Callable interface `(::WavefrontIntegrator)(scene, film)`

## Integration with Hikari

Add to `Hikari.jl`:
```julia
include("integrators/wavefront.jl")
```

## Usage Example

```julia
using Hikari

# Create scene (same as for WhittedIntegrator)
material_scene = MaterialScene(primitives)
scene = Scene(lights, material_scene)

# Create camera and film (same as before)
camera = PerspectiveCamera(...)
film = Film(resolution, ...)

# Create wavefront integrator
integrator = WavefrontIntegrator(camera, max_depth=5, samples_per_pixel=4)

# Render (same API as other integrators)
integrator(scene, film)
```

## Light Type Support Details

| Light Type | sample_li | Shadow Test | Notes |
|------------|-----------|-------------|-------|
| PointLight | ✓ | ✓ | Delta position, pdf=1 |
| SpotLight | ✓ | ✓ | Delta position with falloff |
| DirectionalLight | ✓ | ✓ | Delta direction, needs preprocess |
| AmbientLight | ✓ | N/A | Infinite, no shadow rays needed |
| EnvironmentLight | ✓ | ✓ | Importance sampled, infinite |

## Material Type Support Details

| Material | BSDF Type | Wavefront Support |
|----------|-----------|-------------------|
| MatteMaterial | Lambertian/OrenNayar | ✓ Diffuse shading |
| MirrorMaterial | SpecularReflection | ✓ Perfect reflection bounce |
| GlassMaterial | FresnelSpecular/Microfacet | ✓ Reflection + transmission |
| PlasticMaterial | Lambertian + Microfacet | ✓ Diffuse + specular |

## GPU Considerations

1. **Unroll light iteration**: Use `Base.Cartesian.@nexprs` or `ntuple` for static light count
2. **Avoid tuple iteration**: Use explicit `while` loops instead of `for` over tuples
3. **SoA access**: Use `@get`/`@set` macros for efficient field extraction
4. **Backend agnostic**: Use KernelAbstractions for CPU/GPU compatibility

## Testing Strategy

1. Create test scene with all material types
2. Create test scene with all light types
3. Render with both WhittedIntegrator and WavefrontIntegrator
4. Compare output visually and numerically
