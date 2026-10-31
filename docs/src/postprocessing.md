## Postprocessing and Denoising

Hikari separates rendering from display mapping. The integrator writes sensor-calibrated linear HDR values to the framebuffer. Postprocessing converts those to a display-ready image without re-rendering.

```@setup postprocessing
using GeometryBasics, Hikari, ImageShow
to_mesh(prim) = normal_mesh(prim isa Sphere ? Tesselation(prim, 64) : prim)

function make_scene()
    scene = Hikari.Scene()
    push!(scene, to_mesh(Sphere(Point3f(-0.6f0, 0.4f0, 0f0), 0.4f0)),
          Hikari.Gold(roughness=0.03f0))
    push!(scene, to_mesh(Sphere(Point3f(0.5f0, 0.3f0, 0.2f0), 0.3f0)),
          Hikari.Glass(ior = 1.5f0))
    push!(scene, to_mesh(Rect3f(Vec3f(-3, 0, -3), Vec3f(6, 0.01, 6))),
          Hikari.Diffuse(color = Hikari.RGBSpectrum(0.85f0)))
    push!(scene, Hikari.PointLight(Point3f(2f0, 3f0, -2f0), Hikari.RGBSpectrum(40f0)))
    push!(scene, Hikari.AmbientLight(Hikari.RGBSpectrum(0.5f0)))
    Hikari.sync!(scene)
    return scene
end
```

### Pipeline

The rendering pipeline has three stages:

1. **Render** (sensor simulation baked in, needs re-render to change)
   - ISO, exposure time, white balance, sensor response curves
2. **Denoise** (optional, modifies framebuffer in-place)
   - Edge-aware filtering using normal/depth auxiliary buffers
3. **Postprocess** (display mapping, instant to re-run)
   - Exposure multiplier, tonemapping, gamma correction

```julia
# Full pipeline
integrator(scene, film, camera)           # step 1: render
fill_aux_buffers!(film, scene, camera)    # populate denoiser inputs
denoise!(film)                            # step 2: denoise (optional)
postprocess!(film; exposure=1.0, tonemap=:aces, gamma=2.2)  # step 3: display
```

### Tonemapping

`postprocess!` supports several tonemapping curves. Call it multiple times with different settings without re-rendering:

```@example postprocessing
scene = make_scene()
film = Hikari.Film(Point2f(512, 512))
camera = Hikari.PerspectiveCamera(
    Point3f(2f0, 1.5f0, -2.5f0), Point3f(0f0, 0.3f0, 0f0), film; fov=45f0)
Hikari.clear!(film)
Hikari.VolPath(samples=64, max_depth=8)(scene, film, camera)

# Try different tonemappers on the same render
Array(Hikari.postprocess!(film; tonemap=:aces, exposure=1.0f0, gamma=2.2f0))
```

Available tonemapping curves:

| Curve | Description |
|-------|-------------|
| `:aces` | ACES filmic (default). Industry standard, good highlight rolloff |
| `:reinhard` | Simple Reinhard L/(1+L). Soft, natural look |
| `:reinhard_extended` | Reinhard with configurable white point |
| `:uncharted2` | Uncharted 2 filmic. Good shadow detail |
| `:filmic` | Hejl-Dawson filmic. Punchy contrast |
| `nothing` | No tonemapping (linear clamp to [0,1]) |

### Exposure

The `exposure` parameter is a linear multiplier applied before tonemapping. Adjust it to control overall brightness:

```julia
postprocess!(film; exposure=0.5)   # darker
postprocess!(film; exposure=1.0)   # default
postprocess!(film; exposure=2.0)   # brighter
```

### Denoising

Hikari includes an edge-aware A-trous wavelet denoiser that operates on the HDR framebuffer. It uses normal and depth auxiliary buffers to preserve edges.

```@example postprocessing
# Render with low spp (noisy)
Hikari.clear!(film)
Hikari.VolPath(samples=4, max_depth=8)(scene, film, camera)

# Fill auxiliary buffers for edge detection
Hikari.fill_aux_buffers!(film, scene, camera)

# Denoise the framebuffer
Hikari.denoise!(film)

# Then postprocess as usual
Array(Hikari.postprocess!(film; tonemap=:aces, gamma=2.2f0))
```

`DenoiseConfig` controls the filter parameters:

```julia
config = Hikari.DenoiseConfig(
    iterations = 5,       # filter passes (more = smoother)
    sigma_color = 4.0,    # color edge threshold
    sigma_normal = 128.0, # normal edge threshold
    sigma_depth = 1.0,    # depth edge threshold
    use_variance = true,  # variance-guided filtering
)
Hikari.denoise!(film; config=config)
```

### Sensor Simulation

Sensor parameters (ISO, exposure time, white balance) are configured before rendering and baked into the framebuffer. They match pbrt-v4's `PixelSensor`.

When loading pbrt scenes, sensor settings from the scene file are applied automatically:

```julia
# pbrt scene with Film "rgb" "float iso" 200 "float whitebalance" 5000
fb = Hikari.render_pbrt("scene.pbrt")
# fb already contains sensor-calibrated linear sRGB
```

For the Julia API, the default sensor (ISO=100, exposure_time=1.0) produces physically correct output matching pbrt-v4. Sensor configuration for custom setups is done via `configure_sensor!` on the integrator state before rendering.

### Background Compositing

For scenes without full environment coverage, escaped rays produce black pixels. Use the `background` parameter to composite them:

```julia
postprocess!(film; background=RGB{Float32}(0.1, 0.1, 0.15))  # dark blue sky
```
