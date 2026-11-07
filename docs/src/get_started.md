## Get Started

Hikari is a physically-based, spectrally-aware wavefront path tracer. It uses Vulkan and Metal through Mantle, with Lavapipe for CPU validation, and implements the volumetric path tracing algorithm from [pbrt-v4](https://pbrt.org).

### Installation

The development branch uses backend packages that are not all registered yet.
Its `.ci` project records their published sources:

```sh
git clone --branch sd/lava-refactor https://github.com/JuliaGraphics/Hikari.jl
cd Hikari.jl
julia --project=.ci .ci/setup.jl
julia --project=.ci
```

Use the same project when running examples.

### Your First Scene

Every Hikari render follows the same five steps: build a scene, set up a camera and film, run the integrator, postprocess, and display.

```@setup getstarted
using GeometryBasics, Hikari, Mantle, ImageShow
backend = Mantle.defaultbackend()
to_mesh(prim) = normal_mesh(prim isa Sphere ? Tessellation(prim, 64) : prim)
```

```@example getstarted
# 1. Create materials
red   = Hikari.Diffuse(color = Hikari.RGBSpectrum(0.8f0, 0.25f0, 0.1f0))
white = Hikari.Diffuse(color = Hikari.RGBSpectrum(0.85f0))
gold  = Hikari.Gold(roughness=0.05f0)

# 2. Build the scene
scene = Hikari.Scene(; backend)
push!(scene, to_mesh(Sphere(Point3f(0, 0.5, 0), 0.5f0)), gold)
push!(scene, to_mesh(Rect3f(Vec3f(-3, 0, -3), Vec3f(6, 0.01, 6))), white)
push!(scene, Hikari.PointLight(Point3f(2f0, 3f0, -2f0), Hikari.RGBSpectrum(40f0)))
push!(scene, Hikari.AmbientLight(Hikari.RGBSpectrum(0.5f0)))
Hikari.sync!(scene)  # builds the BVH acceleration structure

# 3. Camera and film
film   = Hikari.Film(backend, Hikari.Film(Point2f(512, 512)))
camera = Hikari.PerspectiveCamera(
    Point3f(2f0, 1.5f0, -2.5f0), Point3f(0f0, 0.3f0, 0f0), film; fov=45f0)

# 4. Render
Hikari.clear!(film)
Hikari.VolPath(samples=32, max_depth=8)(scene, film, camera)

# 5. Postprocess and display
Array(Hikari.postprocess!(film; tonemap=:aces, exposure=1.0f0, gamma=2.2f0))
```

### Core Concepts

#### RGBSpectrum

All colors in Hikari are represented as `RGBSpectrum`. Internally the renderer works spectrally; `RGBSpectrum` values are uplifted to full spectral representations during rendering.

```julia
Hikari.RGBSpectrum(0.8f0)                    # uniform grey
Hikari.RGBSpectrum(0.8f0, 0.2f0, 0.1f0)     # red
Hikari.RGBSpectrum(1f0, 0.84f0, 0f0)        # gold-ish
```

#### Scene

A `Scene` is a collection of meshes with materials, plus light sources. Use `push!` to add objects and lights, then call `sync!` to build the acceleration structure before rendering.

```julia
scene = Hikari.Scene(; backend)
push!(scene, mesh, material)   # add a surface
push!(scene, light)            # add a light
Hikari.sync!(scene)            # required before rendering
```

#### Film and Camera

`Film` holds the image buffer. `PerspectiveCamera` describes the viewpoint.

```julia
film   = Hikari.Film(backend, Hikari.Film(Point2f(width, height)))
camera = Hikari.PerspectiveCamera(eye, lookat, film; fov=55f0)
```

Call `Hikari.clear!(film)` before each new render to reset the sample accumulator.

#### Integrator

`VolPath` is the main integrator — a wavefront volumetric path tracer with MIS and spectral hero wavelength sampling.

```julia
integrator = Hikari.VolPath(samples=64, max_depth=8)
integrator(scene, film, camera)   # renders into film
```

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `samples` | 64 | Samples per pixel |
| `max_depth` | 8 | Maximum path bounces |
| `regularize` | `true` | BSDF regularisation after first non-specular bounce (reduces fireflies) |
| `max_component_value` | 10.0 | Per-sample clamp for firefly suppression |

#### Postprocessing

`postprocess!` converts the HDR film buffer to a displayable image.

```julia
img = Hikari.postprocess!(film;
    tonemap  = :aces,    # :aces | :reinhard | :reinhard_extended | :uncharted2 | :filmic | nothing
    exposure = 1.0f0,    # linear exposure multiplier before tonemapping
    gamma    = 2.2f0,    # sRGB gamma (nothing = skip)
)
```
