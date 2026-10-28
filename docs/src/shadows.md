## Rendering

A complete walkthrough of a Cornell box scene, demonstrating shadows, inter-reflections, caustics through glass, and multiple material types.

```@setup rendering
using GeometryBasics, Hikari, ImageShow
to_mesh(prim) = normal_mesh(prim isa Sphere ? Tesselation(prim, 64) : prim)
```

### Cornell Box

```@example rendering
# --- Materials ---
white  = Hikari.Diffuse(color = Hikari.RGBSpectrum(0.88f0))
red    = Hikari.Diffuse(color = Hikari.RGBSpectrum(0.65f0, 0.1f0, 0.1f0))
green  = Hikari.Diffuse(color = Hikari.RGBSpectrum(0.1f0, 0.6f0, 0.1f0))
mirror = Hikari.Glossy(roughness = 0, color = Hikari.RGBSpectrum(0.95f0))
glass  = Hikari.Glass(ior = 1.5f0)
gold   = Hikari.Gold(roughness=0.04f0)

# --- Scene ---
scene = Hikari.Scene()

# Spheres
push!(scene, to_mesh(Sphere(Point3f(0f0, 0.5f0, 0f0),    0.5f0)), mirror)
push!(scene, to_mesh(Sphere(Point3f(0.8f0, 0.3f0, 0.3f0), 0.3f0)), glass)
push!(scene, to_mesh(Sphere(Point3f(-0.8f0, 0.3f0, 0.3f0), 0.3f0)), gold)

# Room geometry
push!(scene, to_mesh(Rect3f(Vec3f(-2, 0, -2),      Vec3f(4, 0.01, 4))),   white)  # floor
push!(scene, to_mesh(Rect3f(Vec3f(-2, 3-0.01, -2), Vec3f(4, 0.01, 4))),   white)  # ceiling
push!(scene, to_mesh(Rect3f(Vec3f(-2, 0, 2-0.01),  Vec3f(4, 3, 0.01))),   white)  # back wall
push!(scene, to_mesh(Rect3f(Vec3f(-2, 0, -2),      Vec3f(0.01, 3, 4))),   red)    # left wall
push!(scene, to_mesh(Rect3f(Vec3f(2-0.01, 0, -2),  Vec3f(0.01, 3, 4))),   green)  # right wall

# Ceiling area light
push!(scene, to_mesh(Rect3f(Vec3f(-0.4f0, 2.98f0, -0.4f0), Vec3f(0.8f0, 0.01f0, 0.8f0))),
      Hikari.Emission(color = (3.0, 3.0, 2.85), strength = 1f0, two_sided=true))

Hikari.sync!(scene)

# --- Camera ---
film   = Hikari.Film(Point2f(512, 512))
camera = Hikari.PerspectiveCamera(
    Point3f(0f0, 1.5f0, -3.2f0), Point3f(0f0, 1f0, 0f0), film; fov=48f0)

# --- Render ---
Hikari.clear!(film)
Hikari.VolPath(samples=128, max_depth=12)(scene, film, camera)

Array(Hikari.postprocess!(film; tonemap=:aces, exposure=1.2f0, gamma=2.2f0))
```

### GPU Rendering

To render on GPU, pass a GPU backend when constructing the `Scene` and ensure your geometry and film are on device memory.

```julia
import CUDA, KernelAbstractions as KA

scene = Hikari.Scene(; backend=KA.CUDABackend())
# push! geometry and lights as normal ...
Hikari.sync!(scene)

film = Hikari.Film(Point2f(1920, 1080))
Hikari.VolPath(samples=512, max_depth=12)(scene, film, camera)
```

The same code path is used for CPU and GPU — only the `backend` keyword differs.

### Loading pbrt Scenes

Hikari can load and render scenes in the [pbrt-v4 scene format](https://pbrt.org/fileformat-v4):

```julia
fb = Hikari.render_pbrt("scene.pbrt"; samples=256, max_depth=10)
```

The returned framebuffer is an `Array{RGBA{Float32}}` ready for display or saving:

```julia
using FileIO
FileIO.save("output.exr", fb)
```
