## Shadows

This example demonstrates a Cornell box-style scene with multiple spheres, different materials (matte, mirror, conductor), and shadow rendering.

```@example shadows
using GeometryBasics
using Hikari
using ImageShow

# Helper to tessellate primitives
to_mesh(prim) = normal_mesh(prim isa Sphere ? Tesselation(prim, 64) : prim)

# Define materials
material_white = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.9f0))
material_red = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.8f0, 0.2f0, 0.2f0))
material_green = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.2f0, 0.8f0, 0.2f0))
material_blue = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.3f0, 0.5f0, 0.9f0))
mirror = Hikari.MirrorMaterial(Kr=Hikari.RGBSpectrum(0.95f0))

# Build scene
scene = Hikari.Scene()

# Spheres with different materials
push!(scene, to_mesh(Sphere(Point3f(0, 0.5, 0), 0.5f0)), mirror)        # Center mirror
push!(scene, to_mesh(Sphere(Point3f(0.8, 0.3, 0.3), 0.3f0)), material_blue)
push!(scene, to_mesh(Sphere(Point3f(-0.8, 0.3, 0.3), 0.3f0)), material_red)
push!(scene, to_mesh(Sphere(Point3f(0, 0.25, 0.9), 0.25f0)), material_white)

# Room geometry (Cornell box style, Y-up)
push!(scene, to_mesh(Rect3f(Vec3f(-2, 0, -2), Vec3f(4, 0.01, 4))), material_white)   # Floor
push!(scene, to_mesh(Rect3f(Vec3f(-2, 0, 2 - 0.01), Vec3f(4, 3, 0.01))), material_white)  # Back wall
push!(scene, to_mesh(Rect3f(Vec3f(-2, 0, -2), Vec3f(0.01, 3, 4))), material_red)     # Left wall
push!(scene, to_mesh(Rect3f(Vec3f(2 - 0.01, 0, -2), Vec3f(0.01, 3, 4))), material_green)  # Right wall

# Lights
push!(scene, Hikari.PointLight(Point3f(0f0, 2.5f0, 0f0), Hikari.RGBSpectrum(12f0)))
push!(scene, Hikari.PointLight(Point3f(-1f0, 1.5f0, -1.5f0), Hikari.RGBSpectrum(4f0)))

Hikari.sync!(scene)

# Camera and film
resolution = Point2f(512, 512)
film = Hikari.Film(resolution)
camera = Hikari.PerspectiveCamera(
    Point3f(0f0, 1.5f0, -3f0), Point3f(0f0, 0.4f0, 0f0), film; fov=50f0,
)
Hikari.clear!(film)

# Render
integrator = Hikari.VolPath(samples=16, max_depth=10)
integrator(scene, film, camera)

img = Hikari.postprocess!(film; exposure=1.0f0, tonemap=:aces, gamma=2.2f0)
Array(img)
```
