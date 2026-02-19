## Get Started

Hikari is a physically-based ray tracer for Julia. This guide shows you how to create your first rendered scene.

### Basic Scene Setup

A minimal Hikari scene requires:
1. **Geometry** - Meshes to render (spheres, boxes, triangles)
2. **Materials** - Surface properties (matte, mirror, glass)
3. **Lights** - Light sources to illuminate the scene
4. **Camera** - Viewpoint and image settings

```@example getstarted
using GeometryBasics
using Hikari
using ImageShow

# Helper to tessellate GeometryBasics primitives into triangle meshes
to_mesh(prim) = normal_mesh(prim isa Sphere ? Tesselation(prim, 64) : prim)

# Create materials using keyword constructors
red_material = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.8f0, 0.2f0, 0.2f0))
white_material = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.9f0))

# Build the scene using the push! API
scene = Hikari.Scene()
push!(scene, to_mesh(Sphere(Point3f(0, 0.5, 0), 0.5f0)), red_material)
push!(scene, to_mesh(Rect3f(Vec3f(-3, 0, -3), Vec3f(6, 0.01, 6))), white_material)

# Add lights
push!(scene, Hikari.PointLight(Point3f(2f0, 3f0, -2f0), Hikari.RGBSpectrum(15f0)))
push!(scene, Hikari.AmbientLight(Hikari.RGBSpectrum(0.1f0)))

# Finalize the acceleration structure
Hikari.sync!(scene)

# Set up camera and film
resolution = Point2f(512, 512)
film = Hikari.Film(resolution)
camera = Hikari.PerspectiveCamera(
    Point3f(2f0, 2f0, -2f0), Point3f(0f0, 0.3f0, 0f0), film; fov=45f0,
)
Hikari.clear!(film)

# Render with the VolPath integrator
integrator = Hikari.VolPath(samples=16, max_depth=5)
integrator(scene, film, camera)

# Postprocess and display
img = Hikari.postprocess!(film; exposure=1.0f0, tonemap=:aces, gamma=2.2f0)
Array(img)
```
