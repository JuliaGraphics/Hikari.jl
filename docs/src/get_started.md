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
using Raycore
using ImageShow

# Helper to convert GeometryBasics primitives to Hikari meshes
function tmesh(prim, material)
    prim = prim isa Sphere ? Tesselation(prim, 64) : prim
    mesh = normal_mesh(prim)
    m = Raycore.TriangleMesh(mesh)
    return Hikari.GeometricPrimitive(m, material)
end

# Create a simple matte material (diffuse red)
red_material = Hikari.MatteMaterial(
    Hikari.ConstantTexture(Hikari.RGBSpectrum(0.8f0, 0.2f0, 0.2f0)),
    Hikari.ConstantTexture(0f0),
)

# White material for the ground
white_material = Hikari.MatteMaterial(
    Hikari.ConstantTexture(Hikari.RGBSpectrum(0.9f0)),
    Hikari.ConstantTexture(0f0),
)

# Create a sphere sitting on a ground plane
sphere = tmesh(Sphere(Point3f(0, 0, 0.5), 0.5f0), red_material)
ground = tmesh(Rect3f(Vec3f(-3, -3, 0), Vec3f(6, 6, 0.01)), white_material)

# Build the scene with geometry and lights
mat_scene = Hikari.MaterialScene([sphere, ground])
lights = [Hikari.PointLight(Vec3f(2, -2, 3), Hikari.RGBSpectrum(15f0))]
scene = Hikari.Scene(lights, mat_scene)

# Set up the camera
resolution = Point2f(1024)
filter = Hikari.LanczosSincFilter(Point2f(1f0), 3f0)
film = Hikari.Film(
    resolution,
    Hikari.Bounds2(Point2f(0f0), Point2f(1f0)),
    filter, 1f0, 1f0,
)
screen = Hikari.Bounds2(Point2f(-1f0), Point2f(1f0))
# Camera looking at the sphere from above and to the side
camera = Hikari.PerspectiveCamera(
    Hikari.look_at(Point3f(2, 2, 1.5), Point3f(0, 0, 0.3), Vec3f(0, 0, 1)),
    screen, 0f0, 1f0, 0f0, 1f6, 45f0, film,
)

# Render!
integrator = Hikari.WhittedIntegrator(camera, Hikari.UniformSampler(8), 5)
integrator(scene, film, camera)
```
