## Shadows

This example demonstrates a Cornell box-style scene with multiple spheres, different materials (matte, mirror), and shadow rendering.

```@example shadows
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

# Helper to create a sphere resting on the ground (z=0)
LowSphere(radius, contact=Point3f(0)) = Sphere(contact .+ Point3f(0, 0, radius), radius)

# Define materials - colorful Cornell box style
material_white = Hikari.MatteMaterial(
    Hikari.ConstantTexture(Hikari.RGBSpectrum(0.9f0)),
    Hikari.ConstantTexture(0f0),
)
material_red = Hikari.MatteMaterial(
    Hikari.ConstantTexture(Hikari.RGBSpectrum(0.8f0, 0.2f0, 0.2f0)),
    Hikari.ConstantTexture(0f0),
)
material_green = Hikari.MatteMaterial(
    Hikari.ConstantTexture(Hikari.RGBSpectrum(0.2f0, 0.8f0, 0.2f0)),
    Hikari.ConstantTexture(0f0),
)
material_blue = Hikari.MatteMaterial(
    Hikari.ConstantTexture(Hikari.RGBSpectrum(0.3f0, 0.5f0, 0.9f0)),
    Hikari.ConstantTexture(0f0),
)
mirror = Hikari.MirrorMaterial(Hikari.ConstantTexture(Hikari.RGBSpectrum(0.95f0)))

# Create spheres with different materials
s1 = tmesh(LowSphere(0.5f0), mirror)  # Center mirror sphere
s2 = tmesh(LowSphere(0.3f0, Point3f(0.8, 0.3, 0)), material_blue)
s3 = tmesh(LowSphere(0.3f0, Point3f(-0.8, 0.3, 0)), material_red)
s4 = tmesh(LowSphere(0.25f0, Point3f(0, 0.9, 0)), material_white)

# Create room geometry (Cornell box style)
ground = tmesh(Rect3f(Vec3f(-2, -2, 0), Vec3f(4, 4, 0.01)), material_white)
back = tmesh(Rect3f(Vec3f(-2, -2, 0), Vec3f(4, 0.01, 3)), material_white)
left = tmesh(Rect3f(Vec3f(-2, -2, 0), Vec3f(0.01, 4, 3)), material_red)
right = tmesh(Rect3f(Vec3f(2, -2, 0), Vec3f(0.01, 4, 3)), material_green)

# Build scene with multiple lights for interesting shadows
mat_scene = Hikari.MaterialScene([s1, s2, s3, s4, ground, back, left, right])
lights = [
    Hikari.PointLight(Vec3f(0, 0, 2.5), Hikari.RGBSpectrum(12f0)),  # Main overhead light
    Hikari.PointLight(Vec3f(-1, 1.5, 1.5), Hikari.RGBSpectrum(4f0)),  # Fill light
]
scene = Hikari.Scene(lights, mat_scene)

# Set up camera and film
resolution = Point2f(1024)
filter = Hikari.LanczosSincFilter(Point2f(1f0), 3f0)
film = Hikari.Film(
    resolution,
    Hikari.Bounds2(Point2f(0f0), Point2f(1f0)),
    filter, 1f0, 1f0,
)
screen = Hikari.Bounds2(Point2f(-1f0), Point2f(1f0))
camera = Hikari.PerspectiveCamera(
    Hikari.look_at(Point3f(0, 3, 1.5), Point3f(0, 0, 0.4), Vec3f(0, 0, 1)),
    screen, 0f0, 1f0, 0f0, 1f6, 50f0, film,
)

# Render with Whitted integrator
integrator = Hikari.WhittedIntegrator(camera, Hikari.UniformSampler(16), 10)
integrator(scene, film, camera)
```
