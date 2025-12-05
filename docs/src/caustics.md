## Caustics with SPPM

This example demonstrates the Stochastic Progressive Photon Mapping (SPPM) integrator, which excels at rendering caustics - the bright patterns created when light focuses through transparent or reflective objects.

SPPM is particularly effective for scenes with:
- Glass objects that refract light
- Water surfaces creating light patterns
- Metallic surfaces focusing reflections

```@example caustics
using GeometryBasics
using Hikari
using Raycore
using FileIO
using ImageShow

# Load the caustic glass model
model_path = joinpath(@__DIR__, "assets", "models", "caustic-glass.ply")
glass_mesh = load(model_path)

# Get mesh bounds to position camera and lights correctly
bounds = extrema(Rect3f(decompose(Point3f, glass_mesh)))
mesh_center = (bounds[1] .+ bounds[2]) ./ 2
floor_y = bounds[1][2]

# Create glass material with index of refraction 1.5 (typical glass)
glass = Hikari.GlassMaterial(
    Hikari.ConstantTexture(Hikari.RGBSpectrum(1f0)),      # Reflectance
    Hikari.ConstantTexture(Hikari.RGBSpectrum(1f0)),      # Transmittance
    Hikari.ConstantTexture(0.0f0),                          # u_roughness (0 = perfectly smooth)
    Hikari.ConstantTexture(0.0f0),                          # v_roughness
    Hikari.ConstantTexture(1.5f0),                        # Index of refraction
    true,                                                  # Remap roughness
)

# White plastic floor to catch the caustic patterns
floor_material = Hikari.PlasticMaterial(
    Hikari.ConstantTexture(Hikari.RGBSpectrum(0.8f0)),    # Diffuse color
    Hikari.ConstantTexture(Hikari.RGBSpectrum(0.1f0)),    # Specular color
    Hikari.ConstantTexture(0.01f0),                       # Roughness
    true,
)

# Create primitives
glass_nm = normal_mesh(glass_mesh)
glass_obj = Hikari.GeometricPrimitive(Raycore.TriangleMesh(glass_nm), glass)

# Floor positioned at the bottom of the glass mesh
floor_rect = Rect3f(Vec3f(mesh_center[1]-5, floor_y - 0.01f0, mesh_center[3]-5), Vec3f(10, 0.01, 10))
floor_nm = normal_mesh(floor_rect)
floor_obj = Hikari.GeometricPrimitive(Raycore.TriangleMesh(floor_nm), floor_material)

# Create a spotlight above the glass pointing down through it
# This creates focused caustic patterns on the floor
# SpotLight(position, target, intensity, cone_angle, falloff_start)
light_pos = Point3f(mesh_center[1] - 2, mesh_center[2] + 2, mesh_center[3] - 6)
light_target = Point3f(mesh_center...)
spotlight = Hikari.SpotLight(light_pos, light_target, Hikari.RGBSpectrum(60f0), 30f0, 15f0)

# Build scene
mat_scene = Hikari.MaterialScene([glass_obj, floor_obj])
scene = Hikari.Scene([spotlight], mat_scene)

# Set up camera looking at the glass and floor from an angle
resolution = Point2f(512)
filter = Hikari.LanczosSincFilter(Point2f(1f0), 3f0)
film = Hikari.Film(
    resolution,
    Hikari.Bounds2(Point2f(0f0), Point2f(1f0)),
    filter, 1f0, 1f0,
)
screen = Hikari.Bounds2(Point2f(-1f0), Point2f(1f0))

# Position camera to see both the glass object and the caustic pattern on the floor
cam_pos = Point3f(mesh_center[1] + 3, mesh_center[2] + 4f0, mesh_center[3] + 4)
cam_look_at = Point3f(mesh_center[1] - 3, floor_y - 5, mesh_center[3])
camera = Hikari.PerspectiveCamera(
    Hikari.look_at(cam_pos, cam_look_at, Vec3f(0, 1, 0)),
    screen, 0f0, 1f0, 0f0, 1f6, 60f0, film,
)

# Render with SPPM - this captures caustics that Whitted ray tracing misses
# Parameters: camera, initial_search_radius, max_depth, n_iterations
integrator = Hikari.SPPMIntegrator(camera, 0.075f0, 5, 20, film)
integrator(scene, film)
```

### Why SPPM for Caustics?

Standard Whitted-style ray tracing struggles with caustics because it traces rays from the camera and relies on direct light sampling. Caustics require tracing light paths that refract through glass and then hit diffuse surfaces - these "specular-diffuse-specular" paths are difficult to find with camera-based ray tracing.

SPPM solves this by:
1. **Photon emission**: Shooting photons from light sources through the scene
2. **Photon gathering**: Collecting photons near visible points
3. **Progressive refinement**: Iteratively reducing the search radius for sharper results

The `initial_search_radius` parameter controls the initial photon gathering radius - smaller values give sharper but noisier results, while larger values give smoother but blurrier caustics.
