# Cat Scene Example
# Demonstrates the same scene with different integrators
# All integrators use the same interface: integrator(scene, film, camera)
#
# This scene is designed to be comparable with PbrtWavefront/examples/cat_scene.jl

using GeometryBasics
using Colors
using Hikari
using Raycore
import Makie
using Raycore: to_gpu

# =============================================================================
# Scene Setup (shared by all integrators)
# =============================================================================

function create_cat_mesh()
    cat_mesh = Makie.loadasset("cat.obj")
    angle = deg2rad(150f0)
    rotation = Makie.Quaternionf(0, sin(angle/2), 0, cos(angle/2))
    rotated_coords = [rotation * Point3f(v) for v in coordinates(cat_mesh)]

    cat_bbox = Rect3f(rotated_coords)
    floor_y = -1.5f0
    cat_offset = Vec3f(0, floor_y - cat_bbox.origin[2], 0)

    GeometryBasics.normal_mesh(
        [v + cat_offset for v in rotated_coords],
        GeometryBasics.faces(cat_mesh)
    )
end

function tmesh(prim, material)
    prim = prim isa Sphere ? Tesselation(prim, 64) : prim
    mesh = normal_mesh(prim)
    Hikari.GeometricPrimitive(Raycore.TriangleMesh(mesh), material)
end

"""
Create scene matching PbrtWavefront/examples/cat_scene.jl for comparison.

Materials:
- Cat: warm diffuse (orange)
- Floor: green diffuse
- Back wall: metallic copper
- Left wall: diffuse gray-blue
- Sphere1 (left, big): metallic silver mirror
- Sphere2 (right, small): semi-metallic blue
- Glass sphere (front left): clear glass
- Emissive sphere (front right): bright orange glow
"""
function create_scene(; glass_cat=false)
    cat_mesh = create_cat_mesh()

    # Materials matching PbrtWavefront cat_scene
    # Cat - warm diffuse or glass
    cat_material = if glass_cat
        Hikari.GlassMaterial(
            Kr=Hikari.RGBSpectrum(0.95f0, 1.0f0, 0.95f0),
            Kt=Hikari.RGBSpectrum(0.95f0, 1.0f0, 0.95f0),
            index=1.5f0
        )
    else
        Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.8f0, 0.6f0, 0.4f0), σ=0f0)
    end

    # Floor - green diffuse
    floor_material = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.3f0, 0.5f0, 0.3f0), σ=0f0)

    # Back wall - metallic copper
    back_wall_material = Hikari.MetalMaterial(
        reflectance=Hikari.RGBSpectrum(0.8f0, 0.6f0, 0.5f0),
        roughness=0.05f0
    )

    # Left wall - diffuse gray-blue
    left_wall_material = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.7f0, 0.7f0, 0.8f0), σ=0f0)

    # Sphere1 - metallic silver (mirror-like)
    sphere1_material = Hikari.MetalMaterial(
        reflectance=Hikari.RGBSpectrum(0.9f0, 0.9f0, 0.9f0),
        roughness=0.02f0
    )

    # Sphere2 - semi-metallic blue (rougher)
    sphere2_material = Hikari.MetalMaterial(
        reflectance=Hikari.RGBSpectrum(0.3f0, 0.6f0, 0.9f0),
        roughness=0.3f0
    )

    # Glass sphere - clear glass with slight green tint
    glass_sphere_material = Hikari.GlassMaterial(
        Kr=Hikari.RGBSpectrum(0.98f0, 1.0f0, 0.98f0),
        Kt=Hikari.RGBSpectrum(0.98f0, 1.0f0, 0.98f0),
        index=1.5f0
    )

    # Emissive sphere - bright orange glow
    emissive_sphere_material = Hikari.EmissiveMaterial(
        Le=Hikari.RGBSpectrum(1.0f0, 0.5f0, 0.1f0),
        scale=15.0f0
    )

    # Create primitives
    cat = Hikari.GeometricPrimitive(Raycore.TriangleMesh(cat_mesh), cat_material)
    floor = tmesh(Rect3f(Vec3f(-5, -1.5, -2), Vec3f(10, 0.01, 10)), floor_material)
    back_wall = tmesh(Rect3f(Vec3f(-5, -1.5, 8), Vec3f(10, 5, 0.01)), back_wall_material)
    left_wall = tmesh(Rect3f(Vec3f(-5, -1.5, -2), Vec3f(0.01, 5, 10)), left_wall_material)
    sphere1 = tmesh(Sphere(Point3f(-2, -1.5 + 0.8, 2), 0.8f0), sphere1_material)
    sphere2 = tmesh(Sphere(Point3f(2, -1.5 + 0.6, 1), 0.6f0), sphere2_material)

    # New spheres in front of the cat (matching PbrtWavefront)
    glass_sphere = tmesh(Sphere(Point3f(-0.8, -1.5 + 0.5, 0.5), 0.5f0), glass_sphere_material)
    emissive_sphere = tmesh(Sphere(Point3f(0.8, -1.5 + 0.4, 0.3), 0.4f0), emissive_sphere_material)

    mat_scene = Hikari.MaterialScene([cat, floor, back_wall, left_wall, sphere1, sphere2, glass_sphere, emissive_sphere])

    # Lights matching PbrtWavefront
    lights = (
        Hikari.PointLight(Point3f(3, 3, -1), Hikari.RGBSpectrum(15.0f0, 15.0f0, 15.0f0)),  # Key light
        Hikari.PointLight(Point3f(-3, 2, 0), Hikari.RGBSpectrum(5.0f0, 5.0f0, 5.0f0)),     # Fill light
        Hikari.AmbientLight(Hikari.RGBSpectrum(0.5f0, 0.7f0, 1.0f0)),  # Sky color for escaped rays
    )

    Hikari.Scene(lights, mat_scene)
end

"""
Create camera matching PbrtWavefront cat_scene.jl for comparison.

Uses the same position, lookat, and fov as PbrtWavefront:
- pos: (0, -0.9, -2.5)
- lookat: (0, -0.9, 10)
- fov: 45°
"""
function create_film_and_camera(; width=720, height=400, use_pbrt_camera=true)
    resolution = Point2f(width, height)
    film = Hikari.Film(resolution)

    aspect = Float32(width) / Float32(height)
    screen_window = Hikari.Bounds2(Point2f(-aspect, -1f0), Point2f(aspect, 1f0))

    if use_pbrt_camera
        # Camera matching PbrtWavefront example
        camera_pos = Point3f(0f0, -0.9f0, -2.5f0)
        camera_lookat = Point3f(0f0, -0.9f0, 10f0)
        camera_up = Vec3f(0f0, 1f0, 0f0)
    else
        # Original Hikari camera position
        camera_pos = Point3f(3.92f0, 0.43f0, -1.76f0)
        camera_lookat = Point3f(0.7f0, -0.57f0, 1.5f0)
        camera_up = Vec3f(0.25f0, 0.92f0, -0.3f0)
    end

    fov = 45f0
    camera = Hikari.PerspectiveCamera(
        Hikari.look_at(camera_pos, camera_lookat, camera_up),
        screen_window, 0f0, 1f0, 0f0, 1f6, fov, film)

    film, camera
end

# =============================================================================
# Render Functions
# =============================================================================

"""
    render_with_integrator(integrator; width=720, height=400, glass_cat=false)

Render the cat scene with a specified integrator.

# Example usage:
```julia
# FastWavefront (simple, fast)
img = render_with_integrator(Hikari.FastWavefront(samples_per_pixel=16))

# Whitted (classic ray tracing)
img = render_with_integrator(Hikari.WhittedIntegrator(Hikari.UniformSampler(16), 5))

# PhysicalWavefront (spectral path tracing)
img = render_with_integrator(Hikari.PhysicalWavefront(samples_per_pixel=64, max_depth=8))
```
"""
function render_with_integrator(integrator; width=720, height=400, glass_cat=false, use_pbrt_camera=true)
    scene = create_scene(; glass_cat=glass_cat)
    film, camera = create_film_and_camera(; width=width, height=height, use_pbrt_camera=use_pbrt_camera)

    Hikari.clear!(film)
    @time integrator(scene, film, camera)

    # Apply postprocessing
    Hikari.postprocess!(film; exposure=1.0f0, tonemap=:aces, gamma=1.2f0)
    film.postprocess
end

# =============================================================================
# Run if executed directly
# =============================================================================

using pocl_jll, OpenCL
cl.device!(cl.devices(cl.platforms()[1])[1])

# Example: Render with PhysicalWavefront

begin
    scene = create_scene(; glass_cat=false)
    film, camera = create_film_and_camera(; width=720, height=720, use_pbrt_camera=true)
    # Choose integrator:
    integrator = Hikari.FastWavefront(samples=8)
    integrator = Hikari.Whitted(samples=8)
    integrator = Hikari.PhysicalWavefront(samples_per_pixel=200, max_depth=8)
    gpu_film = to_gpu(Array, film)
    gpu_scene = to_gpu(Array, scene)
    Hikari.clear!(gpu_film)
    integrator(gpu_scene, gpu_film, camera)
    Hikari.postprocess!(gpu_film; exposure=2.0f0, tonemap=:aces, gamma=1.2f0)
    Array(gpu_film.postprocess)
end
