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
using Adapt, AMDGPU

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

function to_mesh(prim)
    prim = prim isa Sphere ? Tesselation(prim, 64) : prim
    Raycore.TriangleMesh(normal_mesh(prim))
end

function create_scene(; glass_cat=false, backend=Raycore.KA.CPU())
    scene = Hikari.Scene(; backend)

    # === Add lights ===
    push!(scene.lights, Hikari.PointLight(RGB{Float32}(1f0, 1f0, 1f0), Point3f(3, 3, -1)))  # Key light
    push!(scene.lights, Hikari.PointLight(RGB{Float32}(5f0, 5f0, 5f0), Point3f(-3, 2, 0)))  # Fill light
    push!(scene.lights, Hikari.AmbientLight(RGB{Float32}(0.5f0, 0.7f0, 1f0)))  # Sky color

    # === Add geometry with materials ===
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
    push!(scene, create_cat_mesh(), cat_material)

    # Floor - green diffuse
    push!(scene, to_mesh(Rect3f(Vec3f(-5, -1.5, -2), Vec3f(10, 0.01, 10))),
          Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.3f0, 0.5f0, 0.3f0), σ=0f0))

    # Back wall - metallic copper
    push!(scene, to_mesh(Rect3f(Vec3f(-5, -1.5, 8), Vec3f(10, 5, 0.01))),
          Hikari.MetalMaterial(reflectance=Hikari.RGBSpectrum(0.8f0, 0.6f0, 0.5f0), roughness=0.05f0))

    # Left wall - diffuse gray-blue
    push!(scene, to_mesh(Rect3f(Vec3f(-5, -1.5, -2), Vec3f(0.01, 5, 10))),
          Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.7f0, 0.7f0, 0.8f0), σ=0f0))

    # Sphere1 - metallic silver (mirror-like)
    push!(scene, to_mesh(Sphere(Point3f(-2, -1.5 + 0.8, 2), 0.8f0)),
          Hikari.MetalMaterial(reflectance=Hikari.RGBSpectrum(0.9f0, 0.9f0, 0.9f0), roughness=0.02f0))

    # Sphere2 - semi-metallic blue (rougher)
    push!(scene, to_mesh(Sphere(Point3f(2, -1.5 + 0.6, 1), 0.6f0)),
          Hikari.MetalMaterial(reflectance=Hikari.RGBSpectrum(0.3f0, 0.6f0, 0.9f0), roughness=0.3f0))

    # Glass sphere - clear glass with slight green tint
    glass_material = Hikari.GlassMaterial(
        Kr=Hikari.RGBSpectrum(0.98f0, 1.0f0, 0.98f0),
        Kt=Hikari.RGBSpectrum(0.98f0, 1.0f0, 0.98f0),
        index=1.5f0
    )
    push!(scene, to_mesh(Sphere(Point3f(-0.8, -1.5 + 0.5, 0.5), 0.5f0)), glass_material)

    # Emissive sphere - use glass material (emissive handled via area lights)
    push!(scene, to_mesh(Sphere(Point3f(0.8, -1.5 + 0.4, 0.3), 0.4f0)), glass_material)

    # Build the acceleration structure
    Hikari.sync!(scene)

    scene
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
# Run if executed directly
# =============================================================================

# Example: Render with PhysicalWavefront on OpenCL
begin
    # Create scene directly on OpenCL backend (TLAS built on GPU)
    # backend = OpenCL.OpenCLBackend()
    backend = Hikari.KernelAbstractions.CPU()
    backend = AMDGPU.ROCBackend()
    scene = create_scene(; glass_cat=false, backend=backend)
    film, camera = create_film_and_camera(; width=1820, height=720, use_pbrt_camera=true)
    sensor = Hikari.FilmSensor(iso=10, white_balance=6500)
    integrator = Hikari.VolPath(samples=10, max_depth=8)
    # Film still needs GPU conversion, scene already on GPU
    gpu_film = Adapt.adapt(backend, film)
    Hikari.clear!(gpu_film)
    @time integrator(scene, gpu_film, camera)
    img = Hikari.postprocess!(gpu_film; sensor, exposure=0.5f0, tonemap=:aces, gamma=2.2f0)
    Array(img)
end
