# Cat Scene GPU Example
# Test FastWavefront on AMD GPU via AMDGPU/ROCArray

using GeometryBasics
using Colors
using Hikari
using Raycore
import Makie
using AMDGPU

# =============================================================================
# Scene Setup (same as cat_scene.jl)
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

function create_scene()
    cat_mesh = create_cat_mesh()

    cat_material = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.8f0, 0.6f0, 0.4f0), σ=0f0)
    floor_material = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.3f0, 0.5f0, 0.3f0), σ=0f0)
    back_wall_material = Hikari.MetalMaterial(reflectance=Hikari.RGBSpectrum(0.8f0, 0.6f0, 0.5f0), roughness=0.05f0)
    left_wall_material = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.7f0, 0.7f0, 0.8f0), σ=0f0)
    sphere1_material = Hikari.MetalMaterial(reflectance=Hikari.RGBSpectrum(0.9f0, 0.9f0, 0.9f0), roughness=0.02f0)
    sphere2_material = Hikari.MetalMaterial(reflectance=Hikari.RGBSpectrum(0.3f0, 0.6f0, 0.9f0), roughness=0.3f0)

    cat = Hikari.GeometricPrimitive(Raycore.TriangleMesh(cat_mesh), cat_material)
    floor = tmesh(Rect3f(Vec3f(-5, -1.5, -2), Vec3f(10, 0.01, 10)), floor_material)
    back_wall = tmesh(Rect3f(Vec3f(-5, -1.5, 8), Vec3f(10, 5, 0.01)), back_wall_material)
    left_wall = tmesh(Rect3f(Vec3f(-5, -1.5, -2), Vec3f(0.01, 5, 10)), left_wall_material)
    sphere1 = tmesh(Sphere(Point3f(-2, -1.5 + 0.8, 2), 0.8f0), sphere1_material)
    sphere2 = tmesh(Sphere(Point3f(2, -1.5 + 0.6, 1), 0.6f0), sphere2_material)

    mat_scene = Hikari.MaterialScene([cat, floor, back_wall, left_wall, sphere1, sphere2])

    lights = (
        Hikari.PointLight(Point3f(3, 3, -1), Hikari.RGBSpectrum(15.0f0, 15.0f0, 15.0f0)),
        Hikari.PointLight(Point3f(-3, 2, 0), Hikari.RGBSpectrum(5.0f0, 5.0f0, 5.0f0)),
        Hikari.AmbientLight(Hikari.RGBSpectrum(0.5f0, 0.7f0, 1.0f0)),
    )

    Hikari.Scene(lights, mat_scene)
end

function create_film_and_camera(; width=720, height=400)
    resolution = Point2f(width, height)
    film = Hikari.Film(resolution)

    aspect = Float32(width) / Float32(height)
    screen_window = Hikari.Bounds2(Point2f(-aspect, -1f0), Point2f(aspect, 1f0))

    camera_pos = Point3f(0f0, -0.9f0, -2.5f0)
    camera_lookat = Point3f(0f0, -0.9f0, 10f0)
    camera_up = Vec3f(0f0, 1f0, 0f0)
    fov = 45f0

    camera = Hikari.PerspectiveCamera(
        Hikari.look_at(camera_pos, camera_lookat, camera_up),
        screen_window, 0f0, 1f0, 0f0, 1f6, fov, film)

    film, camera
end

# =============================================================================
# GPU Rendering
# =============================================================================

println("Creating scene on CPU...")
scene = create_scene()
film, camera = create_film_and_camera(; width=720, height=400)

println("Converting scene to GPU (ROCArray)...")
gpu_scene, preserve = Hikari.to_gpu(ROCArray, scene)
gpu_film = Hikari.to_gpu(ROCArray, film)

println("Scene aggregate type: $(typeof(gpu_scene.aggregate))")
println("Film framebuffer type: $(typeof(gpu_film.framebuffer))")

println("\nCreating FastWavefront integrator...")
integrator = Hikari.FastWavefront(samples_per_pixel=4)

println("Clearing film...")
Hikari.clear!(gpu_film)

println("\nRendering on GPU...")
@time integrator(gpu_scene, gpu_film, camera)

println("\nCopying result back to CPU...")
cpu_framebuffer = Array(gpu_film.framebuffer)

println("Done! Framebuffer size: $(size(cpu_framebuffer))")
