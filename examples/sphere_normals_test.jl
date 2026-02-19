# Test scene to verify smooth normal interpolation on spheres
# This creates a simple scene with spheres to check if normals are properly interpolated

using Hikari
using GeometryBasics
using GeometryBasics: Point3f, Vec3f, normal_mesh, Tesselation, Sphere
using Raycore
using FileIO
using LinearAlgebra: normalize

# Helper to create mesh from GeometryBasics primitive
function create_sphere_mesh(center, radius, tesselation=64)
    sphere = Sphere(Point3f(center), Float32(radius))
    mesh = normal_mesh(Tesselation(sphere, tesselation))
    tmesh = Raycore.TriangleMesh(mesh)

    # Debug: check first triangle normals
    if center[1] ≈ -1.5f0
        println("Debug: First triangle of sphere at $center:")
        f_idx = 1
        idx1 = tmesh.indices[f_idx]
        idx2 = tmesh.indices[f_idx + 1]
        idx3 = tmesh.indices[f_idx + 2]
        n1 = tmesh.normals[idx1]
        n2 = tmesh.normals[idx2]
        n3 = tmesh.normals[idx3]
        println("  Indices: $idx1, $idx2, $idx3")
        println("  Normal 1: $n1")
        println("  Normal 2: $n2")
        println("  Normal 3: $n3")
        println("  All same? ", n1 ≈ n2 && n2 ≈ n3)
    end

    return tmesh
end

# Create scene with three spheres
function create_sphere_scene()
    # Materials - different colors for each sphere
    mat1 = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.8f0, 0.2f0, 0.2f0))  # Red
    mat2 = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.2f0, 0.8f0, 0.2f0))  # Green
    mat3 = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.2f0, 0.2f0, 0.8f0))  # Blue

    # Floor
    floor_mat = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.7f0, 0.7f0, 0.7f0))
    floor_mesh = Raycore.TriangleMesh(normal_mesh(Rect3f(Vec3f(-5, -1, -5), Vec3f(10, 0.1f0, 10))))

    # Create spheres at different positions
    sphere1 = create_sphere_mesh(Vec3f(-1.5, 0.5, 0), 0.8f0)
    sphere2 = create_sphere_mesh(Vec3f(0, 0.5, 0), 0.8f0)
    sphere3 = create_sphere_mesh(Vec3f(1.5, 0.5, 0), 0.8f0)

    # Create primitives
    primitives = [
        Hikari.GeometricPrimitive(floor_mesh, floor_mat),
        Hikari.GeometricPrimitive(sphere1, mat1),
        Hikari.GeometricPrimitive(sphere2, mat2),
        Hikari.GeometricPrimitive(sphere3, mat3),
    ]

    # Create material scene
    mat_scene = Hikari.MaterialScene(primitives)

    # Create directional light (like sun)
    light_dir = normalize(Vec3f(-1, -1.5, -0.5))
    light = Hikari.DirectionalLight(
        Hikari.Transformation(),
        Hikari.RGBSpectrum(3.0f0, 3.0f0, 3.0f0),
        light_dir
    )

    return Hikari.Scene((light,), mat_scene)
end

# Create camera and film
function create_camera_and_film(width, height)
    resolution = Point2f(width, height)
    film = Hikari.Film(resolution)

    # Camera positioned to see all three spheres
    cam_pos = Point3f(0, 1.5, 4)
    look_at_pos = Point3f(0, 0.5, 0)
    up = Vec3f(0, 1, 0)

    aspect = Float32(width) / Float32(height)
    screen_window = Raycore.Bounds2(Point2f(-aspect, -1f0), Point2f(aspect, 1f0))

    camera = Hikari.PerspectiveCamera(
        Hikari.look_at(cam_pos, look_at_pos, up),
        screen_window, 0f0, 1f0, 0f0, 1f6, 40f0, film
    )

    return film, camera
end

# Main rendering function
function render_sphere_test(; width=800, height=600, samples=16)
    println("Creating scene...")
    scene = create_sphere_scene()

    println("Creating camera and film...")
    film, camera = create_camera_and_film(width, height)

    println("Creating integrator...")
    integrator = Hikari.VolPath(samples_per_pixel=samples, max_depth=5)

    println("Rendering with VolPath...")
    integrator(scene, film, camera)

    # Also try Physical integrator for comparison
    println("\nTrying Physical integrator...")
    film2 = Hikari.Film(Point2f(width, height))
    camera2 = Hikari.PerspectiveCamera(
        Hikari.look_at(Point3f(0, 1.5, 4), Point3f(0, 0.5, 0), Vec3f(0, 1, 0)),
        Raycore.Bounds2(Point2f(-Float32(width/height), -1f0), Point2f(Float32(width/height), 1f0)),
        0f0, 1f0, 0f0, 1f6, 40f0, film2
    )
    integrator2 = Hikari.PhysicalWavefront(samples_per_pixel=samples, max_depth=5)
    integrator2(scene, film2, camera2)

    img2 = Hikari.postprocess!(film2; exposure=1.0f0, tonemap=:aces, gamma=2.2f0)
    output_path2 = joinpath(@__DIR__, "sphere_normals_test_physical.png")
    save(output_path2, img2)
    println("Saved Physical render to: $output_path2")

    println("Post-processing...")
    img = Hikari.postprocess!(film; exposure=1.0f0, tonemap=:aces, gamma=2.2f0)

    println("Saving image...")
    output_path = joinpath(@__DIR__, "sphere_normals_test.png")
    save(output_path, img)
    println("Saved to: $output_path")

    return img
end

# Run the test
img = render_sphere_test(width=800, height=600, samples=16)
