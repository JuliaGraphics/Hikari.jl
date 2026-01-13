# Minimal test: single triangle with explicit vertex normals to debug normal interpolation

using Hikari
using GeometryBasics
using GeometryBasics: Point3f, Vec3f, TriangleFace
using Raycore
using FileIO

# Create a single triangle mesh with explicit normals
function create_test_triangle()
    # Triangle vertices - a large triangle filling most of the view
    vertices = Point3f[
        Point3f(-1, -0.5, 0),  # Bottom left
        Point3f(1, -0.5, 0),    # Bottom right
        Point3f(0, 1, 0)        # Top center
    ]

    # Face (indices are 1-based in Julia)
    faces = [TriangleFace(1, 2, 3)]

    # Explicit normals - very different at each vertex to make interpolation obvious
    # These should create a smooth gradient across the triangle when rendered
    # Make them NOT cancel out when averaged!
    normals = Vec3f[
        Vec3f(0, 0, 1),        # Vertex 1 (bottom left): pointing straight out
        Vec3f(0.7, 0, 0.714),  # Vertex 2 (bottom right): tilted right and down
        Vec3f(0, 0.7, 0.714)   # Vertex 3 (top): tilted up
    ]

    # UVs
    uvs = Point2f[
        Point2f(0, 0),
        Point2f(1, 0),
        Point2f(0.5, 1)
    ]

    # Create mesh with explicit normals
    mesh = GeometryBasics.Mesh(vertices, faces; normal=normals, uv=uvs)

    # Convert to TriangleMesh
    tmesh = Raycore.TriangleMesh(mesh)

    # Debug: verify normals are preserved
    println("TriangleMesh normals:")
    for (i, n) in enumerate(tmesh.normals)
        println("  Vertex $i: $n")
    end
    println("Triangle indices: ", tmesh.indices)

    return tmesh
end

# Create simple scene
function create_scene()
    triangle_mesh = create_test_triangle()
    mat = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.8f0, 0.8f0, 0.8f0))

    primitives = [Hikari.GeometricPrimitive(triangle_mesh, mat)]
    mat_scene = Hikari.MaterialScene(primitives)

    # Simple directional light
    light = Hikari.DirectionalLight(
        Hikari.Transformation(),
        Hikari.RGBSpectrum(2.0f0, 2.0f0, 2.0f0),
        normalize(Vec3f(0, 0, -1))  # Light from camera direction
    )

    return Hikari.Scene((light,), mat_scene)
end

# Camera looking straight at the triangle
function create_camera_and_film(width, height)
    resolution = Point2f(width, height)
    film = Hikari.Film(resolution)

    cam_pos = Point3f(0, 0, 3)
    look_at_pos = Point3f(0, 0, 0)
    up = Vec3f(0, 1, 0)

    aspect = Float32(width) / Float32(height)
    screen_window = Raycore.Bounds2(Point2f(-aspect, -1f0), Point2f(aspect, 1f0))

    camera = Hikari.PerspectiveCamera(
        Hikari.look_at(cam_pos, look_at_pos, up),
        screen_window, 0f0, 1f0, 0f0, 1f6, 50f0, film
    )

    return film, camera
end

# Render
function render_test(; width=800, height=600, samples=16)
    println("Creating scene...")
    scene = create_scene()

    println("\nCreating camera and film...")
    film, camera = create_camera_and_film(width, height)

    println("\nRendering...")
    integrator = Hikari.VolPath(samples_per_pixel=samples, max_depth=5)
    integrator(scene, film, camera)

    println("\nPost-processing...")
    img = Hikari.postprocess!(film; exposure=1.0f0, tonemap=:aces, gamma=2.2f0)

    println("\nSaving...")
    output_path = joinpath(@__DIR__, "single_triangle_test.png")
    save(output_path, img)
    println("Saved to: $output_path")

    return img
end

# Run
img = render_test(width=800, height=600, samples=16)
