using GeometryBasics
using Trace
using FileIO
using Colors

function tmesh(prim, material)
    prim = prim isa Sphere ? Tesselation(prim, 64) : prim
    mesh = normal_mesh(prim)
    m = Trace.create_triangle_mesh(mesh, Trace.ShapeCore())
    return Trace.GeometricPrimitive(m, material)
end

begin
    # Create materials
    material_red = Trace.MatteMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(0.796f0, 0.235f0, 0.2f0)),
        Trace.ConstantTexture(0f0),
    )
    material_blue = Trace.MatteMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(0.251f0, 0.388f0, 0.847f0)),
        Trace.ConstantTexture(0f0),
    )
    material_white = Trace.MatteMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(1f0)),
        Trace.ConstantTexture(0f0),
    )
    mirror = Trace.MirrorMaterial(Trace.ConstantTexture(Trace.RGBSpectrum(1f0)))
    glass = Trace.GlassMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(1f0)),
        Trace.ConstantTexture(Trace.RGBSpectrum(1f0)),
        Trace.ConstantTexture(0f0),
        Trace.ConstantTexture(0f0),
        Trace.ConstantTexture(1.5f0),
        true,
    )

    # Create spheres using the new API
    # Glass sphere at (0.3, 0.11, -2.2)
    sphere1 = Sphere(Point3f(0.3, 0.11, -2.2), 0.1f0)
    primitive1 = tmesh(sphere1, glass)

    # Blue sphere at (0.2, 0.11, -2.6)
    sphere2 = Sphere(Point3f(0.2, 0.11, -2.6), 0.1f0)
    primitive2 = tmesh(sphere2, material_blue)

    # Large mirror sphere at (0.7, 0.31, -2.8)
    sphere3 = Sphere(Point3f(0.7, 0.31, -2.8), 0.3f0)
    primitive3 = tmesh(sphere3, mirror)

    # Red sphere at (0.7, 0.11, -2.3)
    sphere4 = Sphere(Point3f(0.7, 0.11, -2.3), 0.1f0)
    primitive4 = tmesh(sphere4, material_red)

    # Create floor and wall triangles
    # Floor (two triangles at z=-2, y=0)
    floor_quad = Rect3f(Vec3f(0, 0, -2), Vec3f(1, 0, -1))
    floor1 = tmesh(floor_quad, mirror)

    # Back wall (two triangles at z=-3, spanning x=[0,1], y=[0,1])
    wall_quad = Rect3f(Vec3f(0, 0, -3), Vec3f(1, 1, 0))
    wall1 = tmesh(wall_quad, material_white)

    # Build BVH with new API
    bvh = Trace.no_material_bvh([
        primitive1, primitive2, primitive3, primitive4,
        floor1, wall1,
    ])

    # Create lights
    lights = (
        Trace.PointLight(Vec3f(-1, 1, 0), Trace.RGBSpectrum(25f0)),
    )
    scene = Trace.Scene([lights...], bvh)

    # Set up camera and film
    resolution = Point2f(1024, 1024)
    filter = Trace.LanczosSincFilter(Point2f(1f0), 3f0)
    film = Trace.Film(
        resolution,
        Trace.Bounds2(Point2f(0f0), Point2f(1f0)),
        filter, 1f0, 1f0,
    )
    screen = Trace.Bounds2(Point2f(-1f0), Point2f(1f0))

    # Camera positioned to match the original perspective
    # Low angle view showing spheres with their reflections on the mirror floor
    camera = Trace.PerspectiveCamera(
        Trace.look_at(Point3f(0.15, 0.35, -1.6), Point3f(0.5, 0.25, -2.5), Vec3f(0, 1, 0)),
        screen, 0f0, 1f0, 0f0, 1f6, 55f0, film,
    )

    # Render with Whitted integrator
    integrator = Trace.WhittedIntegrator(camera, Trace.UniformSampler(3), 3)

    println("Rendering Cornell box scene...")
    @time image = Trace.integrator_threaded(integrator, scene, film, camera)

    # Clamp and save
    image_01 = map(c -> mapc(x -> clamp(x, 0f0, 1f0), c), image)
    save("cornell_box_result.png", image_01)

    println("Saved to: cornell_box_result.png")

    image_01
end

# Run the rendering:
# julia> include("cornell-box-scene.jl")
# julia> render()
