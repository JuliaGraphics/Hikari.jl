using GeometryBasics
using Trace
using FileIO
using Colors

function tmesh(prim, material)
    prim = prim isa Sphere ? Tesselation(prim, 64) : prim
    mesh = normal_mesh(prim)
    m = Raycore.TriangleMesh(mesh)
    return Trace.GeometricPrimitive(m, material)
end

function render()
    material_red = Trace.MatteMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(0.796f0, 0.235f0, 0.2f0)),
        Trace.ConstantTexture(0f0),
    )
    material_white = Trace.MatteMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(1f0)),
        Trace.ConstantTexture(0f0),
    )

    # Create sphere using GeometryBasics
    sphere3 = Sphere(Point3f(0.7, 0.31, -2.8), 0.3f0)
    primitive3 = tmesh(sphere3, material_red)

    # Create floor and wall
    floor_quad = Rect3f(Vec3f(0, 0, -2), Vec3f(1, 0, -1))
    floor_primitive = tmesh(floor_quad, material_white)

    wall_quad = Rect3f(Vec3f(0, 0, -3), Vec3f(1, 1, 0))
    wall_primitive = tmesh(wall_quad, material_white)

    bvh = Trace.no_material_bvh([
        primitive3,
        floor_primitive,
        wall_primitive,
    ])

    lights = (
        Trace.PointLight(Vec3f(-1, 1, 0), Trace.RGBSpectrum(25f0)),
    )
    scene = Trace.Scene([lights...], bvh)

    resolution = Point2f(1024 ÷ 3)
    filter = Trace.LanczosSincFilter(Point2f(1f0), 3f0)
    film = Trace.Film(
        resolution,
        Trace.Bounds2(Point2f(0f0), Point2f(1f0)),
        filter, 1f0, 1f0,
    )
    screen = Trace.Bounds2(Point2f(-1f0), Point2f(1f0))
    camera = Trace.PerspectiveCamera(
        Trace.look_at(Point3f(0, 15, 50), Point3f(0, 0, -2), Vec3f(0, 1, 0)),
        screen, 0f0, 1f0, 0f0, 1f6, 90f0, film,
    )

    # Use WhittedIntegrator for faster rendering
    integrator = Trace.WhittedIntegrator(camera, Trace.UniformSampler(8), 8)
    Trace.integrator_threaded(integrator, scene, film, camera)

    # Save the result
    image_01 = map(c -> mapc(x -> clamp(x, 0f0, 1f0), c), film.framebuffer)
    filename = "shadows-sppm-$(Int64(resolution[1]))x$(Int64(resolution[2]))_redSphere.png"
    save(joinpath(@__DIR__, filename), image_01)

    return film.framebuffer
end

render()
