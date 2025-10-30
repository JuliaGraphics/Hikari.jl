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

    # Create spheres using GeometryBasics
    sphere1 = Sphere(Point3f(0.3, 0.11, -2.2), 0.1f0)
    primitive1 = tmesh(sphere1, glass)

    sphere2 = Sphere(Point3f(0.2, 0.11, -2.6), 0.1f0)
    primitive2 = tmesh(sphere2, material_blue)

    sphere3 = Sphere(Point3f(0.7, 0.31, -2.8), 0.3f0)
    primitive3 = tmesh(sphere3, mirror)

    sphere4 = Sphere(Point3f(0.7, 0.11, -2.3), 0.1f0)
    primitive4 = tmesh(sphere4, material_red)

    # Create floor and wall
    floor_quad = Rect3f(Vec3f(0, 0, -2), Vec3f(1, 0, -1))
    floor1 = tmesh(floor_quad, mirror)
    floor2 = tmesh(floor_quad, mirror)  # Two triangles

    wall_quad = Rect3f(Vec3f(0, 0, -3), Vec3f(1, 1, 0))
    wall1 = tmesh(wall_quad, material_white)
    wall2 = tmesh(wall_quad, material_white)  # Two triangles

    bvh = Trace.no_material_bvh([
        primitive1, primitive2, primitive3, primitive4,
        floor1, wall1,
    ])

    lights = (
        Trace.PointLight(Vec3f(-1, 1, 0), Trace.RGBSpectrum(25f0)),
    )
    scene = Trace.Scene([lights...], bvh)

    resolution = Point2f(1024)
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
    filename = "shadows-sppm-$(Int64(resolution[1]))x$(Int64(resolution[2])).png"
    save(joinpath(@__DIR__, filename), image_01)

    return film.framebuffer
end

render()
