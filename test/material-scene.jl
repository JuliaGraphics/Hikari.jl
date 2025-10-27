using GeometryBasics
using Trace
using FileIO
using Colors

function tmesh(prim, material)
    prim =  prim isa Sphere ? Tesselation(prim, 64) : prim
    mesh = normal_mesh(prim)
    m = Trace.create_triangle_mesh(mesh, Trace.ShapeCore())
    return Trace.GeometricPrimitive(m, material)
end

LowSphere(radius, contact=Point3f(0)) = Sphere(contact .+ Point3f(0, 0, radius), radius)

begin

    material_red = Trace.MatteMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(0.796f0, 0.235f0, 0.2f0)),
        Trace.ConstantTexture(0.0f0),
    )
    material_blue = Trace.MatteMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(0.251f0, 0.388f0, 0.847f0)),
        Trace.ConstantTexture(0.0f0),
    )
    material_white = Trace.MatteMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(1.0f0)),
        Trace.ConstantTexture(0.0f0),
    )
    mirror = Trace.MirrorMaterial(Trace.ConstantTexture(Trace.RGBSpectrum(0.1f0)))
    glass = Trace.GlassMaterial(
        Trace.ConstantTexture(Trace.RGBSpectrum(1.0f0)),
        Trace.ConstantTexture(Trace.RGBSpectrum(1.0f0)),
        Trace.ConstantTexture(0.0f0),
        Trace.ConstantTexture(0.0f0),
        Trace.ConstantTexture(1.5f0),
        true,
    )
    s1 = tmesh(LowSphere(0.5f0), material_white)
    s2 = tmesh(LowSphere(0.3f0, Point3f(0.5, 0.5, 0)), material_blue)
    s3 = tmesh(LowSphere(0.3f0, Point3f(-0.5, 0.5, 0)), material_red)
    s4 = tmesh(LowSphere(0.4f0, Point3f(0, 1.0, 0)), glass)

    ground = tmesh(Rect3f(Vec3f(-5, -5, 0), Vec3f(10, 10, 0.01)), mirror)
    back = tmesh(Rect3f(Vec3f(-5, -3, 0), Vec3f(10, 0.01, 10)), material_white)
    l = tmesh(Rect3f(Vec3f(-2, -5, 0), Vec3f(0.01, 10, 10)), material_white)
    r = tmesh(Rect3f(Vec3f(2, -5, 0), Vec3f(0.01, 10, 10)), material_white)

    bvh = Trace.no_material_bvh([s1, s2, s3, s4, ground, back, l, r]);

    lights = (
        # Trace.PointLight(Vec3f(0, -1, 2), Trace.RGBSpectrum(22.0f0)),
        Trace.PointLight(Vec3f(0, 0, 2), Trace.RGBSpectrum(10.0f0)),
        Trace.PointLight(Vec3f(0, 3, 3), Trace.RGBSpectrum(25.0f0)),
    )
    scene = Trace.Scene([lights...], bvh);
    resolution = Point2f(1024)
    f = Trace.LanczosSincFilter(Point2f(1.0f0), 3.0f0)

    film = Trace.Film(resolution,
        Trace.Bounds2(Point2f(0.0f0), Point2f(1.0f0)),
        f, 1.0f0, 1.0f0,
)
    screen_window = Trace.Bounds2(Point2f(-1), Point2f(1))
    cam = Trace.PerspectiveCamera(
        Trace.look_at(Point3f(0, 4, 2), Point3f(0, -4, -1), Vec3f(0, 0, 1)),
        screen_window, 0.0f0, 1.0f0, 0.0f0, 1.0f6, 45.0f0, film,
    )
end
integrator = Trace.WhittedIntegrator(cam, Trace.UniformSampler(8), 5)
image = Trace.integrator_threaded(integrator, scene, film, cam)
image_01 = map(c-> mapc(x-> clamp(x, 0.0f0, 1.0f0), c), image)
save("test.png", image_01)
