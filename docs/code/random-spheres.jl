using GeometryBasics
using Hikari
using FileIO
using Colors

function tmesh(prim, material)
    prim = prim isa Sphere ? Tesselation(prim, 64) : prim
    mesh = normal_mesh(prim)
    m = Raycore.TriangleMesh(mesh)
    return Hikari.GeometricPrimitive(m, material)
end


function random_spheres()
    primitives = []
    push!(primitives, tmesh(Sphere(Point3(0, -1000, 0), 1000.0), Hikari.MirrorMaterial(Hikari.ConstantTexture(Hikari.RGBSpectrum(0.5f0)))))

    function rand_material()
        p = rand()
        if p < 0.8
            Hikari.MatteMaterial(
                Hikari.ConstantTexture(Hikari.RGBSpectrum(0.796f0, 0.235f0, 0.2f0)),
                Hikari.ConstantTexture(0.0f0),
            )
        elseif p < 0.95
            rf = rand(Float32)
            Hikari.MirrorMaterial(Hikari.ConstantTexture(Hikari.RGBSpectrum(rf)))
        else
            Hikari.PlasticMaterial(
                Hikari.ConstantTexture(Hikari.RGBSpectrum(0.6399999857f0, 0.6399999857f0, 0.6399999857f0)),
                Hikari.ConstantTexture(Hikari.RGBSpectrum(0.1000000015f0, 0.1000000015f0, 0.1000000015f0)),
                Hikari.ConstantTexture(0.010408001f0),
                true,
            )
        end
    end

    for a in -11:10, b in -11:10
        center = Point3f(a + 0.9rand(), 0.2, b + 0.9rand())
        if norm(center - Point3f(4, 0.2, 0)) > 0.9
            push!(primitives, tmesh(Sphere(center, 0.2), rand_material()))
        end
    end
    glass = Hikari.GlassMaterial(
        Hikari.ConstantTexture(Hikari.RGBSpectrum(1.0f0)),
        Hikari.ConstantTexture(Hikari.RGBSpectrum(1.0f0)),
        Hikari.ConstantTexture(0.0f0),
        Hikari.ConstantTexture(0.0f0),
        Hikari.ConstantTexture(1.5f0),
        true,
    )

    push!(primitives, tmesh(Sphere(Point3f(0, 1, 0), 1.0), glass))
    push!(primitives, tmesh(Sphere(Point3f(-4, 1, 0), 1.0), rand_material()))
    push!(primitives, tmesh(Sphere(Point3f(4, 1, 0), 1.0), rand_material()))

    return primitives
end


begin
    bvh = Hikari.no_material_bvh(random_spheres())

    lights = (
        Hikari.PointLight(Vec3f(0, 0, 2), Hikari.RGBSpectrum(10.0f0)),
        Hikari.PointLight(Vec3f(0, 3, 3), Hikari.RGBSpectrum(15.0f0)),
    )
    scene = Hikari.Scene([lights...], bvh)

    resolution = Point2f(1024)
    f = Hikari.LanczosSincFilter(Point2f(1.0f0), 3.0f0)
    film = Hikari.Film(
        resolution,
        Hikari.Bounds2(Point2f(0.0f0), Point2f(1.0f0)),
        f, 1.0f0, 1.0f0,
    )
    screen_window = Hikari.Bounds2(Point2f(-1), Point2f(1))
    cam = Hikari.PerspectiveCamera(
        Hikari.look_at(Point3f(0, 4, 2), Point3f(0, -4, -1), Vec3f(0, 0, 1)),
        screen_window, 0.0f0, 1.0f0, 0.0f0, 1.0f6, 45.0f0, film,
    )

    # Render with WhittedIntegrator
    integrator = Hikari.WhittedIntegrator(cam, Hikari.UniformSampler(8), 10)
    @time Hikari.integrator_threaded(integrator, scene, film, cam)

    # Save the result
    image_01 = map(c -> mapc(x -> clamp(x, 0f0, 1f0), c), film.framebuffer)
    save(joinpath(@__DIR__, "shadows_sppm_res.png"), image_01)

    img = reverse(film.framebuffer, dims=1)
end
