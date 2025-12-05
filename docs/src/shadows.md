## shadows

```@example
using GeometryBasics
using Hikari
using FileIO
using ImageCore

function render()
    material_red = Hikari.MatteMaterial(
        Hikari.ConstantTexture(Hikari.RGBSpectrum(0.796f0, 0.235f0, 0.2f0)),
        Hikari.ConstantTexture(0f0),
    )
    material_blue = Hikari.MatteMaterial(
        Hikari.ConstantTexture(Hikari.RGBSpectrum(0.251f0, 0.388f0, 0.847f0)),
        Hikari.ConstantTexture(0f0),
    )
    material_white = Hikari.MatteMaterial(
        Hikari.ConstantTexture(Hikari.RGBSpectrum(1f0)),
        Hikari.ConstantTexture(0f0),
    )
    mirror = Hikari.MirrorMaterial(Hikari.ConstantTexture(Hikari.RGBSpectrum(1f0)))
    glass = Hikari.GlassMaterial(
        Hikari.ConstantTexture(Hikari.RGBSpectrum(1f0)),
        Hikari.ConstantTexture(Hikari.RGBSpectrum(1f0)),
        Hikari.ConstantTexture(0f0),
        Hikari.ConstantTexture(0f0),
        Hikari.ConstantTexture(1.5f0),
        true,
    )

    core = Hikari.ShapeCore(
        Hikari.translate(Vec3f(0.3, 0.11, -2.2)), false,
    )
    sphere = Hikari.Sphere(core, 0.1f0, 360f0)
    primitive = Hikari.GeometricPrimitive(sphere, glass)

    core2 = Hikari.ShapeCore(
        Hikari.translate(Vec3f(0.2, 0.11, -2.6)), false,
    )
    sphere2 = Hikari.Sphere(core2, 0.1f0, 360f0)
    primitive2 = Hikari.GeometricPrimitive(sphere2, material_blue)

    core3 = Hikari.ShapeCore(
        Hikari.translate(Vec3f(0.7, 0.31, -2.8)), false,
    )
    sphere3 = Hikari.Sphere(core3, 0.3f0, 360f0)
    primitive3 = Hikari.GeometricPrimitive(sphere3, mirror)

    core4 = Hikari.ShapeCore(
        Hikari.translate(Vec3f(0.7, 0.11, -2.3)), false,
    )
    sphere4 = Hikari.Sphere(core4, 0.1f0, 360f0)
    primitive4 = Hikari.GeometricPrimitive(sphere4, material_red)

    triangles = Hikari.create_triangle_mesh(
        Hikari.ShapeCore(Hikari.translate(Vec3f(0, 0, -2)), false),
        4,
        UInt32[
            1, 2, 3,
            1, 4, 3,
            2, 3, 5,
            6, 5, 3,
        ],
        6,
        [
            Point3f(0, 0, 0), Point3f(0, 0, -1),
            Point3f(1, 0, -1), Point3f(1, 0, 0),
            Point3f(0, 1, -1), Point3f(1, 1, -1),
        ],
        [
            Hikari.Normal3f(0, 1, 0), Hikari.Normal3f(0, 1, 0),
            Hikari.Normal3f(0, 1, 0), Hikari.Normal3f(0, 1, 0),
            Hikari.Normal3f(0, 0, 1), Hikari.Normal3f(0, 0, 1),
        ],
    )
    triangle_primitive = Hikari.GeometricPrimitive(triangles[1], mirror)
    triangle_primitive2 = Hikari.GeometricPrimitive(triangles[2], mirror)
    triangle_primitive3 = Hikari.GeometricPrimitive(triangles[3], material_white)
    triangle_primitive4 = Hikari.GeometricPrimitive(triangles[4], material_white)

    bvh = Hikari.BVH([
        primitive, primitive2, primitive3, primitive4,
        triangle_primitive, triangle_primitive2,
        triangle_primitive3, triangle_primitive4,
    ], 1)

    lights = [Hikari.PointLight(
        Hikari.translate(Vec3f(-1, 1, 0)), Hikari.RGBSpectrum(25f0),
    )]
    scene = Hikari.Scene(lights, bvh)

    resolution = Point2f(1024 ÷ 3)
    filter = Hikari.LanczosSincFilter(Point2f(1f0), 3f0)
    film = Hikari.Film(resolution,
        Hikari.Bounds2(Point2f(0f0), Point2f(1f0)),
        filter, 1f0, 1f0,
        "shadows_sppm_res.png",
    )
    screen = Hikari.Bounds2(Point2f(-1f0), Point2f(1f0))
    camera = Hikari.PerspectiveCamera(
        Hikari.look_at(Point3f(0, 15, 50), Point3f(0, 0, -2), Vec3f(0, 1, 0)),
        screen, 0f0, 1f0, 0f0, 1f6, 90f0, film,
    )
    # integrator = Hikari.WhittedIntegrator(camera, Hikari.UniformSampler(8), 8)
    integrator = Hikari.SPPMIntegrator(camera, 0.025f0, 5, 10)
    scene |> integrator
end

render()
```

![](shadows_sppm_res.png)
