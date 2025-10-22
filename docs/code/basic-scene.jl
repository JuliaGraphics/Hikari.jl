using GeometryBasics
using Trace
using FileIO
using ImageCore
# using BenchmarkTools
# using FileIO, ImageShow

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
    mirror = Trace.MirrorMaterial(Trace.ConstantTexture(Trace.RGBSpectrum(1.0f0)))
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
    s3 = tmesh(LowSphere(0.3f0, Point3f(-0.5, 0.5, 0)), mirror)
    s4 = tmesh(LowSphere(0.4f0, Point3f(0, 1.0, 0)), glass)

    ground = tmesh(Rect3f(Vec3f(-5, -5, 0), Vec3f(10, 10, 0.01)), mirror)
    back = tmesh(Rect3f(Vec3f(-5, -3, 0), Vec3f(10, 0.01, 10)), material_white)
    l = tmesh(Rect3f(Vec3f(-2, -5, 0), Vec3f(0.01, 10, 10)), material_red)
    r = tmesh(Rect3f(Vec3f(2, -5, 0), Vec3f(0.01, 10, 10)), material_blue)

    bvh = Trace.no_material_bvh([s1, s2, s3, s4, #=ground, back, l, r=#]);

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

using RayCaster
using PProf, Profile

begin
    Trace.clear!(film)
    integrator = Trace.WhittedIntegrator(cam, Trace.UniformSampler(8), 5)
    @time integrator(scene, film, cam)
    img = reverse(film.framebuffer, dims=1);
    summary(img)
end
@profview_allocs integrator(scene, film, cam)

@edit integrator(scene, film, cam)

# Setup for profiling sample_kernel_inner!
begin
    tiles = film.tiles
    tile_size = film.tile_size
    filter_radius = film.filter_radius
    filter_table = film.filter_table
    pixels = film.pixels
    tile_bounds = Trace.Bounds2(Point2f(1, 1), Point2f(16, 16))
    tile_column = Int32(1)
    max_depth = Int32(5)
    sampler = Trace.UniformSampler(8)
    camera = cam
    pixel = Point2f(8, 8)
    spp_sqr = 1.0f0 / √Float32(sampler.samples_per_pixel)
    Trace.sample_kernel_inner!(
        tiles, tile_bounds, tile_column, Point2f(size(pixels)),
        max_depth, scene, sampler, camera,
        pixel, spp_sqr, filter_table, filter_radius
    )
end



begin
    Profile.Allocs.clear()
    Profile.Allocs.@profview_allocs integrator(scene, film, cam)
    PProf.Allocs.pprof()
end

@btime Trace.NoMaterial()

bvh = scene.aggregate.bvh
ray = RayCaster.Ray(o=Point3f(0, 4, 2), d=Point3f(0, -4, -1))
args = Trace.intersect!(scene, ray)
camera_sample = @inline get_camera_sample(sampler, campix)
ray, ω = Trace.generate_ray_differential(cam, camera_sample)
ray = scale_differentials(ray, spp_sqr)
l = RGBSpectrum(0.0f0)


function testo(sampler, max_depth, ray, scene)
    l = Trace.RGBSpectrum(0.0f0)
    for depth in 1:1000
        # For demonstration, we just accumulate some dummy value
        l += Trace.li_iterative(sampler, max_depth, ray, scene)
    end
    return l
end

@allocated testo(sampler, max_depth, ray, scene)

@btime Trace.li_iterative(sampler, max_depth, ray, scene)
t1 = scene.aggregate.bvh.primitives[1]
si = Trace.triangle_to_surface_interaction(t1, ray, Point3f(0))
m = scene.aggregate.materials[1]

@edit m(si, false, Trace.Radiance)

bsdf = Trace.matte_material(m, si, false, Trace.Radiance)
m.type == Trace.MATTE_MATERIAL
Trace.BSDF()


@code_warntype Trace.specular(Trace.Transmit, bsdf, sampler, ray, si)
using JET

# begin
#     resolution = Point2f(1024)
#     Trace.clear!(film)
#     @time render_scene(scene, film, cam)
#     Trace.to_framebuffer!(film, 1.0f0)
#     film.framebuffer
# end


# 6.296157 seconds (17.64 k allocations: 19.796 MiB, 0.13% gc time, 45 lock conflicts)
# After more GPU optimizations
# 4.169616 seconds (17.37 k allocations: 19.777 MiB, 0.14% gc time, 20 lock conflicts)
# After first shading running on GPU
# 3.835527 seconds (17.36 k allocations: 19.779 MiB, 0.16% gc time, 41 lock conflicts)
# 4.191 s (4710 allocations: 18.36 MiB)
# iterative_li: 5.2s -.-


# begin
#     integrator = Trace.SPPMIntegrator(cam, 0.075f0, 5, 1)
#     integrator(scene)
#     img = reverse(film.framebuffer, dims=1)
# end
