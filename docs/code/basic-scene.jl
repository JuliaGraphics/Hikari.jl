using GeometryBasics
using Hikari
using FileIO
using ImageCore
using Raycore

# using BenchmarkTools
# using FileIO, ImageShow

function tmesh(prim, material)
    prim = prim isa Sphere ? Tesselation(prim, 64) : prim
    mesh = normal_mesh(prim)
    m = Raycore.TriangleMesh(mesh)
    return Hikari.GeometricPrimitive(m, material)
end

LowSphere(radius, contact=Point3f(0)) = Sphere(contact .+ Point3f(0, 0, radius), radius)

begin

    material_red = Hikari.MatteMaterial(
        Hikari.ConstantTexture(Hikari.RGBSpectrum(0.796f0, 0.235f0, 0.2f0)),
        Hikari.ConstantTexture(0.0f0),
    )
    material_blue = Hikari.MatteMaterial(
        Hikari.ConstantTexture(Hikari.RGBSpectrum(0.251f0, 0.388f0, 0.847f0)),
        Hikari.ConstantTexture(0.0f0),
    )
    material_white = Hikari.MatteMaterial(
        Hikari.ConstantTexture(Hikari.RGBSpectrum(1.0f0)),
        Hikari.ConstantTexture(0.0f0),
    )
    mirror = Hikari.MirrorMaterial(Hikari.ConstantTexture(Hikari.RGBSpectrum(1.0f0)))
    glass = Hikari.GlassMaterial(
        Hikari.ConstantTexture(Hikari.RGBSpectrum(1.0f0)),
        Hikari.ConstantTexture(Hikari.RGBSpectrum(1.0f0)),
        Hikari.ConstantTexture(0.0f0),
        Hikari.ConstantTexture(0.0f0),
        Hikari.ConstantTexture(1.5f0),
        true,
    )

    s1 = tmesh(LowSphere(0.5f0), material_white)
    s2 = tmesh(LowSphere(0.3f0, Point3f(0.5, 0.5, 0)), material_white)
    s3 = tmesh(LowSphere(0.3f0, Point3f(-0.5, 0.5, 0)), material_white)
    s4 = tmesh(LowSphere(0.4f0, Point3f(0, 1.0, 0)), material_white)

    ground = tmesh(Rect3f(Vec3f(-5, -5, 0), Vec3f(10, 10, 0.01)), material_white)
    back = tmesh(Rect3f(Vec3f(-5, -3, 0), Vec3f(10, 0.01, 10)), material_white)
    l = tmesh(Rect3f(Vec3f(-2, -5, 0), Vec3f(0.01, 10, 10)), material_white)
    r = tmesh(Rect3f(Vec3f(2, -5, 0), Vec3f(0.01, 10, 10)), material_white)

    bvh = Hikari.no_material_bvh([s1, s2, s3, s4, ground, back, l, r]);

    lights = (
        # Hikari.PointLight(Vec3f(0, -1, 2), Hikari.RGBSpectrum(22.0f0)),
        Hikari.PointLight(Vec3f(0, 0, 2), Hikari.RGBSpectrum(10.0f0)),
        # Hikari.PointLight(Vec3f(0, 3, 3), Hikari.RGBSpectrum(25.0f0)),
    )
    scene = Hikari.Scene([lights...], bvh);
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
end


# begin
#     Hikari.clear!(film)
#     integrator = Hikari.WhittedIntegrator(cam, Hikari.UniformSampler(8), 5)
#     @time integrator(scene, film, cam)
# end
pres = []
using AMDGPU
g_scene = Hikari.to_gpu(ROCArray, scene; preserve=pres);
g_film = Hikari.to_gpu(ROCArray, film; preserve=pres);
integrator = Hikari.WhittedIntegrator(cam, Hikari.UniformSampler(8), 5)
integrator(g_scene, g_film, cam);
AMDGPU.@device_code_warntype interactive = true integrator(g_scene, g_film, cam);
@time integrator(scene, film, cam);
@time Hikari.integrator_threaded(integrator, scene, film, cam);


image = @btime (Hikari.integrator_threaded(integrator, scene, film, camera))

@code_warntype Hikari.integrator_threaded(integrator, scene, film, cam)
using OpenCL

cl_scene = Hikari.to_gpu(CLArray, scene)
cl_film = Hikari.to_gpu(CLArray, film);

@time integrator(cl_scene, cl_film, cam);
OpenCL.@device_code_warntype interactive = true integrator(cl_scene, cl_film, cam)
# Setup for profiling sample_kernel_inner!


# begin
#     resolution = Point2f(1024)
#     Hikari.clear!(film)
#     @time render_scene(scene, film, cam)
#     Hikari.to_framebuffer!(film, 1.0f0)
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
#     integrator = Hikari.SPPMIntegrator(cam, 0.075f0, 5, 1)
#     integrator(scene)
#     img = reverse(film.framebuffer, dims=1)
# end
