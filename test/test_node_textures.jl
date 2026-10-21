using Test
using Hikari
using Hikari.TextureNodes
using Hikari: TextureFilterContext, TexKind, RGBSpectrum, device_param, eval_handle, eval_handle_spectrum
using GeometryBasics
using GeometryBasics: Point2f, Point3f, Vec3f
using KernelAbstractions: CPU
using LinearAlgebra
using Raycore
using Mantle
using Statistics: mean

"""The store's texture table on the host: adapting to `CPU()` would give POCL device arrays."""
host_static(store) =
    Raycore.StaticMultiTypeSet((), Tuple(filter(a -> typeof(a) === T, store.texture_gpu_arrays) for T in store.texture_order))

function evaluate(graph; uv = Point2f(0), p = Point3f(0), n = Vec3f(0, 0, 1))
    store = Raycore.MultiTypeSet(CPU())
    handle = device_param(store, graph)
    tfc = TextureFilterContext(uv, p, n)
    rgb = eval_handle_spectrum(host_static(store), handle, tfc)
    return Vec3f(rgb.c[1], rgb.c[2], rgb.c[3])
end

@testset "Node textures" begin
    @testset "inputs" begin
        @test evaluate(xyz(); p = Point3f(1, -2, 3)) == Vec3f(1, -2, 3)
        @test evaluate(uv(); uv = Point2f(0.25, 0.75)) == Vec3f(0.25, 0.75, 0)
        @test evaluate(normal(); n = Vec3f(0, 1, 0)) == Vec3f(0, 1, 0)
    end

    @testset "arithmetic and swizzles" begin
        p = xyz()
        @test evaluate(p.zx * 2 + 1; p = Point3f(1, 2, 3)) == Vec3f(7, 3, 1)
        @test evaluate(p.y; p = Point3f(1, 2, 3)) == Vec3f(2, 2, 2)
        @test evaluate(combine(p.z, p.x, 5); p = Point3f(1, 2, 3)) == Vec3f(3, 1, 5)
        @test evaluate(norm(p.xy); p = Point3f(3, 4, 7)) == Vec3f(5, 5, 5)
        @test evaluate(fract(p) - floor(-p); p = Point3f(1.25, 2.5, -0.75)) ≈ Vec3f(2.25, 3.5, 0.25)
        @test evaluate(p / 0; p = Point3f(1, 2, 3)) == Vec3f(0, 0, 0)
        @test evaluate(mod(p, 2); p = Point3f(-1, 3, 4.5)) == Vec3f(1, 1, 0.5)
        @test evaluate(dot(p, (1, 1, 1)) + cross(p, (0, 0, 1)); p = Point3f(1, 2, 3)) == Vec3f(8, 5, 6)
    end

    @testset "mapping" begin
        moved = mapping(xyz(); location = (1, 0, 0), rotation = (0, 0, π / 2), scale = (2, 2, 2))
        @test evaluate(moved; p = Point3f(1, 0, 0)) ≈ Vec3f(1, 2, 0) atol = 1e-6
        @test evaluate(transform(xyz(), [1 0 0 5; 0 2 0 0; 0 0 1 0; 0 0 0 1]); p = Point3f(1, 1, 1)) == Vec3f(6, 2, 1)
    end

    @testset "color ramp" begin
        ramp(t) = evaluate(color_ramp(xyz().x, 0.2 => (1, 0, 0), 0.6 => (0, 0, 1), 1.0 => (0, 1, 0)); p = Point3f(t, 0, 0))
        @test ramp(0.0) == Vec3f(1, 0, 0)
        @test ramp(0.4) ≈ Vec3f(0.5, 0, 0.5)
        @test ramp(0.8) ≈ Vec3f(0, 0.5, 0.5)
        @test ramp(2.0) == Vec3f(0, 1, 0)
    end

    @testset "procedurals" begin
        p = xyz()
        @test evaluate(noise(p; detail = 0); p = Point3f(3, -2, 7)) == Vec3f(0.5, 0.5, 0.5)
        @test evaluate(noise(p; detail = 0); p = Point3f(3.3, -2, 7)) != Vec3f(0.5, 0.5, 0.5)
        q = Point3f(2.3, -0.4, 5.9)
        @test evaluate(voronoi(p; randomness = 0); p = q) ≈ Vec3f(fill(norm(q - round.(q)), 3)) atol = 1e-6
        @test evaluate(voronoi(p; randomness = 0, output = :position); p = q) == Vec3f(round.(q))
        cell(x) = evaluate(white_noise(floor(p)); p = x)
        @test cell(Point3f(0.2, 0.3, 0.4)) == cell(Point3f(0.9, 0.1, 0.6))
        @test cell(Point3f(0.2, 0.3, 0.4)) != cell(Point3f(1.2, 0.3, 0.4))
    end

    @testset "image at computed coordinates" begin
        img = [RGBSpectrum(Float32(i), Float32(j), 0f0) for i in 1:2, j in 1:2]
        @test evaluate(image_texture(img, xyz().xy); p = Point3f(1, 2, 0)) == Vec3f(1.5, 1.5, 0)
        @test evaluate(image_texture(img); uv = Point2f(0.25, 0.75)) == Vec3f(1, 1, 0)
    end

    @testset "scalar parameters read the first component" begin
        store = Raycore.MultiTypeSet(CPU())
        handle = device_param(store, xyz().z * 2)
        @test handle.kind == TexKind.NODES
        @test eval_handle(host_static(store), handle, TextureFilterContext(Point2f(0), Point3f(0, 0, 0.25))) == 0.5f0
    end

    @testset "registers are reused" begin
        p = xyz()
        terms = [sin(p.x * k) for k in 1:40]
        x = Point3f(0.3, 0, 0)
        @test evaluate(reduce(+, terms); p = x)[1] ≈ sum(sin(0.3f0 * k) for k in 1:40) atol = 1e-5
        @test evaluate(foldr(+, terms); p = x)[1] ≈ sum(sin(0.3f0 * k) for k in 1:40) atol = 1e-5
        @test_throws "more than 16 live values" Hikari.compile_nodes(reduce(+, terms) + reduce(+, [t * t for t in terms]), identity)
    end

    @testset "bump mapping moves the position" begin
        store = Raycore.MultiTypeSet(CPU())
        slope = 0.5f0
        height = device_param(store, xyz().x * slope)
        tfc = TextureFilterContext(Point2f(0.3, 0.3), 0.01f0, 0f0, 0f0, 0.01f0, Point3f(0.3, 0.3, 0), Vec3f(0, 0, 1))
        ns, _ = Hikari.perturb_bump_frame(height, host_static(store), Vec3f(0, 0, 1),
                                          Vec3f(1, 0, 0), Vec3f(0, 1, 0), Vec3f(0), Vec3f(0), Vec3f(0, 0, 1), tfc)
        @test ns ≈ normalize(Vec3f(-slope, 0, 1))
    end

    @testset "updating a graph reuses its slot" begin
        store = Raycore.MultiTypeSet(CPU())
        handle = device_param(store, xyz().x * 2)
        updated = Raycore.update_item(store, handle, xyz().x * 3)
        @test updated == handle
        tfc = TextureFilterContext(Point2f(0), Point3f(1, 0, 0))
        @test eval_handle(host_static(store), updated, tfc) == 3f0
        longer = Raycore.update_item(store, handle, xyz().x * 3 + 1)
        @test longer.idx != handle.idx
        @test eval_handle(host_static(store), longer, tfc) == 4f0
    end

    @testset "rendered by position ($(hw ? "hardware" : "software") tracing)" for hw in (false, true)
        backend = Mantle.defaultbackend()
        scene = Hikari.Scene(; backend, hw_accel = hw)
        sphere = GeometryBasics.normal_mesh(GeometryBasics.Tesselation(GeometryBasics.Sphere(Point3f(0), 1f0), 32))
        push!(scene, sphere, Hikari.Diffuse(Kd = mix((0.8, 0.05, 0.05), (0.05, 0.05, 0.8), greater(xyz().x, 0))))
        push!(scene, Hikari.PointLight(Point3f(0, 0.5, 4), RGBSpectrum(40f0)))
        Hikari.sync!(scene)
        film = Hikari.Film(backend, Hikari.Film(Point2f(32, 32)))
        camera = Hikari.PerspectiveCamera(Point3f(0, 0, 4), Point3f(0), film; fov = 40f0)
        vp = Hikari.VolPath(samples = 4, max_depth = 2, hw_accel = hw)
        vp(scene, film, camera)
        img = Array(film.framebuffer)
        close(vp)
        channels(cols) = (r = mean(p.r for p in img[12:20, cols]), b = mean(p.b for p in img[12:20, cols]))
        negative_x, positive_x = channels(6:12), channels(21:27)
        @test negative_x.r > 10 * negative_x.b
        @test positive_x.b > 10 * positive_x.r
    end
end
