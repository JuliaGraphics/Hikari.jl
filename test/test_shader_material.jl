using Test
using Hikari
using Hikari.TextureNodes
using Hikari: TextureFilterContext, RGBSpectrum, device_param, Lobe, Fresnel, NodeOp
using GeometryBasics
using GeometryBasics: Point2f, Point3f, Vec3f
using KernelAbstractions: CPU
using LinearAlgebra
using Raycore
using Mantle
using Statistics: mean

static_textures(store) =
    Raycore.StaticMultiTypeSet((), Tuple(filter(a -> typeof(a) === T, store.texture_gpu_arrays) for T in store.texture_order))

function closures(shader; p = Point3f(0), n = Vec3f(0, 0, 1), wo = Vec3f(0, 0, 1),
                  dpdx = Vec3f(1f-3, 0, 0), dpdy = Vec3f(0, 1f-3, 0))
    store = Raycore.MultiTypeSet(CPU())
    handle = device_param(store, Hikari.ShaderGraph(shader))
    sc = Hikari.ShaderContext(TextureFilterContext(Point2f(0), p, n), wo, dpdx, dpdy, Vec3f(1, 0, 0), n)
    cl = Hikari.eval_shader_handle(static_textures(store), handle, sc)
    return collect(cl.closures[1:cl.count])
end

function program(shader)
    store = Raycore.MultiTypeSet(CPU())
    handle = device_param(store, Hikari.ShaderGraph(shader))
    return static_textures(store).textures[handle.slot][handle.idx]
end

@testset "Shader materials" begin
    @testset "a diffuse BSDF is one diffuse closure" begin
        c = only(closures(diffuse_bsdf(color = (0.2, 0.4, 0.6))))
        @test c.lobe == Lobe.DIFFUSE
        @test c.weight ≈ Vec3f(0.2, 0.4, 0.6)
        @test c.n == Vec3f(0, 0, 1)
    end

    @testset "mix shader weighs its two shaders by its factor" begin
        red, blue = diffuse_bsdf(color = (1, 0, 0)), diffuse_bsdf(color = (0, 0, 1))
        @test [c.weight for c in closures(mix_shader(0.25, red, blue))] ≈ [Vec3f(0.75, 0, 0), Vec3f(0, 0, 0.25)]
        @test [c.weight for c in closures(mix_shader(xyz().x, red, blue); p = Point3f(0.3, 0, 0))] ≈
              [Vec3f(0.7, 0, 0), Vec3f(0, 0, 0.3)]
        @test [c.weight for c in closures(add_shader(red, blue))] ≈ [Vec3f(1, 0, 0), Vec3f(0, 0, 1)]
    end

    @testset "a metallic principled BSDF is one F82 tint closure" begin
        c = only(closures(principled_bsdf(base_color = (0.9, 0.6, 0.2), metallic = 1, roughness = 0.4)))
        @test (c.lobe, c.fresnel) == (Lobe.MICROFACET, Fresnel.F82_TINT)
        @test c.f0 ≈ Vec3f(0.9, 0.6, 0.2)
        @test c.alpha ≈ 0.16f0
    end

    @testset "the specular layer of a principled BSDF takes its reflection from the diffuse" begin
        specular, diffuse = closures(principled_bsdf(base_color = (0.8, 0.8, 0.8), roughness = 0.3))
        @test (specular.lobe, specular.fresnel) == (Lobe.MICROFACET, Fresnel.DIELECTRIC)
        @test specular.f0 ≈ Vec3f(0.04)
        @test diffuse.lobe == Lobe.DIFFUSE
        @test diffuse.weight ≈ Vec3f(0.8 * 0.96)
    end

    @testset "a coat takes its reflection from the layers under it" begin
        coat, specular, diffuse = closures(principled_bsdf(base_color = (0.8, 0.8, 0.8), coat_weight = 1))
        @test coat.alpha ≈ 0.03f0^2
        @test specular.weight ≈ Vec3f(0.96)
        @test diffuse.weight ≈ Vec3f(0.8 * 0.96 * 0.96)
    end

    @testset "a bump on a linear height tilts the normal by its slope" begin
        c = only(closures(diffuse_bsdf(normal = bump(xyz().x; distance = 0.5))))
        @test c.n ≈ normalize(Vec3f(-0.5, 0, 1))
        c = only(closures(diffuse_bsdf(normal = bump(xyz().x; distance = 0.5, invert = true))))
        @test c.n ≈ normalize(Vec3f(0.5, 0, 1))
        c = only(closures(diffuse_bsdf(normal = bump(xyz().x; distance = 0.5, strength = 0))))
        @test c.n ≈ Vec3f(0, 0, 1)
    end

    @testset "a bump evaluates only what reads the position three times" begin
        fixed = noise(constant((0.3, 0.2, 0.1)))
        ops = [ins.op for ins in program(diffuse_bsdf(normal = bump(xyz().x * fixed)))]
        @test count(==(NodeOp.NOISE), ops) == 1
        @test count(==(NodeOp.MATH), ops) == 3
    end

    @testset "a texture cannot use a bump" begin
        @test_throws "a texture cannot use `bump`" device_param(Raycore.MultiTypeSet(CPU()), bump(xyz().x))
    end

    @testset "renders as the Diffuse material ($(hw ? "hardware" : "software") tracing)" for hw in (false, true)
        function render(material)
            backend = Mantle.defaultbackend()
            scene = Hikari.Scene(; backend, hw_accel = hw)
            sphere = GeometryBasics.normal_mesh(GeometryBasics.Tesselation(GeometryBasics.Sphere(Point3f(0), 1f0), 32))
            push!(scene, sphere, material)
            push!(scene, Hikari.PointLight(Point3f(0, 0.5, 4), RGBSpectrum(40f0)))
            Hikari.sync!(scene)
            film = Hikari.Film(backend, Hikari.Film(Point2f(32, 32)))
            camera = Hikari.PerspectiveCamera(Point3f(0, 0, 4), Point3f(0), film; fov = 40f0)
            vp = Hikari.VolPath(samples = 16, max_depth = 3, hw_accel = hw)
            vp(scene, film, camera)
            img = Array(film.framebuffer)
            close(vp)
            return [mean(getfield(p, c) for p in img) for c in (:r, :g, :b)]
        end
        colour = (0.7, 0.3, 0.1)
        @test render(Hikari.ShaderMaterial(diffuse_bsdf(color = colour))) ≈ render(Hikari.Diffuse(Kd = colour)) rtol = 0.02
    end
end
