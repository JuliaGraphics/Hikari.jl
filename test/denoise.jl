using GeometryBasics: normal_mesh, Tesselation
import Adapt
# `KA.CPU()` arrived with 6ea9555 but nothing ever bound `KA`: Hikari imports
# KernelAbstractions under that alias internally without exporting it, and no
# file earlier in runtests.jl binds it either, so every testset below has been
# erroring with UndefVarError since 2026-08-19.
import KernelAbstractions as KA
include("software_backend.jl")

@testset "Denoising" begin
    to_mesh(prim) = normal_mesh(prim isa Sphere ? Tesselation(prim, 32) : prim)

    function make_noisy_scene(backend = Mantle.defaultbackend())
        scene = Hikari.Scene(; backend = backend)
        push!(scene, to_mesh(Sphere(Point3f(0, 0.5, 0), 0.5f0)),
              Hikari.Diffuse(Kd=Hikari.RGBSpectrum(0.8f0, 0.3f0, 0.2f0)))
        push!(scene, to_mesh(Rect3f(Vec3f(-2, 0, -2), Vec3f(4, 0.01, 4))),
              Hikari.Diffuse(Kd=Hikari.RGBSpectrum(0.85f0)))
        push!(scene, Hikari.PointLight(Point3f(2f0, 3f0, -1f0), Hikari.RGBSpectrum(40f0)))
        Hikari.sync!(scene)
        return scene
    end

    function render_noisy(scene; res=32, spp=2, backend = Mantle.defaultbackend())
        film = Hikari.Film(Point2f(res, res))
        camera = Hikari.PerspectiveCamera(
            Point3f(2f0, 1.5f0, -2f0), Point3f(0f0, 0.3f0, 0f0), film; fov=45f0)
        film = Hikari.Film(backend, film)
        Hikari.clear!(film)
        integrator = Hikari.VolPath(samples=spp, max_depth=4)
        try
            integrator(scene, film, camera)
        finally
            close(integrator)
        end
        Hikari.fill_aux_buffers!(film, scene, camera)
        return film, camera
    end

    @testset "DenoiseConfig defaults" begin
        config = Hikari.DenoiseConfig()
        @test config.iterations == 4
        @test config.sigma_color == 1.0f0
        @test config.sigma_normal == 64.0f0
        @test config.sigma_depth == 0.1f0
    end

    @testset "denoise! runs without error" begin
        scene = make_noisy_scene()
        film, _ = render_noisy(scene)

        fb_before = copy(Array(film.framebuffer))
        Hikari.denoise!(film)
        fb_after = Array(film.framebuffer)

        # Framebuffer was modified
        @test fb_before != fb_after

        # No NaN or Inf in denoised output
        @test !any(px -> isnan(px.r) || isnan(px.g) || isnan(px.b), fb_after)
        @test !any(px -> isinf(px.r) || isinf(px.g) || isinf(px.b), fb_after)

        # Non-negative values
        @test all(px -> px.r >= 0 && px.g >= 0 && px.b >= 0, fb_after)
    end

    @testset "denoise! reduces noise" begin
        scene = make_noisy_scene()
        film, _ = render_noisy(scene; spp=2)

        fb_noisy = Array(film.framebuffer)

        # Compute per-pixel variance in a 3x3 window as a noise proxy
        function local_variance(img)
            h, w = size(img)
            total = 0.0
            for j in 2:w-1, i in 2:h-1
                lum_center = Float64(img[i,j].r + img[i,j].g + img[i,j].b) / 3
                var_sum = 0.0
                for dj in -1:1, di in -1:1
                    lum = Float64(img[i+di,j+dj].r + img[i+di,j+dj].g + img[i+di,j+dj].b) / 3
                    var_sum += (lum - lum_center)^2
                end
                total += var_sum / 9
            end
            return total / ((h-2) * (w-2))
        end

        var_noisy = local_variance(fb_noisy)
        Hikari.denoise!(film)
        fb_denoised = Array(film.framebuffer)
        var_denoised = local_variance(fb_denoised)

        # Denoised image should have less local variance
        @test var_denoised < var_noisy
    end

    @testset "postprocess after denoise" begin
        scene = make_noisy_scene()
        film, _ = render_noisy(scene)

        Hikari.denoise!(film)
        img = Hikari.postprocess!(film; tonemap=:aces, gamma=2.2f0)
        arr = Array(img)

        @test size(arr) == (32, 32)
        @test !any(px -> isnan(px.r) || isnan(px.g) || isnan(px.b), arr)
        # After tonemap + gamma, values should be in [0, 1]
        @test all(px -> 0 <= px.r <= 1 && 0 <= px.g <= 1 && 0 <= px.b <= 1, arr)
    end

    @testset "custom DenoiseConfig" begin
        scene = make_noisy_scene()
        film, _ = render_noisy(scene)

        # An ODD iteration count, so the writeback pass runs: the result of the
        # last à-trous pass lands in the scratch, and the film has to end up
        # holding it.
        config = Hikari.DenoiseConfig(iterations=3, sigma_color=2.0f0)
        Hikari.denoise!(film; config=config)
        fb = Array(film.framebuffer)
        @test !any(px -> isnan(px.r) || isnan(px.g) || isnan(px.b), fb)
    end

    @testset "the plan and its scratch are kept, not rebuilt per call" begin
        scene = make_noisy_scene()
        film, _ = render_noisy(scene)
        Hikari.denoise!(film)
        plan = film.denoise_plan[]
        @test plan !== nothing
        Hikari.denoise!(film)
        # A full-resolution scratch allocation and a plan compile per frame is
        # what this replaced; same config must reuse both.
        @test film.denoise_plan[] === plan
        # A different iteration count is a different number of passes.
        Hikari.denoise!(film; config = Hikari.DenoiseConfig(iterations = 2))
        @test film.denoise_plan[] !== plan
    end

    @testset "sigmas are read per run" begin
        # They ride `Ref`s, so turning the filter up must not recompile — and
        # must not be ignored either.
        scene = make_noisy_scene()
        film, _ = render_noisy(scene)
        Hikari.denoise!(film; config = Hikari.DenoiseConfig(sigma_color = 0.01f0))
        plan = film.denoise_plan[]
        sharp = copy(Array(film.framebuffer))

        film2, _ = render_noisy(scene)
        Hikari.denoise!(film2; config = Hikari.DenoiseConfig(sigma_color = 8.0f0))
        blurry = Array(film2.framebuffer)

        @test film.denoise_plan[] === plan
        @test sharp != blurry
    end

    @testset "GPU denoising agrees with CPU Vulkan" begin
        reference = software_reference_backend()
        if reference === nothing
            @test_skip false # No CPU Vulkan driver installed.
        else
            # The same noisy image and guides, filtered independently by the
            # hardware GPU and the software driver on the CPU.
            backend = Mantle.defaultbackend()
            gpu_scene = make_noisy_scene(backend)
            gfilm, _ = render_noisy(gpu_scene; backend)
            cfilm = Hikari.Film(reference, Hikari.Film(gfilm.resolution))
            try
                src = Array(gfilm.framebuffer)
                copyto!(cfilm.normal, Array(gfilm.normal))
                copyto!(cfilm.depth, Array(gfilm.depth))
                # Odd and even: the odd count exercises the writeback pass.
                for iters in (3, 4)
                    cfg = Hikari.DenoiseConfig(iterations = iters)
                    copyto!(gfilm.framebuffer, src)
                    copyto!(cfilm.framebuffer, src)
                    Hikari.denoise!(gfilm; config = cfg)
                    Hikari.denoise!(cfilm; config = cfg)
                    g = Array(gfilm.framebuffer)
                    c = Array(cfilm.framebuffer)
                    @test all(px -> isfinite(px.r) && isfinite(px.g) && isfinite(px.b), g)
                    @test g != src
                    @test maximum(i -> max(abs(c[i].r - g[i].r), abs(c[i].g - g[i].g),
                                           abs(c[i].b - g[i].b)), eachindex(c)) < 1f-5
                end
            finally
                Hikari.free!(gfilm)
                Hikari.free!(cfilm)
            end
        end
    end
end
