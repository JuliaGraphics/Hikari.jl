using Test, Hikari, Mantle, GeometryBasics
using GeometryBasics: normal_mesh, Tesselation, Sphere, Point2f, Point3f

function wave_scene(backend)
    scene = Hikari.Scene(; backend = backend)
    sphere = normal_mesh(Tesselation(Sphere(Point3f(0, 0, 0), 0.5f0), 16))
    push!(scene, sphere, Hikari.Diffuse(color = Hikari.RGBSpectrum(0.7f0, 0.4f0, 0.4f0)))
    push!(scene, Hikari.PointLight(Point3f(0, -2, 3), Hikari.RGBSpectrum(30f0)))
    Hikari.sync!(scene)
    return scene
end

function rendered(backend, scene; calls, samples)
    return rendered_batches(backend, scene, fill(samples, calls))
end

function rendered_batches(backend, scene, batches)
    film = Hikari.Film(backend, Hikari.Film(Point2f(32, 24)))
    camera = Hikari.PerspectiveCamera(Point3f(0, -3, 1.5f0), Point3f(0, 0, 0.5f0), film; fov = 50f0)
    vp = Hikari.VolPath(samples = sum(batches), max_depth = 3)
    plans = nothing
    accumulators = nothing
    previous_samples = first(batches)
    for samples in batches
        Hikari.render!(vp, scene, film, camera; samples)
        if plans === nothing
            accumulators = (vp.state.accumulation_memory, vp.state.pixel_rgb, vp.state.pixel_weight_sum)
        else
            @test vp.state.accumulation_memory === accumulators[1]
            @test vp.state.pixel_rgb === accumulators[2]
            @test vp.state.pixel_weight_sum === accumulators[3]
            samples == previous_samples && (@test vp.state.plans === plans)
        end
        plans = vp.state.plans
        previous_samples = samples
    end
    Mantle.waitidle(Hikari.mantle_device(backend))
    lanes = vp.state.sobol_rng.lanes
    @test vp.state.shadow_queue.capacity == 1
    image = Array(film.framebuffer)
    close(vp)
    @test Hikari.nallocations(accumulators[1]) == 0
    Hikari.free!(film)
    return image, film.iteration_index[], lanes
end

@testset "changing progressive batch size retains earlier samples" begin
    backend = Mantle.defaultbackend()
    scene = wave_scene(backend)
    expected, count, _ = rendered(backend, scene; calls = 8, samples = 1)
    @test count == 8
    for batches in ((1, 7), (1, 4, 3), (4, 1, 3))
        actual, iteration, _ = rendered_batches(backend, scene, batches)
        @test iteration == 8
        @test all(isapprox(getfield(a, c), getfield(b, c); rtol = 5f-5, atol = 2f-5)
                  for (a, b) in zip(actual, expected) for c in (:r, :g, :b))
    end
end

@testset "an existing state's wave budget stays fixed as memory is consumed" begin
    backend = Mantle.defaultbackend()
    state = Hikari.VolPathState(backend, 2, 2, Hikari.Scene(; backend).lights;
                               has_media = false)
    vp = Hikari.VolPath()
    try
        @test Hikari.wave_path_limit(vp, 16 << 30) == 1 << 22
        @test Hikari.wave_path_limit(vp, 8 << 30) == 1 << 21
        vp.state = state
        vp.path_limit = 1 << 22
        @test Hikari.wave_path_limit(vp, 16 << 30) == 1 << 22
        @test Hikari.wave_path_limit(vp, 8 << 30) == 1 << 22
        @test Hikari.wave_path_limit(vp, typemax(Int)) == 1 << 22
    finally
        close(vp)
    end
end

@testset "medium shadow capacity follows the medium work capacity" begin
    backend = Mantle.defaultbackend()
    lights = Hikari.Scene(; backend).lights
    for has_media in (false, true)
        state = Hikari.VolPathState(backend, 2, 2, lights; has_media)
        try
            @test state.shadow_queue.capacity == (has_media ? 4 : 1)
            @test state.shadow_queue.capacity == state.medium_sample_queue.capacity
        finally
            Hikari.free!(state)
        end
    end
end

@testset "samples rendered in one wave are the samples rendered one by one" begin
    backend = Mantle.defaultbackend()
    scene = wave_scene(backend)
    one_by_one, count_one, lanes_one = rendered(backend, scene; calls = 8, samples = 1)
    waves, count_waves, lanes_waves = rendered(backend, scene; calls = 1, samples = 8)
    @test (count_one, lanes_one) == (8, 1)
    @test (count_waves, lanes_waves) == (8, 8)
    @test maximum(abs(getfield(a, c) - getfield(b, c)) for (a, b) in zip(one_by_one, waves) for c in (:r, :g, :b)) < 1f-5
end

@testset "a wave holds the most samples that divide the count and fit its paths" begin
    @test Hikari.wave_lanes(1024, 256 * 256) == 32
    @test Hikari.wave_lanes(12, 256 * 256) == 12
    @test Hikari.wave_lanes(7, 256 * 256) == 7
    @test Hikari.wave_lanes(64, 1920 * 1080) == 1
    @test Hikari.wave_lanes(1, 64) == 1
    @test Hikari.wave_lanes(64, 512 * 512) == 8
    @test Hikari.wave_lanes(64, 3840 * 2160) == 1
    @test Hikari.wave_path_limit(8 << 30) == 1 << 21
    @test Hikari.wave_path_limit(11 << 30) == 1 << 21
    @test Hikari.wave_path_limit(12 << 30) == 1 << 22
    @test Hikari.wave_path_limit(16 << 30) == 1 << 22
    @test Hikari.wave_path_limit(24 << 30) == 1 << 22
    @test Hikari.wave_path_limit(typemax(Int)) == 1 << 21
    @test Hikari.wave_lanes(64, 512 * 512, 1 << 22) == 16
    @test Hikari.wave_lanes(64, 1024 * 1024, 1 << 22) == 4
end
