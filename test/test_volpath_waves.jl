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
    film = Hikari.Film(backend, Hikari.Film(Point2f(32, 24)))
    camera = Hikari.PerspectiveCamera(Point3f(0, -3, 1.5f0), Point3f(0, 0, 0.5f0), film; fov = 50f0)
    vp = Hikari.VolPath(samples = calls * samples, max_depth = 3)
    for _ in 1:calls
        Hikari.render!(vp, scene, film, camera; samples)
    end
    Mantle.waitidle(Hikari.mantle_device(backend))
    lanes = vp.state.sobol_rng.lanes
    image = Array(film.framebuffer)
    close(vp)
    return image, film.iteration_index[], lanes
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
    @test Hikari.wave_lanes(1024, 256 * 256) == 8
    @test Hikari.wave_lanes(12, 256 * 256) == 6
    @test Hikari.wave_lanes(7, 256 * 256) == 7
    @test Hikari.wave_lanes(64, 1920 * 1080) == 1
    @test Hikari.wave_lanes(1, 64) == 1
end
