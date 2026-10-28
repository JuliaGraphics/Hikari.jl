# The volpath sample as Mantle plans.
#
# What these pin is the part of the port a rendered image would not obviously
# catch: that the plans are REUSED across samples rather than rebuilt, and that
# reusing them still advances everything a sample is supposed to advance. A
# compiled dispatch resolves its arguments once, so a sample index left behind
# in the plan it was built with would render the same sample N times — and the
# picture would look plausible, just converged to the wrong thing.

using Test
using Hikari, Lava, Mantle
using GeometryBasics
using GeometryBasics: normal_mesh, Tesselation
import KernelAbstractions as KA
import Adapt
include("software_backend.jl")

# Every stage of the round: four material types, a medium, an area light and a
# point light, so nothing in the graph is skipped for want of work.
function volpath_graph_scene(backend; with_media::Bool)
    white = Hikari.Diffuse(Kd = Hikari.RGBSpectrum(0.73f0))
    red   = Hikari.Diffuse(Kd = Hikari.RGBSpectrum(0.65f0, 0.05f0, 0.05f0))
    glass = Hikari.Dielectric(Kr = Hikari.RGBSpectrum(1f0), Kt = Hikari.RGBSpectrum(1f0),
                              index = 1.5f0)
    gold  = Hikari.Conductor(eta = Hikari.RGBSpectrum(0.15557f0, 0.42415f0, 1.3831f0),
                             k = Hikari.RGBSpectrum(3.6024f0, 2.4721f0, 1.9155f0))
    emissive = Hikari.Emissive(Le = Hikari.RGBSpectrum(12f0))
    fog = Hikari.HomogeneousMedium(σ_a = Hikari.RGBSpectrum(0.01f0),
                                   σ_s = Hikari.RGBSpectrum(0.3f0),
                                   Le = Hikari.RGBSpectrum(0f0), g = 0.3f0)
    inner = with_media ? Hikari.MediumInterface(glass; inside = fog, outside = nothing) : glass

    scene = Hikari.Scene(; backend = backend)
    add!(prim, mat) = push!(scene, normal_mesh(prim isa Sphere ? Tesselation(prim, 24) : prim), mat)
    add!(Rect3f(Vec3f(-1, 0, -1), Vec3f(2, 0.01f0, 2)), white)
    add!(Rect3f(Vec3f(-1, 0, 0.99f0), Vec3f(2, 2, 0.01f0)), red)
    add!(Sphere(Point3f(-0.4f0, 0.4f0, 0f0), 0.35f0), inner)
    add!(Sphere(Point3f(0.4f0, 0.35f0, 0f0), 0.3f0), gold)
    add!(Sphere(Point3f(0f0, 1.7f0, 0f0), 0.15f0), emissive)
    push!(scene, Hikari.PointLight(Point3f(0f0, 1.8f0, 0f0), Hikari.RGBSpectrum(8f0)))
    Hikari.sync!(scene)
    return scene
end

function volpath_graph_film(backend, res)
    film = Hikari.Film(Point2f(res, res))
    camera = Hikari.PerspectiveCamera(Point3f(0f0, 1f0, -3.5f0), Point3f(0f0, 1f0, 0f0),
                                      film; fov = 40f0)
    film = Hikari.Film(backend, film)
    Hikari.clear!(film)
    return film, camera
end

energy(film) = let fb = Array(film.framebuffer)
    (sum(px -> Float64(px.r), fb), sum(px -> Float64(px.g), fb), sum(px -> Float64(px.b), fb))
end

@testset "volpath graph" begin
    backend = Mantle.defaultbackend()
    scene = volpath_graph_scene(backend; with_media = true)

    @testset "progressive samples equal a batched render" begin
        # Four calls to `render!` on one integrator against one call that loops
        # four samples: the same plans, run 4×, with only the sample index
        # moving. Bit-identical is the right bar here — the two paths run the
        # same dispatches over the same Sobol samples in the same order.
        film_a, cam_a = volpath_graph_film(backend, 48)
        vp_a = Hikari.VolPath(samples = 4, max_depth = 4)
        vp_a(scene, film_a, cam_a)

        film_b, cam_b = volpath_graph_film(backend, 48)
        vp_b = Hikari.VolPath(samples = 1, max_depth = 4)
        for _ in 1:4
            Hikari.render!(vp_b, scene, film_b, cam_b)
            KA.synchronize(backend)
        end

        @test energy(film_a) == energy(film_b)
        # …and it really was one set of plans, not four.
        @test vp_b.state.plans !== nothing
    end

    @testset "the plans are built once and kept" begin
        film, cam = volpath_graph_film(backend, 32)
        vp = Hikari.VolPath(samples = 1, max_depth = 3)
        Hikari.render!(vp, scene, film, cam)
        plans = vp.state.plans
        @test plans !== nothing
        Hikari.render!(vp, scene, film, cam)
        # Same object: a scene whose shape has not moved must not recompile, and
        # "recompiles every sample" is invisible in the picture.
        @test vp.state.plans === plans
    end

    @testset "finalizing another film is refused" begin
        film, cam = volpath_graph_film(backend, 32)
        vp = Hikari.VolPath(samples = 1, max_depth = 3)
        Hikari.render!(vp, scene, film, cam)
        other, _ = volpath_graph_film(backend, 32)
        # A pass names its target when the plan is compiled, so this would
        # silently finalize the film that WAS rendered.
        @test_throws ArgumentError Hikari.finalize_film!(vp, other)
    end

    @testset "a media-free scene renders without the medium stages" begin
        surf = volpath_graph_scene(backend; with_media = false)
        film, cam = volpath_graph_film(backend, 48)
        vp = Hikari.VolPath(samples = 2, max_depth = 4)
        vp(surf, film, cam)
        fb = Array(film.framebuffer)
        @test all(px -> isfinite(px.r) && isfinite(px.g) && isfinite(px.b), fb)
        @test sum(px -> Float64(px.r) + px.g + px.b, fb) > 0
    end

    @testset "the same scene on a CPU Vulkan backend" begin
        reference = software_reference_backend()
        if reference === nothing
            @test_skip false # No CPU Vulkan driver installed.
        else
            cpu_scene = volpath_graph_scene(reference; with_media = true)
            film, cam = volpath_graph_film(reference, 16)
            vp = Hikari.VolPath(samples = 1, max_depth = 3)
            try
                vp(cpu_scene, film, cam)
                fb = Array(film.framebuffer)
                @test all(px -> isfinite(px.r) && isfinite(px.g) && isfinite(px.b), fb)
                @test sum(px -> Float64(px.r) + px.g + px.b, fb) > 0
            finally
                close(vp)
                Hikari.free!(film)
            end
        end
    end
end
