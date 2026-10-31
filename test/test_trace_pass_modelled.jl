# The hardware ray-tracing pass is MODELLED by the graph, not smuggled past it.
#
# Until 2026-08-30 `trace_pass!` declared its work with `Mantle.custom!` — the
# escape hatch for a pass that "declares what it touches but not how" — which
# packed its arguments into queue scratch the pool later rewound, so a hardware
# plan could not be recorded. `custom!` has since been deleted from Mantle, and on
# 2026-10-26 the RT-pipeline trace went too: every traversal now traces in the
# compute pass `vp_trace_kernel!`, with an inline ray query on Vulkan and the
# linked intersector on Metal. What stays to check is that a hardware render's
# plans hold only the pass kinds the graph models, that the trace is one of them,
# and that they record.
using Test
using Hikari
using Mantle
using Raycore
using GeometryBasics
using GeometryBasics: normal_mesh, Tesselation, Rect3f, Sphere, Point3f, Vec3f, Point2f

function _trace_scene(backend)
    scene = Hikari.Scene(; backend = backend, hw_accel = true)
    push!(scene, normal_mesh(Rect3f(Vec3f(-3, -3, -0.05), Vec3f(6, 6, 0.05))),
          Hikari.Diffuse(color = Hikari.RGBSpectrum(0.6f0, 0.6f0, 0.6f0)))
    push!(scene, normal_mesh(Tesselation(Sphere(Point3f(0, 0, 0.35), 0.35f0), 16)),
          Hikari.Diffuse(color = Hikari.RGBSpectrum(0.8f0, 0.3f0, 0.2f0)))
    push!(scene, Hikari.PointLight(Point3f(0, -2, 3), Hikari.RGBSpectrum(30f0)))
    Hikari.sync!(scene)
    return scene
end

@testset "the hardware trace is a modelled pass" begin
    backend = Mantle.defaultbackend()
    scene = _trace_scene(backend)
    film = Hikari.Film(Point2f(48, 48))
    camera = Hikari.PerspectiveCamera(Point3f(0, -3, 1.5), Point3f(0, 0, 0.35),
                                      film; fov = 50f0)
    gpu_film = Hikari.Film(backend, film)
    vp = Hikari.VolPath(samples = 4, max_depth = 3, hw_accel = true)
    vp(scene, gpu_film, camera)

    img = Array(gpu_film.framebuffer)
    # First: the render is real. Every assertion below is about a plan, and a
    # plan that rendered nothing would satisfy all of them.
    @test 0.1 < count(px -> (px.r + px.g + px.b) > 1f-4, img) / length(img) < 0.99

    plans = Hikari.allplans(vp.state.plans)
    @test !isempty(plans)

    @testset "no plan reaches for the escape hatch" begin
        modelled = (:compute, :render, :copy, :update)
        for pl in plans
            for pp in pl.passes
                @test pp.pass.kind in modelled
            end
        end
    end

    @testset "the trace is a compute dispatch, and the plans record" begin
        traces = [pp for pl in plans for pp in pl.passes if pp.pass.name == "trace"]
        @test !isempty(traces)
        @test all(pp -> pp.pass.kind === :compute, traces)
        for pl in plans
            Mantle.record!(pl)
            @test Mantle.recorded(pl)
        end
    end

    close(vp)
end
