# Phase-M regression: a fresh `Hikari.VolPath` per render-iter must not
# trip a GPU device-lost cascade.
#
# Background (full investigation: `iter6-telemetry.jl`, `iter6-bisect2.jl`,
# `iter6-deeper.jl`, `iter6-gc-disable.jl`):
#
#   - Allocating + dropping a `Hikari.VolPath` per render iter triggered
#     a RADV `GPUVM PERMISSION_FAULT` at iter 5-6 on STRIX_HALO.
#   - Bisect proved the trigger is the per-iter VolPath alloc/free cycle
#     (not the per-iter Film alloc, not dispatch volume).
#   - GC.enable(false) around the render call avoided the crash, so the
#     mechanism is Julia GC's finalizer thread destroying a VkBuffer that
#     the main thread's recording batch still references via an
#     unpinned BDA.
#   - Mitigation: `Hikari.close(::VolPath)` forces a synchronous
#     `GC.gc(false)` after dropping `vp.state`, so finalizers run on the
#     current thread before the next render's recording starts.
#
# This test pins the contract: build/destroy a fresh VolPath each iter,
# render, repeat for at least 12 iters, assert no DEVICE_LOST.

using Test
using Hikari
using Lava, Mantle
using Raycore
using Adapt
using GeometryBasics
using GeometryBasics: normal_mesh, Tesselation, Sphere, Point3f, Vec3f, Point2f
using LinearAlgebra: I

const _BACKEND_M = Mantle.defaultbackend()
const TEST_DEVICE = Hikari.mantle_device(Mantle.defaultbackend())

@testset "VolPath per-iter lifecycle — no cascade fault" begin
    # Tiny scene: minimum HW RT setup to exercise the fault trigger.
    scene = Hikari.Scene(; backend=_BACKEND_M, hw_accel=true)
    sphere = normal_mesh(Tesselation(Sphere(Point3f(0, 0, 0.35), 0.35f0), 6))
    push!(scene, sphere, Hikari.Diffuse(color = Hikari.RGBSpectrum(0.6f0, 0.6f0, 0.6f0)))
    push!(scene, Hikari.PointLight(Point3f(0, -2, 3), Hikari.RGBSpectrum(30f0)))
    Hikari.sync!(scene)
    camera = Hikari.PerspectiveCamera(Point3f(0, -3, 1.5), Point3f(0, 0, 0.35),
                                      Hikari.Film(Point2f(8, 8)); fov=50f0)

    # 12 iters: fresh Film + fresh VolPath each iter — the exact pattern that
    # historically tripped the cascade fault around iter 5-6.
    for iter in 1:12
        film = Hikari.Film(Point2f(8, 8))
        gpu_film = Hikari.Film(_BACKEND_M, film)
        vp = Hikari.VolPath(samples=1, max_depth=1, hw_accel=true)
        vp(scene, gpu_film, camera)
        img = Array(gpu_film.framebuffer)
        close(vp)

        # Hard assertion: device must NOT have gone lost on any iter.
        # If close(vp)'s GC.gc(false) mitigation is removed (or breaks),
        # this trips reliably starting iter 5-6.
        @test !Mantle.retired(TEST_DEVICE)
        # Pixels must be sane (non-degenerate output).
        @test any(p -> (p.r + p.g + p.b) > 1f-4, img)
    end
end
