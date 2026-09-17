# MWE-7 (Hikari side): minimum repro of the iter-6 cascade fault.
#
# Found 2026-04-25 — minimum cross-scene scene-switch repro:
#   1. Build scene A (sphere + point light)
#   2. Render A once (compiles 19 pipelines, 38 kernels)
#   3. Build scene B (floor + sphere + light, different)  → adds 1 new pipeline
#   4. Render B once                                       → crashes with
#                                                            PERMISSION_FAULTS=7
#
# Asymmetry test: B→A is clean (B's pipelines are a superset of A's, so
# A's render needs no new compilation).  Same-shape A→A2 is clean too.
# Adding sync + GC.gc() between A and B did not prevent the crash, but
# `GC.enable(false)` around B's render did — confirming it's a GC race
# specifically during B's recording window.
#
# Root cause (fixed 2026-04-25): a freshly-allocated `VkManagedBuffer`
# with `last_write == nothing` could be destroyed by a finalizer-thread
# `vk_free!` mid-recording — the active-batch defer branch only fired
# when `last_write !== nothing`.  Fix:
#   * `pin_leaves!(LavaArray)` now also calls `pin!(buf)` on the
#     underlying `VkManagedBuffer` (was only pinning the wrapper).
#   * `pin!(::CommandBatch, ::VkManagedBuffer)` eagerly advances
#     `buf.last_write` to `(bq, batch.signal_value)` so vk_free!'s
#     active-batch check (memory.jl:289) catches it.
#
# Gated on LAVA_RUN_CASCADE_REPRO=1 because DEVICE_LOST is unrecoverable
# in-session: if a regression breaks this, the whole test session is
# poisoned, so we don't run it in the default test target.

using Test
using Hikari, Lava, Raycore, Adapt
using GeometryBasics
using GeometryBasics: normal_mesh, Tesselation, Rect3f, Sphere, Point3f, Vec3f, Point2f

const _BE_M7  = Mantle.defaultbackend()
const _CTX_M7 = Mantle.vk_context()

_build_a(backend) = let s = Hikari.Scene(; backend=backend, hw_accel=true)
    push!(s, normal_mesh(Tesselation(Sphere(Point3f(0,0,0.35), 0.35f0), 8)),
          Hikari.Diffuse(Kd=Hikari.RGBSpectrum(0.6f0, 0.6f0, 0.6f0)))
    push!(s, Hikari.PointLight(Point3f(0,-2,3), Hikari.RGBSpectrum(30f0)))
    Hikari.sync!(s); s
end

_build_b(backend) = let s = Hikari.Scene(; backend=backend, hw_accel=true)
    push!(s, normal_mesh(Rect3f(Vec3f(-1,-1,-0.05), Vec3f(2,2,0.05))),
          Hikari.Diffuse(Kd=Hikari.RGBSpectrum(0.4f0, 0.7f0, 0.4f0)))
    push!(s, normal_mesh(Tesselation(Sphere(Point3f(0,0,0.35), 0.35f0), 16)),
          Hikari.Diffuse(Kd=Hikari.RGBSpectrum(0.8f0, 0.2f0, 0.2f0)))
    push!(s, Hikari.PointLight(Point3f(1,-2,3), Hikari.RGBSpectrum(20f0)))
    Hikari.sync!(s); s
end

function _render_one(scene)
    cam = Hikari.PerspectiveCamera(Point3f(0,-3,1.5), Point3f(0,0,0.35),
                                   Hikari.Film(Point2f(16,16)); fov=50f0)
    film = Hikari.Film(Point2f(16,16))
    gpu_film = Hikari.Film(_BE_M7, film)
    vp = Hikari.VolPath(samples=1, max_depth=2, hw_accel=true)
    vp(scene, gpu_film, cam)
    img = Array(gpu_film.framebuffer)
    close(vp)
    return img
end

# This test is intentionally NOT inside the normal HW RT testset because
# DEVICE_LOST is unrecoverable in-session.  It needs to run alone in a
# fresh Julia, e.g. with `julia --project=. test_cascade_scene_switch_mwe.jl`.
# The CI runner skips it unless LAVA_RUN_CASCADE_REPRO=1 is set.
if get(ENV, "LAVA_RUN_CASCADE_REPRO", "0") == "1"
    @testset "MWE-7: A→B scene switch (cascade fix regression)" begin
        sa = _build_a(_BE_M7)
        _render_one(sa)
        @test !Mantle.device_lost(_CTX_M7)  # A render must be clean

        sb = _build_b(_BE_M7)
        crashed = false
        try
            _render_one(sb)
        catch e
            crashed = true
            @info "MWE-7 caught exception: $(typeof(e))"
        end
        crashed = crashed || Mantle.device_lost(_CTX_M7)
        @test !crashed   # was @test_broken before the pin!/pin_leaves! fix.
    end
end
