"""
A sample is one `run!`, and for a still scene it is nothing else.

Every per-sample value reaches the recorded plans through a `GPURef`, stored
only when it changed (`storechanged!`); the sample index is not even that — the
sample plan's first pass increments it on the device, and the host stores to it
only to start the count over (`resetsamples!`). So after the first sample of a
still scene nothing is dirty, a `render!` leaves nothing dirty, and a moved
camera stores the camera and nothing else. `test_plan_invalidation.jl` pins that
the same plans object answers a moved camera; this pins WHAT the host did for
it.

The last testset — one submission and no allocation per sample — is what an
open command buffer breaks: a run appended to a batch that allocates a segment
for it and submits when something else asks is neither one submission nor
allocation-free. A run is one submission, made at once, and the recorded plan
allocates nothing on the way.
"""

using Test, Hikari, Mantle, Raycore, GeometryBasics
using GeometryBasics: normal_mesh, Tesselation, Sphere, Point3f, Point2f

# Bound by runtests.jl; bound here so the file also works standalone.

function onerun_scene(backend)
    scene = Hikari.Scene(; backend = backend)
    sphere = normal_mesh(Tesselation(Sphere(Point3f(0, 0, 0), 0.5f0), 8))
    push!(scene, sphere, Hikari.Diffuse(Kd = Hikari.RGBSpectrum(0.7f0, 0.4f0, 0.4f0)))
    push!(scene, Hikari.PointLight(Point3f(0, -2, 3), Hikari.RGBSpectrum(30f0)))
    Hikari.sync!(scene)
    return scene
end

onerun_camera(film, x) = Hikari.PerspectiveCamera(
    Point3f(x, -3, 1.5f0), Point3f(0, 0, 0.5f0), film; fov = 50f0)

"""The device's sample count, read back."""
devicesamples(plans) = (Mantle.waitfor!(plans.sample);
                        Array(Mantle.storage(plans.perrun.sample_idx))[1])

@testset "a still scene stores nothing per sample" begin
    backend = Mantle.defaultbackend()
    scene = onerun_scene(backend)
    film = Hikari.Film(backend, Hikari.Film(Point2f(16, 16)))
    vp = Hikari.VolPath(samples = 1, max_depth = 2)
    cam = onerun_camera(film, 0f0)

    Hikari.render!(vp, scene, film, cam)
    plans = vp.state.plans
    # The per-sample refs are among what the sample plan can land a store into,
    # and none holds one: the first sample compared against what the refs were
    # made with and stored nothing.
    hw = plans.sample.hostwritten
    @test any(r -> r === plans.perrun.camera, hw)
    @test any(r -> r === plans.perrun.sample_idx, hw)
    @test !Mantle.anydirty(hw)
    @test !Mantle.anydirty(plans.finalize.hostwritten)

    # The same values again: identity says nothing changed, nothing is stored.
    Hikari.storechanged!(vp.perrun, cam, vp.perrun.stored_medium, vp.perrun.stored_filter)
    @test !Mantle.anydirty(hw)

    Hikari.render!(vp, scene, film, cam)
    @test vp.state.plans === plans
    @test !Mantle.anydirty(hw)

    # The device counted both samples, and agrees with the film's count.
    @test film.iteration_index[] == 2
    @test devicesamples(plans) == 2
end

@testset "a moved camera stores the camera and nothing else" begin
    backend = Mantle.defaultbackend()
    scene = onerun_scene(backend)
    film = Hikari.Film(backend, Hikari.Film(Point2f(16, 16)))
    vp = Hikari.VolPath(samples = 1, max_depth = 2)
    cam = onerun_camera(film, 0f0)
    Hikari.render!(vp, scene, film, cam)
    plans = vp.state.plans

    cam2 = onerun_camera(film, 1f0)
    Hikari.storechanged!(vp.perrun, cam2, vp.perrun.stored_medium, vp.perrun.stored_filter)
    @test Mantle.isdirty(plans.perrun.camera)
    @test !Mantle.isdirty(plans.perrun.initial_medium)
    @test !Mantle.isdirty(plans.perrun.filter_params)
    @test !Mantle.isdirty(plans.perrun.sample_idx)
    @test plans.perrun.stored_camera === cam2

    # The render lands it, in the run's own submission, and leaves nothing.
    Hikari.render!(vp, scene, film, cam2)
    @test vp.state.plans === plans
    @test !Mantle.anydirty(plans.sample.hostwritten)

    # A cleared film is the one event the sample counter hears: the next
    # sample is sample one again, on the device as on the host.
    Hikari.clear!(film)
    @test film.iteration_index[] == 0
    Hikari.render!(vp, scene, film, cam2)
    @test film.iteration_index[] == 1
    @test devicesamples(plans) == 1
end

# A still scene's sample is one `run!` of a recorded plan: one submission, made
# at once, and nothing allocated on the host.
@testset "a sample is one submission and allocates nothing" begin
    backend = Mantle.defaultbackend()
    scene = onerun_scene(backend)
    film = Hikari.Film(backend, Hikari.Film(Point2f(16, 16)))
    vp = Hikari.VolPath(samples = 1, max_depth = 2)
    cam = onerun_camera(film, 0f0)
    for _ in 1:3
        Hikari.render!(vp, scene, film, cam; finalize_framebuffer = false)
    end
    plans = vp.state.plans
    Mantle.waitfor!(plans.sample)

    bq = Mantle.batchqueue(Hikari.mantle_device(backend))
    # `ctxof(bq)`, not `bq.ctx`: a `SubmitChannel` is core's and holds no
    # context — the driver bundle under it does.
    before = Mantle.ctxof(bq).diag.flush_counter[]
    Hikari.render!(vp, scene, film, cam; finalize_framebuffer = false)
    # One sample, one submission: the sample plan's recording, handed over the
    # moment `run!` is called.
    @test Mantle.ctxof(bq).diag.flush_counter[] - before == 1

    # And the sample adds no per-sample HOST work at all: `render!` of a still
    # scene allocates zero bytes. Three things can each cost some, and each is
    # pinned here by the same number: a run re-walking the recording's
    # `IdSet{Any}` of pins (~640 B; `Recording.sync` snapshots them typed at
    # `record!`); `sync_access!` stamped each synced buffer with a boxed
    # `(queue, value)` tuple (48 B a buffer, 2.3 KB a sample on the materials
    # scene; two plain fields now, see `VkManagedBuffer.last_write_bq`); and
    # `storechanged!` was a dynamic call through the `Any`-typed plans that
    # boxed the camera it compared (684 B; it goes through `VolPath.perrun`
    # now, a union over the camera type). Measured on `render!`, not
    # `runsample!`, so the compare is in the count.
    Mantle.waitfor!(plans.sample)
    for _ in 1:3
        Hikari.render!(vp, scene, film, cam; finalize_framebuffer = false)   # warm the concrete path
    end
    Mantle.waitfor!(plans.sample)
    a1 = @allocated Hikari.render!(vp, scene, film, cam; finalize_framebuffer = false)
    Mantle.waitfor!(plans.sample)
    a2 = @allocated Hikari.render!(vp, scene, film, cam; finalize_framebuffer = false)
    Mantle.waitfor!(plans.sample)
    @test a1 == 0
    @test a2 == 0
    @test !Mantle.anydirty(plans.sample.hostwritten)   # no store landed
end
