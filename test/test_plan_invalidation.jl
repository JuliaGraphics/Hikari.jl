# Plans are invalidated where the change HAPPENS — nothing is compared per
# sample.
#
# - A moved camera is a `GPURef` update: the same plans object answers, and the
#   picture moves.
# - A scene edit drops the plans of every listening integrator, at the verb:
#   `push!` and a `sync!` that rebuilt. The next `render!` builds new ones.
# - `Hikari.invalidate!` is the verb for the integrator's own fields.
#
# Each is pinned from the outside: the identity of `vp.state.plans`, and the
# render still being right afterwards.

using Test, Hikari, Mantle, Raycore, GeometryBasics, Statistics
using GeometryBasics: normal_mesh, Tesselation, Sphere

# Bound by runtests.jl; bound here so the file also works standalone.

function _inv_scene(backend)
    scene = Hikari.Scene(; backend=backend)
    sphere = normal_mesh(Tesselation(Sphere(GeometryBasics.Point3f(0, 0, 0), 0.5f0), 8))
    push!(scene, sphere, Hikari.Diffuse(Kd=Hikari.RGBSpectrum(0.7f0, 0.4f0, 0.4f0)))
    push!(scene, Hikari.PointLight(GeometryBasics.Point3f(0, -2, 3),
                                   Hikari.RGBSpectrum(30f0)))
    Hikari.sync!(scene)
    return scene
end

_inv_camera(film, x) = Hikari.PerspectiveCamera(
    GeometryBasics.Point3f(x, -3, 1.5f0), GeometryBasics.Point3f(0, 0, 0.5f0),
    film; fov=50f0)

_fbmean(film) = mean(Float32(c.r) for c in Array(film.framebuffer))

@testset "plans are invalidated by events, not found by comparison" begin
    backend = Mantle.defaultbackend()
    scene = _inv_scene(backend)
    res = 16
    film = Hikari.Film(backend, Hikari.Film(GeometryBasics.Point2f(res, res)))
    vp = Hikari.VolPath(samples=1, max_depth=2)
    cam = _inv_camera(film, 0f0)

    Hikari.render!(vp, scene, film, cam)
    plans = vp.state.plans
    @test plans !== nothing
    m1 = _fbmean(film)

    # A moved camera between renders is an update, not a rebuild…
    Hikari.render!(vp, scene, film, _inv_camera(film, 1f0))
    @test vp.state.plans === plans

    # …and it took: the film now holds the average of the two views, which a
    # camera that did NOT move could not produce.
    @test _fbmean(film) != m1

    # A scene edit drops the plans AT THE VERB — before any render asks.
    sphere = normal_mesh(Tesselation(Sphere(GeometryBasics.Point3f(0, 0, 2), 0.3f0), 8))
    push!(scene, sphere, Hikari.Diffuse(Kd=Hikari.RGBSpectrum(0.3f0, 0.7f0, 0.3f0)))
    @test vp.state.plans === nothing

    # And the next render builds new ones that see the edit. "See" is the
    # adapted scene the plans were built from: a software TLAS is adapted as a
    # SNAPSHOT of its node and instance arrays, and a rebuild replaces those, so
    # plans rebuilt from a cached adaptation traced the geometry from before the
    # edit — nine spheres after the scene was down to one (RayMakie's
    # meshscatter stress test, 2026-09-08). The edit drops the adaptation with
    # the plans.
    Hikari.render!(vp, scene, film, cam)
    @test vp.state.plans !== nothing
    @test vp.state.plans !== plans
    # The accel was rebuilt for the edit (the flag `push!` set is consumed by
    # the adapt's `sync!`), and the plans were built from THAT snapshot. With
    # the adaptation cached across the edit neither holds: the flag stays set,
    # the stale snapshot is traced, and the two sides of the second assertion
    # are the same object.
    @test !scene.accel.dirty
    @test vp.adapted.accel === scene.accel.static_tlas

    # The integrator's own fields invalidate through `invalidate!`.
    plans3 = vp.state.plans
    Hikari.invalidate!(vp)
    @test vp.state.plans === nothing
    Hikari.render!(vp, scene, film, cam)
    @test vp.state.plans !== nothing
    @test vp.state.plans !== plans3
end
