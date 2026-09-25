# Printing a `VolPathState` walked every work queue element by element, each a
# read of device memory, and ran for longer than 20 minutes. It is what a failing
# `@test vp.state === held` prints, so the test that should have explained a bug
# hung instead.

using Test, Hikari, Mantle, GeometryBasics
using GeometryBasics: normal_mesh, Tesselation, Sphere

@testset "a VolPathState prints its shape, not its buffers" begin
    backend = Mantle.defaultbackend()
    scene = Hikari.Scene(; backend = backend)
    sphere = normal_mesh(Tesselation(Sphere(GeometryBasics.Point3f(0, 0, 0), 0.5f0), 8))
    push!(scene, sphere, Hikari.Diffuse(Kd = Hikari.RGBSpectrum(0.7f0, 0.4f0, 0.4f0)))
    push!(scene, Hikari.PointLight(GeometryBasics.Point3f(0, -2, 3), Hikari.RGBSpectrum(30f0)))
    Hikari.sync!(scene)
    film = Hikari.Film(backend, Hikari.Film(GeometryBasics.Point2f(64, 48)))
    cam = Hikari.PerspectiveCamera(GeometryBasics.Point3f(0, -3, 1.5f0),
                                   GeometryBasics.Point3f(0, 0, 0.5f0), film; fov = 50f0)
    vp = Hikari.VolPath(samples = 1, max_depth = 2)
    Hikari.render!(vp, scene, film, cam)

    shown = sprint(show, vp.state)
    @test startswith(shown, "VolPathState(64×48, 1 lights, max_depth 2, ")
    @test endswith(shown, "plans recorded)")
    @test length(shown) < 200
    # Inside a container, the way a test failure meets it.
    @test length(sprint(show, (vp.state, 1))) < 200

    close(vp)
end
