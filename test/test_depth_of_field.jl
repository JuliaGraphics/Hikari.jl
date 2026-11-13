using Test, Hikari, Mantle, GeometryBasics, LinearAlgebra

@testset "f-stop sets the lens radius from the focal length of a 36 mm sensor" begin
    film = Hikari.Film(Point2f(300, 200))
    fov = 2rad2deg(atan(0.018 / 0.05 / 1.5))
    camera = Hikari.PerspectiveCamera(Point3f(0, 0, 5), Point3f(0), film; fov, fstop = 2)
    @test camera.core.lens_radius ≈ 0.05f0 / 4
    @test Hikari.PerspectiveCamera(Point3f(0, 0, 5), Point3f(0), film; fov).core.lens_radius == 0f0
end

@testset "Focus defaults to the look-at point, else a distance or a point's depth" begin
    film = Hikari.Film(Point2f(30, 20))
    focus(; kw...) = Hikari.PerspectiveCamera(Point3f(0, 0, 5), Point3f(0), film; fstop = 2, kw...).core.focal_distance
    @test focus() ≈ 5f0
    @test focus(focus = 3) ≈ 3f0
    @test focus(focus = Point3f(1, 2, 1)) ≈ 4f0
end

@testset "Bladed apertures sample a regular polygon" begin
    us = [Point2f(i, j) ./ 64f0 for i in 0.5f0:63.5f0 for j in 0.5f0:63.5f0]
    @test Hikari.sample_aperture.(us, Int32(0), 0f0) == Hikari.concentric_sample_disk.(us)

    diamond = Hikari.sample_aperture.(us, Int32(4), 0f0)
    @test maximum(p -> abs(p[1]) + abs(p[2]), diamond) <= 1f0 + 1f-6
    @test maximum(p -> p[1], diamond) > 0.95f0
    @test sum(diamond) / length(diamond) ≈ Point2f(0) atol = 1f-3

    square = Hikari.sample_aperture.(us, Int32(4), Float32(π / 4))
    @test maximum(p -> max(abs(p[1]), abs(p[2])), square) <= sqrt(0.5f0) + 1f-6
end

function render_dot_out_of_focus(; kw...)
    backend = Mantle.defaultbackend()
    scene = Hikari.Scene(; backend)
    push!(scene, normal_mesh(Tessellation(Sphere(Point3f(0), 0.05f0), 24)), Hikari.Emission(color = (1, 1, 1), strength = 20))
    Hikari.sync!(scene)
    film = Hikari.Film(backend, Hikari.Film(Point2f(48, 48)))
    camera = Hikari.PerspectiveCamera(Point3f(0, -4, 0), Point3f(0), film; up = Vec3f(0, 0, 1), fov = 20, kw...)
    integrator = Hikari.VolPath(; samples = 64, max_depth = 1)
    integrator(scene, film, camera)
    image = Array(film.framebuffer)
    close(integrator)
    return count(c -> c.r > 0.02f0, image)
end

@testset "An out-of-focus light spreads into a disk" begin
    sharp = render_dot_out_of_focus()
    @test abs(render_dot_out_of_focus(fstop = 1, focus = 4) - sharp) <= sharp ÷ 4
    @test render_dot_out_of_focus(fstop = 1, focus = 1) > 4sharp
end
