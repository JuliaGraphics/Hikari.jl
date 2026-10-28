# A ray that passes surfaces without a material still reaches what is behind them.
#
# Passing a surface without a material (a medium's boundary, a see-through emitter) keeps a
# path's depth, but in Hikari's wavefront it takes a round: the ray goes on in the next one. A
# sample traced only a round per depth, so a path that crossed boundaries ran out of rounds
# before its depth, and in the suite's fog cube behind null interfaces 0.4 % of the light was
# missing. Scenes with such surfaces now trace a round more per depth (`sample_rounds`).
#
# Two invisible sheets between the camera and a lit floor: at `max_depth = 2` the floor is
# reached in the third round, so with a round per depth it rendered black.

using Test, Hikari, Mantle, GeometryBasics
using Statistics: mean

function crossing_render(sheets::Int; hw_accel)
    backend = Mantle.defaultbackend()
    scene = Hikari.Scene(; backend, hw_accel)
    quad(z, s) = GeometryBasics.Mesh(Point3f[(-s, -s, z), (s, -s, z), (s, s, z), (-s, s, z)],
                                     GLTriangleFace[(1, 2, 3), (1, 3, 4)])
    push!(scene, quad(0f0, 4f0), Hikari.Diffuse(color = (0.5, 0.5, 0.5)))
    for k in 1:sheets
        push!(scene, quad(0.5f0 * k, 4f0), Hikari.MediumInterface(Hikari.NullMaterial()))
    end
    push!(scene, Hikari.PointLight(Point3f(0, 0, 3), Hikari.RGBSpectrum(10f0)))
    Hikari.sync!(scene)
    film = Hikari.Film(backend, Hikari.Film(Point2f(16, 16)))
    camera = Hikari.PerspectiveCamera(Point3f(0, -0.5, 2), Point3f(0), film; fov = 30f0)
    vp = Hikari.VolPath(; samples = 4, max_depth = 2, hw_accel)
    vp(scene, film, camera)
    img = Array(film.framebuffer)
    close(vp)
    return mean(p.r + p.g + p.b for p in img)
end

@testset "rays reach what is behind surfaces without a material ($(hw ? "hardware" : "software") tracing)" for hw in (false, true)
    open_floor = crossing_render(0; hw_accel = hw)
    @test open_floor > 0
    @test crossing_render(2; hw_accel = hw) ≈ open_floor rtol = 1e-3
end
