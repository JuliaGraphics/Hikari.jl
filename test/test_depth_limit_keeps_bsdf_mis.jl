# The last vertex of a path is lit through both halves of its MIS.
#
# A vertex's direct light is the sum of two estimates, the light sample and the BSDF
# sample that hits the light, each weighted by the power heuristic. pbrt-v4's VolPath
# traces the BSDF ray of every vertex below the depth limit and adds the emission it
# reaches before it checks the depth, so `maxdepth 1` is the whole direct light of the
# first hit. Hikari stopped at the deepest vertex before it sampled the BSDF: the light
# sample kept its MIS weight, the BSDF's share was never added, and a surface close
# under a large light, where that share is most of it, went dark. The pbrt suite's
# area-light scenes with a sphere under the light came out at 0.946 of pbrt's energy.
#
# A ground plane under an emitter that scatters nothing has no second bounce, so
# its render at `max_depth = 1` must be the one at `max_depth = 2`. Under a shader's
# Emission node, which no light sample finds, the BSDF ray is all of the light; at the
# depth limit only a material that emits is shaded (`emits`), and this one must be.

using Test, Hikari, Mantle, GeometryBasics
using Hikari.TextureNodes: emission
using Statistics: mean

const AREA_LIGHT = Hikari.MediumInterface(Hikari.NullMaterial();
                                          emission = Hikari.Emissive(; Le = (1, 1, 1), scale = 1f0, two_sided = true))

function depth_limit_render(max_depth, emitter; hw_accel)
    backend = Mantle.defaultbackend()
    scene = Hikari.Scene(; backend, hw_accel)
    quad(z, s) = GeometryBasics.Mesh(Point3f[(-s, -s, z), (s, -s, z), (s, s, z), (-s, s, z)],
                                     GLTriangleFace[(1, 2, 3), (1, 3, 4)])
    push!(scene, quad(0f0, 4f0), Hikari.Diffuse(Kd = (0.5, 0.5, 0.5)))
    push!(scene, quad(0.5f0, 1f0), emitter)
    Hikari.sync!(scene)
    film = Hikari.Film(backend, Hikari.Film(Point2f(24, 24)))
    camera = Hikari.PerspectiveCamera(Point3f(0, -0.6, 0.3), Point3f(0), film; fov = 30f0)
    vp = Hikari.VolPath(; samples = 64, max_depth, hw_accel)
    vp(scene, film, camera)
    img = Array(film.framebuffer)
    close(vp)
    return mean(p.r + p.g + p.b for p in img)
end

@testset "the deepest vertex keeps the BSDF's share of its direct light ($(hw ? "hardware" : "software") tracing)" for hw in (false, true)
    @testset "$name" for (name, emitter) in (("an area light", AREA_LIGHT),
                                             ("an Emission shader", Hikari.ShaderMaterial(emission(strength = 1))))
        first_hit, second_bounce = depth_limit_render(1, emitter; hw_accel = hw), depth_limit_render(2, emitter; hw_accel = hw)
        @test second_bounce > 0
        @test first_hit ≈ second_bounce rtol = 0.01
    end
end
