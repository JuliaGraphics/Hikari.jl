using Test, Hikari, GeometryBasics
import Adapt
import Lava
import KernelAbstractions as KA
using Statistics: mean

# An off-centre emitter makes a Y flip observable. Compare independently
# rendered radiance with primary-hit depth, not with a copy of the indexing
# expression under test. Run the same check on CPU and the hardware RT path.
function test_aux_alignment(backend; hw_accel = false)
    scene = Hikari.Scene(; backend, hw_accel)
    object = normal_mesh(GeometryBasics.Tessellation(Sphere(Point3f(0, 0.55, 0), 0.30f0), 24))
    push!(scene, object, Hikari.Emissive(; Le = (1, 1, 1), scale = 1, two_sided = true))
    Hikari.sync!(scene)
    film = Hikari.Film(Point2f(48, 48))
    camera = Hikari.PerspectiveCamera(Point3f(0, 0, 3), Point3f(0), film; fov = 45f0)
    film = Hikari.Film(backend, film)
    Hikari.clear!(film)
    integrator = Hikari.VolPath(; samples = 64, max_depth = 2, hw_accel)
    integrator(scene, film, camera)
    Hikari.fill_aux_buffers!(film, Adapt.adapt(backend, scene), camera)
    radiance = Array(film.framebuffer)
    depth = Array(film.depth)
    luminance = map(c -> c.r + c.g + c.b, radiance)
    # Half intensity measures the silhouette rather than the reconstruction
    # filter's low-energy fringe around it.
    lit = luminance .> 0.5f0 * maximum(luminance)
    hit = isfinite.(depth)
    @test count(lit) > 10
    @test count(hit) > 10
    litrow = mean(i[1] for i in findall(lit))
    hitrow = mean(i[1] for i in findall(hit))
    @test abs(litrow - 24.5) > 4
    # The radiance pass is spectrally sampled and filtered; the depth pass is
    # a single centre ray. Allow the resulting boundary fringe at 48 px.
    @test abs(litrow - hitrow) < 1.5
    @test count(lit .& hit) / count(lit .| hit) > 0.65
    @test count(lit .& reverse(hit; dims = 1)) == 0
    Hikari.free!(film)
    close(integrator)
end

@testset "Auxiliary buffers align with radiance" begin
    test_aux_alignment(KA.CPU())
    backend = Lava.LavaBackend()
    if Lava.vk_context().rt_pipeline_properties === nothing
        @test_skip false
    else
        test_aux_alignment(backend; hw_accel = true)
    end
end
