using Test, Hikari, GeometryBasics
import Adapt
import Raycore
import Mantle
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
    backend = Mantle.defaultbackend()
    if !Mantle.supports_rt_pipeline(backend)
        @test_skip false
    else
        test_aux_alignment(backend; hw_accel = true)
    end
end

function emitter_beauty_and_guide(backend; hw_accel = false, guide_kwargs...)
    scene = Hikari.Scene(; backend, hw_accel)
    object = normal_mesh(GeometryBasics.Tessellation(Sphere(Point3f(0), 0.6f0), 64))
    push!(scene, object, Hikari.Emissive(; Le = (1, 1, 1), scale = 1, two_sided = true))
    Hikari.sync!(scene)
    filter = Hikari.GaussianFilter()
    film = Hikari.Film(Point2f(32, 32); filter)
    camera = Hikari.PerspectiveCamera(Point3f(0, 0, 3), Point3f(0), film; fov = 45f0)
    film = Hikari.Film(backend, film)
    Hikari.clear!(film)
    integrator = Hikari.VolPath(; samples = 256, max_depth = 1, hw_accel, filter)
    integrator(scene, film, camera)
    Hikari.fill_aux_buffers!(film, Adapt.adapt(backend, scene), camera; guide_kwargs...)
    luminance = map(c -> c.r + c.g + c.b, Array(film.framebuffer))
    albedo = map(c -> c.g, Array(film.albedo))
    hits = Float32.(isfinite.(Array(film.depth)))
    Hikari.free!(film)
    close(integrator)
    return luminance ./ maximum(luminance), albedo, hits
end

@testset "Guide albedo is filtered like the image" begin
    backend = Mantle.defaultbackend()
    for hw_accel in (Mantle.supports_hwtlas(backend) ? (false, true) : (false,))
        coverage, albedo, _ = emitter_beauty_and_guide(backend; hw_accel)
        @test maximum(abs.(albedo .- coverage)) < 0.05f0
    end
end

@testset "One guide ray per pixel with grid = 1" begin
    backend = Mantle.defaultbackend()
    for hw_accel in (Mantle.supports_hwtlas(backend) ? (false, true) : (false,))
        _, albedo, hits = emitter_beauty_and_guide(backend; hw_accel, grid = 1)
        @test albedo == hits
    end
end

# One ray through `Hikari.intersect!`, in a kernel: whether it hit, where, the
# normal there, and the instance's custom index the interaction carries.
KA.@kernel function intersect_one_kernel!(out, accel, ray)
    hit, _, si = Hikari.intersect!(accel, ray)
    out[1] = (hit, si.core.p, Vec3f(si.core.n), si.instance_id)
end

@testset "intersect! puts a moved instance's hit where the ray meets it" begin
    # `closest_hit` returns the hit triangle in world space and the instance's
    # custom index on every structure. The software `intersect!` moved the
    # interaction by the instance's transform once more and stored the
    # instance's position in the structure, so the aux depth and normal of every
    # moved instance were taken somewhere else.
    backend = Mantle.defaultbackend()
    for hw_accel in (Mantle.supports_hwtlas(backend) ? (false, true) : (false,))
        scene = Hikari.Scene(; backend, hw_accel)
        slab = normal_mesh(Rect3f(Vec3f(-0.5, -0.5, 0), Vec3f(1, 1, 0.01)))
        moved = Raycore.Mat4f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 1, 1)
        hs = push!(scene, slab, [Hikari.Diffuse(Kd = Hikari.RGBSpectrum(0.5f0))], [moved])
        Hikari.sync!(scene)
        out = Adapt.adapt(backend, [(false, Point3f(0), Vec3f(0), UInt32(0))])
        intersect_one_kernel!(backend)(out, Adapt.adapt(backend, scene.accel),
                                       Raycore.Ray(o = Point3f(2, 0.1, 5), d = Vec3f(0, 0, -1)); ndrange = 1)
        KA.synchronize(backend)
        hit, p, n, id = only(Array(out))
        @test hit
        @test p ≈ Point3f(2, 0.1, 1.01)
        @test abs(n[3]) ≈ 1f0
        @test id == only(hs).interface != UInt32(0)
    end
end
