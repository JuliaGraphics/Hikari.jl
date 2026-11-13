using Test, Hikari, GeometryBasics
import Adapt
import Raycore
import Mantle
import KernelAbstractions as KA
using Statistics: mean
include("software_backend.jl")

# An off-centre emitter makes a Y flip observable. Compare independently
# rendered radiance with primary-hit depth, not with a copy of the indexing
# expression under test. Run the same check on CPU and the hardware RT path.
function test_aux_alignment(backend; hw_accel = false)
    scene = Hikari.Scene(; backend, hw_accel)
    object = normal_mesh(GeometryBasics.Tessellation(Sphere(Point3f(0, 0.55, 0), 0.30f0), 24))
    push!(scene, object, Hikari.Emission(; color = (1, 1, 1), strength = 1, two_sided = true))
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
    reference = software_reference_backend()
    if reference === nothing
        @test_skip false # No CPU Vulkan driver installed.
    else
        test_aux_alignment(reference)
    end
    backend = Mantle.defaultbackend()
    if !Mantle.supports_hwtlas(backend)
        @test_skip false
    else
        test_aux_alignment(backend; hw_accel = true)
    end
end

function emitter_beauty_and_guide(backend; hw_accel = false, guide_kwargs...)
    scene = Hikari.Scene(; backend, hw_accel)
    object = normal_mesh(GeometryBasics.Tessellation(Sphere(Point3f(0), 0.6f0), 64))
    push!(scene, object, Hikari.Emission(; color = (1, 1, 1), strength = 1, two_sided = true))
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

# A guide fill runs once per offscreen frame. It built and uploaded the film
# filter's sampling tables on every call, four device arrays only the GC could
# free; past the pool's soft cap each one started a full collection, which cost
# crown every other render 0.25-0.45 s. The film uploads them once now.
livecount(pool) = sum(b -> length(b.live), Iterators.flatten(values(pool.blocks)); init = 0)

@testset "A guide fill allocates no device memory" begin
    backend = Mantle.defaultbackend()
    pool = Mantle.pool(Hikari.mantle_device(backend))
    for hw_accel in (Mantle.supports_hwtlas(backend) ? (false, true) : (false,))
        scene = Hikari.Scene(; backend, hw_accel)
        push!(scene, normal_mesh(GeometryBasics.Tessellation(Sphere(Point3f(0), 0.6f0), 16)),
              Hikari.Diffuse(color = (0.5, 0.5, 0.5)))
        Hikari.sync!(scene)
        film = Hikari.Film(Point2f(16, 16); filter = Hikari.GaussianFilter())
        camera = Hikari.PerspectiveCamera(Point3f(0, 0, 3), Point3f(0), film; fov = 45f0)
        film = Hikari.Film(backend, film)
        # Adapted per fill, as RayMakie's `refresh_guides!` does.
        Hikari.fill_aux_buffers!(film, Adapt.adapt(backend, scene), camera; grid = 1)
        before = livecount(pool)
        Hikari.fill_aux_buffers!(film, Adapt.adapt(backend, scene), camera; grid = 1)
        @test livecount(pool) == before
        Hikari.free!(film)
    end
end

@testset "One guide ray per pixel with grid = 1" begin
    backend = Mantle.defaultbackend()
    for hw_accel in (Mantle.supports_hwtlas(backend) ? (false, true) : (false,))
        _, albedo, hits = emitter_beauty_and_guide(backend; hw_accel, grid = 1)
        @test albedo == hits
    end
end

