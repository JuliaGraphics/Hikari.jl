include(joinpath(@__DIR__, "..", "test", "lavapipe.jl"))
using Test, Hikari, Lava, Mantle, GeometryBasics
using GeometryBasics: Point2f, Point3f, Vec3f

function material_cache_probe(; cached = "--cached" in ARGS)
    @test Lava.HAS_OBJECT_CACHE && Lava.ObjCache.enabled()
    backend = Mantle.defaultbackend()
    scene = Hikari.Scene(; backend, hw_accel = false)
    mesh = normal_mesh(Tessellation(Sphere(Point3f(0), 0.7f0), 16))
    push!(scene, mesh, Hikari.Principled(roughness = 0.3, sheen_weight = 0.2))
    push!(scene, Hikari.PointLight(Point3f(-2, -2, 3), Hikari.RGBSpectrum(40f0)))
    Hikari.sync!(scene)
    film = Hikari.Film(backend, Hikari.Film(Point2f(16, 16)))
    camera = Hikari.PerspectiveCamera(Point3f(0, -3, 1), Point3f(0), film; fov = 40f0)
    integrator = Hikari.VolPath(samples = 1, max_depth = 3, hw_accel = false)
    try
        Lava.reset_compile_stats!()
        elapsed = @elapsed integrator(scene, film, camera)
        pixels = Array(film.framebuffer)
        @test all(p -> isfinite(p.r + p.g + p.b), pixels)
        @test any(p -> p.r > 0, pixels)
        stats = Lava.compile_stats()
        llvm = Lava.llvm_compile_stats()
        @info "material cache probe" cached elapsed stats llvm
        if cached
            @test stats.hits > 0
            @test stats.misses == 0
            @test llvm.misses == 0
        end
        return (; elapsed, stats, llvm)
    finally
        close(integrator)
        Hikari.free!(film)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    @testset "material cache across Julia sessions" begin
        material_cache_probe()
    end
end
