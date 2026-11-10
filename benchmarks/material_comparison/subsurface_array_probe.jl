module SubsurfaceArrayProbe
using Hikari, Mantle, GeometryBasics, JSON3, Test, Statistics
using GeometryBasics: Point2f, Point3f, Vec3f

function run(output; hw_accel = false)
    backend = Mantle.defaultbackend()
    scene = Hikari.Scene(; backend, hw_accel)
    sphere = normal_mesh(Tesselation(Sphere(Point3f(0), 1f0), 32))
    ground = normal_mesh(Rect3f(Point3f(-4, -4, -1.05), Vec3f(8, 8, 0.01)))
    push!(scene, sphere, Hikari.Principled(base_color = (0.9, 0.6, 0.5), roughness = 0.4,
        subsurface_weight = 1, subsurface_radius = (1, 0.5, 0.25), subsurface_scale = 0.2))
    push!(scene, ground, Hikari.Diffuse(color = (0.6, 0.6, 0.6)))
    push!(scene, Hikari.PointLight(Point3f(0, -2, 4), Hikari.RGBSpectrum(40f0)))
    Hikari.sync!(scene)
    film = Hikari.Film(backend, Hikari.Film(Point2f(32, 24)))
    camera = Hikari.PerspectiveCamera(Point3f(0, -4, 1), Point3f(0), film; fov = 40f0)
    vp = Hikari.VolPath(; samples = 16, max_depth = 4, hw_accel)
    try
        first_seconds = @elapsed vp(scene, film, camera)
        image = Array(film.framebuffer)
        @test all(p -> isfinite(p.r) && isfinite(p.g) && isfinite(p.b), image)
        brightness = mean(p.r + p.g + p.b for p in image[9:16, 13:20])
        @test brightness > 0.05
        open(io -> JSON3.write(io, (; hw_accel, first_seconds, brightness, passes = true)), output, "w")
        println((; hw_accel, first_seconds, brightness))
    catch err
        open(io -> JSON3.write(io, (; hw_accel, passes = false, error = sprint(showerror, err))), output, "w")
        rethrow()
    finally
        close(vp)
        Hikari.free!(film)
    end
    return nothing
end
end
