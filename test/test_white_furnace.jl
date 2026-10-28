# White furnace: a sphere under a uniform environment of 1, with deep paths. A material that
# neither absorbs nor emits leaves the sphere as bright as the environment, and Cycles' multiscatter
# GGX glass and random walk do, within a fraction of a percent (Blender 5.2.1: 0.9998, 1.0004 and
# 0.9943 for a white glass of roughness 0.1, 0.3 and 0.6).
#
# Regularization roughened a glass lobe after its energy compensation was taken from Cycles' tables,
# for the smoother lobe: a rough glass lost 1.3 % (roughness 0.1) to 3.3 % (0.3) of the light, and
# `regularize` is on by default.
using Test, Hikari, Mantle, GeometryBasics
using GeometryBasics: Point2f, Point3f, Vec3f, Sphere, Tesselation, normal_mesh

function furnace_mean(material; regularize::Bool, res = 96, spp = 256)
    backend = Mantle.defaultbackend()
    scene = Hikari.Scene(; backend)
    push!(scene, normal_mesh(Tesselation(Sphere(Point3f(0), 1f0), 96)), material)
    push!(scene, Hikari.AmbientLight(Hikari.RGBSpectrum(1f0)))
    Hikari.sync!(scene)
    film = Hikari.Film(backend, Hikari.Film(Point2f(res, res)))
    camera = Hikari.PerspectiveCamera(Point3f(0, -4, 0), Point3f(0), film; up = Vec3f(0, 0, 1), fov = 30f0)
    vp = Hikari.VolPath(samples = spp, max_depth = 64, regularize = regularize)
    vp(scene, film, camera)
    img = Array(film.framebuffer)
    close(vp)
    c, r = res ÷ 2, res ÷ 5
    disk = [img[i, j] for i in 1:res, j in 1:res if (i - c)^2 + (j - c)^2 <= r^2]
    return sum(p -> (p.r + p.g + p.b) / 3, disk) / length(disk)
end

@testset "white furnace" begin
    glass(rough) = Hikari.Principled(base_color = (1, 1, 1), transmission_weight = 1, roughness = rough, ior = 1.45)
    for regularize in (false, true), rough in (0.1, 0.3)
        @test furnace_mean(glass(rough); regularize) ≈ 1 atol = 0.003
    end
    walk = Hikari.Principled(base_color = (1, 1, 1), roughness = 0.4, subsurface_weight = 1, subsurface_radius = (1, 1, 1),
                             subsurface_scale = 0.1, ior = 1.45)
    @test furnace_mean(walk; regularize = false) ≈ 1 atol = 0.003
end
