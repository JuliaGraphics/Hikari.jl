# An EMPTY scene renders, on both traversal paths.
#
# A software `TLAS` with no BLAS used to type itself over `Triangle{UInt32}`,
# since there was no first BLAS to read the primitive type from. Hikari's
# `resolve_mi_idx` reads `primitive.metadata.medium_interface_idx`, and a bare
# `UInt32` has no such field, so the empty scene's kernel held a field access
# that could only throw. Metal refuses to compile a throw it cannot report
# (`unsupported call to jl_f_getfield`), so every render before the first plot
# landed, and every one after the last was deleted, failed to compile. RayMakie
# met it twice: a scene emptied and refilled, and the `hw_accel` switch.
#
# The hardware default never had this: `HWTLAS{Triangle{TriangleMeta}}` is told
# its primitive type by its type parameter. The software default is now told the
# same thing, so an empty and a populated scene compile to one kernel.

using Test, Hikari, Mantle, Raycore, GeometryBasics, Statistics

const EMPTY_META = Raycore.Triangle{Hikari.TriangleMeta}

@testset "an empty software TLAS is typed over what Hikari pushes" begin
    backend = Mantle.defaultbackend()
    acc = Hikari.default_accel(backend, Val(false))
    @test eltype(acc) == EMPTY_META
    Raycore.sync!(acc)
    @test eltype(acc.static_tlas) == EMPTY_META
    # And a populated one agrees, which is the point: one kernel, not two.
    scene = Hikari.Scene(; backend)
    push!(scene, GeometryBasics.normal_mesh(Sphere(Point3f(0), 0.5f0)),
          Hikari.Diffuse(Kd = Hikari.RGBSpectrum(0.7f0)))
    Hikari.sync!(scene)
    @test eltype(scene.accel) == EMPTY_META
end

@testset "an empty scene renders ($(hw ? "hardware" : "software") traversal)" for hw in (false, true)
    backend = Mantle.defaultbackend()
    # Ungated, as RayMakie's own `hw_accel = true` default is: every backend
    # Hikari runs on has hardware traversal.
    scene = Hikari.Scene(; backend, hw_accel = hw)
    # A light and no geometry: every ray escapes, and the kernel that decides
    # that has to compile.
    push!(scene, Hikari.AmbientLight(Hikari.RGBSpectrum(0.5f0)))
    Hikari.sync!(scene)
    res = 16
    film = Hikari.Film(backend, Hikari.Film(Point2f(res, res)))
    cam = Hikari.PerspectiveCamera(Point3f(0, -3, 1), Point3f(0, 0, 0), film; fov = 50f0)
    vp = Hikari.VolPath(samples = 1, max_depth = 2, hw_accel = hw)
    Hikari.render!(vp, scene, film, cam)
    px = Array(film.framebuffer)
    @test size(px) == (res, res)
    @test all(c -> isfinite(c.r) && isfinite(c.g) && isfinite(c.b), px)
end
