# Shadow rays through surfaces that let light through, as Cycles traces them: the
# traversal stops at opaque geometry itself and hands back the nearest hits on the
# rest (`Raycore.transparent_hits`), which the shadow ray shades in order. A
# cutout passes `1 - alpha` of the light, deterministically.
#
# A shadow ray decided each cutout by a coin: with one shadow ray a pixel and a
# point light, the ground under a half-transparent sheet was speckled black and
# lit, pixel by pixel. And it walked the surfaces one closest hit at a time.

using Test
using Hikari
using Hikari: RGBSpectrum
using GeometryBasics
using GeometryBasics: Point2f, Point3f, Vec3f
using Raycore
using Mantle
using Statistics: mean, std

@testset "opaque: what a mesh is pushed with" begin
    @test Hikari.opaque(Hikari.Diffuse(Kd = (0.8, 0.8, 0.8)))
    @test !Hikari.opaque(Hikari.Diffuse(Kd = RGBSpectrum(0.8f0, 0.8f0, 0.8f0, 0.5f0)))
    @test !Hikari.opaque(Hikari.Diffuse(Kd = Hikari.Texture(fill(RGBSpectrum(0.5f0, 0.5f0, 0.5f0, 0f0), 2, 2))))
    @test Hikari.opaque(Hikari.Diffuse(Kd = Hikari.Texture(fill(RGBSpectrum(0.5f0, 0.5f0, 0.5f0, 1f0), 2, 2))))
    @test !Hikari.opaque(Hikari.MediumInterface(Hikari.NullMaterial()))
    @test !Hikari.opaque(Hikari.MediumInterface(Hikari.Diffuse(); shadows = 1.33))
    @test Hikari.opaque(Hikari.MediumInterface(Hikari.Diffuse()))
    @test Hikari.opaque(Hikari.Conductor())

    # The software structure carries it in the instance's flags.
    scene = Hikari.Scene(; backend = Mantle.defaultbackend())
    quad = GeometryBasics.normal_mesh(GeometryBasics.Rect3f(Point3f(-1, -1, 0), Vec3f(2, 2, 0.01)))
    solid = push!(scene, quad, Hikari.Diffuse())
    cut = push!(scene, quad, Hikari.Diffuse(Kd = RGBSpectrum(0.8f0, 0.8f0, 0.8f0, 0.5f0)))
    isopaque(h) = Raycore.is_opaque(only(Raycore.get_instances(scene.accel, h.geometry)))
    @test isopaque(solid) && !isopaque(cut)
    # A material that lets light through makes the mesh see-through.
    Hikari.update_material!(scene, solid, Hikari.Diffuse(Kd = RGBSpectrum(0.8f0, 0.8f0, 0.8f0, 0.5f0)))
    @test !isopaque(solid)
end

# The ground seen from under `layers` sheets of constant `alpha`, lit by a point
# light above them: the mean and spread of its brightness. Direct light only, so
# what reaches the ground is the light times what the sheets let through.
function ground_under_sheets(layers, alpha; hw)
    backend = Mantle.defaultbackend()
    scene = Hikari.Scene(; backend, hw_accel = hw)
    ground = GeometryBasics.normal_mesh(GeometryBasics.Rect3f(Point3f(-4, -4, -0.01), Vec3f(8, 8, 0.01)))
    push!(scene, ground, Hikari.Diffuse(Kd = (0.8, 0.8, 0.8)))
    for k in 1:layers
        z = 1f0 + 0.2f0 * k
        sheet = GeometryBasics.normal_mesh(GeometryBasics.Mesh(
            [Point3f(-4, -4, z), Point3f(4, -4, z), Point3f(4, 4, z), Point3f(-4, 4, z)],
            [GeometryBasics.GLTriangleFace(1, 2, 3), GeometryBasics.GLTriangleFace(1, 3, 4)]))
        push!(scene, sheet, Hikari.Diffuse(Kd = RGBSpectrum(0.8f0, 0.8f0, 0.8f0, Float32(alpha))))
    end
    push!(scene, Hikari.PointLight(Point3f(0, 0, 4), RGBSpectrum(60f0)))
    Hikari.sync!(scene)
    film = Hikari.Film(backend, Hikari.Film(Point2f(32, 24)))
    camera = Hikari.PerspectiveCamera(Point3f(0, -2, 0.6f0), Point3f(0, 0, 0), film; fov = 40f0)
    vp = Hikari.VolPath(samples = 1, max_depth = 1, hw_accel = hw)
    vp(scene, film, camera)
    img = Array(film.framebuffer)
    close(vp)
    lum = [p.r + p.g + p.b for p in img]
    return mean(lum), std(lum) / mean(lum)
end

@testset "a cutout passes 1 - alpha of the light ($(hw ? "hardware" : "software") tracing)" for hw in (false, true)
    open_sky, spread0 = ground_under_sheets(0, 1; hw)
    # One sheet passing half: half the light, as smooth as without it.
    half, spread1 = ground_under_sheets(1, 0.5; hw)
    @test half / open_sky ≈ 0.5 rtol = 0.02
    @test spread1 < spread0 + 0.02
    # Six sheets each passing 0.8, more than a query keeps (four): traced on from
    # the fourth, 0.8^6 of the light.
    six, _ = ground_under_sheets(6, 0.2; hw)
    @test six / open_sky ≈ 0.8^6 rtol = 0.02
    # Past ten see-through surfaces a shadow ray counts as blocked (Cycles'
    # transparent bounce limit).
    @test first(ground_under_sheets(11, 0.2; hw)) == 0
end
