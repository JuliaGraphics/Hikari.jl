# A scene whose ONLY object is a medium boundary.
#
# `MediumInterface(NullMaterial(); inside = m)` is pbrt's `Material "interface"`:
# a surface that swaps the ray's medium and does nothing else. A `NullMaterial`
# is never pushed into `scene.materials`, so a scene containing nothing but a
# volume has an EMPTY material set — which is exactly what RayMakie builds for
# `volume!` as the only plot in a scene.
#
# Three call sites asked `with_index(…, materials, idx, …)` about that invalid
# index on the way to the null-material branch that handles it:
# `get_surface_alpha_dispatch`, `is_mix_material_dispatch` and
# `get_perturbed_shading_frame`. `with_index` on an EMPTY `StaticMultiTypeSet`
# expands to `error("with_index: empty StaticMultiTypeSet")`, which a GPU kernel
# cannot raise — so every ray that reached the cube vanished, producing neither a
# continuation ray nor an escape, and the volume rendered as a solid black box at
# exactly zero radiance.
#
# With ANY other material in the scene the same invalid index instead fell into
# `with_index`'s else-branch and read an ARBITRARY material's alpha and
# displacement. That happened to be harmless for an opaque un-bumped material,
# which is why adding an unrelated, off-screen sphere "fixed" the render — and
# why the identity assertion below is the one that matters: an unused material
# type must change nothing.

using Test
using Hikari, Mantle, Raycore, GeometryBasics
using GeometryBasics: normal_mesh, Tesselation, Rect3f, Sphere, Point3f, Point2f, Vec3f
using Colors: red, green, blue
import KernelAbstractions as KA

# A Gaussian blob of pure scattering, dense enough that a ray entering the cube
# cannot cross it unscattered.
function null_only_medium(bounds)
    d = Float32[exp(-((x - 8)^2 + (y - 8)^2 + (z - 8)^2) / 30) for x in 1:16, y in 1:16, z in 1:16]
    nx, ny, nz = size(d)
    σ_s = [Hikari.RGBSpectrum(d[i, j, k], d[i, j, k], d[i, j, k]) for i in 1:nx, j in 1:ny, k in 1:nz]
    return Hikari.RGBGridMedium(; σ_a_grid = fill(Hikari.RGBSpectrum(0.0f0), nx, ny, nz),
                                σ_s_grid = σ_s, sigma_scale = 100.0f0, g = 0.85f0,
                                bounds, majorant_res = Vec{3, Int64}(16, 16, 16))
end

# `unused_material` adds a material TYPE with no geometry referencing it — it
# cannot change a single pixel, and before the fix it changed everything.
function null_only_render(backend; unused_material::Bool, samples = 8, max_depth = 32,
                          res = 48, hw_accel = true)
    # The SCENE decides the traversal: `hw_accel` on the integrator alone left
    # both cases on the software BVH.
    scene = Hikari.Scene(; backend, hw_accel)
    bounds = Raycore.Bounds3(Point3f(0), Point3f(2))
    push!(scene, normal_mesh(Rect3f(Vec3f(0), Vec3f(2))),
          Hikari.MediumInterface(Hikari.NullMaterial(); inside = null_only_medium(bounds)))
    unused_material && push!(scene.materials, Hikari.Diffuse(color = Hikari.RGBSpectrum(1.0f0)))
    push!(scene, Hikari.PointLight(Point3f(4, -6, 6), Hikari.RGBSpectrum(60.0f0)))
    Hikari.sync!(scene)

    film = Hikari.Film(backend, Hikari.Film(Point2f(res, res);
                                            filter = Hikari.LanczosSincFilter(Point2f(1.0f0), 3.0f0),
                                            crop_bounds = Hikari.Bounds2(Point2f(0.0f0), Point2f(1.0f0)),
                                            diagonal = 1.0f0, scale = 1.0f0))
    view = Hikari.look_at(Point3f(4, -6, 4), Point3f(1, 1, 1), Vec3f(0, 0, 1))
    cam = Hikari.PerspectiveCamera(view, Hikari.Bounds2(Point2f(-1, -1), Point2f(1, 1)),
                                   0.0f0, 1.0f0, 0.0f0, 1.0f6, 45.0f0, film)
    Hikari.VolPath(; samples, max_depth, hw_accel)(scene, film, cam)
    KA.synchronize(backend)
    return Array(film.framebuffer)
end

@testset "a scene whose only material is a null medium boundary" begin
    backend = Mantle.defaultbackend()

    @testset "the material set really is empty" begin
        scene = Hikari.Scene(; backend)
        bounds = Raycore.Bounds3(Point3f(0), Point3f(2))
        push!(scene, normal_mesh(Rect3f(Vec3f(0), Vec3f(2))),
              Hikari.MediumInterface(Hikari.NullMaterial(); inside = null_only_medium(bounds)))
        Hikari.sync!(scene)
        # The premise of everything below. A NullMaterial is "no surface", so it
        # contributes no type — this is not an accident to be worked around.
        @test isempty(scene.materials.data_order)
        @test length(scene.media_interfaces) == 1
    end

    for hw_accel in (Mantle.supports_hwtlas(backend) ? (false, true) : (false,))
        @testset "renders, hw_accel=$hw_accel" begin
            img = null_only_render(backend; unused_material = false, hw_accel)
            lum = c -> (red(c) + green(c) + blue(c)) / 3
            # Was exactly 0.0 across the whole frame.
            @test maximum(lum, img) > 0.01
            # And not a handful of stray pixels: the lit cube covers a real area.
            @test count(c -> lum(c) > 0.001, img) > 100
        end

        @testset "an unused material type changes nothing, hw_accel=$hw_accel" begin
            # The scenes differ only by a material type no geometry references, so
            # the two renders must agree. Same sampler seed, same everything.
            without = null_only_render(backend; unused_material = false, hw_accel)
            with = null_only_render(backend; unused_material = true, hw_accel)
            @test size(without) == size(with)
            @test maximum(abs, Float64.(red.(without)) .- Float64.(red.(with))) < 1.0e-5
            @test maximum(abs, Float64.(green.(without)) .- Float64.(green.(with))) < 1.0e-5
            @test maximum(abs, Float64.(blue.(without)) .- Float64.(blue.(with))) < 1.0e-5
        end
    end

    @testset "the host backend agrees" begin
        # On the CPU the same defect was a THROWN `with_index: empty
        # StaticMultiTypeSet` rather than a silently dead invocation, which is
        # how it was found. Cheap, and it fails loudly if a fourth call site
        # ever appears.
        img = null_only_render(KA.CPU(); unused_material = false, hw_accel = false,
                               samples = 2, max_depth = 8, res = 16)
        @test maximum(c -> (red(c) + green(c) + blue(c)) / 3, img) > 0.01
    end
end
