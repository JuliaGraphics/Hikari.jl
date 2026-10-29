# A material updated to another kind of material: another stored type, which no slot of the old
# type can hold. `update_material!` threw ("push it as a new material"), so a node added to a
# RayMakie plot's shader, or a Diffuse swapped for Glass, could not reach the scene. It now takes a
# slot of its type, the interface is pointed at it, and the slot it left is reused by the next
# material of the old type.
using Test, Hikari, Raycore, Mantle, GeometryBasics
using Hikari.TextureNodes
using GeometryBasics: Point2f, Point3f, Vec3f, Rect3f, Sphere, Tesselation, normal_mesh
using Statistics: mean

const TC_BACKEND = Mantle.defaultbackend()

tc_interface(scene, handle) = Raycore.@allowscalar scene.media_interfaces[handle.interface]

function tc_scene(material)
    scene = Hikari.Scene(; backend = TC_BACKEND)
    handle = push!(scene, normal_mesh(Tesselation(Sphere(Point3f(0, 0, 0.5), 0.5f0), 24)), material)
    push!(scene, normal_mesh(Rect3f(Vec3f(-3, -3, -0.01), Vec3f(6, 6, 0.01))), Hikari.Diffuse(color = (0.6, 0.6, 0.6)))
    push!(scene, Hikari.PointLight(Point3f(0, -2, 3), Hikari.RGBSpectrum(30.0f0)))
    Hikari.sync!(scene)
    return scene, handle
end

function tc_render(scene)
    film = Hikari.Film(TC_BACKEND, Hikari.Film(Point2f(48, 48)))
    camera = Hikari.PerspectiveCamera(Point3f(0, -3, 1.5f0), Point3f(0, 0, 0.5f0), film; fov = 50.0f0)
    vp = Hikari.VolPath(samples = 8, max_depth = 4)
    vp(scene, film, camera)
    img = Array(film.framebuffer)
    close(vp)
    return img
end

tc_mean(img) = mean(p -> Float64(p.r + p.g + p.b), img)

@testset "a material updated to another type" begin
    red = Hikari.Diffuse(color = (0.8, 0.1, 0.1))
    glass = Hikari.Glass(ior = 1.5)
    noisy = Hikari.Principled(base_color = noise(xyz()).color, roughness = 0.4)

    @testset "renders as the same material pushed" begin
        for target in (
                glass, noisy, Hikari.Principled(distribution = :ggx, base_color = (0.05, 0.05, 0.05), metallic = 1, roughness = 0.7),
                Hikari.Metallic(fresnel_type = :physical_conductor, ior = (1.5, 1.5, 1.5), extinction = (0.1, 0.1, 0.1), roughness = 0.7),
            )
            scene, handle = tc_scene(red)
            original = tc_render(scene)
            Hikari.update_material!(scene, handle.interface, target)
            updated = tc_render(scene)
            fresh = tc_render(first(tc_scene(target)))
            # Not vacuous: the update changed the image.
            @test abs(tc_mean(updated) - tc_mean(original)) > 0.05 * tc_mean(original)
            # The same sampler sequence over the same scene content: the same image.
            @test updated ≈ fresh
        end
    end

    @testset "the interface moves, the slot it left is reused" begin
        scene, handle = tc_scene(red)
        store = scene.materials
        before = tc_interface(scene, handle).material
        n = length(store)
        Hikari.update_material!(scene, handle, glass)
        moved = tc_interface(scene, handle).material
        @test moved.type_idx != before.type_idx
        @test store[moved] isa Hikari.ShaderMaterial{<:Hikari.ShaderProgram, Nothing, Tuple{Hikari.GlassBSDF}}
        @test length(store) == n
        # Back to the first type: into the slot it left, so the store does not grow.
        Hikari.update_material!(scene, handle, Hikari.Diffuse(color = (0.1, 0.1, 0.8)))
        @test tc_interface(scene, handle).material == before
        @test length(store) == n
        # And a value update of the type it has now is in place again.
        layout = store.layout
        Hikari.update_material!(scene, handle, Hikari.Diffuse(color = (0.1, 0.8, 0.1)))
        @test tc_interface(scene, handle).material == before
        @test store.layout == layout
    end

    @testset "a shader of other node families" begin
        scene, handle = tc_scene(Hikari.Principled(base_color = (0.8, 0.3, 0.2), roughness = 0.4))
        before = tc_interface(scene, handle).material
        Hikari.update_material!(scene, handle, noisy)
        moved = tc_interface(scene, handle).material
        @test moved.type_idx != before.type_idx
        # A value change of the new type stays where it is.
        Hikari.update_material!(scene, handle, Hikari.Principled(base_color = noise(xyz(); scale = 3).color, roughness = 0.4))
        @test tc_interface(scene, handle).material == moved
    end

    @testset "a Principled distribution switches slots, values reuse them" begin
        multi = Hikari.Principled(roughness = 0.4)
        single = Hikari.with_field(multi, :distribution, :ggx)
        scene, handle = tc_scene(multi)
        before = tc_interface(scene, handle).material
        Hikari.update_material!(scene, handle, single)
        moved = tc_interface(scene, handle).material
        @test moved.type_idx != before.type_idx
        Hikari.update_material!(scene, handle, Hikari.with_field(single, :roughness, 0.7))
        @test tc_interface(scene, handle).material == moved
        Hikari.update_material!(scene, handle, multi)
        @test tc_interface(scene, handle).material == before
        Hikari.update_material!(scene, handle, single)
        @test tc_interface(scene, handle).material == moved
    end

    # A mix occupies one graph slot, with no separately stored sub-materials.
    @testset "a mix in and out" begin
        scene, handle = tc_scene(red)
        store = scene.materials
        n = length(store)
        mix = Hikari.Mix(Hikari.Gold(), Hikari.CoatedDiffuse(reflectance = (0.4, 0.4, 0.4)); factor = 0.3f0)
        Hikari.update_material!(scene, handle, mix)
        stored = store[tc_interface(scene, handle).material]
        @test stored isa Hikari.ShaderMaterial
        @test fieldcount(Hikari.bsdf_nodes(typeof(stored))) == 2
        @test length(store) == n
        Hikari.update_material!(scene, handle, red)
        @test Hikari.bsdf_nodes(typeof(store[tc_interface(scene, handle).material])) === Tuple{Hikari.DiffuseBSDF}
        @test length(store) == n
        @test !isempty(tc_render(scene))
    end

    @testset "a medium interface's sides" begin
        scene = Hikari.Scene(; backend = TC_BACKEND)
        medium(σ) = Hikari.HomogeneousMedium(σ_a = Hikari.RGBSpectrum(0.1f0), σ_s = Hikari.RGBSpectrum(σ), Le = Hikari.RGBSpectrum(0.0f0))
        cube = normal_mesh(Rect3f(Vec3f(0), Vec3f(1)))
        handle = push!(scene, cube, Hikari.MediumInterface(Hikari.NullMaterial(); inside = medium(0.5f0)))
        mi = tc_interface(scene, handle)
        inside = mi.inside
        @test !Raycore.is_valid(mi.material) && Raycore.is_valid(inside)
        # A surface where there was none, and no medium where there was one.
        Hikari.update_material!(scene, handle.interface, Hikari.MediumInterface(glass))
        mi = tc_interface(scene, handle)
        @test Raycore.is_valid(mi.material) && !Raycore.is_valid(mi.inside)
        @test length(scene.media) == 0
        # The medium back, into the slot it had.
        Hikari.update_material!(scene, handle.interface, Hikari.MediumInterface(glass; inside = medium(0.2f0)))
        @test tc_interface(scene, handle).inside == inside
        @test length(scene.media) == 1
    end
end
