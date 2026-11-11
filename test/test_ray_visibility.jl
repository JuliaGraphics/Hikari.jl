using Test, Hikari, GeometryBasics, LinearAlgebra, Mantle

@testset "Visibility instance masks" begin
    @test Hikari.instance_mask(Hikari.Visibility()) == 0xff
    @test Hikari.instance_mask(Hikari.Visibility(camera = false)) == 0xfe
    @test Hikari.instance_mask(Hikari.Visibility(diffuse = false)) == 0xfd
    @test Hikari.instance_mask(Hikari.Visibility(glossy = false)) == 0xfb
    @test Hikari.instance_mask(Hikari.Visibility(transmission = false)) == 0xf7
    @test Hikari.instance_mask(Hikari.Visibility(volume_scatter = false)) == 0xef
    @test Hikari.instance_mask(Hikari.Visibility(shadow = false)) == 0xdf
    @test Hikari.instance_mask(Hikari.Visibility(camera = false, shadow = false)) == 0xde
end

@testset "Lobes light sampling leaves out" begin
    hidden(; kw...) = Hikari.hidden_rays(Hikari.Visibility(; kw...))
    @test Hikari.hidden_lobes(hidden(diffuse = false), false) == Hikari.DIFFUSE_RAYS
    @test Hikari.hidden_lobes(hidden(diffuse = false), true) == 0x00
    @test Hikari.hidden_lobes(hidden(glossy = false, camera = false, shadow = false), false) == Hikari.GLOSSY_RAYS
    @test Hikari.hidden_lobes(hidden(transmission = false), false) == 0x00
    @test Hikari.hidden_lobes(hidden(transmission = false), true) == Hikari.DIFFUSE_RAYS | Hikari.GLOSSY_RAYS
    @test Hikari.emitter_hidden_rays(Hikari.Visibility(camera = false, shadow = false)) == 0x00
end

@testset "A light keeps its visibility through updates" begin
    scene = Hikari.Scene(; backend = Mantle.defaultbackend())
    key = push!(scene, Hikari.PointLight(Vec3f(0, 0, 1), Hikari.RGBSpectrum(1f0)); visibility = Hikari.Visibility(diffuse = false))
    @test Hikari.hidden_rays(scene.lights[key]) == Hikari.DIFFUSE_RAYS
    Hikari.update_light!(scene, key, Hikari.PointLight(Vec3f(0, 0, 2), Hikari.RGBSpectrum(1f0)))
    @test Hikari.hidden_rays(scene.lights[key]) == Hikari.DIFFUSE_RAYS
    Hikari.update_light!(scene, key, Hikari.PointLight(Vec3f(0, 0, 2), Hikari.RGBSpectrum(1f0)); visibility = Hikari.Visibility())
    @test Hikari.hidden_rays(scene.lights[key]) == 0x00
end

@testset "Ray type of a bounce" begin
    @test Hikari.bounce_ray_type(Hikari.BXDF_DIFFUSE_REFLECTION) == Hikari.DIFFUSE_RAYS
    @test Hikari.bounce_ray_type(Hikari.BXDF_GLOSSY_REFLECTION) == Hikari.GLOSSY_RAYS
    @test Hikari.bounce_ray_type(Hikari.BXDF_SPECULAR_REFLECTION) == Hikari.GLOSSY_RAYS
    @test Hikari.bounce_ray_type(Hikari.BXDF_DIFFUSE_TRANSMISSION) == Hikari.TRANSMISSION_RAYS
    @test Hikari.bounce_ray_type(Hikari.BXDF_GLOSSY_TRANSMISSION) == Hikari.TRANSMISSION_RAYS
    @test Hikari.bounce_ray_type(Hikari.BXDF_SPECULAR_TRANSMISSION) == Hikari.TRANSMISSION_RAYS
end

function visibility_panel(center, normal; halfsize = 0.6f0)
    n = normalize(normal)
    u = normalize(cross(n, abs(n[3]) < 0.9f0 ? Vec3f(0, 0, 1) : Vec3f(1, 0, 0)))
    v = cross(u, n)
    corners = [Point3f(center .+ halfsize .* (su .* u .+ sv .* v)) for (su, sv) in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return GeometryBasics.Mesh(corners, [TriangleFace(1, 2, 3), TriangleFace(1, 3, 4)]; normal = fill(n, 4))
end

function render_radiance(scene, camera_position, lookat; backend, resolution = (320, 160), samples = 32)
    Hikari.sync!(scene)
    film = Hikari.Film(backend, Hikari.Film(Point2f(resolution...)))
    camera = Hikari.PerspectiveCamera(camera_position, lookat, film; up = Vec3f(0, 0, 1), fov = 50)
    Hikari.clear!(film)
    integrator = Hikari.VolPath(; samples, max_depth = 4)
    integrator(scene, film, camera)
    radiance = map(c -> c.r + c.g + c.b, Array(film.framebuffer))
    close(integrator)
    return radiance
end

white_emission() = Hikari.Emission(color = (1, 1, 1), strength = 2, two_sided = true)

function render_panel_and_mirror(visibility; hw_accel, backend = Mantle.defaultbackend())
    scene = Hikari.Scene(; backend, hw_accel)
    push!(scene, visibility_panel(Vec3f(-1.2, 4, 0), Vec3f(1, -1, 0)), white_emission(); visibility)
    push!(scene, normal_mesh(Tessellation(Sphere(Point3f(1.2, 4, 0), 0.8f0), 128)),
          Hikari.Glossy(roughness = 0, color = (1, 1, 1)))
    return render_radiance(scene, Point3f(0, 0, 0), Point3f(0, 4, 0); backend)
end

function render_panel_over_floor(visibility; hw_accel, backend = Mantle.defaultbackend())
    scene = Hikari.Scene(; backend, hw_accel)
    push!(scene, visibility_panel(Vec3f(0, 0, 1.5), Vec3f(0, 0, -1)), white_emission(); visibility)
    push!(scene, normal_mesh(Rect3f(Vec3f(-3, -3, -0.01), Vec3f(6, 6, 0.01))), Hikari.Diffuse(color = (0.8, 0.8, 0.8)))
    return render_radiance(scene, Point3f(0, -2, 0.6), Point3f(0, 0, 0); backend, samples = 256)
end

function render_occluded_floor(occluder_visibility; hw_accel, occluder = true, backend = Mantle.defaultbackend())
    scene = Hikari.Scene(; backend, hw_accel)
    push!(scene, Hikari.PointLight(Vec3f(0, 0, 3), Hikari.RGBSpectrum(20f0)))
    push!(scene, normal_mesh(Rect3f(Vec3f(-3, -3, -0.01), Vec3f(6, 6, 0.01))), Hikari.Diffuse(color = (0.8, 0.8, 0.8)))
    occluder && push!(scene, visibility_panel(Vec3f(0, 0, 1.5), Vec3f(0, 0, 1); halfsize = 0.5f0),
                      Hikari.Diffuse(color = (0, 0, 0)); visibility = occluder_visibility)
    return render_radiance(scene, Point3f(0, -2, 2.5), Point3f(0, 0, 0); backend)
end

@testset "Ray visibility in renders (hardware=$hw)" for hw in (false, true)
    @testset "camera" begin
        visible = render_panel_and_mirror(Hikari.Visibility(); hw_accel = hw)
        hidden = render_panel_and_mirror(Hikari.Visibility(camera = false); hw_accel = hw)
        left, right = 1:160, 161:320
        @test sum(visible[:, left]) > 0
        @test all(iszero, hidden[:, left])
        @test sum(hidden[:, right]) > 0
        @test sum(hidden[:, right]) ≈ sum(visible[:, right]) rtol = 0.05
    end

    @testset "glossy" begin
        visible = render_panel_and_mirror(Hikari.Visibility(); hw_accel = hw)
        hidden = render_panel_and_mirror(Hikari.Visibility(glossy = false); hw_accel = hw)
        left, right = 1:160, 161:320
        @test sum(hidden[:, left]) ≈ sum(visible[:, left]) rtol = 0.01
        @test sum(visible[:, right]) > 0
        @test all(iszero, hidden[:, right])
    end

    @testset "diffuse" begin
        visible = render_panel_over_floor(Hikari.Visibility(); hw_accel = hw)
        @test sum(visible) > 0
        @test all(iszero, render_panel_over_floor(Hikari.Visibility(diffuse = false); hw_accel = hw))
        @test sum(render_panel_over_floor(Hikari.Visibility(camera = false); hw_accel = hw)) ≈ sum(visible) rtol = 0.02
        @test sum(render_panel_over_floor(Hikari.Visibility(glossy = false); hw_accel = hw)) ≈ sum(visible) rtol = 0.05
    end

    @testset "shadow" begin
        unoccluded = render_occluded_floor(nothing; hw_accel = hw, occluder = false)
        shadowed = render_occluded_floor(Hikari.Visibility(camera = false); hw_accel = hw)
        unshadowed = render_occluded_floor(Hikari.Visibility(camera = false, shadow = false); hw_accel = hw)
        @test sum(shadowed) < 0.9 * sum(unoccluded)
        @test unshadowed ≈ unoccluded
    end
end

floor_mesh() = normal_mesh(Rect3f(Vec3f(-3, -3, -0.01), Vec3f(6, 6, 0.01)))
constant_sky() = Hikari.EnvironmentLight(Hikari.EnvironmentMap(fill(Hikari.RGBSpectrum(1f0), 32, 32)))

function render_point_lit(material, light_visibility; hw_accel, backend = Mantle.defaultbackend())
    scene = Hikari.Scene(; backend, hw_accel)
    push!(scene, Hikari.PointLight(Vec3f(0, 0, 3), Hikari.RGBSpectrum(20f0)); visibility = light_visibility)
    push!(scene, floor_mesh(), material)
    return render_radiance(scene, Point3f(0, -2, 2.5), Point3f(0, 0, 0); backend)
end

function render_backlit_panel(light_visibility; hw_accel, backend = Mantle.defaultbackend())
    scene = Hikari.Scene(; backend, hw_accel)
    push!(scene, Hikari.PointLight(Vec3f(0, 0, -1), Hikari.RGBSpectrum(20f0)); visibility = light_visibility)
    push!(scene, visibility_panel(Vec3f(0, 0, 0), Vec3f(0, 0, 1); halfsize = 2f0),
          Hikari.DiffuseTransmission(reflectance = (0, 0, 0), transmittance = (1, 1, 1)))
    return render_radiance(scene, Point3f(0, -1, 2), Point3f(0, 0, 0); backend)
end

function render_point_lit_fog(light_visibility; hw_accel, backend = Mantle.defaultbackend())
    scene = Hikari.Scene(; backend, hw_accel)
    fog = Hikari.HomogeneousMedium(σ_a = Hikari.RGBSpectrum(0.01f0), σ_s = Hikari.RGBSpectrum(0.5f0),
                                   Le = Hikari.RGBSpectrum(0f0), g = 0f0)
    push!(scene, normal_mesh(Rect3f(Vec3f(-1), Vec3f(2))), Hikari.MediumInterface(Hikari.NullMaterial(); inside = fog))
    push!(scene, Hikari.PointLight(Vec3f(0, 0, 0), Hikari.RGBSpectrum(20f0)); visibility = light_visibility)
    return render_radiance(scene, Point3f(0, -4, 0), Point3f(0, 0, 0); backend)
end

function render_sky_lit_floor(sky_visibility; hw_accel, with_floor = true, backend = Mantle.defaultbackend())
    scene = Hikari.Scene(; backend, hw_accel)
    push!(scene, constant_sky(); visibility = sky_visibility)
    with_floor && push!(scene, floor_mesh(), Hikari.Diffuse(color = (0.8, 0.8, 0.8)))
    return render_radiance(scene, Point3f(0, -0.5, 1), Point3f(0, 0, 0); backend)
end

@testset "Light ray visibility (hardware=$hw)" for hw in (false, true)
    @testset "point light, diffuse floor" begin
        diffuse_floor = Hikari.Diffuse(color = (0.8, 0.8, 0.8))
        visible = render_point_lit(diffuse_floor, Hikari.Visibility(); hw_accel = hw)
        @test sum(visible) > 0
        @test all(iszero, render_point_lit(diffuse_floor, Hikari.Visibility(diffuse = false); hw_accel = hw))
        @test render_point_lit(diffuse_floor, Hikari.Visibility(camera = false, glossy = false, transmission = false,
                                                                volume_scatter = false); hw_accel = hw) ≈ visible
    end

    @testset "point light, glossy floor" begin
        glossy_floor = Hikari.Glossy(roughness = 0.3, color = (1, 1, 1))
        visible = render_point_lit(glossy_floor, Hikari.Visibility(); hw_accel = hw)
        @test sum(visible) > 0
        @test all(iszero, render_point_lit(glossy_floor, Hikari.Visibility(glossy = false); hw_accel = hw))
        @test render_point_lit(glossy_floor, Hikari.Visibility(diffuse = false); hw_accel = hw) ≈ visible
    end

    @testset "point light through a translucent panel" begin
        visible = render_backlit_panel(Hikari.Visibility(); hw_accel = hw)
        @test sum(visible) > 0
        @test all(iszero, render_backlit_panel(Hikari.Visibility(transmission = false); hw_accel = hw))
        @test render_backlit_panel(Hikari.Visibility(diffuse = false, glossy = false); hw_accel = hw) ≈ visible
    end

    @testset "point light in fog" begin
        visible = render_point_lit_fog(Hikari.Visibility(); hw_accel = hw)
        @test sum(visible) > 0
        @test all(iszero, render_point_lit_fog(Hikari.Visibility(volume_scatter = false); hw_accel = hw))
    end

    @testset "light casting no shadow" begin
        unoccluded = render_occluded_floor(nothing; hw_accel = hw, occluder = false)
        scene = Hikari.Scene(; backend = Mantle.defaultbackend(), hw_accel = hw)
        push!(scene, Hikari.PointLight(Vec3f(0, 0, 3), Hikari.RGBSpectrum(20f0)); visibility = Hikari.Visibility(shadow = false))
        push!(scene, floor_mesh(), Hikari.Diffuse(color = (0.8, 0.8, 0.8)))
        push!(scene, visibility_panel(Vec3f(0, 0, 1.5), Vec3f(0, 0, 1); halfsize = 0.5f0), Hikari.Diffuse(color = (0, 0, 0));
              visibility = Hikari.Visibility(camera = false))
        @test render_radiance(scene, Point3f(0, -2, 2.5), Point3f(0, 0, 0); backend = Mantle.defaultbackend()) ≈ unoccluded
    end

    @testset "sky" begin
        sky = render_sky_lit_floor(Hikari.Visibility(); hw_accel = hw, with_floor = false)
        @test sum(sky) > 0
        @test all(iszero, render_sky_lit_floor(Hikari.Visibility(camera = false); hw_accel = hw, with_floor = false))

        lit = render_sky_lit_floor(Hikari.Visibility(); hw_accel = hw)
        @test sum(lit) > 0
        @test render_sky_lit_floor(Hikari.Visibility(camera = false); hw_accel = hw) ≈ lit
        @test all(iszero, render_sky_lit_floor(Hikari.Visibility(diffuse = false); hw_accel = hw))
    end
end
