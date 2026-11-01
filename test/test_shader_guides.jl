module ShaderGuideTests
using Test, Hikari, Mantle, Raycore, GeometryBasics, KernelAbstractions
using GeometryBasics: Point2f, Point3f, Vec3f

function reference(description)
    store = Raycore.MultiTypeSet(KernelAbstractions.CPU())
    material = Hikari.to_device_material(store, Hikari.surface_material(description))
    textures = Raycore.StaticMultiTypeSet((), Tuple(T[a for a in store.texture_gpu_arrays if typeof(a) === T] for T in store.texture_order))
    program = Hikari.shader_programs(textures, material.program)[material.program.idx]
    n = Vec3f(0, 0, 1)
    tfc = Hikari.TextureFilterContext(Point2f(0), Point3f(0), n)
    sc = Hikari.ShaderContext(tfc, n, Vec3f(0), Vec3f(0), Vec3f(1, 0, 0), n, 1f0, Vec3f(0), Point2f(0))
    color, _ = Hikari.shader_guides(Hikari.bsdf_nodes(typeof(material)), program, textures, sc, Val(3))
    return clamp.(color, 0f0, 1f0)
end

function check(description, expected; hardware)
    backend = Mantle.defaultbackend()
    scene = Hikari.Scene(; backend, hw_accel = hardware)
    mesh = normal_mesh(Tessellation(Sphere(Point3f(0), 1f0), 16))
    push!(scene, mesh, description)
    Hikari.sync!(scene)
    film = Hikari.Film(backend, Hikari.Film(Point2f(8, 8)))
    camera = Hikari.PerspectiveCamera(Point3f(0, 0, 3), Point3f(0), film; fov = 30f0)
    integrator = Hikari.VolPath(samples = 1, hw_accel = hardware)
    try
        adapted = Hikari.adapt_scene_for_render(backend, scene, integrator)
        Hikari.fill_aux_buffers!(film, adapted, camera; grid = 1)
        colors, normals, depth = Array(film.albedo), Array(film.normal), Array(film.depth)
        @test all(p -> isfinite(p.r + p.g + p.b), colors)
        @test all(isfinite, depth[4:5, 4:5])
        @test all(n -> all(isfinite, n) && sum(abs2, n) > 0.9f0, normals[4:5, 4:5])
        @test all(p -> Vec3f(p.r, p.g, p.b) ≈ expected, colors[4:5, 4:5])
    finally
        close(integrator)
        Hikari.free!(film)
    end
end

@testset "native graph guide readers stay off the GPU heap" begin
    image = Hikari.TextureNodes.image_texture(fill(Hikari.RGBSpectrum(0.3f0, 0.5f0, 0.7f0), 2, 2))
    materials = (
        Hikari.Conductor(),
        Hikari.Conductor(reflectance = (0.2, 0.4, 0.7)),
        Hikari.Mix(Hikari.Diffuse(color = image), Hikari.CoatedDiffuse(reflectance = image); factor = 0.35),
    )
    for material in materials
        expected = reference(material)
        for hardware in (Mantle.supports_hwtlas(Mantle.defaultbackend()) ? (false, true) : (false,))
            @testset "$(nameof(typeof(material))) / hardware=$hardware" begin
                check(material, expected; hardware)
            end
        end
    end
end
end
