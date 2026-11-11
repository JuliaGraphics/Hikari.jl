using Test, Hikari, Mantle, Raycore, KernelAbstractions, GeometryBasics, LinearAlgebra

@testset "Cycles terms retain shared storage in native compositions" begin
    lambda = Hikari.sample_wavelengths_stratified((0.11f0, 0.27f0, 0.53f0, 0.79f0))
    n = Vec3f(0, 0, 1)
    wo = normalize(Vec3f(0.3, 0.2, 1))
    sc = Hikari.ShaderContext(Hikari.TextureFilterContext(Point2f(0.5), Point3f(0), n), wo, Vec3f(0), Vec3f(0), Vec3f(1, 0, 0), n, 1.0f0, Vec3f(0), Point2f(0))
    for (native, layered) in ((Hikari.Conductor(roughness = 0.2), false), (Hikari.CoatedDiffuse(roughness = 0.2), true))
        description = Hikari.Add(Hikari.Diffuse(color = (0.2, 0.4, 0.7)), native)
        store = Raycore.MultiTypeSet(KernelAbstractions.CPU())
        mat = Hikari.to_device_material(store, Hikari.surface_material(description))
        textures = Raycore.StaticMultiTypeSet((), Tuple(T[a for a in store.texture_gpu_arrays if typeof(a) === T] for T in store.texture_order))
        program = Hikari.shader_programs(textures, mat.program)[mat.program.idx]
        bsdf = Hikari.shader_bsdf(typeof(mat).parameters[3], program, textures, sc, false, Hikari.get_srgb_table(), lambda)
        if layered
            @test bsdf isa Hikari.ComponentBSDF
            @test any(c -> c.bsdf isa Hikari.LobeBSDF && c.bsdf.lobes isa Hikari.SurfaceLobeSpan, bsdf.components)
        else
            @test bsdf isa Hikari.LobeBSDF
            @test any(w -> w.lobe isa Hikari.SurfaceStoredLobe, bsdf.lobes)
        end
    end
end

@testset "packed subsurface retains its interval beside native lobes" begin
    lambda = Hikari.sample_wavelengths_stratified((0.11f0, 0.27f0, 0.53f0, 0.79f0))
    n = Vec3f(0, 0, 1)
    wo = normalize(Vec3f(0.3, 0.2, 1))
    sc = Hikari.ShaderContext(Hikari.TextureFilterContext(Point2f(0.5), Point3f(0), n), wo, Vec3f(0), Vec3f(0), Vec3f(1, 0, 0), n, 1.0f0, Vec3f(0), Point2f(0))
    description = Hikari.Add(Hikari.Principled(subsurface_weight = 0.6), Hikari.Conductor(roughness = 0.2))
    store = Raycore.MultiTypeSet(KernelAbstractions.CPU())
    mat = Hikari.to_device_material(store, Hikari.surface_material(description))
    textures = Raycore.StaticMultiTypeSet((), Tuple(T[a for a in store.texture_gpu_arrays if typeof(a) === T] for T in store.texture_order))
    program = Hikari.shader_programs(textures, mat.program)[mat.program.idx]
    m = typeof(mat).parameters[3]
    table = Hikari.get_srgb_table()
    actual = Hikari.shader_bsdf(m, program, textures, sc, false, table, lambda)
    rgb, transparency, _ = Hikari.shader_lobes(m, program, textures, sc, false)
    reference = Hikari.lift_shader_terms(rgb, 1.0f0 / (1.0f0 - Hikari._avg(transparency)), table, lambda)
    @test Hikari.subsurface_weight(reference) > 0
    @test Hikari.subsurface_slot(typeof(actual.lobes)) > 0
    @test Hikari.subsurface_weight(actual.lobes) ≈ Hikari.subsurface_weight(reference)
    for fraction in (0.0f0, 0.1f0, 0.9f0)
        target = fraction * actual.total
        @test Hikari.subsurface_choice(actual.lobes, target) == Hikari.subsurface_choice(reference, target)
    end
    chosen = Hikari.subsurface_choice(actual.lobes, 0.0f0)
    a, aw = Hikari.picked_walk_params(actual.lobes, chosen, Val(Hikari.subsurface_slot(typeof(actual.lobes))))
    b, bw = Hikari.picked_walk_params(reference, chosen, Val(Hikari.subsurface_slot(typeof(reference))))
    @test aw ≈ bw
    @test a.n ≈ b.n
    @test a.albedo.data == b.albedo.data
    @test a.radius.data == b.radius.data
end

@testset "native linear composition retains graph emission" begin
    table = Hikari.get_srgb_table()
    lambda = Hikari.sample_wavelengths_stratified((0.11f0, 0.27f0, 0.53f0, 0.79f0))
    n = Vec3f(0, 0, 1)
    sc = Hikari.ShaderContext(Hikari.TextureFilterContext(Point2f(0.5), Point3f(0), n), n, Vec3f(0), Vec3f(0), Vec3f(1, 0, 0), n, 1.0f0, Vec3f(0), Point2f(0))
    emission = Hikari.Emission(color = (0.2, 0.4, 0.6), strength = 2)
    results = map((emission, Hikari.Add(Hikari.Conductor(), emission))) do description
        store = Raycore.MultiTypeSet(KernelAbstractions.CPU())
        mat = Hikari.to_device_material(store, Hikari.ShaderMaterial(Hikari.shader_node(description)))
        textures = Raycore.StaticMultiTypeSet((), Tuple(T[a for a in store.texture_gpu_arrays if typeof(a) === T] for T in store.texture_order))
        program = Hikari.shader_programs(textures, mat.program)[mat.program.idx]
        Hikari.shader_bsdf(typeof(mat).parameters[3], program, textures, sc, false, table, lambda).emission
    end
    @test maximum(last(results).data) > 0
    @test first(results).data == last(results).data
end
