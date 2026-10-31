module SurfaceProgramCPUCheck
using Hikari, Mantle, Raycore, KernelAbstractions, Test, GeometryBasics, LinearAlgebra
using Hikari.TextureNodes
using GeometryBasics: Point2f, Point3f, Vec3f
function run(; selected = nothing)
    lambda = Hikari.Wavelengths((450.0f0, 510.0f0, 570.0f0, 630.0f0), (0.25f0, 0.25f0, 0.25f0, 0.25f0))
    table = Hikari.get_srgb_table()
    wo = normalize(Vec3f(0.2, 0.1, 1))
    tfc = Hikari.TextureFilterContext(Point2f(0.3, 0.6), Point3f(0.3, -0.2, 0.7), Vec3f(0, 0, 1))
    sc = Hikari.ShaderContext(tfc, wo, Vec3f(1.0f-3, 0, 0), Vec3f(0, 1.0f-3, 0), Vec3f(1, 0, 0), Vec3f(0, 0, 1), 1.0f0, Vec3f(0), Point2f(0))
    materials = Any[
        Hikari.Diffuse(roughness = 0.4), Hikari.Sheen(), Hikari.Translucent(),
        Hikari.Toon(), Hikari.Toon(component = :glossy), Hikari.Principled(),
        Hikari.Principled(metallic = 0.5, coat_weight = 0.6, sheen_weight = 0.4),
        Hikari.Principled(subsurface_weight = 0.6),
        Hikari.Principled(subsurface_weight = 0.6, thin_wall = true),
        Hikari.Principled(transmission_weight = 0.6, thin_wall = true),
        Hikari.Principled(transmission_weight = 1, transmission_dispersion_scale = 0.5),
        Hikari.SubsurfaceScattering(), Hikari.SubsurfaceScattering(scale = 0),
        Hikari.Mix(Hikari.Glass(roughness = 0.3), Hikari.Metallic(); factor = 0.4),
        Hikari.Mix(Hikari.Principled(subsurface_weight = 0.5), Hikari.Principled(thin_wall = true); factor = 0.4),
        Hikari.Add(Hikari.Principled(), Hikari.Emission(strength = 0.3)),
        Hikari.Add(Hikari.Transparent(color = (0.1, 0.2, 0.3)), Hikari.Glass()),
    ]
    for d in (:multi_ggx, :ggx, :beckmann),r in (0, 0.4),film in (0, 400)
        push!(materials, Hikari.Glass(roughness = r, distribution = d, thin_film_thickness = film))
        push!(materials, Hikari.Glossy(roughness = r, distribution = d))
        push!(materials, Hikari.Metallic(roughness = r, distribution = d, thin_film_thickness = film))
        push!(materials, Hikari.Metallic(roughness = r, distribution = d, fresnel_type = :physical_conductor, thin_film_thickness = film))
    end
    for d in (:ggx, :beckmann)
        push!(materials, Hikari.Refraction(distribution = d))
    end
    selected === nothing || (materials = materials[selected])
    for (index, m) in enumerate(materials)
        println("Surface program case ", index, "/", length(materials), ": ", nameof(typeof(m)));flush(stdout)
        store = Raycore.MultiTypeSet(KernelAbstractions.CPU())
        mat = Hikari.to_device_material(store, Hikari.surface_material(m))
        textures = Raycore.StaticMultiTypeSet((), Tuple(T[a for a in store.texture_gpu_arrays if typeof(a) === T] for T in store.texture_order))
        prog = Hikari.shader_programs(textures, mat.program)[mat.program.idx]
        M = typeof(mat).parameters[3]
        produced = Hikari.shader_lobes(M, prog, textures, sc, false)
        reference = Hikari.lobe_bsdf(M, produced, table, lambda)
        for looped in (false, true)
            s, i = Hikari.run_values(prog, textures, sc)
            inputs = Hikari.ShaderInputs(fill(NaN32, max(Int(mat.inputs), 1) * 3), zeros(Int32, 3), Int32(3))
            Hikari.gather_inputs!(inputs, Int32(2), s, prog, i)
            P = Hikari.surface_program_class(M, looped ? Mantle.MetalAPI() : Mantle.VulkanAPI())
            @test P !== nothing
            candidate = Base.invokelatest(
                Hikari.surface_program_bsdf, P, Hikari.GatheredStack(inputs.values, Int32(3), Int32(2)),
                Hikari.closure_programs(textures)[mat.closures], sc, Hikari.LobeSetup(Hikari.lookup_tables(textures), false), table, lambda
            )
            packed = candidate.lobes
            @test candidate.total ≈ reference.total
            @test all(isapprox.(candidate.emission.data, reference.emission.data))
            @test isbitstype(typeof(candidate))
            @test Hikari.disperses(packed) == Hikari.disperses(reference.lobes)
            for restriction in (Hikari.AnyScattering(), Hikari.OnlyReflection(), Hikari.OnlyTransmission())
                @test Hikari.share(packed, restriction) ≈ Hikari.share(reference.lobes, restriction)
                for wi in (normalize(Vec3f(-0.3, 0.2, 1)), normalize(Vec3f(-0.3, 0.2, -1)))
                    a, ap = Hikari.mixture(reference.lobes, wo, wi, tfc.n, lambda, restriction)
                    b, bp = Hikari.mixture(packed, wo, wi, tfc.n, lambda, restriction)
                    @test all(isapprox.(a.data, b.data))
                    @test ap ≈ bp
                end
                for r in (0.0f0, 0.1f0, 0.5f0, 0.9999f0),u in (Point2f(0.1, 0.2), Point2f(0.55, 0.75))
                    a = Hikari.pick_sample(reference.lobes, r * Hikari.share(reference.lobes, restriction), 0.0f0, wo, u, lambda, restriction)
                    b = Hikari.pick_sample(packed, r * Hikari.share(packed, restriction), 0.0f0, wo, u, lambda, restriction)
                    @test a[1] ≈ b[1]
                    @test all(isapprox.(a[2].data, b[2].data))
                    @test a[3] ≈ b[3]
                    @test a[4:5] == b[4:5]
                    @test a[6] ≈ b[6]
                    @test a[7] ≈ b[7]
                    @test a[8] == b[8]
                end
            end
        end
    end
    return println("Validated ", length(materials), " material configurations; weighted record bytes=", sizeof(Hikari.SurfaceStoredWeight{false, true}))
end
end
