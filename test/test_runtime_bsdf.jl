module RuntimeBSDFTests
using Test, Hikari, Mantle, Raycore, KernelAbstractions, LinearAlgebra
using Hikari.TextureNodes
using GeometryBasics: Point2f, Point3f, Vec3f

@testset "bounded BSDF kernel classes" begin
    D, S, G = Hikari.DiffuseBSDF, Hikari.SheenBSDF, Hikari.GlossyBSDF
    for api in (Mantle.VulkanAPI(), Mantle.MetalAPI())
        select(nodes...) = Hikari.bsdf_kernel_nodes(Tuple{nodes...}, api)
        @test select(D) === Tuple{D}
        @test select(S) === Tuple{S}
        @test select(G) === Tuple{G}
        @test select(D,S,G) === select(G,D,S) === select(S,G,D)
        @test select(D,S) === select(S,G)
        @test select(D,D,D,D) === select(G,G,G,G)
        @test select(ntuple(_->D,8)...) === select(ntuple(_->S,8)...)
        @test select(ntuple(_->D,9)...) === Tuple{ntuple(_->D,9)...}
        @test select(D,Hikari.PrincipledBSDF{false, false, false}) === Tuple{D,Hikari.PrincipledBSDF{false, false, false}}
        @test select(D,Hikari.EmissionBSDF) === Tuple{D,Hikari.EmissionBSDF}
        @test select(D,Hikari.TransparentBSDF) === Tuple{D,Hikari.TransparentBSDF}
        @test !Hikari.emits(select(D,S))
    end
    @test sizeof(Hikari.RuntimeWeighted{false}) == 72
    @test sizeof(Hikari.RuntimeWeighted{true}) == 72
end

function description(ops; computed = false, bump_normal = false, mirror = false)
    color = computed ? noise(xyz(); scale = 3).color : (0.4, 0.2, 0.1)
    n = bump_normal ? bump(noise(xyz(); scale = 6).fac; strength = 0.1) : nothing
    nodes = (Hikari.Diffuse(; color, roughness = 0.4, normal = n),
             Hikari.Sheen(color = (0.25, 0.3, 0.4), roughness = 0.35, normal = n),
             Hikari.Glossy(color = (0.3, 0.3, 0.3), roughness = mirror ? 0 : 0.25,
                           anisotropy = 0.35, rotation = 0.2, normal = n))
    return foldl((a, b) -> Hikari.Mix(a, b; factor = 0.35), (nodes[i] for i in ops))
end

@testset "runtime BSDF differential setup and sampling" begin
    lambda = Hikari.Wavelengths((450f0, 510f0, 570f0, 630f0), (0.25f0, 0.25f0, 0.25f0, 0.25f0))
    table = Hikari.get_srgb_table()
    for ops in ((1,), (2,), (3,), (1, 2, 3), (3, 1, 2), (1, 2, 3, 1, 2, 3, 1, 2)),
        computed in (false, true), mirror in (false, true), looped in (false, true)
        desc = description(ops; computed, mirror)
        for alpha in (1f0, 0.3f0)
            alpha == 1f0 || (desc = Hikari.Mix(desc, Hikari.Diffuse(color = Hikari.RGBA{Float32}(0.4,0.2,0.1,alpha)); factor = 0.3))
            store = Raycore.MultiTypeSet(KernelAbstractions.CPU())
            mat = Hikari.to_device_material(store, Hikari.surface_material(desc))
            textures = Raycore.StaticMultiTypeSet((), Tuple(T[a for a in store.texture_gpu_arrays if typeof(a) === T]
                for T in store.texture_order))
            wo = normalize(Vec3f(0.2, 0.1, 1))
            tfc = Hikari.TextureFilterContext(Point2f(0.3, 0.6), Point3f(0.3, -0.2, 0.7), Vec3f(0, 0, 1))
            sc = Hikari.ShaderContext(tfc, wo, Vec3f(1f-3, 0, 0), Vec3f(0, 1f-3, 0), Vec3f(1, 0, 0),
                                     Vec3f(0, 0, 1), 1f0, Vec3f(0), Point2f(0))
            prog = Hikari.shader_programs(textures, mat.program)[mat.program.idx]
            s, i = Hikari.run_values(prog, textures, sc)
            inputs = Hikari.ShaderInputs(fill(NaN32, max(Int(mat.inputs), 1) * 3), zeros(Int32, 3), Int32(3))
            Hikari.gather_inputs!(inputs, Int32(2), s, prog, i)
            M = typeof(mat).parameters[3]
            produced = Hikari.node_lobes(M, s, prog, i, sc, Hikari.LobeSetup(Hikari.lookup_tables(textures), false))
            reference = Hikari.lobe_bsdf(M, produced, table, lambda)
            n = length(M.parameters)
            N = n == 1 ? 1 : n <= 4 ? 4 : n <= 8 ? 8 : 16
            candidate = Hikari.runtime_bsdf(Hikari.RuntimeBSDFNodes{N, looped}, Hikari.GatheredStack(inputs.values, Int32(3), Int32(2)),
                Hikari.closure_programs(textures)[mat.closures], sc,
                Hikari.LobeSetup(Hikari.lookup_tables(textures), false), table, lambda)
            @test candidate.total ≈ reference.total
            for u in (Point2f(0.1, 0.2), Point2f(0.55, 0.75)), r in (0f0, 0.1f0, 0.5f0, 0.9999f0)
                args = (table, textures, wo, tfc.n, sc.dpdus, tfc, lambda, u, r)
                a = Hikari.sample_bsdf_spectral(reference, args...)
                b = Hikari.sample_bsdf_spectral(candidate, args...)
                @test b.flags == a.flags
                @test b.wi ≈ a.wi
                @test all(isapprox.(b.f.data, a.f.data))
                @test b.pdf ≈ a.pdf
            end
            wi = normalize(Vec3f(-0.3, 0.2, 1))
            args = (table, textures, wo, wi, tfc.n, sc.dpdus, tfc, lambda)
            a, ap = Hikari.evaluate_bsdf_spectral(reference, args...)
            b, bp = Hikari.evaluate_bsdf_spectral(candidate, args...)
            @test all(isapprox.(b.data, a.data))
            @test bp ≈ ap
        end
    end
end

end
