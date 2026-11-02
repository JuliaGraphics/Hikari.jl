using Test, Hikari, GeometryBasics, LinearAlgebra, Raycore, KernelAbstractions

@testset "direct shader evaluation uses shared storage" begin
    table = Hikari.get_srgb_table()
    lambda = Hikari.Wavelengths((450.0f0, 510.0f0, 570.0f0, 630.0f0), (0.25f0, 0.25f0, 0.25f0, 0.25f0))
    n = GeometryBasics.Vec3f(0, 0, 1)
    wo = LinearAlgebra.normalize(GeometryBasics.Vec3f(0.2, 0.1, 1))
    sc = Hikari.ShaderContext(Hikari.TextureFilterContext(GeometryBasics.Point2f(0.3, 0.6), GeometryBasics.Point3f(0.3, -0.2, 0.7), n), wo, GeometryBasics.Vec3f(0), GeometryBasics.Vec3f(0), GeometryBasics.Vec3f(1, 0, 0), n, 1.0f0, GeometryBasics.Vec3f(0), GeometryBasics.Point2f(0))
    nodes = Hikari.TextureNodes
    for desc in (
            Hikari.Diffuse(), Hikari.Glass(roughness = 0.2),
            Hikari.Principled(base_color = nodes.noise(nodes.xyz()).color, roughness = 0.3),
            Hikari.Mix(Hikari.Glass(roughness = 0.3, distribution = :beckmann), Hikari.Metallic(distribution = :ggx); factor = 0.4),
            Hikari.Add(Hikari.Glossy(distribution = :beckmann), Hikari.Glass(roughness = 0.3, distribution = :ggx)),
        )
        store = Raycore.MultiTypeSet(KernelAbstractions.CPU())
        mat = Hikari.to_device_material(store, Hikari.surface_material(desc))
        textures = Raycore.StaticMultiTypeSet((), Tuple(T[a for a in store.texture_gpu_arrays if typeof(a) === T] for T in store.texture_order))
        program = Hikari.shader_programs(textures, mat.program)[mat.program.idx]
        m = typeof(mat).parameters[3]
        actual = @inferred Hikari.shader_bsdf(m, program, textures, sc, false, table, lambda)
        expected = Hikari.lobe_bsdf(m, Hikari.shader_lobes(m, program, textures, sc, false), table, lambda)
        @test actual.lobes isa Hikari.SurfaceLobeSpan
        @test actual.total ≈ expected.total
        for side in (1, -1)
            wi = LinearAlgebra.normalize(GeometryBasics.Vec3f(-0.3, 0.2, side))
            a, apdf = Hikari.mixture(actual.lobes, wo, wi, n, lambda)
            b, bpdf = Hikari.mixture(expected.lobes, wo, wi, n, lambda)
            @test all(isapprox.(a.data, b.data))
            @test apdf ≈ bpdf
        end
    end
end

@testset "one-record storage observes its live count" begin
    n = normalize(Vec3f(0.2, -0.3, 1))
    wo = normalize(Vec3f(0.3, 0.2, 1))
    lambda = Hikari.Wavelengths((450f0, 510f0, 570f0, 630f0), (0.25f0, 0.25f0, 0.25f0, 0.25f0))
    native = Hikari.Weighted(Hikari.LambertLobe(Hikari.SpectralRadiance), Hikari.SpectralRadiance(0.4f0), 0.4f0, n, Vec3f(0))
    stored = Hikari.surface_store(native, Val(true), Val(Hikari.SURFACE_SIMPLE_MASK))
    for count in (Int32(0), Int32(1)), wi in (n, -n), restriction in (Hikari.AnyScattering(), Hikari.OnlyReflection(), Hikari.OnlyTransmission())
        span = Hikari.SurfaceLobeSpan((stored,), count)
        expected = count == 0 ? (Hikari.SpectralRadiance(0f0), 0f0) : Hikari.mixture((native,), wo, wi, n, lambda, restriction)
        actual = Hikari.mixture(span, wo, wi, n, lambda, restriction)
        @test actual[1].data == expected[1].data
        @test actual[2] == expected[2]
    end
    inactive = Hikari.Weighted(stored.lobe, stored.weight, 0f0, stored.n, stored.t)
    f, pdf = Hikari.mixture(Hikari.SurfaceLobeSpan((inactive,), Int32(1)), wo, n, n, lambda)
    @test f.data == Hikari.SpectralRadiance(0f0).data
    @test pdf == 0f0
end
