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
            Hikari.Principled(metallic = 0.35, coat_weight = 0.4, sheen_weight = 0.25),
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
            if length(actual.lobes.data) >= 5 && actual.lobes.count <= 5
                stored = ntuple(j -> actual.lobes.data[j], 5)
                args = (stored, actual.lobes.count, wo, wi, n, lambda, Hikari.AnyScattering())
                reference = invoke(Hikari.surface_mixture, Tuple{Any, Any, Any, Any, Any, Any, Any}, args...)
                bounded = Hikari.mixture(Hikari.SurfaceLobeSpan(stored, actual.lobes.count), wo, wi, n, lambda)
                @test isequal(bounded, reference)
            end
        end
    end
end

@testset "local-array spans retain tuple sampling and live-prefix semantics" begin
    n = Vec3f(0, 0, 1)
    wo = normalize(Vec3f(0.2, 0.1, 1))
    lambda = Hikari.Wavelengths((450f0, 510f0, 570f0, 630f0), (0.25f0, 0.25f0, 0.25f0, 0.25f0))
    for capacity in (1, 4, 8, 16)
        records = ntuple(capacity) do j
            native = Hikari.Weighted(Hikari.LambertLobe(Hikari.SpectralRadiance),
                Hikari.SpectralRadiance(Float32(j) * 0.01f0), Float32(j) / 7f0, n, Vec3f(0))
            Hikari.surface_store(native, Val(true), Val(Hikari.SURFACE_PRINCIPLED_MASK))
        end
        data = Hikari.MVector{capacity, typeof(first(records))}(records)
        @test Hikari.surface_sample_total(data) == Hikari.surface_sample_total(records)
        for count in Int32(0):Int32(capacity)
            span = Hikari.SurfaceLobeSpan(data, count)
            tuple_span = Hikari.SurfaceLobeSpan(records, count)
            for wi in (n, -n), restriction in (Hikari.AnyScattering(), Hikari.OnlyTransmission())
                @test isequal(Hikari.mixture(span, wo, wi, n, lambda, restriction),
                    Hikari.mixture(tuple_span, wo, wi, n, lambda, restriction))
            end
            for target in (0f0, 0.4f0, 2f0)
                args = (target, 0f0, wo, GeometryBasics.Point2f(0.3, 0.7), lambda)
                @test isequal(Hikari.pick_sample(span, args...), Hikari.pick_sample(tuple_span, args...))
            end
            empty_bsdf = Hikari.no_samples(typeof(span), lambda)
            @test empty_bsdf.total == 0f0
            @test empty_bsdf.lobes.count == 0
            @test Hikari.subsurface_slot(typeof(span)) == 0
            @test Hikari.subsurface_weight(span) == 0f0
            @test Hikari.subsurface_choice(span, 0f0) == 0
        end
    end
end

@testset "local-array subsurface exits preserve the BSDF storage type" begin
    n = Vec3f(0, 0, 1)
    lambda = Hikari.Wavelengths((450f0, 510f0, 570f0, 630f0), (0.25f0, 0.25f0, 0.25f0, 0.25f0))
    lobe = Hikari.SubsurfaceLobe(Hikari.SpectralRadiance(0.5f0), Hikari.SpectralRadiance(0.1f0),
        0.3f0, 1.4f0, 0f0, false)
    native = Hikari.Weighted(lobe, Hikari.SpectralRadiance(0.5f0), 0.5f0, n, Vec3f(0))
    stored = Hikari.surface_store(native, Val(true), Val(Hikari.SURFACE_PRINCIPLED_MASK | (UInt32(1) << 14)))
    data = Hikari.MVector{8, typeof(stored)}(ntuple(_ -> stored, 8))
    span = Hikari.SurfaceLobeSpan(data, Int32(2))
    @test Hikari.subsurface_slot(typeof(span)) == 1
    @test Hikari.subsurface_weight(span) == 1f0
    @test Hikari.subsurface_choice(span, 0.7f0) == 2
    @test Hikari.subsurface_choice(span, 1.1f0) == 0
    params, weight = Hikari.picked_walk_params(span, Int32(2), Val(1))
    @test weight == 0.5f0
    exited = Hikari.exit_lobe(eltype(typeof(data)), params, n)
    exit_span = Hikari.with_lobe(typeof(span), Val(1), exited)
    @test typeof(exit_span) == typeof(span)
    @test exit_span.count == 1
    @test exit_span[1].sample_weight == 1f0
    @test all(w -> w.sample_weight == 0f0, exit_span.data[2:end])
    @test Hikari.no_samples(typeof(span), lambda).lobes.count == 0
end

@testset "bounded mixture retains live prefix and accumulation order" begin
    n = Vec3f(0, 0, 1)
    wo = normalize(Vec3f(0.2, 0.1, 1))
    lambda = Hikari.Wavelengths((450f0, 510f0, 570f0, 630f0), (0.25f0, 0.25f0, 0.25f0, 0.25f0))
    # Padding deliberately has nonzero weights: `count` must exclude it even
    # when a live slot has zero weight. Small unequal terms expose reordering.
    for capacity in (2, 5), zero_slot in 0:capacity
        stored = ntuple(capacity) do j
            weight = zero_slot == j ? 0f0 : Float32(j) / 5f0
            native = Hikari.Weighted(Hikari.LambertLobe(Hikari.SpectralRadiance),
                Hikari.SpectralRadiance(Float32(j) * 1f-3), weight, n, Vec3f(0))
            Hikari.surface_store(native, Val(true), Val(Hikari.SURFACE_PRINCIPLED_MASK))
        end
        for count in Int32(0):Int32(capacity), restriction in (Hikari.AnyScattering(), Hikari.OnlyReflection(), Hikari.OnlyTransmission()), wi in (n, -n)
            expected = invoke(Hikari.surface_mixture,
                Tuple{Any, Any, Any, Any, Any, Any, Any}, stored, count, wo, wi, n, lambda, restriction)
            actual = Hikari.mixture(Hikari.SurfaceLobeSpan(stored, count), wo, wi, n, lambda, restriction)
            @test isequal(actual, expected)
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
