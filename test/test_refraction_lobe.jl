using Test, Hikari, LinearAlgebra
using GeometryBasics: Vec3f, Point2f

@testset "Cycles Refraction lobes" begin
    lambda = Hikari.sample_wavelengths_visible(0.4f0)
    color = Hikari.SpectralRadiance(1f0)
    n = Vec3f(0, 0, 1)
    @testset "Beckmann math" begin
        # Independently evaluated double-precision reference values.
        for (x, expected) in (
            (-4f0, -0.9999999845827421),
            (-1f0, -0.8427007929497149),
            (0f0, 0.0),
            (0.3f0, 0.32862677175271543),
            (1f0, 0.8427007929497149),
            (4f0, 0.9999999845827421),
        )
            @test Hikari.beckmann_erf(x) ≈ expected atol = 2e-6
        end
        for (x, expected) in (
            (-0.999f0, -2.326756326796155),
            (-0.5f0, -0.4769362762044698),
            (0f0, 0.0),
            (0.7f0, 0.732869059882759),
            (0.999f0, 2.326756326796155),
        )
            @test Hikari.beckmann_erfinv(x) ≈ expected atol = 1e-6
        end
        @test Hikari.beckmann_d(n, 0.3f0, 0.5f0) ≈ 1 / (π * 0.3 * 0.5)
        @test Hikari.beckmann_d(Vec3f(1, 0, 0), 0.3f0, 0.5f0) == 0f0
        @test Hikari.beckmann_lambda(n, 0.3f0, 0.5f0) == 0f0
        # At normal incidence the squared slope is exponentially distributed,
        # with mean alpha^2. This checks the sampler against the analytic NDF.
        count = 4096
        moment = sum(1:count) do i
            h = Hikari.beckmann_sample_wm(n, Point2f((i - 0.5) / count, mod(i * 0.61803398875, 1)), 0.4f0, 0.4f0)
            (h[1]^2 + h[2]^2) / h[3]^2
        end / count
        @test moment ≈ 0.16 atol = 3e-5
        @test isfinite(Hikari.beckmann_erfinv(1f0))
        for wo in (n, normalize(Vec3f(0.6, 0, 0.8))), u in (Point2f(0), Point2f(0, 0.5), Point2f(1))
            h = Hikari.beckmann_sample_wm(wo, u, 0.4f0, 0.4f0)
            @test all(isfinite, h)
            @test norm(h) ≈ 1 atol = 2e-6
        end
    end
    for convention in (Hikari.CyclesBeckmann(), Hikari.CyclesGGX())
        lobe = Hikari.MicrofacetLobe{:refraction}(Hikari.NoFresnel(color); alpha_x = 0f0, ior = 1.5f0, convention)
        wo = Vec3f(0.6, 0, 0.8)
        wi, f, pdf, flags, eta = Hikari.lobe_sample(lobe, wo, Point2f(0.3, 0.7), 0.1f0, lambda)
        @test wi ≈ Vec3f(-0.4, 0, -sqrt(0.84))
        @test f.data[1] * abs(wi[3]) ≈ 1f0
        @test pdf == 1f0
        @test eta == 1.5f0
        @test flags == Hikari.BXDF_SPECULAR_TRANSMISSION
        @test !Hikari.can_scatter(lobe, Hikari.OnlyReflection())
        @test Hikari.lobe_sample(lobe, wo, Point2f(0.3), 0.5f0, lambda, Hikari.OnlyReflection()) == Hikari.NO_LOBE_SAMPLE
        inside = Hikari.other_side(lobe)
        @test inside.ior ≈ 1f0 / 1.5f0
        @test Hikari.lobe_sample(inside, Vec3f(0.8, 0, 0.6), Point2f(0.5), 0.5f0, lambda) == Hikari.NO_LOBE_SAMPLE
        rough = Hikari.MicrofacetLobe{:refraction}(Hikari.NoFresnel(color); alpha_x = 0.25f0, ior = 1.45f0, convention)
        @test Hikari.lobe_f(rough, n, n, lambda) == (Hikari.SpectralRadiance(0f0), 0f0)
        # At normal incidence every sampled microfacet refracts below the
        # surface for eta > 1. Integrate the directional PDF independently.
        count = 65536
        mass = sum(1:count) do i
            z = Float32((i - 0.5) / count)
            wi = Vec3f(sqrt(max(1f0 - z^2, 0f0)), 0, -z)
            last(Hikari.lobe_f(rough, n, wi, lambda))
        end * (2π / count)
        @test mass ≈ 1 atol = 0.003
        samples = [Hikari.lobe_sample(rough, wo, Point2f((i - 0.5) / 1024, mod(i * 0.61803398875, 1)),
                                     0.5f0, lambda) for i in 1:1024]
        valid = filter(s -> s[4] != 0x00, samples)
        @test length(valid) > 900
        @test all(s -> s[1][3] < 0f0 && isapprox(norm(s[1]), 1; atol = 2e-6), valid)
        @test all(s -> Hikari.lobe_f(rough, wo, s[1], lambda)[2] > 0f0, valid)
        @test all(s -> s[4] == Hikari.BXDF_GLOSSY_TRANSMISSION, valid)
    end
end
