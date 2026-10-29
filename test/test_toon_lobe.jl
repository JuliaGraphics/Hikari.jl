using Test, Hikari, LinearAlgebra
using GeometryBasics: Vec3f, Point2f

@testset "Cycles Toon lobes" begin
    lambda = Hikari.sample_wavelengths_visible(0.4f0)
    n = Vec3f(0, 0, 1)
    for glossy in (false, true), (size, smooth) in ((0.4f0, 0f0), (0.4f0, 0.2f0), (0.8f0, 0.6f0))
        lobe = Hikari.ToonLobe{glossy}(size, smooth)
        @test Hikari.is_diffuse_lobe(lobe) == !glossy
        # Integrate f*cos and the PDF over the cone. This checks Hikari's
        # cosine convention independently of its sample/evaluate consistency.
        integral, mass = 0.0, 0.0
        count = 4096
        for i in 1:count
            z = 1f0 - Float32((i - 0.5) / count) * lobe.one_minus_cos_angle
            wi = Vec3f(sqrt(max(1f0 - z^2, 0f0)), 0, z)
            f, pdf = Hikari.lobe_f(lobe, n, wi, lambda)
            integral += f.data[1] * z
            mass += pdf
        end
        solid_angle = 2π * lobe.one_minus_cos_angle
        @test mass * solid_angle / count ≈ 1 atol = 2e-5
        delta = Float64(lobe.sample_angle) - lobe.size
        expected = smooth == 0f0 ? 1.0 :
            (1 - cos(lobe.sample_angle) + delta / lobe.smooth * cos(lobe.sample_angle) -
             (sin(Float64(lobe.sample_angle)) - sin(Float64(lobe.size))) / lobe.smooth) / lobe.one_minus_cos_angle
        @test integral * solid_angle / count ≈ expected atol = 2e-5

        wo = normalize(Vec3f(0.6, 0, 0.8))
        samples = [Hikari.lobe_sample(lobe, wo, Point2f((i - 0.5) / 256, mod(i * 0.61803398875, 1)),
                                     0.5f0, lambda) for i in 1:256]
        valid = filter(s -> s[4] != 0x00, samples)
        @test !isempty(valid)
        @test all(s -> isapprox(norm(s[1]), 1; atol = 2e-6), valid)
        @test all(s -> s[1][3] > 0f0, valid)
        @test all(s -> Hikari.lobe_f(lobe, wo, s[1], lambda)[2] > 0f0, valid)
        @test all(s -> s[4] == (glossy ? Hikari.BXDF_GLOSSY_REFLECTION : Hikari.BXDF_DIFFUSE_REFLECTION), valid)
        @test Hikari.lobe_f(lobe, wo, -n, lambda)[2] == 0f0
        @test Hikari.lobe_f(lobe, wo, n, lambda, Hikari.OnlyTransmission())[2] == 0f0
    end
    tiny = Hikari.ToonLobe{false}(0f0, 0f0)
    @test tiny.one_minus_cos_angle > 0f0
    @test all(isfinite, first(Hikari.lobe_f(tiny, n, n, lambda)).data)
    @test Hikari.lobe_sample(Hikari.ToonLobe{true}(0.5f0, 0f0), -n, Point2f(0.5), 0.5f0, lambda) == Hikari.NO_LOBE_SAMPLE
end
