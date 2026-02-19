@testset "Fresnel Dielectric" begin
    # Vacuum gives no reflectance.
    @test Hikari.fresnel_dielectric(1f0, 1f0, 1f0) ≈ 0f0
    @test Hikari.fresnel_dielectric(0.5f0, 1f0, 1f0) ≈ 0f0
end

@testset "Fresnel Conductor" begin
    s = Hikari.RGBSpectrum(1f0)
    @test Hikari.fresnel_conductor(0f0, s, s, s) == s
    @test all(Hikari.fresnel_conductor(cos(π / 4f0), s, s, s).c .> 0f0)
    @test all(Hikari.fresnel_conductor(1f0, s, s, s).c .> 0f0)
end

@testset "SpecularReflection" begin
    sr = Hikari.SpecularReflection(true, Hikari.RGBSpectrum(1f0), Hikari.FresnelNoOp())
    @test sr & (Hikari.BSDF_SPECULAR | Hikari.BSDF_REFLECTION)
end

@testset "SpecularTransmission" begin
    st = Hikari.SpecularTransmission(
        true, Hikari.RGBSpectrum(1f0), 1f0, 1f0,
        Hikari.UInt8(1),
    )
    @test st & (Hikari.BSDF_SPECULAR | Hikari.BSDF_TRANSMISSION)
end

@testset "FresnelSpecular" begin
    f = Hikari.FresnelSpecular(
        true, Hikari.RGBSpectrum(1f0), Hikari.RGBSpectrum(1f0),
        1f0, 1f0, Hikari.UInt8(1),
    )
    @test f & (Hikari.BSDF_SPECULAR | Hikari.BSDF_REFLECTION | Hikari.BSDF_TRANSMISSION)

    wo = Vec3f(0, 0, 1)
    u = Point2f(0, 0)
    wi, pdf, bxdf_value, sampled_type = Hikari.sample_f(f, wo, u)
    @test wi ≈ -wo
    @test pdf ≈ 1f0
    @test sampled_type == Hikari.BSDF_SPECULAR | Hikari.BSDF_TRANSMISSION
end

@testset "MicrofacetReflection" begin
    m = Hikari.MicrofacetReflection(
        true, Hikari.RGBSpectrum(1f0),
        Hikari.TrowbridgeReitzDistribution(1f0, 1f0),
        Hikari.FresnelNoOp(),
        Hikari.UInt8(1),
    )
    @test m & (Hikari.BSDF_REFLECTION | Hikari.BSDF_GLOSSY)
    wo = Vec3f(0, 0, 1)
    u = Point2f(0, 0)
    wi, pdf, bxdf_value, sampled_type = Hikari.sample_f(m, wo, u)
    @test wi ≈ Vec3f(0, 0, 1)
end

@testset "MicrofacetTransmission" begin
    m = Hikari.MicrofacetTransmission(
        true, Hikari.RGBSpectrum(1f0),
        Hikari.TrowbridgeReitzDistribution(1f0, 1f0),
        1f0, 2f0,
        Hikari.UInt8(1),
    )
    @test m & (Hikari.BSDF_TRANSMISSION | Hikari.BSDF_GLOSSY)
    wo = Vec3f(0, 0, 1)
    u = Point2f(0, 0)
    wi, pdf, bxdf_value, sampled_type = Hikari.sample_f(m, wo, u)
    @test wi ≈ Vec3f(0, 0, -1)
end
