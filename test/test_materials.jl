@testset "Fresnel Dielectric" begin
    # Vacuum gives no reflectance.
    @test Trace.fresnel_dielectric(1f0, 1f0, 1f0) ≈ 0f0
    @test Trace.fresnel_dielectric(0.5f0, 1f0, 1f0) ≈ 0f0
end

@testset "Fresnel Conductor" begin
    s = Trace.RGBSpectrum(1f0)
    @test Trace.fresnel_conductor(0f0, s, s, s) == s
    @test all(Trace.fresnel_conductor(cos(π / 4f0), s, s, s).c .> 0f0)
    @test all(Trace.fresnel_conductor(1f0, s, s, s).c .> 0f0)
end

@testset "SpecularReflection" begin
    sr = Trace.SpecularReflection(true, Trace.RGBSpectrum(1f0), Trace.FresnelNoOp())
    @test sr & (Trace.BSDF_SPECULAR | Trace.BSDF_REFLECTION)
end

@testset "SpecularTransmission" begin
    st = Trace.SpecularTransmission(
        true, Trace.RGBSpectrum(1f0), 1f0, 1f0,
        Trace.UInt8(1),
    )
    @test st & (Trace.BSDF_SPECULAR | Trace.BSDF_TRANSMISSION)
end

@testset "FresnelSpecular" begin
    f = Trace.FresnelSpecular(
        true, Trace.RGBSpectrum(1f0), Trace.RGBSpectrum(1f0),
        1f0, 1f0, Trace.UInt8(1),
    )
    @test f & (Trace.BSDF_SPECULAR | Trace.BSDF_REFLECTION | Trace.BSDF_TRANSMISSION)

    wo = Vec3f(0, 0, 1)
    u = Point2f(0, 0)
    wi, pdf, bxdf_value, sampled_type = Trace.sample_f(f, wo, u)
    @test wi ≈ -wo
    @test pdf ≈ 1f0
    @test sampled_type == Trace.BSDF_SPECULAR | Trace.BSDF_TRANSMISSION
end

@testset "MicrofacetReflection" begin
    m = Trace.MicrofacetReflection(
        true, Trace.RGBSpectrum(1f0),
        Trace.TrowbridgeReitzDistribution(1f0, 1f0),
        Trace.FresnelNoOp(),
        Trace.UInt8(1),
    )
    @test m & (Trace.BSDF_REFLECTION | Trace.BSDF_GLOSSY)
    wo = Vec3f(0, 0, 1)
    u = Point2f(0, 0)
    wi, pdf, bxdf_value, sampled_type = Trace.sample_f(m, wo, u)
    @test wi ≈ Vec3f(0, 0, 1)
end

@testset "MicrofacetTransmission" begin
    m = Trace.MicrofacetTransmission(
        true, Trace.RGBSpectrum(1f0),
        Trace.TrowbridgeReitzDistribution(1f0, 1f0),
        1f0, 2f0,
        Trace.UInt8(1),
    )
    @test m & (Trace.BSDF_TRANSMISSION | Trace.BSDF_GLOSSY)
    wo = Vec3f(0, 0, 1)
    u = Point2f(0, 0)
    wi, pdf, bxdf_value, sampled_type = Trace.sample_f(m, wo, u)
    @test wi ≈ Vec3f(0, 0, -1)
end
