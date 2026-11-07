@testset "Fresnel Dielectric" begin
    # Vacuum gives no reflectance.
    @test Hikari.fresnel_dielectric(1f0, 1f0, 1f0) ≈ 0f0
    @test Hikari.fresnel_dielectric(0.5f0, 1f0, 1f0) ≈ 0f0
end

# NOTE: there are no tests here for standalone BxDF objects
# (`SpecularReflection`, `SpecularTransmission`, `MicrofacetReflection`,
# `MicrofacetTransmission`, `FresnelSpecular`, `fresnel_conductor`) or the
# `BSDF_*` flag constants: materials are the wrappers (`Diffuse`, `Conductor`,
# `Dielectric`, `CoatedDiffuse`, `CoatedConductor`, …). Per-material
# BSDF sampling is exercised end-to-end by `test/pbrt/test_pbrt_all_
# materials.jl` against pbrt-v4 reference EXRs, which gives stronger
# coverage than re-implementing unit tests against the new wrappers'
# internal `sample_bsdf_spectral` / `evaluate_bsdf_spectral` methods.
