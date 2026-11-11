# The uniform sky reaches a surface through the BSDF's samples only, as in pbrt-v4.
#
# pbrt's path tracers ask lights for samples with `allowIncompletePDF`, and a
# `UniformInfiniteLight` then gives none and reports zero density for the BSDF's samples that
# escape: those find it at full weight. Hikari sampled the sky uniformly and weighed the two
# strategies against each other. For an exact BSDF the sum is the same, but a layered BSDF's
# random-walk `f` and the paths its `Sample_f` walks do not agree exactly, and every coated
# material under the sky came out 0.9 to 1.8 % brighter than pbrt's (at 4096 samples on both
# sides, where pbrt agrees with itself to 0.07 %).

using Test, Hikari, Mantle, GeometryBasics, Colors
isdefined(@__MODULE__, :render_scene) || include(joinpath(@__DIR__, "pbrt", "suite.jl"))

@testset "the uniform sky is found by the BSDF's samples only" begin
    sky = Hikari.AmbientLight(RGB{Float32}(0.5f0, 0.5f0, 0.5f0))
    lambda = Hikari.sample_wavelengths_visible(0.3f0)
    sample = Hikari.sample_light_spectral(sky, Hikari.get_srgb_table(), nothing, Point3f(0), lambda, Point2f(0.3f0, 0.6f0))
    @test sample.pdf == 0
    @test Hikari.env_light_pdf_single(sky, nothing, Vec3f(0, 0, 1)) == 0

    name = "mat_coatedconductor_rough_light_ambient"
    ref, ours = ensure_reference(name), render_scene(name; samples = REFERENCE_SPP)
    @test compute_metrics(ref, ours).energy_ratio ≈ 1 rtol = 0.005
end
