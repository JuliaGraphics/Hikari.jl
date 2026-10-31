# A conductor can be given its colour instead of its IOR, as pbrt-v4's.
#
# pbrt makes the metal of reflectance `r` at each wavelength: IOR 1 + i 2√r/√(1 - r), whose
# reflectance at normal incidence is `r`. `Conductor`'s `reflectance` was documented as a tint and
# read by nothing, so a pbrt conductor given one rendered as the default metal, and a plot colour
# on a metal changed nothing. `CoatedConductor` converted its reflectance in RGB before the
# wavelengths were known; the conversion is not linear, and a red car paint came out 15 % too
# green, 3.2 % too bright over all.

using Test, Hikari, Mantle, Colors
isdefined(@__MODULE__, :render_scene) || include(joinpath(@__DIR__, "pbrt", "suite.jl"))

@testset "a conductor given its reflectance" begin
    @testset "reflects that reflectance at normal incidence" begin
        table = Hikari.get_srgb_table()
        lambda = Hikari.sample_wavelengths_visible(0.4f0)
        rgb = Hikari.RGBSpectrum(0.9f0, 0.5f0, 0.2f0)
        eta, k = Hikari.reflectance_ior(table, rgb, lambda)
        r = Hikari.uplift_rgb(table, rgb, lambda)
        f0 = ((eta - Hikari.SpectralRadiance(1f0)) * (eta - Hikari.SpectralRadiance(1f0)) + k * k) /
             ((eta + Hikari.SpectralRadiance(1f0)) * (eta + Hikari.SpectralRadiance(1f0)) + k * k)
        @test all(isapprox.(f0.data, r.data; rtol = 1f-5))
    end

    @testset "and not its IOR as well" begin
        m = Hikari.Conductor(reflectance = (0.9, 0.5, 0.2))
        @test isnothing(m.ior) && isnothing(m.extinction) && !isnothing(m.reflectance)
        @test isnothing(Hikari.Conductor(ior = (0.2, 0.3, 0.4), extinction = (3, 3, 3)).reflectance)
        @test isnothing(Hikari.Gold().reflectance)
        @test_throws ArgumentError Hikari.Conductor(ior = (0.2, 0.3, 0.4), reflectance = (0.9, 0.5, 0.2))
        c = Hikari.CoatedConductor(reflectance = (0.85, 0.1, 0.1))
        @test isnothing(c.conductor_ior) && !isnothing(c.conductor_reflectance)
        @test_throws ArgumentError Hikari.CoatedConductor(conductor_ior = (0.2, 0.3, 0.4), conductor_extinction = (3, 3, 3),
                                                          reflectance = (0.9, 0.5, 0.2))
    end

    @testset "renders as pbrt's: $name" for name in ("mat_conductor_reflectance_light_point",
                                                     "mat_coatedconductor_reflectance_light_point")
        ref, ours = ensure_reference(name), render_scene(name; samples = REFERENCE_SPP)
        for channel in (red, green, blue)
            @test sum(c -> Float64(channel(c)), ours) / sum(c -> Float64(channel(c)), ref) ≈ 1 rtol = 0.005
        end
    end
end
