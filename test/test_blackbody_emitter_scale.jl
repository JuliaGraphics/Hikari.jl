# The blackbody -> RGB emitter conversion normalizes LUMINANCE to 1, and that is
# deliberate. It looks wrong next to pbrt-v4, which normalizes its Planck
# spectrum so the PEAK value is 1 (`BlackbodySpectrum`: 1/Blackbody(λ_max, T),
# λ_max from Wien). Those differ by a lot — the peak-normalized spectrum has
# luminance 0.9796 at 5500 K and 0.2763 at 2700 K — so "make it match pbrt" is a
# tempting and plausible-looking change.
#
# It is a REGRESSION. pbrt is a spectral renderer: it never builds an emitter
# RGB, it samples wavelengths and resolves them in `PixelSensor`, which applies
# its own imaging ratio and white balance. Hikari's RGB pipeline consumes a
# luminance-normalized emitter instead. Measured on the two blackbody suite
# scenes (5500 K, `scale 10`), same harness, same pbrt reference:
#
#     formula                     smoothgold      bumpgold
#     Y normalized to 1           1.0004          0.9990
#     pbrt peak normalization     0.9808          0.9793   <- 2 % dark, and
#                                                             bumpgold's tile
#                                                             score crosses the
#                                                             0.07 suite gate
#
# This test pins the convention itself so the swap fails here, in a fast CPU-only
# test, rather than as a 2 % energy drift someone has to bisect out of a render.
using Test
using Hikari

# Rec.709 / sRGB luminance weights.
luminance(rgb) = 0.2126 * rgb[1] + 0.7152 * rgb[2] + 0.0722 * rgb[3]

@testset "blackbody emitter is luminance-normalized" begin
    for T in (2700f0, 4000f0, 5500f0, 6500f0, 9000f0)
        rgb = Hikari.blackbody_to_rgb(T)
        @test luminance(rgb) ≈ 1.0 rtol = 1e-3
        @test all(>=(0.0), rgb)     # negatives wreck the radiance estimator
    end
end

@testset "chromaticity tracks the Planckian locus" begin
    for T in (2700f0, 5500f0, 9000f0)
        r, g, b = Hikari.blackbody_to_rgb(T)
        X = 0.4124r + 0.3576g + 0.1805b
        Y = 0.2126r + 0.7152g + 0.0722b
        Z = 0.0193r + 0.1192g + 0.9505b
        s = X + Y + Z
        x_ref, y_ref = Hikari.planckian_xy(T)
        @test X / s ≈ x_ref atol = 2e-3
        @test Y / s ≈ y_ref atol = 2e-3
    end
    # Warmer is redder, cooler is bluer — guards against a channel swap that
    # luminance alone would not catch.
    warm = Hikari.blackbody_to_rgb(2700f0)
    cool = Hikari.blackbody_to_rgb(9000f0)
    @test warm[1] / warm[3] > cool[1] / cool[3]
end

@testset "NOT pbrt's peak normalization" begin
    # If someone swaps in pbrt's BlackbodySpectrum scaling, luminance stops being
    # 1 and picks up these values instead. Asserting the gap explicitly documents
    # which convention is in force.
    peak_normalized_luminance = Dict(2700f0 => 0.2763, 5500f0 => 0.9796, 9000f0 => 0.5509)
    for (T, Y_pbrt) in peak_normalized_luminance
        @test !isapprox(luminance(Hikari.blackbody_to_rgb(T)), Y_pbrt; rtol = 1e-2)
    end
end

@testset "blackbody_normalized matches pbrt's peak convention" begin
    # The SPD primitive itself does follow pbrt (max over wavelengths == 1); it
    # is only the emitter RGB conversion that differs. Building `λ_max` as a
    # Vector{Float64} throws a MethodError on every call, since
    # `blackbody` only accepted Vector{Float32} — so it was dead code.
    T = 5500f0
    λ_peak = Hikari.blackbody_peak_wavelength(T)
    @test Hikari.blackbody_normalized(λ_peak, T) ≈ 1.0 rtol = 1e-6
    λs = Float32.(360:830)
    spd = Hikari.blackbody_normalized(λs, T)
    @test length(spd) == length(λs)
    @test maximum(spd) ≈ 1.0 rtol = 1e-4
    @test all(0 .<= spd .<= 1.0 + 1e-4)
    # Wien's law: hotter bodies peak at shorter wavelengths.
    @test Hikari.blackbody_peak_wavelength(9000f0) < Hikari.blackbody_peak_wavelength(2700f0)
end
