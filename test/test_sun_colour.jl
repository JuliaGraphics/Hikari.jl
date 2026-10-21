using Test, Hikari, GeometryBasics, LinearAlgebra, Colors

luminance(c) = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b
towards(elevation) = Vec3f(cosd(elevation), 0, sind(elevation))
redness(c) = c.r / c.b

@testset "the sun dims and reddens as it sets" begin
    overhead = Hikari.sunsky_sun_rgb(1.0, towards(90))
    @test luminance(overhead) ≈ 5 rtol = 1e-5
    @test Hikari.sunsky_sun_rgb(1.0) == overhead
    @test luminance(Hikari.sunsky_sun_rgb(2.0, towards(90))) ≈ 10 rtol = 1e-5

    setting = [Hikari.sunsky_sun_rgb(1.0, towards(e)) for e in (60, 30, 15, 5, 2)]
    @test issorted(luminance.(setting); rev = true)
    @test issorted(redness.(setting))
    @test Hikari.sunsky_sun_rgb(1.0, towards(-1)) == RGB{Float32}(0, 0, 0)
end

@testset "haze reddens a low sun further" begin
    clear, hazy = (Hikari.sunsky_sun_rgb(1.0, towards(8); turbidity) for turbidity in (2, 6))
    @test redness(hazy) > redness(clear)
end

@testset "a baked sun-and-sky lights with the sun at its elevation" begin
    dir = normalize(Vec3f(1, 0.5, 0.1))
    _, sun = Hikari.sunsky_to_envlight(direction = dir, turbidity = 3f0, resolution = 8)
    # Baked spectral skies use the model's disc-integrated irradiance, in the
    # same units as the sky, rather than the legacy normalized RGB helper.
    expected = Hikari.SunLight(Hikari.sunsky_sun_rgb(dir; intensity = 1, turbidity = 3), -dir)
    @test (sun.i.poly, sun.i.scale) == (expected.i.poly, expected.i.scale)
    @test sun.scale == expected.scale
end
