using Test, Hikari, GeometryBasics, LinearAlgebra

below = [normalize(Vec3f(x, y, -0.2)) for (x, y) in ((1, 0), (0, 1), (-1, -0.5), (0.3, -1))]
sky_at(light, d) = light.env_map(d)

@testset "a sky without a ground is dark below the horizon" begin
    sky, _ = Hikari.sunsky_to_envlight(direction = normalize(Vec3f(1, 0, 0.1)), ground_enabled = false, resolution = 64)
    @test all(d -> sky_at(sky, d) == Hikari.RGBSpectrum(0f0), below)
    @test Hikari.luminance(sky_at(sky, Vec3f(0, 0, 1))) > 0
end

@testset "a sky with a ground shows the ground below the horizon" begin
    ground = Hikari.RGBSpectrum(0.25f0)
    sky, _ = Hikari.sunsky_to_envlight(direction = normalize(Vec3f(1, 0, 0.1)), ground_albedo = ground, resolution = 64)
    @test all(d -> sky_at(sky, d) == ground * 0.3f0, below)
end
