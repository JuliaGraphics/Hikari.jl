using Test, Hikari, GeometryBasics, LinearAlgebra, Colors

angle_between(a, b) = atan(norm(cross(Vec3{Float64}(a), Vec3{Float64}(b))), dot(Vec3{Float64}(a), Vec3{Float64}(b)))

@testset "sun is sampled over its disc" begin
    travel = normalize(Vec3f(0.3, -0.4, -0.8))
    sun = Hikari.SunLight(RGB{Float32}(1, 1, 1), travel; angular_diameter = 0.01f0)
    towards_sun = -sun.direction

    @test Hikari.sun_disc_direction(sun, Point2f(0, 0.3)) ≈ towards_sun
    rim = [angle_between(Hikari.sun_disc_direction(sun, Point2f(1, ϕ)), towards_sun) for ϕ in 0:0.125:0.875]
    @test rim ≈ fill(0.005, 8) rtol = 1e-3
    @test angle_between(Hikari.sun_disc_direction(sun, Point2f(0.25, 0.6)), towards_sun) ≈ 0.0025 rtol = 1e-3
    @test all(u -> norm(Hikari.sun_disc_direction(sun, u)) ≈ 1, [Point2f(a, b) for a in 0:0.1:1, b in 0:0.1:1])

    point_sun = Hikari.SunLight(RGB{Float32}(1, 1, 1), travel; angular_diameter = 0f0)
    @test Hikari.sun_disc_direction(point_sun, Point2f(0.7, 0.2)) == -point_sun.direction
end
