# Test file for environment light compatibility with PBRT-v4
# Verifies that our environment light sampling matches pbrt-v4 GPU volpath integrator
#
# Key differences from PBRT-v4:
# - PBRT uses Equal-Area sphere mapping (constant Jacobian 1/(4π))
# - We use Equirectangular mapping (variable Jacobian 1/(2π² sin(θ)))
# Both are valid for their respective image formats.
#
# What MUST match:
# 1. Monte Carlo estimator convergence (integral over sphere)
# 2. Unbounded RGB->Spectral conversion for HDR emission values
# 3. PDF * radiance / pdf should give unbiased estimate

using Test
using GeometryBasics
using LinearAlgebra

# =============================================================================
# Mock scene for testing
# =============================================================================
struct MockScene <: Hikari.AbstractScene
    world_radius::Float32
    world_center::Point3f
end
MockScene() = MockScene(100f0, Point3f(0, 0, 0))

# =============================================================================
# Test 1: Verify equirectangular PDF formula
# For equirectangular: pdf_solidangle = pdf_image / (2π² sin(θ))
# =============================================================================
@testset "Equirectangular PDF Formula" begin
    # The solid angle element in equirectangular is:
    # dω = sin(θ) dθ dφ = sin(θ) * (π * dv) * (2π * du) = 2π² sin(θ) du dv
    # So: pdf_solidangle = pdf_image / (2π² sin(θ))

    # At equator (v = 0.5, θ = π/2): sin(θ) = 1, so pdf = pdf_image / (2π²)
    θ_equator = Float32(π) / 2
    jacobian_equator = 2f0 * Float32(π)^2 * sin(θ_equator)
    @test jacobian_equator ≈ 2f0 * Float32(π)^2 atol=1e-5

    # Near pole (v = 0.1, θ = 0.1π): sin(θ) ≈ 0.309
    θ_pole = 0.1f0 * Float32(π)
    jacobian_pole = 2f0 * Float32(π)^2 * sin(θ_pole)
    @test jacobian_pole < jacobian_equator  # Less solid angle near poles

    # Verify sin(θ) weighting compensates for this
    # Pixels near poles cover less solid angle, so need lower weight in distribution
end

# =============================================================================
# Test 2: Verify Distribution2D sin(θ) weighting
# =============================================================================
@testset "Distribution2D sin(θ) Weighting" begin
    h, w = 32, 64

    # Uniform radiance environment map
    data = [Hikari.RGBSpectrum(1f0) for _ in 1:h, _ in 1:w]
    env_map = Hikari.EnvironmentMap(Matrix(data))

    # Check that marginal distribution reflects sin(θ) weighting
    # Row at equator (v=h/2) should have highest weight
    # Rows near poles (v=1, v=h) should have lowest weight

    equator_row = div(h, 2)
    pole_row = 1

    marginal_equator = env_map.distribution.p_marginal.func[equator_row]
    marginal_pole = env_map.distribution.p_marginal.func[pole_row]

    # Equator should have much higher marginal than pole
    @test marginal_equator > marginal_pole * 2

    # The ratio should approximately be sin(π/2) / sin(π/h) ≈ 1 / sin(π/h)
    expected_ratio = sin(Float32(π) * (equator_row - 0.5f0) / h) /
                     sin(Float32(π) * (pole_row - 0.5f0) / h)
    actual_ratio = marginal_equator / marginal_pole
    @test actual_ratio ≈ expected_ratio rtol=0.15
end

# =============================================================================
# Test 3: Direction<->UV roundtrip consistency
# =============================================================================
@testset "Direction UV Roundtrip" begin
    for _ in 1:100
        # Random direction on sphere (avoiding exact poles)
        θ = 0.01f0 + rand(Float32) * (Float32(π) - 0.02f0)
        φ = rand(Float32) * 2f0 * Float32(π) - Float32(π)
        dir = Vec3f(sin(θ) * cos(φ), cos(θ), sin(θ) * sin(φ))
        dir = normalize(dir)

        # Convert to UV and back
        uv = Hikari.direction_to_uv(dir)
        dir2 = Hikari.uv_to_direction(uv)

        @test isapprox(dir[1], dir2[1], atol=1e-4)
        @test isapprox(dir[2], dir2[2], atol=1e-4)
        @test isapprox(dir[3], dir2[3], atol=1e-4)
    end
end

# =============================================================================
# Test 4: Monte Carlo integration - uniform environment
# The integral of constant radiance L over sphere should be 4π*L
# =============================================================================
@testset "Monte Carlo Integration - Uniform" begin
    h, w = 16, 32
    L = 2.5f0  # Constant radiance
    data = [Hikari.RGBSpectrum(L) for _ in 1:h, _ in 1:w]
    env_map = Hikari.EnvironmentMap(Matrix(data))
    env_light = Hikari.EnvironmentLight(env_map)

    interaction = Hikari.Interaction(
        Point3f(0, 0, 0), 0f0, Vec3f(0, 1, 0), Hikari.Normal3f(0, 1, 0)
    )

    # Monte Carlo estimate: E[L/pdf] should converge to integral
    n_samples = 50000
    estimate = Hikari.RGBSpectrum(0f0)
    mock_scene = MockScene()
    for _ in 1:n_samples
        u = Point2f(rand(Float32), rand(Float32))
        radiance, wi, pdf, _ = Hikari.sample_li(env_light, interaction, u, mock_scene)
        if pdf > 0
            estimate = estimate + radiance / pdf
        end
    end
    estimate = estimate / n_samples

    # Expected: integral of L over sphere = 4π * L
    expected = L * 4f0 * Float32(π)
    actual = Hikari.to_Y(estimate)  # Luminance (grayscale)

    @test isapprox(actual, expected, rtol=0.05)
end

# =============================================================================
# Test 5: Monte Carlo integration - non-uniform (bright spot)
# =============================================================================
@testset "Monte Carlo Integration - Bright Spot" begin
    h, w = 64, 128

    # Create environment with bright spot at center
    data = Matrix{Hikari.RGBSpectrum}(undef, h, w)
    total_power = 0f0
    for v in 1:h, u in 1:w
        # Bright spot at center
        uv_center = Point2f(0.5f0, 0.5f0)
        uv_pixel = Point2f((u - 0.5f0) / w, (v - 0.5f0) / h)
        dist = sqrt((uv_pixel[1] - uv_center[1])^2 + (uv_pixel[2] - uv_center[2])^2)
        brightness = exp(-dist^2 * 50f0) * 10f0 + 0.1f0
        data[v, u] = Hikari.RGBSpectrum(Float32(brightness))

        # Compute expected integral (weighted by sin(θ))
        θ = Float32(π) * (v - 0.5f0) / h
        # dω = 2π² sin(θ) du dv, with du = 1/w, dv = 1/h
        dω = 2f0 * Float32(π)^2 * sin(θ) / (w * h)
        total_power += brightness * dω
    end

    env_map = Hikari.EnvironmentMap(data)
    env_light = Hikari.EnvironmentLight(env_map)

    interaction = Hikari.Interaction(
        Point3f(0, 0, 0), 0f0, Vec3f(0, 1, 0), Hikari.Normal3f(0, 1, 0)
    )

    # Monte Carlo estimate
    n_samples = 100000
    estimate = 0f0
    mock_scene = MockScene()
    for _ in 1:n_samples
        u = Point2f(rand(Float32), rand(Float32))
        radiance, wi, pdf, _ = Hikari.sample_li(env_light, interaction, u, mock_scene)
        if pdf > 0
            estimate += Hikari.to_Y(radiance) / pdf
        end
    end
    estimate /= n_samples

    @test isapprox(estimate, total_power, rtol=0.1)
end

# =============================================================================
# Test 6: PDF consistency - sample_li and pdf_li should match
# =============================================================================
@testset "PDF Consistency" begin
    h, w = 32, 64

    # Non-uniform environment
    data = Matrix{Hikari.RGBSpectrum}(undef, h, w)
    for v in 1:h, u in 1:w
        brightness = 0.5f0 + 0.5f0 * sin(Float32(u) / w * 4f0 * Float32(π))
        data[v, u] = Hikari.RGBSpectrum(Float32(brightness))
    end

    env_map = Hikari.EnvironmentMap(data)
    env_light = Hikari.EnvironmentLight(env_map)

    interaction = Hikari.Interaction(
        Point3f(0, 0, 0), 0f0, Vec3f(0, 1, 0), Hikari.Normal3f(0, 1, 0)
    )

    mock_scene = MockScene()
    for _ in 1:100
        u = Point2f(rand(Float32), rand(Float32))
        _, wi, pdf_sample, _ = Hikari.sample_li(env_light, interaction, u, mock_scene)

        if pdf_sample > 1e-6  # Avoid near-pole samples
            pdf_eval = Hikari.pdf_li(env_light, interaction, wi)
            @test isapprox(pdf_sample, pdf_eval, rtol=0.05)
        end
    end
end

# =============================================================================
# Test 7: HDR Values - Unbounded RGB conversion (critical for sun/shadows!)
# =============================================================================
@testset "HDR Unbounded RGB Conversion" begin
    table = Hikari.get_srgb_table()
    lambda = Hikari.sample_wavelengths_uniform(0.5f0)

    # Test with HDR sun-like values (matching your sky image extrema)
    # extrema were: (Float32[0.0, 0.0, 0.0], Float32[20896.0, 20272.0, 18304.0])
    hdr_rgb = Hikari.RGBSpectrum(20896f0, 20272f0, 18304f0)

    # Convert using unbounded method
    spectral = Hikari.uplift_rgb_unbounded(table, hdr_rgb, lambda)

    # The spectral values should be proportional to the input RGB
    # Not clamped to [0,1] like bounded method would do
    @test !Hikari.is_black(spectral)
    @test spectral.data[1] > 1000f0  # Should preserve HDR magnitude
    @test spectral.data[2] > 1000f0
    @test spectral.data[3] > 1000f0
    @test spectral.data[4] > 1000f0

    # Compare with bounded method (which would clip)
    spectral_bounded = Hikari.uplift_rgb(table, hdr_rgb, lambda)

    # Unbounded should be much larger than bounded for HDR values
    @test spectral.data[1] > spectral_bounded.data[1] * 100
end

# =============================================================================
# Test 8: Environment Light Direct Lighting Sample - HDR sun
# =============================================================================
@testset "Environment Light Direct Lighting - HDR Sun" begin
    table = Hikari.get_srgb_table()

    h, w = 32, 64

    # Create environment with very bright sun
    data = Matrix{Hikari.RGBSpectrum}(undef, h, w)
    sun_intensity = 20000f0
    sky_intensity = 0.15f0

    for v in 1:h, u in 1:w
        # Small bright sun at specific location
        sun_u, sun_v = 16, 16  # Center
        dist = sqrt(Float32((u - sun_u)^2 + (v - sun_v)^2))
        if dist < 2
            data[v, u] = Hikari.RGBSpectrum(sun_intensity)
        else
            data[v, u] = Hikari.RGBSpectrum(sky_intensity)
        end
    end

    env_map = Hikari.EnvironmentMap(data)
    env_light = Hikari.EnvironmentLight(env_map)

    p = Point3f(0, 0, 0)
    lambda = Hikari.sample_wavelengths_uniform(0.5f0)

    # Sample many times and check that we sometimes get very bright samples
    max_Li = 0f0
    sun_samples = 0
    n_samples = 10000

    for _ in 1:n_samples
        u = Point2f(rand(Float32), rand(Float32))
        sample = Hikari.sample_light_spectral(table, env_light, p, lambda, u)

        if !Hikari.is_black(sample.Li) && sample.pdf > 0
            Li_max = maximum(sample.Li.data)
            max_Li = max(max_Li, Li_max)

            # Count samples that hit the sun
            if Li_max > sun_intensity * 0.5f0
                sun_samples += 1
            end
        end
    end

    # We should get some very bright samples hitting the sun
    @test max_Li > sun_intensity * 0.5f0
    # And importance sampling should find the sun frequently
    @test sun_samples > n_samples * 0.01  # At least 1% should hit sun

    println("Max Li sampled: $max_Li (sun intensity: $sun_intensity)")
    println("Sun hit rate: $(sun_samples / n_samples * 100)%")
end

# =============================================================================
# Test 9: Verify sample_light_spectral uses unbounded uplift
# =============================================================================
@testset "sample_light_spectral Uses Unbounded Uplift" begin
    table = Hikari.get_srgb_table()

    h, w = 8, 16

    # Create environment with HDR value
    hdr_value = 10000f0
    data = [Hikari.RGBSpectrum(hdr_value) for _ in 1:h, _ in 1:w]
    env_map = Hikari.EnvironmentMap(Matrix(data))
    env_light = Hikari.EnvironmentLight(env_map)

    p = Point3f(0, 0, 0)
    lambda = Hikari.sample_wavelengths_uniform(0.5f0)

    # Sample the light
    u = Point2f(0.5f0, 0.5f0)
    sample = Hikari.sample_light_spectral(table, env_light, p, lambda, u)

    # The Li should preserve HDR magnitude (not be clamped to ~1)
    @test sample.Li.data[1] > hdr_value * 0.5f0
    @test sample.Li.data[2] > hdr_value * 0.5f0
    @test sample.Li.data[3] > hdr_value * 0.5f0
    @test sample.Li.data[4] > hdr_value * 0.5f0

    println("Sampled Li: $(sample.Li.data)")
    println("Expected > $(hdr_value * 0.5f0)")
end

# =============================================================================
# Test 10: Shadow ray contribution should be proportional to sun intensity
# =============================================================================
@testset "Shadow Ray Contribution Magnitude" begin
    table = Hikari.get_srgb_table()

    h, w = 16, 32

    # Test two environments: one with normal sun, one 100x brighter
    sun_intensity_1 = 100f0
    sun_intensity_2 = 10000f0

    function create_sun_env(intensity)
        data = Matrix{Hikari.RGBSpectrum}(undef, h, w)
        for v in 1:h, u in 1:w
            dist = sqrt(Float32((u - w÷2)^2 + (v - h÷2)^2))
            if dist < 2
                data[v, u] = Hikari.RGBSpectrum(intensity)
            else
                data[v, u] = Hikari.RGBSpectrum(0.1f0)
            end
        end
        env_map = Hikari.EnvironmentMap(data)
        return Hikari.EnvironmentLight(env_map)
    end

    env1 = create_sun_env(sun_intensity_1)
    env2 = create_sun_env(sun_intensity_2)

    p = Point3f(0, 0, 0)
    lambda = Hikari.sample_wavelengths_uniform(0.5f0)

    # Sample both and compare - use same random seed
    n_samples = 1000
    total_1 = 0f0
    total_2 = 0f0

    for i in 1:n_samples
        # Use deterministic samples for fair comparison
        u = Point2f(Float32(i % 100) / 100f0, Float32(i ÷ 100) / (n_samples ÷ 100))

        sample1 = Hikari.sample_light_spectral(table, env1, p, lambda, u)
        sample2 = Hikari.sample_light_spectral(table, env2, p, lambda, u)

        if sample1.pdf > 0
            total_1 += maximum(sample1.Li.data) / sample1.pdf
        end
        if sample2.pdf > 0
            total_2 += maximum(sample2.Li.data) / sample2.pdf
        end
    end

    # The ratio should approximately match the sun intensity ratio
    ratio = total_2 / total_1
    expected_ratio = sun_intensity_2 / sun_intensity_1

    # Should be within 50% of expected (accounting for Monte Carlo variance)
    @test ratio > expected_ratio * 0.5
    @test ratio < expected_ratio * 2.0

    println("Intensity ratio: $ratio (expected: $expected_ratio)")
end

println("\n" * "="^60)
println("Environment Light PBRT-v4 Compatibility Tests")
println("="^60 * "\n")
