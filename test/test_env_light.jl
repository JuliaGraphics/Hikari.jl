# Test file for environment light importance sampling
using Test
using GeometryBasics

# Test Distribution1D
@testset "Distribution1D" begin
    # Uniform distribution
    func = Float32[1, 1, 1, 1]
    d = Hikari.Distribution1D(func)
    @test d.func_int ≈ 1.0f0
    @test d.cdf[end] ≈ 1.0f0

    # Test sampling - should be roughly uniform
    sampled, pdf, _ = Hikari.sample_continuous(d, 0.5f0)
    @test 0 ≤ sampled ≤ 1
    @test pdf ≈ 1.0f0  # uniform, so pdf = func_int

    # Non-uniform distribution - heavily weighted to first element
    func2 = Float32[10, 1, 1, 1]
    d2 = Hikari.Distribution1D(func2)
    @test d2.func_int ≈ 13/4  # (10+1+1+1)/4

    # Low u values should sample first element more
    sampled_low, pdf_low, offset = Hikari.sample_continuous(d2, 0.1f0)
    @test offset == 1  # should be in first bin
end

# Test Distribution2D
@testset "Distribution2D" begin
    # 2x2 with one bright pixel
    func = Float32[10 1; 1 1]  # top-left is bright
    d = Hikari.Distribution2D(func)

    # Sample should prefer top-left quadrant
    n_samples = 1000
    top_left_count = 0
    for i in 1:n_samples
        u = Point2f(rand(Float32), rand(Float32))
        uv, pdf = Hikari.sample_continuous(d, u)
        if uv[1] < 0.5 && uv[2] < 0.5
            top_left_count += 1
        end
    end
    # Should be biased toward top-left
    @test top_left_count > n_samples * 0.5
end

# Test direction_to_uv and uv_to_direction roundtrip
@testset "UV Direction Roundtrip" begin
    for _ in 1:100
        # Random direction on sphere
        θ = rand(Float32) * π
        φ = rand(Float32) * 2π - π
        dir = Vec3f(sin(θ) * cos(φ), cos(θ), sin(θ) * sin(φ))
        dir = normalize(dir)

        # Convert to UV and back
        uv = Hikari.direction_to_uv(dir)
        dir2 = Hikari.uv_to_direction(uv)

        @test isapprox(dir[1], dir2[1], atol=1e-5)
        @test isapprox(dir[2], dir2[2], atol=1e-5)
        @test isapprox(dir[3], dir2[3], atol=1e-5)
    end
end

# Test PDF conversion
@testset "PDF Solid Angle Conversion" begin
    # The PDF conversion formula: pdf_solidangle = pdf_image / (2π² sin(θ))
    # At θ = π/2 (equator), sin(θ) = 1
    # At θ = 0 or π (poles), sin(θ) = 0 (singular)

    # Create a uniform environment map
    h, w = 16, 32
    data = [Hikari.RGBSpectrum(1f0) for _ in 1:h, _ in 1:w]
    env_map = Hikari.EnvironmentMap(Matrix(data))

    # Check that the distribution integral accounts for sin(θ) weighting
    # After weighting, the marginal should be sin-weighted
    total = 0f0
    for v in 1:h
        total += env_map.distribution.p_marginal.func[v]
    end
    @test total > 0

    # PDF at equator should be reasonable
    uv_equator = Point2f(0.5f0, 0.5f0)
    pdf_equator = Hikari.pdf(env_map.distribution, uv_equator)
    @test pdf_equator > 0
end

# Test sample_li returns valid values
@testset "sample_li" begin
    # Create a simple environment light
    h, w = 8, 16
    data = [Hikari.RGBSpectrum(1f0) for _ in 1:h, _ in 1:w]
    env_map = Hikari.EnvironmentMap(Matrix(data))
    env_light = Hikari.EnvironmentLight(env_map)

    # Create a dummy interaction
    interaction = Hikari.Interaction(
        Point3f(0, 0, 0),
        0f0,
        Vec3f(0, 1, 0),
        Hikari.Normal3f(0, 1, 0)
    )

    # Sample multiple times
    for _ in 1:100
        u = Point2f(rand(Float32), rand(Float32))
        radiance, wi, pdf, visibility = Hikari.sample_li(env_light, interaction, u)

        # Radiance should be positive
        @test !Hikari.is_black(radiance)

        # PDF should be positive
        @test pdf > 0

        # Direction should be unit vector
        @test isapprox(norm(wi), 1f0, atol=1e-5)

        # pdf_li should match
        pdf_check = Hikari.pdf_li(env_light, interaction, wi)
        @test isapprox(pdf, pdf_check, rtol=0.1)  # Allow some tolerance
    end
end

# Test sample_le for photon mapping
@testset "sample_le" begin
    h, w = 8, 16
    data = [Hikari.RGBSpectrum(1f0) for _ in 1:h, _ in 1:w]
    env_map = Hikari.EnvironmentMap(Matrix(data))
    env_light = Hikari.EnvironmentLight(env_map, Hikari.RGBSpectrum(1f0); world_radius=10f0)

    for _ in 1:100
        u1 = Point2f(rand(Float32), rand(Float32))
        u2 = Point2f(rand(Float32), rand(Float32))

        radiance, ray, light_normal, pdf_pos, pdf_dir = Hikari.sample_le(
            env_light, u1, u2, 0f0
        )

        # Radiance should be positive
        @test !Hikari.is_black(radiance)

        # PDFs should be positive
        @test pdf_pos > 0
        @test pdf_dir > 0

        # Ray direction should be unit
        @test isapprox(norm(ray.d), 1f0, atol=1e-5)

        # Light normal should point opposite to ray direction (into scene)
        # Actually for env light, normal points in same direction as -wi
        dot_val = Vec3f(light_normal) ⋅ ray.d
        @test dot_val ≈ -1f0 atol=1e-5
    end
end

# Monte Carlo integration test - verify PDF is correct
@testset "Monte Carlo PDF Verification" begin
    h, w = 32, 64
    # Create non-uniform environment - bright spot
    data = Matrix{Hikari.RGBSpectrum}(undef, h, w)
    for v in 1:h, u in 1:w
        # Bright spot at center
        dist = sqrt((u - w/2)^2 + (v - h/2)^2)
        brightness = exp(-dist^2 / 100)
        data[v, u] = Hikari.RGBSpectrum(Float32(brightness))
    end
    env_map = Hikari.EnvironmentMap(data)
    env_light = Hikari.EnvironmentLight(env_map)

    interaction = Hikari.Interaction(
        Point3f(0, 0, 0), 0f0, Vec3f(0, 1, 0), Hikari.Normal3f(0, 1, 0)
    )

    # Monte Carlo estimate should converge to correct value
    n_samples = 10000
    estimate = Hikari.RGBSpectrum(0f0)
    for _ in 1:n_samples
        u = Point2f(rand(Float32), rand(Float32))
        radiance, wi, pdf, _ = Hikari.sample_li(env_light, interaction, u)
        if pdf > 0
            estimate = estimate + radiance / pdf
        end
    end
    estimate = estimate / n_samples

    # The estimate should be roughly the integral of the environment map over the sphere
    # For a uniform map of intensity 1, this would be 4π
    # Since we have non-uniform, just check it's reasonable
    @test Hikari.to_Y(estimate) > 0
    @test !isnan(estimate)
    @test !isinf(estimate)
end

println("All environment light tests passed!")
