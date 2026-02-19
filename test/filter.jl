using Test
using Hikari
using GeometryBasics
using Statistics

@testset "Filter Sampling" begin

    @testset "Gaussian Filter Integral" begin
        # Test that func_integral matches numerical integration
        for (radius, sigma) in [(1.5f0, 0.5f0), (2.0f0, 0.5f0), (3.0f0, 1.0f0), (5.0f0, 2.0f0)]
            filter = Hikari.GaussianFilter(Point2f(radius, radius), sigma)
            data = Hikari.GPUFilterSamplerData(filter)

            # Numerical integration of Gaussian filter
            n = 1000
            r = radius
            xs = range(-r, r, n)
            ys = range(-r, r, n)
            dx_num = xs[2] - xs[1]
            dy_num = ys[2] - ys[1]

            exp_r = exp(-r^2 / (2 * sigma^2))
            numerical_integral = 0.0
            for x in xs, y in ys
                g = exp(-(x^2 + y^2) / (2 * sigma^2)) - exp_r
                numerical_integral += max(0.0, g) * dx_num * dy_num
            end

            @test isapprox(data.func_integral, numerical_integral, rtol=0.02) ||
                  @show (radius, sigma, data.func_integral, numerical_integral)
        end
    end

    @testset "Filter Weight = func_integral (importance sampling property)" begin
        # With proper importance sampling, weight = f(x) / pdf(x) = integral (constant)
        for (radius, sigma) in [(1.5f0, 0.5f0), (2.0f0, 0.5f0), (3.0f0, 1.0f0)]
            filter = Hikari.GaussianFilter(Point2f(radius, radius), sigma)
            data = Hikari.GPUFilterSamplerData(filter)

            # Sample many points and check weights
            n_samples = 10000
            weights = Float32[]
            for _ in 1:n_samples
                u = Point2f(rand(Float32), rand(Float32))
                fs = Hikari.filter_sample_tabulated(data, u)
                push!(weights, fs.weight)
            end

            # All weights should be approximately equal to func_integral
            @test isapprox(mean(weights), data.func_integral, rtol=0.001)
            @test isapprox(minimum(weights), data.func_integral, rtol=0.01)
            @test isapprox(maximum(weights), data.func_integral, rtol=0.01)
        end
    end

    @testset "Sampled positions are within filter domain" begin
        for (radius, sigma) in [(1.5f0, 0.5f0), (3.0f0, 1.0f0), (5.0f0, 2.0f0)]
            filter = Hikari.GaussianFilter(Point2f(radius, radius), sigma)
            data = Hikari.GPUFilterSamplerData(filter)

            for _ in 1:1000
                u = Point2f(rand(Float32), rand(Float32))
                fs = Hikari.filter_sample_tabulated(data, u)

                @test -radius <= fs.p[1] <= radius
                @test -radius <= fs.p[2] <= radius
            end
        end
    end

    @testset "PDF integrates to 1" begin
        # The PDF over the domain should integrate to 1
        for (radius, sigma) in [(1.5f0, 0.5f0), (2.0f0, 0.5f0)]
            filter = Hikari.GaussianFilter(Point2f(radius, radius), sigma)
            data = Hikari.GPUFilterSamplerData(filter)

            # Numerical integration of PDF = f(x,y) / func_integral
            n = 100
            r = radius
            xs = range(-r, r, n)
            ys = range(-r, r, n)
            dx = xs[2] - xs[1]
            dy = ys[2] - ys[1]

            pdf_integral = 0.0
            for (ix, x) in enumerate(xs), (iy, y) in enumerate(ys)
                # Get the tabulated function value at this cell
                # Map (x,y) to cell indices
                tx = (x + r) / (2r)
                ty = (y + r) / (2r)
                cell_x = clamp(Int(floor(tx * data.nx)) + 1, 1, Int(data.nx))
                cell_y = clamp(Int(floor(ty * data.ny)) + 1, 1, Int(data.ny))
                f_val = data.func[cell_y, cell_x]
                pdf = f_val / data.func_integral
                pdf_integral += pdf * dx * dy
            end

            @test isapprox(pdf_integral, 1.0, rtol=0.05) ||
                  @show (radius, sigma, pdf_integral)
        end
    end

    @testset "Large radius filters" begin
        # Test with large filter radii to stress test the sampling
        for radius in [5.0f0, 10.0f0]
            sigma = radius / 3
            filter = Hikari.GaussianFilter(Point2f(radius, radius), sigma)
            data = Hikari.GPUFilterSamplerData(filter)

            # Weights should still be constant
            weights = [Hikari.filter_sample_tabulated(data, Point2f(rand(), rand())).weight
                       for _ in 1:1000]

            weight_variance = var(weights)
            @test weight_variance < 1e-6 * data.func_integral^2 ||
                  @show (radius, sigma, weight_variance, data.func_integral)
        end
    end

    @testset "Box filter weight = 1" begin
        # Box filter uses analytical sampling with weight = 1
        filter = Hikari.BoxFilter(Point2f(0.5f0, 0.5f0))
        params = Hikari.GPUFilterParams(filter)

        for _ in 1:100
            u = Point2f(rand(Float32), rand(Float32))
            fs = Hikari.filter_sample(params, nothing, u)
            @test fs.weight == 1.0f0
            @test -0.5f0 <= fs.p[1] <= 0.5f0
            @test -0.5f0 <= fs.p[2] <= 0.5f0
        end
    end

    @testset "Triangle filter weight = 1" begin
        # Triangle filter uses analytical tent sampling with weight = 1
        filter = Hikari.TriangleFilter(Point2f(1.0f0, 1.0f0))
        params = Hikari.GPUFilterParams(filter)

        for _ in 1:100
            u = Point2f(rand(Float32), rand(Float32))
            fs = Hikari.filter_sample(params, nothing, u)
            @test fs.weight == 1.0f0
            @test -1.0f0 <= fs.p[1] <= 1.0f0
            @test -1.0f0 <= fs.p[2] <= 1.0f0
        end
    end

    @testset "Mitchell filter" begin
        filter = Hikari.MitchellFilter(Point2f(2.0f0, 2.0f0), 1/3, 1/3)
        data = Hikari.GPUFilterSamplerData(filter)

        # Weights should be constant
        weights = [Hikari.filter_sample_tabulated(data, Point2f(rand(), rand())).weight
                   for _ in 1:1000]

        @test isapprox(mean(weights), data.func_integral, rtol=0.001)
        @test var(weights) < 1e-6 * data.func_integral^2
    end

end
