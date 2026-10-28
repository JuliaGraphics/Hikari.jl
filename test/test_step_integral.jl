using Test, Hikari

@testset "Emission quadrature is stable in vacuum and thin gas" begin
    @test !Hikari.integrates_emission(Hikari.HomogeneousMedium())
    for sigma in Float32[0, 1e-10, 1e-6, 1e-3, 0.1, 1, 100], h in Float32[0, 1e-3, 0.5, 2]
        coefficients = (sigma, 2sigma, sigma / 2, 0f0)
        spectrum = Hikari.SpectralRadiance(coefficients...)
        integral, transmittance = Hikari.step_integral(spectrum, h)
        half, half_transmittance = Hikari.step_integral(spectrum, h / 2)
        for i in 1:4
            s = Float64(coefficients[i])
            reference = s == 0 ? Float64(h) : -expm1(-s * Float64(h)) / s
            @test isfinite(integral[i]) && integral[i] >= 0
            @test integral[i] ≈ reference rtol=4e-5 atol=1e-12
            @test transmittance[i] ≈ exp(-s * Float64(h)) rtol=2e-6 atol=1e-7
            @test integral[i] ≈ half[i] * (1 + half_transmittance[i]) rtol=5e-5 atol=1e-12
        end
    end
end
