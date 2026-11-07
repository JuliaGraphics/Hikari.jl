using Test
@testset "host synchrotron coefficients (no snapshot dependency)" begin
    @test Hikari.electron_density(Hikari.MagneticEtaNe(0.01),10.,1.,1.,2.5,10.) ≈
        0.01 * 10^2/(8π) * 0.5/(1.5 * Hikari.MEC2 * 10)
    @test Hikari.synchrotron_jν(1e5,10.,2.5,2.3e11,10.,1e5) ≈ 1.212181764287848e-18 rtol=1e-6
    a=Hikari.synchrotron_αν(1e5,10.,2.5,2.3e11,10.,1e5)
    @test isfinite(a) && a>0
    @test Hikari.synchrotron_αν(2e5,10.,2.5,2.3e11,10.,1e5) ≈ 2a
    @test Hikari.synchrotron_αν(1e5,20.,2.5,2.3e11,10.,1e5) ≈ 2^2.25*a
    @test Hikari.synchrotron_αν(1e5,10.,2.5,4.6e11,10.,1e5) ≈ 2^-3.25*a
end
