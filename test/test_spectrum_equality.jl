using Test
using Hikari: RGBSpectrum, Spectrum

@testset "Spectrum equality" begin
    one = RGBSpectrum(1f0)
    zero_rgb = RGBSpectrum(0f0)
    @test one == RGBSpectrum(1f0, 1f0, 1f0, 1f0)
    @test one != RGBSpectrum(1f0, 1f0, 1f0, 0f0)
    @test one == 1
    @test one != 0
    @test zero_rgb != 0 # Alpha remains part of equality.
    @test RGBSpectrum(0f0, 0f0, 0f0, 0f0) == 0
    @test one == ones(Float32, 4)
    @test one == (1f0, 1f0, 1f0, 1f0)
    @test one != (1f0, 1f0, 1f0, 0f0)
    @test (one == nothing) === false
    @test (one == :unrelated) === false
    @test (one == missing) === missing

    # Loading Hikari must not add a candidate to arbitrary `==(Any, T)` calls.
    @test which(==, Tuple{Spectrum, Task}) === which(==, Tuple{Any, Any})
end
