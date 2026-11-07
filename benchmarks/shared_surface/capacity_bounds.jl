module CapacityBoundsProbe
using Hikari, Mantle, Test, Statistics, JSON3
using ..SurfaceProgramGPUProbe
const Probe = SurfaceProgramGPUProbe

# Differential control: the former single shared-storage dispatch. A more
# specific temporary method leaves the production method intact when deleted.
function draw_unfiltered(f)
    before = Set(methods(Hikari.shade_queues))
    @eval Hikari shade_queues(q::ClosureQueue{SurfaceProgramNodes{8, L, K, E}}) where {L, K, E} = (q,)
    added = only(setdiff(Set(methods(Hikari.shade_queues)), before))
    try
        return Base.invokelatest(Probe.draw, f)
    finally
        Base.delete_method(added)
    end
end

function compare(material; resolution = 512, samples = 8, timing_samples = 51, hw_accel = true)
    fixtures = Any[]
    first_seconds = Float64[]
    try
        for draw in (Probe.draw_reference, draw_unfiltered, Probe.draw)
            f = Probe.fixture(material; resolution, samples, hw_accel)
            push!(fixtures, f)
            push!(first_seconds, @elapsed Base.invokelatest(draw, f))
        end
        expected = Probe.image(first(fixtures))
        errors = map(fixtures) do f
            actual = Probe.image(f)
            @test all(isfinite, actual)
            @test maximum(actual) > 0f0
            @test actual ≈ expected rtol = 5f-5 atol = 2f-5
            maximum(abs.(actual .- expected))
        end
        signature_counts = map(f -> length(Probe.signatures(f)), fixtures)
        @test signature_counts == [1, 1, 3]
        signatures = map(Probe.signatures, fixtures)
        for _ in 1:5, f in fixtures
            Base.invokelatest(Probe.draw, f)
        end
        times = ntuple(_ -> Float64[], 3)
        for i in 1:timing_samples, k in (isodd(i) ? (1, 2, 3) : (3, 2, 1))
            push!(times[k], @elapsed Base.invokelatest(Probe.draw, fixtures[k]))
        end
        @test map(Probe.signatures, fixtures) == signatures
        return (; resolution, samples, hw_accel, first_seconds, signature_counts,
                 max_errors = errors, medians = map(median, times), frame_seconds = times)
    finally
        foreach(Probe.release, fixtures)
    end
end
end
