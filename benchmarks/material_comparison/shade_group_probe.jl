module ShadeGroupProbe
using Hikari, Mantle, JSON3, Test, Statistics
using ..MaterialComparison
const Probe = MaterialComparison

function run(output; device, resolution = 1024, groups = (64, 32, 128), repeats = 31,
             names = ("matte", "rough_metal"))
    Mantle.defaultdevice!(device)
    original = Main.Revise.CodeTracking.definition(Expr, only(methods(Hikari.shader_bsdf_workgroupsize)))
    original isa Expr || error("Cannot recover shade group definition")
    results = Any[]
    for row in Probe.configuration().cases
        row.name in names || continue
        fixtures = Any[]
        try
            for group in groups
                Core.eval(Hikari, :(shader_bsdf_workgroupsize(group) = min(group, $group)))
                f = Base.invokelatest(Probe.fixture, Probe.principled_material(row), Probe.configuration(); resolution)
                push!(fixtures, f)
                Base.invokelatest(Probe.draw, f)
            end
            expected = Probe.pixels(first(fixtures))
            for f in fixtures
                @test Probe.pixels(f) ≈ expected rtol = 5f-5 atol = 2f-5
            end
            for _ in 1:5, f in fixtures
                Base.invokelatest(Probe.draw, f)
            end
            times = [Float64[] for _ in fixtures]
            for i in 1:repeats, k in (isodd(i) ? eachindex(fixtures) : reverse(eachindex(fixtures)))
                push!(times[k], @elapsed Base.invokelatest(Probe.draw, fixtures[k]))
            end
            for f in fixtures
                @test Probe.pixels(f) ≈ expected rtol = 5f-5 atol = 2f-5
            end
            result = (; name = row.name, resolution, groups, median_seconds = map(median, times), frame_seconds = times)
            push!(results, result)
            mkpath(dirname(output))
            open(io -> JSON3.write(io, results), output, "w")
            println(row.name, " shade groups ", groups, " ms: ", result.median_seconds .* 1000)
            flush(stdout)
        finally
            Core.eval(Hikari, original)
            foreach(Probe.release, fixtures)
        end
    end
    nothing
end
end
