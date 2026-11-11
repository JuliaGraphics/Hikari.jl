module PrimaryIndexProbe
using Hikari, Mantle, JSON3, Statistics, Test
using ..MaterialComparison
const Probe = MaterialComparison

function rewrite(x)
    x isa Expr || return x
    x.head == :(=) && x.args[1] == :(rays.items.pixel_index[path]) && return :(nothing)
    x == :(rays.pixel_index[path]) && return :path
    Expr(x.head, (rewrite(a) for a in x.args)...)
end

function run(output; device, repeats = 61, names = ("matte", "rough_metal"))
    occursin("rays.pixel_index", string(Main.Revise.CodeTracking.definition(Expr,
        only(methods(Hikari.primary_work))))) || error("This historical control requires stored primary indices.")
    Mantle.defaultdevice!(device)
    store_method = only(filter(m -> occursin("PrimaryRayStore", string(m.sig)), methods(Hikari.publish_traced_work!)))
    originals = [Main.Revise.CodeTracking.definition(Expr, m) for m in
                 (store_method, only(methods(Hikari.primary_work)))]
    all(x -> x isa Expr, originals) || error("Cannot recover primary path definitions")
    records = Any[]
    for row in Probe.configuration().cases
        row.name in names || continue
        baseline = Probe.fixture(Probe.principled_material(row), Probe.configuration())
        candidate = nothing
        try
            Base.invokelatest(Probe.draw, baseline)
            expected = Probe.pixels(baseline)
            foreach(x -> Core.eval(Hikari, rewrite(x)), originals)
            candidate = Base.invokelatest(Probe.fixture, Probe.principled_material(row), Probe.configuration())
            Base.invokelatest(Probe.draw, candidate)
            @test Probe.pixels(candidate) == expected
            times = (Float64[], Float64[])
            for _ in 1:5, f in (baseline, candidate)
                Base.invokelatest(Probe.draw, f)
            end
            for i in 1:repeats, k in (isodd(i) ? (1, 2) : (2, 1))
                push!(times[k], @elapsed Base.invokelatest(Probe.draw, k == 1 ? baseline : candidate))
            end
            @test Probe.pixels(candidate) == expected
            record = (; name = row.name, median_seconds = map(median, times), frame_seconds = times)
            push!(records, record)
            mkpath(dirname(output))
            open(io -> JSON3.write(io, records), output, "w")
            println(row.name, " stored/implicit primary index ms: ", record.median_seconds .* 1000)
            flush(stdout)
        finally
            foreach(x -> Core.eval(Hikari, x), originals)
            candidate === nothing || Probe.release(candidate)
            Probe.release(baseline)
        end
    end
    nothing
end
end
