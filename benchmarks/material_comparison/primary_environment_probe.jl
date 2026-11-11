module PrimaryEnvironmentProbe
using Hikari, Mantle, JSON3, Test, Statistics
using ..MaterialComparison
const Probe = MaterialComparison

function inline_misses(ex, count)
    ex isa Expr || return ex
    if ex.head === :call && length(ex.args) == 3 && ex.args[1] === :push! && ex.args[2] === :escaped_queue
        count[] += 1
        return quote
            if rays isa PrimaryRayStore && work.depth == Int32(0) && !windowed(work.prev_intr_n)
                primary_Le = evaluate_escaped_ray_spectral(rgb2spec_table, lights, work.ray.d, work.lambda)
                primary_contribution = work.beta * primary_Le
                if !is_black(primary_contribution)
                    accumulate_spectrum!(pixel_L, (work.pixel_index - Int32(1)) * Int32(4),
                                         primary_contribution / average(work.r_u))
                end
            else
                $ex
            end
        end
    end
    ex.args = map(a -> inline_misses(a, count), ex.args)
    return ex
end

function run(output; spec, repeats = 31, names = ("matte", "rough_glass", "rough_metal", "plastic", "layered_principled"))
    original = Main.Revise.CodeTracking.definition(Expr, only(methods(Hikari.vp_trace!)))
    original isa Expr || error("Cannot recover trace method")
    count = Ref(0)
    candidate_definition = inline_misses(deepcopy(original), count)
    count[] == 1 || error("This historical control requires the pre-inline surface escape queue.")
    records = Any[]
    for name in names
        row = only(filter(r -> r.name == name, spec.cases))
        baseline = Probe.fixture(Probe.principled_material(row, spec), spec)
        candidate = nothing
        try
            Base.invokelatest(Probe.draw, baseline)
            expected = Probe.pixels(baseline)
            Core.eval(Hikari, candidate_definition)
            candidate = Base.invokelatest(Probe.fixture, Probe.principled_material(row, spec), spec)
            first_seconds = @elapsed Base.invokelatest(Probe.draw, candidate)
            actual = Probe.pixels(candidate)
            @test actual ≈ expected rtol = 5f-5 atol = 2f-5
            times = (Float64[], Float64[])
            for _ in 1:5, f in (baseline, candidate)
                Base.invokelatest(Probe.draw, f)
            end
            for i in 1:repeats, k in (isodd(i) ? (1,2) : (2,1))
                push!(times[k], @elapsed Base.invokelatest(Probe.draw, k == 1 ? baseline : candidate))
            end
            @test Probe.pixels(candidate) ≈ expected rtol = 5f-5 atol = 2f-5
            record = (; material = name, first_seconds, median_seconds = map(median, times), frame_seconds = times,
                        max_pixel_error = maximum(abs.(actual .- expected)))
            push!(records, record)
            mkpath(dirname(output))
            open(io -> JSON3.write(io, records), output, "w")
            println(name, " queued/inline primary misses: ", record.median_seconds .* 1000, " ms")
            flush(stdout)
        finally
            Core.eval(Hikari, original)
            candidate === nothing || Probe.release(candidate)
            Probe.release(baseline)
        end
    end
    return nothing
end
end
