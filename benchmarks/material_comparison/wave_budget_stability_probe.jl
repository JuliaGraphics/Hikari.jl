module WaveBudgetStabilityProbe
using Hikari, Mantle, Test, JSON3
using ..MaterialComparison
const Probe = MaterialComparison

function run(output; device)
    # Scale the budget down to a few thousand paths so the feedback can be
    # reproduced without holding several gigabytes of artificial GPU pressure.
    budget = Ref(4096)
    devtype = typeof(device)
    original = Main.Revise.CodeTracking.definition(Expr,
        which(Hikari.wave_path_limit, Tuple{Hikari.VolPath, devtype}))
    Core.eval(Hikari, :(wave_path_limit(dev::$devtype) = $budget[]))
    budget_method = which(Hikari.wave_path_limit, Tuple{devtype})
    records = Any[]
    reference = nothing
    try
        for policy in (:stable, :resampled)
            if policy === :resampled
                Core.eval(Hikari, :(wave_path_limit(vp::VolPath, dev) = wave_path_limit(dev)))
            end
            budget[] = 4096
            f = Probe.fixture(Probe.principled_material(first(Probe.configuration().cases)),
                              Probe.configuration(); resolution = 16)
            try
                first_plan = nothing
                for reported in (4096, 2048, 4096)
                    budget[] = reported
                    Mantle.Lava.reset_compile_stats!()
                    Base.invokelatest(Probe.draw, f)
                    plan = f.vp.state.plans
                    image = Probe.pixels(f)
                    reference === nothing && (reference = image)
                    @test image ≈ reference rtol = 5f-5 atol = 2f-5
                    same = first_plan === nothing || plan === first_plan
                    push!(records, (; policy, reported, lanes = Int(f.vp.state.sobol_rng.lanes),
                        same_plan = same, cache = Mantle.Lava.compile_stats()))
                    if first_plan !== nothing
                        @test same == (policy === :stable)
                    end
                    first_plan === nothing && (first_plan = plan)
                end
            finally
                Probe.release(f)
            end
        end
        mkpath(dirname(output))
        open(io -> JSON3.write(io, records), output, "w")
        foreach(println, records)
    finally
        Core.eval(Hikari, original)
        Base.delete_method(budget_method)
    end
    return nothing
end
end
