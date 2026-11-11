module PassProbe
using Hikari, Mantle, JSON3, Statistics, Test
using ..MaterialComparison
const Probe = MaterialComparison

# Diagnostic only: expand the recorded bounce loop so that every bounce has
# distinct timestamp slots. Otherwise the loop's last iteration overwrites
# earlier measurements. Empty queues still size their dispatches to zero.
const unrolled = quote
    function sample_graph(dev, state::VolPathState, @nospecialize(refs), d::Int;
                          has_media::Bool, has_lights::Bool)
        g = Mantle.Graph(dev)
        Mantle.dispatch!(g, vp_next_sample_kernel!, (refs.sample_idx, state.sobol_rng.lanes), 1;
                         group=1, name="next sample")
        if d == 0
            setup_passes!(g, state, refs, state.ray_queue_a)
        else
            primary_passes!(g, state, refs; has_media, has_lights)
        end
        cur,nxt = state.ray_queue_b,state.ray_queue_a
        for _ in 2:d
            round_passes!(g,state,refs,cur,nxt;has_media,has_lights)
            cur,nxt = nxt,cur
        end
        accumulate_passes!(g, state, refs)
        g
    end
end

function run(output; repeats=5, hw_accel=true, select="RTX 4000 Ada",
             names=("matte","layered_principled"), implementations=(:typed_cycles,:current_cycles),
             spec=Probe.configuration())
    Mantle.defaultdevice!(select)
    method=only(methods(Hikari.sample_graph))
    original=Main.Revise.CodeTracking.definition(Expr,method)
    original === nothing && error("Cannot recover sample_graph for restoration")
    records=Any[]
    for name in names, implementation in implementations
        row=only(filter(r->r.name==name,spec.cases))
        reference=Probe.fixture(Probe.principled_material(row),spec; hw_accel)
        expected=try
            Base.invokelatest(Probe.first_draw,reference,implementation)
            Probe.pixels(reference)
        finally
            Probe.release(reference)
        end
        Core.eval(Hikari,unrolled)
        try
            f=Base.invokelatest(Probe.fixture,Probe.principled_material(row),spec; hw_accel)
            try
                Base.invokelatest(Probe.first_draw,f,implementation)
                plans=Hikari.allplans(f.vp.state.plans)
                @test all(isempty(p.loops) for p in plans)
                for p in plans
                    p.profiler === nothing && error("No timestamp profiler")
                    p.profile=true
                    Mantle.invalidate!(p)
                    foreach(empty!,p.profiler.gpu_ns)
                end
                times=Float64[]
                for _ in 1:repeats
                    push!(times,@elapsed Base.invokelatest(Probe.draw,f))
                end
                actual=Probe.pixels(f)
                @test actual≈expected rtol=5f-5 atol=2f-5
                passrecords=Any[]
                for (planid,p) in enumerate(plans)
                    ts=Mantle.timings(p)
                    for (i,t) in enumerate(ts)
                        push!(passrecords,(;plan=planid,index=i,name=t.name,kind=string(t.kind),
                                           gpu_ms=t.gpu_ms,samples=t.samples))
                    end
                end
                record=(;material=name,implementation=string(implementation),hw_accel,
                        frame_seconds=times,max_pixel_error=maximum(abs.(actual.-expected)),
                        lanes=Int(f.vp.state.sobol_rng.lanes),samples=Int(spec.samples),passes=passrecords)
                push!(records,record)
                mkpath(dirname(output))
                open(io->JSON3.write(io,records),output,"w")
                println(name," / ",implementation,": instrumented ",median(times)*1000," ms")
                groups=Dict{String,Float64}()
                for t in passrecords
                    isfinite(t.gpu_ms) || continue
                    groups[t.name]=get(groups,t.name,0.0)+t.gpu_ms
                end
                println(sort!(collect(groups);by=last,rev=true))
                flush(stdout)
            finally
                Probe.release(f)
            end
        finally
            Core.eval(Hikari,original)
        end
    end
    nothing
end
end
