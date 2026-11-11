module EscapedQueueProbe
using Hikari, Mantle, JSON3, Test, Statistics
using ..MaterialComparison
const Probe=MaterialComparison

function rewrite(ex)
    ex isa Expr || return ex
    if ex.head===:call && length(ex.args)==3 && ex.args[1]===:push! && ex.args[2]===:escaped_queue
        return :(if num_infinite_lights>0; $ex; end)
    end
    ex.args=map(rewrite,ex.args)
    return ex
end

function run(output;repeats=11)
    Mantle.defaultdevice!("RTX 4000 Ada")
    spec=Probe.configuration()
    original=Main.Revise.CodeTracking.definition(Expr,only(methods(Hikari.vp_trace!)))
    original isa Expr || error("Cannot recover trace method")
    candidate_definition=rewrite(deepcopy(original))
    records=Any[]
    for name in ("matte","layered_principled")
        row=only(filter(r->r.name==name,spec.cases))
        baseline=Probe.fixture(Probe.principled_material(row),spec)
        candidate=nothing
        try
            Base.invokelatest(Probe.draw,baseline)
            @test baseline.vp.state.num_infinite_lights==0
            expected=Probe.pixels(baseline)
            Core.eval(Hikari,candidate_definition)
            candidate=Base.invokelatest(Probe.fixture,Probe.principled_material(row),spec)
            first_seconds=@elapsed Base.invokelatest(Probe.draw,candidate)
            actual=Probe.pixels(candidate)
            @test actual≈expected rtol=5f-5 atol=2f-5
            times=(Float64[],Float64[])
            for _ in 1:5,f in (baseline,candidate)
                Base.invokelatest(Probe.draw,f)
            end
            for i in 1:repeats,k in (isodd(i) ? (1,2) : (2,1))
                push!(times[k],@elapsed Base.invokelatest(Probe.draw,k==1 ? baseline : candidate))
            end
            record=(;material=name,first_seconds,median_seconds=map(median,times),frame_seconds=times,
                    max_pixel_error=maximum(abs.(actual.-expected)))
            push!(records,record)
            mkpath(dirname(output));open(io->JSON3.write(io,records),output,"w")
            println(name," original/skip-unused-escape queue: ",record.median_seconds.*1000," ms");flush(stdout)
        finally
            Core.eval(Hikari,original)
            candidate===nothing || Probe.release(candidate)
            Probe.release(baseline)
        end
    end
    nothing
end
end
