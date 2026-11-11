module DispatchGroupProbe
using Hikari, Mantle, JSON3, Test, Statistics
using ..MaterialComparison
const Probe=MaterialComparison

function rewrite(ex,group)
    ex isa Expr || return ex
    ex.head===:kw && ex.args[1]===:group && return Expr(:kw,:group,group)
    ex.args=[rewrite(a,group) for a in ex.args]
    return ex
end

function run(output;groups=(32,64,128,256),repeats=11)
    Mantle.defaultdevice!("RTX 4000 Ada")
    spec=Probe.configuration()
    original=Main.Revise.CodeTracking.definition(Expr,only(methods(Hikari.trace_pass!)))
    original isa Expr || error("Cannot recover trace launch")
    rows=Any[]
    for name in ("matte","layered_principled")
        row=only(filter(r->r.name==name,spec.cases))
        fixtures=Any[];first_seconds=Float64[]
        try
            for group in groups
                Core.eval(Hikari,rewrite(deepcopy(original),group))
                f=Base.invokelatest(Probe.fixture,Probe.principled_material(row),spec)
                push!(fixtures,f)
                push!(first_seconds,@elapsed Base.invokelatest(Probe.draw,f))
            end
            Core.eval(Hikari,original)
            expected=Probe.pixels(last(fixtures))
            errors=Float64[]
            for f in fixtures
                actual=Probe.pixels(f)
                @test actual≈expected rtol=5f-5 atol=2f-5
                push!(errors,maximum(abs.(actual.-expected)))
            end
            for _ in 1:5,f in fixtures
                Base.invokelatest(Probe.draw,f)
            end
            times=[Float64[] for _ in fixtures]
            for i in 1:repeats,k in (isodd(i) ? eachindex(fixtures) : reverse(eachindex(fixtures)))
                push!(times[k],@elapsed Base.invokelatest(Probe.draw,fixtures[k]))
            end
            record=(;material=name,groups,first_seconds,median_seconds=map(median,times),frame_seconds=times,
                    max_pixel_errors=errors)
            push!(rows,record)
            mkpath(dirname(output));open(io->JSON3.write(io,rows),output,"w")
            println(name," trace groups ",groups,": ",record.median_seconds.*1000," ms");flush(stdout)
        finally
            Core.eval(Hikari,original)
            foreach(Probe.release,fixtures)
        end
    end
    nothing
end
end
