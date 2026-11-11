module ZeroWeightSpecularProbe
using Hikari, Mantle, JSON3, Statistics, Test
using ..MaterialComparison
const Probe=MaterialComparison

function rewrite(x)
    x isa Expr || return x
    if x.head==:(=) && x.args[1]==:l_specular
        branch=deepcopy(x.args[2])
        @assert branch.head==:if
        branch.args[1]=:($(branch.args[1]) && w!=Vec3f(0f0))
        return Expr(:(=),:l_specular,branch)
    end
    Expr(x.head,(rewrite(a) for a in x.args)...)
end

function run(output;device,repeats=31,names=nothing)
    Mantle.defaultdevice!(device)
    original=Main.Revise.CodeTracking.definition(Expr,only(methods(Hikari.principled_specular_layer)))
    original isa Expr || error("Cannot recover Principled specular layer")
    replacement=rewrite(original)
    spec=Probe.configuration()
    records=Any[]
    for row in spec.cases
        names===nothing || row.name in names || continue
        baseline=Probe.fixture(Probe.principled_material(row),spec)
        candidate=nothing
        try
            Base.invokelatest(Probe.draw,baseline)
            expected=Probe.pixels(baseline)
            Core.eval(Hikari,replacement)
            candidate=Base.invokelatest(Probe.fixture,Probe.principled_material(row),spec)
            first_seconds=@elapsed Base.invokelatest(Probe.draw,candidate)
            actual=Probe.pixels(candidate)
            @test all(isfinite,actual)
            @test actual≈expected rtol=5f-5 atol=2f-5
            @test Probe.shade_signatures(candidate)==Probe.shade_signatures(baseline)
            times=(Float64[],Float64[])
            for _ in 1:5,f in (baseline,candidate)
                Base.invokelatest(Probe.draw,f)
            end
            for i in 1:repeats,k in (isodd(i) ? (1,2) : (2,1))
                push!(times[k],@elapsed Base.invokelatest(Probe.draw,k==1 ? baseline : candidate))
            end
            @test Probe.pixels(candidate)≈actual rtol=5f-5 atol=2f-5
            result=(;name=row.name,first_seconds,frame_seconds=times,median_seconds=map(median,times),
                     max_pixel_error=maximum(abs.(actual.-expected)))
            push!(records,result)
            mkpath(dirname(output));open(io->JSON3.write(io,records),output,"w")
            println(row.name," specular setup/zero guard ms: ",result.median_seconds.*1000,
                    "; maximum pixel error ",result.max_pixel_error)
            flush(stdout)
        finally
            Core.eval(Hikari,original)
            candidate===nothing || Probe.release(candidate)
            Probe.release(baseline)
        end
    end
    nothing
end
end
