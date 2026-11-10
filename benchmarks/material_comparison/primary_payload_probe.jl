module PrimaryPayloadProbe
using Hikari, Mantle, JSON3, Statistics, Test
using ..MaterialComparison
const Probe=MaterialComparison

# Diagnostic encoding only: a sentinel depth identifies the partially stored
# primary item. A production layout would tag the hit explicitly instead.
@inline function load_work(rays,path)
    @inbounds if rays.depth[path]==Int32(-1)
        return Hikari.VPRayWorkItem(rays.ray[path],Int32(0),rays.lambda[path],rays.pixel_index[path],
            rays.beta[path],Hikari.SpectralRadiance(1f0),Hikari.SpectralRadiance(1f0),
            Hikari.Point3f(0f0),Hikari.Vec3f(0,0,1),1f0,false,false,rays.medium_idx[path])
    end
    @inbounds return rays[path]
end

function rewrite(x)
    x isa Expr || return x
    if x.head==:(=) && x.args[1]==:work
        @assert x.args[2] isa Expr && x.args[2].head==:if
        return :(work=sh.path>Int32(0) ? $load_work(cur.items,sh.path) : arrivals.items[-sh.path])
    end
    Expr(x.head,(rewrite(a) for a in x.args)...)
end

function with_payload(f)
    method=only(filter(m->occursin("PrimaryRayStore",string(m.sig)),collect(methods(Hikari.publish_traced_work!))))
    store=Main.Revise.CodeTracking.definition(Expr,method)
    expand=Main.Revise.CodeTracking.definition(Expr,only(methods(Hikari.expanded_hit)))
    all(x->x isa Expr,(store,expand)) || error("Cannot recover primary payload definitions")
    try
        Core.eval(Hikari,quote
            @inline function publish_traced_work!(rays::PrimaryRayStore,path,work,tinted)
                @inbounds begin
                    rays.items.ray[path]=work.ray
                    rays.items.depth[path]=Int32(-1)
                    rays.items.lambda[path]=work.lambda
                    rays.items.pixel_index[path]=work.pixel_index
                    rays.items.beta[path]=work.beta
                    rays.items.medium_idx[path]=work.medium_idx
                end
                nothing
            end
        end)
        Core.eval(Hikari,rewrite(expand))
        Base.invokelatest(f)
    finally
        Core.eval(Hikari,expand)
        Core.eval(Hikari,store)
    end
end

function run(output;device,repeats=31,names=("matte","rough_glass","rough_metal","plastic","layered_principled"))
    :primary in fieldnames(Hikari.VPSurfaceHit) && error(
        "This historical control requires source before compact primary payloads were integrated.")
    Mantle.defaultdevice!(device)
    spec=Probe.configuration();results=Any[]
    for row in spec.cases
        row.name in names || continue
        baseline=Probe.fixture(Probe.principled_material(row),spec)
        candidate=nothing
        try
            Base.invokelatest(Probe.draw,baseline)
            expected=Probe.pixels(baseline)
            with_payload() do
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
                push!(results,result)
                mkpath(dirname(output));open(io->JSON3.write(io,results),output,"w")
                println(row.name," full/compact primary payload ms: ",result.median_seconds.*1000,
                        "; maximum pixel error ",result.max_pixel_error)
                flush(stdout)
            end
        finally
            candidate===nothing || Probe.release(candidate)
            Probe.release(baseline)
        end
    end
    nothing
end
end
