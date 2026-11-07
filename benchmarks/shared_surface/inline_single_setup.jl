# Diagnostic: retain shared runtime storage and inline only its one-record setup.
# Load renderer.jl first. A temporary method is removed in the finally block.
module InlineOneSetupProbe
using Hikari, Mantle, Test, Statistics, JSON3
using Main.SurfaceProgramGPUProbe: fixture, draw, draw_reference, image, release

function resources(f)
    launches = unique([d.launch for pl in Hikari.allplans(f.vp.state.plans)
        for pp in pl.passes if pp.pass.name == "shade" for d in pp.dispatches])
    return [(; identity=objectid(l.pipeline), stats=Mantle.pipeline_exec_stats(f.dev,l.pipeline)) for l in launches]
end

function run(material=Hikari.Metallic(); pairs=51, result_path=nothing)
    reference=fixture(material;resolution=512,samples=8)
    baseline=fixture(material;resolution=512,samples=8)
    candidate=nothing
    added=nothing
    try
        draw_reference(reference)
        Base.invokelatest(draw,baseline)
        expected=image(reference)
        baseimage=image(baseline)
        @test baseimage ≈ expected rtol=5f-5 atol=2f-5
        before=resources(baseline)
        @assert !any(m -> Base.unwrap_unionall(m.sig).parameters[2] === Val{1}, methods(Hikari.surface_setup_slots))
        @eval Hikari Base.@constprop :none @inline function surface_setup_slots(
            ::Val{1}, s, prog, i, sc, lut, table, lambda, scale,
            mode::Val{Looped}, mask::Val{Mask}) where {Looped,Mask}
            storage=MVector{1,SurfaceStoredWeight{Looped,Mask}}(surface_empty(Val(1),mode,mask))
            at=Int32(1); emission=Vec3f(0)
            @inbounds while prog[i].op != NodeOp.RESULT
                i,prefix=surface_zero_prefix(s,prog,i,sc)
                emission+=prefix
                prog[i].op==NodeOp.RESULT && break
                i,at,e=surface_store_dispatch!(storage,at,s,prog,i,sc,lut,table,lambda,scale,mode,mask)
                emission+=e
            end
            return Tuple(storage),emission,at-Int32(1)
        end
        added=only(m for m in methods(Hikari.surface_setup_slots)
            if Base.unwrap_unionall(m.sig).parameters[2] === Val{1})
        candidate=fixture(material;resolution=512,samples=8)
        first=@elapsed Base.invokelatest(draw,candidate)
        actual=image(candidate)
        @test all(isfinite,actual)
        @test maximum(expected)>0
        @test actual ≈ expected rtol=5f-5 atol=2f-5
        fs=(reference,baseline,candidate)
        times=(Float64[],Float64[],Float64[])
        for _ in 1:5, f in fs
            Base.invokelatest(draw,f)
        end
        for j in 1:pairs, k in (isodd(j) ? (1,2,3) : (3,2,1))
            push!(times[k],@elapsed Base.invokelatest(draw,fs[k]))
        end
        @test resources(baseline)==before
        row=(;material=string(nameof(typeof(material))), first, max_error=maximum(abs.(actual.-expected)),
            reference=median(times[1]),baseline=median(times[2]),candidate=median(times[3]),
            before,after=resources(candidate),reference_resources=resources(reference))
        println(JSON3.write(row))
        if result_path !== nothing
            mkpath(dirname(result_path))
            write(result_path, JSON3.write(row))
        end
        return row
    finally
        added===nothing || Base.delete_method(added)
        candidate===nothing || release(candidate)
        release(baseline);release(reference)
    end
end
end
