module SubmissionProbe
using Hikari, Mantle, JSON3, Statistics, Test
using ..MaterialComparison
const Probe = MaterialComparison

# Same existing render! API and sample count, without the convenience call's
# device-wide wait after every wave. All work is still synchronized at the end.
function batched(f)
    f.film.iteration_index[] = Int32(0)
    Hikari.clear!(f.vp)
    Hikari.render!(f.vp,f.scene,f.film,f.camera; samples=Int(f.vp.samples_per_pixel))
    Hikari.postprocess!(f.film)
    Mantle.waitidle(f.dev)
    return nothing
end

function run(output; repeats=11)
    Mantle.defaultdevice!("RTX 4000 Ada")
    spec=Probe.configuration()
    result=Any[]
    for name in ("matte","layered_principled"), implementation in (:typed_cycles,:current_cycles)
        row=only(filter(r->r.name==name,spec.cases))
        f=Probe.fixture(Probe.principled_material(row),spec)
        try
            Base.invokelatest(Probe.first_draw,f,implementation)
            expected=Probe.pixels(f)
            Base.invokelatest(batched,f)
            actual=Probe.pixels(f)
            @test actual≈expected rtol=5f-5 atol=2f-5
            @test f.film.iteration_index[]==spec.samples
            for _ in 1:3, draw in (Probe.draw,batched)
                Base.invokelatest(draw,f)
            end
            times=(Float64[],Float64[])
            submits=(Int[],Int[])
            host=(Float64[],Float64[])
            for i in 1:repeats, k in (isodd(i) ? (1,2) : (2,1))
                before=Mantle.submissions(f.dev)
                timing=@timed Base.invokelatest(k==1 ? Probe.draw : batched,f)
                push!(times[k],timing.time)
                push!(submits[k],Int(Mantle.submissions(f.dev)-before))
                push!(host[k],timing.compile_time)
            end
            record=(; material=name,implementation=string(implementation),
                     samples=Int(spec.samples),lanes=Int(f.vp.state.sobol_rng.lanes),
                     median_seconds=map(median,times),frame_seconds=times,
                     submissions=submits,jit_seconds=host,max_pixel_error=maximum(abs.(actual.-expected)))
            push!(result,record)
            mkpath(dirname(output))
            open(io->JSON3.write(io,result),output,"w")
            println(name," / ",implementation,": normal/batched ",record.median_seconds.*1000," ms; submissions ",map(median,submits))
            flush(stdout)
        finally
            Probe.release(f)
        end
    end
    nothing
end
end
