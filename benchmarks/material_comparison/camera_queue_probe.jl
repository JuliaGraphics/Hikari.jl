module CameraQueueProbe
using Hikari, Mantle, Adapt, JSON3, Test, Statistics
using ..MaterialComparison
const Probe=MaterialComparison

struct DirectCameraQueue{Q}
    queue::Q
    count::Int32
end
Adapt.adapt_structure(to,q::DirectCameraQueue)=DirectCameraQueue(Adapt.adapt(to,q.queue),q.count)

# The fixture's perspective camera emits exactly one ray per sample. Its sample
# index is already a unique queue slot; consumers run after the graph barrier.
@inline function Base.push!(q::DirectCameraQueue,item)
    index=item.pixel_index
    @inbounds q.queue.items[index]=item
    if index==1
        @inbounds q.queue.size[1]=q.count
    end
    return index
end

function rewrite(ex)
    ex isa Expr || return ex
    if ex.head===:call
        at=findfirst(==(:vp_generate_camera_rays_kernel!),ex.args)
        if at!==nothing
            args=ex.args[at+1]
            @assert args isa Expr && args.head===:tuple && args.args[1]===:cur
            args.args[1]=:($DirectCameraQueue(cur,Int32(state.sobol_rng.pixels*state.sobol_rng.lanes)))
            return ex
        end
    end
    ex.args=map(rewrite,ex.args)
    return ex
end

function run(output; repeats=11)
    isdefined(Hikari, :vp_generate_primary_rays_kernel!) && error(
        "This historical camera-queue probe requires pre-primary-fusion source.")
    Mantle.defaultdevice!("RTX 4000 Ada")
    spec=Probe.configuration()
    original=Main.Revise.CodeTracking.definition(Expr,only(methods(Hikari.setup_passes!)))
    original isa Expr || error("Cannot recover camera setup for restoration")
    candidate_definition=rewrite(deepcopy(original))
    records=Any[]
    for name in ("matte","layered_principled")
        row=only(filter(r->r.name==name,spec.cases))
        baseline=Probe.fixture(Probe.principled_material(row),spec)
        candidate=nothing
        try
            Base.invokelatest(Probe.draw,baseline)
            expected=Probe.pixels(baseline)
            Core.eval(Hikari,candidate_definition)
            candidate=Base.invokelatest(Probe.fixture,Probe.principled_material(row),spec)
            first_seconds=@elapsed Base.invokelatest(Probe.draw,candidate)
            actual=Probe.pixels(candidate)
            @test actual≈expected rtol=5f-5 atol=2f-5
            times=(Float64[],Float64[])
            for _ in 1:5, f in (baseline,candidate)
                Base.invokelatest(Probe.draw,f)
            end
            for i in 1:repeats,k in (isodd(i) ? (1,2) : (2,1))
                push!(times[k],@elapsed Base.invokelatest(Probe.draw,k==1 ? baseline : candidate))
            end
            record=(;material=name,first_seconds,median_seconds=map(median,times),frame_seconds=times,
                    max_pixel_error=maximum(abs.(actual.-expected)))
            push!(records,record)
            mkpath(dirname(output))
            open(io->JSON3.write(io,records),output,"w")
            println(name," atomic/direct camera queue: ",record.median_seconds.*1000," ms")
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
