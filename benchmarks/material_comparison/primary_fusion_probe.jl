module PrimaryFusionProbe
using Hikari, Mantle, JSON3, Test, Statistics
using ..MaterialComparison
const Probe=MaterialComparison

struct DeferredRayStore{A}
    items::A
end
@inline function publish!(rays,path,work,tinted)
    tinted && (@inbounds rays.beta[path]=work.beta)
    nothing
end
@inline function publish!(rays::DeferredRayStore,path,work,tinted)
    @inbounds rays.items[path]=work
    nothing
end

function replace_publish(ex)
    ex isa Expr || return ex
    if ex.head===:&& && occursin("rays.beta[path]",string(ex))
        return :($publish!(rays,path,work,tinted))
    end
    ex.args=map(replace_publish,ex.args)
    ex
end

function camera_function(ex)
    ex isa Expr || return nothing
    if ex.head===:function && ex.args[1] isa Expr && ex.args[1].head===:call &&
       ex.args[1].args[1]===:vp_generate_camera_rays_kernel!
        return ex
    end
    for a in ex.args
        found=camera_function(a)
        found===nothing || return found
    end
    nothing
end

function replace_camera_push(ex)
    ex isa Expr || return ex
    if ex.head===:call && ex.args==[:push!,:ray_queue,:work_item]
        return :(vp_trace!(work_item,Int32(idx),$DeferredRayStore(ray_queue.items),Base.tail(trace_args)...))
    end
    ex.args=map(replace_camera_push,ex.args)
    ex
end

function install_kernel!()
    isdefined(Hikari, :vp_generate_primary_rays_kernel!) && error(
        "This historical prototype requires the pre-integration camera kernel. Use test_primary_ray_fusion.jl on current source.")
    # Parse only to derive this diagnostic kernel, not to execute package source.
    parsed=Meta.parseall(read(joinpath(pkgdir(Hikari),"src/integrators/volpath/volpath.jl"),String))
    f=camera_function(parsed)
    f===nothing && error("Cannot locate camera kernel")
    f=replace_camera_push(deepcopy(f))
    f.args[1].args[1]=:vp_primary_trace_probe_kernel!
    push!(f.args[1].args,:trace_args)
    Core.eval(Hikari,:(@kernel inbounds=true $f))
end

function graph_definition(fused)
    quote
        function sample_graph(dev,state::VolPathState,@nospecialize(refs),d::Int;has_media::Bool,has_lights::Bool)
            a,b=state.ray_queue_a,state.ray_queue_b
            g=Mantle.Graph(dev)
            Mantle.dispatch!(g,vp_next_sample_kernel!,(refs.sample_idx,state.sobol_rng.lanes),1;group=1,name="next sample")
            setup_passes!(g,state,refs,a)
            cur,nxt=a,b
            for round in 1:d
                round_passes!(g,state,refs,cur,nxt;has_media,has_lights)
                if $fused && round==1
                    camera=only(filter(p->p.name=="camera",g.passes))
                    trace=only(filter(p->p.name=="trace",g.passes))
                    cd,td=only(camera.dispatches),only(trace.dispatches)
                    trace.dispatches[1]=Mantle.Dispatch(vp_primary_trace_probe_kernel!,
                        (cd.args...,td.args),cd.ndrange,64)
                    trace.name="primary"
                    deleteat!(g.passes,findfirst(p->p===camera,g.passes))
                end
                cur,nxt=nxt,cur
            end
            accumulate_passes!(g,state,refs)
            g
        end
    end
end

function run(output;resolution=64,samples=4,repeats=11)
    Mantle.defaultdevice!("RTX 4000 Ada")
    install_kernel!()
    graph=Main.Revise.CodeTracking.definition(Expr,only(methods(Hikari.sample_graph)))
    trace=Main.Revise.CodeTracking.definition(Expr,only(methods(Hikari.vp_trace!)))
    graph isa Expr && trace isa Expr || error("Cannot restore primary fusion control")
    spec=Probe.configuration();records=Any[]
    for name in ("matte","layered_principled")
        row=only(filter(r->r.name==name,spec.cases))
        fixtures=Any[];first_seconds=Float64[]
        try
            for mode in (:normal,:unrolled,:fused)
                mode===:normal || Core.eval(Hikari,graph_definition(mode===:fused))
                mode===:fused && Core.eval(Hikari,replace_publish(deepcopy(trace)))
                f=Base.invokelatest(Probe.fixture,Probe.principled_material(row),spec;resolution,samples)
                push!(fixtures,f)
                push!(first_seconds,@elapsed Base.invokelatest(Probe.draw,f))
            end
            expected=Probe.pixels(first(fixtures));errors=Float64[]
            for f in fixtures
                actual=Probe.pixels(f)
                @test all(isfinite,actual)
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
            record=(;material=name,resolution,samples,first_seconds,median_seconds=map(median,times),
                    frame_seconds=times,max_pixel_errors=errors,modes=("normal","unrolled","fused"))
            push!(records,record)
            mkpath(dirname(output));open(io->JSON3.write(io,records),output,"w")
            println(name," normal/unrolled/fused ms: ",record.median_seconds.*1000," errors=",errors);flush(stdout)
        finally
            Core.eval(Hikari,graph)
            Core.eval(Hikari,trace)
            foreach(Probe.release,fixtures)
        end
    end
    nothing
end
end
