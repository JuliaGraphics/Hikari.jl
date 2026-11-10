module DeferredWavelengthProbe
using Hikari, Mantle, Raycore, KernelAbstractions, JSON3, Statistics, Test
using ..MaterialComparison
const Probe = MaterialComparison

function camera_definition(x,defer=true)
    x isa Expr || return x
    if x.head==:call && first(x.args)==:generate_camera_work!
        return Expr(:call,:_probe_deferred_camera_work!,x.args[2:end]...)
    elseif defer && x.head==:(=) && first(x.args)==:lambda
        return :(lambda=Wavelengths((0f0,0f0,0f0,0f0),(0f0,0f0,0f0,0f0)))
    elseif defer && x.head==:(=) && first(x.args) isa Expr && first(x.args).head==:ref &&
           first(first(x.args).args) in (:wavelengths_per_pixel,:pdf_per_pixel)
        return :(nothing)
    elseif x.head==:return
        @assert x.args[1]==Expr(:tuple,:work_item,:weight)
        return :(return work_item,weight,wavelength_u)
    end
    Expr(x.head,(camera_definition(a,defer) for a in x.args)...)
end

function trace_definition(x,loop=false)
    x isa Expr || return x
    if x.head==:call && first(x.args)==:vp_trace!
        return Expr(:call,:_probe_trace_known_hit!,x.args[2:end]...,:first_hit)
    elseif x.head==:for
        iteration=deepcopy(x.args[1]);iteration.args[1]=:_probe_attempt
        return Expr(:for,iteration,trace_definition(x.args[2],true))
    elseif loop && x.head==:(=) && first(x.args)==:rec
        return :(rec=_probe_attempt==1 ? first_hit : $(x.args[2]))
    end
    Expr(x.head,(trace_definition(a,loop) for a in x.args)...)
end

function replace_primary(x,defer=true)
    x===:vp_generate_primary_rays_kernel! && return GlobalRef(@__MODULE__,defer ? :primary! : :primary_eager_record!)
    x isa Expr || return x
    Expr(x.head,(replace_primary(a,defer) for a in x.args)...)
end

@inline function wavelengths!(work,u,wavelengths,pdfs,path)
    lambda=Hikari.sample_wavelengths_visible(u)
    base=(path-Int32(1))*Int32(4)
    @inbounds begin
        wavelengths[base+Int32(1)]=lambda.lambda[1]
        wavelengths[base+Int32(2)]=lambda.lambda[2]
        wavelengths[base+Int32(3)]=lambda.lambda[3]
        wavelengths[base+Int32(4)]=lambda.lambda[4]
        pdfs[base+Int32(1)]=lambda.pdf[1]
        pdfs[base+Int32(2)]=lambda.pdf[2]
        pdfs[base+Int32(3)]=lambda.pdf[3]
        pdfs[base+Int32(4)]=lambda.pdf[4]
    end
    Hikari.VPRayWorkItem(work.ray,work.depth,lambda,work.pixel_index,work.beta,work.r_u,work.r_l,
        work.prev_intr_p,work.prev_intr_n,work.eta_scale,work.specular_bounce,
        work.any_non_specular_bounces,work.medium_idx)
end

@inline function trace!(work,u,path,rays,wavelengths,pdfs,args)
    next_queue,escaped,medium_queue,material_queues,hits,pixel_L,accel,interfaces,media,materials,
        lights,table,bvh,trails,num_infinite,num_bvh,max_depth=args
    if Hikari.has_medium(work.medium_idx)
        work=wavelengths!(work,u,wavelengths,pdfs,path)
        Hikari.vp_trace!(work,path,rays,args...)
    else
        rec=Raycore.closest_hit_record(accel,work.ray)
        # A black miss contributes no spectral energy. Its filter weight was
        # already stored; the zero radiance needs no wavelength/PDF work.
        if rec.kind!=Raycore.HIT_MISS || num_infinite>0
            work=wavelengths!(work,u,wavelengths,pdfs,path)
            Hikari._probe_trace_known_hit!(work,path,rays,args...,rec)
        end
    end
    nothing
end

@kernel inbounds=true function primary!(ray_queue,wavelengths,pdfs,filter_weights,
        @Const(height::Int32),@Const(camera),@Const(sample),@Const(medium),@Const(filter),sampler,rng,trace_args)
    idx=@index(Global)
    if idx<=rng.pixels*rng.lanes
        work,weight,u=Hikari._probe_deferred_camera_work!(wavelengths,pdfs,filter_weights,
            height,camera,sample,medium,filter,sampler,rng,Int32(idx))
        weight>0f0 && trace!(work,u,Int32(idx),Hikari.PrimaryRayStore(ray_queue.items),wavelengths,pdfs,trace_args)
    end
end

@kernel inbounds=true function primary_eager_record!(ray_queue,wavelengths,pdfs,filter_weights,
        @Const(height::Int32),@Const(camera),@Const(sample),@Const(medium),@Const(filter),sampler,rng,trace_args)
    idx=@index(Global)
    if idx<=rng.pixels*rng.lanes
        work,weight,u=Hikari._probe_deferred_camera_work!(wavelengths,pdfs,filter_weights,
            height,camera,sample,medium,filter,sampler,rng,Int32(idx))
        if weight>0f0
            rays=Hikari.PrimaryRayStore(ray_queue.items)
            if Hikari.has_medium(work.medium_idx)
                Hikari.vp_trace!(work,Int32(idx),rays,trace_args...)
            else
                rec=Raycore.closest_hit_record(trace_args[7],work.ray)
                Hikari._probe_trace_known_hit!(work,Int32(idx),rays,trace_args...,rec)
            end
        end
    end
end

function with_deferred(f;defer=true)
    camera=Main.Revise.CodeTracking.definition(Expr,only(methods(Hikari.generate_camera_work!)))
    trace=Main.Revise.CodeTracking.definition(Expr,only(methods(Hikari.vp_trace!)))
    graph=Main.Revise.CodeTracking.definition(Expr,only(methods(Hikari.primary_passes!)))
    all(x->x isa Expr,(camera,trace,graph)) || error("Cannot recover primary definitions")
    names=(:_probe_deferred_camera_work!,:_probe_trace_known_hit!)
    old=map(names) do name
        isdefined(Hikari,name) ? Set(methods(Base.invokelatest(getfield,Hikari,name))) : Set{Method}()
    end
    try
        Core.eval(Hikari,camera_definition(camera,defer))
        Core.eval(Hikari,trace_definition(trace))
        Core.eval(Hikari,replace_primary(graph,defer))
        Base.invokelatest(f)
    finally
        Core.eval(Hikari,graph)
        for (name,previous) in zip(names,old)
            isdefined(Hikari,name) || continue
            foreach(Base.delete_method,setdiff(Set(methods(Base.invokelatest(getfield,Hikari,name))),previous))
        end
    end
end

function run(output;device,repeats=31,profile=false,defer=true,names=("matte","rough_glass","rough_metal","plastic","layered_principled"))
    Mantle.defaultdevice!(device)
    spec=Probe.configuration()
    results=Any[]
    for row in spec.cases
        row.name in names || continue
        baseline=Probe.fixture(Probe.principled_material(row),spec)
        candidate=nothing
        try
            Base.invokelatest(Probe.draw,baseline)
            expected=Probe.pixels(baseline)
            with_deferred(;defer) do
                candidate=Base.invokelatest(Probe.fixture,Probe.principled_material(row),spec)
                first_seconds=@elapsed Base.invokelatest(Probe.draw,candidate)
                actual=Probe.pixels(candidate)
                @test all(isfinite,actual)
                @test actual≈expected rtol=5f-5 atol=2f-5
                times=(Float64[],Float64[])
                for _ in 1:5,f in (baseline,candidate)
                    Base.invokelatest(Probe.draw,f)
                end
                for i in 1:repeats,k in (isodd(i) ? (1,2) : (2,1))
                    push!(times[k],@elapsed Base.invokelatest(Probe.draw,k==1 ? baseline : candidate))
                end
                @test Probe.pixels(candidate)≈actual rtol=5f-5 atol=2f-5
                gpu=Any[]
                if profile
                    for (label,f) in zip(("eager","deferred"),(baseline,candidate))
                        plans=Hikari.allplans(f.vp.state.plans)
                        for p in plans
                            p.profiler===nothing && error("No GPU timestamp profiler")
                            p.profile=true
                            Mantle.invalidate!(p)
                            foreach(empty!,p.profiler.gpu_ns)
                        end
                        for _ in 1:8
                            Base.invokelatest(Probe.draw,f)
                        end
                        passes=[t for p in plans for t in Mantle.timings(p) if t.name in ("primary","accumulate")]
                        launches=[d.launch for p in plans for pp in p.passes if pp.pass.name=="primary" for d in pp.dispatches]
                        launch=only(launches)
                        mkpath(dirname(output))
                        write(joinpath(dirname(output),"$(row.name)_$label.spv"),launch.compiled.spirv_bytes)
                        native=Mantle.pipeline_exec_stats(f.dev,launch.pipeline)
                        pdf=Array(f.vp.state.pdf_per_pixel)
                        sampled=count(!iszero,view(pdf,1:4:length(pdf)))
                        push!(gpu,(;label,passes,native,stored_pdf_paths=sampled,total_paths=length(pdf)÷4,
                                   spirv_bytes=length(launch.compiled.spirv_bytes)))
                    end
                end
                result=(;name=row.name,defer,first_seconds,frame_seconds=times,median_seconds=map(median,times),
                         max_pixel_error=maximum(abs.(actual.-expected)),gpu)
                push!(results,result)
                mkpath(dirname(output));open(io->JSON3.write(io,results),output,"w")
                println(row.name," eager/deferred wavelength ms: ",result.median_seconds.*1000,
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
