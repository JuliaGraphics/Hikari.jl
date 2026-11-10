module SamplerProbe
using Hikari, Mantle, KernelAbstractions, JSON3, Test, Random, Statistics
using ..MaterialComparison
const Probe=MaterialComparison

const folded_mod24=quote
    @inline function mod24(x::UInt64)::UInt32
        # x = 8q+r. Since 2^16 and 2^32 are 1 modulo 3, q's three
        # chunks sum to the same residue without overflowing UInt32.
        qsum=UInt32(x>>35)+UInt32((x>>19)&0xffff)+UInt32((x>>3)&0xffff)
        return ((qsum%UInt32(3))<<3)|UInt32(x&7)
    end
end

@kernel inbounds=true function camera_stage!(out,rng,cref,filter_ref,filter_data,height,sample_idx,::Val{Stage}) where {Stage}
    j=@index(Global)
    if j<=size(out,1)
        x,y,s=Hikari.path_pixel_sample(rng,Int32(j),sample_idx)
        camera=cref[Int32(1)]
        p=Hikari.compute_pixel_sample(rng,x,y,s,Hikari.camera_uses_motion_blur(camera),Hikari.camera_uses_lens(camera))
        out[j,1]=p.jitter_x;out[j,2]=p.jitter_y;out[j,3]=p.wavelength_u
        if Stage>=1
            lambda=Hikari.sample_wavelengths_visible(p.wavelength_u)
            for k in 1:4
                out[j,3+k]=lambda.lambda[k]
                out[j,7+k]=lambda.pdf[k]
            end
        end
        if Stage>=2
            fs=Hikari.filter_sample(filter_ref[Int32(1)],filter_data,Hikari.Point2f(p.jitter_x,p.jitter_y))
            pf=Hikari.Point2f(Float32(x)-0.5f0+fs.p[1],Float32(height)-Float32(y)+0.5f0+fs.p[2])
            ray,_=Hikari.generate_ray(camera,Hikari.CameraSample(pf,Hikari.Point2f(p.lens_u,p.lens_v),p.time))
            for k in 1:3
                out[j,11+k]=ray.o[k]
                out[j,14+k]=ray.d[k]
            end
        end
    end
end

function run_stages(output;pairs=21,group=64)
    isdefined(Hikari, :vp_generate_primary_rays_kernel!) && error(
        "This historical stage probe requires the separate camera pass before primary fusion.")
    dev=Mantle.defaultdevice!("RTX 4000 Ada")
    spec=Probe.configuration()
    fixture=Probe.fixture(Probe.principled_material(first(spec.cases)),spec)
    arrays=Any[];plans=Any[]
    try
        Probe.draw(fixture)
        d=only([d for pl in Hikari.allplans(fixture.vp.state.plans) for pp in pl.passes
            if pp.pass.name=="camera" for d in pp.pass.dispatches])
        args=d.args
        rng=fixture.vp.state.sobol_rng;count=Int(rng.pixels)*Int(rng.lanes)
        for stage in 0:2
            out=Mantle.devicearray(dev,Float32,(count,17));push!(arrays,out)
            g=Mantle.Graph(dev)
            Mantle.dispatch!(g,camera_stage!,
                (out,rng,args[6],args[9],args[10],args[5],Int32(64),Val(stage)),count;group,name="camera stage")
            push!(plans,Mantle.record!(Mantle.Plan(g;profile=true)))
        end
        g=Mantle.Graph(dev)
        Mantle.dispatch!(g,Hikari.zero_size_counters_kernel!,((first(args).size,),),1;group=1,name="reset")
        Mantle.dispatch!(g,Hikari.vp_generate_camera_rays_kernel!,args,count;group,name="camera stage")
        push!(plans,Mantle.record!(Mantle.Plan(g;profile=true)))
        for _ in 1:5,p in plans
            Mantle.run!(p);Mantle.waitidle(dev)
        end
        @test Array(arrays[1])[:,1:3]==Array(arrays[2])[:,1:3]
        @test Array(arrays[2])[:,1:11]≈Array(arrays[3])[:,1:11]
        for i in 1:pairs,k in (isodd(i) ? (1:4) : (4:-1:1))
            Mantle.run!(plans[k]);Mantle.waitidle(dev)
        end
        gpu=[only(filter(t->t.name=="camera stage",Mantle.timings(p))).gpu_ms for p in plans]
        labels=("sampling","sampling and wavelengths","sampling, wavelengths and ray","actual camera queue")
        mkpath(dirname(output));open(io->JSON3.write(io,(;count,group,labels,gpu_ms=gpu)),output,"w")
        println("Camera stage GPU ms: ",collect(zip(labels,gpu)))
    finally
        foreach(Mantle.free!,plans)
        foreach(Mantle.unsafe_free!,arrays)
        Probe.release(fixture)
    end
    nothing
end

function validate()
    rng=MersenneTwister(0x24)
    values=[UInt64(0),typemax(UInt64),rand(rng,UInt64,100_000)...]
    for k in 0:63
        x=UInt64(1)<<k
        push!(values,x,x-UInt64(1))
    end
    @test all(Base.invokelatest(Hikari.mod24,x)==UInt32(x%UInt64(24)) for x in values)
end

@kernel inbounds=true function samples!(out,rng,sample_idx,needs_time,needs_lens)
    j=@index(Global)
    if j<=size(out,2)
        x,y,s=Hikari.path_pixel_sample(rng,Int32(j),sample_idx)
        p=Hikari.compute_pixel_sample(rng,x,y,s,needs_time,needs_lens)
        values=(p.jitter_x,p.jitter_y,p.wavelength_u,p.lens_u,p.lens_v,p.time)
        for k in 1:6
            out[k,j]=values[k]
        end
    end
end

function plan(dev,out,rng,sample_idx,needs_time)
    g=Mantle.Graph(dev)
    Mantle.dispatch!(g,samples!,(out,rng,sample_idx,needs_time,false),size(out,2);group=128,name="sampler")
    Mantle.record!(Mantle.Plan(g;profile=true))
end

function run(output; pairs=21)
    dev=Mantle.defaultdevice!("RTX 4000 Ada")
    spec=Probe.configuration()
    fixture=Probe.fixture(Probe.principled_material(first(spec.cases)),spec)
    original=Main.Revise.CodeTracking.definition(Expr,only(methods(Hikari.mod24)))
    original isa Expr || error("Cannot restore sampler")
    arrays=Any[]; plans=Any[];rows=Any[]
    try
        Probe.draw(fixture)
        rng=fixture.vp.state.sobol_rng
        count=Int(rng.pixels)*Int(rng.lanes)
        for needs_time in (false,true)
            a=Mantle.devicearray(dev,Float32,(6,count));push!(arrays,a)
            b=Mantle.devicearray(dev,Float32,(6,count));push!(arrays,b)
            baseline=plan(dev,a,rng,Int32(64),needs_time);push!(plans,baseline)
            Core.eval(Hikari,folded_mod24)
            validate()
            candidate=Base.invokelatest(plan,dev,b,rng,Int32(64),needs_time);push!(plans,candidate)
            for p in (baseline,candidate)
                Mantle.run!(p);Mantle.waitidle(dev)
            end
            @test Array(a)==Array(b)
            times=(Float64[],Float64[])
            for _ in 1:5,p in (baseline,candidate)
                Mantle.run!(p);Mantle.waitidle(dev)
            end
            for i in 1:pairs,k in (isodd(i) ? (1,2) : (2,1))
                p=k==1 ? baseline : candidate
                push!(times[k],@elapsed (Mantle.run!(p);Mantle.waitidle(dev)))
            end
            gpu=map(p->only(filter(t->t.name=="sampler",Mantle.timings(p))).gpu_ms,(baseline,candidate))
            row=(;needs_time,count,median_wall_seconds=map(median,times),gpu_ms=gpu,wall_seconds=times)
            push!(rows,row)
            mkpath(dirname(output));open(io->JSON3.write(io,rows),output,"w")
            println("Sampler needs_time=",needs_time," old/folded GPU ms=",gpu);flush(stdout)
            Core.eval(Hikari,original)
        end
    finally
        Core.eval(Hikari,original)
        foreach(Mantle.free!,plans)
        foreach(Mantle.unsafe_free!,arrays)
        Probe.release(fixture)
    end
    nothing
end
end
