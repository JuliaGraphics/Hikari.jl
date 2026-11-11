module IndexProbe
using Hikari, Mantle, JSON3, Statistics, Test, GeometryBasics, LinearAlgebra
using ..MaterialComparison
const Probe=MaterialComparison

function validate_control(original_world)
    n=Vec3f(0,0,1)
    wo=normalize(Vec3f(0.2,0.1,1))
    lambda=Hikari.Wavelengths((450f0,510f0,570f0,630f0),(0.25f0,0.25f0,0.25f0,0.25f0))
    # Nonzero backing records beyond `count` must never contribute. Vary zero
    # weights inside the live prefix independently of the live count as well.
    @testset "bounded traversal controls" begin
        for zero_slot in 0:5
            lobes=ntuple(5) do j
                weight=zero_slot==j ? 0f0 : Float32(j)/5f0
                native=Hikari.Weighted(Hikari.LambertLobe(Hikari.SpectralRadiance),
                    Hikari.SpectralRadiance(0.4f0),weight,n,Vec3f(0))
                Hikari.surface_store(native,Val(true),Val(Hikari.SURFACE_PRINCIPLED_MASK))
            end
            for count in Int32(0):Int32(5), s in (Hikari.AnyScattering(),Hikari.OnlyReflection(),Hikari.OnlyTransmission())
                for wi in (n,-n)
                    a=Base.invoke_in_world(original_world,Hikari.surface_mixture,lobes,count,wo,wi,n,lambda,s)
                    b=Base.invokelatest(Hikari.surface_mixture,lobes,count,wo,wi,n,lambda,s)
                    @test isequal(a,b)
                end
                for target in (0f0,0.1f0,0.99f0,3f0)
                    args=(lobes,count,target,0f0,wo,Point2f(0.3,0.7),lambda,s)
                    a=Base.invoke_in_world(original_world,Hikari.surface_pick_sample,args...)
                    b=Base.invokelatest(Hikari.surface_pick_sample,args...)
                    @test isequal(a,b)
                end
            end
        end
    end
    nothing
end

# Change only mixture traversal of the existing five-record storage class.
# Constants replace the runtime tuple index; order and count checks stay the same.
const constant_indices=quote
    @inline function surface_mixture(lobes::NTuple{5,W}, count, wo, wi, ns, lambda, s) where {W <: SurfaceStoredWeight}
        f,pdf=SpectralRadiance(0f0),0f0
        Base.Cartesian.@nexprs 5 k -> begin
            if count >= 6-k
                w=lobes[6-k]
                if w.sample_weight > 0f0
                    fi,pi=weighted_f(w,wo,wi,ns,lambda,s)
                    f+=fi
                    pdf+=pi*w.sample_weight
                end
            end
        end
        return f,pdf
    end
end

const sampling_after_selection=quote
    @inline function surface_pick_sample(lobes::NTuple{5,W}, count, target::Float32,
                                        before::Float32, wo::Vec3f, u::Point2f,
                                        lambda::Wavelengths, s::Scattering) where {W <: SurfaceStoredWeight}
        selected=Int32(0)
        selected_sw,uc=0f0,0f0
        @inbounds for j in 1:count
            sw=share(lobes[j],s)
            after=before+sw
            if sw>0f0 && (target<after || j==count)
                selected=Int32(j)
                selected_sw=sw
                uc=clamp((target-before)/max(sw,1f-12),0f0,0.99999994f0)
                break
            end
            before=after
        end
        if selected!=0
            @inbounds w=lobes[selected]
            n,t,b=lobe_frame(w.lobe,w.n,w.t,wo)
            wi,f,pdf,flags,eta=lobe_sample(w.lobe,world_to_local(wo,n,t,b),u,uc,lambda,s)
            return (normalize(local_to_world(wi,n,t,b)),w.weight*f,pdf,flags,eta,
                    selected_sw,n,terminates_secondary(w.lobe,flags))
        end
        return pick_sample((),target,before,wo,u,lambda,s)
    end
end

function resources(f)
    launches=unique([d.launch for pl in Hikari.allplans(f.vp.state.plans)
        for pp in pl.passes if pp.pass.name=="shade" for d in pp.dispatches])
    [(;pipeline_id=objectid(l.pipeline),spirv_bytes=length(l.compiled.spirv_bytes),
        stats=Mantle.pipeline_exec_stats(f.dev,l.pipeline)) for l in launches]
end

function run(output; repeats=11, mode=:mixture)
    mode in (:mixture,:sampling,:both) || error("Unknown traversal probe $mode")
    if mode!==:sampling
        W=Hikari.SurfaceStoredWeight{true,Hikari.SURFACE_PRINCIPLED_MASK}
        reference=which(Hikari.surface_mixture,Tuple{NTuple{5,W},Int32,Vec3f,Vec3f,Vec3f,Hikari.Wavelengths,Hikari.AnyScattering})
        Base.unwrap_unionall(reference.sig).parameters[2]===Any ||
            error("Mixture probe needs the pre-optimization source; refusing to replace its existing specialized method")
    end
    Mantle.enable_pipeline_executable_properties!()
    Mantle.defaultdevice!("RTX 4000 Ada")
    spec=Probe.configuration()
    records=Any[]
    for name in ("matte","layered_principled")
        row=only(filter(r->r.name==name,spec.cases))
        baseline=Probe.fixture(Probe.principled_material(row),spec)
        candidate=nothing
        added=Method[]
        try
            Base.invokelatest(Probe.draw,baseline)
            expected=Probe.pixels(baseline)
            before=resources(baseline)
            for (function_,definition) in ((Hikari.surface_mixture,constant_indices),
                                           (Hikari.surface_pick_sample,sampling_after_selection))
                function_===Hikari.surface_mixture && mode===:sampling && continue
                function_===Hikari.surface_pick_sample && mode===:mixture && continue
                previous=Set(methods(function_))
                Core.eval(Hikari,definition)
                push!(added,only(setdiff(Set(methods(function_)),previous)))
            end
            candidate=Base.invokelatest(Probe.fixture,Probe.principled_material(row),spec)
            first_seconds=@elapsed Base.invokelatest(Probe.draw,candidate)
            actual=Probe.pixels(candidate)
            @test actual≈expected rtol=5f-5 atol=2f-5
            for _ in 1:5, f in (baseline,candidate)
                Base.invokelatest(Probe.draw,f)
            end
            times=(Float64[],Float64[])
            for i in 1:repeats, k in (isodd(i) ? (1,2) : (2,1))
                f=k==1 ? baseline : candidate
                push!(times[k],@elapsed Base.invokelatest(Probe.draw,f))
            end
            @test resources(baseline)==before
            record=(;material=name,mode=string(mode),first_seconds,baseline_seconds=times[1],candidate_seconds=times[2],
                    median_seconds=map(median,times),max_pixel_error=maximum(abs.(actual.-expected)),
                    baseline_resources=before,candidate_resources=resources(candidate))
            push!(records,record)
            mkpath(dirname(output))
            open(io->JSON3.write(io,records),output,"w")
            println(name," / ",mode,": ",record.median_seconds.*1000," ms; first ",first_seconds," s")
            flush(stdout)
        finally
            foreach(Base.delete_method,added)
            candidate===nothing || Probe.release(candidate)
            Probe.release(baseline)
        end
    end
    nothing
end
end
