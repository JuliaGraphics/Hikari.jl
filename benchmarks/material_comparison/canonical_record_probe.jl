module CanonicalRecordProbe
using Hikari, Mantle, JSON3, Test, Statistics
using ..MaterialComparison
const Probe=MaterialComparison

# Capacity five is reserved for a single opaque Principled program. Keep its
# five known lobe layouts, including zero-weight terms, instead of packing and
# dispatching five universal records. This does not change the queue signature.
const definition=quote
    @inline function surface_program_bsdf(::Type{SurfaceProgramNodes{5,L,M,E}},s,prog,sc,lut,table,lambda,
                                         start::Int32=Int32(1)) where {L,M,E}
        p=surface_proven_inputs(principled_inputs(s,prog,start,sc),Val(M))
        convention=iszero(M & (UInt32(1)<<18)) ? CyclesGGX() : CyclesBeckmann()
        produced=principled_lobes(Val(false),Val(false),p,sc,lut,prog[start].mode[2]==0x00,convention)
        rgb,transparency,emission=produced
        # The fourth term is the known-absent solid transmission lobe. Removing
        # only that term also excludes its unused dispersion from metadata.
        terms=(rgb[1],rgb[2],rgb[3],rgb[5],rgb[6])
        return lobe_bsdf(Tuple{PrincipledBSDF{false,false,E}},(terms,transparency,emission),table,lambda)
    end
end

function resources(f)
    launches=unique([d.launch for pl in Hikari.allplans(f.vp.state.plans)
        for pp in pl.passes if pp.pass.name=="shade" for d in pp.dispatches])
    [(;bytes=length(l.compiled.spirv_bytes),native=Mantle.pipeline_exec_stats(f.dev,l.pipeline))
        for l in launches]
end

function run(output;device,resolution=512,samples=64,repeats=11)
    mask=Hikari.SURFACE_PRINCIPLED_MASK|Hikari.SURFACE_OPAQUE_STORAGE
    signature(P)=Tuple{Type{P},Any,Any,Any,Any,Any,Any,Int32}
    which(Hikari.surface_program_bsdf,signature(Hikari.SurfaceProgramNodes{5,true,mask,false})) ===
        which(Hikari.surface_program_bsdf,signature(Hikari.SurfaceProgramNodes{8,true,mask,false})) || error(
            "This historical layout probe requires source before the opaque Principled layout was integrated.")
    Mantle.defaultdevice!(device)
    spec=Probe.configuration()
    row=only(filter(r->r.name=="layered_principled",spec.cases))
    baseline=Probe.fixture(Probe.principled_material(row),spec;resolution,samples)
    candidate=nothing;added=Method[]
    try
        Base.invokelatest(Probe.draw,baseline)
        expected=Probe.pixels(baseline)
        existing=Set(methods(Hikari.surface_program_bsdf))
        Core.eval(Hikari,definition)
        append!(added,setdiff(Set(methods(Hikari.surface_program_bsdf)),existing))
        @assert length(added)==2
        candidate=Base.invokelatest(Probe.fixture,Probe.principled_material(row),spec;resolution,samples)
        first_seconds=@elapsed Base.invokelatest(Probe.draw,candidate)
        actual=Probe.pixels(candidate)
        @test all(isfinite,actual)
        @test actual≈expected rtol=5f-5 atol=2f-5
        @test Probe.shade_signatures(candidate)==Probe.shade_signatures(baseline)
        times=(Float64[],Float64[])
        for _ in 1:5,f in (baseline,candidate)
            Base.invokelatest(Probe.draw,f)
        end
        @test Probe.pixels(candidate)≈actual rtol=5f-5 atol=2f-5
        for i in 1:repeats,k in (isodd(i) ? (1,2) : (2,1))
            push!(times[k],@elapsed Base.invokelatest(Probe.draw,k==1 ? baseline : candidate))
        end
        result=(;resolution,samples,first_seconds,frame_seconds=times,median_seconds=map(median,times),
            max_pixel_error=maximum(abs.(actual.-expected)),baseline_resources=resources(baseline),
            candidate_resources=resources(candidate))
        mkpath(dirname(output));open(io->JSON3.write(io,result),output,"w")
        println("Layered packed/canonical record ms: ",result.median_seconds.*1000,
                "; image error: ",result.max_pixel_error,
                "; SPIR-V bytes: ",[r.bytes for r in result.candidate_resources])
    finally
        foreach(Base.delete_method,added)
        candidate===nothing||Probe.release(candidate)
        Probe.release(baseline)
    end
    nothing
end
end
