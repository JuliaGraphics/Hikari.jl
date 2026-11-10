module PrincipledLayoutProbe
using Hikari, Mantle, JSON3, Test, Statistics
using ..MaterialComparison
const Probe = MaterialComparison

# Temporary controls for the existing capacity-one glass and capacity-two
# opaque queues. The queue signatures and all Principled formulas stay fixed.
function definitions()
    fields = map(fieldnames(Hikari.PrincipledInputs)) do f
        f in (:coat_weight, :sheen_weight) ? :(0f0) : :(getfield(p, $(QuoteNode(f))))
    end
    two = quote
        @inline function surface_program_bsdf(::Type{SurfaceProgramNodes{2,L,M,E}}, s, prog, sc, lut, table,
                                             lambda, start::Int32=Int32(1)) where {L,M,E}
            p = surface_proven_inputs(principled_inputs(s, prog, start, sc), Val(M))
            p = PrincipledInputs($(fields...))
            convention = iszero(M & (UInt32(1)<<18)) ? CyclesGGX() : CyclesBeckmann()
            rgb, transparency, emission = principled_lobes(Val(false), Val(false), p, sc, lut,
                                                          prog[start].mode[2]==0x00, convention)
            terms = (rgb[3], rgb[5], rgb[6])
            return lobe_bsdf(Tuple{PrincipledBSDF{false,false,E}}, (terms,transparency,emission), table,lambda)
        end
    end
    glass = map((UInt32(1)<<7, (UInt32(1)<<7)|(UInt32(1)<<18))) do mask
        quote
            @inline function surface_program_bsdf(::Type{SurfaceProgramNodes{1,L,$mask,E}}, s, prog, sc, lut, table,
                                                 lambda, start::Int32=Int32(1)) where {L,E}
                p = surface_proven_inputs(principled_inputs(s,prog,start,sc), Val($mask))
                convention = iszero($mask & (UInt32(1)<<18)) ? CyclesGGX() : CyclesBeckmann()
                rgb, transparency, emission = principled_lobes(Val(false),Val(false),p,sc,lut,
                                                              prog[start].mode[2]==0x00,convention)
                return lobe_bsdf(Tuple{PrincipledBSDF{false,false,E}}, ((rgb[4],),transparency,emission), table,lambda)
            end
        end
    end
    return (two, glass...)
end

function with_layouts(f)
    mask = Hikari.SURFACE_PRINCIPLED_MASK | Hikari.SURFACE_OPAQUE_STORAGE
    signature(P) = Tuple{Type{P},Any,Any,Any,Any,Any,Any,Int32}
    which(Hikari.surface_program_bsdf,signature(Hikari.SurfaceProgramNodes{2,true,mask,false})) ===
        which(Hikari.surface_program_bsdf,signature(Hikari.SurfaceProgramNodes{8,true,mask,false})) ||
        error("This historical control requires source before the compact Principled layouts were integrated.")
    old = Set(methods(Hikari.surface_program_bsdf))
    added = Method[]
    try
        foreach(def -> Core.eval(Hikari,def), definitions())
        append!(added,setdiff(Set(methods(Hikari.surface_program_bsdf)),old))
        @assert length(added)==6
        return Base.invokelatest(f)
    finally
        foreach(Base.delete_method,added)
    end
end

function run(output;device,resolution=512,samples=64,repeats=11)
    Mantle.defaultdevice!(device)
    spec = Probe.configuration()
    rows = filter(r->r.name!="layered_principled",spec.cases)
    baseline = Any[]; candidates = Any[]; images = Any[]
    results = Any[]
    try
        for row in rows
            f = Probe.fixture(Probe.principled_material(row),spec;resolution,samples)
            push!(baseline,f)
            Base.invokelatest(Probe.draw,f)
            push!(images,Probe.pixels(f))
        end
        with_layouts() do
            for (i,row) in enumerate(rows)
                f = Base.invokelatest(Probe.fixture,Probe.principled_material(row),spec;resolution,samples)
                push!(candidates,f)
                first_seconds = @elapsed Base.invokelatest(Probe.draw,f)
                actual = Probe.pixels(f)
                @test all(isfinite,actual)
                @test actual≈images[i] rtol=5f-5 atol=2f-5
                @test Probe.shade_signatures(f)==Probe.shade_signatures(baseline[i])
                times = (Float64[],Float64[])
                for _ in 1:5, fixture in (baseline[i],f)
                    Base.invokelatest(Probe.draw,fixture)
                end
                @test Probe.pixels(f)≈actual rtol=5f-5 atol=2f-5
                for j in 1:repeats,k in (isodd(j) ? (1,2) : (2,1))
                    push!(times[k],@elapsed Base.invokelatest(Probe.draw,k==1 ? baseline[i] : f))
                end
                @test Probe.pixels(f)≈actual rtol=5f-5 atol=2f-5
                result = (;name=row.name,first_seconds,frame_seconds=times,median_seconds=map(median,times),
                           max_pixel_error=maximum(abs.(actual.-images[i])))
                push!(results,result)
                println(row.name," packed/fixed layout ms: ",result.median_seconds.*1000,
                        "; maximum pixel error: ",result.max_pixel_error)
            end
        end
        mkpath(dirname(output))
        open(io->JSON3.write(io,(;resolution,samples,rows=results)),output,"w")
    finally
        foreach(Probe.release,candidates)
        foreach(Probe.release,baseline)
    end
    nothing
end
end
