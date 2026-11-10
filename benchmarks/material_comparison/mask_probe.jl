module MaskProbe
using Hikari, Mantle, JSON3, Test, Statistics, GeometryBasics, LinearAlgebra
using ..MaterialComparison
const Probe = MaterialComparison

function guarded(ex)
    ex isa Expr || return ex
    if ex.head === :call && ex.args[1] === :(==) && length(ex.args) == 3 &&
       ex.args[2] == :(w.lobe.kind) && ex.args[3] in (1,11)
        return :(storage_kind_probe(w.lobe, Val($(ex.args[3]))) && $ex)
    end
    ex.args = map(guarded, ex.args)
    return ex
end

const definitions = quote
    @inline storage_kind_probe(::SurfaceStoredLobe{L,M}, ::Val{K}) where {L,M,K} =
        M isa Bool || !iszero(M & (UInt32(1) << K))
    @inline lobe_frame(l::SurfaceStoredLobe, n::Vec3f, t::Vec3f, wo::Vec3f) =
        storage_kind_probe(l,Val(2)) && l.kind == 2 ? lobe_frame(surface_restore(l,Val(2)),n,t,wo) :
        (t == Vec3f(0) ? (n, coordinate_system(n)...) : (n,t,cross(n,t)))
    @inline is_diffuse_lobe(l::SurfaceStoredLobe{L,M}) where {L,M} =
        (M isa Bool || !iszero(M & UInt32(0x5806))) && l.kind in (1,2,11,12,14)
end

function run(output; device, repeats = 11)
    Mantle.defaultdevice!(device)
    T = Hikari.SurfaceStoredLobe{true,Hikari.SURFACE_PRINCIPLED_MASK}
    W = Hikari.SurfaceStoredWeight{true,Hikari.SURFACE_PRINCIPLED_MASK}
    methods_to_restore = (
        which(Hikari.lobe_frame,Tuple{T,Vec3f,Vec3f,Vec3f}),
        which(Hikari.is_diffuse_lobe,Tuple{T}),
        which(Hikari.weighted_f,Tuple{W,Vec3f,Vec3f,Vec3f,Hikari.Wavelengths,Hikari.AnyScattering}))
    originals = map(m -> Main.Revise.CodeTracking.definition(Expr,m),methods_to_restore)
    all(x -> x isa Expr, originals) || error("Cannot restore mask controls")
    spec = Probe.configuration()
    rows = Any[]
    for row in spec.cases
        baseline = Probe.fixture(Probe.principled_material(row),spec)
        candidate = nothing
        try
            Base.invokelatest(Probe.draw,baseline)
            expected = Probe.pixels(baseline)
            Core.eval(Hikari,definitions)
            Core.eval(Hikari,guarded(deepcopy(originals[3])))
            candidate = Base.invokelatest(Probe.fixture,Probe.principled_material(row),spec)
            first_seconds = @elapsed Base.invokelatest(Probe.draw,candidate)
            actual = Probe.pixels(candidate)
            @test all(isfinite,actual)
            if !isapprox(actual,expected;rtol=5f-5,atol=2f-5)
                difference = (;material=String(row.name),max_error=maximum(abs.(actual.-expected)),
                    relative_norm=norm(actual.-expected)/norm(expected))
                mkpath(dirname(output))
                open(io -> JSON3.write(io,difference),output*".failure.json","w")
            end
            @test actual ≈ expected rtol=5f-5 atol=2f-5
            times = (Float64[],Float64[])
            for _ in 1:5, f in (baseline,candidate)
                Base.invokelatest(Probe.draw,f)
            end
            for i in 1:repeats, k in (isodd(i) ? (1,2) : (2,1))
                push!(times[k],@elapsed Base.invokelatest(Probe.draw,k==1 ? baseline : candidate))
            end
            record=(;material=String(row.name),first_seconds,median_seconds=map(median,times),
                frame_seconds=times,max_pixel_error=maximum(abs.(actual.-expected)))
            push!(rows,record)
            mkpath(dirname(output));open(io -> JSON3.write(io,rows),output,"w")
            println(row.name," unguarded/mask-guarded ms: ",record.median_seconds.*1000,
                    "; image error: ",record.max_pixel_error)
            flush(stdout)
        finally
            foreach(ex -> Core.eval(Hikari,ex),originals)
            candidate === nothing || Probe.release(candidate)
            Probe.release(baseline)
        end
    end
    return nothing
end
end
