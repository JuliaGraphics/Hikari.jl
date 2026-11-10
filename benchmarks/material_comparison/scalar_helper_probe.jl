module ScalarHelperProbe
using Hikari, Mantle, JSON3, Test, Statistics, GeometryBasics, LinearAlgebra
using ..MaterialComparison
const Probe=MaterialComparison

function function_body(ex)
    ex isa Expr || return nothing
    ex.head===:function && return ex.args[2]
    ex.head===:(=) && return Expr(:block,ex.args[2])
    for a in ex.args
        body=function_body(a)
        body===nothing || return body
    end
    nothing
end

function install_helper!(original, scope)
    ds=[Symbol(:d,k) for k in 1:15]
    ws=[Symbol(:w,k) for k in 1:4]
    ns=[Symbol(:n,k) for k in 1:3];ts=[Symbol(:t,k) for k in 1:3]
    os=[Symbol(:o,k) for k in 1:3];is=[Symbol(:i,k) for k in 1:3]
    ss=[Symbol(:s,k) for k in 1:3]
    ls=[Symbol(:l,k) for k in 1:4];ps=[Symbol(:p,k) for k in 1:4]
    floats=scope===:weighted ? [ds;ws;:sw;ns;ts;os;is;ss;ls;ps] : [ds;os;is;ls;ps]
    params=[:(_l::Val{L}),:(_m::Val{M}),:(kind::UInt32),
            [Expr(:(::),x,:Float32) for x in floats]...,:(s::Scattering)]
    signature=Expr(:where,Expr(:call,:surface_weighted_scalar_probe!,params...),:L,:M)
    initialize_weighted=quote
        w=Weighted(SurfaceStoredLobe{L,M}(kind,($(ds...),)),
            SpectralRadiance(($(ws...),)),sw,Vec3f($(ns...)),Vec3f($(ts...)))
        wo=Vec3f($(os...));wi=Vec3f($(is...));ns=Vec3f($(ss...))
        lambda=Wavelengths(($(ls...),),($(ps...),))
    end
    initialize_lobe=quote
        l=SurfaceStoredLobe{L,M}(kind,($(ds...),))
        wo=Vec3f($(os...));wi=Vec3f($(is...))
        lambda=Wavelengths(($(ls...),),($(ps...),))
    end
    initialize=scope===:weighted ? initialize_weighted : initialize_lobe
    body=function_body(original)
    body===nothing && error("Cannot recover evaluator body")
    definition=Expr(:function,signature,Expr(:block,initialize.args...,deepcopy(body).args...))
    Core.eval(Hikari,:(Base.@constprop :none @noinline $definition))
end

const wrapper=quote
    @inline function weighted_f(w::SurfaceStoredWeight{L,M},wo::Vec3f,wi::Vec3f,ns::Vec3f,
                               lambda::Wavelengths,s::Scattering=AnyScattering()) where {L,M}
        surface_weighted_scalar_probe!(Val(L),Val(M),w.lobe.kind,w.lobe.data...,
            w.weight.data...,w.sample_weight,w.n...,w.t...,wo...,wi...,ns...,
            lambda.lambda...,lambda.pdf...,s)
    end
end

const lobe_wrapper=quote
    @inline function lobe_f(l::SurfaceStoredLobe{L,M},wo::Vec3f,wi::Vec3f,
                           lambda::Wavelengths,s::Scattering) where {L,M}
        surface_weighted_scalar_probe!(Val(L),Val(M),l.kind,l.data...,
            wo...,wi...,lambda.lambda...,lambda.pdf...,s)
    end
end

const loop_mixture=quote
    @inline function surface_mixture(lobes::NTuple{5,W},count,wo,wi,ns,lambda,s) where {W <: SurfaceStoredWeight}
        invoke(surface_mixture,Tuple{Any,Any,Any,Any,Any,Any,Any},lobes,count,wo,wi,ns,lambda,s)
    end
end

function resources(f)
    launches=unique([d.launch for pl in Hikari.allplans(f.vp.state.plans)
        for pp in pl.passes if pp.pass.name=="shade" for d in pp.dispatches])
    [(;bytes=length(l.compiled.spirv_bytes),functions=Mantle.spirv_op_stats(l.compiled.spirv_bytes).n_functions,
        native=Mantle.pipeline_exec_stats(f.dev,l.pipeline))
        for l in launches]
end

function run(output;resolution=32,samples=4,repeats=11,profile=false,only_case=nothing,rr_depth=3,looped=false,device="RTX 4000 Ada",scope=:weighted)
    @assert scope in (:weighted,:lobe)
    Mantle.enable_pipeline_executable_properties!(profile)
    Mantle.defaultdevice!(device)
    W=Hikari.SurfaceStoredWeight{true,Hikari.SURFACE_PRINCIPLED_MASK}
    method=scope===:weighted ?
        which(Hikari.weighted_f,Tuple{W,Vec3f,Vec3f,Vec3f,Hikari.Wavelengths,Hikari.AnyScattering}) :
        which(Hikari.lobe_f,Tuple{Hikari.SurfaceStoredLobe{true,Hikari.SURFACE_PRINCIPLED_MASK},Vec3f,Vec3f,Hikari.Wavelengths,Hikari.AnyScattering})
    original=Main.Revise.CodeTracking.definition(Expr,method)
    original isa Expr || error("Cannot restore weighted evaluator")
    mixture=Main.Revise.CodeTracking.definition(Expr,which(Hikari.surface_mixture,
        Tuple{NTuple{5,W},Int32,Vec3f,Vec3f,Vec3f,Hikari.Wavelengths,Hikari.AnyScattering}))
    install_helper!(original,scope)
    spec=Probe.configuration();rows=Any[]
    for name in ("matte","layered_principled")
        only_case===nothing || name==only_case || continue
        row=only(filter(r->r.name==name,spec.cases))
        baseline=Probe.fixture(Probe.principled_material(row),spec;resolution,samples,rr_depth)
        candidate=nothing
        try
            Base.invokelatest(Probe.draw,baseline)
            expected=Probe.pixels(baseline)
            Core.eval(Hikari,scope===:weighted ? wrapper : lobe_wrapper)
            looped && Core.eval(Hikari,loop_mixture)
            candidate=Base.invokelatest(Probe.fixture,Probe.principled_material(row),spec;resolution,samples,rr_depth)
            first_seconds=@elapsed Base.invokelatest(Probe.draw,candidate)
            actual=Probe.pixels(candidate)
            @test all(isfinite,actual)
            if !isapprox(actual,expected;rtol=5f-5,atol=2f-5)
                path=dirname(output);mkpath(path)
                open(io->write(io,reinterpret(UInt8,actual)),joinpath(path,"scalar_$(scope)_$(name)_actual.f32"),"w")
                open(io->write(io,reinterpret(UInt8,expected)),joinpath(path,"scalar_$(scope)_$(name)_expected.f32"),"w")
                difference=(;max_error=maximum(abs.(actual.-expected)),relative_norm=norm(actual.-expected)/norm(expected),
                            mean_ratio=mean(actual)/mean(expected),reference_resources=resources(baseline),candidate_resources=resources(candidate))
                open(io->JSON3.write(io,difference),output*".failure.json","w")
                println("Scalar evaluator discrepancy: ",difference.max_error," max; ",difference.relative_norm," relative norm; mean ratio ",difference.mean_ratio)
            end
            @test actual≈expected rtol=5f-5 atol=2f-5
            times=(Float64[],Float64[])
            for _ in 1:5,f in (baseline,candidate)
                Base.invokelatest(Probe.draw,f)
            end
            for i in 1:repeats,k in (isodd(i) ? (1,2) : (2,1))
                push!(times[k],@elapsed Base.invokelatest(Probe.draw,k==1 ? baseline : candidate))
            end
            record=(;material=name,resolution,samples,rr_depth,looped,scope,first_seconds,median_seconds=map(median,times),
                    frame_seconds=times,max_pixel_error=maximum(abs.(actual.-expected)),
                    baseline_resources=resources(baseline),candidate_resources=resources(candidate))
            push!(rows,record);mkpath(dirname(output));open(io->JSON3.write(io,rows),output,"w")
            println(name," inline/scalar helper ms=",record.median_seconds.*1000," resources=",record.candidate_resources);flush(stdout)
        finally
            Core.eval(Hikari,original)
            looped && Core.eval(Hikari,mixture)
            candidate===nothing || Probe.release(candidate)
            Probe.release(baseline)
        end
    end
    nothing
end
end
