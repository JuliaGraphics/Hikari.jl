module AggregateCallProbe
using Hikari, Mantle, JSON3, Statistics, Test
using ..MaterialComparison
const L = Mantle.Lava

function renamed(x)
    x === :force_inline_all! && return :composition_original_inline_policy!
    x isa Expr && return Expr(x.head, map(renamed, x.args)...)
    return x
end
function noinline_source(x)
    x === Symbol("@inline") && return Symbol("@noinline")
    x === :inline && return :noinline
    x isa QuoteNode && return QuoteNode(noinline_source(x.value))
    x isa Expr && return Expr(x.head,map(noinline_source,x.args)...)
    return x
end
function resource(f)
    launches = unique([d.launch for plan in Hikari.allplans(f.vp.state.plans)
        for pass in plan.passes if pass.pass.name == "shade" for d in pass.dispatches])
    map(launches) do launch
        words = reinterpret(UInt32, launch.compiled.spirv_bytes)
        at = 6; functions = 0
        while at <= length(words)
            functions += (words[at] & 0xffff) == 54
            at += Int(words[at] >> 16)
        end
        (; functions, bytes = length(launch.compiled.spirv_bytes),
            native = Mantle.pipeline_exec_stats(f.dev, launch.pipeline))
    end
end

function run(output; spec, repeats = 21)
    source = read(joinpath(pkgdir(Hikari),"src","materials","bsdf-program.jl"),String)
    field_source = Meta.parse(source,first(findfirst("Base.@constprop :none @inline @generated function surface_apply_fields",source)))[1]
    source = read(joinpath(pkgdir(L),"src","compiler","compilation.jl"),String)
    inline_source = Meta.parse(source,first(findfirst("function force_inline_all!",source)))[1]
    source = read(joinpath(pkgdir(L),"src","compiler","cache.jl"),String)
    lookup = Meta.parse(source,first(findfirst("@noinline function compile_or_lookup",source)))[1]
    rows = Any[]
    for settings in filter(row -> haskey(row,:mix),spec.cases)
        fixtures = Any[]
        helpers = Method[]
        world, cache = L._initialization_world[], L.ObjCache.enabled_state[]
        try
            baseline = MaterialComparison.fixture(MaterialComparison.principled_material(settings,spec),spec)
            push!(fixtures,baseline)
            Base.invokelatest(MaterialComparison.draw,baseline)
            expected = MaterialComparison.pixels(baseline)
            Core.eval(L,renamed(inline_source))
            helper = Base.invokelatest(getglobal,L,:composition_original_inline_policy!)
            append!(helpers,methods(helper))
            Core.eval(L, :(function force_inline_all!(mod::LLVM.Module, entry_fn::LLVM.Function; force_inline_all::Bool=false)
                type_sret_boxes!(mod,mod.datalayout)
                composition_original_inline_policy!(mod,entry_fn;force_inline_all)
            end))
            Core.eval(L, :(function compile_or_lookup(job::LavaCompilerJob)::LavaGPUKernel
                invoke_frozen(lava_compile_gpu_from_job,job)
            end))
            Core.eval(Hikari,noinline_source(field_source))
            L.ObjCache.enabled_state[] = false
            L._initialization_world[] = Base.get_world_counter()
            candidate = Base.invokelatest(MaterialComparison.fixture,
                MaterialComparison.principled_material(settings,spec),spec)
            push!(fixtures,candidate)
            first_seconds = @elapsed Base.invokelatest(MaterialComparison.draw,candidate)
            actual = MaterialComparison.pixels(candidate)
            @test actual ≈ expected rtol=5f-5 atol=2f-5
            for _ in 1:5, f in fixtures
                Base.invokelatest(MaterialComparison.draw,f)
            end
            times = (Float64[],Float64[])
            for i in 1:repeats,j in (isodd(i) ? (1,2) : (2,1))
                push!(times[j],@elapsed Base.invokelatest(MaterialComparison.draw,fixtures[j]))
            end
            @test MaterialComparison.pixels(candidate) ≈ expected rtol=5f-5 atol=2f-5
            record = (; material=settings.name,first_seconds,baseline_seconds=median(times[1]),
                candidate_seconds=median(times[2]),max_pixel_error=maximum(abs.(actual.-expected)),
                resources=map(resource,fixtures),frame_seconds=times)
            push!(rows,record)
            mkpath(dirname(output));open(io->JSON3.write(io,rows),output,"w")
            println((;record.material,record.baseline_seconds,record.candidate_seconds,
                record.max_pixel_error,functions=map(r->map(v->v.functions,r),record.resources)))
            flush(stdout)
        finally
            Core.eval(Hikari,field_source)
            Core.eval(L,lookup)
            Core.eval(L,inline_source)
            foreach(Base.delete_method,helpers)
            L.ObjCache.enabled_state[] = cache
            L._initialization_world[] = world
            foreach(MaterialComparison.release,fixtures)
        end
    end
    return nothing
end
end
