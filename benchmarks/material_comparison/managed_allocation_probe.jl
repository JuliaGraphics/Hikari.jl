"""Cache-free compiler controls for the existing Cycles material reference batch."""
module ManagedAllocationProbe
using Hikari, Mantle, JSON3, SHA
const LL = Mantle.Lava.LLVM

function apply_policy!(mod, policy)
    policy == :local && return Mantle.Lava.inline_managed_arguments!(mod)
    policy == :none && return nothing
    for fn in mod.functions
        isempty(fn.blocks) && continue
        any(ty -> ty isa LL.PointerType && 10 <= ty.addrspace <= 13,
            fn.function_type.parameters) || continue
        if policy in (:returning_dispatch, :throwing_dispatch, :dispatch)
            startswith(fn.name, "julia_with_index_") || continue
            returns = any(ins -> ins isa LL.RetInst, (ins for bb in fn.blocks for ins in bb.instructions))
            policy == :returning_dispatch && !returns && continue
            policy == :throwing_dispatch && returns && continue
        elseif policy != :blanket
            error("Unknown compiler control: $policy")
        end
        delete!(fn.function_attributes, LL.EnumAttribute("noinline"))
        push!(fn.function_attributes, LL.EnumAttribute("alwaysinline"))
    end
    return nothing
end

function reference_module(; hw_accel = true)
    m = Module(gensym(:MaterialReferenceControl))
    Core.eval(m, :(include(path) = Base.include(@__MODULE__, path)))
    path = joinpath(pkgdir(Hikari), "test", "test_cycles_materials.jl")
    source = first(split(read(path, String), "@testset \"materials against Cycles\" begin"))
    if !hw_accel
        source = replace(source, "scene = Hikari.Scene(; backend)" => "scene = Hikari.Scene(; backend, hw_accel = false)",
            "Hikari.VolPath(; samples," => "Hikari.VolPath(; samples, hw_accel = false,")
    end
    Base.include_string(m, source, path)
    return m
end

function run_loaded(output, m; policy = :local, batch = 7, samples = 64, capture = nothing)
    L = Mantle.Lava
    world, cache = L._initialization_world[], L.ObjCache.enabled_state[]
    source = read(joinpath(pkgdir(L), "src", "compiler", "cache.jl"), String)
    lookup = Meta.parse(source, first(findfirst("@noinline function compile_or_lookup", source)))[1]
    source = read(joinpath(pkgdir(L), "src", "compiler", "target.jl"), String)
    hook = Meta.parse(source, first(findfirst("function GPUCompiler.finish_linked_module!", source)))[1]
    control = function (mod)
        if capture !== nothing
            mkpath(capture)
            for fn in mod.functions
                isempty(fn.blocks) && continue
                startswith(fn.name, "julia_with_index_") || continue
                write(joinpath(capture, fn.name * ".ll"), string(fn))
            end
        end
        apply_policy!(mod, policy)
    end
    try
        Core.eval(L, :(function GPUCompiler.finish_linked_module!(job::LavaCompilerJob, mod::LLVM.Module)
            native!(mod); $control(mod); nothing
        end))
        Core.eval(L, :(function compile_or_lookup(job::LavaCompilerJob)::LavaGPUKernel
            @atomic COMPILE_STATS.misses += 1
            invoke_frozen(lava_compile_gpu_from_job, job)
        end))
        L.ObjCache.enabled_state[] = false
        L._initialization_world[] = Base.get_world_counter()
        L.reset_compile_stats!()
        cases = m.cycles_batches()[batch]
        image = Base.invokelatest(m.render_cycles_batch, cases; samples)
        rows = map(enumerate(cases)) do (i, (name, _))
            reference, tolerance = m.CYCLES_REFERENCE[name]
            value = m.cycles_mean(image, m.cycles_sphere_pixels(i))
            (; name, mean = value, reference, passes = all(isapprox.(value, reference; rtol = tolerance)))
        end
        result = (; policy, batch, samples, rows, passes = all(row -> row.passes, rows),
            image_sha256 = bytes2hex(sha256(reinterpret(UInt8, vec(image)))),
            kernel = L.compile_stats(), functions = L.function_compile_stats(), llvm = L.llvm_compile_stats())
        mkpath(dirname(output))
        open(io -> JSON3.write(io, result), output, "w")
        println((; policy, result.passes, rows)); flush(stdout)
        return result
    finally
        Core.eval(L, lookup)
        Core.eval(L, hook)
        L.ObjCache.enabled_state[] = cache
        L._initialization_world[] = world
    end
end

run(output; hw_accel = true, kwargs...) = Base.invokelatest(run_loaded, output, reference_module(; hw_accel); kwargs...)
end
