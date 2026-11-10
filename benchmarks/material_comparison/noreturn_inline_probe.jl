module NoreturnInlineProbe
using Mantle, Test, KernelInterface
const L = Mantle.Lava
const LL = L.LLVM

struct Gate
    tag::UInt32
    index::UInt32
end
@noinline empty_dispatch(gate::Gate, context::NTuple{3,Float32}) = error("empty dispatch")

function scalar_path(x, tag)
    result = 0f0
    for bounce in 1:Int32(8)
        tag != 0 && empty_dispatch(Gate(tag, UInt32(bounce)), (x, x + 1f0, x + 2f0))
        weight = x > 0.5f0 ? 0.8f0 : 0.3f0
        result = muladd(weight, x, result)
        x = muladd(x, 0.73f0, 0.13f0)
        bounce > 4 && result > 2f0 && break
    end
    return result
end
function kernel!(dst, src, tags)
    i = KernelInterface.get_global_id().x
    i > length(src) && return nothing
    @inbounds dst[i] = scalar_path(src[i], tags[i])
    return nothing
end

function run()
    dev = Mantle.Device(Mantle.defaultbackend())
    source = Float32.(range(-1, 2; length = 1024))
    dst = Mantle.devicearray(dev, zeros(Float32, length(source)))
    src = Mantle.devicearray(dev, source)
    tags = Mantle.devicearray(dev, zeros(UInt32, length(source)))
    target = read(joinpath(pkgdir(L), "src", "compiler", "target.jl"), String)
    hook = Meta.parse(target, first(findfirst("function GPUCompiler.finish_linked_module!", target)))[1]
    text = read(joinpath(pkgdir(L), "src", "compiler", "cache.jl"), String)
    lookup = Meta.parse(text, first(findfirst("@noinline function compile_or_lookup", text)))[1]
    world, cache = L._initialization_world[], L.ObjCache.enabled_state[]
    try
        Core.eval(L, :(function compile_or_lookup(job::LavaCompilerJob)::LavaGPUKernel
            invoke_frozen(lava_compile_gpu_from_job, job)
        end))
        L.ObjCache.enabled_state[] = false
        @testset "normal lanes around inlined nonreturning dispatch" begin
            for early in (false, true)
                policy = function (mod)
                    early || return nothing
                    for fn in mod.functions
                        startswith(fn.name, "julia_empty_dispatch_") || continue
                        delete!(fn.function_attributes, LL.EnumAttribute("noinline"))
                        push!(fn.function_attributes, LL.EnumAttribute("alwaysinline"))
                    end
                    return nothing
                end
                Core.eval(L, :(function GPUCompiler.finish_linked_module!(job::LavaCompilerJob, mod::LLVM.Module)
                    native!(mod); $policy(mod); nothing
                end))
                L._initialization_world[] = Base.get_world_counter()
                Base.invokelatest() do
                    KernelInterface.@launch dev ndrange=length(source) kernel!(dst, src, tags)
                end
                Mantle.waitidle(dev)
                actual = Array(dst)
                expected = scalar_path.(source, UInt32(0))
                @test all(isfinite, actual)
                @test actual ≈ expected rtol = 1f-6 atol = 1f-6
                println((; early, maximum_error = maximum(abs.(actual .- expected))))
            end
        end
    finally
        Core.eval(L, lookup)
        Core.eval(L, hook)
        L._initialization_world[], L.ObjCache.enabled_state[] = world, cache
        foreach(Mantle.unsafe_free!, (dst, src, tags))
    end
    return nothing
end
end
