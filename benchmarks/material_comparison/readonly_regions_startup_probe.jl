module ReadonlyRegionsStartupProbe
using Hikari, Mantle, JSON3, SHA, Test
using ..MaterialComparison

struct Region
    owner::Any
    first::UInt128
    past::UInt128
    bounded::Bool
end

region(x) = Region(Mantle.bufferof(x), 0, 0, false)
function region(x::Mantle.LavaArray)
    base, offset = Mantle.GPUArrays.memory_location(x)
    start = UInt128(base) + offset
    return Region(Mantle.bufferof(x), start, start + sizeof(x), true)
end
overlap(a::Region, b::Region) = a.bounded && b.bounded ?
    a.first < a.past && b.first < b.past && a.first < b.past && b.first < a.past :
    a.owner === b.owner

function run(output; device, expected_hash, trace = true)
    method = which(Mantle.readonlyargs, Tuple{Mantle.Graph,Vector{Mantle.Touch},Tuple,Tuple})
    original = Main.Revise.CodeTracking.definition(Expr, method)
    f = nothing
    try
        Core.eval(Mantle, quote
            @noinline Base.@nospecializeinfer function readonlyargs(g::Graph, touches::AbstractVector{Touch},
                    @nospecialize(raw::Tuple), @nospecialize(adapted::Tuple))
                any(isreadonly, touches) || return adapted
                written = $(Region)[]
                for i in eachindex(touches)
                    touches[i].write || continue
                    for r in resourceleaves!(Any[], g, raw[i])
                        push!(written, $(region)(r))
                    end
                end
                result = Vector{Any}(undef, length(adapted))
                for i in eachindex(adapted)
                    wrap = isreadonly(touches[i])
                    if wrap
                        for resource in resourceleaves!(Any[], g, raw[i])
                            r = $(region)(resource)
                            for w in written
                                if $(overlap)(r, w)
                                    wrap = false
                                    break
                                end
                            end
                            wrap || break
                        end
                    end
                    result[i] = wrap ? Adapt.adapt(KA.ConstAdaptor(), adapted[i]) : adapted[i]
                end
                return Tuple(result)
            end
        end)
        spec = MaterialComparison.configuration()
        f = Base.invokelatest(MaterialComparison.fixture,
            MaterialComparison.principled_material(first(spec.cases)), spec)
        Mantle.Lava.reset_compile_stats!()
        action() = Mantle.no_pipeline_compilation(; ctx = device) do
            Base.invokelatest(MaterialComparison.draw, f)
        end
        draw = @timed if trace
            open(output * ".trace.jl", "w") do io
                redirect_stderr(io) do
                    Base.@trace_compile action()
                end
            end
        else
            action()
        end
        stats = Mantle.Lava.compile_stats()
        digest = bytes2hex(sha256(reinterpret(UInt8, MaterialComparison.pixels(f))))
        @test stats.misses == device.diag.pipeline_compiles_refused == 0
        @test digest == expected_hash
        record = (; draw_seconds = draw.time, compile_seconds = draw.compile_time,
            stats, image_sha256 = digest)
        open(io -> JSON3.write(io, record), output, "w")
        println(record)
        return record
    finally
        Core.eval(Mantle, original)
        f === nothing || MaterialComparison.release(f)
    end
end
end
