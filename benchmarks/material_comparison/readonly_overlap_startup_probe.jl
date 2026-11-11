module ReadonlyOverlapStartupProbe
using Hikari, Mantle, JSON3, SHA, Test
using ..MaterialComparison

function replace_overlap(ex, count, direct)
    ex isa Expr || return ex
    if ex.head === :call && ex.args[1] === :any && length(ex.args) == 3 &&
            ex.args[3] === :written_resources
        count[] += 1
        if direct
            return quote
                local overlaps = false
                for written in written_resources
                    if readonlyoverlap(resource, written)
                        overlaps = true
                        break
                    end
                end
                overlaps
            end
        end
        return :(_readonly_overlap_probe(resource, written_resources))
    end
    return Expr(ex.head, map(x -> replace_overlap(x, count, direct), ex.args)...)
end

function run(output; device, expected_hash, direct = false, trace = false)
    method = which(Mantle.readonlyargs, Tuple{Mantle.Graph,Vector{Mantle.Touch},Tuple,Tuple})
    original = Main.Revise.CodeTracking.definition(Expr, method)
    count = Ref(0)
    replacement = replace_overlap(original, count, direct)
    @assert count[] == 1
    helper = nothing
    if !direct
        Core.eval(Mantle, quote
        @noinline Base.@nospecializeinfer function _readonly_overlap_probe(
                @nospecialize(resource), written_resources::Vector{Any})
            for written in written_resources
                readonlyoverlap(resource, written) && return true
            end
            return false
        end
        end)
        helper = which(Base.invokelatest(getglobal, Mantle, :_readonly_overlap_probe), Tuple{Any,Vector{Any}})
    end
    f = nothing
    try
        Core.eval(Mantle, replacement)
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
        record = (; direct, trace, draw_seconds = draw.time, compile_seconds = draw.compile_time,
            stats, image_sha256 = digest)
        open(io -> JSON3.write(io, record), output, "w")
        println(record)
        return record
    finally
        Core.eval(Mantle, original)
        helper === nothing || Base.delete_method(helper)
        f === nothing || MaterialComparison.release(f)
    end
end
end
