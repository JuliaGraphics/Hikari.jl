module RecordPackStartupProbe
using Hikari, Mantle, JSON3, SHA, Test
using ..MaterialComparison

function replace_packer(ex)
    ex isa Expr || return ex
    if ex.head === :call && ex.args[1] === :pack_args_direct!
        return Expr(:call, :_record_pack_probe!, ex.args[2:end]...)
    end
    return Expr(ex.head, map(replace_packer, ex.args)...)
end

function run(output; device, expected_hash, generic = false, erase = false, trace = false)
    method = which(Mantle.packdispatch!, Tuple{Mantle.Emitter,Mantle.LavaDispatch,Any})
    original = Main.Revise.CodeTracking.definition(Expr, method)
    @assert original.head === :function
    replacement = deepcopy(original)
    signature = replacement.args[1]
    @assert signature.head === :call && signature.args[1] === :packdispatch!
    helper = nothing
    if generic
        Core.eval(Mantle, quote
            @noinline Base.@nospecializeinfer function _record_pack_probe!(owner::Closed, site,
                    ptr::Ptr{UInt8}, address::UInt64, offsets::Vector{Int}, base::Int,
                    sizes::Vector{Int}, @nospecialize(args::Tuple))
                cursor = base
                for (layout, arg) in enumerate(passedslots(typeof(args).parameters))
                    cursor = pack_arg!(args[arg], ptr, address, offsets[layout],
                                       sizes[layout], cursor, owner, site)::Int
                end
                return nothing
            end
        end)
        helper = only(methods(Base.invokelatest(getglobal, Mantle, :_record_pack_probe!)))
        replacement = replace_packer(replacement)
    end
    if !generic || erase
        signature = replacement.args[1]
        for i in (3, 4)
            signature.args[i] = :(@nospecialize($(signature.args[i])))
        end
        replacement = :(Base.@nospecializeinfer $replacement)
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
        repeated = @timed Base.invokelatest(MaterialComparison.draw, f)
        @test bytes2hex(sha256(reinterpret(UInt8, MaterialComparison.pixels(f)))) == expected_hash
        record = (; generic, erase, trace, draw_seconds = draw.time, compile_seconds = draw.compile_time,
            repeat_seconds = repeated.time, stats, image_sha256 = digest)
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
