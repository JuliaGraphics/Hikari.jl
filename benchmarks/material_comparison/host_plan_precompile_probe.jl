module HostPlanPrecompileProbe
using Hikari, Mantle, JSON3, SHA, Test
using ..MaterialComparison

function prime(device::Type{<:Mantle.Device})
    signatures = (Tuple{Type{Mantle.Plan},Mantle.Graph{device}},
                  Tuple{typeof(Mantle.compile!),Mantle.Compile{device}})
    return count(Base.precompile, signatures), length(signatures)
end

function run(output; device, expected_hash, trace = false)
    priming = @timed prime(typeof(device))
    @test priming.value == (2, 2)
    spec = MaterialComparison.configuration()
    setup = @timed MaterialComparison.fixture(
        MaterialComparison.principled_material(first(spec.cases)), spec)
    f = setup.value
    try
        Mantle.Lava.reset_compile_stats!()
        action() = Mantle.no_pipeline_compilation(() -> MaterialComparison.draw(f); ctx = device)
        draw = @timed if trace
            open(output * ".trace.jl", "w") do io
                redirect_stderr(io) do
                    Base.@trace_compile action()
                end
            end
        else
            action()
        end
        cache = Mantle.Lava.compile_stats()
        native_misses = device.diag.pipeline_compiles_refused
        digest = bytes2hex(sha256(reinterpret(UInt8, MaterialComparison.pixels(f))))
        @test cache.misses == native_misses == 0
        @test digest == expected_hash
        record = (; trace, priming_seconds = priming.time, priming_compile_seconds = priming.compile_time,
            setup_seconds = setup.time, draw_seconds = draw.time,
            draw_compile_seconds = draw.compile_time, cache, native_misses, image_sha256 = digest)
        open(io -> JSON3.write(io, record), output, "w")
        println(record)
        return record
    finally
        MaterialComparison.release(f)
    end
end
end
