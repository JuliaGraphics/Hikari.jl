module HostPackOutlineProbe
using Hikari, Mantle, JSON3, Test
using ..MaterialComparison
const Probe = MaterialComparison

function run(output; device, reference)
    Mantle.defaultdevice!(device)
    originals = [Main.Revise.CodeTracking.definition(Expr, m) for m in methods(Mantle.pack_arg!)]
    all(x -> x isa Expr && x.head == :macrocall && x.args[1] == Symbol("@inline"), originals) ||
        error("Expected explicitly inline argument packers")
    f = nothing
    try
        for definition in originals
            replacement = deepcopy(definition)
            replacement.args[1] = Symbol("@noinline")
            Core.eval(Mantle, replacement)
        end
        f = Base.invokelatest(Probe.fixture, Probe.principled_material(first(Probe.configuration().cases)), Probe.configuration())
        Mantle.Lava.reset_compile_stats!()
        timing = @timed Base.invokelatest(Probe.draw, f)
        cache = Mantle.Lava.compile_stats()
        @test cache.misses == 0
        @test Probe.pixels(f) == reinterpret(Float32, read(reference))
        repeated = @timed Base.invokelatest(Probe.draw, f)
        @test Probe.pixels(f) == reinterpret(Float32, read(reference))
        mkpath(dirname(output))
        record = (; seconds = timing.time, julia_compile_seconds = timing.compile_time,
                  repeat_seconds = repeated.time, cache)
        open(io -> JSON3.write(io, record), output, "w")
        println("OUTLINED_PACKER_RESULT ", record)
    finally
        foreach(definition -> Core.eval(Mantle, definition), originals)
        f === nothing || Probe.release(f)
    end
    nothing
end
end
