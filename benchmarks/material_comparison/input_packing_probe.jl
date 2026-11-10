module InputPackingProbe
using Hikari, Mantle, JSON3, Statistics, Test
using ..MaterialComparison

function unpacked_program(ex, hits)
    ex isa Expr || return ex
    if ex.head === :call && ex.args[1] === :pack_principled_inputs
        hits[] += 1
        return ex.args[2]
    end
    return Expr(ex.head, map(x -> unpacked_program(x, hits), ex.args)...)
end

function run(output; spec, repeats = 31, names = ("mix_matte_metal", "mix_layered_glass"))
    program_method = which(Hikari.closure_program, Tuple{Hikari.CompiledProgram{UInt32(0)}})
    reader_method = which(Hikari.principled_inputs,
        Tuple{Hikari.GatheredStack{Vector{Float32}}, Any, Int32, Hikari.ShaderContext})
    program = Main.Revise.CodeTracking.definition(Expr, program_method)
    reader = Main.Revise.CodeTracking.definition(Expr, reader_method)
    program isa Expr && reader isa Expr || error("Cannot restore packed input compilation")
    hits = Ref(0)
    previous_program = unpacked_program(program, hits)
    hits[] == 1 || error("Expected one parameter-packing call")
    rows = Any[]
    for settings in filter(row -> row.name in names, spec.cases)
        fixtures = Any[]
        try
            Core.eval(Hikari, previous_program)
            @eval Hikari @inline function principled_inputs(s::GatheredStack, prog, i::Int32, sc::ShaderContext)
                invoke(principled_inputs, Tuple{Any, Any, Int32, ShaderContext}, s, prog, i, sc)
            end
            setup_a = @timed Base.invokelatest(MaterialComparison.fixture,
                MaterialComparison.principled_material(settings, spec), spec)
            push!(fixtures, setup_a.value)
            first_a = @timed Base.invokelatest(MaterialComparison.draw, setup_a.value)
            expected = MaterialComparison.pixels(setup_a.value)
            Core.eval(Hikari, program)
            Core.eval(Hikari, reader)
            setup_b = @timed Base.invokelatest(MaterialComparison.fixture,
                MaterialComparison.principled_material(settings, spec), spec)
            push!(fixtures, setup_b.value)
            first_b = @timed Base.invokelatest(MaterialComparison.draw, setup_b.value)
            actual = MaterialComparison.pixels(setup_b.value)
            @test all(isfinite, actual)
            @test actual ≈ expected rtol = 5f-5 atol = 2f-5
            @test MaterialComparison.shade_signatures(setup_a.value) == MaterialComparison.shade_signatures(setup_b.value)
            for _ in 1:3, fixture in fixtures
                Base.invokelatest(MaterialComparison.draw, fixture)
            end
            times = (Float64[], Float64[])
            for i in 1:repeats, j in (isodd(i) ? (1, 2) : (2, 1))
                push!(times[j], @elapsed Base.invokelatest(MaterialComparison.draw, fixtures[j]))
            end
            @test MaterialComparison.pixels(fixtures[1]) == expected
            @test MaterialComparison.pixels(fixtures[2]) == actual
            record = (; material = settings.name,
                baseline_seconds = median(times[1]), candidate_seconds = median(times[2]),
                first_seconds = (first_a.time, first_b.time),
                julia_compile_seconds = (first_a.compile_time, first_b.compile_time),
                setup_seconds = (setup_a.time, setup_b.time),
                max_pixel_error = maximum(abs.(actual .- expected)), frame_seconds = times)
            push!(rows, record)
            mkpath(dirname(output))
            open(io -> JSON3.write(io, (; device = Mantle.devicename(fixtures[1].dev),
                resolution = spec.resolution, samples = spec.samples, repeats, rows)), output, "w")
            println((; record.material, record.baseline_seconds, record.candidate_seconds, record.max_pixel_error))
            flush(stdout)
        finally
            Core.eval(Hikari, program)
            Core.eval(Hikari, reader)
            foreach(MaterialComparison.release, fixtures)
        end
    end
    return nothing
end
end
