module InlineEscapeProbe
using Hikari, Mantle, Adapt, JSON3, Test, Statistics
using ..MaterialComparison
const Probe = MaterialComparison

struct InlineEscapes{P,R,L,B,T,I}
    pixel_L::P
    rgb2spec_table::R
    lights::L
    bvh_nodes::B
    light_to_bit_trail::T
    infinite_indices::I
    num_infinite::Int32
    num_bvh::Int32
end
Adapt.@adapt_structure InlineEscapes

Base.@propagate_inbounds function Base.push!(q::InlineEscapes, work::Hikari.VPEscapedRayWorkItem)
    Hikari.vp_handle_escaped_rays_kernel!(work, q.pixel_L, q.rgb2spec_table, q.lights,
        q.bvh_nodes, q.light_to_bit_trail, q.infinite_indices, q.num_infinite, q.num_bvh)
    return Int32(0)
end

function inline_queue(ex, count)
    ex isa Expr || return ex
    if ex == :(state.escaped_queue)
        count[] += 1
        return :($InlineEscapes(state.pixel_L, state.rgb2spec_table, refs.lights,
            state.bvh_nodes, state.light_to_bit_trail, state.infinite_light_indices,
            state.num_infinite_lights, state.num_bvh_lights))
    end
    ex.args = map(a -> inline_queue(a, count), ex.args)
    return ex
end

function run(output; spec, repeats = 31, names = ("matte", "rough_glass", "rough_metal", "plastic", "layered_principled"))
    original = Main.Revise.CodeTracking.definition(Expr, only(methods(Hikari.trace_arguments)))
    original isa Expr || error("Cannot recover trace arguments")
    count = Ref(0)
    candidate_definition = inline_queue(deepcopy(original), count)
    count[] == 1 || error("This historical control requires the pre-inline surface escape queue.")
    records = Any[]
    for name in names
        row = only(filter(r -> r.name == name, spec.cases))
        baseline = Probe.fixture(Probe.principled_material(row, spec), spec)
        candidate = nothing
        try
            Base.invokelatest(Probe.draw, baseline)
            expected = Probe.pixels(baseline)
            Core.eval(Hikari, candidate_definition)
            candidate = Base.invokelatest(Probe.fixture, Probe.principled_material(row, spec), spec)
            first_seconds = @elapsed Base.invokelatest(Probe.draw, candidate)
            actual = Probe.pixels(candidate)
            @test actual ≈ expected rtol = 5f-5 atol = 2f-5
            times = (Float64[], Float64[])
            for _ in 1:5, f in (baseline, candidate)
                Base.invokelatest(Probe.draw, f)
            end
            for i in 1:repeats, k in (isodd(i) ? (1,2) : (2,1))
                push!(times[k], @elapsed Base.invokelatest(Probe.draw, k == 1 ? baseline : candidate))
            end
            @test Probe.pixels(candidate) ≈ expected rtol = 5f-5 atol = 2f-5
            record = (; material = name, first_seconds, median_seconds = map(median, times), frame_seconds = times,
                        max_pixel_error = maximum(abs.(actual .- expected)))
            push!(records, record)
            mkpath(dirname(output))
            open(io -> JSON3.write(io, records), output, "w")
            println(name, " queued/inline all surface misses: ", record.median_seconds .* 1000, " ms")
            flush(stdout)
        finally
            Core.eval(Hikari, original)
            candidate === nothing || Probe.release(candidate)
            Probe.release(baseline)
        end
    end
    return nothing
end
end
