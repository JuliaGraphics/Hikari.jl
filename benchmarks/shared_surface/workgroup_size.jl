# Diagnostic: retain shared runtime storage and vary only the BSDF workgroup size.
# Load renderer.jl first. A temporary method is removed in the finally block.
module SharedGroupSizeProbe
using Hikari, Mantle, Test, Statistics, JSON3
using Main.SurfaceProgramGPUProbe: fixture, draw, draw_reference, image, release

function resources(f)
    launches = unique(
        [
            d.launch for pl in Hikari.allplans(f.vp.state.plans)
                for pp in pl.passes if pp.pass.name == "shade" for d in pp.dispatches
        ]
    )
    return [(; identity = objectid(l.pipeline), stats = Mantle.pipeline_exec_stats(f.dev, l.pipeline)) for l in launches]
end

function run(material = Hikari.Metallic(); pairs = 51, result_path = nothing, group = 128)
    reference = fixture(material; resolution = 512, samples = 8)
    baseline = fixture(material; resolution = 512, samples = 8)
    candidate = nothing
    added = nothing
    try
        draw_reference(reference)
        Base.invokelatest(draw, baseline)
        expected = image(reference)
        baseimage = image(baseline)
        @test baseimage ≈ expected rtol = 5.0f-5 atol = 2.0f-5
        before = resources(baseline)
        @assert !any(m -> Base.unwrap_unionall(m.sig).parameters[2] === Int, methods(Hikari.shader_bsdf_workgroupsize))
        @eval Hikari shader_bsdf_workgroupsize(group::Int) = min(group, $group)
        added = which(Hikari.shader_bsdf_workgroupsize, Tuple{Int})
        candidate = fixture(material; resolution = 512, samples = 8)
        first = @elapsed Base.invokelatest(draw, candidate)
        actual = image(candidate)
        @test all(isfinite, actual)
        @test maximum(expected) > 0
        @test actual ≈ expected rtol = 5.0f-5 atol = 2.0f-5
        fs = (reference, baseline, candidate)
        times = (Float64[], Float64[], Float64[])
        for _ in 1:5, f in fs
            Base.invokelatest(draw, f)
        end
        for j in 1:pairs, k in (isodd(j) ? (1, 2, 3) : (3, 2, 1))
            push!(times[k], @elapsed Base.invokelatest(draw, fs[k]))
        end
        @test resources(baseline) == before
        row = (;
            group, material = string(nameof(typeof(material))), first, max_error = maximum(abs.(actual .- expected)),
            reference = median(times[1]), baseline = median(times[2]), candidate = median(times[3]),
            before, after = resources(candidate), reference_resources = resources(reference),
        )
        println(JSON3.write(row))
        if result_path !== nothing
            mkpath(dirname(result_path))
            write(result_path, JSON3.write(row))
        end
        return row
    finally
        added === nothing || Base.delete_method(added)
        candidate === nothing || release(candidate)
        release(baseline);release(reference)
    end
end
end
