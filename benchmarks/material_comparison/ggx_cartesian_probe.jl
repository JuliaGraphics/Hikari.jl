module GGXCartesianProbe
using Hikari, Mantle, JSON3, Statistics, Test
using LinearAlgebra: normalize, norm
using ..MaterialComparison
const H = Hikari

# Cycles' anisotropic GGX uses Cartesian components directly. The current
# Hikari expressions recover azimuth through sin(theta), then cancel it again.
# Keep Hikari's grazing cutoff while isolating that arithmetic change.
@inline function distribution(h, ax, ay)
    z2 = h[3] * h[3]
    z2 * z2 < 1f-16 && return 0f0
    denominator = (h[1] / ax)^2 + (h[2] / ay)^2 + z2
    return 1f0 / (Float32(pi) * ax * ay * denominator^2)
end

@inline function masking(w, ax, ay)
    z2 = w[3] * w[3]
    iszero(z2) && return 0f0
    slope = ((ax * w[1])^2 + (ay * w[2])^2) / z2
    return (sqrt(1f0 + slope) - 1f0) * 0.5f0
end

function check_math()
    old_error, new_error = 0.0, 0.0
    @testset "Cartesian GGX against Float64 slope equations" begin
        for ax in (1f-4, 4f-4, 0.04f0, 0.25f0, 1f0),
                ay in (1f-4, 0.04f0, 0.5f0, 1f0),
                slope in (0.0, 1e-5, 1e-4, 4e-4, 0.001, 0.1, 1.0, 10.0, 1000.0),
                phi in (0.0, 0.3, 1.0, 2.0)
            h = H.Vec3f(normalize([slope * cos(phi), slope * sin(phi), 1.0]))
            x, y, z = Float64.(h)
            a, b = Float64(ax), Float64(ay)
            # An independent slope form, evaluated before any Float32 rounding.
            expected_d = inv(pi * a * b * z^4 * (1 + (x / (z * a))^2 + (y / (z * b))^2)^2)
            expected_l = (sqrt(1 + ((a * x)^2 + (b * y)^2) / z^2) - 1) / 2
            d, l = distribution(h, ax, ay), masking(h, ax, ay)
            @test d ≈ expected_d rtol=3e-6
            @test l ≈ expected_l rtol=3e-5 atol=1e-7
            old = H.trowbridge_reitz_d(h, ax, ay)
            old_error = max(old_error, abs(old / expected_d - 1))
            new_error = max(new_error, abs(d / expected_d - 1))
        end
        @test distribution(H.Vec3f(1, 0, 0), 0.2f0, 0.3f0) == 0f0
        @test masking(H.Vec3f(1, 0, 0), 0.2f0, 0.3f0) == 0f0
    end
    return (; old_max_relative_error=old_error, new_max_relative_error=new_error)
end

function resources(fixture)
    seen = Set{UInt}()
    result = Any[]
    for plan in H.allplans(fixture.vp.state.plans), pass in plan.passes
        pass.pass.name == "shade" || continue
        for dispatch in pass.dispatches
            pipeline = dispatch.launch.pipeline
            id = objectid(pipeline)
            id in seen && continue
            push!(seen, id)
            stats = Mantle.pipeline_exec_stats(fixture.dev, pipeline)
            stats === nothing && error("Request pipeline executable properties before device creation")
            push!(result, (; spirv_bytes=length(dispatch.launch.compiled.spirv_bytes), native=stats))
        end
    end
    return result
end

function run(output; spec, repeats=31, profile=false, names=("mix_matte_metal", "mix_layered_glass"))
    definitions = [(name, Main.Revise.CodeTracking.definition(Expr,
        which(getglobal(H, name), Tuple{H.Vec3f,Float32,Float32})))
        for name in (:trowbridge_reitz_d, :trowbridge_reitz_lambda)]
    all(item -> item[2] isa Expr, definitions) || error("Cannot restore GGX methods")
    checks = check_math()
    records = Any[]
    for settings in filter(row -> row.name in names, spec.cases)
        fixtures = Any[]
        try
            baseline = MaterialComparison.fixture(MaterialComparison.principled_material(settings, spec), spec)
            push!(fixtures, baseline)
            first_a = @timed Base.invokelatest(MaterialComparison.draw, baseline)
            expected = MaterialComparison.pixels(baseline)
            @eval H begin
                @inline trowbridge_reitz_d(h::Vec3f, ax::Float32, ay::Float32)::Float32 =
                    $distribution(h, ax, ay)
                @inline trowbridge_reitz_lambda(w::Vec3f, ax::Float32, ay::Float32)::Float32 =
                    $masking(w, ax, ay)
            end
            candidate = Base.invokelatest(MaterialComparison.fixture,
                MaterialComparison.principled_material(settings, spec), spec)
            push!(fixtures, candidate)
            first_b = @timed Base.invokelatest(MaterialComparison.draw, candidate)
            actual = MaterialComparison.pixels(candidate)
            @test all(isfinite, actual)
            @test actual ≈ expected rtol=5f-5 atol=2f-5
            for _ in 1:5, fixture in fixtures
                Base.invokelatest(MaterialComparison.draw, fixture)
            end
            times = (Float64[], Float64[])
            for i in 1:repeats, j in (isodd(i) ? (1, 2) : (2, 1))
                push!(times[j], @elapsed Base.invokelatest(MaterialComparison.draw, fixtures[j]))
            end
            @test MaterialComparison.pixels(baseline) == expected
            @test MaterialComparison.pixels(candidate) == actual
            record = (; material=settings.name, baseline_seconds=median(times[1]),
                candidate_seconds=median(times[2]), frame_seconds=times,
                max_pixel_error=maximum(abs.(actual .- expected)),
                relative_image_error=norm(actual .- expected) / norm(expected),
                first_seconds=(first_a.time, first_b.time),
                julia_compile_seconds=(first_a.compile_time, first_b.compile_time),
                resources=profile ? map(resources, fixtures) : nothing)
            push!(records, record)
            mkpath(dirname(output))
            open(io -> JSON3.write(io, (; device=Mantle.devicename(baseline.dev), checks,
                resolution=spec.resolution, samples=spec.samples, repeats, rows=records)), output, "w")
            println((; record.material, record.baseline_seconds, record.candidate_seconds,
                record.max_pixel_error)); flush(stdout)
        finally
            for (_, definition) in definitions
                Core.eval(H, definition)
            end
            foreach(MaterialComparison.release, fixtures)
        end
    end
    return nothing
end
end
