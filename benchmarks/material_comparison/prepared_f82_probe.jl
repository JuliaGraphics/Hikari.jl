module PreparedF82Probe
using Hikari, Mantle, JSON3, Statistics, Test
using ..MaterialComparison

# Cycles stores the F82 edge coefficient at setup. Keep the same record size,
# and retain the original tint for film-bearing records, which need it.
@inline function fields(f::Hikari.F82Tint{Hikari.SpectralRadiance})
    one = Hikari.SpectralRadiance(1f0)
    s = 6f0 / 7f0
    s5 = s^5
    b = Hikari.has_film(f) ? f.tint :
        (f.f0 + (one - f.f0) * s5) * (one - f.tint) * (7f0 / (s5 * s))
    return (f.f0.data..., b.data..., f.film_thickness, f.film_ior)
end

@inline function stored_fresnel(d, mu::Float32, ior::Float32, lambda::Hikari.Wavelengths)
    f0 = Hikari.surface_spectrum(d, Val(6))
    tint_or_b = Hikari.surface_spectrum(d, Val(10))
    if d[14] > Hikari.THIN_FILM_THICKNESS_CUTOFF
        return Hikari.fresnel_rt(Hikari.F82Tint(f0, tint_or_b, d[14], d[15]), mu, ior, lambda)
    end
    one = Hikari.SpectralRadiance(1f0)
    m = clamp(1f0 - mu, 0f0, 1f0)
    m5 = m^5
    return (clamp(f0 + (one - f0) * m5 - tint_or_b * (mu * m5 * m), 0f0, 1f0),
            Hikari.SpectralRadiance(0f0))
end

function replace_f82(ex, hits)
    ex isa Expr || return ex
    if ex.head === :call && ex.args[1] === :fresnel_rt && length(ex.args) == 5
        f = ex.args[2]
        if f isa Expr && f.head === :call && f.args[1] === :surface_fresnel &&
                f.args[2] == :(F82Tint{SpectralRadiance})
            hits[] += 1
            return :($stored_fresnel($(f.args[3]), $(ex.args[3:end]...)))
        end
    end
    return Expr(ex.head, map(x -> replace_f82(x, hits), ex.args)...)
end

function check_coefficients()
    lambda = Hikari.Wavelengths((450f0, 510f0, 570f0, 630f0), (0.25f0, 0.25f0, 0.25f0, 0.25f0))
    @testset "prepared F82 coefficient preserves angular and film response" begin
        for tint in (0f0, 0.25f0, 0.7f0, 1f0), thickness in (0f0, 450f0), mu in Float32.(0:0.05:1)
            f = Hikari.F82Tint(Hikari.SpectralRadiance((0.1f0, 0.3f0, 0.7f0, 0.99f0)),
                Hikari.SpectralRadiance(tint), thickness, 1.3f0)
            data = Hikari.surface_data((0.2f0, 0.2f0, 1.5f0, 1f0, 0f0, fields(f)...))
            expected = Hikari.fresnel_rt(f, mu, 1.5f0, lambda)
            actual = stored_fresnel(data, mu, 1.5f0, lambda)
            @test actual[1].data == expected[1].data
            @test actual[2].data == expected[2].data
        end
    end
    return nothing
end

function run(output; spec, repeats = 11, names = ("mix_matte_metal", "mix_layered_glass"))
    fresnel_method = which(Hikari.fresnel_rt,
        Tuple{Hikari.SurfaceMicrofacetFresnel{UInt32(0), :reflection}, Float32, Float32, Hikari.Wavelengths})
    original = Main.Revise.CodeTracking.definition(Expr, fresnel_method)
    original isa Expr || error("Cannot restore shared Fresnel evaluation")
    hits = Ref(0)
    candidate_definition = replace_f82(original, hits)
    hits[] == 1 || error("Expected exactly one stored F82 evaluation, got $(hits[])")
    rows = Any[]
    for settings in filter(row -> row.name in names, spec.cases)
        fixtures = Any[]
        packing_method = nothing
        try
            setup_a = @timed MaterialComparison.fixture(MaterialComparison.principled_material(settings, spec), spec)
            push!(fixtures, setup_a.value)
            first_a = @timed Base.invokelatest(MaterialComparison.draw, setup_a.value)
            expected = MaterialComparison.pixels(setup_a.value)
            @eval Hikari @inline surface_fresnel_fields(f::F82Tint{SpectralRadiance}) = $fields(f)
            packing_method = which(Hikari.surface_fresnel_fields, Tuple{Hikari.F82Tint{Hikari.SpectralRadiance}})
            Core.eval(Hikari, candidate_definition)
            setup_b = @timed Base.invokelatest(MaterialComparison.fixture,
                MaterialComparison.principled_material(settings, spec), spec)
            push!(fixtures, setup_b.value)
            first_b = @timed Base.invokelatest(MaterialComparison.draw, setup_b.value)
            actual = MaterialComparison.pixels(setup_b.value)
            @test all(isfinite, actual)
            @test actual ≈ expected rtol = 5f-5 atol = 2f-5
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
            packing_method === nothing || Base.delete_method(packing_method)
            Core.eval(Hikari, original)
            foreach(MaterialComparison.release, fixtures)
        end
    end
    return nothing
end
end
