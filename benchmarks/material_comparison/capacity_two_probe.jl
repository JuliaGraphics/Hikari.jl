module CapacityTwoProbe
using Hikari, Mantle, JSON3, Test, Statistics, GeometryBasics
using ..MaterialComparison
const Probe = MaterialComparison

function smaller_queue(ex, capacity)
    ex isa Expr || return ex
    if ex.head === :curly && first(ex.args) === :SurfaceProgramNodes && ex.args[2] == 5
        ex.args[2] = capacity
    end
    ex.args = map(x -> smaller_queue(x, capacity), ex.args)
    return ex
end

function smaller_mixture(ex)
    ex isa Int && return ex == 5 ? 2 : ex == 6 ? 3 : ex
    ex isa Expr || return ex
    return Expr(ex.head, map(smaller_mixture, ex.args)...)
end

function resources(f)
    launches=unique([d.launch for pl in Hikari.allplans(f.vp.state.plans)
        for pp in pl.passes if pp.pass.name=="shade" for d in pp.dispatches])
    return [(;bytes=length(l.compiled.spirv_bytes),native=Mantle.pipeline_exec_stats(f.dev,l.pipeline))
        for l in launches]
end

# This is a storage-size control for fixed inputs, not a production classifier.
# These fixtures have no transmission, subsurface, coat or sheen. The maximum
# is diffuse + specular, or one metallic lobe. Never apply it to other graphs.
function run(output; device, repeats = 11, capacity = 2)
    @assert capacity in (1,2)
    program=Hikari.closure_program(Hikari.compiled(Hikari.surface_material(Hikari.Principled()).program))[1]
    Hikari.closure_capacity_hint(first(program)) == 2 && error(
        "This historical storage control requires source before two-record routing was integrated.")
    Mantle.defaultdevice!(device)
    method = only(filter(m -> occursin("SurfaceProgramNodes", string(m.sig)), methods(Hikari.shade_queues)))
    original = Main.Revise.CodeTracking.definition(Expr, method)
    W = Hikari.SurfaceStoredWeight{true, Hikari.SURFACE_PRINCIPLED_MASK}
    mix = Main.Revise.CodeTracking.definition(Expr, which(Hikari.surface_mixture,
        Tuple{NTuple{5,W},Int32,Vec3f,Vec3f,Vec3f,Hikari.Wavelengths,Hikari.AnyScattering}))
    original isa Expr && mix isa Expr || error("Cannot restore capacity control")
    spec = Probe.configuration()
    rows = Any[]
    for name in (capacity == 1 ? ("matte", "rough_metal") : ("matte", "rough_metal", "plastic"))
        row = only(filter(r -> r.name == name, spec.cases))
        material = Probe.principled_material(row)
        for field in (:coat_weight, :sheen_weight, :transmission_weight, :subsurface_weight)
            @assert getfield(material, field) == 0
        end
        @assert material.metallic in (0,1)
        if capacity == 1
            @assert material.metallic == 1 || material.specular_ior_level == 0
        end
        baseline = Probe.fixture(material, spec)
        candidate = nothing
        mix_method = nothing
        try
            Base.invokelatest(Probe.draw, baseline)
            expected = Probe.pixels(baseline)
            Core.eval(Hikari, smaller_queue(deepcopy(original), capacity))
            if capacity == 2
                Core.eval(Hikari, smaller_mixture(deepcopy(mix)))
                mix_method = which(Hikari.surface_mixture,
                    Tuple{NTuple{2,W},Int32,Vec3f,Vec3f,Vec3f,Hikari.Wavelengths,Hikari.AnyScattering})
            end
            candidate = Base.invokelatest(Probe.fixture, material, spec)
            first_seconds = @elapsed Base.invokelatest(Probe.draw, candidate)
            actual = Probe.pixels(candidate)
            @test all(isfinite, actual)
            @test actual ≈ expected rtol = 5f-5 atol = 2f-5
            @test any(s -> occursin("SurfaceProgramNodes{$capacity,", s), Probe.shade_signatures(candidate))
            @test !any(s -> occursin("SurfaceProgramNodes{5,", s), Probe.shade_signatures(candidate))
            times = (Float64[], Float64[])
            for _ in 1:5, f in (baseline, candidate)
                Base.invokelatest(Probe.draw, f)
            end
            for i in 1:repeats, k in (isodd(i) ? (1,2) : (2,1))
                push!(times[k], @elapsed Base.invokelatest(Probe.draw, k == 1 ? baseline : candidate))
            end
            record = (; material = name, capacity, first_seconds, median_seconds = map(median, times),
                frame_seconds = times, max_pixel_error = maximum(abs.(actual .- expected)),
                baseline_resources=resources(baseline),candidate_resources=resources(candidate))
            push!(rows, record)
            mkpath(dirname(output))
            open(io -> JSON3.write(io, rows), output, "w")
            println(name, " capacity five/", capacity, " ms: ", record.median_seconds .* 1000,
                    "; image error: ", record.max_pixel_error)
            flush(stdout)
        finally
            Core.eval(Hikari, original)
            mix_method === nothing || Base.delete_method(mix_method)
            candidate === nothing || Probe.release(candidate)
            Probe.release(baseline)
        end
    end
    return nothing
end
end
