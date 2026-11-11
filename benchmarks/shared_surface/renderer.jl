module SurfaceProgramGPUProbe
using Hikari, Mantle, KernelAbstractions, GeometryBasics, Raycore, Test, Statistics, JSON3
using Hikari.TextureNodes
using GeometryBasics: Point2f, Point3f, Vec3f, Sphere, Tesselation, normal_mesh
# Differential reference only: record a plan using the original typed formulas.
# Normal rendering always selects Hikari's integrated shared storage.
function draw_reference(f)
    @eval Hikari.bsdf_kernel_nodes(::Type{M}, ::Mantle.Backend) where {M <: Tuple} = M
    method = which(Hikari.bsdf_kernel_nodes, Tuple{Type{Tuple{}}, Mantle.VulkanAPI})
    try
        return Base.invokelatest(draw, f)
    finally
        Base.delete_method(method)
    end
end
function fixture(material; resolution = 96, samples = 8, hw_accel = false)
    backend = Mantle.defaultbackend()
    scene = Hikari.Scene(; backend, hw_accel)
    push!(scene, normal_mesh(Tesselation(Sphere(Point3f(0), 0.7f0), 32)), material)
    push!(scene, Hikari.PointLight(Point3f(-2, -2, 3), Hikari.RGBSpectrum(40.0f0)))
    Hikari.sync!(scene)
    film = Hikari.Film(backend, Hikari.Film(Point2f(resolution, resolution)))
    camera = Hikari.PerspectiveCamera(Point3f(0, -3, 1), Point3f(0), film; fov = 40.0f0)
    vp = Hikari.VolPath(; samples, max_depth = 4)
    return (; scene, film, camera, vp, dev = Hikari.mantle_device(backend))
end
draw(f) = (f.vp(f.scene, f.film, f.camera); Mantle.waitidle(f.dev); nothing)
image(f) = [Float32(c) for p in Array(f.film.framebuffer) for c in (p.r, p.g, p.b)]
release(f) = (close(f.vp); Hikari.free!(f.film); nothing)
signatures(f) = unique(
    [
        Mantle.devicetype(f.dev, d.args) for pl in Hikari.allplans(f.vp.state.plans)
            for pp in pl.passes if pp.pass.name == "shade" for d in pp.pass.dispatches
    ]
)
native_stats(f, ::Mantle.Backend) = nothing
function native_stats(f, ::Mantle.VulkanAPI)
    launches = unique(
        [
            d.launch for pl in Hikari.allplans(f.vp.state.plans)
                for pp in pl.passes if pp.pass.name == "shade" for d in pp.dispatches
        ]
    )
    return map(launches) do launch
        spirv = Mantle.spirv_op_stats(launch.compiled.spirv_bytes)
        native = Mantle.pipeline_exec_stats(f.dev, launch.pipeline)
        return (;
            pipeline_id = objectid(launch.pipeline), bytes = spirv.bytes,
            functions = spirv.n_functions,
            registers = native === nothing ? nothing : native.registers,
            scratch_bytes = native === nothing ? nothing : native.scratch_bytes,
        )
    end
end
function compare(material; resolution = 96, samples = 8, timing_samples = 0, failure_dump = nothing, hw_accel = false)
    original = fixture(material; resolution, samples, hw_accel);candidate = nothing
    try
        draw_reference(original);expected = image(original)
        reference_signatures = signatures(original)
        reference_native = native_stats(original, Mantle.syncbackend(original.dev))
        candidate = fixture(material; resolution, samples, hw_accel)
        firsttime = @elapsed Base.invokelatest(draw, candidate)
        actual = image(candidate)
        @test any(T -> occursin("SurfaceProgramNodes", string(T)), signatures(candidate))
        @test all(isfinite, actual)
        @test maximum(expected) > 0.0f0
        @test actual ≈ expected rtol = 5.0f-5 atol = 2.0f-5
        times = [Float64[], Float64[]]
        if timing_samples > 0
            for _ in 1:5,f in (original, candidate)
                Base.invokelatest(draw, f)
            end
            for i in 1:timing_samples,k in (isodd(i) ? (1, 2) : (2, 1))
                f = k == 1 ? original : candidate
                push!(times[k], @elapsed Base.invokelatest(draw, f))
            end
        end
        @test signatures(original) == reference_signatures
        @test native_stats(original, Mantle.syncbackend(original.dev)) == reference_native
        selected = Base.invokelatest(
            Hikari.bsdf_kernel_nodes,
            Hikari.bsdf_nodes(typeof(Hikari.surface_material(material))),
            Mantle.syncbackend(candidate.dev)
        )
        row = (;
            material = string(nameof(typeof(material))), resolution, samples, hw_accel,
            capacity = selected.parameters[1], looped = selected.parameters[2], mask = string(selected.parameters[3]),
            max_error = maximum(abs.(actual .- expected)), first_seconds = firsttime,
            reference_native, candidate_native = native_stats(candidate, Mantle.syncbackend(candidate.dev)),
            warm = timing_samples == 0 ? nothing : (; reference = median(times[1]), candidate = median(times[2]), ratio = median(times[2]) / median(times[1])),
        )
        println(JSON3.write(row));return row
    catch
        if failure_dump !== nothing && candidate !== nothing && candidate.vp.state.plans !== nothing
            mkpath(failure_dump)
            launches = unique([d.launch for pl in Hikari.allplans(candidate.vp.state.plans) for pp in pl.passes if pp.pass.name == "shade" for d in pp.dispatches])
            for (index, launch) in enumerate(launches)
                write(joinpath(failure_dump, "shade_$index.spv"), launch.compiled.spirv_bytes)
            end
            println("Saved failed shade pipelines to ", failure_dump)
        end
        rethrow()
    finally
        candidate === nothing||release(candidate);release(original)
    end
end
end
