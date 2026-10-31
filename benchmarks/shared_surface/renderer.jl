module SurfaceProgramGPUProbe
using Hikari, Mantle, KernelAbstractions, GeometryBasics, Raycore, Test, Statistics, JSON3
using Hikari.TextureNodes
using GeometryBasics: Point2f, Point3f, Vec3f, Sphere, Tesselation, normal_mesh
function install!(; force_looped = false)
    @eval Hikari function bsdf_kernel_nodes(::Type{M}, api::Mantle.Backend) where {M <: Tuple}
        selected = surface_program_class(M, api)
        if selected !== nothing && $force_looped
            N, _, mask, emission = selected.parameters
            return SurfaceProgramNodes{N, true, mask, emission}
        end
        return selected === nothing ? invoke(bsdf_kernel_nodes, Tuple{Type{M}, Any}, M, api) : selected
    end
    return which(Hikari.bsdf_kernel_nodes, Tuple{Type{Tuple{Hikari.PrincipledBSDF{false, false, false}}}, Mantle.VulkanAPI})
end
function fixture(material; resolution = 96, samples = 8)
    backend = Mantle.defaultbackend()
    scene = Hikari.Scene(; backend)
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
function compare(material; resolution = 96, samples = 8, timing_samples = 0, force_looped = false)
    original = fixture(material; resolution, samples);candidate = nothing;method = nothing
    try
        draw(original);expected = image(original)
        reference_signatures = signatures(original)
        reference_native = native_stats(original, Mantle.syncbackend(original.dev))
        method = install!(; force_looped)
        candidate = fixture(material; resolution, samples)
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
            material = string(nameof(typeof(material))), resolution, samples,
            capacity = selected.parameters[1], looped = selected.parameters[2], mask = string(selected.parameters[3]),
            max_error = maximum(abs.(actual .- expected)), first_seconds = firsttime,
            reference_native, candidate_native = native_stats(candidate, Mantle.syncbackend(candidate.dev)),
            warm = timing_samples == 0 ? nothing : (; reference = median(times[1]), candidate = median(times[2]), ratio = median(times[2]) / median(times[1])),
        )
        println(JSON3.write(row));return row
    finally
        method === nothing||Base.delete_method(method)
        candidate === nothing||release(candidate);release(original)
    end
end
end
