module MetalBSDFStageProbe
using Hikari, Mantle, Metal, KernelAbstractions, Raycore, GeometryBasics, LinearAlgebra, Test, Statistics, JSON3
using GeometryBasics: Point2f, Point3f, Vec3f

@inline function context(j)
    x = Float32(j % 1024) / 1024.0f0
    wo = normalize(Vec3f(0.2f0, 0.1f0, 0.3f0 + x))
    tfc = Hikari.TextureFilterContext(Point2f(0.3, 0.6), Point3f(0.3, -0.2, 0.7), Vec3f(0, 0, 1))
    sc = Hikari.ShaderContext(tfc, wo, Vec3f(1.0f-3, 0, 0), Vec3f(0, 1.0f-3, 0), Vec3f(1, 0, 0), Vec3f(0, 0, 1), 1.0f0, Vec3f(0), Point2f(0))
    lambda = Hikari.Wavelengths((450.0f0 + x, 510.0f0 + x, 570.0f0 + x, 630.0f0 + x), (0.25f0, 0.25f0, 0.25f0, 0.25f0))
    return sc, lambda, x
end
@inline function build(::Val{false}, ::Val{M}, ::Val{P}, stack, prog, sc, lut, table, lambda) where {M, P}
    return Hikari.material_bsdf(M, Hikari.node_lobes(M, stack, prog, Int32(1), sc, lut), table, lambda)
end
@inline function build(::Val{true}, ::Val{M}, ::Val{P}, stack, prog, sc, lut, table, lambda) where {M, P}
    return Hikari.surface_program_bsdf(P, stack, prog, sc, lut, table, lambda)
end
@inline function metrics(bsdf, sc, lambda, x, ::Val{Stage}) where {Stage}
    if Stage == 0
        # Only setup data observed here; LLVM may omit unused scattering fields.
        return (bsdf.total, bsdf.emission.data..., Float32(bsdf.dispersive), ntuple(_ -> 0.0f0, Val(10))...)
    end
    f, pdf = Hikari.mixture(bsdf.lobes, sc.wo, normalize(Vec3f(-0.3, 0.2, 1)), sc.tfc.n, lambda)
    if Stage == 1
        return (f.data..., pdf, ntuple(_ -> 0.0f0, Val(11))...)
    end
    sampled = Hikari.pick_sample(bsdf.lobes, x * bsdf.total, 0.0f0, sc.wo, Point2f(0.1f0 + x / 4, 0.2f0 + x / 4), lambda)
    return (f.data..., pdf, sampled[1]..., sampled[2].data..., sampled[3], Float32(sampled[4]), sampled[5], sampled[6])
end
@kernel inbounds = true function stage_kernel!(out, textures, values, table, program::Int32, shared, m, p, stage)
    j = @index(Global)
    if j <= size(out, 1)
        sc, lambda, x = context(j)
        prog = Hikari.closure_programs(textures)[program]
        stack = Hikari.GatheredStack(values, Int32(1), Int32(1))
        bsdf = build(shared, m, p, stack, prog, sc, Hikari.LobeSetup(Hikari.lookup_tables(textures), false), table, lambda)
        v = metrics(bsdf, sc, lambda, x, stage)
        for k in 1:16
            out[j, k] = v[k]
        end
    end
end

function run(; count = 65536, pairs = 31, result_path = nothing)
    rows = []
    for (name, material) in (("Principled", Hikari.Principled(roughness = 0.3, metallic = 0.35, coat_weight = 0.4, sheen_weight = 0.25)), ("Glass", Hikari.Glass(roughness = 0.2)))
        f = Main.SurfaceProgramGPUProbe.fixture(material; resolution = 16, samples = 1)
        arrays = []
        try
            Main.SurfaceProgramGPUProbe.draw(f)
            store = f.scene.materials
            mat = Raycore.@allowscalar first(first(store.static.data))
            textures = Hikari.closure_set(store.static)
            M = Hikari.bsdf_nodes(typeof(Hikari.surface_material(material)))
            P = Hikari.surface_program_class(M, Mantle.syncbackend(f.dev))
            a = Mantle.devicearray(f.dev, Float32, (count, 16));push!(arrays, a)
            b = Mantle.devicearray(f.dev, Float32, (count, 16));push!(arrays, b)
            values = Mantle.devicearray(f.dev, zeros(Float32, 1));push!(arrays, values)
            for stage in 0:2
                println("BSDF_STAGE ", name, " ", stage);flush(stdout)
                launches = map(((a, false), (b, true))) do (out, shared)
                    () -> begin
                        stage_kernel!(f.dev, 64)(out, textures, values, f.vp.state.rgb2spec_table, mat.closures, Val(shared), Val(M), Val(P), Val(stage); ndrange = count)
                        Mantle.waitidle(f.dev)
                    end
                end
                foreach(f -> f(), launches)
                expected, actual = Array(a), Array(b)
                @test all(isfinite, actual)
                @test actual ≈ expected rtol = 5.0f-5 atol = 2.0f-5
                for _ in 1:5
                    foreach(f -> f(), launches)
                end
                times = (Float64[], Float64[])
                for i in 1:pairs, k in (isodd(i) ? (1, 2) : (2, 1))
                    push!(times[k], @elapsed launches[k]())
                end
                profiles = map(launches) do launch
                    profile = Metal.@profile for _ in 1:5
                        launch()
                    end
                    (; device = profile.device, wall = profile.wall)
                end
                row = (; name, stage, count, reference = median(times[1]), shared = median(times[2]), ratio = median(times[2]) / median(times[1]), max_error = maximum(abs.(actual .- expected)), profiles)
                push!(rows, row);println(JSON3.write(row));flush(stdout)
                if result_path !== nothing
                    mkpath(dirname(result_path)); write(result_path, JSON3.write(rows))
                end
            end
        finally
            foreach(Mantle.unsafe_free!, arrays)
            Main.SurfaceProgramGPUProbe.release(f)
        end
    end
    return rows
end
end
