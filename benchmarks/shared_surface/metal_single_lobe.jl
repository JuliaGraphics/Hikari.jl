module MetalSingleLobeProbe
using Hikari, Mantle, Metal, KernelAbstractions, Raycore, Test, Statistics, JSON3
using Main.MetalBSDFStageProbe: context, metrics

# Diagnostic restricted to one active Diffuse/Sheen/Glossy term. Runtime
# tags share one capability kernel; construct each concrete BSDF inside
# its branch so inference does not widen a tuple containing a union.
@inline function decoded_weight(w, v)
    return Hikari.Weighted(Hikari.surface_restore(w.lobe, v), w.weight, w.sample_weight, w.n, w.t)
end
@inline function decoded_bsdf(bsdf, v)
    rw = decoded_weight(bsdf.lobes.data[1], v)
    return Hikari.LobeBSDF((rw,), bsdf.total, bsdf.lambda, bsdf.emission, bsdf.dispersive)
end
@inline function decode_one(bsdf)
    kind = bsdf.lobes.data[1].lobe.kind
    return kind == 1 ? decoded_bsdf(bsdf, Val(1)) :
        kind == 2 ? decoded_bsdf(bsdf, Val(2)) : decoded_bsdf(bsdf, Val(3))
end

@kernel inbounds = true function decode_kernel!(out, textures, values, table, programs, ::Val{Decode}) where {Decode}
    j = @index(Global)
    if j <= size(out, 1)
        sc, lambda, x = context(j)
        # Per-lane indirection models the renderer's material-instance lookup.
        prog = Hikari.closure_programs(textures)[programs[j]]
        p = Hikari.SurfaceProgramNodes{1, true, Hikari.SURFACE_SIMPLE_MASK, false}
        bsdf = Hikari.surface_program_bsdf(
            p, Hikari.GatheredStack(values, Int32(1), Int32(1)), prog, sc,
            Hikari.LobeSetup(Hikari.lookup_tables(textures), false), table, lambda
        )
        Decode && (bsdf = decode_one(bsdf))
        v = metrics(bsdf, sc, lambda, x, Val(2))
        for k in 1:16
            out[j, k] = v[k]
        end
    end
end

function run(; count = 65536, pairs = 51, result_path = nothing)
    rows = []
    for (index, (name, m)) in enumerate(
            (
                ("Diffuse", Hikari.Diffuse(color = (0.4, 0.2, 0.1), roughness = 0.3)),
                ("Sheen", Hikari.Sheen(color = (0.4, 0.2, 0.1))),
                ("Glossy", Hikari.Glossy(color = (0.4, 0.2, 0.1), roughness = 0.3)),
            )
        )
        f = Main.SurfaceProgramGPUProbe.fixture(m; resolution = 16, samples = 1);arrays = []
        try
            Main.SurfaceProgramGPUProbe.draw(f)
            store = f.scene.materials
            mat = Raycore.@allowscalar first(first(store.static.data))
            textures = Hikari.closure_set(store.static)
            a = Mantle.devicearray(f.dev, Float32, (count, 16));push!(arrays, a)
            b = Mantle.devicearray(f.dev, Float32, (count, 16));push!(arrays, b)
            values = Mantle.devicearray(f.dev, zeros(Float32, 1));push!(arrays, values)
            programs = Mantle.devicearray(f.dev, fill(mat.closures, count));push!(arrays, programs)
            launches = map(((a, false), (b, true))) do (out, uniform)
                () -> begin
                    decode_kernel!(f.dev, 64)(out, textures, values, f.vp.state.rgb2spec_table, programs, Val(uniform); ndrange = count)
                    Mantle.waitidle(f.dev)
                end
            end
            Metal.reset_compile_stats!()
            foreach(f -> f(), launches)
            compiled = Metal.compile_stats()
            if index > 1
                @test compiled.misses == 0
            end
            expected, actual = Array(a), Array(b)
            @test all(isfinite, actual)
            @test actual ≈ expected rtol = 5.0f-5 atol = 2.0f-5
            for _ in 1:5
                foreach(f -> f(), launches)
            end
            times = (Float64[], Float64[])
            for i in 1:pairs,k in (isodd(i) ? (1, 2) : (2, 1))
                push!(times[k], @elapsed launches[k]())
            end
            profiles = (Any[], Any[])
            for block in 1:6, k in (isodd(block) ? (1, 2) : (2, 1))
                launch = launches[k]
                profile = Metal.@profile for _ in 1:5
                    launch()
                end
                push!(profiles[k], (; device = profile.device, wall = profile.wall))
            end
            row = (; name, count, profiles, compiled, baseline = median(times[1]), decoded = median(times[2]), ratio = median(times[2]) / median(times[1]), max_error = maximum(abs.(actual .- expected)))
            push!(rows, row);println("DECODE_ONE ", JSON3.write((; name, ratio = row.ratio, max_error = row.max_error)));flush(stdout)
            if result_path !== nothing
                mkpath(dirname(result_path));write(result_path, JSON3.write(rows))
            end
        finally
            foreach(Mantle.unsafe_free!, arrays)
            Main.SurfaceProgramGPUProbe.release(f)
        end
    end
    return rows
end
end
