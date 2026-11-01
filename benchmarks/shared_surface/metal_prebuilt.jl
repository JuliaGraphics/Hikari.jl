module MetalBSDFPrebuiltProbe
using Hikari, Mantle, Metal, KernelAbstractions, Raycore, Test, Statistics, JSON3
using ..MetalBSDFStageProbe: context, metrics

function prebuilt(material)
    store = Raycore.MultiTypeSet(KernelAbstractions.CPU())
    mat = Hikari.to_device_material(store, Hikari.surface_material(material))
    textures = Raycore.StaticMultiTypeSet((), Tuple(T[a for a in store.texture_gpu_arrays if typeof(a) === T] for T in store.texture_order))
    prog = Hikari.shader_programs(textures, mat.program)[mat.program.idx]
    M = typeof(mat).parameters[3]
    P = Hikari.surface_program_class(M, Mantle.MetalAPI())
    sc, lambda, _ = context(512)
    stack, start = Hikari.run_values(prog, textures, sc)
    lut = Hikari.LobeSetup(Hikari.lookup_tables(textures), false)
    table = Hikari.get_srgb_table()
    original = Hikari.material_bsdf(M, Hikari.node_lobes(M, stack, prog, start, sc, lut), table, lambda)
    shared = Hikari.surface_program_bsdf(P, stack, prog, sc, lut, table, lambda, start)
    @assert isbitstype(typeof(original)) && isbitstype(typeof(shared))
    return original, shared
end

@kernel inbounds = true function prebuilt_kernel!(out, bsdf, stage)
    j = @index(Global)
    if j <= size(out, 1)
        sc, _, x = context(j)
        v = metrics(bsdf, sc, bsdf.lambda, x, stage)
        for k in 1:16
            out[j, k] = v[k]
        end
    end
end

function run(; count = 65536, pairs = 31, result_path = nothing)
    rows = [];dev = Mantle.Device()
    a = Mantle.devicearray(dev, Float32, (count, 16))
    b = Mantle.devicearray(dev, Float32, (count, 16))
    try
        for (name, m) in (("Principled", Hikari.Principled(roughness = 0.3, metallic = 0.35, coat_weight = 0.4, sheen_weight = 0.25)), ("Glass", Hikari.Glass(roughness = 0.2)))
            original, shared = prebuilt(m)
            for stage in 1:2
                println("PREBUILT_STAGE ", name, " ", stage);flush(stdout)
                launches = map(((a, original), (b, shared))) do (out, bsdf)
                    () -> begin
                        prebuilt_kernel!(dev, 64)(out, bsdf, Val(stage); ndrange = count)
                        Mantle.waitidle(dev)
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
                for i in 1:pairs,k in (isodd(i) ? (1, 2) : (2, 1))
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
        end
    finally
        Mantle.unsafe_free!(a);Mantle.unsafe_free!(b)
    end
    return rows
end
end
