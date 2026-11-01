# Diagnostic only: trim a CPU-prepared BSDF to its active count. This deliberately
# changes the capacity specialization, so it is not a renderer update strategy.
# Load metal_stages.jl and metal_prebuilt.jl into Main first.
module MetalActiveCapacityProbe
using Hikari, Mantle, Metal, Test, Statistics, JSON3
using Main.MetalBSDFPrebuiltProbe: prebuilt, prebuilt_kernel!

function trim_capacity(bsdf)
    span = bsdf.lobes
    n = Int(span.count)
    @assert all(w -> w.sample_weight > 0, span.data[1:n])
    @assert all(w -> w.sample_weight == 0, span.data[(n + 1):end])
    trimmed = Hikari.SurfaceLobeSpan(ntuple(i -> span.data[i], n), span.count)
    return Hikari.LobeBSDF(trimmed, bsdf.total, bsdf.lambda, bsdf.emission, bsdf.dispersive)
end

function run(; count = 65536, pairs = 51, result_path = nothing)
    rows = []
    dev = Mantle.Device()
    arrays = [Mantle.devicearray(dev, Float32, (count, 16)) for _ in 1:3]
    try
        for (name, material) in (
                ("Principled", Hikari.Principled(roughness = 0.3, metallic = 0.35, coat_weight = 0.4, sheen_weight = 0.25)),
                ("Glass", Hikari.Glass(roughness = 0.2)),
            )
            typed, shared = prebuilt(material)
            trimmed = trim_capacity(shared)
            for stage in 1:2
                println("ACTIVE_CAPACITY_BEGIN ", name, " ", stage)
                flush(stdout)
                launches = map(arrays, (typed, shared, trimmed)) do out, bsdf
                    () -> begin
                        prebuilt_kernel!(dev, 64)(out, bsdf, Val(stage); ndrange = count)
                        Mantle.waitidle(dev)
                    end
                end
                foreach(f -> f(), launches)
                expected = Array(first(arrays))
                for output in arrays[2:3]
                    actual = Array(output)
                    @test all(isfinite, actual)
                    @test actual ≈ expected rtol = 5.0f-5 atol = 2.0f-5
                end
                for _ in 1:5
                    foreach(f -> f(), launches)
                end
                times = (Float64[], Float64[], Float64[])
                for i in 1:pairs, k in (isodd(i) ? (1, 2, 3) : (3, 2, 1))
                    push!(times[k], @elapsed launches[k]())
                end
                profiles = (Any[], Any[], Any[])
                for block in 1:6, k in (isodd(block) ? (1, 2, 3) : (3, 2, 1))
                    launch = launches[k]
                    p = Metal.@profile for _ in 1:5
                        launch()
                    end
                    push!(profiles[k], (; device = p.device, wall = p.wall))
                end
                gpu = map(profiles) do ps
                    median([b - a for p in ps for (a, b) in zip(p.device.start, p.device.stop)])
                end
                row = (;
                    name, stage, count, active = shared.lobes.count,
                    capacity = length(shared.lobes.data), original_bytes = sizeof(shared.lobes),
                    trimmed_bytes = sizeof(trimmed.lobes), wall = map(median, times), gpu,
                    trimmed_over_shared = gpu[3] / gpu[2], profiles,
                )
                push!(rows, row)
                println("ACTIVE_CAPACITY_RESULT ", JSON3.write((;
                    name, stage, active = row.active, wall = row.wall, gpu,
                    trimmed_over_shared = row.trimmed_over_shared,
                )))
                flush(stdout)
                if result_path !== nothing
                    mkpath(dirname(result_path))
                    write(result_path, JSON3.write(rows))
                end
            end
        end
    finally
        foreach(Mantle.unsafe_free!, arrays)
    end
    return rows
end
end
