"""Backend-independent paired timings; load `hikari.jl` first for the shared fixture."""
module PairedMaterialRuntime
using Hikari, Mantle, JSON3, Statistics, SHA, Test
using ..MaterialComparison

function run(output; spec = MaterialComparison.configuration(), select = nothing,
             repeats = Int(spec.repeats), warmups = Int(spec.warmups))
    select === nothing || Mantle.defaultdevice!(select)
    device = Mantle.Device(Mantle.defaultbackend())
    rows = Any[]
    for row in spec.cases
        fixtures = Any[]
        setup_times, first_times, compile_times = Float64[], Float64[], Float64[]
        try
            for implementation in (:typed_cycles, :current_cycles)
                setup = @timed MaterialComparison.fixture(
                    MaterialComparison.principled_material(row, spec), spec)
                push!(fixtures, setup.value)
                first = @timed Base.invokelatest(MaterialComparison.first_draw,
                                                 setup.value, implementation)
                push!(setup_times, setup.time)
                push!(first_times, first.time)
                push!(compile_times, first.compile_time)
            end
            images = MaterialComparison.pixels.(fixtures)
            @test all(image -> all(isfinite, image) && maximum(image) > 0, images)
            @test images[1] ≈ images[2] rtol = 5f-5 atol = 2f-5
            for _ in 1:warmups, f in fixtures
                Base.invokelatest(MaterialComparison.draw, f)
            end
            times = (Float64[], Float64[])
            for i in 1:repeats, j in (isodd(i) ? (1, 2) : (2, 1))
                push!(times[j], @elapsed Base.invokelatest(MaterialComparison.draw, fixtures[j]))
            end
            for j in 1:2
                @test MaterialComparison.pixels(fixtures[j]) == images[j]
            end
            record = (; material = row.name, setup_seconds = setup_times,
                first_seconds = first_times, julia_compile_seconds = compile_times,
                typed_seconds = median(times[1]), current_seconds = median(times[2]),
                ratio = median(times[2]) / median(times[1]),
                max_pixel_error = maximum(abs.(images[1] .- images[2])),
                typed_frame_seconds = times[1], current_frame_seconds = times[2],
                image_sha256 = map(image -> bytes2hex(sha256(reinterpret(UInt8, image))), images))
            push!(rows, record)
            println((; record.material, record.typed_seconds, record.current_seconds, record.ratio))
            flush(stdout)
            mkpath(dirname(output))
            open(output, "w") do io
                JSON3.write(io, (; device = Mantle.devicename(device), resolution = spec.resolution,
                    samples = spec.samples, max_depth = spec.max_depth, warmups, repeats, rows))
            end
        finally
            foreach(MaterialComparison.release, fixtures)
        end
    end
    return nothing
end
end
