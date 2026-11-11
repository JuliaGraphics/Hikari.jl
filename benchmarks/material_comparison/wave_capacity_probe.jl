module WaveCapacityProbe
using Hikari, Mantle, JSON3, Test, Statistics
using ..MaterialComparison
const Probe = MaterialComparison

function run(output;device,repeats=15,capacities=(1<<19,1<<20,1<<21),
             names=("matte","rough_glass","layered_principled"), resolution=Probe.configuration().resolution)
    Mantle.defaultdevice!(device)
    original = Main.Revise.CodeTracking.definition(Expr,which(Hikari.wave_path_limit,Tuple{typeof(device)}))
    original isa Expr || error("Cannot recover wave_path_limit for restoration")
    spec = Probe.configuration()
    records = Any[]
    try
        for name in names
            row = only(filter(r->r.name==name,spec.cases))
            f = Probe.fixture(Probe.principled_material(row),spec;resolution)
            expected = nothing
            try
                for capacity in capacities
                    Core.eval(Hikari, :(wave_path_limit(dev) = $capacity))
                    before = Mantle.Lava.compile_stats()
                    first_seconds = @elapsed Base.invokelatest(Probe.draw,f)
                    after = Mantle.Lava.compile_stats()
                    actual = Probe.pixels(f)
                    @test all(isfinite,actual)
                    expected === nothing && (expected=actual)
                    @test actual≈expected rtol=5f-5 atol=2f-5
                    for _ in 1:5
                        Base.invokelatest(Probe.draw,f)
                    end
                    times = [@elapsed Base.invokelatest(Probe.draw,f) for _ in 1:repeats]
                    @test Probe.pixels(f)≈actual rtol=5f-5 atol=2f-5
                    bytes = sum(b->length(Mantle.storage(b))*sizeof(eltype(Mantle.storage(b))),f.vp.state.memory.owned;init=0)
                    result = (;name,resolution,capacity,lanes=Int(f.vp.state.sobol_rng.lanes),first_seconds,
                               spirv_misses=after.misses-before.misses,frame_seconds=times,median_seconds=median(times),
                               logical_buffer_bytes=bytes,max_pixel_error=maximum(abs.(actual.-expected)))
                    push!(records,result)
                    mkpath(dirname(output))
                    open(io->JSON3.write(io,records),output,"w")
                    println(name," / ",result.lanes," lanes: ",result.median_seconds*1000,
                            " ms; buffers ",bytes," bytes; image error ",result.max_pixel_error)
                    flush(stdout)
                end
            finally
                Probe.release(f)
            end
        end
    finally
        Core.eval(Hikari,original)
    end
    nothing
end
end
