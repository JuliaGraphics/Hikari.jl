module MaterialCacheRestartProbe
using Hikari,Mantle,Test,JSON3,SHA
using ..MaterialComparison
const Probe=MaterialComparison

function run(output;device,require_hits=true,require_native_hits=false,
             save_native_cache=true,spec=Probe.configuration())
    Mantle.defaultdevice!(device)
    records=Any[]
    for row in spec.cases
        Mantle.Lava.reset_compile_stats!()
        setup=@timed Mantle.no_pipeline_compilation(;ctx=device) do
            Probe.fixture(Probe.principled_material(row,spec),spec)
        end
        native_setup_misses=device.diag.pipeline_compiles_refused
        setup_cache=Mantle.Lava.compile_stats()
        f=setup.value
        try
            Mantle.Lava.reset_compile_stats!()
            first=@timed Mantle.no_pipeline_compilation(;ctx=device) do
                Probe.draw(f)
            end
            native_first_misses=device.diag.pipeline_compiles_refused
            cache=Mantle.Lava.compile_stats()
            expected=Probe.pixels(f)
            Mantle.Lava.reset_compile_stats!()
            repeated=@timed Probe.draw(f)
            repeated_cache=Mantle.Lava.compile_stats()
            repeated_image_matches = Probe.pixels(f)==expected
            push!(records,(;material=row.name,setup_seconds=setup.time,setup_compile_seconds=setup.compile_time,
                setup_cache,native_setup_misses,native_first_misses,
                first_seconds=first.time,julia_compile_seconds=first.compile_time,
                cache,repeat_seconds=repeated.time,repeated_cache,repeated_image_matches,
                image_sha256=bytes2hex(sha256(reinterpret(UInt8,expected)))))
            mkpath(dirname(output))
            source_hash=Probe.source_hash()
            identities=[(;name=string(nameof(m)),build_id=string(Base.module_build_id(m)))
                        for m in (Hikari,Mantle,Mantle.Lava)]
            dependency_sources=[(;name=string(nameof(m)),sha256=Probe.source_hash(m))
                                for m in (Hikari,Mantle,Mantle.Lava)]
            open(io->JSON3.write(io,(;source_hash,identities,dependency_sources,
                device=Mantle.devicename(device),rows=records)),output,"w")
            @test repeated_cache.misses==0
            @test repeated_image_matches
            if require_hits
                @test setup_cache.misses==0
                @test cache.misses==0
            end
            if require_native_hits
                @test native_setup_misses==0
                @test native_first_misses==0
            end
            println(row.name," fresh-process first frame ",first.time," s; misses ",cache.misses)
            flush(stdout)
        finally
            Probe.release(f)
        end
    end
    # A persistent Julia seed has not exited: flush the driver's cache before
    # another process tests it, just as normal device shutdown does.
    if !require_hits && save_native_cache
        Mantle.save_pipeline_cache!(device)
    end
    nothing
end
end
