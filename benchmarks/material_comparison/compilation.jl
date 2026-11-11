module CompilationComparison
using Hikari, Mantle, JSON3, Test
using SHA: sha256
using ..MaterialComparison
const Probe = MaterialComparison
const Compiler = Mantle.Lava

struct Capture
    lookups::Vector{Any}
    pipelines::Vector{Any}
end
Capture() = Capture(Any[], Any[])

# Session-only timers around the original methods, invoked in their original world.
# The exact tracked method definitions are restored before this benchmark returns.
function instrument!(capture, cache_dir)
    lookup = which(Compiler.compile_or_lookup, Tuple{Compiler.LavaCompilerJob})
    pipeline = which(Mantle.create_compute_pipeline, Tuple{Mantle.LavaDevice, Mantle.VK.ComputePipelineCreateInfo})
    path = which(Mantle.lava_pipeline_cache_path, Tuple{String,String})
    definitions = [(Compiler, Main.Revise.CodeTracking.definition(Expr, lookup)),
                   (Mantle, Main.Revise.CodeTracking.definition(Expr, pipeline)),
                   (Mantle, Main.Revise.CodeTracking.definition(Expr, path))]
    @assert all(item -> item[2] isa Expr, definitions)
    world = Base.get_world_counter()
    Core.eval(Compiler, quote
        @noinline function compile_or_lookup(job::LavaCompilerJob)::LavaGPUKernel
            before = compile_stats()
            t0 = time_ns()
            result = Base.invoke_in_world($world, compile_or_lookup, job)
            seconds = (time_ns() - t0) / 1e9
            after = compile_stats()
            push!($(capture.lookups), (; signature=string(job.source.specTypes), seconds,
                                        hits=after.hits-before.hits, misses=after.misses-before.misses))
            return result
        end
    end)
    Core.eval(Mantle, quote
        function create_compute_pipeline(ctx::LavaDevice, ci::VK.ComputePipelineCreateInfo)
            jit0 = Base.cumulative_compile_time_ns()
            t0 = time_ns()
            result = Base.invoke_in_world($world, create_compute_pipeline, ctx, ci)
            seconds = (time_ns() - t0) / 1e9
            jit = Base.cumulative_compile_time_ns() .- jit0
            push!($(capture.pipelines), (; seconds, julia_compile_seconds=jit[1]/1e9,
                                          julia_recompile_seconds=jit[2]/1e9))
            return result
        end
        lava_pipeline_cache_path(::AbstractString, ::AbstractString) = $(joinpath(cache_dir,"vulkan-pipelines.bin"))
    end)
    return definitions
end

function measure(action, capture)
    empty!(capture.lookups)
    empty!(capture.pipelines)
    Compiler.reset_compile_stats!()
    Compiler.reset_phase_log!()
    timing = @timed Base.invokelatest(action)
    data = Dict{String,Any}(
        "wall_seconds" => timing.time, "julia_compile_seconds" => timing.compile_time,
        "julia_recompile_seconds" => timing.recompile_time, "gc_seconds" => timing.gctime,
        "spirv_cache" => Compiler.compile_stats(), "llvm_cache" => Compiler.llvm_compile_stats(),
        "function_cache" => Compiler.function_compile_stats(),
        "compiler_lookups" => copy(capture.lookups), "pipeline_creations" => copy(capture.pipelines),
        "phase_table" => Compiler.phase_table(),
        "phase_records" => [(; r.group, r.label, r.seconds, r.kernel) for r in Compiler.phase_records()])
    data["compiler_lookup_wall_seconds"] = sum(r.seconds for r in capture.lookups; init=0.0)
    data["compiler_miss_wall_seconds"] = sum(r.seconds for r in capture.lookups if r.misses > 0; init=0.0)
    data["pipeline_creation_wall_seconds"] = sum(r.seconds for r in capture.pipelines; init=0.0)
    return timing.value, data
end

function run(output, implementation::Symbol, cache_state::Symbol)
    @assert implementation in (:original_hikari, :typed_cycles, :current_cycles)
    @assert cache_state in (:cold, :restart)
    cache_dir = ENV["JULIA_OBJCACHE_PATH"]
    mkpath(output)
    mkpath(cache_dir)
    capture = Capture()
    definitions = instrument!(capture, cache_dir)
    capturing = Compiler.enable_phase_capture!()
    spec = Probe.configuration()
    result = Dict{String,Any}("implementation" => string(implementation),
                              "cache_state" => string(cache_state), "cache_dir" => cache_dir,
                              "julia" => string(VERSION), "rows" => Any[])
    try
        device = Base.invokelatest(Mantle.defaultdevice!, "RTX 4000 Ada")
        result["device"] = Mantle.devicename(device)
        result["hikari_commit"] = strip(read(Cmd(["git","-C",pkgdir(Hikari),"rev-parse","HEAD"]),String))
        result["hikari_source_diff_sha256"] = bytes2hex(sha256(read(Cmd(["git","-C",pkgdir(Hikari),"diff","--","src"]))))
        result["hikari_source_sha256"] = Probe.source_hash()
        result["pipeline_cache_loaded_bytes"] = filesize_or_zero(joinpath(cache_dir,"vulkan-pipelines.bin"))
        result["initial_native_cache_bytes"] = let (size, ptr) = Mantle.VK.unwrap(Mantle.VK.get_pipeline_cache_data(device.device,device.pipeline_cache))
            try
                Int(size)
            finally
                Libc.free(ptr)
            end
        end
        for row in spec.cases
            implementation == :original_hikari && row.original === nothing && continue
            println("COMPILE ", implementation, " / ", cache_state, " / ", row.name); flush(stdout)
            material = implementation == :original_hikari ? Probe.original_material(row.original) : Probe.principled_material(row)
            fixture, setup = measure(() -> Probe.fixture(material,spec),capture)
            record = Dict{String,Any}("material" => String(row.name), "setup" => setup)
            try
                _, first = measure(() -> Probe.first_draw(fixture,implementation),capture)
                image = Probe.pixels(fixture)
                @test all(isfinite,image) && maximum(image) > 0
                if isempty(result["rows"])
                    launches=unique([d.launch for pl in Hikari.allplans(fixture.vp.state.plans)
                        for pp in pl.passes if pp.pass.name=="shade" for d in pp.dispatches])
                    for (index,launch) in enumerate(launches)
                        write(joinpath(output,"shade_$index.spv"),launch.compiled.spirv_bytes)
                    end
                end
                record["first_render"] = first
                record["shade_signatures"] = Probe.shade_signatures(fixture)
                _, repeat = measure(() -> Probe.draw(fixture),capture)
                record["same_session_repeat"] = repeat
                @test repeat["spirv_cache"].misses == 0
                @test isempty(repeat["pipeline_creations"])
                push!(result["rows"],record)
                open(io -> JSON3.write(io,result),joinpath(output,"$cache_state.json"),"w")
                println("  first=",round(first["wall_seconds"];digits=3),
                        "s; compiler lookups=",round(first["compiler_lookup_wall_seconds"];digits=3),
                        "s; pipelines=",round(first["pipeline_creation_wall_seconds"];digits=3),
                        "s; misses=",first["spirv_cache"].misses)
                flush(stdout)
            finally
                Probe.release(fixture)
            end
        end
        Base.invokelatest(Mantle.save_pipeline_cache!,device)
        result["pipeline_cache_saved_bytes"] = filesize_or_zero(joinpath(cache_dir,"vulkan-pipelines.bin"))
        open(io -> JSON3.write(io,result),joinpath(output,"$cache_state.json"),"w")
    finally
        Compiler.enable_phase_capture!(capturing)
        for (mod, definition) in reverse(definitions)
            Core.eval(mod,definition)
        end
    end
    return nothing
end
filesize_or_zero(path) = isfile(path) ? filesize(path) : 0
end
