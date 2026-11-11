module CacheLookupTrace
using Hikari, Mantle, JSON3, SHA
using ..MaterialComparison
const L = Mantle.Lava

function run(output; device)
    Mantle.defaultdevice!(device)
    lookups = Any[]
    current = Ref("")
    hook = job -> push!(lookups, (; material = current[], job, before = L.compile_stats()))
    L.reset_compile_stats!()
    L.GPUCompiler.with_compile_hook(hook) do
        for row in MaterialComparison.configuration().cases
            current[] = String(row.name)
            f = Mantle.no_pipeline_compilation(; ctx = device) do
                MaterialComparison.fixture(MaterialComparison.principled_material(row),
                                           MaterialComparison.configuration())
            end
            try
                Mantle.no_pipeline_compilation(; ctx = device) do
                    MaterialComparison.draw(f)
                end
                expected = MaterialComparison.pixels(f)
                MaterialComparison.draw(f)
                @assert MaterialComparison.pixels(f) == expected
            finally
                MaterialComparison.release(f)
            end
        end
    end
    final = L.compile_stats()
    records = Any[]
    for i in eachindex(lookups)
        entry = lookups[i]
        after = i == length(lookups) ? final : lookups[i+1].before
        misses = after.misses - entry.before.misses
        sig = string(entry.job.source.specTypes)
        push!(records, (; material = entry.material, name = string(entry.job.source.def.name),
            signature_hash = bytes2hex(sha256(sig)), misses,
            signature = misses > 0 ? sig : nothing))
        misses > 0 && println("CACHE MISS ", entry.material, " ", entry.job.source.def.name)
    end
    mkpath(dirname(output))
    open(io -> JSON3.write(io, records), output, "w")
    println((lookups = length(lookups), stats = final))
    return records
end

function fingerprints(output; device)
    Mantle.defaultdevice!(device)
    jobs = Any[]
    row = only(filter(r -> r.name == "plastic", MaterialComparison.configuration().cases))
    f = MaterialComparison.fixture(MaterialComparison.principled_material(row),
                                   MaterialComparison.configuration())
    try
        L.GPUCompiler.with_compile_hook(job -> push!(jobs, job)) do
            Mantle.no_pipeline_compilation(() -> MaterialComparison.draw(f); ctx = device)
        end
    finally
        MaterialComparison.release(f)
    end
    records = Any[]
    for job in unique(j -> j.source.specTypes, jobs)
        key = L.invoke_frozen(L.inferred_kernel_key, job)
        sig = string(job.source.specTypes)
        push!(records, (; name = string(job.source.def.name),
            signature = sig, signature_hash = bytes2hex(sha256(sig)),
            key = key === nothing ? nothing : bytes2hex(key)))
    end
    open(io -> JSON3.write(io, records), output, "w")
    println((jobs = length(jobs), signatures = length(records)))
    return records
end
end
