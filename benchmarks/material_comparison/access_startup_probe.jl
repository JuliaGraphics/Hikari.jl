module AccessStartupProbe
using Hikari, Mantle, Test, JSON3, Profile, Serialization, SHA
using ..MaterialComparison
const Probe = MaterialComparison

function access_kernel!(out, src, scale)
    @inbounds for i in eachindex(out)
        out[i] = muladd(src[i], scale, out[i])
    end
    return nothing
end

function prime()
    A = Mantle.Lava.LavaDeviceArray{Float32,1}
    interp = Mantle.Lava.kernelinterpreter(access_kernel!, Tuple{A,A,Float32})
    touches = Mantle.accessof(interp, access_kernel!, (A,A,Float32))
    @test touches == [Mantle.NOTOUCH, Mantle.READ | Mantle.WRITE, Mantle.READ, Mantle.NOTOUCH]
    return nothing
end

function prime_common(signatures, cache)
    for sig in Serialization.deserialize(signatures)
        F = sig.parameters[1]
        @assert Base.issingletontype(F)
        f = F.instance
        args = Tuple(sig.parameters[2:end])
        interp = Mantle.Lava.kernelinterpreter(f, Tuple{args...}; workgroup_size = (32,1,1))
        Mantle.accessof(interp, f, args; cache)
    end
    return nothing
end

function run(output; device, reference = nothing, expected_hash = nothing,
             prime_access = false, prime_signatures = nothing,
             profile = false, trace_compile = false, save_profile_samples = false)
    reference !== nothing || expected_hash !== nothing ||
        throw(ArgumentError("A reference image or its SHA256 is required"))
    primer = @timed if prime_signatures !== nothing
        Base.invokelatest(prime_common, prime_signatures, Mantle.accesscache(device))
    elseif prime_access
        Base.invokelatest(prime)
    end
    setup = @timed Probe.fixture(Probe.principled_material(first(Probe.configuration().cases)),
                                 Probe.configuration())
    f = setup.value
    try
        Mantle.Lava.reset_compile_stats!()
        if profile
            Profile.clear()
            Profile.init(; n = 5_000_000, delay = 0.01)
        end
        draw() = Mantle.no_pipeline_compilation(() -> Probe.draw(f); ctx = device)
        action() = if profile
            Profile.@profile draw()
        else
            draw()
        end
        first = @timed if trace_compile
            open(output * ".trace.jl", "w") do io
                redirect_stderr(io) do
                    Base.@trace_compile action()
                end
            end
        else
            action()
        end
        stats = Mantle.Lava.compile_stats()
        native_misses = device.diag.pipeline_compiles_refused
        @test stats.misses == 0
        @test native_misses == 0
        image = Probe.pixels(f)
        digest = bytes2hex(sha256(reinterpret(UInt8, image)))
        reference === nothing || @test image == reinterpret(Float32, read(reference))
        expected_hash === nothing || @test digest == expected_hash
        record = (; prime_access, prime_signatures, profile, trace_compile,
                  source_hash = Probe.source_hash(), image_sha256 = digest,
                  primer_seconds = primer.time, primer_compile_seconds = primer.compile_time,
                  setup_seconds = setup.time, first_seconds = first.time, compile_seconds = first.compile_time,
                  stats, native_misses)
        mkpath(dirname(output))
        open(io -> JSON3.write(io, record), output, "w")
        if profile
            open(io -> Profile.print(io; format = :flat, sortedby = :count, C = true, threads = 1), output * ".flat.txt", "w")
            save_profile_samples && Serialization.serialize(output * ".profile.jls", Profile.retrieve())
        end
        println(record)
    finally
        Probe.release(f)
    end
    return nothing
end
end
