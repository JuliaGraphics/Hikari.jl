# Intel Open Image Denoise (OIDN): its ray tracing filter, a network trained on
# path-traced frames, run on the CPU. The library is the official Linux build, a
# lazy artifact (see `Artifacts.toml`) fetched the first time a film is denoised.

"""
    OIDN(; quality = :high, guides = true)

Denoise with Intel Open Image Denoise's ray tracing filter, on the CPU. `quality` is
`:fast`, `:balanced` or `:high`. With `guides` the filter also gets the film's albedo
and normal (see [`fill_aux_buffers!`](@ref)), noise-free, which keep the edges and
textures of what the camera sees first.
"""
struct OIDN <: Denoiser
    quality::Int32
    guides::Bool
end

function OIDN(; quality::Symbol = :high, guides::Bool = true)
    level = quality === :fast ? 4 : quality === :balanced ? 5 : quality === :high ? 6 :
            throw(ArgumentError("OIDN quality is :fast, :balanced or :high, not :$quality"))
    return OIDN(Int32(level), guides)
end

"""
The OIDN device and filter for one film, and the host images they share: each
`(width, height)`, so that Julia's column order is OIDN's rows.
"""
mutable struct OIDNPlan
    framebuffer::Any                 # the film's, which these images are for
    config::OIDN
    lib::Ptr{Cvoid}
    device::Ptr{Cvoid}
    filter::Ptr{Cvoid}
    staging::Matrix{RGB{Float32}}    # the film's layout, `(height, width)`
    color::Matrix{RGB{Float32}}
    albedo::Matrix{RGB{Float32}}
    normal::Matrix{Vec3f}
    output::Matrix{RGB{Float32}}
end

"A function of the OIDN library `lib`."
oidnfunction(lib::Ptr{Cvoid}, name::Symbol) = Libc.Libdl.dlsym(lib, name)

"Throw the error OIDN's `device` holds, if it holds one."
function oidncheck(plan::OIDNPlan)
    message = Ref{Cstring}(C_NULL)
    code = ccall(oidnfunction(plan.lib, :oidnGetDeviceError), Cint, (Ptr{Cvoid}, Ref{Cstring}), plan.device, message)
    code == 0 && return nothing
    error("OIDN error $code: ", message[] == C_NULL ? "" : unsafe_string(message[]))
end

"Share the host `image` with the filter as its image `name` (three floats a pixel)."
function oidnimage!(plan::OIDNPlan, name::String, image::Matrix)
    width, height = size(image)
    ccall(oidnfunction(plan.lib, :oidnSetSharedFilterImage), Cvoid,
          (Ptr{Cvoid}, Cstring, Ptr{Cvoid}, Cint, Csize_t, Csize_t, Csize_t, Csize_t, Csize_t),
          plan.filter, name, pointer(image), 3, width, height, 0, 0, 0)   # 3: OIDN_FORMAT_FLOAT3
    return plan
end

function free!(plan::OIDNPlan)
    plan.filter == C_NULL || ccall(oidnfunction(plan.lib, :oidnReleaseFilter), Cvoid, (Ptr{Cvoid},), plan.filter)
    plan.device == C_NULL || ccall(oidnfunction(plan.lib, :oidnReleaseDevice), Cvoid, (Ptr{Cvoid},), plan.device)
    plan.filter = C_NULL
    plan.device = C_NULL
    return nothing
end

"""The OIDN filter for this film and `config`, made if there is not one."""
function denoise_plan!(film::Film, config::OIDN)
    cached = film.denoise_plan[]
    cached isa OIDNPlan && cached.framebuffer === film.framebuffer && cached.config == config && return cached
    if cached !== nothing
        free!(cached)
        film.denoise_plan[] = nothing
    end
    lib = Libc.Libdl.dlopen(joinpath(artifact"oidn", "oidn-2.5.1.x86_64.linux", "lib", "libOpenImageDenoise.so.2"))
    height, width = size(film.framebuffer)
    device = ccall(Libc.Libdl.dlsym(lib, :oidnNewDevice), Ptr{Cvoid}, (Cint,), 1)   # 1: OIDN_DEVICE_TYPE_CPU
    plan = OIDNPlan(film.framebuffer, config, lib, device, C_NULL,
                    Matrix{RGB{Float32}}(undef, height, width),
                    Matrix{RGB{Float32}}(undef, width, height), Matrix{RGB{Float32}}(undef, width, height),
                    Matrix{Vec3f}(undef, width, height), Matrix{RGB{Float32}}(undef, width, height))
    ccall(oidnfunction(lib, :oidnCommitDevice), Cvoid, (Ptr{Cvoid},), device)
    oidncheck(plan)
    plan.filter = ccall(oidnfunction(lib, :oidnNewFilter), Ptr{Cvoid}, (Ptr{Cvoid}, Cstring), device, "RT")
    oidnimage!(plan, "color", plan.color)
    oidnimage!(plan, "output", plan.output)
    if config.guides
        oidnimage!(plan, "albedo", plan.albedo)
        oidnimage!(plan, "normal", plan.normal)
        # The guides are traced through each pixel's centre, without noise.
        ccall(oidnfunction(lib, :oidnSetFilterBool), Cvoid, (Ptr{Cvoid}, Cstring, Bool), plan.filter, "cleanAux", true)
    end
    # The framebuffer is radiance, before exposure and tone mapping.
    ccall(oidnfunction(lib, :oidnSetFilterBool), Cvoid, (Ptr{Cvoid}, Cstring, Bool), plan.filter, "hdr", true)
    ccall(oidnfunction(lib, :oidnSetFilterInt), Cvoid, (Ptr{Cvoid}, Cstring, Cint), plan.filter, "quality", config.quality)
    ccall(oidnfunction(lib, :oidnCommitFilter), Cvoid, (Ptr{Cvoid},), plan.filter)
    oidncheck(plan)
    film.denoise_plan[] = plan
    return plan
end

"""
    denoise!(film::Film, config::OIDN)

Denoise `film.framebuffer` in place with Intel Open Image Denoise (see [`OIDN`](@ref)).
The frame goes to the host and back; with `config.guides`, `film.albedo` and
`film.normal` must be filled (`fill_aux_buffers!`).
"""
function denoise!(film::Film, config::OIDN)
    plan = denoise_plan!(film, config)
    copyto!(plan.staging, film.framebuffer)
    permutedims!(plan.color, plan.staging, (2, 1))
    if config.guides
        permutedims!(plan.albedo, Array(film.albedo), (2, 1))
        permutedims!(plan.normal, Array(film.normal), (2, 1))
    end
    ccall(oidnfunction(plan.lib, :oidnExecuteFilter), Cvoid, (Ptr{Cvoid},), plan.filter)
    oidncheck(plan)
    permutedims!(plan.staging, plan.output, (2, 1))
    copyto!(film.framebuffer, plan.staging)
    return nothing
end
