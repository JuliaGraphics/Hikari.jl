# GPU-safe integer conversion helpers (avoid InexactError)
u_int32(x) = Base.unsafe_trunc(Int32, x)

# Unchecked unsigned div/rem, for operands known to be non-negative.
#
# `div`/`mod` on signed integers emit a div-by-zero guard and an INT_MIN/-1
# guard. Both are throw paths, and a throw cannot exist on a GPU: it compiles
# to error-reporting machinery that can never run, and — per Lava's own note in
# `device/quirks.jl` — the resulting control flow is a pattern NVIDIA's shader
# compiler miscompiles. So this is a correctness fix as much as a speed one.
#
# Every caller feeds a pixel index (>= 0, it is `pixel_index - 1` with
# `pixel_index >= 1`) and an image width (> 0), so the unsigned reading is the
# correct one.
@inline u_div(a::Int32, b::Int32) =
    Base.bitcast(Int32, Base.udiv_int(Base.bitcast(UInt32, a), Base.bitcast(UInt32, b)))
@inline u_mod(a::Int32, b::Int32) =
    Base.bitcast(Int32, Base.urem_int(Base.bitcast(UInt32, a), Base.bitcast(UInt32, b)))
u_int(x) = Base.unsafe_trunc(Int, x)
u_uint32(x) = Base.unsafe_trunc(UInt32, x)
u_uint64(x) = Base.unsafe_trunc(UInt64, x)
floor_int32(x) = Base.unsafe_trunc(Int32, floor(x))
floor_int(x) = Base.unsafe_trunc(Int, floor(x))
round_int32(x) = Base.unsafe_trunc(Int32, round(x))

struct Film{
    FB<:AbstractMatrix{RGB{Float32}},
    PP<:AbstractMatrix{RGBA{Float32}},
    AB<:AbstractMatrix{RGB{Float32}},
    NB<:AbstractMatrix{Vec3f},
    DB<:AbstractMatrix{Float32},
}
    resolution::Point2f
    crop_bounds::Bounds2
    diagonal::Float32

    # Filter parameters (for pixel reconstruction in wavefront integrator)
    filter_table::Matrix{Float32}
    filter_table_width::Int32
    filter_radius::Point2f
    filter_params::GPUFilterParams
    scale::Float32

    # Sensor-calibrated linear HDR output (written by integrator)
    # Contains raw linear sRGB with imaging_ratio already applied.
    # Equivalent to pbrt-v4's GetPixelRGB output.
    framebuffer::FB

    # Auxiliary buffers for denoising (first-hit data)
    albedo::AB
    normal::NB
    depth::DB

    # Display-ready output (written by postprocess!)
    postprocess::PP

    # Progressive rendering state
    iteration_index::Base.RefValue{Int32}

    # The denoiser's compiled plan and its ping-pong scratch (see `denoise!`).
    # Kept across frames rather than rebuilt per call, and `Any` so the plan's
    # type stays out of `Film`'s.
    #
    # `aux_rays`/`aux_results` used to sit here too — scratch for a three-step
    # HW aux pass that `hw_fill_aux_kernel!` replaced. Nothing has assigned them
    # since; they were declared, carried through every adapt and freed in
    # `free!`, and always `nothing`.
    denoise_plan::Base.RefValue{Any}

    # Every device array above comes from here and goes back through it. See
    # `device-memory.jl`: nothing in this package finalizes a GPU resource.
    memory::DeviceMemory
end

"""
    Film(resolution; filter, crop_bounds, diagonal, scale, filter_table_width)

Create a film buffer for rendering.

- `resolution`: Image size in pixels, e.g. `Point2f(1920, 1080)`
- `filter`: Pixel reconstruction filter (default: LanczosSinc)
- `crop_bounds`: Region to render in [0,1] range (default: full image)
"""
function Film(
        resolution::Point2f;
        filter=Hikari.LanczosSincFilter(Point2f(1f0), 3f0),
        crop_bounds::Bounds2=Hikari.Bounds2(Point2f(0f0), Point2f(1f0)),
        diagonal=1f0, scale=1f0,
        tile_size=4, filter_table_width=16,
    )
    crop_bounds = Bounds2(
        ceil.(resolution .* crop_bounds.p_min) .+ 1.0f0,
        ceil.(resolution .* crop_bounds.p_max),
    )
    crop_resolution = Int32.(inclusive_sides(crop_bounds))

    # Precompute filter weight table
    filter_table = Matrix{Float32}(undef, filter_table_width, filter_table_width)
    r = filter.radius ./ filter_table_width
    for y in 0:filter_table_width-1, x in 0:filter_table_width-1
        p = Point2f((x + 0.5f0) * r[1], (y + 0.5f0) * r[2])
        filter_table[y+1, x+1] = filter(p)
    end

    # A host film's memory is pooled too, and costs nothing to forget:
    # `Mantle.Device(HostAPI())` builds a FRESH pool per call, so the whole thing
    # is reachable only from this film and the GC reclaims it if `free!` is
    # never reached. On a device the pool is shared and cached, which is what
    # `retire!` and `Mantle.reclaim!` are for.
    mem = DeviceMemory(KA.CPU())
    pixel_size = (Int(crop_resolution[end]), Int(crop_resolution[begin]))
    framebuffer = alloc!(mem, RGB{Float32}, pixel_size)
    albedo = alloc!(mem, RGB{Float32}, pixel_size, RGB{Float32}(0, 0, 0))
    normal = alloc!(mem, Vec3f, pixel_size, Vec3f(0, 0, 0))
    depth = alloc!(mem, Float32, pixel_size, 0.0f0)
    postprocess = alloc!(mem, RGBA{Float32}, pixel_size)

    return Film(
        resolution,
        crop_bounds,
        diagonal * 0.001f0,
        filter_table,
        Int32(filter_table_width),
        filter.radius,
        GPUFilterParams(filter),
        scale,
        framebuffer,
        albedo,
        normal,
        depth,
        postprocess,
        Ref(Int32(0)),
        Ref{Any}(nothing),
        mem,
    )
end

"""
    Film(backend, film::Film) -> Film

`film`'s layers on `backend`, in memory this film owns.

Replaces `Adapt.adapt(backend, film)`, which was doing the allocating — `adapt`
converts, and a conversion that reaches for an allocator has nowhere to put one,
which is why the result could only ever be freed by the GC. `free!(film)`
releases these; forgetting it retires them instead of losing them.

`filter_table` stays on the host: it is read when the plans are built, never by a
kernel. `iteration_index` is shared with `film`, as it was before — it is the
progressive sample counter, and two films disagreeing about it is the bug.
"""
function Film(backend, film::Film)
    mem = DeviceMemory(backend)
    return Film(
        film.resolution,
        film.crop_bounds,
        film.diagonal,
        film.filter_table,
        film.filter_table_width,
        film.filter_radius,
        film.filter_params,
        film.scale,
        upload!(mem, film.framebuffer),
        upload!(mem, film.albedo),
        upload!(mem, film.normal),
        upload!(mem, film.depth),
        upload!(mem, film.postprocess),
        film.iteration_index,
        # A fresh slot, not the source film's: the plan names the buffers it
        # filters, and this film's are these.
        Ref{Any}(nothing),
        mem,
    )
end

function clear!(film::Film)
    film.iteration_index[] = Int32(0)
    fill!(film.albedo, RGB{Float32}(0, 0, 0))
    fill!(film.normal, Vec3f(0, 0, 0))
    fill!(film.depth, 0.0f0)
end

# ============================================================================
# Auxiliary Buffer Filling (for denoising)
# ============================================================================

"""
The depth written where a primary ray hit nothing.

Finite, not `Inf`, when the scene has an infinite light: the denoiser filters
across depth, and a difference against `Inf` is `NaN` — with a sky there, the
missed pixel is content to be filtered, not missing data. Without one, `Inf` is
kept because it is the honest answer and nothing downstream reads it.

Both are far beyond any scene, so [`missed`](@ref) reads either, and every
consumer asks that instead of testing for one of the two. Testing `isinf`
directly is what made a transparent background come out opaque: with an
`AmbientLight` in the scene the misses were 1f30, no pixel counted as escaped,
and the alpha the compositing caller reads was 1 everywhere.
"""
const MISS_DEPTH = 1.0f30

"Whether the primary ray at this depth hit nothing — see [`MISS_DEPTH`](@ref)."
missed(depth::Real) = depth >= MISS_DEPTH

"""
    fill_aux_buffers!(film, scene, camera; has_infinite_lights=false)

Fill auxiliary buffers (albedo, normal, depth) by tracing primary rays.
These are used by `denoise!` for edge-aware filtering.
"""
function fill_aux_buffers!(film::Film, scene, camera; has_infinite_lights::Bool=false)
    albedo = film.albedo
    normal = film.normal
    depth = film.depth
    resolution = film.resolution
    crop_bounds = film.crop_bounds
    miss_depth = has_infinite_lights ? MISS_DEPTH : Inf32

    backend = KA.get_backend(albedo)
    # Adapt the acceleration structure for kernel dispatch (TLAS -> StaticTLAS)
    accel = Adapt.adapt(backend, scene.accel)
    kernel! = aux_buffer_kernel!(backend)
    kernel!(
        albedo, normal, depth,
        resolution, crop_bounds,
        accel, camera, miss_depth;
        ndrange = length(albedo)
    )
    Mantle.waitidle(mantle_device(backend))
    return film
end

@kernel inbounds=true function aux_buffer_kernel!(albedo, normal, depth, resolution, crop_bounds, accel, camera, miss_depth::Float32)
    idx = @index(Global)

    h, _ = size(albedo)
    row = ((idx - 1) % h) + 1
    col = ((idx - 1) ÷ h) + 1

    px = Float32(col) + crop_bounds.p_min[1] - 1f0
    # Match the camera raster orientation used by the rendered framebuffer.
    py = Float32(h - row) + crop_bounds.p_min[2]
    pixel = Point2f(px + 0.5f0, py + 0.5f0)

    camera_sample = CameraSample(pixel, Point2f(0.5f0), 0f0)
    ray, ω = generate_ray(camera, camera_sample)

     if ω > 0f0
        hit, _primitive, si = intersect!(accel, ray)
        if hit
            normal[idx] = Vec3f(si.core.n)
            cam_pos = ray.o
            hit_pos = si.core.p
            depth[idx] = sqrt(sum((hit_pos .- cam_pos) .^ 2))
            albedo[idx] = RGB{Float32}(0.8f0, 0.8f0, 0.8f0)
        else
            normal[idx] = Vec3f(0f0, 0f0, 0f0)
            depth[idx] = miss_depth
            albedo[idx] = RGB{Float32}(0f0, 0f0, 0f0)
        end
    else
        normal[idx] = Vec3f(0f0, 0f0, 0f0)
        depth[idx] = Inf32
        albedo[idx] = RGB{Float32}(0f0, 0f0, 0f0)
    end
end

# ============================================================================
# Resource Cleanup
# ============================================================================

"""
    free!(film::Film)

Release GPU memory held by the film.

No precondition. This used to require an idle GPU and warn that calling it
during a render was a use-after-free; the regions are retired now, so the pool
decides when they are safe rather than the caller.
"""
function free!(film::Film)
    # The denoiser first: it holds a compiled plan and its own scratch, and both
    # are its to give back. `filter_table` appears nowhere because it is a host
    # `Matrix{Float32}` — it was in the `finalize` list this replaces, doing
    # nothing while reading as if it released GPU memory.
    dp = film.denoise_plan[]
    if dp !== nothing
        free!(dp)
        film.denoise_plan[] = nothing
    end
    free!(film.memory)
    return nothing
end
