# VolPath state container
# Uses the unified WorkQueue from integrators/workqueue.jl

import KernelAbstractions as KA

# ============================================================================
# SOA Override for VolPath Work Items
# ============================================================================

# Override should_use_soa for VolPath work items that benefit from SOA layout.
#
# pbrt-v4's wavefront integrator declares its work items in a .soa DSL
# (wavefront/workitems.soa) that the build system expands into per-field
# SOA<T> templates. The reason: every queue is large enough (≥ scene-pixel-
# count entries) and each shading kernel reads many fields out of each work
# item across many threads — the AOS stride wastes most of the memory
# bandwidth on cache lines whose other fields the warp doesn't need this
# pass. SOA fixes this by giving each field its own coalesced array.
#
# The big items (≥ 64 B) are the ones where the bandwidth difference shows
# up at render scale. The defaults below cover every queue that's heavy
# enough to matter: the two ray queues, surface-hit, material-eval, shadow,
# escaped, and the two medium queues. Trait dispatch is read by
# `WorkQueue{T}(backend, capacity)` which defaults `soa=should_use_soa(T)` —
# so flagging a new work-item type here is sufficient to opt it in.
should_use_soa(::Type{VPRayWorkItem}) = true
should_use_soa(::Type{VPShadowRayWorkItem}) = true
should_use_soa(::Type{VPEscapedRayWorkItem}) = true
should_use_soa(::Type{VPHitSurfaceWorkItem}) = true
should_use_soa(::Type{VPMaterialEvalWorkItem}) = true
should_use_soa(::Type{VPMediumSampleWorkItem}) = true
should_use_soa(::Type{VPMediumScatterWorkItem}) = true
# VPRaySamples used to be a per-pixel Sobol cache filled by
# vp_generate_ray_samples_kernel!; that kernel and its buffer are gone now
# (samples are computed inline in each consumer), so the SOA hint is no
# longer needed.

# ============================================================================
# VolPath State Container
# ============================================================================

"""
    VolPathState

Contains all work queues and buffers for VolPath wavefront rendering.
"""
mutable struct VolPathState{Backend}
    backend::Backend

    # Every device allocation below comes from here and goes back through it.
    # See `device-memory.jl`: nothing in this package finalizes a GPU resource.
    memory::DeviceMemory

    # Ray queues (double-buffered for iteration)
    ray_queue_a::WorkQueue{VPRayWorkItem}
    ray_queue_b::WorkQueue{VPRayWorkItem}

    # Medium sample queue (rays in medium with bounded t_max from intersection)
    medium_sample_queue::WorkQueue{VPMediumSampleWorkItem}

    # Medium scatter queue (real scattering events from delta tracking)
    medium_scatter_queue::WorkQueue{VPMediumScatterWorkItem}

    # Shared surface-hit store. Every surface hit (fused SW trace, HW chit
    # legacy path, medium survive-to-surface) is written here ONCE; the
    # per-material typed queues hold 4-byte `TypedHitRef{T}` indices into it.
    hit_surface_queue::WorkQueue{VPHitSurfaceWorkItem}

    # Shadow ray queue
    shadow_queue::WorkQueue{VPShadowRayWorkItem}

    # Escaped ray queue
    escaped_queue::WorkQueue{VPEscapedRayWorkItem}

    # Hit area-light queue — pbrt-v4 hitAreaLightQueue (wavefront/integrator.h).
    # Surface hits on emissive triangles are pushed here in PARALLEL to the
    # per-material queue push (see `enqueue_after_intersection!`); a dedicated
    # emitter kernel (`vp_handle_emitters!`) drains it between trace and
    # shade. Keeps the per-material kernels free of emission-MIS code +
    # eliminates the corresponding fields (`arealight_flat_idx`,
    # `triangle_area`, `t_hit`, `prev_intr_p`, `prev_intr_n`) from the path
    # that surface BSDF eval traverses.
    hit_area_light_queue::WorkQueue{VPHitAreaLightWorkItem}

    # Film buffer (spectral radiance per pixel, 4 wavelengths)
    pixel_L::AbstractVector{Float32}

    # Accumulators for progressive rendering (moved from main loop)
    # NOTE: Default Float64 matches pbrt-v4's double precision accumulators (film.h:304-305)
    # Use Float32 for OpenCL backends that don't support double precision atomics
    pixel_rgb::AbstractVector  # n_pixels * 3 (RGB accumulator)
    pixel_weight_sum::AbstractVector  # n_pixels (filter weight accumulator)
    wavelengths_per_pixel::AbstractVector{Float32}  # n_pixels * 4 (wavelength samples)
    pdf_per_pixel::AbstractVector{Float32}  # n_pixels * 4 (wavelength PDFs)
    filter_weight_per_pixel::AbstractVector{Float32}  # n_pixels (filter weight per sample)

    # RGB to spectrum table
    rgb2spec_table::RGBToSpectrumTable

    # Spectral response table (CIE XYZ or camera sensor curves)
    cie_table::CIEXYZTable

    # Sensor output: response → output RGB matrix + exposure scaling
    output_matrix::Mat3f
    imaging_ratio::Float32

    # BVH Light Sampler data (spatially-aware importance sampling)
    bvh_nodes::AbstractVector{LightBVHNode}           # BVH node array (GPU)
    light_to_bit_trail::AbstractVector{UInt32}         # Per-light bit trail (GPU)
    infinite_light_indices::AbstractVector{Int32}       # Flat indices of infinite lights (GPU)
    num_bvh_lights::Int32                               # Count of bounded lights in BVH
    num_infinite_lights::Int32                          # Count of infinite lights
    num_lights::Int32

    # Render parameters
    max_depth::Int32
    rr_depth::Int32
    width::Int32
    height::Int32

    # Sobol RNG for low-discrepancy sampling (allocated once, reused across frames)
    sobol_rng::Any  # SobolRNG or nothing

    # Per-material-type shading queues (one `WorkQueue{TypedHitRef{T}}` per
    # concrete material type `T` in the scene).  Built lazily on first render
    # from the scene's adapted `StaticMultiTypeSet` of materials; rebuilt
    # when the type tuple changes (different scene).  Type is left `Any`
    # because the concrete `MultiTypeMaterialQueue{Qs}` signature depends on
    # which materials are present.
    per_material_queue::Any              # MultiTypeMaterialQueue{...} | nothing
    per_material_queue_signature::Any    # type tuple, used as freshness check

    # Their own `DeviceMemory` and not the one above, because they have their own
    # lifetime: a scene with different material types replaces them, and regions
    # that go back on a rebuild must not be mixed in with the ones that live as
    # long as the state. One `free!` per group is the whole reason this is a
    # second allocator rather than a second list.
    per_material_memory::DeviceMemory

    # The compiled Mantle plans for a sample (see graph.jl). Here rather than on
    # the integrator because they name this state's queues and accumulators: a
    # state that is rebuilt takes its plans with it.  `Any` for the same reason
    # the queues above are — the concrete type carries the scene's material
    # types and the adapted scene's.
    plans::Any                           # VPPlans{...} | nothing
end

# The default `show` printed every work queue element by element, each one a read
# of device memory: printing one state (a failing test's operand) ran for longer
# than 20 minutes. What is worth seeing is its shape and whether it is live.
function Base.show(io::IO, state::VolPathState)
    print(io, "VolPathState(", state.width, "×", state.height, ", ", state.num_lights,
          " lights, max_depth ", state.max_depth, ", ",
          nallocations(state.memory), " allocations, ",
          state.plans === nothing ? "no plans" : "plans recorded", ")")
end

"""
    free!(state::VolPathState)

Release all GPU memory held by the VolPath render state (work queues,
pixel buffers, tables).

No precondition: the regions are retired, not released, so calling this with a
render still in flight is fine. It used to require an idle GPU and say so.
"""
function free!(state::VolPathState)
    # The plans first: they name the buffers below, and a plan gives its own pool
    # regions back rather than freeing anything the state owns.
    state.plans === nothing || free!(state.plans)
    state.plans = nothing
    # Then everything at once. The queues, the accumulators, the light BVH and
    # the Sobol matrices are all regions of one pool, so there is no order to get
    # right and nothing here to keep in step with the constructor.
    free!(state.memory)
    free!(state.per_material_memory)
    return nothing
end

function VolPathState(
    backend,
    width::Integer,
    height::Integer,
    lights::Raycore.MultiTypeSet;  # MultiTypeSet of lights (has backend for GPU allocation)
    max_depth::Integer = 8,
    rr_depth::Integer = 1,
    queue_capacity::Integer = width * height,
    scene_radius::Float32 = 10f0,  # Scene bounding sphere radius for light power estimation
    samples_per_pixel::Integer = 1,  # For SobolRNG parameter computation
    sampler_seed::UInt32 = UInt32(0),  # Scrambling seed for Sobol
    accumulation_eltype::DataType = Float32,  # Element type for accumulators (Float32 for OpenCL)
    sensor::PixelSensor = PixelSensor(),  # Pixel sensor for spectral → RGB conversion
    # When HW per-material chit slots own the shading, the post-hoc
    # `vp_handle_emitters!` and `vp_shade_surfaces!` kernels never run, so the
    # `hit_area_light_queue`, the shared `hit_surface_queue` and the typed
    # per-material queues are unused. `hw_accel=true` allocates them with
    # capacity=1 (placeholder slots that the chit body never touches) so the
    # queue allocations don't blow past GPU memory.
    hw_accel::Bool = false,
    # Scenes without participating media never push to the medium queues;
    # allocating them at full pixel capacity wasted ~580 MiB on a 1.4 Mpx
    # render (VPMediumSampleWorkItem is 320 B). `has_media=false` allocates
    # capacity-1 placeholders. The freshness check in `render!` rebuilds the
    # state when a scene with media shows up.
    has_media::Bool = true,
)
    mem = DeviceMemory(backend)
    n_pixels = width * height
    vestigial_capacity = hw_accel ? 1 : queue_capacity
    medium_capacity = has_media ? queue_capacity : 1
    # The shared hit store feeds the SW typed-queue shading path and the
    # medium survive-to-surface path; with neither (HW chit + no media) a
    # placeholder suffices.
    hit_surface_capacity = hw_accel ? 1 : queue_capacity

    # Create work queues
    ray_queue_a = WorkQueue{VPRayWorkItem}(mem, queue_capacity)
    ray_queue_b = WorkQueue{VPRayWorkItem}(mem, queue_capacity)
    medium_sample_queue = WorkQueue{VPMediumSampleWorkItem}(mem, medium_capacity)
    medium_scatter_queue = WorkQueue{VPMediumScatterWorkItem}(mem, medium_capacity)
    hit_surface_queue = WorkQueue{VPHitSurfaceWorkItem}(mem, hit_surface_capacity)
    shadow_queue = WorkQueue{VPShadowRayWorkItem}(mem, queue_capacity)
    escaped_queue = WorkQueue{VPEscapedRayWorkItem}(mem, queue_capacity)
    hit_area_light_queue = WorkQueue{VPHitAreaLightWorkItem}(mem, vestigial_capacity)

    # Film buffer (4 wavelengths per pixel)
    pixel_L = alloc!(mem, Float32, n_pixels * 4, 0f0)

    # Accumulators for progressive rendering (configurable eltype for OpenCL compatibility)
    pixel_rgb = alloc!(mem, accumulation_eltype, n_pixels * 3, zero(accumulation_eltype))
    pixel_weight_sum = alloc!(mem, accumulation_eltype, n_pixels, zero(accumulation_eltype))
    wavelengths_per_pixel = alloc!(mem, Float32, n_pixels * 4, 0f0)
    pdf_per_pixel = alloc!(mem, Float32, n_pixels * 4, 0f0)
    filter_weight_per_pixel = alloc!(mem, Float32, n_pixels, 0f0)

    # Load lookup tables to GPU (sensor determines response curves)
    rgb2spec_table = to_gpu(mem, get_srgb_table())
    cie_table = to_gpu(mem, sensor_response_table(sensor.sensor_name))

    # Build BVH light sampler (spatially-aware importance sampling)
    n_lights = length(lights)
    if n_lights > 0
        bvh_sampler = BVHLightSampler(lights; scene_radius=scene_radius)
        bvh_gpu = bvh_to_gpu(mem, bvh_sampler)
        bvh_nodes = bvh_gpu.nodes
        light_to_bit_trail = bvh_gpu.light_to_bit_trail
        infinite_light_indices = bvh_gpu.infinite_light_indices
        num_bvh_lights = bvh_gpu.num_bvh_lights
        num_infinite_lights = bvh_gpu.num_infinite_lights
    else
        bvh_nodes = alloc!(mem, LightBVHNode, 1)
        light_to_bit_trail = alloc!(mem, UInt32, 1)
        infinite_light_indices = alloc!(mem, Int32, 1)
        num_bvh_lights = Int32(0)
        num_infinite_lights = Int32(0)
    end

    # Create SobolRNG (allocated once, reused across frames)
    sobol_rng = SobolRNG(mem, sampler_seed, width, height, samples_per_pixel)

    return VolPathState(
        backend,
        mem,
        ray_queue_a, ray_queue_b,
        medium_sample_queue, medium_scatter_queue,
        hit_surface_queue, shadow_queue, escaped_queue,
        hit_area_light_queue,
        pixel_L, pixel_rgb, pixel_weight_sum,
        wavelengths_per_pixel, pdf_per_pixel, filter_weight_per_pixel,
        rgb2spec_table, cie_table,
        sensor.output_from_sensor, sensor.imaging_ratio,
        bvh_nodes, light_to_bit_trail, infinite_light_indices,
        num_bvh_lights, num_infinite_lights, Int32(n_lights),
        Int32(max_depth), Int32(rr_depth), Int32(width), Int32(height),
        sobol_rng,
        # Per-material typed queues (built lazily once we see the scene's
        # adapted materials).
        nothing, nothing, DeviceMemory(backend),
        # Plans (built lazily on the first render, once the scene shape and the
        # film are known).
        nothing,
    )
end

"""
    ensure_per_material_queue!(state, materials_static, capacity)

Lazily build the per-material typed queue from the scene's adapted
`StaticMultiTypeSet` of materials.  Rebuilt when the type signature
changes (different scene); a no-op cache hit otherwise.  Called once at
the start of each `render!` after scene adaptation.
"""
function ensure_per_material_queue!(state::VolPathState, materials_static, capacity::Integer)
    sig = typeof(materials_static).parameters[1]   # Data tuple of StaticMultiTypeSet
    if state.per_material_queue === nothing || state.per_material_queue_signature !== sig
        # The previous scene's queues go back before the new ones are taken, so
        # switching scenes does not accumulate a set per scene. No wait: the
        # regions are retired, and the allocator that wants them next is the one
        # that decides whether the device is done with them.
        state.per_material_queue === nothing || free!(state.per_material_memory)
        state.per_material_queue =
            build_per_material_queues(materials_static, capacity, state.per_material_memory)
        state.per_material_queue_signature = sig
    end
    return state.per_material_queue
end

# ============================================================================
# State Helpers
# ============================================================================


