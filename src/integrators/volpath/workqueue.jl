# Work queues for VolPath wavefront integrator
# Reuses PWWorkQueue pattern from PhysicalWavefront

# ============================================================================
# Generic Work Queue (same as PhysicalWavefront)
# ============================================================================

"""
    VPWorkQueue{T}

Thread-safe work queue for wavefront rendering.
Uses atomic counter for GPU-safe push operations.
"""
struct VPWorkQueue{T, V<:AbstractVector{T}, S<:AbstractVector{Int32}}
    items::V
    size::S      # Single-element array for atomic operations
    capacity::Int32
end

function VPWorkQueue{T}(backend, capacity::Integer) where T
    items = KernelAbstractions.allocate(backend, T, capacity)
    size = KernelAbstractions.allocate(backend, Int32, 1)
    KernelAbstractions.fill!(size, Int32(0))
    VPWorkQueue{T, typeof(items), typeof(size)}(items, size, Int32(capacity))
end

"""Get current queue size (copies from GPU if needed)"""
@propagate_inbounds function queue_size(q::VPWorkQueue)
    s = Array(q.size)
    return Int(s[1])
end

"""Reset queue to empty"""
function reset_queue!(backend, q::VPWorkQueue)
    KernelAbstractions.fill!(q.size, Int32(0))
    KernelAbstractions.synchronize(backend)
end

# ============================================================================
# VolPath State Container
# ============================================================================

"""
    VolPathState

Contains all work queues and buffers for VolPath wavefront rendering.
"""
mutable struct VolPathState{Backend}
    backend::Backend

    # Ray queues (double-buffered for iteration)
    ray_queue_a::VPWorkQueue{VPRayWorkItem}
    ray_queue_b::VPWorkQueue{VPRayWorkItem}
    current_ray_queue::Symbol  # :a or :b

    # Medium sample queue (rays in medium with bounded t_max from intersection)
    medium_sample_queue::VPWorkQueue{VPMediumSampleWorkItem}

    # Medium scatter queue (real scattering events from delta tracking)
    medium_scatter_queue::VPWorkQueue{VPMediumScatterWorkItem}

    # Surface hit queue (rays that hit surfaces and are NOT in medium)
    hit_surface_queue::VPWorkQueue{VPHitSurfaceWorkItem}

    # Material evaluation queue
    material_queue::VPWorkQueue{VPMaterialEvalWorkItem}

    # Shadow ray queue
    shadow_queue::VPWorkQueue{VPShadowRayWorkItem}

    # Escaped ray queue
    escaped_queue::VPWorkQueue{VPEscapedRayWorkItem}

    # Film buffer (spectral radiance per pixel, 4 wavelengths)
    pixel_L::AbstractVector{Float32}

    # RGB to spectrum table
    rgb2spec_table::RGBToSpectrumTable

    # Render parameters
    max_depth::Int32
    width::Int32
    height::Int32
end

function VolPathState(
    backend,
    width::Integer,
    height::Integer;
    max_depth::Integer = 8,
    queue_capacity::Integer = width * height
)
    n_pixels = width * height

    # Create work queues
    ray_queue_a = VPWorkQueue{VPRayWorkItem}(backend, queue_capacity)
    ray_queue_b = VPWorkQueue{VPRayWorkItem}(backend, queue_capacity)
    medium_sample_queue = VPWorkQueue{VPMediumSampleWorkItem}(backend, queue_capacity)
    medium_scatter_queue = VPWorkQueue{VPMediumScatterWorkItem}(backend, queue_capacity)
    hit_surface_queue = VPWorkQueue{VPHitSurfaceWorkItem}(backend, queue_capacity)
    material_queue = VPWorkQueue{VPMaterialEvalWorkItem}(backend, queue_capacity)
    shadow_queue = VPWorkQueue{VPShadowRayWorkItem}(backend, queue_capacity)
    escaped_queue = VPWorkQueue{VPEscapedRayWorkItem}(backend, queue_capacity)

    # Film buffer (4 wavelengths per pixel)
    pixel_L = KernelAbstractions.allocate(backend, Float32, n_pixels * 4)
    KernelAbstractions.fill!(pixel_L, 0f0)

    # Load RGB to spectrum table and convert to appropriate array type
    rgb2spec_table_cpu = get_srgb_table()
    # Determine array type from an allocated array
    ArrayType = backend isa KernelAbstractions.CPU ? Array : typeof(pixel_L).name.wrapper
    rgb2spec_table = to_gpu(ArrayType, rgb2spec_table_cpu)

    VolPathState(
        backend,
        ray_queue_a,
        ray_queue_b,
        :a,
        medium_sample_queue,
        medium_scatter_queue,
        hit_surface_queue,
        material_queue,
        shadow_queue,
        escaped_queue,
        pixel_L,
        rgb2spec_table,
        Int32(max_depth),
        Int32(width),
        Int32(height)
    )
end

"""Get current input ray queue"""
@propagate_inbounds function current_ray_queue(state::VolPathState)
    state.current_ray_queue == :a ? state.ray_queue_a : state.ray_queue_b
end

"""Get next output ray queue"""
@propagate_inbounds function next_ray_queue(state::VolPathState)
    state.current_ray_queue == :a ? state.ray_queue_b : state.ray_queue_a
end

"""Swap ray queue buffers"""
function swap_ray_queues!(state::VolPathState)
    state.current_ray_queue = state.current_ray_queue == :a ? :b : :a
end

"""Reset all queues for new iteration"""
function reset_iteration_queues!(state::VolPathState)
    backend = state.backend
    reset_queue!(backend, next_ray_queue(state))
    reset_queue!(backend, state.medium_sample_queue)
    reset_queue!(backend, state.medium_scatter_queue)
    reset_queue!(backend, state.hit_surface_queue)
    reset_queue!(backend, state.material_queue)
    reset_queue!(backend, state.shadow_queue)
    reset_queue!(backend, state.escaped_queue)
end

"""Reset film for new sample"""
function reset_film!(state::VolPathState)
    KernelAbstractions.fill!(state.pixel_L, 0f0)
    KernelAbstractions.synchronize(state.backend)
end
