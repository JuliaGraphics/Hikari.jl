# Work queue implementation for PhysicalWavefront path tracing
# Uses KernelAbstractions for CPU/GPU portability
# Adapted from PbrtWavefront to use Hikari types

using Atomix: @atomic

# ============================================================================
# Generic Work Queue
# ============================================================================

"""
    PWWorkQueue{T, ItemsArray, CounterArray}

A GPU-compatible work queue that stores items of type T.
Uses atomic operations for thread-safe push operations.

The queue stores items in a pre-allocated buffer and uses an atomic
counter to track the current size.
"""
struct PWWorkQueue{T, ItemsArray <: AbstractVector{T}, CounterArray <: AbstractVector{Int32}}
    items::ItemsArray
    size::CounterArray  # Single-element array for atomic access
    capacity::Int32
end

"""
    PWWorkQueue{T}(capacity::Integer, ArrayType=Vector)

Create a new work queue with the given capacity.
For GPU usage, pass ROCArray or CuArray as ArrayType.
"""
function PWWorkQueue{T}(capacity::Integer, ArrayType=Vector) where {T}
    items = ArrayType{T}(undef, capacity)
    # Initialize size on CPU then convert - avoids scalar indexing on GPU
    size_cpu = Int32[0]
    size_arr = ArrayType(size_cpu)
    return PWWorkQueue{T, typeof(items), typeof(size_arr)}(items, size_arr, Int32(capacity))
end

"""
    to_gpu(ArrayType, queue::PWWorkQueue{T}) -> PWWorkQueue{T}

Convert a PWWorkQueue to use GPU arrays.
Creates a new empty queue with the same capacity.
"""
function to_gpu(ArrayType, queue::PWWorkQueue{T}) where {T}
    return PWWorkQueue{T}(queue.capacity, ArrayType)
end

"""
    reset!(queue::PWWorkQueue)

Reset the queue to empty state. Works on both CPU and GPU arrays.
"""
@propagate_inbounds function reset!(queue::PWWorkQueue)
    fill!(queue.size, Int32(0))
    return nothing
end

"""
    queue_size(queue::PWWorkQueue) -> Int32

Return the current number of items in the queue.
Note: This copies from GPU to CPU, so use sparingly (between kernel dispatches).
"""
@propagate_inbounds queue_size(queue::PWWorkQueue) =  Array(queue.size)[1]

"""
    push_work!(queue::PWWorkQueue{T}, item::T) -> Int32

Atomically push an item to the queue and return its index (1-based).
Thread-safe for concurrent GPU access.
"""
@propagate_inbounds function push_work!(queue::PWWorkQueue{T}, item::T) where {T}
    # Atomically increment size and get index
    idx = @atomic queue.size[1] += Int32(1)
    # Store item at the allocated index
     queue.items[idx] = item
    return idx
end

"""
    get_item(queue::PWWorkQueue, idx::Integer)

Get item at the given index (1-based).
"""
@propagate_inbounds function get_item(queue::PWWorkQueue, idx::Integer)
     return queue.items[idx]
end

# Allow indexing
@propagate_inbounds Base.getindex(queue::PWWorkQueue, idx::Integer) = get_item(queue, idx)

# ============================================================================
# Reset Kernel - GPU-safe queue reset
# ============================================================================

@kernel inbounds=true function pw_reset_queue_kernel!(queue_size)
     queue_size[1] = Int32(0)
end

"""
    reset_queue!(backend, queue::PWWorkQueue)

Reset queue using a kernel (GPU-safe).
"""
function reset_queue!(backend, queue::PWWorkQueue)
    kernel = pw_reset_queue_kernel!(backend)
    kernel(queue.size; ndrange=1)
    return nothing
end

# ============================================================================
# Specialized Queue Type Aliases
# ============================================================================

"""Ray queue for main path tracing rays."""
const PWRayQueue{I, C} = PWWorkQueue{PWRayWorkItem, I, C}

"""Queue for escaped rays (environment light evaluation)."""
const PWEscapedRayQueue{I, C} = PWWorkQueue{PWEscapedRayWorkItem, I, C}

"""Queue for rays that hit area lights."""
const PWHitAreaLightQueue{I, C} = PWWorkQueue{PWHitAreaLightWorkItem, I, C}

"""Queue for shadow rays (direct lighting visibility)."""
const PWShadowRayQueue{I, C} = PWWorkQueue{PWShadowRayWorkItem, I, C}

"""Queue for material evaluation work items."""
const PWMaterialEvalQueue{I, C} = PWWorkQueue{PWMaterialEvalWorkItem, I, C}

# ============================================================================
# Convenience Push Functions
# ============================================================================

"""
    push_camera_ray!(queue, ray, lambda, pixel_index)

Push a new camera ray to the queue with default path state.
"""
@propagate_inbounds function push_camera_ray!(
    queue::PWWorkQueue{PWRayWorkItem},
    ray::Raycore.Ray,
    lambda::Wavelengths,
    pixel_index::Int32
)
    item = PWRayWorkItem(ray, lambda, pixel_index)
    return push_work!(queue, item)
end

"""
    push_indirect_ray!(queue, ray, depth, prev_ctx, beta, r_u, r_l, lambda,
                       eta_scale, specular_bounce, any_non_specular, pixel_index)

Push an indirect (bounced) ray to the queue.
"""
@propagate_inbounds function push_indirect_ray!(
    queue::PWWorkQueue{PWRayWorkItem},
    ray::Raycore.Ray,
    depth::Int32,
    prev_ctx::PWLightSampleContext,
    beta::SpectralRadiance,
    r_u::SpectralRadiance,
    r_l::SpectralRadiance,
    lambda::Wavelengths,
    eta_scale::Float32,
    specular_bounce::Bool,
    any_non_specular::Bool,
    pixel_index::Int32
)
    item = PWRayWorkItem(
        ray,
        depth,
        lambda,
        pixel_index,
        beta,
        r_u,
        r_l,
        prev_ctx,
        eta_scale,
        specular_bounce,
        any_non_specular
    )
    return push_work!(queue, item)
end

"""
    push_shadow_ray!(queue, ray, t_max, lambda, Ld, r_u, r_l, pixel_index)

Push a shadow ray for direct lighting visibility test.
"""
@propagate_inbounds function push_shadow_ray!(
    queue::PWWorkQueue{PWShadowRayWorkItem},
    ray::Raycore.Ray,
    t_max::Float32,
    lambda::Wavelengths,
    Ld::SpectralRadiance,
    r_u::SpectralRadiance,
    r_l::SpectralRadiance,
    pixel_index::Int32
)
    item = PWShadowRayWorkItem(ray, t_max, lambda, Ld, r_u, r_l, pixel_index)
    return push_work!(queue, item)
end

"""
    push_escaped_ray!(queue, ray_work_item)

Push an escaped ray work item (created from a ray that missed geometry).
"""
@propagate_inbounds function push_escaped_ray!(
    queue::PWWorkQueue{PWEscapedRayWorkItem},
    w::PWRayWorkItem
)
    item = PWEscapedRayWorkItem(w)
    return push_work!(queue, item)
end

"""
    push_hit_area_light!(queue, p, n, uv, wo, lambda, depth, beta, r_u, r_l,
                         prev_ctx, specular_bounce, pixel_index, material_idx)

Push a hit area light work item (ray hit an emissive surface).
"""
@propagate_inbounds function push_hit_area_light!(
    queue::PWWorkQueue{PWHitAreaLightWorkItem},
    p::Point3f,
    n::Vec3f,
    uv::Point2f,
    wo::Vec3f,
    lambda::Wavelengths,
    depth::Int32,
    beta::SpectralRadiance,
    r_u::SpectralRadiance,
    r_l::SpectralRadiance,
    prev_ctx::PWLightSampleContext,
    specular_bounce::Bool,
    pixel_index::Int32,
    material_idx::MaterialIndex
)
    item = PWHitAreaLightWorkItem(
        p, n, uv, wo, lambda, depth, beta, r_u, r_l,
        prev_ctx, specular_bounce, pixel_index, material_idx
    )
    return push_work!(queue, item)
end

"""
    push_material_eval!(queue, pi, n, dpdu, dpdv, ns, dpdus, dpdvs, uv,
                        depth, lambda, pixel_index, any_non_specular,
                        wo, beta, r_u, eta_scale, material_idx)

Push a material evaluation work item.
"""
@propagate_inbounds function push_material_eval!(
    queue::PWWorkQueue{PWMaterialEvalWorkItem},
    pi::Point3f,
    n::Vec3f,
    dpdu::Vec3f,
    dpdv::Vec3f,
    ns::Vec3f,
    dpdus::Vec3f,
    dpdvs::Vec3f,
    uv::Point2f,
    depth::Int32,
    lambda::Wavelengths,
    pixel_index::Int32,
    any_non_specular::Bool,
    wo::Vec3f,
    beta::SpectralRadiance,
    r_u::SpectralRadiance,
    eta_scale::Float32,
    material_idx::MaterialIndex
)
    item = PWMaterialEvalWorkItem(
        pi, n, dpdu, dpdv, ns, dpdus, dpdvs, uv,
        depth, lambda, pixel_index, any_non_specular,
        wo, beta, r_u, eta_scale, material_idx
    )
    return push_work!(queue, item)
end

# ============================================================================
# Queue Processing Kernel Infrastructure
# ============================================================================

"""
    for_all_queued_kernel!(queue_items, queue_size, max_queued, process_func, args...)

Generic kernel that processes all items in a queue.
Each thread checks if its index is within the current queue size.
"""
@kernel inbounds=true function for_all_queued_kernel!(
    queue_items,
    queue_size,
    @Const(max_queued),
    process_func,
    args...
)
    idx = @index(Global)

     if idx <= max_queued
        # Check if this index has a valid item
        current_size = queue_size[1]
        if idx <= current_size
            item = queue_items[idx]
            process_func(item, args...)
        end
    end
end

"""
    process_queue!(backend, queue::PWWorkQueue, process_func, args...; workgroup_size=256)

Process all items in a queue using the given function.
"""
function process_queue!(backend, queue::PWWorkQueue, process_func, args...; workgroup_size=256)
    n = queue_size(queue)
    n == 0 && return nothing

    kernel = for_all_queued_kernel!(backend, workgroup_size)
    kernel(queue.items, queue.size, n, process_func, args...; ndrange=n)
    return nothing
end
