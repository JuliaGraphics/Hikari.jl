# Unified WorkQueue for wavefront integrators
# GPU-compatible thread-safe work queue with optional SOA layout
#
# This replaces both PWWorkQueue and VPWorkQueue with a single implementation.

using KernelAbstractions
using KernelAbstractions: @kernel, @index
import KernelAbstractions as KA
using Atomix: @atomic
using StructArrays
using Adapt

# ============================================================================
# SOA/AOS Array Allocation (following pbrt-v4's SOA pattern)
# ============================================================================

# Trait: should this type be decomposed into SOA?
# Override this for work item types that benefit from SOA layout.
# Nested types (Ray, Vec3f, Spectrum, etc.) stay flat since they're small
# and accessed as units. The SOA benefit comes from coalesced access when
# threads read the same field across different work items.
should_use_soa(::Type{T}) where T = false

"""
    allocate_array(mem, T, n; soa=false)

Allocate array with AOS (soa=false) or SOA (soa=true) layout, from the memory
the render state owns. Both support identical indexing: arr[i] returns T,
arr[i] = val stores T.
"""
function allocate_array(mem::DeviceMemory, ::Type{T}, n::Integer; soa::Bool=false) where T
    soa ? allocate_soa(mem, T, n) : alloc!(mem, T, n)
end

function allocate_soa(mem::DeviceMemory, ::Type{T}, n::Integer) where T
    if !should_use_soa(T)
        return alloc!(mem, T, n)
    end
    if fieldcount(T) > 0
        fnames = fieldnames(T)
        ftypes = fieldtypes(T)
        components = NamedTuple{fnames}(
            ntuple(i -> allocate_soa(mem, ftypes[i], n), length(fnames))
        )
        return StructArray{T}(components)
    end
    return alloc!(mem, T, n)
end

# ============================================================================
# WorkQueue - Unified GPU Work Queue
# ============================================================================

"""
    WorkQueue{T, V, S}

A GPU-compatible work queue that stores items of type T.
Uses atomic operations for thread-safe push operations.

The queue stores items in a pre-allocated buffer and uses an atomic
counter to track the current size. Supports both AOS and SOA layouts.

# Fields
- `items::V`: Pre-allocated array of work items
- `size::S`: Single-element array for atomic counter
- `capacity::Int32`: Maximum number of items

# Example
```julia
# Create a queue in the memory a render state owns
queue = WorkQueue{MyWorkItem}(mem, 1024)

# In a kernel, push items atomically
idx = push!(queue, item)

# Check if push succeeded (within capacity)
if idx <= length(queue.items)
    # item was stored at queue.items[idx]
end
```
"""
struct WorkQueue{T, V <: AbstractVector{T}, S <: AbstractVector{Int32}}
    items::V
    size::S      # Single-element array for atomic operations
    capacity::Int32
end

# A queue has no `free!` of its own: its arrays are views into regions the
# state's `DeviceMemory` owns, so `free!(mem)` releases them and there is nothing
# per-queue to remember. Walking the SOA components to `finalize` each is the
# alternative, because `finalize(::StructArray)` is a no-op and would leave them
# alive until the GC ran: one more piece of a lifetime protocol that belongs to
# the pool.

"""
    WorkQueue{T}(mem::DeviceMemory, capacity; soa=should_use_soa(T))

Create a new work queue with the given capacity, in memory the render state
owns.

# Arguments
- `mem`: the state's [`DeviceMemory`](@ref); `free!(mem)` releases this queue
- `capacity`: Maximum number of items the queue can hold
- `soa`: If true, use Structure-of-Arrays layout for better GPU memory coalescing.
  Defaults to `should_use_soa(T)` so flagged item types automatically get the
  faster layout without every caller having to remember the kwarg. (Before
  this default was wired up, the trait was set on `VPRayWorkItem` and friends
  but never actually read — every queue was AOS regardless.)
"""
function WorkQueue{T}(mem::DeviceMemory, capacity::Integer; soa::Bool=should_use_soa(T)) where T
    items = allocate_array(mem, T, capacity; soa=soa)
    size = alloc!(mem, Int32, 1, Int32(0))
    WorkQueue{T, typeof(items), typeof(size)}(items, size, Int32(capacity))
end

# ============================================================================
# Queue Operations
# ============================================================================

function Base.length(queue::WorkQueue)
    s = Array(queue.size)
    return Int(s[1])
end

@inline function Base.push!(queue::WorkQueue{T}, item::T) where T
    # Atomically increment size and get index
    idx = @atomic queue.size[1] += Int32(1)
    # Only store if within bounds
    if idx <= length(queue.items)
        @inbounds queue.items[idx] = item
    end
    return idx
end

@propagate_inbounds function Base.getindex(queue::WorkQueue, idx::Integer)
    return queue.items[idx]
end

@propagate_inbounds function Base.setindex!(queue::WorkQueue{T}, item::T, idx::Integer) where T
    queue.items[idx] = item
    return item
end

function Base.empty!(queue::WorkQueue)
    fill!(queue.size, Int32(0))
    return queue
end


# ============================================================================
# Adapt.jl Integration for GPU Kernels
# ============================================================================

"""
    Adapt.adapt_structure(backend, queue::WorkQueue)

Adapt WorkQueue for use inside GPU kernels. This converts the host-side
arrays (e.g., CLArray, CuArray) to device-compatible representations
(e.g., CLDeviceArray, CuDeviceArray) that can be used inside kernels.

This allows passing the entire WorkQueue to a kernel instead of
passing items and size arrays separately.
"""
function Adapt.adapt_structure(backend, queue::WorkQueue)
    WorkQueue(
        Adapt.adapt(backend, queue.items),
        Adapt.adapt(backend, queue.size),
        queue.capacity
    )
end

# ============================================================================
# Map Operations for GPU Kernel Execution
# ============================================================================
@kernel function workqueue_map_kernel!(f, queue, args...)
    i = @index(Global)
    if i <= queue.size[1]
        @inbounds f(queue.items[i], args...)
    end
end

# Default workgroupsize=256 gives ~14% speedup on CUDA (Ampere) vs auto-selection.
# Static workgroupsize helps GPU compilers optimize register allocation and enables
# more concurrent blocks per SM.
#
# How a queue is dispatched over is `Mantle.DeviceRange(q.size; max = q.capacity)`
# in the graph — the count lives on the device, the backend turns it into
# workgroups there, and the graph orders the stage after whoever wrote the count.
const DEFAULT_WORKGROUPSIZE = 256

# ============================================================================
# MultiTypeWorkQueue — one WorkQueue per concrete item type
# ============================================================================
#
# Mirrors pbrt-v4's per-material-type queue pattern (MaterialEvalQueue<T> in
# wavefront/workitems.h). One `WorkQueue{ItemFor[T]}` per concrete type T in
# the scene. The trace kernel routes each work item into the queue for its
# concrete type via `with_index` dispatch; the shading stage drains all of them
# — one dispatch per type, all in one graph pass, since they are independent.
#
# Each per-type kernel is fully monomorphised by Julia (no `with_index`
# switch inside the kernel), so the resulting SPIR-V is small and the warps
# never diverge on material-type branches.

"""
    MultiTypeWorkQueue{Qs <: Tuple}

Heterogeneous tuple of `WorkQueue`s, one per concrete item type. Iteration
order matches the type-tuple order, and the number of queues is known at
compile time, so a stage that dispatches over all of them unrolls with full
type stability per arm.

# Fields
- `queues::Qs`: NTuple of WorkQueue, one per concrete item type
"""
struct MultiTypeWorkQueue{Qs <: Tuple}
    queues::Qs
end

"""
    MultiTypeWorkQueue(item_types::Tuple, capacity, backend; soa=false)

Build a MultiTypeWorkQueue with one `WorkQueue{T}(backend, capacity)` per `T`
in `item_types`.

```julia
mtwq = MultiTypeWorkQueue((HitWorkA, HitWorkB, HitWorkC), 1024, backend)
```
"""
function MultiTypeWorkQueue(item_types::Tuple, capacity::Integer, mem::DeviceMemory; soa::Bool=false)
    qs = map(T -> WorkQueue{T}(mem, capacity; soa=soa), item_types)
    return MultiTypeWorkQueue(qs)
end

Base.empty!(mtwq::MultiTypeWorkQueue) = (foreach(empty!, mtwq.queues); mtwq)

# Adapt walks into the tuple so each per-queue Adapt.adapt_structure runs
# and the kernel sees device-side arrays.
function Adapt.adapt_structure(backend, mtwq::MultiTypeWorkQueue)
    MultiTypeWorkQueue(map(q -> Adapt.adapt(backend, q), mtwq.queues))
end

# ============================================================================
# Batched counter reset — one dispatch for all queues
# ============================================================================
#
# `empty!(queue)` is a `fill!` on a 1-element array → one GPU dispatch (plus
# barrier) per queue. The volpath bounce loop resets ~20 queues per round;
# at ~20µs effective cost per command that was ~0.4ms of pure overhead per
# round (measured on Crown: 27 single-thread fills per round). One kernel
# writing every counter replaces all of them.

@inline _zero_counters!(::Tuple{}) = nothing
@inline function _zero_counters!(counters::Tuple)
    @inbounds counters[1][1] = Int32(0)
    _zero_counters!(Base.tail(counters))
    return nothing
end

@kernel function zero_size_counters_kernel!(counters)
    i = @index(Global)
    if i == 1
        _zero_counters!(counters)
    end
end

# Flatten WorkQueues / MultiTypeWorkQueues into a tuple of size-counter arrays.
_collect_size_counters(acc::Tuple) = acc
_collect_size_counters(acc::Tuple, q::WorkQueue, rest...) =
    _collect_size_counters((acc..., q.size), rest...)
_collect_size_counters(acc::Tuple, m::MultiTypeWorkQueue, rest...) =
    _collect_size_counters((acc..., map(q -> q.size, m.queues)...), rest...)

# ============================================================================
# Convenience Aliases
# ============================================================================

# For backwards compatibility during migration
const GPUWorkQueue = WorkQueue
