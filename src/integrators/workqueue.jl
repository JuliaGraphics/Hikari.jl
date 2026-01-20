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
# Backend Utilities
# ============================================================================

"""
    get_array_type(backend) -> Type

Get the concrete array type for a KernelAbstractions backend.
Returns the wrapper type (e.g., `CuArray`, `ROCArray`, `Array`).
"""
function get_array_type(backend)
    dummy = KA.allocate(backend, Float32, 1)
    ArrayType = typeof(dummy).name.wrapper
    finalize(dummy)  # Clean up immediately
    return ArrayType
end

"""
    transfer(backend, array::AbstractArray) -> AbstractArray

Transfer a CPU array to the specified backend.
Allocates memory on the backend and copies the data.

This is a convenience function for the common pattern:
```julia
gpu_array = KA.allocate(backend, eltype(array), size(array)...)
KA.copyto!(backend, gpu_array, array)
```

# Example
```julia
cpu_data = rand(Float32, 100)
gpu_data = transfer(CUDABackend(), cpu_data)
```
"""
function transfer(backend, array::AbstractArray{T,N}) where {T,N}
    gpu_array = KA.allocate(backend, T, size(array)...)
    KA.copyto!(backend, gpu_array, array)
    return gpu_array
end

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
    allocate_array(backend, T, n; soa=false)

Allocate array with AOS (soa=false) or SOA (soa=true) layout.
Both support identical indexing: arr[i] returns T, arr[i] = val stores T.
"""
function allocate_array(backend, ::Type{T}, n::Integer; soa::Bool=false) where T
    soa ? _allocate_soa(backend, T, n) : KA.allocate(backend, T, n)
end

function _allocate_soa(backend, ::Type{T}, n::Integer) where T
    if !should_use_soa(T)
        return KA.allocate(backend, T, n)
    end
    if fieldcount(T) > 0
        fnames = fieldnames(T)
        ftypes = fieldtypes(T)
        components = NamedTuple{fnames}(
            ntuple(i -> _allocate_soa(backend, ftypes[i], n), length(fnames))
        )
        return StructArray{T}(components)
    end
    return KA.allocate(backend, T, n)
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
# Create a queue on GPU backend
queue = WorkQueue{MyWorkItem}(backend, 1024)

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

"""
    WorkQueue{T}(backend, capacity; soa=false)

Create a new work queue with the given capacity on the specified backend.

# Arguments
- `backend`: KernelAbstractions backend (e.g., `CPU()`, `CUDABackend()`, `ROCBackend()`)
- `capacity`: Maximum number of items the queue can hold
- `soa`: If true, use Structure-of-Arrays layout for better GPU memory coalescing
"""
function WorkQueue{T}(backend, capacity::Integer; soa::Bool=false) where T
    items = allocate_array(backend, T, capacity; soa=soa)
    size = KA.allocate(backend, Int32, 1)
    KA.fill!(size, Int32(0))
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

function cleanup!(queue::WorkQueue)
    finalize(queue.items)
    finalize(queue.size)
    return nothing
end

# ============================================================================
# Adapt.jl Integration for GPU Kernels
# ============================================================================

"""
    Adapt.adapt_structure(to, queue::WorkQueue)

Adapt WorkQueue for use inside GPU kernels. This converts the host-side
arrays (e.g., CLArray, CuArray) to device-compatible representations
(e.g., CLDeviceArray, CuDeviceArray) that can be used inside kernels.

This allows passing the entire WorkQueue to a kernel instead of
passing items and size arrays separately.
"""
function Adapt.adapt_structure(to, queue::WorkQueue)
    WorkQueue(
        Adapt.adapt(to, queue.items),
        Adapt.adapt(to, queue.size),
        queue.capacity
    )
end

# ============================================================================
# Map Operations for GPU Kernel Execution
# ============================================================================
@kernel function _workqueue_map_kernel!(f, queue, args)
    i = @index(Global)
    if i <= queue.size[1]
        @inbounds f(queue.items[i], args...)
    end
end

function Base.foreach(f, queue::WorkQueue, args...; workgroupsize=nothing)
    n = length(queue)
    n == 0 && return nothing
    backend = KA.get_backend(queue.items)
    kernel! = _workqueue_map_kernel!(backend)
    if workgroupsize === nothing
        kernel!(f, queue, args; ndrange=n)
    else
        kernel!(f, queue, args; ndrange=n, workgroupsize=workgroupsize)
    end
    return nothing
end

# ============================================================================
# Convenience Aliases
# ============================================================================

# For backwards compatibility during migration
const GPUWorkQueue = WorkQueue
