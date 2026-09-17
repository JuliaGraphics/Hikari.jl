# Where a render state's device memory comes from, and the only thing that gives
# it back.
#
# One place, and not `KA.allocate` with `finalize` scattered across the state,
# the work queues, the light BVH upload and the Sobol matrices. That is the
# pattern the project rule forbids in as many words: memory management is
# centralized, and high-level code never finalizes GPU resources. What it costs
# otherwise is a lifetime protocol maintained by hand — a bare reassign racing
# GC finalizers and doubling peak memory, a proactive free from a finalizer
# taking arrays the render loop is still using.
#
# Here every allocation is a `Mantle.Buffer` from the device's pool, and the
# whole tree goes back in one call. Three things follow that are not just
# tidiness:
#
#   * one allocator sees the renderer's memory. `Mantle.peakbytes` and the pool's
#     own accounting now cover ~1.9 GiB of work queues that were invisible to it.
#   * nothing in this package finalizes a GPU resource. A missed release is a
#     leak the pool can report, not a use-after-free.
#   * the release order stops mattering. `free!` walks one list.
#
# What the buffers hold is unchanged: `Mantle.storage` hands back a `LavaArray`
# view that owns nothing, so the queues and accumulators are the same arrays the
# kernels always saw, and `Adapt`, `use!` and every `@kernel` are untouched.

"""
    DeviceMemory(backend)

The device allocations one `VolPathState` owns.

Constructed from a KernelAbstractions backend because that is what a renderer
has; it resolves the Mantle device once and allocates from that device's pool
thereafter. One per state, and it dies with the state.
"""
mutable struct DeviceMemory
    dev::Any
    backend::Any
    owned::Vector{Any}
end
function DeviceMemory(backend)
    mem = DeviceMemory(mantle_device(backend), backend, Any[])
    # The safety net, not the plan. `free!` is still how memory goes back, at a
    # point the caller chose. This only decides what happens when it is NOT
    # called: the regions are retired rather than lost, and `Mantle.reclaim!`
    # releases them once the device is done. The finalizer never touches a free
    # list — it appends under a lock, which is the whole reason it is allowed to
    # exist at all.
    #
    # On this type and not on `Mantle.Buffer`, which deliberately has no
    # finalizer: `Mantle.storage` hands out a view that owns nothing, so
    # finalizing a `Buffer` would turn "kept the view, dropped the buffer" into
    # a use-after-free. A `DeviceMemory` outlives every view taken from it by
    # construction — whoever holds one of those arrays holds the state that
    # holds this — so it is the level where a finalizer is safe.
    finalizer(free!, mem)
    return mem
end

"Keep the buffer, hand back the array a kernel takes."
own!(mem::DeviceMemory, b) = (push!(mem.owned, b); Mantle.storage(b))

"""
    alloc!(mem, T, n) -> array
    alloc!(mem, T, n, value) -> array filled with `value`

`n` elements of `T` from the pool, as the array a kernel takes. The `Buffer`
behind it is kept by `mem`; what comes back is a view that owns nothing, so
dropping it does nothing and `free!(mem)` is the only release.
"""
alloc!(mem::DeviceMemory, ::Type{T}, n::Integer) where {T} =
    own!(mem, Mantle.Buffer(mem.dev, T, max(Int(n), 1)))

"""
    alloc!(mem, T, dims::Dims) -> array

`dims` elements of `T`, keeping the shape. What a film's layers are: a
framebuffer indexed by pixel is a matrix, and flattening it here would put the
width back into every kernel that writes one.
"""
alloc!(mem::DeviceMemory, ::Type{T}, dims::Dims) where {T} =
    own!(mem, Mantle.Buffer(mem.dev, T, dims))

function alloc!(mem::DeviceMemory, ::Type{T}, dims::Dims, value) where {T}
    a = alloc!(mem, T, dims)
    KA.fill!(a, convert(T, value))
    return a
end

function alloc!(mem::DeviceMemory, ::Type{T}, n::Integer, value) where {T}
    a = alloc!(mem, T, n)
    KA.fill!(a, convert(T, value))
    return a
end

"""
    upload!(mem, data) -> array

`data` on the device, in memory this state owns. The host-side counterpart of
[`alloc!`](@ref), for a table that is computed once and read for ever.

The capacity floor is not defensive: a kernel indexes the light BVH and the
infinite-light list unconditionally, and a zero-length array has no address to
give it. `Mantle.Buffer` uploads nothing when `data` is empty, so an empty table
costs one element and reads as garbage that nothing looks at.
"""
upload!(mem::DeviceMemory, data::AbstractVector) =
    own!(mem, Mantle.Buffer(mem.dev, data; capacity = max(length(data), 1)))

"""
    upload!(mem, data::AbstractArray) -> array

The same, keeping `data`'s shape. The spectral tables are the reason: the sRGB
coefficient table is 5-D and is indexed as such by every kernel that converts a
colour, so flattening it here would put the strides back into the shader.
"""
upload!(mem::DeviceMemory, data::AbstractArray) = own!(mem, Mantle.Buffer(mem.dev, data))

"""
    free!(mem::DeviceMemory)

Give every region back.

**No precondition, and that is the point.** `Mantle.free!` retires: the bytes go
back on a free list when the device says it is finished, not when this returns.
So there is no "the GPU must be idle" rule here for a caller to get wrong, and
no `KA.synchronize` guarding any of it — one before every plan rebuild, every
queue rebuild and every teardown is one more place to forget.

Safe from the GC thread for the same reason, which is why it can be the
finalizer as well as the explicit path. There is no separate `retire!`: this was
it.
"""
function free!(mem::DeviceMemory)
    for r in mem.owned
        Mantle.free!(r)
    end
    empty!(mem.owned)
    return nothing
end

"""How many device allocations this state is holding. For tests that assert a
render loop does not grow one per frame."""
nallocations(mem::DeviceMemory) = length(mem.owned)
