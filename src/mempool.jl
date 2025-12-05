struct MemoryPool
    bytes::Vector{UInt8}
    last_alloc::Base.RefValue{Int}
end

function MemoryPool(max_size::Int)
    MemoryPool(Vector{UInt8}(undef, max_size), Base.RefValue(0))
end

function free_all(pool::MemoryPool)
    pool.last_alloc[] = 0
end

last_alloc(pool::MemoryPool) = pool.last_alloc[]

# pointer need to be byte8 aligned, so our views need to start at 8 aligned indices
function next_byte8_aligned(index)
    m = mod(index, 8)
    m === 1 && return index
    m === 0 && return index + 1
    return index - m + 8 + 1
end

"""
    next_free_segment(pool::MemoryPool, bytes_to_allocate::Int)

Returns the next free segment in `pool`. The segment returned is a byte8 aligned index
to where the segment starts and stops `(start, stop)::Tuple{Int, Int}`.
If pool is full, returns nothing.
"""
function next_free_segment(pool::MemoryPool, bytes_to_allocate::Int)::Union{Nothing,Tuple{Int,Int}}

    len = length(pool.bytes)
    last_free = next_byte8_aligned(last_alloc(pool) + 1)
    segment_stop = last_free + bytes_to_allocate - 1
    # When we have enough room in pool give out next segment
    if segment_stop <= len
        segment = (last_free, segment_stop)
        pool.last_alloc[] = segment_stop
        return segment
    end
    # we don't have space left
    return nothing
end

struct MutableRef{T}
    ptr::Ptr{UInt8}
end
MutableRef{T}() where {T} = MutableRef{T}(C_NULL)
Base.isnothing(x::MutableRef) = x.ptr == C_NULL
type_pointer(x::MutableRef{T}) where {T} = Base.unsafe_convert(Ptr{T}, getfield(x, :ptr))

function Base.copy(pool::MemoryPool, x::MutableRef{T}) where {T}
    mem = allocate(pool, T)
    dest = type_pointer(mem)
    Base.unsafe_copyto!(dest, type_pointer(x), 1)
    return mem
end

@inline allocate(pool::MemoryPool, ::Type{MutableRef{T}}) where {T} = allocate(pool, T)

# Creates an actual array out of a view into the pool
function reinterpret_segment(pool::MemoryPool, ::Type{T}, start::Int, dims::NTuple{N,Int}) where {T,N}
    @assert isbitstype(T)
    start_ptr = pointer(pool.bytes, start)
    ptr_t = Base.unsafe_convert(Ptr{T}, start_ptr)
    # Note, that this is only safe, since we should never loose the reference to pool.bytes
    # Sadly, allocates ~96 bytes ... Somehow allocates much more in threaded code!?
    return unsafe_wrap(Array, ptr_t, dims)
end

@inline function allocate(pool::MemoryPool, ::Type{<:Array{T}}, dims::NTuple{N}) where {T, N}
    len = prod(dims)
    nbytes = sizeof(T) * len
    segment = next_free_segment(pool, nbytes)
    # If pool is not big enough, we do a gc tracked allocation
    if isnothing(segment)
        return Array{T,N}(undef, dims)
    else
        return reinterpret_segment(pool, T, segment[1], dims)
    end
end

function allocate(pool, obj::T) where {T}
    mem = allocate(pool, T)
    ptr = type_pointer(mem)
    unsafe_store!(ptr, obj)
    return mem
end

@inline function allocate(pool::MemoryPool, ::Type{T}) where {T}
    @assert isbitstype(T) "$T"
    nbytes = sizeof(T)
    segment = next_free_segment(pool, nbytes)
    # If pool is not big enough, we do a gc tracked allocation
    if isnothing(segment)
        error("Out of mem")
        return MutableRef{T}(C_NULL)
    else
        return MutableRef{T}(pointer(pool.bytes, segment[1]))
    end
end

@inline allocate(pool::MemoryPool, ::Type{MutableRef{T}}, args::AT) where {T, AT<:Tuple} = allocate(pool, T, args)

@inline function allocate(pool::MemoryPool, ::Type{T}, args::AT) where {T,AT<:Tuple}
    obj = T(args...)
    mem = allocate(pool, T)
    ptr = Base.unsafe_convert(Ptr{T}, getfield(mem, :ptr))
    unsafe_store!(ptr, obj)
    return mem
end

Base.@propagate_inbounds function Base.setindex!(mref::MutableRef{T}, value, idx::Int) where {T}
    fieldoffset = Base.fieldoffset(T, idx)
    VT = fieldtype(T, idx)
    converted = convert(VT, value)
    ptr = Base.unsafe_convert(Ptr{VT}, getfield(mref, :ptr) + fieldoffset)
    unsafe_store!(ptr, converted)
    return converted
end

Base.@propagate_inbounds function Base.setproperty!(mref::MutableRef{T}, field::Symbol, value) where {T}
    idx = findfirst(x -> x === field, fieldnames(T))
    return mref[idx] = value
end

Base.@propagate_inbounds function Base.getproperty(mref::MutableRef{T}, field::Symbol) where {T}
    idx = findfirst(x -> x === field, fieldnames(T))
    if isnothing(idx)
        error("Field $field not found in type $T")
    end
    return mref[idx]
end

Base.@propagate_inbounds function Base.getindex(mref::MutableRef{T}, idx::Int) where {T}
    fieldoffset = Base.fieldoffset(T, idx)
    VT = fieldtype(T, idx)
    ptr = Ptr{VT}(getfield(mref, :ptr) + fieldoffset)
    return unsafe_load(ptr)
end


function Base.getindex(mref::MutableRef{T}) where {T}
    return unsafe_load(Ptr{T}(getfield(mref, :ptr)))
end

function LifeCycle(f::Function, pool)
    try
        f(pool)
    finally
        pool.last_alloc[] = 0
    end
end
