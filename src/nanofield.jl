"""
    load_nanotree(path; backend=nothing)

Read a ZIP-compressed Float NanoVDB grid into a `NanoTree`. With `backend=nothing`
the words stay on the host, so a file-series reader can cache neighbouring files
without allocating GPU storage. A supplied backend uploads the words once.

This representation requires a positive, uniform, axis-aligned voxel transform.
Other transforms are rejected; use `NanoVDBMedium` for general affine grids.
"""
function load_nanotree(path::AbstractString; backend=nothing)
    buffer, meta = parse_nanovdb_buffer(String(path))
    meta.leaf_count > 0 || throw(ArgumentError("NanoTree requires at least one leaf"))
    reinterpret(UInt32, buffer[637:640])[1] == 1 ||
        throw(ArgumentError("NanoTree requires a Float NanoVDB grid"))
    inv = meta.inv_mat
    scale = inv[1]
    isfinite(scale) && scale > 0 &&
        all(i -> isapprox(inv[i], scale; rtol=1f-6), (5, 9)) &&
        all(i -> iszero(inv[i]), (2, 3, 4, 6, 7, 8)) ||
        throw(ArgumentError("NanoTree requires a positive uniform axis-aligned transform"))
    offsets = (meta.upper_offset, meta.lower_offset, meta.leaf_offset)
    all(o -> o >= meta.root_offset && (o-meta.root_offset) % 4 == 0, offsets) ||
        throw(ArgumentError("Unsupported NanoVDB node layout"))
    stop = meta.leaf_offset + Int(meta.leaf_count) * LEAFDATA_SIZE - 1
    stop <= length(buffer) || throw(ArgumentError("Truncated NanoVDB leaves"))
    words = copy(reinterpret(UInt32, buffer[meta.root_offset:stop]))
    voxel = Float32(1 / scale)
    lo = Point3f(meta.vec) .- Vec3f(0.5f0 * voxel)
    background = wordfloat(words, Int32(ROOTDATA_BACKGROUND_OFFSET ÷ 4))
    storage = backend === nothing ? words : Mantle.devicearray(backend, words)
    return NanoTree(storage, Int32.((offsets .- meta.root_offset) .÷ 4)...,
        meta.upper_count, meta.lower_count, meta.leaf_count, meta.root_table_size,
        Point3f(lo), voxel, background, Vec{3,Int32}(meta.index_min), Vec{3,Int32}(meta.index_max))
end

"""
    NanoBlend(a::NanoTree, b::NanoTree, weight)

Two independently positioned NanoVDB fields, interpolated at the sampled world
position. Missing voxels use each grid's own background. This descriptor retains
the two input buffers and never constructs a third voxel array. `weight` must be
finite and in `[0,1]`. A file-input consumer can pass its `a`, `b` and `weight`
directly here without introducing a dependency on an editor package.
"""
struct NanoBlend{A<:NanoTree,B<:NanoTree}
    a::A
    b::B
    weight::Float32
    function NanoBlend(a::A, b::B, weight::Real) where {A<:NanoTree,B<:NanoTree}
        isfinite(weight) && 0 <= weight <= 1 || throw(ArgumentError("Invalid NanoVDB blend weight"))
        return new{A,B}(a, b, Float32(weight))
    end
end
Adapt.@adapt_structure NanoBlend
const NanoField = Union{NanoTree,NanoBlend}

"The same two field buffers with a new interpolation weight."
with_blend_weight(t::NanoBlend, weight::Real) = NanoBlend(t.a,t.b,weight)

@inline nanofield_sample(t::NanoTree, media, p) = nanosample(t, Raycore.deref(media, t.words), p)
@inline nanofield_sample(t::NanoTree, ::Nothing, p) = nanosample(t, t.words, p)
@inline function nanofield_sample(t::NanoBlend, media, p)
    # Exact endpoints also avoid touching a file that contributes no density.
    t.weight == 0f0 && return nanofield_sample(t.a, media, p)
    t.weight == 1f0 && return nanofield_sample(t.b, media, p)
    return (1f0-t.weight)*nanofield_sample(t.a, media, p) + t.weight*nanofield_sample(t.b, media, p)
end

function nanobounds(t::NanoTree)
    return Bounds3(Point3f(t.lo + (Vec3f(t.imin) .- 0.5f0)*t.voxel),
                   Point3f(t.lo + (Vec3f(t.imax) .+ 0.5f0)*t.voxel))
end
function nanobounds(t::NanoBlend)
    a, b = nanobounds(t.a), nanobounds(t.b)
    return Bounds3(Point3f(min.(a.p_min,b.p_min)), Point3f(max.(a.p_max,b.p_max)))
end

function nanomajorant(t::NanoBlend, bounds::Bounds3, res::Vec{3,Int32}; tight::Bool=true)
    a = nanomajorant(t.a, bounds, res; tight)
    b = nanomajorant(t.b, bounds, res; tight)
    # Bound every possible weight for this pair. The acceleration data can stay
    # unchanged while a keyframed weight moves in either direction.
    a.voxels .= max.(a.voxels, b.voxels)
    return a
end
