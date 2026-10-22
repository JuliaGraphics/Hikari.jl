# A NanoVDB float grid made on the device, a frame at a time.
#
# `build_nanovdb_from_dense` (integrators/volpath/nanovdb.jl) makes a grid on the host
# from a dense array, for a cloud read once. Water, foam, smoke and fire change every
# frame and come from the device already sparse: a set of 8³ blocks. `NanoTree` lays
# those out as NanoVDB does (Museth 2021) — root table, upper (32³) and lower (16³)
# internal nodes, leaves (8³) — in one device buffer of 32-bit words, every node and
# mask where NanoVDB keeps it, so the same bytes save as a `.nvdb` file (`save_nanovdb`)
# that NanoVDB, OpenVDB and pbrt-v4 read. The tree is built by a few passes over dense
# flag maps of the nodes (no sorting, no host round trip of the voxels), and read by
# `nanovalue`, NanoVDB's `getValue`.

"Words per node and offsets in words of the fields this file writes and reads (the byte constants of nanovdb.jl over 4)."
const NT_LEAF_WORDS = Int32(LEAFDATA_SIZE ÷ 4)
const NT_LEAF_MASK = Int32(LEAFDATA_MASK_OFFSET ÷ 4)
const NT_LEAF_MIN = Int32(LEAFDATA_MIN_OFFSET ÷ 4)
const NT_LEAF_VALUES = Int32(LEAFDATA_VALUES_OFFSET ÷ 4)
const NT_LOWER_WORDS = Int32(LOWER_NODE_SIZE ÷ 4)
const NT_LOWER_VALUEMASK = Int32(LOWER_VALUEMASK_OFFSET ÷ 4)
const NT_LOWER_CHILDMASK = Int32(LOWER_CHILDMASK_OFFSET ÷ 4)
const NT_LOWER_MIN = Int32(LOWER_MIN_OFFSET ÷ 4)
const NT_LOWER_TABLE = Int32(LOWER_TABLE_OFFSET ÷ 4)
const NT_UPPER_WORDS = Int32(UPPER_NODE_SIZE ÷ 4)
const NT_UPPER_VALUEMASK = Int32(UPPER_VALUEMASK_OFFSET ÷ 4)
const NT_UPPER_CHILDMASK = Int32(UPPER_CHILDMASK_OFFSET ÷ 4)
const NT_UPPER_MIN = Int32(UPPER_MIN_OFFSET ÷ 4)
const NT_UPPER_TABLE = Int32(UPPER_TABLE_OFFSET ÷ 4)
const NT_ROOT_HEADER = Int32(ROOTDATA_HEADER_SIZE ÷ 4)
const NT_ROOT_TILE = Int32(ROOTTILE_SIZE ÷ 4)

"""
    NanoTree

A NanoVDB float grid in a device buffer of 32-bit words (`words`, adapted to the
device with the tree): where its nodes start (in words, from 0), how many there are,
and its voxels' placement in the world, `voxel` metres apart, the centre of voxel
`(0, 0, 0)` half a voxel inside `lo`. Values outside every leaf and tile are
`background`.
"""
struct NanoTree{W<:AbstractVector{UInt32}}
    words::W
    upper::Int32          # first upper node
    lower::Int32          # first lower node
    leaf::Int32           # first leaf
    uppers::Int32
    lowers::Int32
    leaves::Int32
    roottiles::Int32
    lo::Point3f
    voxel::Float32
    background::Float32
    imin::Vec{3,Int32}    # index bounding box of the leaves (inclusive min, exclusive max)
    imax::Vec{3,Int32}
end

"""
    nanomajorant(tree, bounds, res) -> MajorantGrid

Conservative maxima of a nonnegative, trilinearly sampled grid in `res` cells of
`bounds`. A leaf contributes its maximum wherever its interpolation support reaches.
Overlapping leaves of one field take a maximum; independent fields can then be added.
"""
function nanomajorant(t::NanoTree, bounds::Bounds3, res::Vec{3,Int32}; tight::Bool=true)
    grid = MajorantGrid(res, Vector{Float32})
    fill!(grid.voxels, max(0f0, t.background))
    lo = Point3f(bounds.p_min)
    cell = Vec3f(bounds.p_max - bounds.p_min) ./ Vec3f(res)
    words = Array(t.words)
    for l in 1:Int(t.leaves)
        at = t.leaf + (l - 1) * NT_LEAF_WORDS
        origin = Vec3f(reinterpret(Int32, words[at + 1]), reinterpret(Int32, words[at + 2]),
                       reinterpret(Int32, words[at + 3]))
        top = max(0f0, reinterpret(Float32, words[at + NT_LEAF_MIN + 2]))
        top > 0f0 || continue
        if tight
            # Trilinear interpolation is a convex combination. Each nonzero
            # sample bounds exactly the cells overlapping its tent support,
            # rather than inflating an entire 8³ leaf to its peak density.
            for i in 0:7, j in 0:7, k in 0:7
                v = max(0f0, wordfloat(words, at + NT_LEAF_VALUES + 64i + 8j + k))
                v > 0f0 || continue
                centre = t.lo + (origin + Vec3f(i,j,k) .+ 0.5f0) * t.voxel
                a = centre .- (1.00001f0 * t.voxel)
                b = centre .+ (1.00001f0 * t.voxel)
                (any(b .< lo) || any(a .> bounds.p_max)) && continue
                c0 = clamp.(floor.(Int32, (a-lo)./cell), Int32(0), res .- Int32(1))
                c1 = clamp.(floor.(Int32, (b-lo)./cell), Int32(0), res .- Int32(1))
                for z in c0[3]:c1[3], y in c0[2]:c1[2], x in c0[1]:c1[1]
                    idx = x + res[1]*(y+res[2]*z)+1
                    grid.voxels[idx] = max(grid.voxels[idx],v)
                end
            end
            continue
        end
        # A sample uses a voxel up to one voxel away from its centre. Round the
        # half-voxel support outwards so boundaries remain conservative in Float32.
        a = t.lo + (origin .- 1f0) * t.voxel
        b = t.lo + (origin .+ 9f0) * t.voxel
        (any(b .< lo) || any(a .> bounds.p_max)) && continue
        c0 = clamp.(floor.(Int32, (a - lo) ./ cell), Int32(0), res .- Int32(1))
        c1 = clamp.(floor.(Int32, (b - lo) ./ cell), Int32(0), res .- Int32(1))
        for k in c0[3]:c1[3], j in c0[2]:c1[2], i in c0[1]:c1[1]
            idx = i + res[1] * (j + res[2] * k) + 1
            grid.voxels[idx] = max(grid.voxels[idx], top)
        end
    end
    return grid
end
Adapt.@adapt_structure NanoTree

"`p` in index space: voxel centres at whole numbers."
@inline indexspace(t::NanoTree, p) = Vec3f(p - t.lo) / t.voxel - Vec3f(0.5f0)
"The world position of index-space point `q`."
@inline worldspace(t::NanoTree, q) = t.lo + (Vec3f(q) + Vec3f(0.5f0)) * t.voxel

@inline wordfloat(w, i) = reinterpret(Float32, @inbounds w[i + 1])
@inline function wordint64(w, i)
    lo = UInt64(@inbounds w[i + 1]); hi = UInt64(@inbounds w[i + 2])
    return reinterpret(Int64, lo | (hi << 32))
end
@inline maskbit(w, mask, n) = (@inbounds(w[mask + (n >> 5) + 1]) >> (UInt32(n) & 0x1f)) & 0x00000001 != 0x00000000

"""
    nanolocate(t::NanoTree, words, ijk) -> (leaf, value)

The first word of the leaf holding voxel `ijk`, or -1 and the value of the tile (or the
background) it lies in.
"""
@inline function nanolocate(t::NanoTree, words, ijk::Vec{3,Int32})
    key = coord_to_root_key((ijk[1], ijk[2], ijk[3]))
    tile = Int32(-1)
    for i in Int32(0):t.roottiles - Int32(1)
        at = NT_ROOT_HEADER + i * NT_ROOT_TILE
        if reinterpret(UInt64, wordint64(words, at)) == key
            tile = at
            break
        end
    end
    tile < Int32(0) && return (Int32(-1), t.background)
    child = wordint64(words, tile + Int32(2))
    child == Int64(0) && return (Int32(-1), wordfloat(words, tile + Int32(5)))
    up = Int32(child ÷ 4)
    n = upper_coord_to_offset((ijk[1], ijk[2], ijk[3]))
    maskbit(words, up + NT_UPPER_CHILDMASK, n) || return (Int32(-1), wordfloat(words, up + NT_UPPER_TABLE + 2n))
    low = up + Int32(wordint64(words, up + NT_UPPER_TABLE + 2n) ÷ 4)
    m = lower_coord_to_offset((ijk[1], ijk[2], ijk[3]))
    maskbit(words, low + NT_LOWER_CHILDMASK, m) || return (Int32(-1), wordfloat(words, low + NT_LOWER_TABLE + 2m))
    return (low + Int32(wordint64(words, low + NT_LOWER_TABLE + 2m) ÷ 4), 0f0)
end

"""
    nanoleafvalue(t::NanoTree, words, ijk) -> (Float32, Bool)

The value at voxel `ijk` (index space, whole numbers), NanoVDB's `getValue` through the
root's table, the upper and the lower node, to the leaf or the tile on the way; and
whether a leaf held it. `words` is `t.words` as the device sees it.
"""
@inline function nanoleafvalue(t::NanoTree, words, ijk::Vec{3,Int32})
    leaf, v = nanolocate(t, words, ijk)
    leaf < Int32(0) && return (v, false)
    return (wordfloat(words, leaf + NT_LEAF_VALUES + leaf_coord_to_offset((ijk[1], ijk[2], ijk[3]))), true)
end

"The value at voxel `ijk` (see [`nanoleafvalue`](@ref))."
@inline nanovalue(t::NanoTree, words, ijk::Vec{3,Int32}) = first(nanoleafvalue(t, words, ijk))

"Four values within one leaf, at offsets `0`, `a`, `b`, and `a+b`."
@inline function nanoquadvalues(t::NanoTree, words, ijk::Vec{3,Int32}, a::Int32, b::Int32)
    leaf, value = nanolocate(t, words, ijk)
    leaf < Int32(0) && return (value, value, value, value)
    base = leaf + NT_LEAF_VALUES + leaf_coord_to_offset((ijk[1], ijk[2], ijk[3]))
    return (wordfloat(words, base), wordfloat(words, base+a),
            wordfloat(words, base+b), wordfloat(words, base+a+b))
end

"""
The world's value at `p`, trilinear between the voxel centres round it. When all eight
lie in one leaf (two thirds of the time with 8³ leaves) the tree is walked once, as
NanoVDB's cached accessor would. Crossing just one leaf face walks its two leaves;
the uncommon footprints crossing two or three leaf faces look up each corner.
"""
@inline function nanosample(t::NanoTree, words, p)
    q = indexspace(t, p)
    i = floor.(Int32, q)
    # Reject only when the entire interpolation footprint is outside the
    # grid's index bounds. The one-cell halo retains cross-boundary values.
    if any(i .+ Int32(1) .< t.imin) || any(i .>= t.imax)
        return t.background
    end
    f = q - Vec3f(i)
    bx = (i[1] & Int32(7)) == Int32(7)
    by = (i[2] & Int32(7)) == Int32(7)
    bz = (i[3] & Int32(7)) == Int32(7)
    if !bx && !by && !bz
        leaf, v = nanolocate(t, words, i)
        leaf < Int32(0) && return v
        base = leaf + NT_LEAF_VALUES + leaf_coord_to_offset((i[1], i[2], i[3]))
        # Within a leaf, x steps 64 values, y 8, z 1 (see `leaf_coord_to_offset`).
        w(d) = wordfloat(words, base + d)
        x00 = w(Int32(0)) * (1f0 - f[1]) + w(Int32(64)) * f[1]
        x10 = w(Int32(8)) * (1f0 - f[1]) + w(Int32(72)) * f[1]
        x01 = w(Int32(1)) * (1f0 - f[1]) + w(Int32(65)) * f[1]
        x11 = w(Int32(9)) * (1f0 - f[1]) + w(Int32(73)) * f[1]
        return ((x00 * (1f0 - f[2]) + x10 * f[2]) * (1f0 - f[3]) + (x01 * (1f0 - f[2]) + x11 * f[2]) * f[3])
    elseif bx && !by && !bz
        a = nanoquadvalues(t, words, i, Int32(8), Int32(1))
        b = nanoquadvalues(t, words, i+Vec{3,Int32}(1,0,0), Int32(8), Int32(1))
        x00 = a[1]*(1f0-f[1]) + b[1]*f[1]
        x10 = a[2]*(1f0-f[1]) + b[2]*f[1]
        x01 = a[3]*(1f0-f[1]) + b[3]*f[1]
        x11 = a[4]*(1f0-f[1]) + b[4]*f[1]
        return ((x00 * (1f0 - f[2]) + x10 * f[2]) * (1f0 - f[3]) + (x01 * (1f0 - f[2]) + x11 * f[2]) * f[3])
    elseif by && !bx && !bz
        a = nanoquadvalues(t, words, i, Int32(64), Int32(1))
        b = nanoquadvalues(t, words, i+Vec{3,Int32}(0,1,0), Int32(64), Int32(1))
        x00 = a[1]*(1f0-f[1]) + a[2]*f[1]
        x10 = b[1]*(1f0-f[1]) + b[2]*f[1]
        x01 = a[3]*(1f0-f[1]) + a[4]*f[1]
        x11 = b[3]*(1f0-f[1]) + b[4]*f[1]
        return ((x00 * (1f0 - f[2]) + x10 * f[2]) * (1f0 - f[3]) + (x01 * (1f0 - f[2]) + x11 * f[2]) * f[3])
    elseif bz && !bx && !by
        a = nanoquadvalues(t, words, i, Int32(64), Int32(8))
        b = nanoquadvalues(t, words, i+Vec{3,Int32}(0,0,1), Int32(64), Int32(8))
        x00 = a[1]*(1f0-f[1]) + a[2]*f[1]
        x10 = a[3]*(1f0-f[1]) + a[4]*f[1]
        x01 = b[1]*(1f0-f[1]) + b[2]*f[1]
        x11 = b[3]*(1f0-f[1]) + b[4]*f[1]
        return ((x00 * (1f0 - f[2]) + x10 * f[2]) * (1f0 - f[3]) + (x01 * (1f0 - f[2]) + x11 * f[2]) * f[3])
    end
    at(a, b, c) = nanovalue(t, words, i + Vec{3,Int32}(a, b, c))
    x00 = at(0, 0, 0) * (1f0 - f[1]) + at(1, 0, 0) * f[1]
    x10 = at(0, 1, 0) * (1f0 - f[1]) + at(1, 1, 0) * f[1]
    x01 = at(0, 0, 1) * (1f0 - f[1]) + at(1, 0, 1) * f[1]
    x11 = at(0, 1, 1) * (1f0 - f[1]) + at(1, 1, 1) * f[1]
    return ((x00 * (1f0 - f[2]) + x10 * f[2]) * (1f0 - f[3]) + (x01 * (1f0 - f[2]) + x11 * f[2]) * f[3])
end

# ─── building ────────────────────────────────────────────────────────────────

"The upper node (from 0, in the dense box of uppers from `u0`, `nu` a side) leaf origin `o` is in."
@inline upperslot(o, u0, nu) = let u = (o .>> Int32(12)) .- u0
    u[1] + nu[1] * (u[2] + nu[2] * u[3])
end
"The lower node's slot (0–32767) in its upper node."
@inline lowerslot(o) = upper_coord_to_offset((o[1], o[2], o[3]))
"The leaf's slot (0–4095) in its lower node."
@inline leafslot(o) = lower_coord_to_offset((o[1], o[2], o[3]))

@kernel inbounds=true function nt_flaguppers_kernel!(flags, @Const(origins), u0::Vec{3,Int32}, nu::Vec{3,Int32})
    l = @index(Global)
    flags[upperslot(origins[l], u0, nu) + 1] = Int32(1)
end
@kernel inbounds=true function nt_flaglowers_kernel!(flags, @Const(upperid), @Const(origins), u0::Vec{3,Int32}, nu::Vec{3,Int32})
    l = @index(Global)
    o = origins[l]
    u = upperid[upperslot(o, u0, nu) + 1] - Int32(1)
    flags[u * Int32(32768) + lowerslot(o) + 1] = Int32(1)
end
@kernel inbounds=true function nt_mapleaves_kernel!(leafof, @Const(upperid), @Const(lowerid), @Const(origins), u0::Vec{3,Int32},
                                                     nu::Vec{3,Int32})
    l = @index(Global)
    o = origins[l]
    u = upperid[upperslot(o, u0, nu) + 1] - Int32(1)
    low = lowerid[u * Int32(32768) + lowerslot(o) + 1] - Int32(1)
    leafof[low * Int32(4096) + leafslot(o) + 1] = Int32(l)
end

@inline function putint64!(w, i, v::Int64)
    u = reinterpret(UInt64, v)
    @inbounds w[i + 1] = UInt32(u & 0xffffffff)
    @inbounds w[i + 2] = UInt32(u >> 32)
end
@inline putfloat!(w, i, v::Float32) = (@inbounds w[i + 1] = reinterpret(UInt32, v))
@inline putcoord!(w, i, c) = for a in 1:3
    @inbounds w[i + a] = reinterpret(UInt32, Int32(c[a]))
end

"Leaf `l`: its origin, a full value mask (every voxel of a leaf is in the band), its values' extremes and its values."
@kernel inbounds=true function nt_leaves_kernel!(w, @Const(origins), @Const(values), first::Int32)
    l = @index(Global)
    at = first + (Int32(l) - Int32(1)) * Int32(NT_LEAF_WORDS)
    putcoord!(w, at, origins[l])
    w[at + 4] = UInt32(0x00070707)          # bbox extent 7, 7, 7; flags 0
    for k in Int32(0):Int32(15)
        w[at + NT_LEAF_MASK + k + 1] = 0xffffffff
    end
    lo = Inf32; hi = -Inf32; sum = 0f0
    base = (Int32(l) - Int32(1)) * Int32(512)
    for v in Int32(1):Int32(512)
        x = values[base + v]
        lo = min(lo, x); hi = max(hi, x); sum += x
        putfloat!(w, at + NT_LEAF_VALUES + v - Int32(1), x)
    end
    putfloat!(w, at + NT_LEAF_MIN, lo); putfloat!(w, at + NT_LEAF_MIN + 1, hi); putfloat!(w, at + NT_LEAF_MIN + 2, sum / 512f0)
end

"""
Lower node `low`'s word `m` of its child and value masks (32 of its 4096 slots), and
those slots' table entries: the offset from the node to its leaf, or the background.
"""
@kernel inbounds=true function nt_lowers_kernel!(w, @Const(leafof), @Const(lowerorigin), first::Int32, leaffirst::Int32,
                                                 background::Float32)
    t = @index(Global)
    low = (Int32(t) - Int32(1)) ÷ Int32(128)
    m = (Int32(t) - Int32(1)) % Int32(128)
    at = first + low * Int32(NT_LOWER_WORDS)
    if m == Int32(0)
        o = lowerorigin[low + 1]
        putcoord!(w, at, o)
        putcoord!(w, at + 3, o + Vec{3,Int32}(127, 127, 127))
    end
    bits = UInt32(0)
    for b in Int32(0):Int32(31)
        n = m * Int32(32) + b
        leaf = leafof[low * Int32(4096) + n + 1]
        if leaf > Int32(0)
            bits |= UInt32(1) << UInt32(b)
            leafat = leaffirst + (leaf - Int32(1)) * Int32(NT_LEAF_WORDS)
            putint64!(w, at + NT_LOWER_TABLE + 2n, Int64(leafat - at) * 4)
        else
            putfloat!(w, at + NT_LOWER_TABLE + 2n, background)
        end
    end
    w[at + NT_LOWER_CHILDMASK + m + 1] = bits
    w[at + NT_LOWER_VALUEMASK + m + 1] = bits
end

"The same for upper node `up`'s word `m` (32 of its 32768 slots), whose children are lower nodes."
@kernel inbounds=true function nt_uppers_kernel!(w, @Const(lowerid), @Const(upperorigin), first::Int32, lowerfirst::Int32,
                                                 background::Float32)
    t = @index(Global)
    up = (Int32(t) - Int32(1)) ÷ Int32(1024)
    m = (Int32(t) - Int32(1)) % Int32(1024)
    at = first + up * Int32(NT_UPPER_WORDS)
    if m == Int32(0)
        o = upperorigin[up + 1]
        putcoord!(w, at, o)
        putcoord!(w, at + 3, o + Vec{3,Int32}(4095, 4095, 4095))
    end
    bits = UInt32(0)
    for b in Int32(0):Int32(31)
        n = m * Int32(32) + b
        low = lowerid[up * Int32(32768) + n + 1]
        if low > Int32(0)
            bits |= UInt32(1) << UInt32(b)
            lowat = lowerfirst + (low - Int32(1)) * Int32(NT_LOWER_WORDS)
            putint64!(w, at + NT_UPPER_TABLE + 2n, Int64(lowat - at) * 4)
        else
            putfloat!(w, at + NT_UPPER_TABLE + 2n, background)
        end
    end
    w[at + NT_UPPER_CHILDMASK + m + 1] = bits
    w[at + NT_UPPER_VALUEMASK + m + 1] = bits
end

@kernel inbounds=true function nt_lowerorigins_kernel!(origin, @Const(lowerid), @Const(upperorigin))
    t = @index(Global)
    id = lowerid[t]
    if id > Int32(0)
        up = (Int32(t) - Int32(1)) ÷ Int32(32768)
        n = (Int32(t) - Int32(1)) % Int32(32768)
        # The inverse of `upper_coord_to_offset`: (i >> 7 & 31) << 10 | (j >> 7 & 31) << 5 | (k >> 7 & 31).
        origin[id] = upperorigin[up + 1] + Vec{3,Int32}((n >> 10) & 31, (n >> 5) & 31, n & 31) * Int32(128)
    end
end

"""
    NanoTree(origins, values, lo, voxel; background = 0, backend)

The NanoVDB float grid of the leaves at `origins` (device, `Vec{3,Int32}` index-space
corners, multiples of 8, distinct) with `values` (device, 512 per leaf in NanoVDB's
order, `x` slowest: see `leaf_coord_to_offset`), voxels `voxel` apart from `lo` (see
[`NanoTree`](@ref)). The root has a tile per upper node the leaves reach; the nodes
follow it in NanoVDB's order: uppers, lowers, leaves.
"""
function NanoTree(origins::AbstractVector{Vec{3,Int32}}, values::AbstractVector{Float32}, lo, voxel; background = 0f0, backend)
    n = length(origins)
    n > 0 || throw(ArgumentError("a NanoTree needs at least one leaf"))
    # The leaves' index box, and the upper nodes (4096³ voxels each) that cover it.
    host = Array(origins)
    imin = reduce((a, b) -> min.(a, b), host); imax = reduce((a, b) -> max.(a, b), host) .+ Int32(8)
    u0 = imin .>> Int32(12); nu = ((imax .- Int32(1)) .>> Int32(12)) .- u0 .+ Int32(1)
    uflags = KA.zeros(backend, Int32, prod(nu))
    nt_flaguppers_kernel!(backend, 256)(uflags, origins, u0, nu; ndrange = n)
    upperid = cumsum(Array(uflags)) .* (Array(uflags) .> 0)    # few: on the host
    nupper = Int(maximum(upperid))
    upperorigin = Vector{Vec{3,Int32}}(undef, nupper)
    for s in eachindex(upperid)
        upperid[s] > 0 || continue
        c = s - 1
        upperorigin[upperid[s]] = (u0 .+ Vec{3,Int32}(c % nu[1], (c ÷ nu[1]) % nu[2], c ÷ (nu[1] * nu[2]))) .* Int32(4096)
    end
    dupperid = Mantle.devicearray(backend, Int32.(upperid))
    dupperorigin = Mantle.devicearray(backend, upperorigin)
    # The lower nodes: a flag per slot of every upper node (32768 each), numbered by a
    # running sum on the host, where so few flags are no cost.
    lflags = KA.zeros(backend, Int32, nupper * 32768)
    nt_flaglowers_kernel!(backend, 256)(lflags, dupperid, origins, u0, nu; ndrange = n)
    hflags = Array(lflags)
    hlowerid = cumsum(hflags) .* hflags
    nlower = Int(maximum(hlowerid))
    lowerid = Mantle.devicearray(backend, hlowerid)
    lowerorigin = Mantle.devicearray(backend, Vec{3,Int32}, nlower)
    nt_lowerorigins_kernel!(backend, 256)(lowerorigin, lowerid, dupperorigin; ndrange = length(lowerid))
    leafof = KA.zeros(backend, Int32, nlower * 4096)
    nt_mapleaves_kernel!(backend, 256)(leafof, dupperid, lowerid, origins, u0, nu; ndrange = n)
    # Layout, in words: root header and tiles, uppers, lowers, leaves.
    upper = NT_ROOT_HEADER + nupper * NT_ROOT_TILE
    lower = upper + nupper * NT_UPPER_WORDS
    leaf = lower + nlower * NT_LOWER_WORDS
    words = KA.zeros(backend, UInt32, leaf + n * NT_LEAF_WORDS)
    nt_leaves_kernel!(backend, 64)(words, origins, values, Int32(leaf); ndrange = n)
    nt_lowers_kernel!(backend, 128)(words, leafof, lowerorigin, Int32(lower), Int32(leaf), Float32(background);
                                     ndrange = nlower * 128)
    nt_uppers_kernel!(backend, 256)(words, lowerid, dupperorigin, Int32(upper), Int32(lower), Float32(background);
                                     ndrange = nupper * 1024)
    # The root: its index box, table size and background, and a tile per upper node.
    root = zeros(UInt32, upper)
    putcoord!(root, 0, imin); putcoord!(root, 3, imax .- Int32(1))
    root[ROOTDATA_TABLESIZE_OFFSET ÷ 4 + 1] = UInt32(nupper)
    putfloat!(root, ROOTDATA_BACKGROUND_OFFSET ÷ 4, Float32(background))
    for (i, o) in enumerate(upperorigin)
        at = NT_ROOT_HEADER + (i - 1) * NT_ROOT_TILE
        putint64!(root, at, reinterpret(Int64, coord_to_root_key((o[1], o[2], o[3]))))
        putint64!(root, at + 2, Int64(upper + (i - 1) * NT_UPPER_WORDS) * 4)
        root[at + 4 + 1] = UInt32(1)                # state: active
        putfloat!(root, at + 5, Float32(background))
    end
    copyto!(view(words, 1:upper), root)
    return NanoTree(words, Int32(upper), Int32(lower), Int32(leaf), Int32(nupper), Int32(nlower), Int32(n), Int32(nupper),
                    Point3f(lo), Float32(voxel), Float32(background), imin, imax)
end

"""
    save_nanovdb(path, t::NanoTree; name="density", gridclass=2)

The grid as a `.nvdb` file (see `save_nanovdb`): its words, which are NanoVDB's node
bytes, behind the grid's header.
"""
function save_nanovdb(path::String, t::NanoTree; name::AbstractString="density", gridclass::Integer=2)
    bytes = reinterpret(UInt8, Array(t.words))
    h = Float64(t.voxel)
    meta = (world_min = Tuple(Float32.(t.lo .+ Vec3f(t.imin) .* t.voxel)), world_max = Tuple(Float32.(t.lo .+ Vec3f(t.imax) .* t.voxel)),
            inv_mat = (Float32(1 / h), 0f0, 0f0, 0f0, Float32(1 / h), 0f0, 0f0, 0f0, Float32(1 / h)),
            vec = Tuple(Float32.(t.lo .+ Vec3f(0.5f0 * t.voxel))),
            root_offset = Int64(1), upper_offset = Int64(4t.upper + 1), lower_offset = Int64(4t.lower + 1),
            leaf_offset = Int64(4t.leaf + 1), leaf_count = t.leaves, lower_count = t.lowers, upper_count = t.uppers,
            root_table_size = t.roottiles, index_min = Tuple(t.imin), index_max = Tuple(t.imax .- Int32(1)))
    return save_nanovdb(path, bytes, meta; name, gridclass)
end
