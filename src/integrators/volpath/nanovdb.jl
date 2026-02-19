# NanoVDB Direct Volume Sampling Implementation
# Matches pbrt-v4's C++ NanoVDBMedium exactly, keeping the buffer in native format

# ============================================================================
# NanoVDB Constants and Offsets
# ============================================================================

# GridData structure size (from NanoVDB.h line 2185)
const NANOVDB_GRIDDATA_SIZE = 672

# TreeData structure offsets (starts at byte 672)
# TreeData layout (64 bytes total):
#   mNodeOffset[4]: 4 × UInt64 = 32 bytes (offsets to leaf, lower, upper, root)
#   mNodeCount[3]: 3 × UInt32 = 12 bytes (counts of leaf, lower, upper nodes)
#   mTileCount[3]: 3 × UInt32 = 12 bytes (tile counts)
#   mVoxelCount: UInt64 = 8 bytes
const TREEDATA_NODE_OFFSET_START = NANOVDB_GRIDDATA_SIZE + 1  # Julia 1-indexed
const TREEDATA_NODE_COUNT_START = NANOVDB_GRIDDATA_SIZE + 32 + 1
const TREEDATA_SIZE = 64

# Map structure offsets within GridData (starts at byte 296 in C++, 297 in Julia 1-indexed)
# Map layout (264 bytes):
#   mMatF[9]: 36 bytes (index→world, not needed)
#   mInvMatF[9]: 36 bytes (world→index, NEEDED)
#   mVecF[3]: 12 bytes (translation, NEEDED)
#   ... rest not needed for sampling
const MAP_OFFSET = 296 + 1  # Convert to Julia 1-indexed
const MAP_INVMATF_OFFSET = MAP_OFFSET + 36  # Skip mMatF (9 floats = 36 bytes)
const MAP_VECF_OFFSET = MAP_OFFSET + 72     # Skip mMatF + mInvMatF

# World bbox offsets in GridData (offset 560, 6 doubles = 48 bytes)
const WORLDBBOX_OFFSET = 560 + 1  # Julia 1-indexed

# VDB tree configuration (standard NanoVDB for floats):
# Root → Upper (32³, LOG2DIM=5, TOTAL=12) → Lower (16³, LOG2DIM=4, TOTAL=7) → Leaf (8³, LOG2DIM=3, TOTAL=3)
const LEAF_LOG2DIM = 3
const LEAF_DIM = 1 << LEAF_LOG2DIM  # 8
const LEAF_SIZE = 1 << (3 * LEAF_LOG2DIM)  # 512 voxels
const LEAF_MASK = (1 << LEAF_LOG2DIM) - 1  # 7

const LOWER_LOG2DIM = 4
const LOWER_DIM = 1 << LOWER_LOG2DIM  # 16
const LOWER_SIZE = 1 << (3 * LOWER_LOG2DIM)  # 4096 children/tiles
const LOWER_TOTAL = LEAF_LOG2DIM + LOWER_LOG2DIM  # 7
const LOWER_MASK = (1 << LOWER_TOTAL) - 1  # 127

const UPPER_LOG2DIM = 5
const UPPER_DIM = 1 << UPPER_LOG2DIM  # 32
const UPPER_SIZE = 1 << (3 * UPPER_LOG2DIM)  # 32768 children/tiles
const UPPER_TOTAL = LOWER_TOTAL + UPPER_LOG2DIM  # 12
const UPPER_MASK = (1 << UPPER_TOTAL) - 1  # 4095

# ============================================================================
# LeafData structure layout for Float32 (from NanoVDB.h line 3354)
# ============================================================================
# LeafData<float> layout:
#   CoordT mBBoxMin: 12 bytes (3 × Int32)
#   uint8_t mBBoxDif[3]: 3 bytes
#   uint8_t mFlags: 1 byte
#   MaskT<3> mValueMask: 64 bytes (512 bits)
#   float mMinimum: 4 bytes
#   float mMaximum: 4 bytes
#   float mAverage: 4 bytes
#   float mStdDevi: 4 bytes
#   alignas(32) float mValues[512]: 2048 bytes
# Total: 12 + 3 + 1 + 64 + 4 + 4 + 4 + 4 + 2048 = 2144 bytes

const LEAFDATA_BBOXMIN_OFFSET = 0
const LEAFDATA_MASK_OFFSET = 16  # After coords (12) + bbox (3) + flags (1)
const LEAFDATA_MIN_OFFSET = 80   # After mask (64 bytes)
const LEAFDATA_VALUES_OFFSET = 96  # After min(4) + max(4) + avg(4) + dev(4) = 16 bytes, but aligned to 32
const LEAFDATA_SIZE = 2144

# ============================================================================
# InternalData structure layout (Upper: LOG2DIM=5, Lower: LOG2DIM=4)
# ============================================================================
# InternalData layout:
#   BBox<CoordT> mBBox: 24 bytes
#   uint64_t mFlags: 8 bytes
#   MaskT mValueMask: 4096B (upper) or 512B (lower)
#   MaskT mChildMask: 4096B (upper) or 512B (lower)
#   ValueT mMinimum: 4 bytes
#   ValueT mMaximum: 4 bytes
#   StatsT mAverage: 4 bytes
#   StatsT mStdDevi: 4 bytes
#   alignas(32) Tile mTable[SIZE]: SIZE × 8 bytes (union of float + int64 child offset)

# Upper node (32³ = 32768 entries):
#   24 + 8 + 4096 + 4096 + 4 + 4 + 4 + 4 = 8240 bytes header
#   mTable: alignas(32) Tile[32768] × 8 = 262144 bytes
const UPPER_BBOX_OFFSET = 0
const UPPER_FLAGS_OFFSET = 24
const UPPER_VALUEMASK_OFFSET = 32
const UPPER_CHILDMASK_OFFSET = 32 + 4096  # 4128
const UPPER_MIN_OFFSET = 32 + 4096 + 4096  # 8224
const UPPER_TABLE_OFFSET = 8256  # Header is 8240, aligned to 32 -> 8256

# Lower node (16³ = 4096 entries):
#   24 + 8 + 512 + 512 + 4 + 4 + 4 + 4 = 1072 bytes header
#   mTable: alignas(32) Tile[4096] × 8 = 32768 bytes
const LOWER_BBOX_OFFSET = 0
const LOWER_FLAGS_OFFSET = 24
const LOWER_VALUEMASK_OFFSET = 32
const LOWER_CHILDMASK_OFFSET = 32 + 512  # 544
const LOWER_MIN_OFFSET = 32 + 512 + 512  # 1056
const LOWER_TABLE_OFFSET = 1088  # Header is 1072, aligned to 32 -> 1088

# ============================================================================
# RootData structure layout (from NanoVDB.h line 2680)
# ============================================================================
# RootData layout:
#   BBox<CoordT> mBBox: 24 bytes
#   uint32_t mTableSize: 4 bytes
#   ValueT mBackground: 4 bytes
#   ValueT mMinimum: 4 bytes
#   ValueT mMaximum: 4 bytes
#   StatsT mAverage: 4 bytes
#   StatsT mStdDevi: 4 bytes
# Total header: 48 bytes
# Then: Tile[mTableSize] where each Tile is 32 bytes (aligned to NANOVDB_DATA_ALIGNMENT)
# Raw data: key 8B + child 8B + state 4B + value 4B = 24B, padded to 32B

const ROOTDATA_BBOX_OFFSET = 0
const ROOTDATA_TABLESIZE_OFFSET = 24
const ROOTDATA_BACKGROUND_OFFSET = 28
const ROOTDATA_HEADER_SIZE = 64  # 48 bytes data + 16 bytes padding (aligned to 32)

# Root Tile layout (from NanoVDB.h line 2727):
#   KeyT key: 8 bytes (uint64 with USE_SINGLE_ROOT_KEY)
#   int64_t child: 8 bytes (offset from RootData to child, 0 means tile value)
#   uint32_t state: 4 bytes
#   ValueT value: 4 bytes
#   padding: 8 bytes (to align to 32 bytes)
const ROOTTILE_SIZE = 32  # Aligned to NANOVDB_DATA_ALIGNMENT (32 bytes)
const ROOTTILE_KEY_OFFSET = 0
const ROOTTILE_CHILD_OFFSET = 8
const ROOTTILE_STATE_OFFSET = 16
const ROOTTILE_VALUE_OFFSET = 20

# ============================================================================
# NanoVDBMedium struct
# ============================================================================

"""
    NanoVDBMedium

Direct NanoVDB volume sampling medium that keeps the buffer in native format.
Matches pbrt-v4's NanoVDBMedium implementation exactly.

The buffer contains the entire decompressed NanoVDB grid data.
Tree traversal is done on-the-fly using byte offsets.
"""
struct NanoVDBMedium{B<:AbstractVector{UInt8}, M<:MajorantGrid} <: Medium
    # Raw NanoVDB buffer (can be GPU array)
    buffer::B

    # Cached byte offsets from TreeData (relative to buffer start, 1-indexed for Julia)
    root_offset::Int64      # Offset to RootData
    upper_offset::Int64     # Offset to first upper internal node
    lower_offset::Int64     # Offset to first lower internal node
    leaf_offset::Int64      # Offset to first leaf node

    # Node counts
    leaf_count::Int32
    lower_count::Int32
    upper_count::Int32
    root_table_size::Int32

    # Affine transform: world → index
    # From Map structure: p_index = inv_mat * (p_world - vec)
    inv_mat::NTuple{9, Float32}  # 3×3 matrix, row-major
    vec::NTuple{3, Float32}      # Translation

    # World space bounds (from GridData.mWorldBBox)
    bounds::Bounds3

    # Index space bounds (for majorant grid mapping)
    index_bbox_min::NTuple{3, Int32}
    index_bbox_max::NTuple{3, Int32}

    # Optical properties
    σ_a::RGBSpectrum
    σ_s::RGBSpectrum
    g::Float32

    # Majorant grid (64³, same as pbrt-v4)
    majorant_grid::M

    # Precomputed max density (global maximum of majorant_grid.voxels)
    max_density::Float32
end

# ============================================================================
# Buffer reading utilities (works for both CPU Ptr and GPU LLVMPtr)
# ============================================================================

# Reinterpret pointer to different element type - works for both CPU and GPU
@inline as_pointer(::Type{T}, ptr::Core.LLVMPtr{T2, AS}) where {T, T2, AS} =
    reinterpret(Core.LLVMPtr{T, AS}, ptr)
@inline as_pointer(::Type{T}, ptr::Ptr) where {T} =
    reinterpret(Ptr{T}, ptr)

# Unified read functions using pointer + reinterpret (CPU and GPU)
# ptr is a UInt8 pointer, byte_offset is 1-indexed Julia offset
@inline function read_float32(ptr, byte_offset::Integer)::Float32
    ptr32 = as_pointer(Float32, ptr)
    elem_idx = ((byte_offset - 1) >> 2) + 1  # ÷4
    @inbounds Base.unsafe_load(ptr32, elem_idx)
end

@inline function read_int32(ptr, byte_offset::Integer)::Int32
    ptr32 = as_pointer(Int32, ptr)
    elem_idx = ((byte_offset - 1) >> 2) + 1
    @inbounds Base.unsafe_load(ptr32, elem_idx)
end

@inline function read_uint32(ptr, byte_offset::Integer)::UInt32
    ptr32 = as_pointer(UInt32, ptr)
    elem_idx = ((byte_offset - 1) >> 2) + 1
    @inbounds Base.unsafe_load(ptr32, elem_idx)
end

@inline function read_int64(ptr, byte_offset::Integer)::Int64
    ptr64 = as_pointer(Int64, ptr)
    elem_idx = ((byte_offset - 1) >> 3) + 1  # ÷8
    @inbounds Base.unsafe_load(ptr64, elem_idx)
end

@inline function read_uint64(ptr, byte_offset::Integer)::UInt64
    ptr64 = as_pointer(UInt64, ptr)
    elem_idx = ((byte_offset - 1) >> 3) + 1
    @inbounds Base.unsafe_load(ptr64, elem_idx)
end

# Check if bit n is set in a bitmask (CPU and GPU)
# ptr is UInt8 pointer, mask_offset is 1-indexed, n is 0-indexed bit number
@inline function bitmask_is_on(ptr, mask_offset::Integer, n::Integer)::Bool
    byte_idx = (n >> 3)  # Which byte (0-indexed from mask_offset)
    bit_idx = n & 7      # Which bit within byte
    @inbounds byte = Base.unsafe_load(ptr, mask_offset + byte_idx)
    return (byte >> bit_idx) & 1 != 0
end


# ============================================================================
# Coordinate to Key conversion (matching C++ CoordToKey)
# ============================================================================

# For USE_SINGLE_ROOT_KEY (64-bit key), coordinate is hashed as:
# key = (z >> TOTAL) | ((y >> TOTAL) << 21) | ((x >> TOTAL) << 42)
# where TOTAL = 12 for upper nodes
@inline function coord_to_root_key(ijk::NTuple{3, Int32})::UInt64
    x, y, z = ijk
    # Use reinterpret to get unsigned bit pattern (like C++ uint32_t cast)
    xu, yu, zu = reinterpret(UInt32, x), reinterpret(UInt32, y), reinterpret(UInt32, z)
    # Shift by UPPER_TOTAL (12) to get the root-level tile coordinates
    zk = UInt64((zu >> UPPER_TOTAL) & 0x1fffff)  # 21 bits
    yk = UInt64((yu >> UPPER_TOTAL) & 0x1fffff) << 21
    xk = UInt64((xu >> UPPER_TOTAL) & 0x1fffff) << 42
    return zk | yk | xk
end

# ============================================================================
# CoordToOffset functions (matching C++ InternalNode/LeafNode::CoordToOffset)
# ============================================================================

# Upper node: 32³, TOTAL=12, LOG2DIM=5
# Offset = ((x >> 7) & 31) << 10 | ((y >> 7) & 31) << 5 | ((z >> 7) & 31)
@inline function upper_coord_to_offset(ijk::NTuple{3, Int32})::Int32
    x, y, z = ijk
    # Use reinterpret to handle negative coordinates (like C++ uint32_t cast)
    xu, yu, zu = reinterpret(UInt32, x), reinterpret(UInt32, y), reinterpret(UInt32, z)
    ox = Int32((xu >> LOWER_TOTAL) & (UPPER_DIM - 1)) << (2 * UPPER_LOG2DIM)
    oy = Int32((yu >> LOWER_TOTAL) & (UPPER_DIM - 1)) << UPPER_LOG2DIM
    oz = Int32((zu >> LOWER_TOTAL) & (UPPER_DIM - 1))
    return ox | oy | oz
end

# Lower node: 16³, TOTAL=7, LOG2DIM=4
# Offset = ((x >> 3) & 15) << 8 | ((y >> 3) & 15) << 4 | ((z >> 3) & 15)
@inline function lower_coord_to_offset(ijk::NTuple{3, Int32})::Int32
    x, y, z = ijk
    # Use reinterpret to handle negative coordinates (like C++ uint32_t cast)
    xu, yu, zu = reinterpret(UInt32, x), reinterpret(UInt32, y), reinterpret(UInt32, z)
    ox = Int32((xu >> LEAF_LOG2DIM) & (LOWER_DIM - 1)) << (2 * LOWER_LOG2DIM)
    oy = Int32((yu >> LEAF_LOG2DIM) & (LOWER_DIM - 1)) << LOWER_LOG2DIM
    oz = Int32((zu >> LEAF_LOG2DIM) & (LOWER_DIM - 1))
    return ox | oy | oz
end

# Leaf node: 8³, LOG2DIM=3
# Offset = (x & 7) << 6 | (y & 7) << 3 | (z & 7)
@inline function leaf_coord_to_offset(ijk::NTuple{3, Int32})::Int32
    x, y, z = ijk
    ox = (x & LEAF_MASK) << (2 * LEAF_LOG2DIM)
    oy = (y & LEAF_MASK) << LEAF_LOG2DIM
    oz = z & LEAF_MASK
    return ox | oy | oz
end

# ============================================================================
# Tree Traversal: getValue
# ============================================================================

"""
    nanovdb_get_value(medium::NanoVDBMedium, media, ijk::NTuple{3, Int32}) -> Float32

Get the voxel value at integer index coordinates using full tree traversal.
Matches pbrt-v4/NanoVDB's Tree::getValue exactly.

The `media` parameter is used to deref TextureRef fields when NanoVDBMedium is stored in a MultiTypeSet.

Uses pointer(buffer) with as_pointer for type conversion - works on both CPU and GPU.
"""
@inline @propagate_inbounds function nanovdb_get_value(medium::NanoVDBMedium, media, ijk::NTuple{3, Int32})::Float32
    # Deref buffer from TextureRef (no-op if already array)
    buffer = Raycore.deref(media, medium.buffer)
    # Get pointer to buffer - works for both CPU Vector and GPU CLArray
    ptr = pointer(buffer)
    root_offset = medium.root_offset

    # 1. Root node lookup (linear search through tiles)
    table_size = medium.root_table_size
    key = coord_to_root_key(ijk)

    # Linear search through root tiles (matching C++ RootNode::findTile)
    tile_found = false
    tile_offset = Int64(0)
    tile_base = root_offset + ROOTDATA_HEADER_SIZE

    for i in Int32(0):table_size-Int32(1)
        t_offset = tile_base + i * ROOTTILE_SIZE
        t_key = read_uint64(ptr, t_offset + ROOTTILE_KEY_OFFSET)
        if t_key == key
            tile_found = true
            tile_offset = t_offset
            break
        end
    end

    if !tile_found
        # Return background value
        return read_float32(ptr, root_offset + ROOTDATA_BACKGROUND_OFFSET)
    end

    # Check if tile has child or is a constant tile
    child_offset = read_int64(ptr, tile_offset + ROOTTILE_CHILD_OFFSET)
    if child_offset == 0
        # Constant tile - return tile value
        return read_float32(ptr, tile_offset + ROOTTILE_VALUE_OFFSET)
    end

    # 2. Upper internal node (32³)
    # child_offset is relative to RootData, convert to absolute buffer offset
    upper_off = root_offset + child_offset
    n_upper = upper_coord_to_offset(ijk)

    # Check child mask
    if !bitmask_is_on(ptr, upper_off + UPPER_CHILDMASK_OFFSET, n_upper)
        # This is a tile value, not a child
        value_off = upper_off + UPPER_TABLE_OFFSET + n_upper * 8
        return read_float32(ptr, value_off)
    end

    # Get child offset (from upper node's table)
    upper_child_offset = read_int64(ptr, upper_off + UPPER_TABLE_OFFSET + n_upper * 8)

    # 3. Lower internal node (16³)
    lower_off = upper_off + upper_child_offset
    n_lower = lower_coord_to_offset(ijk)

    # Check child mask
    if !bitmask_is_on(ptr, lower_off + LOWER_CHILDMASK_OFFSET, n_lower)
        # This is a tile value, not a child
        value_off = lower_off + LOWER_TABLE_OFFSET + n_lower * 8
        return read_float32(ptr, value_off)
    end

    # Get child offset
    lower_child_offset = read_int64(ptr, lower_off + LOWER_TABLE_OFFSET + n_lower * 8)

    # 4. Leaf node (8³)
    leaf_off = lower_off + lower_child_offset
    n_leaf = leaf_coord_to_offset(ijk)

    # Read value from leaf
    return read_float32(ptr, leaf_off + LEAFDATA_VALUES_OFFSET + n_leaf * 4)
end

# ============================================================================
# World to Index transform
# ============================================================================

"""
    world_to_index_f(medium::NanoVDBMedium, p::Point3f) -> Point3f

Transform a point from world space to index space (floating point).
Matches pbrt-v4's Grid::worldToIndexF exactly.
"""
@inline function world_to_index_f(medium::NanoVDBMedium, p::Point3f)::Point3f
    # p_index = inv_mat * (p - vec)
    px = p[1] - medium.vec[1]
    py = p[2] - medium.vec[2]
    pz = p[3] - medium.vec[3]

    # Row-major 3×3 matrix multiply (matching C++ NanoVDB Map::applyInverseMapF)
    ix = medium.inv_mat[1]*px + medium.inv_mat[2]*py + medium.inv_mat[3]*pz
    iy = medium.inv_mat[4]*px + medium.inv_mat[5]*py + medium.inv_mat[6]*pz
    iz = medium.inv_mat[7]*px + medium.inv_mat[8]*py + medium.inv_mat[9]*pz

    return Point3f(ix, iy, iz)
end

# ============================================================================
# Trilinear Interpolation
# ============================================================================

"""
    sample_nanovdb_density(medium::NanoVDBMedium, media, p_world::Point3f) -> Float32

Sample the density at a world-space point using trilinear interpolation.
Matches pbrt-v4's SampleFromVoxels<TreeT, 1, false> sampler.

The `media` parameter is used to deref TextureRef fields when NanoVDBMedium is stored in a MultiTypeSet.
"""
@inline @propagate_inbounds function sample_nanovdb_density(medium::NanoVDBMedium, media, p_world::Point3f)::Float32
    # Transform to index space
    p_idx = world_to_index_f(medium, p_world)

    # Floor to get base voxel (use floor_int32 for GPU compatibility)
    ix = floor_int32(p_idx[1])
    iy = floor_int32(p_idx[2])
    iz = floor_int32(p_idx[3])

    # Fractional parts
    fx = p_idx[1] - Float32(ix)
    fy = p_idx[2] - Float32(iy)
    fz = p_idx[3] - Float32(iz)

    # Sample 8 corners (matching C++ TrilinearSampler::stencil order)
    # v[i][j][k] where i=x+di, j=y+dj, k=z+dk
    v000 = nanovdb_get_value(medium, media, (ix, iy, iz))
    v001 = nanovdb_get_value(medium, media, (ix, iy, iz + Int32(1)))
    v010 = nanovdb_get_value(medium, media, (ix, iy + Int32(1), iz))
    v011 = nanovdb_get_value(medium, media, (ix, iy + Int32(1), iz + Int32(1)))
    v100 = nanovdb_get_value(medium, media, (ix + Int32(1), iy, iz))
    v101 = nanovdb_get_value(medium, media, (ix + Int32(1), iy, iz + Int32(1)))
    v110 = nanovdb_get_value(medium, media, (ix + Int32(1), iy + Int32(1), iz))
    v111 = nanovdb_get_value(medium, media, (ix + Int32(1), iy + Int32(1), iz + Int32(1)))

    # Trilinear interpolation (matching C++ TrilinearSampler::sample)
    # lerp(a, b, t) = a + t * (b - a)
    fx1 = 1f0 - fx
    fy1 = 1f0 - fy
    fz1 = 1f0 - fz

    # Interpolate along z
    v00 = v000 * fz1 + v001 * fz
    v01 = v010 * fz1 + v011 * fz
    v10 = v100 * fz1 + v101 * fz
    v11 = v110 * fz1 + v111 * fz

    # Interpolate along y
    v0 = v00 * fy1 + v01 * fy
    v1 = v10 * fy1 + v11 * fy

    # Interpolate along x
    return v0 * fx1 + v1 * fx
end

# ============================================================================
# Medium Interface Implementation
# ============================================================================

@propagate_inbounds is_emissive(::NanoVDBMedium) = false

@propagate_inbounds function sample_point(
    medium::NanoVDBMedium,
    media,  # StaticMultiTypeSet for deref of TextureRef fields
    table::RGBToSpectrumTable,
    p::Point3f,
    λ::Wavelengths
)::MediumProperties
    # Sample density using trilinear interpolation
    d = sample_nanovdb_density(medium, media, p)

    # Scale coefficients by density
    σ_a = uplift_rgb_unbounded(table, medium.σ_a, λ) * d
    σ_s = uplift_rgb_unbounded(table, medium.σ_s, λ) * d

    return MediumProperties(σ_a, σ_s, SpectralRadiance(0f0), medium.g)
end

@propagate_inbounds function get_majorant(
    medium::NanoVDBMedium,
    table::RGBToSpectrumTable,
    ray::Raycore.Ray,
    t_min::Float32,
    t_max::Float32,
    λ::Wavelengths
)::RayMajorantSegment
    # Use precomputed global max from majorant grid (conservative bound)
    σ_a = uplift_rgb_unbounded(table, medium.σ_a, λ)
    σ_s = uplift_rgb_unbounded(table, medium.σ_s, λ)
    σ_maj = (σ_a + σ_s) * medium.max_density
    return RayMajorantSegment(t_min, t_max, σ_maj)
end

@propagate_inbounds function create_majorant_iterator(
    medium::NanoVDBMedium,
    table::RGBToSpectrumTable,
    ray::Raycore.Ray,
    t_max::Float32,
    λ::Wavelengths
)
    # Compute ray-bounds intersection
    t_enter, t_exit = ray_bounds_intersect(ray.o, ray.d, medium.bounds)

    # Clamp to requested range
    t_enter = max(t_enter, 0f0)
    t_exit = min(t_exit, t_max)

    if t_enter >= t_exit
        return RayMajorantIterator(medium.majorant_grid)
    end

    # Compute base extinction coefficient
    σ_a = uplift_rgb_unbounded(table, medium.σ_a, λ)
    σ_s = uplift_rgb_unbounded(table, medium.σ_s, λ)
    σ_t = σ_a + σ_s

    # Create DDA iterator (ray is already in world space, majorant grid is in world space)
    dda_iter = create_dda_iterator(
        medium.majorant_grid,
        medium.bounds,
        ray.o,
        ray.d,
        t_enter,
        t_exit,
        σ_t
    )
    return RayMajorantIterator(dda_iter)
end

@propagate_inbounds function create_majorant_iterator(
    medium::NanoVDBMedium,
    table::RGBToSpectrumTable,
    ray::Raycore.Ray,
    t_max::Float32,
    λ::Wavelengths,
    ::MajorantGrid
)
    return create_majorant_iterator(medium, table, ray, t_max, λ)
end

# ============================================================================
# Dense Array → NanoVDB Buffer Construction
# ============================================================================

# Node sizes (bytes)
const UPPER_NODE_SIZE = UPPER_TABLE_OFFSET + UPPER_SIZE * 8  # 270400
const LOWER_NODE_SIZE = LOWER_TABLE_OFFSET + LOWER_SIZE * 8  # 33856

# Buffer write helpers (1-indexed byte offsets, matching the read helpers)
@inline function write_buf!(buffer::Vector{UInt8}, offset::Integer, value::Float32)
    GC.@preserve buffer unsafe_store!(reinterpret(Ptr{Float32}, pointer(buffer, offset)), value)
end
@inline function write_buf!(buffer::Vector{UInt8}, offset::Integer, value::Int32)
    GC.@preserve buffer unsafe_store!(reinterpret(Ptr{Int32}, pointer(buffer, offset)), value)
end
@inline function write_buf!(buffer::Vector{UInt8}, offset::Integer, value::UInt32)
    GC.@preserve buffer unsafe_store!(reinterpret(Ptr{UInt32}, pointer(buffer, offset)), value)
end
@inline function write_buf!(buffer::Vector{UInt8}, offset::Integer, value::Int64)
    GC.@preserve buffer unsafe_store!(reinterpret(Ptr{Int64}, pointer(buffer, offset)), value)
end
@inline function write_buf!(buffer::Vector{UInt8}, offset::Integer, value::UInt64)
    GC.@preserve buffer unsafe_store!(reinterpret(Ptr{UInt64}, pointer(buffer, offset)), value)
end

# Set bit n in a bitmask at mask_offset (1-indexed, n is 0-indexed)
@inline function bitmask_set!(buffer::Vector{UInt8}, mask_offset::Integer, n::Integer)
    byte_idx = n >> 3
    bit_idx = n & 7
    buffer[mask_offset + byte_idx] |= UInt8(1) << bit_idx
end

"""
    build_nanovdb_from_dense(data::Array{Float32,3}, origin::Point3f, extent::Vec3f; background=0f0)

Convert a dense 3D Float32 array to a NanoVDB-compatible binary buffer.
Only stores leaf blocks (8³) that contain non-background voxels, giving
significant memory savings for sparse data (e.g. cloud fields with ~1% fill).

Returns `(buffer::Vector{UInt8}, metadata::NamedTuple)` suitable for
constructing a `NanoVDBMedium`.

The NanoVDB tree hierarchy is: Root → Upper (32³) → Lower (16³) → Leaf (8³).
Buffer layout: [Root | Upper nodes | Lower nodes | Leaf nodes]
All child offsets are stored relative to the parent node.
"""
function build_nanovdb_from_dense(
    data::Array{Float32, 3},
    origin::Point3f,
    extent::Vec3f;
    background::Float32 = 0f0
)
    nx, ny, nz = size(data)
    dx, dy, dz = extent[1] / nx, extent[2] / ny, extent[3] / nz

    # ---- Phase 1: Collect active leaf blocks (8³) ----
    n_bx = cld(nx, LEAF_DIM)
    n_by = cld(ny, LEAF_DIM)
    n_bz = cld(nz, LEAF_DIM)

    leaf_coords = NTuple{3, Int32}[]   # NanoVDB base coordinate per leaf
    leaf_values = Vector{Vector{Float32}}()  # 512 values per leaf

    # Reuse a single scratch buffer to avoid 622k temporary allocations
    scratch = Vector{Float32}(undef, 512)

    for bz in 0:n_bz-1, by in 0:n_by-1, bx in 0:n_bx-1
        has_active = false
        fill!(scratch, background)

        for lz in 0:LEAF_DIM-1, ly in 0:LEAF_DIM-1, lx in 0:LEAF_DIM-1
            ix = bx * LEAF_DIM + lx + 1  # Julia 1-indexed
            iy = by * LEAF_DIM + ly + 1
            iz = bz * LEAF_DIM + lz + 1
            v = (ix <= nx && iy <= ny && iz <= nz) ? data[ix, iy, iz] : background
            leaf_idx = (lx << (2*LEAF_LOG2DIM)) | (ly << LEAF_LOG2DIM) | lz
            scratch[leaf_idx + 1] = v
            if v != background
                has_active = true
            end
        end

        if has_active
            base = (Int32(bx * LEAF_DIM), Int32(by * LEAF_DIM), Int32(bz * LEAF_DIM))
            push!(leaf_coords, base)
            push!(leaf_values, copy(scratch))
        end
    end

    n_leaves = length(leaf_coords)

    # ---- Phase 2: Group leaves into lower nodes (each covers 128³) ----
    lower_to_leaves = Dict{NTuple{3,Int32}, Vector{Int}}()

    for (li, coord) in enumerate(leaf_coords)
        lb = (coord[1] & ~Int32(LOWER_MASK),
              coord[2] & ~Int32(LOWER_MASK),
              coord[3] & ~Int32(LOWER_MASK))
        leaves = get!(lower_to_leaves, lb, Int[])
        push!(leaves, li)
    end

    lower_bases = sort!(collect(keys(lower_to_leaves)))
    n_lowers = length(lower_bases)

    # ---- Phase 3: Group lower nodes into upper nodes (each covers 4096³) ----
    upper_to_lowers = Dict{NTuple{3,Int32}, Vector{Int}}()

    for (low_i, lb) in enumerate(lower_bases)
        ub = (lb[1] & ~Int32(UPPER_MASK),
              lb[2] & ~Int32(UPPER_MASK),
              lb[3] & ~Int32(UPPER_MASK))
        lowers = get!(upper_to_lowers, ub, Int[])
        push!(lowers, low_i)
    end

    upper_bases = sort!(collect(keys(upper_to_lowers)))
    n_uppers = length(upper_bases)

    # ---- Phase 4: Compute buffer layout ----
    root_n_tiles = n_uppers
    root_size = ROOTDATA_HEADER_SIZE + root_n_tiles * ROOTTILE_SIZE
    upper_section = n_uppers * UPPER_NODE_SIZE
    lower_section = n_lowers * LOWER_NODE_SIZE
    leaf_section = n_leaves * LEAFDATA_SIZE
    total_size = root_size + upper_section + lower_section + leaf_section

    buffer = zeros(UInt8, total_size)

    # 1-indexed byte positions
    root_pos = 1
    upper_pos(i) = root_pos + root_size + (i - 1) * UPPER_NODE_SIZE
    lower_pos(i) = root_pos + root_size + upper_section + (i - 1) * LOWER_NODE_SIZE

    # Sort leaves for deterministic layout; build index mapping
    leaf_sorted_order = sortperm(leaf_coords)
    leaf_buf_pos = Vector{Int64}(undef, n_leaves)
    for (slot, li) in enumerate(leaf_sorted_order)
        leaf_buf_pos[li] = root_pos + root_size + upper_section + lower_section + (slot - 1) * LEAFDATA_SIZE
    end

    # ---- Phase 5: Write leaf nodes ----
    for li in 1:n_leaves
        coord = leaf_coords[li]
        values = leaf_values[li]
        off = leaf_buf_pos[li]

        # BBoxMin (3 × Int32)
        write_buf!(buffer, off + LEAFDATA_BBOXMIN_OFFSET, coord[1])
        write_buf!(buffer, off + LEAFDATA_BBOXMIN_OFFSET + 4, coord[2])
        write_buf!(buffer, off + LEAFDATA_BBOXMIN_OFFSET + 8, coord[3])

        # BBoxDif (3 × UInt8)
        buffer[off + 12] = 0x07
        buffer[off + 13] = 0x07
        buffer[off + 14] = 0x07

        # ValueMask + min/max
        vmin = typemax(Float32)
        vmax = typemin(Float32)
        for i in 0:511
            v = values[i + 1]
            if v != background
                bitmask_set!(buffer, off + LEAFDATA_MASK_OFFSET, i)
            end
            vmin = min(vmin, v)
            vmax = max(vmax, v)
        end
        write_buf!(buffer, off + LEAFDATA_MIN_OFFSET, vmin)
        write_buf!(buffer, off + LEAFDATA_MIN_OFFSET + 4, vmax)

        # Values (512 × Float32)
        for i in 0:511
            write_buf!(buffer, off + LEAFDATA_VALUES_OFFSET + i * 4, values[i + 1])
        end
    end

    # ---- Phase 6: Write lower nodes ----
    lower_pos_map = Dict{NTuple{3,Int32}, Int}()
    for (i, lb) in enumerate(lower_bases)
        lower_pos_map[lb] = i
    end

    for (low_i, lb) in enumerate(lower_bases)
        off = lower_pos(low_i)

        # BBox (min + max, 6 × Int32)
        write_buf!(buffer, off + LOWER_BBOX_OFFSET, lb[1])
        write_buf!(buffer, off + LOWER_BBOX_OFFSET + 4, lb[2])
        write_buf!(buffer, off + LOWER_BBOX_OFFSET + 8, lb[3])
        # max = lb + 128 - 1
        write_buf!(buffer, off + LOWER_BBOX_OFFSET + 12, lb[1] + Int32(LEAF_DIM * LOWER_DIM - 1))
        write_buf!(buffer, off + LOWER_BBOX_OFFSET + 16, lb[2] + Int32(LEAF_DIM * LOWER_DIM - 1))
        write_buf!(buffer, off + LOWER_BBOX_OFFSET + 20, lb[3] + Int32(LEAF_DIM * LOWER_DIM - 1))

        # Write child mask and table entries for active leaves
        for li in lower_to_leaves[lb]
            coord = leaf_coords[li]
            # Compute child offset within this lower node (0..4095)
            n = lower_coord_to_offset(coord)

            # Set child mask bit
            bitmask_set!(buffer, off + LOWER_CHILDMASK_OFFSET, n)
            # Also set value mask bit
            bitmask_set!(buffer, off + LOWER_VALUEMASK_OFFSET, n)

            # Write child offset (relative: leaf_position - lower_position)
            child_off = Int64(leaf_buf_pos[li] - off)
            write_buf!(buffer, off + LOWER_TABLE_OFFSET + n * 8, child_off)
        end

        # Tile entries for inactive children get background value (already 0 from zeros())
    end

    # ---- Phase 7: Write upper nodes ----
    for (up_i, ub) in enumerate(upper_bases)
        off = upper_pos(up_i)

        # BBox
        write_buf!(buffer, off + UPPER_BBOX_OFFSET, ub[1])
        write_buf!(buffer, off + UPPER_BBOX_OFFSET + 4, ub[2])
        write_buf!(buffer, off + UPPER_BBOX_OFFSET + 8, ub[3])
        write_buf!(buffer, off + UPPER_BBOX_OFFSET + 12, ub[1] + Int32(LEAF_DIM * LOWER_DIM * UPPER_DIM - 1))
        write_buf!(buffer, off + UPPER_BBOX_OFFSET + 16, ub[2] + Int32(LEAF_DIM * LOWER_DIM * UPPER_DIM - 1))
        write_buf!(buffer, off + UPPER_BBOX_OFFSET + 20, ub[3] + Int32(LEAF_DIM * LOWER_DIM * UPPER_DIM - 1))

        # Write child mask and table entries for active lower nodes
        for low_i in upper_to_lowers[ub]
            lb = lower_bases[low_i]
            n = upper_coord_to_offset(lb)

            bitmask_set!(buffer, off + UPPER_CHILDMASK_OFFSET, n)
            bitmask_set!(buffer, off + UPPER_VALUEMASK_OFFSET, n)

            child_off = Int64(lower_pos(low_i) - off)
            write_buf!(buffer, off + UPPER_TABLE_OFFSET + n * 8, child_off)
        end
    end

    # ---- Phase 8: Write root node ----
    # Root header
    # Compute index bbox from all leaf coords
    idx_min = (typemax(Int32), typemax(Int32), typemax(Int32))
    idx_max = (typemin(Int32), typemin(Int32), typemin(Int32))
    for coord in leaf_coords
        idx_min = (min(idx_min[1], coord[1]), min(idx_min[2], coord[2]), min(idx_min[3], coord[3]))
        idx_max = (max(idx_max[1], coord[1] + Int32(LEAF_DIM)),
                   max(idx_max[2], coord[2] + Int32(LEAF_DIM)),
                   max(idx_max[3], coord[3] + Int32(LEAF_DIM)))
    end

    write_buf!(buffer, root_pos + ROOTDATA_BBOX_OFFSET, idx_min[1])
    write_buf!(buffer, root_pos + ROOTDATA_BBOX_OFFSET + 4, idx_min[2])
    write_buf!(buffer, root_pos + ROOTDATA_BBOX_OFFSET + 8, idx_min[3])
    write_buf!(buffer, root_pos + ROOTDATA_BBOX_OFFSET + 12, idx_max[1])
    write_buf!(buffer, root_pos + ROOTDATA_BBOX_OFFSET + 16, idx_max[2])
    write_buf!(buffer, root_pos + ROOTDATA_BBOX_OFFSET + 20, idx_max[3])
    write_buf!(buffer, root_pos + ROOTDATA_TABLESIZE_OFFSET, UInt32(root_n_tiles))
    write_buf!(buffer, root_pos + ROOTDATA_BACKGROUND_OFFSET, background)

    # Root tiles (one per upper node)
    tile_base = root_pos + ROOTDATA_HEADER_SIZE
    for (ti, ub) in enumerate(upper_bases)
        t_off = tile_base + (ti - 1) * ROOTTILE_SIZE
        key = coord_to_root_key(ub)
        write_buf!(buffer, t_off + ROOTTILE_KEY_OFFSET, key)
        # Child offset: relative to root_pos
        child_off = Int64(upper_pos(ti) - root_pos)
        write_buf!(buffer, t_off + ROOTTILE_CHILD_OFFSET, child_off)
        write_buf!(buffer, t_off + ROOTTILE_STATE_OFFSET, UInt32(1))  # active
        write_buf!(buffer, t_off + ROOTTILE_VALUE_OFFSET, background)
    end

    # ---- Build metadata ----
    # World-to-index transform: p_index = inv_mat * (p_world - vec)
    # vec = world position of index origin (center of first voxel)
    vec = (Float32(origin[1] + dx/2), Float32(origin[2] + dy/2), Float32(origin[3] + dz/2))
    inv_mat = (Float32(1/dx), 0f0, 0f0,
               0f0, Float32(1/dy), 0f0,
               0f0, 0f0, Float32(1/dz))

    world_min = (Float32(origin[1]), Float32(origin[2]), Float32(origin[3]))
    world_max = (Float32(origin[1] + extent[1]), Float32(origin[2] + extent[2]), Float32(origin[3] + extent[3]))

    metadata = (
        world_min = world_min,
        world_max = world_max,
        inv_mat = inv_mat,
        vec = vec,
        root_offset = Int64(root_pos),
        upper_offset = Int64(upper_pos(1)),
        lower_offset = Int64(lower_pos(1)),
        leaf_offset = Int64(leaf_buf_pos[leaf_sorted_order[1]]),
        leaf_count = Int32(n_leaves),
        lower_count = Int32(n_lowers),
        upper_count = Int32(n_uppers),
        root_table_size = Int32(root_n_tiles),
        index_min = idx_min,
        index_max = idx_max,
    )

    return buffer, metadata
end

"""
    save_nanovdb(filepath, buffer, metadata)

Save a NanoVDB buffer (from `build_nanovdb_from_dense`) to a `.nanovdb` file
that can be loaded back with `NanoVDBMedium(filepath; ...)`.

Prepends GridData (672 bytes) + TreeData (64 bytes) header, compresses with zlib.
"""
function save_nanovdb(filepath::String, buffer::Vector{UInt8}, metadata::NamedTuple)
    header_size = NANOVDB_GRIDDATA_SIZE + TREEDATA_SIZE  # 672 + 64 = 736
    full_buffer = zeros(UInt8, header_size + length(buffer))

    # Copy node data after header
    copyto!(full_buffer, header_size + 1, buffer, 1, length(buffer))

    # ---- Write Map in GridData ----
    inv_mat = metadata.inv_mat
    for (i, v) in enumerate(inv_mat)
        write_buf!(full_buffer, MAP_INVMATF_OFFSET + (i - 1) * 4, Float32(v))
    end

    # matF = inverse of invMatF (3x3 matrix inverse via cofactors)
    m = inv_mat
    cof = (m[5]*m[9] - m[6]*m[8], m[3]*m[8] - m[2]*m[9], m[2]*m[6] - m[3]*m[5],
           m[6]*m[7] - m[4]*m[9], m[1]*m[9] - m[3]*m[7], m[3]*m[4] - m[1]*m[6],
           m[4]*m[8] - m[5]*m[7], m[2]*m[7] - m[1]*m[8], m[1]*m[5] - m[2]*m[4])
    det = m[1]*cof[1] + m[2]*cof[4] + m[3]*cof[7]
    for (i, v) in enumerate(cof)
        write_buf!(full_buffer, MAP_OFFSET + (i - 1) * 4, Float32(v / det))
    end

    # vecF
    for (i, v) in enumerate(metadata.vec)
        write_buf!(full_buffer, MAP_VECF_OFFSET + (i - 1) * 4, Float32(v))
    end

    # World bounding box (6 × Float64)
    wmin, wmax = metadata.world_min, metadata.world_max
    GC.@preserve full_buffer begin
        ptr = pointer(full_buffer, WORLDBBOX_OFFSET)
        unsafe_store!(reinterpret(Ptr{Float64}, ptr), Float64(wmin[1]), 1)
        unsafe_store!(reinterpret(Ptr{Float64}, ptr), Float64(wmin[2]), 2)
        unsafe_store!(reinterpret(Ptr{Float64}, ptr), Float64(wmin[3]), 3)
        unsafe_store!(reinterpret(Ptr{Float64}, ptr), Float64(wmax[1]), 4)
        unsafe_store!(reinterpret(Ptr{Float64}, ptr), Float64(wmax[2]), 5)
        unsafe_store!(reinterpret(Ptr{Float64}, ptr), Float64(wmax[3]), 6)
    end

    # ---- Write TreeData ----
    # mNodeOffset[4]: byte offsets from TreeData start to each node level
    # extract_nanovdb_metadata computes: absolute = tree_data_start + mNodeOffset
    # where tree_data_start = NANOVDB_GRIDDATA_SIZE + 1 = 673 (Julia 1-indexed)
    # Nodes are at (header_size + orig_offset) in full buffer
    # So: mNodeOffset = header_size + orig_offset - tree_data_start
    #                 = 736 + orig_offset - 673 = orig_offset + 63
    GC.@preserve full_buffer begin
        ptr = pointer(full_buffer, TREEDATA_NODE_OFFSET_START)
        unsafe_store!(reinterpret(Ptr{UInt64}, ptr), UInt64(metadata.leaf_offset + 63), 1)
        unsafe_store!(reinterpret(Ptr{UInt64}, ptr), UInt64(metadata.lower_offset + 63), 2)
        unsafe_store!(reinterpret(Ptr{UInt64}, ptr), UInt64(metadata.upper_offset + 63), 3)
        unsafe_store!(reinterpret(Ptr{UInt64}, ptr), UInt64(metadata.root_offset + 63), 4)
    end

    # mNodeCount[3]: leaf, lower, upper
    GC.@preserve full_buffer begin
        ptr = pointer(full_buffer, TREEDATA_NODE_COUNT_START)
        unsafe_store!(reinterpret(Ptr{UInt32}, ptr), UInt32(metadata.leaf_count), 1)
        unsafe_store!(reinterpret(Ptr{UInt32}, ptr), UInt32(metadata.lower_count), 2)
        unsafe_store!(reinterpret(Ptr{UInt32}, ptr), UInt32(metadata.upper_count), 3)
    end

    # Compress and write
    compressed = compress_zlib(full_buffer)
    write(filepath, compressed)

    return nothing
end

"""
    save_nanovdb(filepath, data::Array{Float32,3}, origin, extent)

Build a NanoVDB tree from a dense 3D array and save to file.
"""
function save_nanovdb(filepath::String, data::Array{Float32,3}, origin, extent)
    buffer, metadata = build_nanovdb_from_dense(data, Point3f(origin...), Vec3f(extent...))
    save_nanovdb(filepath, buffer, metadata)
end

"""
    NanoVDBMedium(data::Array{Float32,3}; bounds, σ_a, σ_s, g, majorant_res, buffer_alloc)

Construct a NanoVDBMedium from a dense 3D array by building a sparse NanoVDB tree.
Only leaf blocks (8³) containing non-zero voxels are stored, giving significant
memory savings for sparse volumetric data like cloud fields.

# Arguments
- `data`: Dense 3D Float32 array of density/opacity values
- `bounds`: World-space bounding box (`Bounds3`)
- `σ_a`: Absorption coefficient (default: 0 for pure scattering clouds)
- `σ_s`: Scattering coefficient (default: 1)
- `g`: Henyey-Greenstein asymmetry parameter
- `majorant_res`: Resolution of majorant grid (default: 64³)
- `buffer_alloc`: Array constructor for target device (default: `Vector{UInt8}`)
"""
function NanoVDBMedium(
    data::Array{Float32, 3};
    bounds::Bounds3,
    σ_a::RGBSpectrum = RGBSpectrum(0f0),
    σ_s::RGBSpectrum = RGBSpectrum(1f0),
    g::Float32 = 0f0,
    majorant_res::Vec3i = Vec3i(64, 64, 64),
    buffer_alloc = Vector{UInt8}
)
    origin = Point3f(bounds.p_min)
    extent = Vec3f(bounds.p_max - bounds.p_min)

    buffer, metadata = build_nanovdb_from_dense(data, origin, extent)

    # Build majorant grid
    majorant_grid = build_nanovdb_majorant_grid(buffer, metadata, bounds, majorant_res)

    gpu_buffer = buffer_alloc(buffer)

    NanoVDBMedium(
        gpu_buffer,
        metadata.root_offset,
        metadata.upper_offset,
        metadata.lower_offset,
        metadata.leaf_offset,
        metadata.leaf_count,
        metadata.lower_count,
        metadata.upper_count,
        metadata.root_table_size,
        metadata.inv_mat,
        metadata.vec,
        bounds,
        metadata.index_min,
        metadata.index_max,
        σ_a,
        σ_s,
        g,
        majorant_grid,
        Float32(maximum(majorant_grid.voxels))
    )
end

# ============================================================================
# NanoVDB File Parsing and Medium Construction
# ============================================================================

# Zlib stream structure for decompression (must be defined before use)
mutable struct ZStream
    next_in::Ptr{UInt8}
    avail_in::Cuint
    total_in::Culong
    next_out::Ptr{UInt8}
    avail_out::Cuint
    total_out::Culong
    msg::Ptr{Cchar}
    state::Ptr{Cvoid}
    zalloc::Ptr{Cvoid}
    zfree::Ptr{Cvoid}
    opaque::Ptr{Cvoid}
    data_type::Cint
    adler::Culong
    reserved::Culong
end

ZStream() = ZStream(
    C_NULL, 0, 0,
    C_NULL, 0, 0,
    C_NULL, C_NULL,
    C_NULL, C_NULL, C_NULL,
    0, 0, 0
)


function compress_zlib(data::Vector{UInt8})
    out_buf = Vector{UInt8}(undef, length(data) + div(length(data), 100) + 1024)

    z = Ref(ZStream())
    z[].next_in = pointer(data)
    z[].avail_in = length(data)
    z[].next_out = pointer(out_buf)
    z[].avail_out = length(out_buf)

    ret = ccall((:deflateInit_, Zlib_jll.libz), Cint,
        (Ref{ZStream}, Cint, Cstring, Cint), z, 6, "1.2.11", sizeof(ZStream))
    ret != 0 && error("deflateInit failed: $ret")

    ret = ccall((:deflate, Zlib_jll.libz), Cint, (Ref{ZStream}, Cint), z, 4)  # Z_FINISH=4
    compressed_size = z[].total_out
    ccall((:deflateEnd, Zlib_jll.libz), Cint, (Ref{ZStream},), z)

    return out_buf[1:compressed_size]
end


function decompress_nanovdb(compressed_data::Vector{UInt8})
    # Use Zlib_jll for decompression
    output_buffer = Vector{UInt8}(undef, 200_000_000)  # 200MB should be enough

    z = Ref(ZStream())
    z[].next_in = pointer(compressed_data)
    z[].avail_in = length(compressed_data)
    z[].next_out = pointer(output_buffer)
    z[].avail_out = length(output_buffer)

    ret = ccall((:inflateInit_, Zlib_jll.libz), Cint,
                (Ref{ZStream}, Cstring, Cint), z, "1.2.11", sizeof(ZStream))
    ret != 0 && error("inflateInit failed: $ret")

    ret = ccall((:inflate, Zlib_jll.libz), Cint, (Ref{ZStream}, Cint), z, 4)  # Z_FINISH=4
    decompressed_size = z[].total_out
    ccall((:inflateEnd, Zlib_jll.libz), Cint, (Ref{ZStream},), z)

    return output_buffer[1:decompressed_size]
end

"""
    parse_nanovdb_buffer(filepath::String) -> (buffer, metadata)

Parse a NanoVDB file and return the decompressed buffer along with metadata.
Returns raw buffer suitable for GPU upload.
"""
function parse_nanovdb_buffer(filepath::String)
    nvdb_data = read(filepath)

    # Find zlib header (0x78 0x9c for default compression)
    compressed_start = 0
    for i in 1:min(500, length(nvdb_data)-1)
        if nvdb_data[i] == 0x78 && nvdb_data[i+1] in (0x01, 0x5e, 0x9c, 0xda)
            compressed_start = i
            break
        end
    end
    compressed_start == 0 && error("Could not find zlib header in NanoVDB file")

    # Decompress using external zlib
    compressed_data = nvdb_data[compressed_start:end]
    buffer = decompress_nanovdb(compressed_data)

    # Extract metadata
    metadata = extract_nanovdb_metadata(buffer)

    return buffer, metadata
end

"""Extract metadata from decompressed NanoVDB buffer"""
function extract_nanovdb_metadata(buffer::Vector{UInt8})
    ptr = pointer(buffer)

    # Read world bounds from GridData (offset 561-608, 6 doubles)
    world_bbox = reinterpret(Float64, buffer[WORLDBBOX_OFFSET:WORLDBBOX_OFFSET+47])
    world_min = (Float32(world_bbox[1]), Float32(world_bbox[2]), Float32(world_bbox[3]))
    world_max = (Float32(world_bbox[4]), Float32(world_bbox[5]), Float32(world_bbox[6]))

    # Read affine transform from Map structure
    inv_mat = ntuple(i -> read_float32(ptr, MAP_INVMATF_OFFSET + (i-1)*4), 9)
    vec = ntuple(i -> read_float32(ptr, MAP_VECF_OFFSET + (i-1)*4), 3)

    # Read tree offsets
    node_offsets = ntuple(i -> read_uint64(ptr, TREEDATA_NODE_OFFSET_START + (i-1)*8), 4)
    node_counts = ntuple(i -> read_uint32(ptr, TREEDATA_NODE_COUNT_START + (i-1)*4), 3)

    # Node offsets are relative to TreeData start, convert to absolute (1-indexed)
    tree_data_start = NANOVDB_GRIDDATA_SIZE + 1
    leaf_offset = tree_data_start + node_offsets[1]
    lower_offset = tree_data_start + node_offsets[2]
    upper_offset = tree_data_start + node_offsets[3]
    root_offset = tree_data_start + node_offsets[4]

    # Read root table size
    root_table_size = read_uint32(ptr, root_offset + ROOTDATA_TABLESIZE_OFFSET)

    # Compute index bounding box from leaf coordinates
    leaf_count = Int(node_counts[1])
    index_min = (typemax(Int32), typemax(Int32), typemax(Int32))
    index_max = (typemin(Int32), typemin(Int32), typemin(Int32))

    for i in 0:leaf_count-1
        leaf_off = leaf_offset + i * LEAFDATA_SIZE
        # LeafData starts with mBBoxMin (3 x Int32 = 12 bytes)
        cx = read_int32(ptr, leaf_off)
        cy = read_int32(ptr, leaf_off + 4)
        cz = read_int32(ptr, leaf_off + 8)
        index_min = (min(index_min[1], cx), min(index_min[2], cy), min(index_min[3], cz))
        index_max = (max(index_max[1], Int32(cx + LEAF_DIM)), max(index_max[2], Int32(cy + LEAF_DIM)), max(index_max[3], Int32(cz + LEAF_DIM)))
    end

    return (
        world_min = world_min,
        world_max = world_max,
        inv_mat = inv_mat,
        vec = vec,
        root_offset = Int64(root_offset),
        upper_offset = Int64(upper_offset),
        lower_offset = Int64(lower_offset),
        leaf_offset = Int64(leaf_offset),
        leaf_count = Int32(node_counts[1]),
        lower_count = Int32(node_counts[2]),
        upper_count = Int32(node_counts[3]),
        root_table_size = Int32(root_table_size),
        index_min = index_min,
        index_max = index_max
    )
end

"""
    build_nanovdb_majorant_grid(medium_partial, res::Vec3i) -> MajorantGrid

Build a majorant grid for NanoVDBMedium by sampling max density in each cell.
Matches pbrt-v4's majorant grid construction.
"""
function build_nanovdb_majorant_grid(
    buffer::Vector{UInt8},
    metadata::NamedTuple,
    bounds::Bounds3,
    res::Vec3i = Vec3i(64, 64, 64)
)
    grid = MajorantGrid(res, Vector{Float32})

    # Create temporary medium-like structure for sampling
    temp_medium = (
        buffer = buffer,
        root_offset = metadata.root_offset,
        root_table_size = metadata.root_table_size,
        inv_mat = metadata.inv_mat,
        vec = metadata.vec
    )

    # For each majorant cell, find max density
    diag = bounds.p_max - bounds.p_min

    for iz in 0:res[3]-1
        for iy in 0:res[2]-1
            for ix in 0:res[1]-1
                # World bounds of this majorant cell
                p_min = Point3f(
                    bounds.p_min[1] + diag[1] * ix / res[1],
                    bounds.p_min[2] + diag[2] * iy / res[2],
                    bounds.p_min[3] + diag[3] * iz / res[3]
                )
                p_max = Point3f(
                    bounds.p_min[1] + diag[1] * (ix + 1) / res[1],
                    bounds.p_min[2] + diag[2] * (iy + 1) / res[2],
                    bounds.p_min[3] + diag[3] * (iz + 1) / res[3]
                )

                # Transform to index space
                i_min = world_to_index_f_raw(metadata.inv_mat, metadata.vec, p_min)
                i_max = world_to_index_f_raw(metadata.inv_mat, metadata.vec, p_max)

                # Get integer bounds with filter slop (matching pbrt's delta=1)
                delta = 1f0
                nx0 = max(floor(Int32, min(i_min[1], i_max[1]) - delta), metadata.index_min[1])
                nx1 = min(ceil(Int32, max(i_min[1], i_max[1]) + delta), metadata.index_max[1])
                ny0 = max(floor(Int32, min(i_min[2], i_max[2]) - delta), metadata.index_min[2])
                ny1 = min(ceil(Int32, max(i_min[2], i_max[2]) + delta), metadata.index_max[2])
                nz0 = max(floor(Int32, min(i_min[3], i_max[3]) - delta), metadata.index_min[3])
                nz1 = min(ceil(Int32, max(i_min[3], i_max[3]) + delta), metadata.index_max[3])

                # Find max value in this range
                max_val = 0f0
                for nz in nz0:nz1, ny in ny0:ny1, nx in nx0:nx1
                    val = nanovdb_get_value_raw(buffer, metadata, (Int32(nx), Int32(ny), Int32(nz)))
                    max_val = max(max_val, val)
                end

                majorant_set!(grid, ix, iy, iz, max_val)
            end
        end
    end

    return grid
end

# Raw versions of functions that work without full NanoVDBMedium struct
@inline function world_to_index_f_raw(inv_mat::NTuple{9,Float32}, vec::NTuple{3,Float32}, p::Point3f)::Point3f
    px = p[1] - vec[1]
    py = p[2] - vec[2]
    pz = p[3] - vec[3]
    ix = inv_mat[1]*px + inv_mat[2]*py + inv_mat[3]*pz
    iy = inv_mat[4]*px + inv_mat[5]*py + inv_mat[6]*pz
    iz = inv_mat[7]*px + inv_mat[8]*py + inv_mat[9]*pz
    return Point3f(ix, iy, iz)
end

@inline @propagate_inbounds function nanovdb_get_value_raw(buffer::Vector{UInt8}, metadata::NamedTuple, ijk::NTuple{3,Int32})::Float32
    ptr = pointer(buffer)
    root_offset = metadata.root_offset
    table_size = metadata.root_table_size

    # Root lookup
    key = coord_to_root_key(ijk)
    tile_found = false
    tile_offset = Int64(0)
    tile_base = root_offset + ROOTDATA_HEADER_SIZE

    for i in Int32(0):table_size-Int32(1)
        t_offset = tile_base + i * ROOTTILE_SIZE
        t_key = read_uint64(ptr, t_offset + ROOTTILE_KEY_OFFSET)
        if t_key == key
            tile_found = true
            tile_offset = t_offset
            break
        end
    end

    if !tile_found
        return read_float32(ptr, root_offset + ROOTDATA_BACKGROUND_OFFSET)
    end

    child_offset = read_int64(ptr, tile_offset + ROOTTILE_CHILD_OFFSET)
    if child_offset == 0
        return read_float32(ptr, tile_offset + ROOTTILE_VALUE_OFFSET)
    end

    # Upper node
    upper_off = root_offset + child_offset
    n_upper = upper_coord_to_offset(ijk)

    if !bitmask_is_on(ptr, upper_off + UPPER_CHILDMASK_OFFSET, n_upper)
        return read_float32(ptr, upper_off + UPPER_TABLE_OFFSET + n_upper * 8)
    end

    upper_child_offset = read_int64(ptr, upper_off + UPPER_TABLE_OFFSET + n_upper * 8)

    # Lower node
    lower_off = upper_off + upper_child_offset
    n_lower = lower_coord_to_offset(ijk)

    if !bitmask_is_on(ptr, lower_off + LOWER_CHILDMASK_OFFSET, n_lower)
        return read_float32(ptr, lower_off + LOWER_TABLE_OFFSET + n_lower * 8)
    end

    lower_child_offset = read_int64(ptr, lower_off + LOWER_TABLE_OFFSET + n_lower * 8)

    # Leaf node
    leaf_off = lower_off + lower_child_offset
    n_leaf = leaf_coord_to_offset(ijk)
    return read_float32(ptr, leaf_off + LEAFDATA_VALUES_OFFSET + n_leaf * 4)
end

"""
    NanoVDBMedium(filepath::String; σ_a, σ_s, g, transform, majorant_res)

Construct a NanoVDBMedium from a NanoVDB file.

# Arguments
- `filepath`: Path to the .nvdb file
- `σ_a`: Absorption coefficient (RGBSpectrum)
- `σ_s`: Scattering coefficient (RGBSpectrum)
- `g`: Henyey-Greenstein asymmetry parameter
- `transform`: Optional 3x3 rotation matrix (medium-to-world, like pbrt's Rotate)
- `majorant_res`: Resolution of majorant grid (default 64³)

For pbrt bunny-cloud scene, use:
  transform = RotZ(π) * RotX(π/2)  # Rotate 180° around Z, then 90° around X
"""
function NanoVDBMedium(
    filepath::String;
    σ_a::RGBSpectrum = RGBSpectrum(0.5f0),
    σ_s::RGBSpectrum = RGBSpectrum(10f0),
    g::Float32 = 0f0,
    transform::Mat3f = Mat3f(I),
    majorant_res::Vec3i = Vec3i(64, 64, 64),
    buffer_alloc = Vector{UInt8}
)
    # Parse file
    buffer, metadata = parse_nanovdb_buffer(filepath)

    # Get NanoVDB's internal transform as Mat3f
    nanovdb_inv_mat = Mat3f(
        metadata.inv_mat[1], metadata.inv_mat[2], metadata.inv_mat[3],
        metadata.inv_mat[4], metadata.inv_mat[5], metadata.inv_mat[6],
        metadata.inv_mat[7], metadata.inv_mat[8], metadata.inv_mat[9]
    )

    # Compose transforms: world → medium → index
    # transform is medium-to-world (like pbrt Rotate), so we need its inverse
    # Combined: p_index = nanovdb_inv_mat * (inv(transform) * p_world - nanovdb_vec)
    #         = nanovdb_inv_mat * inv(transform) * p_world - nanovdb_inv_mat * nanovdb_vec
    # Let R = inv(transform), then:
    #   combined_mat = nanovdb_inv_mat * R
    #   combined_vec = nanovdb_vec (applied after R, before nanovdb_inv_mat)
    # Actually: p_index = nanovdb_inv_mat * (R * p_world - nanovdb_vec)
    #                   = (nanovdb_inv_mat * R) * p_world - nanovdb_inv_mat * nanovdb_vec
    # So: combined_inv_mat = nanovdb_inv_mat * inv(transform)
    #     combined_vec stays the same (subtracted before matrix multiply in original)
    # Wait, the original is: p_index = inv_mat * (p - vec)
    # With rotation: p_medium = inv(transform) * p_world
    #                p_index = inv_mat * (p_medium - vec)
    #                        = inv_mat * (inv(transform) * p_world - vec)
    # This means vec is in medium space, not world space.
    # For the bunny scene, vec = (0,0,0), so this is fine.

    inv_transform = inv(transform)
    combined_inv_mat = nanovdb_inv_mat * inv_transform

    # Convert back to tuple for storage
    inv_mat_tuple = (
        combined_inv_mat[1,1], combined_inv_mat[1,2], combined_inv_mat[1,3],
        combined_inv_mat[2,1], combined_inv_mat[2,2], combined_inv_mat[2,3],
        combined_inv_mat[3,1], combined_inv_mat[3,2], combined_inv_mat[3,3]
    )

    # Transform world bounds using the medium-to-world transform
    # The original bounds are in medium space, transform to world space
    corners_medium = [
        Vec3f(metadata.world_min...),
        Vec3f(metadata.world_min[1], metadata.world_min[2], metadata.world_max[3]),
        Vec3f(metadata.world_min[1], metadata.world_max[2], metadata.world_min[3]),
        Vec3f(metadata.world_min[1], metadata.world_max[2], metadata.world_max[3]),
        Vec3f(metadata.world_max[1], metadata.world_min[2], metadata.world_min[3]),
        Vec3f(metadata.world_max[1], metadata.world_min[2], metadata.world_max[3]),
        Vec3f(metadata.world_max[1], metadata.world_max[2], metadata.world_min[3]),
        Vec3f(metadata.world_max...)
    ]
    corners_world = [transform * c for c in corners_medium]
    xs = [c[1] for c in corners_world]
    ys = [c[2] for c in corners_world]
    zs = [c[3] for c in corners_world]
    world_min = Point3f(minimum(xs), minimum(ys), minimum(zs))
    world_max = Point3f(maximum(xs), maximum(ys), maximum(zs))
    bounds = Bounds3(world_min, world_max)

    # Build majorant grid with transformed metadata
    transformed_metadata = (
        inv_mat = inv_mat_tuple,
        vec = metadata.vec,
        index_min = metadata.index_min,
        index_max = metadata.index_max,
        root_offset = metadata.root_offset,
        root_table_size = metadata.root_table_size,
    )
    majorant_grid = build_nanovdb_majorant_grid(buffer, transformed_metadata, bounds, majorant_res)

    # Convert buffer to target array type
    gpu_buffer = buffer_alloc(buffer)

    NanoVDBMedium(
        gpu_buffer,
        metadata.root_offset,
        metadata.upper_offset,
        metadata.lower_offset,
        metadata.leaf_offset,
        metadata.leaf_count,
        metadata.lower_count,
        metadata.upper_count,
        metadata.root_table_size,
        inv_mat_tuple,
        metadata.vec,
        bounds,
        metadata.index_min,
        metadata.index_max,
        σ_a,
        σ_s,
        g,
        majorant_grid,
        Float32(maximum(majorant_grid.voxels))
    )
end
