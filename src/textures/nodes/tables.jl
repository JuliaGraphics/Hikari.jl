# The tables are Blender Cycles' (v5.2.1): intern/cycles/scene/shader.tables, read as
# kernel/util/lookup_table.h reads them. Copyright 2011-2022 Blender Foundation.
# SPDX-License-Identifier: Apache-2.0. See LICENSES/Apache-2.0.txt.

# ============================================================================
# Cycles' precomputed lookup tables
# ============================================================================
#
# The albedos of the GGX lobes, their averages, the generalized Schlick
# albedos, the sheen LTC fit and the thin film colour matching functions, as
# `intern/cycles/scene/shader.tables` holds them, concatenated in
# `materials/data/cycles_tables.f32`.

"""One value of the lookup tables: its own element type, so the table's texture slot is known at compile time."""
struct LookupValue
    v::Float32
end

module LookupTable
const GGX_E = 0
const GGX_EAVG = 1024
const GGX_GLASS_E = 1056
const GGX_GLASS_EAVG = 5152
const GGX_GLASS_INV_E = 5408
const GGX_GLASS_INV_EAVG = 9504
const SHEEN_LTC = 9760
const GGX_GEN_SCHLICK_IOR_S = 12832
const GGX_GEN_SCHLICK_S = 16928
const THIN_FILM_CMF = 21024
const LENGTH = 24096
end

const LOOKUP_TABLE_PATH = joinpath(@__DIR__, "..", "..", "materials", "data", "cycles_tables.f32")

function lookup_table_values()
    raw = reinterpret(Float32, read(LOOKUP_TABLE_PATH))
    length(raw) == LookupTable.LENGTH || error("lookup tables in $LOOKUP_TABLE_PATH have $(length(raw)) values, expected $(LookupTable.LENGTH)")
    return LookupValue.(raw)
end

"""Store the lookup tables in `dhv` once: every shader reads the same ones."""
function store_lookup_tables!(dhv)
    any(T -> eltype(T) === LookupValue, dhv.texture_order) && return nothing
    Raycore.store_texture(dhv, lookup_table_values())
    return nothing
end

"""The lookup tables in a texture set, found by their element type; `nothing` in a set without them."""
@generated function lookup_tables(textures::Raycore.StaticMultiTypeSet{D, Tex}) where {D, Tex}
    slot = findfirst(T -> eltype(eltype(T)) === LookupValue, Tex.parameters)
    slot === nothing && return :(nothing)
    return :(@inbounds textures.textures[$slot][1])
end
lookup_tables(textures) = nothing

@inline function lookup_table_read(lut::AbstractVector{LookupValue}, x::Float32, offset::Int32, size::Int32)
    x = clamp(x, 0f0, 1f0) * Float32(size - Int32(1))
    index = min(unsafe_trunc(Int32, x), size - Int32(1))
    nindex = min(index + Int32(1), size - Int32(1))
    t = x - Float32(index)
    data0 = (@inbounds lut[offset + index + Int32(1)]).v
    t == 0f0 && return data0
    data1 = (@inbounds lut[offset + nindex + Int32(1)]).v
    return (1f0 - t) * data0 + t * data1
end

@inline function lookup_table_read_2d(lut::AbstractVector{LookupValue}, x::Float32, y::Float32, offset::Int32,
                                      xsize::Int32, ysize::Int32)
    y = clamp(y, 0f0, 1f0) * Float32(ysize - Int32(1))
    index = min(unsafe_trunc(Int32, y), ysize - Int32(1))
    nindex = min(index + Int32(1), ysize - Int32(1))
    t = y - Float32(index)
    data0 = lookup_table_read(lut, x, offset + xsize * index, xsize)
    t == 0f0 && return data0
    data1 = lookup_table_read(lut, x, offset + xsize * nindex, xsize)
    return (1f0 - t) * data0 + t * data1
end

@inline function lookup_table_read_3d(lut::AbstractVector{LookupValue}, x::Float32, y::Float32, z::Float32,
                                      offset::Int32, xsize::Int32, ysize::Int32, zsize::Int32)
    z = clamp(z, 0f0, 1f0) * Float32(zsize - Int32(1))
    index = min(unsafe_trunc(Int32, z), zsize - Int32(1))
    nindex = min(index + Int32(1), zsize - Int32(1))
    t = z - Float32(index)
    data0 = lookup_table_read_2d(lut, x, y, offset + xsize * ysize * index, xsize, ysize)
    t == 0f0 && return data0
    data1 = lookup_table_read_2d(lut, x, y, offset + xsize * ysize * nindex, xsize, ysize)
    return (1f0 - t) * data0 + t * data1
end

lookup_table_read(::Nothing, x, offset, size) = 1f0
lookup_table_read_2d(::Nothing, x, y, offset, xsize, ysize) = 1f0
lookup_table_read_3d(::Nothing, x, y, z, offset, xsize, ysize, zsize) = 1f0
