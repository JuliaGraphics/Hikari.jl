# The same uniform-tile walk as TiledMajorantValues, without retaining a dense
# copy of the empty/constant cells. Every nonuniform 8³ tile stays Float32.
struct PackedMajorantValues{F,T,O} <: AbstractVector{Float32}
    fine::F
    tiles::T
    offsets::O
    res::Vec{3,Int32}
    tile_res::Vec{3,Int32}
end
Adapt.@adapt_structure PackedMajorantValues
Base.size(v::PackedMajorantValues) = (prod(Int.(v.res)),)
Base.IndexStyle(::Type{<:PackedMajorantValues}) = IndexLinear()

@inline function packed_majorant_value(v,media,x,y,z)
    qx,qy,qz = Int32(x),Int32(y),Int32(z)
    tile = (qx>>3)+v.tile_res[1]*((qy>>3)+v.tile_res[2]*(qz>>3))+Int32(1)
    offset = Raycore.deref(media,v.offsets)[tile]
    offset==UInt32(0) && return Raycore.deref(media,v.tiles)[tile]
    local_index = (qx&Int32(7))+Int32(8)*((qy&Int32(7))+Int32(8)*(qz&Int32(7)))
    return Raycore.deref(media,v.fine)[offset+UInt32(local_index)+UInt32(1)]
end

@inline function Base.getindex(v::PackedMajorantValues,i::Int)
    @boundscheck checkbounds(v,i)
    q = i-1
    x = q%v.res[1]
    y = (q÷v.res[1])%v.res[2]
    z = q÷(Int(v.res[1])*v.res[2])
    @inbounds return packed_majorant_value(v,nothing,x,y,z)
end

function Raycore.maybe_convert_field(set::Raycore.MultiTypeSet,v::PackedMajorantValues)
    if !(set.backend isa KA.CPU)
        tiled=device_tiledmajorants(MajorantGrid(v,v.res),set.backend)
        return Raycore.maybe_convert_field(set,tiled.voxels)
    end
    PackedMajorantValues(Raycore.maybe_convert_field(set,v.fine),
        Raycore.maybe_convert_field(set,v.tiles),Raycore.maybe_convert_field(set,v.offsets),v.res,v.tile_res)
end

@kernel inbounds=true function unpack_majorant_cells!(destination,source,first::Int32,n::Int32)
    local_index=@index(Global)
    if local_index<=n
        index=first+Int32(local_index)
        q=index-Int32(1)
        x=q%source.res[1]
        y=(q÷source.res[1])%source.res[2]
        z=q÷(source.res[1]*source.res[2])
        destination[index]=packed_majorant_value(source,nothing,x,y,z)
    end
end

"Expand compact host bounds directly into reusable device tiles, preserving every cell."
function device_tiledmajorants(grid::MajorantGrid{<:PackedMajorantValues},backend;reuse=nothing)
    count=prod(Int.(grid.res))
    count<=typemax(Int32) || error("Device auxiliary grid exceeds Int32 cell indexing")
    source=Adapt.adapt(backend,grid.voxels)
    fine=reuse===nothing ? Mantle.devicearray(backend,Float32,count) : reuse.voxels.fine
    length(fine)==count || throw(ArgumentError("Reusable majorant extent differs"))
    tiles=if reuse===nothing
        source.tiles
    else
        copyto!(reuse.voxels.tiles,source.tiles)
        reuse.voxels.tiles
    end
    chunk=8*1024*1024
    GC.@preserve source fine tiles begin
        for first in 0:chunk:count-1
            n=min(chunk,count-first)
            unpack_majorant_cells!(backend,256)(fine,source,Int32(first),Int32(n);ndrange=n)
        end
        Mantle.waitidle(mantle_device(backend))
    end
    return MajorantGrid(TiledMajorantValues(fine,tiles,grid.res,source.tile_res),grid.res)
end

function Raycore.update_item(set::Raycore.MultiTypeSet,old::TiledMajorantValues,new::PackedMajorantValues)
    if old.res!=new.res
        tiled=device_tiledmajorants(MajorantGrid(new,new.res),set.backend)
        return Raycore.update_item(set,old,tiled.voxels)
    end
    fine,tiles=Raycore.deref(set,old.fine),Raycore.deref(set,old.tiles)
    reuse=MajorantGrid(TiledMajorantValues(fine,tiles,old.res,old.tile_res),old.res)
    device_tiledmajorants(MajorantGrid(new,new.res),set.backend;reuse)
    return old
end
function Raycore.update_item(set::Raycore.MultiTypeSet,old::PackedMajorantValues,new::PackedMajorantValues)
    PackedMajorantValues(Raycore.update_item(set,old.fine,new.fine),
        Raycore.update_item(set,old.tiles,new.tiles),Raycore.update_item(set,old.offsets,new.offsets),new.res,new.tile_res)
end

@inline uniform_majorant_tile(v,x,y,z) = NaN32
@inline function uniform_majorant_tile(v::Union{TiledMajorantValues,PackedMajorantValues},x,y,z)
    @inbounds return v.tiles[x+v.tile_res[1]*(y+v.tile_res[2]*z)+1]
end

function build_packed_majorants(res,sample,uniform_tile)
    tr = (res .+ Int32(7)) .>> Int32(3)
    tiles = Vector{Float32}(undef,prod(Int.(tr)))
    offsets = zeros(UInt32,length(tiles))
    # Keep a valid, minimal device allocation even for a completely uniform grid.
    fine = Float32[0]
    block = Vector{Float32}(undef,512)
    @inbounds for z in 0:tr[3]-1,y in 0:tr[2]-1,x in 0:tr[1]-1
        tile = x+tr[1]*(y+tr[2]*z)+1
        value = uniform_tile(x,y,z)
        if !isnan(value)
            tiles[tile] = value
            continue
        end
        first_value = Float32(sample(8x,8y,8z))
        fill!(block,first_value)
        uniform = true
        for k in 0:min(Int32(7),res[3]-1-8z),j in 0:min(Int32(7),res[2]-1-8y),i in 0:min(Int32(7),res[1]-1-8x)
            value = Float32(sample(8x+i,8y+j,8z+k))
            block[i+8*(j+8*k)+1] = value
            uniform &= value==first_value
        end
        tiles[tile] = uniform ? first_value : NaN32
        if !uniform
            length(fine)<=typemax(UInt32)-512 || error("Packed majorant pool exceeds UInt32 addressing")
            offsets[tile] = UInt32(length(fine))
            append!(fine,block)
        end
    end
    return MajorantGrid(PackedMajorantValues(fine,tiles,offsets,res,tr),res)
end

"Losslessly pack a host majorant grid; physical NanoVDB density is untouched."
function packmajorants(grid::MajorantGrid)
    values,res = grid.voxels,grid.res
    sample(x,y,z) = values[x+res[1]*(y+res[2]*z)+1]
    uniform(x,y,z) = uniform_majorant_tile(values,x,y,z)
    return build_packed_majorants(res,sample,uniform)
end
packmajorants(grid::MajorantGrid{<:PackedMajorantValues}) = grid

"The exact cellwise maximum, retaining uniform tiles without expanding a dense grid."
function maxmajorants(a::MajorantGrid{<:PackedMajorantValues},b::MajorantGrid{<:PackedMajorantValues})
    a.res==b.res || throw(ArgumentError("Majorant grid resolutions differ"))
    va,vb = a.voxels,b.voxels
    sample(x,y,z) = max(packed_majorant_value(va,nothing,x,y,z),packed_majorant_value(vb,nothing,x,y,z))
    uniform(x,y,z) = max(uniform_majorant_tile(va,x,y,z),uniform_majorant_tile(vb,x,y,z))
    return build_packed_majorants(a.res,sample,uniform)
end

@inline @propagate_inbounds function majorant_lookup(grid::MajorantGrid{<:PackedMajorantValues},media,x::Integer,y::Integer,z::Integer)
    return packed_majorant_value(grid.voxels,media,x,y,z)
end

@inline @propagate_inbounds function ray_majorant_next_dda(iter::RayMajorantIterator{<:MajorantGrid{<:PackedMajorantValues}},media)
    iter.t_min>=iter.t_max && return (RayMajorantSegment(),iter,false)
    values = iter.grid.voxels
    q = ntuple(a->iter.voxel[a]>>Int32(3),3)
    idx = q[1]+values.tile_res[1]*(q[2]+values.tile_res[2]*q[3])+Int32(1)
    density = Raycore.deref(media,values.tiles)[idx]
    isnan(density) && return ray_majorant_next_dda_step(iter,media,
        majorant_lookup(iter.grid,media,iter.voxel...))
    return majorant_tile_step(iter,density,Int32(8))
end
