# A flat cache of uniform 8³ auxiliary-bound tiles. The fine dense grid is
# retained verbatim; NaN marks a nonuniform tile and falls back to fine DDA.
# Physical density sampling remains on its original NanoVDB grids.
struct TiledMajorantValues{F,T} <: AbstractVector{Float32}
    fine::F
    tiles::T
    res::Vec{3,Int32}
    tile_res::Vec{3,Int32}
end
Adapt.@adapt_structure TiledMajorantValues
Base.size(v::TiledMajorantValues) = (Int(prod(v.res)),)
Base.IndexStyle(::Type{<:TiledMajorantValues}) = IndexLinear()
function Base.show(io::IO,v::TiledMajorantValues)
    # Converted fields contain TextureRefs, which cannot be indexed on the
    # host. Error formatting must not try to enumerate a device-side grid.
    print(io,"TiledMajorantValues(",Tuple(v.res),", ",typeof(v.fine),", ",typeof(v.tiles),")")
end
Base.show(io::IO,::MIME"text/plain",v::TiledMajorantValues)=show(io,v)
@inline Base.getindex(v::TiledMajorantValues,i::Int) = v.fine[i]
function Raycore.maybe_convert_field(set::Raycore.MultiTypeSet,v::TiledMajorantValues)
    TiledMajorantValues(Raycore.maybe_convert_field(set,v.fine),
        Raycore.maybe_convert_field(set,v.tiles),v.res,v.tile_res)
end
function Raycore.update_item(set::Raycore.MultiTypeSet,old::TiledMajorantValues,new::TiledMajorantValues)
    TiledMajorantValues(Raycore.update_item(set,old.fine,new.fine),
        Raycore.update_item(set,old.tiles,new.tiles),new.res,new.tile_res)
end

function tiledmajorants(grid::MajorantGrid)
    fine=Array(grid.voxels)
    res=grid.res
    tr=(res .+ Int32(7)) .>> Int32(3)
    tiles=Vector{Float32}(undef,prod(tr))
    @inbounds for z in 0:tr[3]-1,y in 0:tr[2]-1,x in 0:tr[1]-1
        first_value=fine[8x+res[1]*(8y+res[2]*8z)+1]
        uniform=true
        for k in 8z:min(8z+7,res[3]-1),j in 8y:min(8y+7,res[2]-1),i in 8x:min(8x+7,res[1]-1)
            uniform &= fine[i+res[1]*(j+res[2]*k)+1]==first_value
        end
        tiles[x+tr[1]*(y+tr[2]*z)+1]=uniform ? first_value : NaN32
    end
    MajorantGrid(TiledMajorantValues(fine,tiles,res,tr),res)
end
tiledmajorants(grid::MajorantGrid{<:TiledMajorantValues}) = grid

@inline @propagate_inbounds function majorant_lookup(grid::MajorantGrid{<:TiledMajorantValues},media,x::Integer,y::Integer,z::Integer)
    v=grid.voxels
    idx=Int32(x)+v.res[1]*(Int32(y)+v.res[2]*Int32(z))+Int32(1)
    return Raycore.deref(media,v.fine)[idx]
end

@inline @propagate_inbounds function ray_majorant_next_dda(iter::RayMajorantIterator{<:MajorantGrid{<:TiledMajorantValues}},media)
    iter.t_min>=iter.t_max && return (RayMajorantSegment(),iter,false)
    values=iter.grid.voxels
    q=ntuple(a->iter.voxel[a]>>Int32(3),3)
    idx=q[1]+values.tile_res[1]*(q[2]+values.tile_res[2]*q[3])+Int32(1)
    density=Raycore.deref(media,values.tiles)[idx]
    isnan(density) && return ray_majorant_next_dda_step(iter,media,
        majorant_lookup(iter.grid,media,iter.voxel...))
    return majorant_tile_step(iter,density,Int32(8))
end

"Cross one aligned uniform tile, then resume with the original fine DDA state."
@inline function majorant_tile_step(iter::RayMajorantIterator,density::Float32,dimension::Int32)
    exits=ntuple(3) do a
        v=iter.voxel[a];d=iter.delta_t[a]
        aligned=v & -dimension
        boundary=iter.step[a]>0 ? min(aligned+dimension,iter.grid_res[a]) : aligned
        cells=iter.step[a]>0 ? boundary-v : v-boundary+Int32(1)
        isfinite(d) ? iter.next_crossing_t[a]+Float32(cells-Int32(1))*d : Inf32
    end
    stop=min(iter.t_max,exits[1],exits[2],exits[3])
    crossed=ntuple(3) do a
        d=iter.delta_t[a];first=iter.next_crossing_t[a]
        if exits[a]==stop && isfinite(d)
            v=iter.voxel[a];aligned=v & -dimension
            boundary=iter.step[a]>0 ? min(aligned+dimension,iter.grid_res[a]) : aligned
            iter.step[a]>0 ? boundary-v : v-boundary+Int32(1)
        else
            isfinite(d) && stop>=first ? floor_int32((stop-first)/d)+Int32(1) : Int32(0)
        end
    end
    voxel=ntuple(a->iter.voxel[a]+iter.step[a]*crossed[a],3)
    next_t=ntuple(a->isfinite(iter.delta_t[a]) ? iter.next_crossing_t[a]+Float32(crossed[a])*iter.delta_t[a] : Inf32,3)
    mode=all(a->Int32(0)<=voxel[a]<iter.grid_res[a],1:3) && stop<iter.t_max ? Int32(2) : Int32(0)
    next=typeof(iter)(mode,iter.σ_t,stop,iter.t_max,false,iter.grid,iter.grid_res,next_t,
        iter.delta_t,iter.step,iter.voxel_limit,voxel)
    return (RayMajorantSegment(iter.t_min,stop,iter.σ_t*density),next,true)
end
