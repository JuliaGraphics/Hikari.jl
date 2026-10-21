# NanoVDB-backed auxiliary extinction bounds. The traversal follows NanoVDB's
# HDDA/getDim: inactive 8, 128 or 4096-voxel tiles are crossed in one interval.
# Active leaf voxels retain the original conservative local bounds.
# Retired rendering experiment: retain the accessor and integral probes for
# diagnosis, but WaterMedium construction accepts dense auxiliary bounds only.
# Isolated traversal passes on both GPUs; the assembled native wavefront shaders
# lose the AMD device and are substantially slower on NVIDIA.
struct NativeMajorantValues{T} <: AbstractVector{Float32}
    tree::T
    res::Vec{3,Int32}
    top::Float32
end
Base.size(v::NativeMajorantValues) = (Int(prod(v.res)),)
Base.IndexStyle(::Type{<:NativeMajorantValues}) = IndexLinear()
Base.maximum(v::NativeMajorantValues) = v.top
Base.only(v::NativeMajorantValues) = length(v)==1 ? v.top : throw(ArgumentError("grid has more than one value"))
@inline function Base.getindex(v::NativeMajorantValues,i::Int)
    @boundscheck checkbounds(v,i)
    n = i-1; r = v.res
    q = Vec{3,Int32}(n % r[1],(n÷r[1]) % r[2],n÷(r[1]*r[2]))
    return nanovalue(v.tree,v.tree.words,q)
end
function Raycore.maybe_convert_field(set::Raycore.MultiTypeSet, v::NativeMajorantValues)
    return NativeMajorantValues(Raycore.maybe_convert_field(set,v.tree),v.res,v.top)
end
function Raycore.update_item(set::Raycore.MultiTypeSet, old::NativeMajorantValues, new::NativeMajorantValues)
    return NativeMajorantValues(Raycore.update_item(set,old.tree,new.tree),new.res,new.top)
end

function sparsemajorants(grid::MajorantGrid, bounds::Bounds3, background; backend)
    res = grid.res
    origins = Vec{3,Int32}[]; values = Float32[]
    scratch = Vector{Float32}(undef,512)
    for z in Int32(0):Int32(8):res[3]-Int32(1), y in Int32(0):Int32(8):res[2]-Int32(1),
        x in Int32(0):Int32(8):res[1]-Int32(1)
        active = false
        for i in Int32(0):Int32(7), j in Int32(0):Int32(7), k in Int32(0):Int32(7)
            a,b,c = x+i,y+j,z+k
            v = (a<res[1] && b<res[2] && c<res[3]) ? grid.voxels[a+res[1]*(b+res[2]*c)+1] : Float32(background)
            scratch[64i+8j+k+1] = v
            active |= v != background
        end
        if active
            push!(origins,Vec{3,Int32}(x,y,z)); append!(values,scratch)
        end
    end
    # A leaf-free homogeneous grid needs no sparse allocation or traversal.
    isempty(origins) && return grid
    cell = Vec3f(bounds.p_max-bounds.p_min)./Vec3f(res)
    all(isapprox.(cell,cell[1];rtol=1f-5)) || throw(ArgumentError("NanoVDB majorants require uniform voxels"))
    tree = NanoTree(Mantle.devicearray(backend,origins),Mantle.devicearray(backend,values),
                    Point3f(bounds.p_min),cell[1]; background=Float32(background),backend)
    return MajorantGrid(NativeMajorantValues(tree,res,maximum(grid.voxels)),res)
end

@inline function majorant_lookup(grid::MajorantGrid{<:NativeMajorantValues}, media, x::Integer,y::Integer,z::Integer)
    t = grid.voxels.tree
    return nanovalue(t,Raycore.deref(media,t.words),Vec{3,Int32}(x,y,z))
end

"NanoVDB accessor getDim and value, sharing a single tree walk."
@noinline function majorant_tile(t::NanoTree,w,q::Vec{3,Int32})
    @inbounds begin
        key = coord_to_root_key(Tuple(q)); tile = Int32(-1)
        for i in Int32(0):t.roottiles-Int32(1)
            at = NT_ROOT_HEADER+i*NT_ROOT_TILE
            if reinterpret(UInt64,wordint64(w,at)) == key
                tile = at; break
            end
        end
        tile < 0 && return (Int32(4096),t.background)
        child = wordint64(w,tile+Int32(2))
        child == 0 && return (Int32(4096),wordfloat(w,tile+Int32(5)))
        up = Int32(child÷4); n = upper_coord_to_offset(Tuple(q))
        maskbit(w,up+NT_UPPER_CHILDMASK,n) || return (Int32(128),wordfloat(w,up+NT_UPPER_TABLE+2n))
        low = up+Int32(wordint64(w,up+NT_UPPER_TABLE+2n)÷4); m = lower_coord_to_offset(Tuple(q))
        maskbit(w,low+NT_LOWER_CHILDMASK,m) || return (Int32(8),wordfloat(w,low+NT_LOWER_TABLE+2m))
        leaf = low+Int32(wordint64(w,low+NT_LOWER_TABLE+2m)÷4)
        return (Int32(1),wordfloat(w,leaf+NT_LEAF_VALUES+leaf_coord_to_offset(Tuple(q))))
    end
end

@inline @propagate_inbounds function ray_majorant_next_dda(iter::RayMajorantIterator{<:MajorantGrid{<:NativeMajorantValues}}, media)
    iter.t_min >= iter.t_max && return (RayMajorantSegment(),iter,false)
    tree = iter.grid.voxels.tree
    dimension,density = majorant_tile(tree,Raycore.deref(media,tree.words),Vec{3,Int32}(iter.voxel))
    dimension == Int32(1) && return ray_majorant_next_dda_step(iter,media,density)
    # Aligned tile boundaries, exactly as HDDA::update; retain fine DDA times
    # so an active tile can immediately resume voxel traversal.
    exits = ntuple(3) do a
        v = iter.voxel[a]; d = iter.delta_t[a]
        # NanoVDB tile dimensions are powers of two. Align with the HDDA bit
        # mask rather than signed division, whose exception guards are unsafe
        # in device code (see u_div in film.jl).
        aligned = v & -dimension
        boundary = iter.step[a] > 0 ? min(aligned+dimension,iter.grid_res[a]) : aligned
        cells = iter.step[a] > 0 ? boundary-v : v-boundary+Int32(1)
        isfinite(d) ? iter.next_crossing_t[a]+Float32(cells-Int32(1))*d : Inf32
    end
    stop = min(iter.t_max,exits[1],exits[2],exits[3])
    crossed = ntuple(3) do a
        d = iter.delta_t[a]; first = iter.next_crossing_t[a]
        # A finite crossing count is nonnegative and bounded by the grid.
        # Avoid the InexactError path of checked floor(Int32, ...).
        isfinite(d) && stop>=first ? floor_int32((stop-first)/d)+Int32(1) : Int32(0)
    end
    voxel = ntuple(a->iter.voxel[a]+iter.step[a]*crossed[a],3)
    next_t = ntuple(a->isfinite(iter.delta_t[a]) ? iter.next_crossing_t[a]+Float32(crossed[a])*iter.delta_t[a] : Inf32,3)
    mode = all(a->Int32(0)<=voxel[a]<iter.grid_res[a],1:3) && stop<iter.t_max ? Int32(2) : Int32(0)
    next = typeof(iter)(mode,iter.σ_t,stop,iter.t_max,false,iter.grid,iter.grid_res,next_t,
                       iter.delta_t,iter.step,iter.voxel_limit,voxel)
    return (RayMajorantSegment(iter.t_min,stop,iter.σ_t*density),next,true)
end
