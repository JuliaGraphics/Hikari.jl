using Test,Hikari,GeometryBasics,Random,Mantle,LinearAlgebra
import Hikari.Raycore as RC
import Hikari.Adapt as Adapt
import KernelAbstractions as KA
using KernelAbstractions: @kernel,@index

@kernel function packed_bound_values!(out,media,grid,coordinates)
    i=@index(Global)
    q=coordinates[i]
    out[i]=Hikari.majorant_lookup(grid,media,q[1],q[2],q[3])
end

function packed_bound_integral(grid,media,bounds,origin,direction)
    enter,leave=Hikari.ray_bounds_intersect(origin,direction,bounds)
    enter=max(enter,0f0)
    enter>=leave && return 0f0
    iter=Hikari.RayMajorantIterator(Hikari.create_dda_iterator(grid,bounds,origin,direction,
        enter,leave,Hikari.SpectralRadiance(1f0)))
    integral=0f0
    for _ in 1:4096
        segment,iter,valid=Hikari.ray_majorant_next(iter,media)
        valid || break
        integral+=(segment.t_max-segment.t_min)*segment.σ_maj[1]
    end
    return integral
end

@kernel function packed_bound_integrals!(out,media,grid,bounds,origins,directions)
    i=@index(Global)
    out[i]=packed_bound_integral(grid,media,bounds,origins[i],directions[i])
end

@kernel function packed_stored_integrals!(out,media,key,bounds,origins,directions)
    i=@index(Global)
    out[i]=RC.with_index(packed_bound_integral,media,key,media,bounds,origins[i],directions[i])
end

function test_packed_majorants(;backend=nothing)
    @testset "Packed auxiliary bounds preserve exact cells and updates" begin
        rng=MersenneTwister(20261022)
        for dims in ((33,17,9),(32,24,16),(1,1,1))
            res=Vec{3,Int32}(dims)
            data=fill(.3f0,prod(dims))
            for _ in 1:50
                data[rand(rng,eachindex(data))]=rand(rng,Float32)*4f0
            end
            a=Hikari.MajorantGrid(data,res)
            packed=Hikari.packmajorants(a)
            @test collect(packed.voxels)==data
            @test Hikari.packmajorants(packed)===packed
            @test collect(Hikari.packmajorants(Hikari.tiledmajorants(a)).voxels)==data
            changed=reverse(data)
            b=Hikari.packmajorants(Hikari.MajorantGrid(changed,res))
            @test collect(Hikari.maxmajorants(packed,b).voxels)==max.(data,changed)
            if backend!==nothing
                set=RC.MultiTypeSet(backend)
                key=push!(set,packed)
                stored=set[key]
                slots=length(set.texture_gpu_arrays)
                fine_storage=RC.deref(set,stored.voxels.fine)
                tile_storage=RC.deref(set,stored.voxels.tiles)
                coordinates=Vec{3,Int32}[Vec{3,Int32}(x,y,z) for z in 0:dims[3]-1
                    for y in 0:dims[2]-1 for x in 0:dims[1]-1]
                points=KA.allocate(backend,eltype(coordinates),length(coordinates));copyto!(points,coordinates)
                out=KA.zeros(backend,Float32,length(data))
                packed_bound_values!(backend,64)(out,Adapt.adapt(backend,set.static),stored,points;ndrange=length(data))
                @test Array(out)==data
                RC.update!(set,key,b)
                @test length(set.texture_gpu_arrays)==slots
                @test RC.deref(set,set[key].voxels.fine)===fine_storage
                @test RC.deref(set,set[key].voxels.tiles)===tile_storage
                packed_bound_values!(backend,64)(out,Adapt.adapt(backend,set.static),set[key],points;ndrange=length(data))
                @test Array(out)==changed
                bounds=Hikari.Bounds3(Point3f(0),Point3f(1))
                origins=Point3f[Point3f(-1,rand(rng,Float32),rand(rng,Float32)) for _ in 1:128]
                directions=Vec3f[normalize(Vec3f(1,.2f0*(rand(rng,Float32)-.5f0),
                    .2f0*(rand(rng,Float32)-.5f0))) for _ in origins]
                integrals=KA.zeros(backend,Float32,length(origins))
                packed_bound_integrals!(backend,64)(integrals,Adapt.adapt(backend,set.static),set[key],bounds,
                    Mantle.devicearray(backend,origins),Mantle.devicearray(backend,directions);ndrange=length(origins))
                expected=packed_bound_integral.(Ref(b),nothing,Ref(bounds),origins,directions)
                @test Array(integrals)≈expected rtol=1f-5 atol=1f-6
                packed_stored_integrals!(backend,64)(integrals,Adapt.adapt(backend,set.static),key,bounds,
                    Mantle.devicearray(backend,origins),Mantle.devicearray(backend,directions);ndrange=length(origins))
                @test Array(integrals)≈expected rtol=1f-5 atol=1f-6
            end
        end
        res=Vec{3,Int32}(884,884,574)
        huge=Hikari.build_packed_majorants(res,(x,y,z)->error("uniform tile expanded"),(x,y,z)->.3f0)
        v=huge.voxels
        @test length(v.fine)==1
        @test all(iszero,v.offsets)
        @test v[1]==v[length(v)]==.3f0
        @test 4*(length(v.fine)+length(v.tiles)+length(v.offsets))<8_000_000
    end
end
test_packed_majorants()
