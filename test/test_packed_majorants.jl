using Test,Hikari,GeometryBasics,Random
import Hikari.Raycore as RC
import Hikari.Adapt as Adapt
import KernelAbstractions as KA
using KernelAbstractions: @kernel,@index

@kernel function packed_bound_values!(out,media,grid,coordinates)
    i=@index(Global)
    q=coordinates[i]
    out[i]=Hikari.majorant_lookup(grid,media,q[1],q[2],q[3])
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
                coordinates=Vec{3,Int32}[Vec{3,Int32}(x,y,z) for z in 0:dims[3]-1
                    for y in 0:dims[2]-1 for x in 0:dims[1]-1]
                points=KA.allocate(backend,eltype(coordinates),length(coordinates));copyto!(points,coordinates)
                out=KA.zeros(backend,Float32,length(data))
                packed_bound_values!(backend,64)(out,Adapt.adapt(backend,set.static),stored,points;ndrange=length(data))
                @test Array(out)==data
                RC.update!(set,key,b)
                @test length(set.texture_gpu_arrays)==slots
                packed_bound_values!(backend,64)(out,Adapt.adapt(backend,set.static),set[key],points;ndrange=length(data))
                @test Array(out)==changed
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
