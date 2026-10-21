using Test,Hikari,GeometryBasics,Random,LinearAlgebra

function test_tiled_majorants()
    H=Hikari
    rng=MersenneTwister(210215)
    function intervals(grid,box,o,d)
        lo,hi=H.ray_bounds_intersect(o,d,box);lo=max(lo,0f0)
        lo>=hi && return (integral=0.0,distance=0.0,count=0)
        it=H.RayMajorantIterator(H.create_dda_iterator(grid,box,o,d,lo,hi,H.SpectralRadiance(1f0)))
        integral=0.0;distance=0.0;count=0;last=lo
        for _ in 1:H.majorant_segment_limit(it)
            seg,it,valid=H.ray_majorant_next(it,nothing)
            valid || break
            @test seg.t_max>=seg.t_min
            @test seg.t_min≈last atol=2f-4
            last=seg.t_max
            integral+=(seg.t_max-seg.t_min)*seg.σ_maj[1]
            distance+=seg.t_max-seg.t_min
            count+=1
        end
        @test last≈hi rtol=2f-4 atol=2f-4
        (;integral,distance,count)
    end
    @testset "Uniform auxiliary tiles retain fine-grid integrals" begin
        for res in (Vec{3,Int32}(33,17,9),Vec{3,Int32}(32,24,16)),scale in (.00625f0,.05f0,1f0)
            grid=H.MajorantGrid(fill(.3f0,prod(res)),res)
            for z in 3:7,y in 2:10,x in 4:12
                grid.voxels[x+res[1]*(y+res[2]*z)+1]=Float32(1+(x+y+z)%5)
            end
            tiled=H.tiledmajorants(grid)
            @test collect(tiled.voxels)==grid.voxels
            lo=Point3f(-18,-18,-12.2)
            box=H.Bounds3(lo,lo+Vec3f(res)*scale)
            rays=[(Point3f(lo+Vec3f(rand(rng,Float32,3).*Vec3f(res))*scale),normalize(Vec3f(randn(rng,Float32,3)))) for _ in 1:1000]
            append!(rays,[(Point3f(lo+Vec3f(res)*scale*.5f0),Vec3f(d)) for d in ((1,0,0),(-1,0,0),(0,1,0),(0,0,-1),(1,1,1),(-1,-1,-1))])
            for (o,d) in rays
                a=intervals(grid,box,o,d);b=intervals(tiled,box,o,d)
                @test a.integral≈b.integral rtol=8f-4 atol=2f-5
                @test a.distance≈b.distance rtol=2f-4 atol=2f-5
                @test b.count<=a.count
            end
        end
    end
end

test_tiled_majorants()
