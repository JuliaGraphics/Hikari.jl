using Test, Hikari, GeometryBasics, Mantle, Random, LinearAlgebra
import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index

@testset "Supported water majorant layout" begin
    bounds = Hikari.Bounds3(Point3f(0),Point3f(1))
    @test Hikari.validate_water_majorants(:dense) === nothing
    @test_throws ArgumentError Hikari.WaterMedium(nothing;bounds,majorants=:native,backend=nothing)
    @test_throws ArgumentError Hikari.WaterMedium(nothing;bounds,majorants=:unknown,backend=nothing)
    medium = Hikari.WaterMedium(nothing;bounds,cell=0.5f0,backend=nothing)
    @test medium.majorant_grid.voxels isa Vector{Float32}
    @test all(iszero,medium.majorant_grid.voxels)
end

@kernel function majorant_gpu_integrals!(result, counts, tree, res, bounds, origins, directions)
    i = @index(Global)
    grid = Hikari.MajorantGrid(Hikari.NativeMajorantValues(tree,res,3f0),res)
    o=origins[i]; d=directions[i]
    enter,leave=Hikari.ray_bounds_intersect(o,d,bounds)
    enter=max(0f0,enter)
    total=0f0; n=Int32(0)
    if enter<leave
        it=Hikari.RayMajorantIterator(Hikari.create_dda_iterator(grid,bounds,o,d,enter,leave,Hikari.SpectralRadiance(1f0)))
        for step in Int32(1):Int32(1000)
            seg,it,valid=Hikari.ray_majorant_next(it,nothing)
            valid || break
            total+=(seg.t_max-seg.t_min)*seg.σ_maj[1]
            n+=Int32(1)
        end
    end
    result[i]=total
    counts[i]=n
end

@testset "Native NanoVDB majorant HDDA" begin
    backend = Mantle.defaultbackend()
    res = Vec{3,Int32}(280,64,64)
    grid = Hikari.MajorantGrid(res)
    fill!(grid.voxels,0.3f0)
    for z in 2:4, y in 3:6, x in (4:9...,257:264...)
        grid.voxels[x+res[1]*(y+res[2]*z)+1] = Float32(1+x%3)
    end
    bounds = Hikari.Bounds3(Point3f(0),Point3f(res))
    sparse = Hikari.sparsemajorants(grid,bounds,0.3f0;backend)
    device = sparse.voxels.tree
    host = Hikari.NanoTree(Array(device.words),device.upper,device.lower,device.leaf,device.uppers,
        device.lowers,device.leaves,device.roottiles,device.lo,device.voxel,device.background,device.imin,device.imax)
    native = Hikari.MajorantGrid(Hikari.NativeMajorantValues(host,res,sparse.voxels.top),res)
    @test length(host.words)*4 < length(grid.voxels)*4
    @test Hikari.majorant_tile(host,host.words,Vec{3,Int32}(150,1,1))[1] == 128
    @test Hikari.majorant_tile(host,host.words,Vec{3,Int32}(40,1,1))[1] >= 8
    function integrate(g,o,d,box=bounds)
        enter,leave = Hikari.ray_bounds_intersect(o,d,box)
        enter=max(0f0,enter)
        enter>=leave && return (0f0,0,0f0)
        it=Hikari.RayMajorantIterator(Hikari.create_dda_iterator(g,box,o,d,enter,leave,Hikari.SpectralRadiance(1f0)))
        total=0f0; n=0; finish=enter
        while true
            seg,it,valid=Hikari.ray_majorant_next(it,nothing)
            valid || break
            @test seg.t_min ≈ finish atol=1f-4
            total+=(seg.t_max-seg.t_min)*seg.σ_maj[1]
            finish=seg.t_max; n+=1
            n<=1000 || error("HDDA failed to advance")
        end
        return (total,n,finish-enter)
    end
    rays=[(Point3f(-1,3.5,3.5),Vec3f(1,0,0)),(Point3f(281,3.5,3.5),Vec3f(-1,0,0)),
          (Point3f(140,3.5,-1),Vec3f(0,0,1))]
    rng=MersenneTwister(8442)
    for i in 1:100
        p=Point3f(rand(rng,Float32,3).*Vec3f(res))
        d=normalize(Vec3f(randn(rng,Float32,3)))
        push!(rays,(p,d))
    end
    for (o,d) in rays
        a=integrate(grid,o,d); b=integrate(native,o,d)
        @test a[1] ≈ b[1] rtol=2f-4 atol=1f-4
        @test a[3] ≈ b[3] rtol=1f-5 atol=1f-4
        @test b[2]<=a[2]
    end
    output=KA.allocate(backend,Float32,length(rays))
    counts=KA.allocate(backend,Int32,length(rays))
    majorant_gpu_integrals!(backend,64)(output,counts,device,res,bounds,
        Mantle.devicearray(backend,first.(rays)),Mantle.devicearray(backend,last.(rays));ndrange=length(rays))
    gpu=Array(output); gpu_counts=Array(counts)
    for (i,(o,d)) in enumerate(rays)
        expected=integrate(grid,o,d)
        @test gpu[i] ≈ expected[1] rtol=2f-4 atol=1f-4
        @test gpu_counts[i]<1000
    end

    # Physical liquid grids use centimetre voxels and negative world origins.
    # Preserve optical depth under scale/translation, through both active
    # leaves and empty 8/128-voxel tiles, in both ray directions.
    @testset "Translated physical voxel scales" for scale in (0.0125f0,0.05f0,2f0)
        lo=Point3f(-18,-18,-12.2)
        box=Hikari.Bounds3(lo,lo+Vec3f(res)*scale)
        scaled_rays=[(Point3f(lo+Vec3f(o)*scale),d) for (o,d) in rays]
        majorant_gpu_integrals!(backend,64)(output,counts,device,res,box,
            Mantle.devicearray(backend,first.(scaled_rays)),
            Mantle.devicearray(backend,last.(scaled_rays));ndrange=length(rays))
        scaled_gpu=Array(output);scaled_counts=Array(counts)
        for (i,(o,d)) in enumerate(scaled_rays)
            dense=integrate(grid,o,d,box)
            sparse=integrate(native,o,d,box)
            original=integrate(grid,rays[i]...)
            @test dense[1] ≈ scale*original[1] rtol=5f-4 atol=1f-4
            @test sparse[1] ≈ dense[1] rtol=5f-4 atol=1f-4
            @test scaled_gpu[i] ≈ dense[1] rtol=5f-4 atol=1f-4
            @test scaled_counts[i]<1000
        end
    end
end
