using Test, Hikari, GeometryBasics, Random, Statistics

function test_water_shadow_scalar(;n=20000)
    H=Hikari
    V=Vec{3,Int32}
    tree=H.NanoTree(UInt32[],Int32(0),Int32(0),Int32(0),Int32(0),Int32(0),Int32(0),Int32(0),
        Point3f(-10),1f0,1f0,V(-100),V(100))
    bounds=H.Bounds3(Point3f(0),Point3f(1))
    mc=H.MediumProperties(H.SpectralRadiance(.1f0,.2f0,.4f0,.8f0),H.SpectralRadiance(0f0),H.SpectralRadiance(0f0),0f0)
    rng=MersenneTwister(21437)
    origins=[Point3f(0,rand(rng,Float32),rand(rng,Float32)) for _ in 1:n]
    direction=Vec3f(1,0,0)
    lambda=H.sample_wavelengths_visible(.5f0)
    table=H.get_rgb2spec_table()
    @testset "Scalar water shadows retain spectral control and residual expectation" begin
        for residual in (0f0,.01f0,.7f0,3f0,8f0),cells in (1,17)
            base=1f0
            grid=H.MajorantGrid(fill(base+residual*1.7f0,cells),V(cells,1,1))
            medium=H.WaterMedium(Float32[],360f0,5f0,Int32(2),0f0,base,
                ((tree,residual),),1f0,bounds,grid)
            media=H.Raycore.StaticMultiTypeSet(([medium],),())
            key=H.SetKey(UInt32(1),UInt32(1))
            results=zeros(Float64,n,4)
            difference=0.0
            for (i,origin) in enumerate(origins)
                iter=H.RayMajorantIterator(H.create_dda_iterator(grid,bounds,origin,direction,0f0,1f0,H.SpectralRadiance(1f0)))
                actual=H.water_ratio_tracking(iter,origin,direction,nothing,medium,mc)
                general=H.ratio_tracking_dda(iter,origin,direction,media,key,table,lambda,mc,base)
                @test all(isfinite,actual[1].data)
                @test all(isfinite,actual[2].data)
                @test all(==(1f0),actual[3].data)
                results[i,:].=actual[1].data
                difference=max(difference,maximum(abs.(actual[1].data.-general[1].data)))
            end
            @test difference < 2e-4
            for channel in 1:4
                expected=exp(-Float64(mc.σ_a[channel])-Float64(residual))
                estimate=mean(view(results,:,channel))
                error=std(view(results,:,channel))/sqrt(n)
                @test abs(estimate-expected) < 6error+2e-5
            end
        end
        empty_grid=H.MajorantGrid(Float32[0],V(1))
        medium=H.WaterMedium(Float32[],360f0,5f0,Int32(2),0f0,0f0,(),1f0,bounds,empty_grid)
        result=H.water_ratio_tracking(H.RayMajorantIterator(empty_grid),Point3f(0),direction,nothing,medium,mc)
        @test all(t->all(==(1f0),t.data),result)
    end
end

test_water_shadow_scalar()
