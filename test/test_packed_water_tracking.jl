using Test,Hikari,Mantle,GeometryBasics,LinearAlgebra
import KernelAbstractions as KA
import Hikari.Raycore as RC
import Hikari.Adapt as Adapt
using KernelAbstractions: @kernel,@index

@inline function packed_water_probe(m,media,origin,direction)
    enter,leave=Hikari.ray_bounds_intersect(origin,direction,m.bounds)
    iter=Hikari.RayMajorantIterator(Hikari.create_dda_iterator(m.majorant_grid,m.bounds,
        origin,direction,max(enter,0f0),leave,Hikari.SpectralRadiance(1f0)))
    zero=Hikari.SpectralRadiance(0f0)
    control=Hikari.MediumProperties(zero,zero,zero,0f0)
    transmission=Hikari.water_ratio_tracking(iter,origin,direction,media,m,control)[1][1]
    return Vec3f(transmission,Hikari.airscattering(media,m.air,Point3f(0)),RC.deref(media,m.spectra)[1])
end

@kernel function packed_water_probes!(out,media,key,origins,directions)
    i=@index(Global)
    out[i]=RC.with_index(packed_water_probe,media,key,media,origins[i],directions[i])
end

function test_packed_water_tracking(;backend=nothing,packed_layout=true)
    origins=Vec{3,Int32}[Vec{3,Int32}(8x,8y,8z) for x in 0:3 for y in 0:3 for z in 0:3]
    lo=Point3f(-1.6)
    function field(centre,scale)
        values=Float32[scale*exp(-40f0*(norm(lo+(Vec3f(o+Vec{3,Int32}(x,y,z)).+.5f0)*.1f0-centre)-.6f0)^2)
            for o in origins for x in 0:7 for y in 0:7 for z in 0:7]
        Hikari.NanoTree(origins,values,lo,.1f0;backend=KA.CPU())
    end
    a,b=field(Point3f(0),.005f0),field(Point3f(.2,0,0),.007f0)
    c,d=field(Point3f(0,0,-.1),.001f0),field(Point3f(.2,0,-.1),.002f0)
    bounds=Hikari.Bounds3(Point3f(-2),Point3f(2))
    rays=Point3f[Point3f(-3,Float32(y),Float32(z)) for y in -.8:.2:.8 for z in -.8:.2:.8]
    directions=fill(Vec3f(1,0,0),length(rays))
    @testset "Packed water bounds in stochastic tracking" begin
        for weight in (0f0,.25f0,1f0)
            blend=Hikari.NanoBlend(a,b,weight)
            other=Hikari.NanoBlend(c,d,weight)
            media=(Hikari.WaterMedium(Hikari.Seawater();air=((other,.001f0,.85f0),(blend,.001f0,.85f0)),
                    bounds,cell=.4f0,majorants=:tiled,backend=KA.CPU()),
                Hikari.WaterMedium(nothing;air=((blend,.001f0,.85f0),(other,.0002f0,.87f0)),
                    bounds,cell=.4f0,ior=1f0,majorants=:tiled,backend=KA.CPU()),
                Hikari.WaterMedium(Hikari.Seawater();bounds,cell=1000f0,backend=KA.CPU()))
            packed=map(media) do m
                Hikari.WaterMedium(m.spectra,m.from,m.step,m.n,m.g,m.base,m.air,m.ior,m.bounds,
                    Hikari.packmajorants(m.majorant_grid))
            end
            packed_layout || (packed=map(media) do m
                Hikari.WaterMedium(m.spectra,m.from,m.step,m.n,m.g,m.base,m.air,m.ior,m.bounds,
                    Hikari.tiledmajorants(m.majorant_grid))
            end)
            set=backend===nothing ? nothing : RC.MultiTypeSet(backend)
            keys=backend===nothing ? nothing : map(m->push!(set,m),packed)
            for (i,m) in enumerate(packed)
                expected=packed_water_probe.(Ref(m),nothing,rays,directions)
                @test all(p->all(isfinite,p),expected)
                if backend!==nothing
                    out=KA.zeros(backend,Vec3f,length(rays))
                    device_rays=Mantle.devicearray(backend,rays)
                    device_directions=Mantle.devicearray(backend,directions)
                    actual=GC.@preserve set out device_rays device_directions begin
                        packed_water_probes!(backend,64)(out,Adapt.adapt(backend,set.static),keys[i],
                            device_rays,device_directions;ndrange=length(rays))
                        KA.synchronize(backend)
                        Array(out)
                    end
                    @test actual≈expected rtol=1f-5 atol=1f-6
                end
            end
        end
    end
    return nothing
end

test_packed_water_tracking()
