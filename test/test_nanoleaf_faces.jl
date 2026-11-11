using Test, Hikari, GeometryBasics, Random, Mantle
import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index

# Independent eight-corner lookup, including missing leaves and the grid halo.
function nanoleaf_reference(t,p)
    q=Hikari.indexspace(t,p);i=floor.(Int32,q);f=q-Vec3f(i)
    if any(i .+ Int32(1) .< t.imin) || any(i .>= t.imax)
        return t.background
    end
    at(x,y,z)=Hikari.nanovalue(t,t.words,i+Vec{3,Int32}(x,y,z))
    x00=at(0,0,0)*(1f0-f[1])+at(1,0,0)*f[1]
    x10=at(0,1,0)*(1f0-f[1])+at(1,1,0)*f[1]
    x01=at(0,0,1)*(1f0-f[1])+at(1,0,1)*f[1]
    x11=at(0,1,1)*(1f0-f[1])+at(1,1,1)*f[1]
    return (x00*(1f0-f[2])+x10*f[2])*(1f0-f[3])+(x01*(1f0-f[2])+x11*f[2])*f[3]
end

@kernel function nanoleaf_face_samples!(out,tree,points)
    i=@index(Global)
    out[i]=Hikari.nanosample(tree,tree.words,points[i])
end

function test_nanoleaf_faces(;backend=nothing)
    @testset "NanoVDB leaf-face interpolation" begin
        rng=MersenneTwister(20261021)
        origins=Vec{3,Int32}[Vec{3,Int32}(8x,8y,8z) for x in -1:1 for y in -1:1 for z in -1:1
            if (x,y,z)!=(0,0,0) && (x,y,z)!=(1,-1,0)]
        values=rand(rng,Float32,512length(origins)).-.25f0
        host=Hikari.NanoTree(origins,values,Point3f(-.23,.17,-.51),.1f0;background=.125f0,backend=KA.CPU())
        points=Point3f[]
        # All leaf-boundary combinations, including negative coordinates, corners,
        # absent leaves, and the interpolation halo at the grid's outer edges.
        for x in -9:16,y in -9:16,z in -9:16
            push!(points,Hikari.worldspace(host,Vec3f(x+.37f0,y+.61f0,z+.23f0)))
        end
        append!(points,[Point3f(-1.5f0 .+ 4f0.*rand(rng,Float32,3)) for _ in 1:2000])
        expected=nanoleaf_reference.(Ref(host),points)
        actual=Hikari.nanosample.(Ref(host),Ref(host.words),points)
        for (a,b) in zip(actual,expected)
            @test isequal(a,b)
        end
        if backend!==nothing
            device=Hikari.NanoTree(Mantle.devicearray(backend,origins),Mantle.devicearray(backend,values),
                host.lo,host.voxel;background=host.background,backend)
            output=KA.allocate(backend,Float32,length(points))
            nanoleaf_face_samples!(backend,256)(output,device,Mantle.devicearray(backend,points);ndrange=length(points))
            @test maximum(abs.(Array(output).-expected))<=2f-6
        end
    end
end

# The ordinary suite runs the exhaustive host reference; the dedicated device
# check additionally calls test_nanoleaf_faces(; backend=...) on its GPU queue.
test_nanoleaf_faces()
