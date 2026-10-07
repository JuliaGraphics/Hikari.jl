using Test, Hikari, GeometryBasics
import KernelAbstractions as KA
import Raycore

@testset "retained mesh emission follows its material" begin
    scene = Hikari.Scene(; backend = KA.CPU())
    mesh = GeometryBasics.Mesh(Point3f[(0,0,0),(1,0,0),(1,1,0),(0,1,0)],
        GLTriangleFace[(1,2,3),(1,3,4)])
    emission(v; scale=1f0) = Hikari.MediumInterface(Hikari.NullMaterial();
        emission=Hikari.Emissive(Hikari.Texture(fill(Hikari.RGBSpectrum(v), 4, 4)), scale, true))
    handle = push!(scene, mesh, emission(0f0))
    keys = copy(handle.area_lights)
    @test length(keys) == 2
    geometry = handle.geometry
    slots = length(scene.lights.texture_gpu_arrays)
    @test slots == 1
    for v in (1f0, .2f0, .9f0)
        Hikari.update_material!(scene, handle, emission(v; scale=2f0))
        @test handle.geometry === geometry && handle.area_lights == keys
        @test length(scene.lights.texture_gpu_arrays) == slots
        @test length(scene.lights) == 2
        lights = [scene.lights[key] for key in keys]
        @test all(l -> l.scale == 2f0 && l.two_sided, lights)
        @test lights[1].Le === lights[2].Le
        @test all(c -> c == Hikari.RGBSpectrum(v), only(scene.lights.texture_gpu_arrays))
    end
    Hikari.update_material!(scene, handle, Hikari.NullMaterial())
    @test all(key -> scene.lights[key].scale == 0f0, keys)
    Hikari.update_material!(scene, handle, emission(.8f0))
    @test all(key -> scene.lights[key].scale == 1f0, keys)
    for i in 1:3
        delete!(scene.accel, handle.geometry)
        moved = GeometryBasics.Mesh(Point3f[(i,0,0),(i+1,0,0),(i+1,1,0),(i,1,0)],
            GLTriangleFace[(1,2,3),(1,3,4)])
        handle = push!(scene, moved, handle.interface, emission(.3f0*i); area_lights=handle.area_lights)
        @test handle.area_lights == keys
        @test length(scene.lights) == 2
        @test length(scene.lights.texture_gpu_arrays) == slots
        @test scene.lights[first(keys)].vertices[1] == Point3f(i,0,0)
    end
    constant(v) = Hikari.MediumInterface(Hikari.NullMaterial();
        emission=Hikari.Emissive(; Le=(v,v,v), scale=1f0, two_sided=true))
    solid = push!(scene, mesh, constant(1f0))
    Hikari.update_material!(scene, solid, constant(2f0))
    @test all(key -> scene.lights[key].Le == Hikari.RGBSpectrum(2f0), solid.area_lights)
end
