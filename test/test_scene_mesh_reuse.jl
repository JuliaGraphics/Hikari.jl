using Test, Hikari, GeometryBasics
import Lava
import KernelAbstractions as KA

function test_scene_mesh_texture_reuse(backend; hw_accel = false)
    scene = Hikari.Scene(; backend, hw_accel)
    mesh = normal_mesh(GeometryBasics.Tessellation(Sphere(Point3f(0), 0.5f0), 12))
    material(v) = Hikari.Diffuse(; Kd = Hikari.Texture(fill(Hikari.RGBSpectrum(v), 4, 4)))
    handle = push!(scene, mesh, material(0.2f0))
    interface = handle.interface
    slots = length(scene.materials.texture_gpu_arrays)
    @test slots > 0
    for i in 1:12
        delete!(scene.accel, handle.geometry)
        mat = material(0.2f0 + i*0.01f0)
        Hikari.update_material!(scene, interface, mat)
        handle = push!(scene, mesh, interface, mat)
        @test handle.interface == interface
        @test length(scene.materials.texture_gpu_arrays) == slots
    end
    Hikari.sync!(scene)
end

@testset "Geometry replacement reuses material textures" begin
    test_scene_mesh_texture_reuse(KA.CPU())
    backend = Lava.LavaBackend()
    if Lava.vk_context().rt_pipeline_properties === nothing
        @test_skip false
    else
        test_scene_mesh_texture_reuse(backend; hw_accel = true)
    end
end
