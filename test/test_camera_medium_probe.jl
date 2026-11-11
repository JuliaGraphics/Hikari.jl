# Finding the medium the camera is in keeps no device memory.
#
# A render probes it with one ray, once per camera position. The probe wrote its answer into an
# array of its own, a device allocation nothing released: on Vulkan, whose arrays have no
# finalizer, every new position left one behind, a camera moving through an interactive view one
# per frame, towards the driver's limit on allocations. It also ran in every scene with geometry,
# media or not. Its scratch now comes from the memory it is handed, and goes back to it.

using Test, Hikari, Mantle, Raycore, GeometryBasics, Adapt
using GeometryBasics: normal_mesh, Tesselation, Rect3f, Sphere

@testset "probing the camera's medium keeps no device memory" begin
    backend = Mantle.defaultbackend()
    scene = Hikari.Scene(; backend)
    fog = Hikari.HomogeneousMedium(σ_a = Hikari.RGBSpectrum(0.01f0), σ_s = Hikari.RGBSpectrum(0.2f0),
                                   Le = Hikari.RGBSpectrum(0f0), g = 0f0)
    push!(scene, normal_mesh(Rect3f(Vec3f(-5), Vec3f(10))), Hikari.MediumInterface(Hikari.NullMaterial(); inside = fog))
    push!(scene, normal_mesh(Tesselation(Sphere(Point3f(0), 0.5f0), 16)), Hikari.Diffuse())
    Hikari.sync!(scene)
    adapted = Adapt.adapt(backend, scene)
    mem = Hikari.DeviceMemory(backend)
    probe(z) = Hikari.detect_camera_medium(mem, adapted.accel, adapted.media_interfaces, Point3f(0, -3, z))

    inside = probe(1f0)
    @test inside != Raycore.SetKey()               # inside the fog
    @test Hikari.nallocations(mem) == 0            # and nothing kept
    @test all(z -> probe(z) == inside, range(0.5f0, 2f0; length = 16))
    @test Hikari.nallocations(mem) == 0
end
