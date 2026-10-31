using Test
module SharedBSDFClassTests
    using Hikari, Mantle, Test
    struct UnstoredBSDF <: Hikari.ShaderBSDF end

    class(material, api = Mantle.VulkanAPI()) =
        Hikari.bsdf_kernel_nodes(Hikari.bsdf_nodes(typeof(Hikari.surface_material(material))), api)

    function run()
        @testset "Shared surface capability classes" begin
            @test_throws ArgumentError Hikari.bsdf_kernel_nodes(Tuple{UnstoredBSDF}, Mantle.VulkanAPI())
            principled = class(Hikari.Principled())
            @test class(Hikari.Glass()) === principled
            simple = class(Hikari.Sheen())
            @test simple === class(Hikari.Glossy())
            @test simple === class(Hikari.Diffuse())
            @test simple.parameters[1] == 1
            @test class(Hikari.Add(Hikari.Diffuse(), Hikari.Sheen())) ===
                class(Hikari.Add(Hikari.Sheen(), Hikari.Glossy()))
            @test class(Hikari.Principled(thin_wall = true)) === principled
            @test class(Hikari.Principled(distribution = :ggx)) === principled
            @test class(Hikari.Glass(distribution = :beckmann)) ===
                class(Hikari.Principled(distribution = :beckmann))

            a, b, c = Hikari.Principled(), Hikari.Sheen(), Hikari.Glossy()
            reference = class(Hikari.Add(a, Hikari.Add(b, c)))
            for material in (
                    Hikari.Add(c, Hikari.Add(a, b)),
                    Hikari.Mix(Hikari.Add(b, c), a; factor = 0.3),
                    Hikari.Add(Hikari.Glass(), Hikari.Add(b, c)),
                )
                @test class(material) === reference
            end
            @test reference === principled

            @test class(Hikari.Metallic(fresnel_type = :physical_conductor)) ===
                class(Hikari.Metallic(fresnel_type = :physical_conductor, thin_film_thickness = 400))
            @test class(Hikari.Dielectric()) <: Tuple
            @test class(Hikari.Principled(subsurface_weight = 0.5)) !== principled
            @test class(Hikari.Principled(), Mantle.MetalAPI()) === principled
            @test sizeof(Hikari.SurfaceStoredWeight{false, Hikari.SURFACE_PRINCIPLED_MASK}) == 108

            for n in (1, 4, 8, 16, 32, 33)
                material = foldl(Hikari.Add, fill(Hikari.Diffuse(), n))
                selected = class(material)
                @test selected <: Hikari.SurfaceProgramNodes
                @test selected.parameters[1] >= n
                @test selected.parameters[1] in (1, 4, 8, 16, 32, 64)
            end
        end
        return nothing
    end
end

SharedBSDFClassTests.run()
