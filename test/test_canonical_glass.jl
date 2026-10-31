module CanonicalGlassTests
using Test, Hikari
using Hikari.TextureNodes

program(m) = Hikari.link(Hikari.compile_shader(Hikari.ShaderGraph(Hikari.shader_node(m)).closures), ())

@testset "Glass is a canonical Principled configuration" begin
    for distribution in (:multi_ggx, :ggx, :beckmann), thin_wall in (false, true),
            roughness in (0, 0.45), film in (0, 400)
        color = (0.25, 0.6, 0.9)
        glass = Hikari.Glass(;
            base_color = color, roughness, ior = 1.6,
            thin_film_thickness = film, distribution, thin_wall
        )
        explicit = Hikari.Principled(;
            base_color = color, roughness, ior = 1.6,
            transmission_weight = 1, thin_film_thickness = film,
            distribution, thin_wall
        )
        @test glass isa Hikari.Principled
        @test typeof(glass) === typeof(explicit)
        @test all(f -> getfield(glass, f) == getfield(explicit, f), fieldnames(typeof(glass)))
        @test program(glass) == program(explicit)
        @test typeof(Hikari.surface_material(glass)) === typeof(Hikari.surface_material(explicit))
        changed = Hikari.with_field(glass, :roughness, 0.2)
        @test changed isa Hikari.Principled && changed.transmission_weight == 1
        @test Hikari.merge_color_with_material((1, 0, 0), glass).base_color == (1, 0, 0)
    end
    color = noise(xyz()).color
    glass = Hikari.Glass(;
        color, normal = bump(noise(xyz()).fac),
        transmission_dispersion_scale = 0.5
    )
    @test glass.base_color === color
    @test glass.normal isa Hikari.TexNode
    @test glass.transmission_dispersion_scale == 0.5
    @test !isdefined(Hikari, :GlassBSDF)
    @test !isdefined(Hikari, :GlassFilmBSDF)
    @test_throws ArgumentError Hikari.shader_node(Hikari.Glass(distribution = :invalid))
end
end
