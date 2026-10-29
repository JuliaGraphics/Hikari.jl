# A material is a description: Blender's BSDFs and pbrt's models with readable names, any field a
# number, a colour, an image or a node graph, kept as given. `surface_material` decides what renders
# it, and glTF's and OpenPBR's materials are front ends of the Principled BSDF.

using Test, Hikari, Raycore
using Hikari.TextureNodes
using Hikari: TextureFilterContext, RGBSpectrum, ShaderMaterial
using GeometryBasics: Point2f, Point3f, Vec3f
using Colors: RGB, RGBA
using KernelAbstractions: CPU

nodes(::ShaderMaterial{P, A, M}) where {P, A, M} = M
program(m) = Hikari.link(Hikari.compile_shader(Hikari.ShaderGraph(Hikari.shader_node(m)).closures), ())

@testset "Material descriptions" begin
    @testset "a scalar image field stores and updates one texture slot" begin
        store = Raycore.MultiTypeSet(CPU())
        image = fill(0.25, 4, 4)
        mat = Hikari.to_device_material(store, Hikari.Diffuse(roughness = image))
        h = mat.roughness
        @test h.kind == Hikari.TexKind.IMAGE
        @test Hikari.stored_array(store, (Int(h.slot), Int(h.idx))) == fill(0.25f0, 4, 4)
        arrays, layout = length(store.texture_gpu_arrays), store.layout
        next = Raycore.update_item(store, h, fill(0.75, 4, 4))
        @test next === h
        @test (length(store.texture_gpu_arrays), store.layout) == (arrays, layout)
        @test Hikari.stored_array(store, (Int(h.slot), Int(h.idx))) == fill(0.75f0, 4, 4)
        rgb = fill(RGBSpectrum(0.2f0, 0.4f0, 0.6f0), 4, 4)
        colored = Hikari.to_device_material(store, Hikari.Diffuse(color = rgb))
        c = colored.color
        @test Hikari.stored_array(store, (Int(c.slot), Int(c.idx))) == rgb
        fill!(rgb, RGBSpectrum(0.7f0))
        @test Raycore.update_item(store, c, rgb) === c
        @test Hikari.stored_array(store, (Int(c.slot), Int(c.idx))) == rgb
    end

    @testset "a description keeps its fields as given" begin
        d = Hikari.Diffuse(color = (1, 0, 0), roughness = 0.5)
        @test d.color === (1, 0, 0) && d.roughness === 0.5
        p = Hikari.Principled(base_color = noise(xyz()).color, coat_weight = 1)
        @test p.base_color isa Hikari.TexNode && p.coat_weight === 1
        @test Hikari.with_field(d, :roughness, 0.25).roughness === 0.25
        @test Hikari.merge_color_with_material((0, 1, 0), p).base_color === (0, 1, 0)
    end

    @testset "what renders a description" begin
        own = Hikari.Diffuse(color = (0.5, 0.5, 0.5))
        @test Hikari.surface_material(own) === own
        @test nodes(Hikari.surface_material(Hikari.Diffuse(normal = bump(xyz().x)))) == Tuple{Hikari.DiffuseBSDF}
        @test nodes(Hikari.surface_material(Hikari.Glass())) == Tuple{Hikari.GlassBSDF}
        @test nodes(Hikari.surface_material(Hikari.Mix(Hikari.Diffuse(), Hikari.Glossy()))) ==
              Tuple{Hikari.DiffuseBSDF, Hikari.GlossyBSDF}
        @test nodes(Hikari.surface_material(Hikari.Add(Hikari.Diffuse(), Hikari.Emission()))) ==
              Tuple{Hikari.DiffuseBSDF, Hikari.EmissionBSDF}
        # A mix holding one of pbrt's models picks one material per hit, as pbrt's MixMaterial.
        picks = Hikari.surface_material(Hikari.Mix(Hikari.Principled(), Hikari.CoatedDiffuse(); factor = 0.3))
        @test picks isa Hikari.Mix && picks.a isa ShaderMaterial && picks.b isa Hikari.CoatedDiffuse
        # So does a mix of a Diffuse with a pbrt displacement, which no shader node has.
        @test Hikari.surface_material(Hikari.Mix(Hikari.Diffuse(displacement = 0.1), Hikari.Diffuse())) isa Hikari.Mix
        # Alone, an Emission is an area light.
        e = Hikari.Emission(color = (1, 0.5, 0.25), strength = 2)
        @test Hikari.surface_material(e) === e
        info = Hikari.get_emission_info(e)
        @test info.Le == RGBSpectrum(1f0, 0.5f0, 0.25f0) && info.scale == 2f0 / Hikari.D65_PHOTOMETRIC
        @test_throws ArgumentError Hikari.surface_material(Hikari.Add(Hikari.Diffuse(), Hikari.Gold()))
    end

    @testset "a Principled with constant inputs is its constants' program" begin
        @test program(Hikari.Principled(base_color = (0.2, 0.3, 0.4), roughness = 0.25)) ==
              program(Hikari.Principled(base_color = RGBSpectrum(0.2f0, 0.3f0, 0.4f0), roughness = 0.25f0))
    end

    @testset "glTF's material is the Principled BSDF Blender's importer makes of it" begin
        g = Hikari.GLTF(base_color_factor = (0.8, 0.2, 0.1, 1), metallic_factor = 0.3, roughness_factor = 0.6,
                        ior = 1.45, clearcoat_factor = 1, clearcoat_roughness_factor = 0.1, specular_factor = 0.8)
        p = Hikari.Principled(g)
        @test (p.base_color, p.metallic, p.roughness, p.ior) == ((0.8, 0.2, 0.1), 0.3, 0.6, 1.45)
        @test (p.coat_weight, p.coat_roughness, p.specular_ior_level) == (1, 0.1, 0.4)
        @test p.alpha == 1 && p.emission_strength == 0 && p.sheen_weight == 0
        # No emission and no sheen leave their code out of the material.
        @test nodes(Hikari.surface_material(g)) == Tuple{Hikari.PrincipledBSDF{false, false, false}}
        @test nodes(Hikari.surface_material(Hikari.GLTF(emissive_factor = (1, 1, 1)))) ==
              Tuple{Hikari.PrincipledBSDF{false, false, true}}
        # Textures multiply their factors, metallic from blue and roughness from green.
        t = Hikari.Principled(Hikari.GLTF(metallic_roughness_texture = fill(RGB{Float32}(0, 0.5, 1), 2, 2)))
        @test t.metallic isa Hikari.TexNode && t.roughness isa Hikari.TexNode
        @test_throws ArgumentError Hikari.Principled(Hikari.GLTF(alpha_mode = :dither))
    end

    @testset "a glTF mask cuts out where its texture's alpha is below the cutoff" begin
        img = [RGBA{Float32}(0.8, 0.2, 0.1, j <= 2 ? 0 : 1) for i in 1:4, j in 1:4]
        @test Hikari.opaque(Hikari.GLTF(base_color_texture = img))
        cut = Hikari.GLTF(base_color_texture = img, alpha_mode = :mask)
        @test !Hikari.opaque(cut)
        store = Raycore.MultiTypeSet(CPU())
        mat = Hikari.to_device_material(store, Hikari.surface_material(cut))
        textures = Raycore.StaticMultiTypeSet((), Tuple(T[a for a in store.texture_gpu_arrays if typeof(a) === T]
                                                        for T in store.texture_order))
        through(u) = Hikari.surface_transparency(mat, textures, TextureFilterContext(Point2f(u, 0.5f0), Point3f(0), Vec3f(0, 0, 1)))
        @test through(0.1f0) == Vec3f(1)
        @test through(0.9f0) == Vec3f(0)
    end

    @testset "OpenPBR's surface is the Principled BSDF it corresponds to" begin
        p = Hikari.Principled(Hikari.OpenPBR(base_weight = 0.5, base_color = (0.8, 0.4, 0.2), specular_roughness = 0.2,
                                             specular_weight = 0.6, fuzz_weight = 0.5, fuzz_color = (1, 0, 0),
                                             thin_film_weight = 1, thin_film_thickness = 0.3, geometry_thin_walled = true))
        @test p.base_color == (0.4, 0.2, 0.1)
        @test (p.roughness, p.specular_ior_level) == (0.2, 0.3)
        @test (p.sheen_weight, p.sheen_tint) == (0.5, (1, 0, 0))
        @test p.thin_film_thickness ≈ 300 && p.thin_wall
        @test Hikari.Principled(Hikari.OpenPBR(transmission_weight = 1, transmission_color = (0.9, 1, 0.9))).base_color ==
              (0.9, 1, 0.9)
    end
end
