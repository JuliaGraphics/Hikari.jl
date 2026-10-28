# A glass whose index of refraction is a spectrum disperses light as pbrt-v4's does.
#
# pbrt ends a path's secondary wavelengths where it meets such a BSDF, before the lights are
# sampled there (`GetBxDF`), and its film divides the whole sample by the path's last wavelength
# pdfs. Hikari ended them only on refraction, and its film divided by the pdfs the camera ray
# started with, so every wavelength still followed the first one's path, the glass's colour
# fringes smeared, and under the suite's `maxcomponentvalue` the picture was 8 % brighter than
# pbrt's, whose four-times-weighted samples meet the clamp. The importer read a named spectrum as
# its default, 1.5, and an inline one as its first wavelength. Termination also divided again
# where a path refracted twice, as pbrt's does not.

using Test, Hikari, Mantle, Colors
isdefined(@__MODULE__, :render_scene) || include(joinpath(@__DIR__, "pbrt", "suite.jl"))

@testset "dispersion" begin
    @testset "the secondary wavelengths end once" begin
        lambda = Hikari.terminate_secondary_wavelengths(Hikari.sample_wavelengths_visible(0.3f0))
        @test Hikari.terminate_secondary_wavelengths(lambda).pdf == lambda.pdf
    end

    @testset "a glass of pbrt's names is its spectrum" begin
        glass(eta) = let pb = Hikari.parse_pbrt_string("""
                Material "dielectric" $eta
                Shape "trianglemesh" "point3 P" [ 0 0 0  1 0 0  0 1 0 ] "integer indices" [ 0 1 2 ]""")
            Hikari.build_pbrt_material(only(pb.shapes).material_inline, pb)
        end
        @test glass("\"spectrum eta\" \"glass-BK7\"").ior === Hikari.GLASS_BK7_ETA_SPECTRUM
        @test Hikari.sample(Hikari.GLASS_BK7_ETA_SPECTRUM, 587.6f0) ≈ 1.5168f0 atol = 1f-4
        inline = glass("\"spectrum eta\" [ 400 1.6  700 1.5 ]").ior
        @test Hikari.sample(inline, 550f0) ≈ 1.55f0
        @test_throws ArgumentError glass("\"spectrum eta\" \"glass-unknown\"")
    end

    # A coat's IOR may be a spectrum too, as pbrt's coated materials take one: the importer
    # refused it, Hikari's coats holding one number, and before that read it as 1.5 or as its
    # first wavelength. Taken at the path's first wavelength, it ends the others at the vertex.
    @testset "a coat of pbrt's names is its spectrum, and ends the other wavelengths" begin
        coat(material) = let pb = Hikari.parse_pbrt_string("""
                $material
                Shape "trianglemesh" "point3 P" [ 0 0 0  1 0 0  0 1 0 ] "integer indices" [ 0 1 2 ]""")
            Hikari.build_pbrt_material(only(pb.shapes).material_inline, pb)
        end
        d = coat("""Material "coateddiffuse" "spectrum eta" "glass-F11" """)
        c = coat("""Material "coatedconductor" "spectrum interface.eta" [ 400 1.6  700 1.5 ]""")
        @test d.ior === Hikari.GLASS_SF11_ETA_SPECTRUM
        @test Hikari.sample(c.interface_ior, 550f0) ≈ 1.55f0
        @test coat("""Material "coateddiffuse" "float eta" 1.33""").ior == 1.33f0

        table = Hikari.get_srgb_table()
        lambda = Hikari.sample_wavelengths_visible(0.3f0)
        tfc = Hikari.TextureFilterContext(Hikari.Point2f(0.5f0, 0.5f0))
        # The material as the device holds it, and its BSDF.
        function bsdf(m)
            scene = Hikari.Scene(; hw_accel = false)
            key = scene.media_interfaces[push!(scene, m)].material
            static = scene.materials.static
            return Hikari.get_bxdf(static.data[key.type_idx][key.vec_idx], table, static, tfc, lambda, false,
                                   Hikari.Vec3f(0, 0, 1), Hikari.Vec3f(0, 0, 1), Hikari.Vec3f(1, 0, 0))
        end
        @test Hikari.terminates_at_vertex(bsdf(d))
        @test Hikari.terminates_at_vertex(bsdf(c))
        @test !Hikari.terminates_at_vertex(bsdf(Hikari.CoatedDiffuse(ior = 1.5)))
    end

    @testset "renders as pbrt's" begin
        name = "mat_dielectric_bk7_light_point"
        ref, ours = ensure_reference(name), render_scene(name; samples = REFERENCE_SPP)
        for channel in (red, green, blue)
            @test sum(c -> Float64(channel(c)), ours) / sum(c -> Float64(channel(c)), ref) ≈ 1 rtol = 0.015
        end
    end
end
