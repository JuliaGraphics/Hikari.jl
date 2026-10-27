# A pbrt material arrives with every parameter pbrt reads for it.
#
# The importer read a coated diffuse material's `roughness` alone: its `uroughness` and
# `vroughness`, which pbrt reads first, and its `thickness`, `albedo`, `g`, `maxdepth` and
# `nsamples` were dropped for the defaults. The pearl of `shadow_pearl_on_velvet_arealight`,
# `uroughness` 0.001, rendered with a mirror-smooth coat, 1.35 % darker than pbrt's. The coated
# conductor read only the two `roughness`es and the IOR, and a dielectric whose `roughness` was a
# texture lost it when no `uroughness` was given. And `CoatedConductor()` without an IOR or a
# reflectance was not pbrt's copper, as its comment said, but IOR 1 + 1i. A conductor's
# `uroughness` and `vroughness` were folded into their larger, `Conductor` having one roughness.

using Test, Hikari, Mantle
isdefined(@__MODULE__, :render_scene) || include(joinpath(@__DIR__, "pbrt", "suite.jl"))

# A stored texture, not a constant: what a constant fallback cannot stand in for.
const ROUGH_TEXTURE = Hikari.TexHandle(Hikari.TexKind.IMAGE, 0f0, Hikari.RGBSpectrum(0f0), Int32(1), Int32(1))

material(text) = let pb = Hikari.parse_pbrt_string("""
        $text
        Shape "trianglemesh" "point3 P" [ 0 0 0  1 0 0  0 1 0 ] "integer indices" [ 0 1 2 ]""")
    Hikari.build_pbrt_material(only(pb.shapes).material_inline, pb, Dict("rough" => ROUGH_TEXTURE))
end
value(h) = Hikari.const_float(h, NaN32)

@testset "pbrt materials keep their parameters" begin
    @testset "coateddiffuse" begin
        m = material("""Material "coateddiffuse" "float roughness" 0.3 "float uroughness" 0.001
                        "float thickness" 0.05 "rgb albedo" [0.2 0.3 0.4] "float g" 0.5
                        "integer maxdepth" 4 "integer nsamples" 3""")
        @test value(m.u_roughness) == 0.001f0
        @test value(m.v_roughness) == 0.3f0
        @test value(m.thickness) == 0.05f0
        @test value(m.g) == 0.5f0
        @test m.albedo != Hikari.matparam(Hikari.RGBSpectrum(0f0))
        @test (m.max_depth, m.n_samples) == (4, 3)
    end

    @testset "coatedconductor" begin
        m = material("""Material "coatedconductor" "float interface.roughness" 0.2 "float interface.vroughness" 0.05
                        "float conductor.uroughness" 0.4 "spectrum conductor.eta" "metal-Au-eta"
                        "spectrum conductor.k" "metal-Au-k" "float thickness" 0.02 "float g" -0.3
                        "integer maxdepth" 6 "integer nsamples" 2 "bool remaproughness" false""")
        @test (value(m.interface_u_roughness), value(m.interface_v_roughness)) == (0.2f0, 0.05f0)
        @test (value(m.conductor_u_roughness), value(m.conductor_v_roughness)) == (0.4f0, 0f0)
        @test value(m.thickness) == 0.02f0
        @test value(m.g) == -0.3f0
        @test (m.max_depth, m.n_samples, m.remap_roughness) == (6, 2, false)
    end

    @testset "an anisotropic conductor" begin
        m = material("""Material "conductor" "spectrum eta" "metal-Au-eta" "spectrum k" "metal-Au-k"
                        "float uroughness" 0.02 "float vroughness" 0.4""")
        @test (value(m.u_roughness), value(m.v_roughness)) == (0.02f0, 0.4f0)
        @test m.eta === Hikari.AU_ETA_SPECTRUM && m.k === Hikari.AU_K_SPECTRUM
    end

    @testset "a dielectric's textured roughness" begin
        m = material("""Material "dielectric" "texture roughness" "rough" """)
        @test m.u_roughness === ROUGH_TEXTURE
        @test m.v_roughness === ROUGH_TEXTURE
    end

    @testset "CoatedConductor's default conductor is copper" begin
        m = Hikari.CoatedConductor()
        @test m.conductor_eta === Hikari.CU_ETA_SPECTRUM
        @test m.conductor_k === Hikari.CU_K_SPECTRUM
    end

    @testset "the pearl's coat is pbrt's" begin
        name = "shadow_pearl_on_velvet_arealight"
        @test compute_metrics(ensure_reference(name), render_scene(name; samples = REFERENCE_SPP)).energy_ratio ≈ 1 rtol = 0.005
    end
end
