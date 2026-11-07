using Test
import Adapt
using Hikari
using Raycore
using GeometryBasics: Point2f
using Colors: RGB

# Value-node families can require different value programs, but spellings of
# the same model share BSDF types. Native spectra live in program data, and
# mixtures store one graph rather than recursively stored material records.
device_types(scene) = [eltype(v) for v in scene.materials.static.data]
fresh_scene() = Hikari.Scene(; hw_accel = false)
include("host_materials.jl")
img_tex(v) = Hikari.Texture(fill(Hikari.RGBSpectrum(v, v, v), 4, 4))
float_tex(v) = Hikari.Texture(fill(Float32(v), 4, 4))
host_static(store) = host_materials(store)

@testset "Material type collapse" begin
    @testset "constant and image spellings share a BSDF kernel" begin
        scene = fresh_scene()
        push!(scene, Hikari.Diffuse(color = (0.5, 0.4, 0.3)))
        push!(scene, Hikari.Diffuse(color = Hikari.RGBSpectrum(0.1f0, 0.2f0, 0.3f0)))
        push!(scene, Hikari.Diffuse(color = Hikari.ConstTexture(Hikari.RGBSpectrum(0.7f0))))
        push!(scene, Hikari.Diffuse(color = img_tex(0.9f0), roughness = float_tex(0.2f0)))
        ts = device_types(scene)
        @test length(ts) == 2 # constant and image value programs
        @test all(T -> T <: Hikari.ShaderMaterial, ts)
        @test all(T -> Hikari.bsdf_nodes(T) === Tuple{Hikari.DiffuseBSDF}, ts)
        @test all(T -> sizeof(T) < sizeof(Hikari.TexHandle), ts)
        @test length(scene.materials) == 4
    end
    @testset "named metals share their spectral BSDF representation" begin
        scene = fresh_scene()
        push!(scene, Hikari.Gold(roughness = 0.1f0))
        push!(scene, Hikari.Gold(roughness = float_tex(0.3f0)))
        push!(scene, Hikari.Silver(roughness = 0.0f0))
        push!(scene, Hikari.Copper(roughness = float_tex(0.05f0)))
        ts = device_types(scene)
        @test length(ts) == 2 # split constants and fused image reads
        @test all(T -> T <: Hikari.ShaderMaterial, ts)
        @test length(unique(Hikari.bsdf_nodes.(ts))) == 1
        @test all(T -> sizeof(T) < sizeof(Hikari.TexHandle), ts)
    end
    @testset "displacement is a graph input" begin
        scene = fresh_scene()
        plain = Hikari.Gold(roughness = 0.05f0)
        bumped = Hikari.set_displacement(plain, Hikari.TexHandle(0.25f0))
        @test isnothing(Hikari.displacement(plain))
        @test !Hikari.is_none(Hikari.displacement(bumped))
        push!(scene, plain)
        push!(scene, bumped)
        push!(scene, Hikari.set_displacement(plain, Hikari.TexHandle(0.5f0)))
        @test length(device_types(scene)) == 2
        @test all(T -> T <: Hikari.ShaderMaterial, device_types(scene))
    end
    @testset "legacy mix displacement is absent" begin
        mix = Hikari.MixMaterial(Hikari.TexHandle(0.5f0), Raycore.SetKey(), Raycore.SetKey())
        @test Hikari.is_none(Hikari.displacement(mix))
        @test Hikari.set_displacement(mix, Hikari.TexHandle(1.0f0)) === mix
    end
    @testset "mixes store graphs without sub-material slots" begin
        scene = fresh_scene()
        push!(scene, Hikari.Mix(Hikari.Diffuse(color = (0.2, 0.2, 0.2)), Hikari.Gold(); factor = 0.25f0))
        push!(
            scene, Hikari.Mix(
                Hikari.CoatedDiffuse(reflectance = (0.4, 0.4, 0.4)), Hikari.Silver();
                factor = float_tex(0.5f0)
            )
        )
        ts = device_types(scene)
        @test length(ts) == 2
        @test length(scene.materials) == 2
        @test all(T -> T <: Hikari.ShaderMaterial, ts)
        @test all(T -> fieldcount(Hikari.bsdf_nodes(T)) == 2, ts)
    end
    @testset "a mix's nodes and programs are stored together" begin
        scene = fresh_scene()
        push!(scene, Hikari.Mix(Hikari.Diffuse(color = (0.2, 0.3, 0.4)), Hikari.Gold(); factor = 0.0f0))
        @test length(scene.materials) == 1
        stored = only(only(host_materials(scene.materials).data))
        @test stored isa Hikari.ShaderMaterial
        @test fieldcount(Hikari.bsdf_nodes(typeof(stored))) == 2
        @test stored.program.idx > 0 && stored.closures > 0
    end
end

# ============================================================================
# TexHandle evaluation
# ============================================================================

@testset "TexHandle" begin
    @testset "constants live inline" begin
        h = Hikari.TexHandle(0.25f0)
        @test h.kind == Hikari.TexKind.CONST_FLOAT
        @test h.slot == 0 && h.idx == 0
        @test Hikari.const_float(h) == 0.25f0

        c = Hikari.TexHandle((0.1, 0.2, 0.3))
        @test c.kind == Hikari.TexKind.CONST_SPECTRUM
        @test Hikari.const_spectrum(c).c[1] ≈ 0.1f0

        # A 0-d `Texture` is how `to_texture` spells a constant; it must not
        # occupy a texture slot.
        @test Hikari.TexHandle(Hikari.ConstTexture(0.5f0)).kind == Hikari.TexKind.CONST_FLOAT
        @test Hikari.is_none(Hikari.TexHandle(nothing))
    end

    @testset "const_spectrum refuses out-of-line kinds" begin
        # Host-side callers (area-light registration) have no texture store, so
        # this must fail loudly rather than silently register a black light.
        h = Hikari.TexHandle(
            Hikari.TexKind.IMAGE, 0.0f0, Hikari.RGBSpectrum(0.0f0),
            Int32(1), Int32(1)
        )
        @test_throws ErrorException Hikari.const_spectrum(h)
    end

    @testset "every kind evaluates against a store holding all of them" begin
        # `with_texture` emits one arm per SLOT and compiles all of them, so a
        # CHECKER handle's switch still generates the image arm for the slot
        # holding a float map. Each arm has to be total over the array types
        # actually present; before the fallbacks existed this was a MethodError,
        # which on the GPU is an unsupported call rather than an error.
        scene = fresh_scene()
        mats = scene.materials
        h_img_f = Hikari.device_param(mats, float_tex(0.5f0))
        h_img_s = Hikari.device_param(mats, img_tex(0.75f0))
        h_check = Hikari.device_param(
            mats, Hikari.CheckerboardTexture(
                2.0f0, 2.0f0, 0.0f0, 0.0f0, 1.0f0, 0.0f0
            )
        )
        @test h_img_f.kind == Hikari.TexKind.IMAGE
        @test h_img_s.kind == Hikari.TexKind.IMAGE
        @test h_check.kind == Hikari.TexKind.CHECKER
        # Distinct array types land in distinct slots.
        @test h_img_f.slot != h_img_s.slot

        static = host_static(mats)
        tfc = Hikari.TextureFilterContext(Point2f(0.25f0, 0.25f0))
        @test Hikari.eval_handle(static, h_img_f, tfc) ≈ 0.5f0
        @test Hikari.eval_handle_spectrum(static, h_img_s, tfc).c[1] ≈ 0.75f0
        # Cross-evaluating a handle through the "wrong" accessor must return a
        # value rather than throw — that is exactly what the unreachable arms
        # compile to on the device.
        @test Hikari.eval_handle(static, h_img_s, tfc) isa Float32
        @test Hikari.eval_handle_spectrum(static, h_img_f, tfc) isa Hikari.RGBSpectrum
        @test Hikari.eval_handle(static, h_check, tfc) isa Float32
        @test Hikari.eval_handle_spectrum(static, h_check, tfc) isa Hikari.RGBSpectrum
        # NONE reads as zero, never as garbage.
        @test Hikari.eval_handle(static, Hikari.TexHandle(), tfc) == 0.0f0
    end

    @testset "checker handle matches direct evaluation" begin
        scene = fresh_scene()
        cb = Hikari.CheckerboardTexture(4.0f0, 4.0f0, 0.0f0, 0.0f0, 1.0f0, 0.0f0)
        h = Hikari.device_param(scene.materials, cb)
        static = host_static(scene.materials)
        for uv in (Point2f(0.05f0, 0.05f0), Point2f(0.3f0, 0.1f0), Point2f(0.7f0, 0.6f0))
            tfc = Hikari.TextureFilterContext(uv)
            @test Hikari.eval_handle(static, h, tfc) ≈ Float32(Hikari.eval_tex(static, cb, tfc))
        end
    end
end
