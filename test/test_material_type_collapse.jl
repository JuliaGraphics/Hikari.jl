using Test
import Adapt
using Hikari
using Raycore
using GeometryBasics: Point2f
using Colors: RGB

# ============================================================================
# The material type explosion must stay collapsed
# ============================================================================
#
# The per-material closest-hit path compiles ONE shader per concrete material
# type that reaches the device, so "the same material spelled differently" used
# to cost a whole extra shader each. Crown had 12 concrete types where it uses
# 5 material CLASSES; the other 7 were constant-vs-texture combinatorics and
# wrapper types, and the two MixMaterial types were the most expensive of them
# because a mix's shading kernel inlines the whole shading system.
#
# These tests pin the three mechanisms that keep it collapsed:
#
#   1. `to_device_material` rewrites every texture-carrying field to a
#      `TexHandle`, so the spelling stops mattering at push time.
#   2. `displacement` is a field, not a `BumpMapped{…}` wrapper.
#   3. `MixMaterial` references its sub-materials by `SetKey`, not by value.
#
# Each is checked on the DEVICE-SIDE type — what `scene.materials` actually
# stores — because that, not the host type, is what the SBT is built from.

# The set's stored element types, in slot order.
device_types(scene) = [eltype(v) for v in scene.materials.static.data]

fresh_scene() = Hikari.Scene(; hw_accel = false)

# Download texture owners, not the device pointer table: Array(static.textures)
# copies handles whose payloads still name device memory.
function host_materials(mats)
    data = Tuple(copy(mats.data_vectors[T]) for T in mats.data_order)
    textures = Tuple([Array(a) for a in mats.texture_gpu_arrays if typeof(a) === T]
                     for T in mats.texture_order)
    return Raycore.StaticMultiTypeSet(data, textures)
end

img_tex(v) = Hikari.Texture(fill(Hikari.RGBSpectrum(v, v, v), 4, 4))
float_tex(v) = Hikari.Texture(fill(Float32(v), 4, 4))

# Evaluate host fixtures against downloaded texture owners, rather than device pointers.
host_static(store) = host_materials(store)

@testset "Material type collapse" begin

    @testset "constant vs texture is one device type" begin
        scene = fresh_scene()
        # Four spellings of the same material class: raw tuple, RGBSpectrum,
        # a 0-d ConstTexture, and a real 4x4 image map.
        push!(scene, Hikari.Diffuse(color = (0.5, 0.4, 0.3)))
        push!(scene, Hikari.Diffuse(color = Hikari.RGBSpectrum(0.1f0, 0.2f0, 0.3f0)))
        push!(scene, Hikari.Diffuse(color = Hikari.ConstTexture(Hikari.RGBSpectrum(0.7f0))))
        push!(scene, Hikari.Diffuse(color = img_tex(0.9f0), roughness = float_tex(0.2f0)))
        ts = device_types(scene)
        @test length(ts) == 1
        @test ts[1] <: Hikari.Diffuse
        # Every parameter is a handle — checked without pinning the arity, so
        # adding a parameter (as `displacement` did) doesn't break the test for
        # the wrong reason.
        @test all(p -> p === Hikari.TexHandle, ts[1].parameters)
    end

    @testset "roughness texture does not fork Conductor" begin
        scene = fresh_scene()
        push!(scene, Hikari.Gold(roughness = 0.1f0))
        push!(scene, Hikari.Gold(roughness = float_tex(0.3f0)))
        # Measured spectra stay out of line: a PiecewiseLinearSpectrum{56} is
        # 448 bytes and cannot live inline in a 32-byte handle. Every named
        # metal shares the same representation, so this is one type, not five.
        push!(scene, Hikari.Silver(roughness = 0f0))
        push!(scene, Hikari.Copper(roughness = float_tex(0.05f0)))
        ts = device_types(scene)
        @test length(ts) == 1
        @test ts[1] <: Hikari.Conductor
    end

    @testset "bumped and unbumped are one device type" begin
        scene = fresh_scene()
        plain = Hikari.Gold(roughness = 0.05f0)
        bumped = Hikari.set_displacement(plain, Hikari.TexHandle(0.25f0))
        @test isnothing(Hikari.displacement(plain))
        @test !Hikari.is_none(Hikari.displacement(bumped))
        push!(scene, plain)
        push!(scene, bumped)
        @test length(device_types(scene)) == 1
    end

    @testset "displacement is absent, not wrong, on materials without the field" begin
        # `MixMaterial` has no height field — pbrt-v4 resolves the mix to a
        # sub-material before consulting GetDisplacement. `set_displacement`
        # must leave such a material alone rather than invent a field.
        mix = Hikari.MixMaterial(Hikari.TexHandle(0.5f0), Raycore.SetKey(), Raycore.SetKey())
        @test Hikari.is_none(Hikari.displacement(mix))
        @test Hikari.set_displacement(mix, Hikari.TexHandle(1f0)) === mix
    end

    @testset "MixMaterial is one type whatever it mixes" begin
        scene = fresh_scene()
        # Two mixes over different sub-material PAIRS, and different amount
        # spellings. Before, each pair was its own MixMaterial{M1,M2} — and a
        # mix's shading kernel dispatches over every material type in the scene, so each
        # extra one is a full copy of the shading system.
        push!(scene, Hikari.Mix(Hikari.Diffuse(color = (0.2, 0.2, 0.2)), Hikari.Gold(); factor = 0.25f0))
        push!(scene, Hikari.Mix(Hikari.CoatedDiffuse(reflectance = (0.4, 0.4, 0.4)), Hikari.Silver();
                                factor = float_tex(0.5f0)))
        ts = device_types(scene)
        mixes = filter(t -> t <: Hikari.MixMaterial, ts)
        @test length(mixes) == 1
        @test mixes[1] === Hikari.MixMaterial{Hikari.TexHandle}
        # The sub-materials are in the set too, each collapsed to one type.
        @test count(t -> t <: Hikari.Diffuse, ts) == 1
        @test count(t -> t <: Hikari.Conductor, ts) == 1
        @test count(t -> t <: Hikari.CoatedDiffuse, ts) == 1
    end

    @testset "a mix references its sub-materials, and they resolve" begin
        scene = fresh_scene()
        d = Hikari.Diffuse(color = (0.2, 0.3, 0.4))
        g = Hikari.Gold()
        push!(scene, Hikari.Mix(d, g; factor = 0f0))
        stored = only(filter(m -> m isa Hikari.MixMaterial,
                             [v[1] for v in host_materials(scene.materials).data if length(v) >= 1]))
        @test Raycore.is_valid(stored.material1_idx)
        @test Raycore.is_valid(stored.material2_idx)
        @test stored.material1_idx != stored.material2_idx
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
        h = Hikari.TexHandle(Hikari.TexKind.IMAGE, 0f0, Hikari.RGBSpectrum(0f0),
                             Int32(1), Int32(1))
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
        h_check = Hikari.device_param(mats, Hikari.CheckerboardTexture(
            2f0, 2f0, 0f0, 0f0, 1f0, 0f0))
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
        @test Hikari.eval_handle(static, Hikari.TexHandle(), tfc) == 0f0
    end

    @testset "checker handle matches direct evaluation" begin
        scene = fresh_scene()
        cb = Hikari.CheckerboardTexture(4f0, 4f0, 0f0, 0f0, 1f0, 0f0)
        h = Hikari.device_param(scene.materials, cb)
        static = host_static(scene.materials)
        for uv in (Point2f(0.05f0, 0.05f0), Point2f(0.3f0, 0.1f0), Point2f(0.7f0, 0.6f0))
            tfc = Hikari.TextureFilterContext(uv)
            @test Hikari.eval_handle(static, h, tfc) ≈ Float32(Hikari.eval_tex(static, cb, tfc))
        end
    end
end
