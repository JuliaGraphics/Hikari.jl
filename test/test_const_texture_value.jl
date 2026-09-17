using Test
using Hikari
using Hikari: Texture, ConstTexture, TexHandle, TexKind, RGBSpectrum, constant_value,
              const_float, device_param, Diffuse

# A 0-d `Texture` carries its value in one of two places depending on which
# constructor built it:
#
#   Texture(arr)      -> value in `data`,     `constval` zeroed, isconst=false
#   ConstTexture(val) -> value in `constval`, `data` is Array{T,0}(undef)
#
# A conversion to a `TexHandle` that reads `data[]` unconditionally gives a
# `ConstTexture` a handle holding *uninitialised memory*. Nothing in
# the pbrt builder hit it (scalar parameters arrive as raw `Real`/`RGBSpectrum`
# and take the value overloads), but RayMakie's `merge_color_with_material`
# passes the plot's colour through as a const 0-d `Texture` — so a coloured
# Makie plot got a material whose Kd was garbage. Garbage that happened to be
# negative renders black, which is how it surfaced: geometry in the TLAS,
# nothing in the image.
#
# The value is compared against `constval` rather than any particular number:
# the point is which FIELD is read, not what is in it.

@testset "const 0-d texture keeps its value" begin
    @testset "constant_value reads the field the constructor used" begin
        c = ConstTexture(RGBSpectrum(0.2f0, 0.55f0, 0.75f0, 1.0f0))
        @test c.isconst
        @test constant_value(c) == c.constval

        # The non-const 0-d form still reads `data`, and is unaffected.
        arr = Array{RGBSpectrum, 0}(undef)
        arr[] = RGBSpectrum(0.1f0, 0.2f0, 0.3f0, 1.0f0)
        t = Texture(arr)
        @test !t.isconst
        @test constant_value(t) == arr[]
    end

    @testset "TexHandle from a const texture" begin
        val = RGBSpectrum(0.2f0, 0.55f0, 0.75f0, 1.0f0)
        h = TexHandle(ConstTexture(val))
        @test h.kind == TexKind.CONST_SPECTRUM
        @test h.rgb == val

        hf = TexHandle(ConstTexture(0.625f0))
        @test hf.kind == TexKind.CONST_FLOAT
        @test hf.f == 0.625f0
    end

    @testset "const_float from a const texture" begin
        @test const_float(ConstTexture(0.625f0)) == 0.625f0
    end

    # `device_param` is what the scene builder calls, and is the exact path
    # RayMakie's material merge takes. `nothing` stands in for the texture
    # store: a constant never reaches it.
    @testset "device_param folds a const texture inline" begin
        val = RGBSpectrum(0.2f0, 0.55f0, 0.75f0, 1.0f0)
        h = device_param(nothing, ConstTexture(val))
        @test h isa TexHandle
        @test h.kind == TexKind.CONST_SPECTRUM
        @test h.rgb == val
    end

    # The whole point: a material built with a const-texture parameter must
    # arrive on the device holding that colour. Repeated because uninitialised
    # memory can coincidentally match once.
    @testset "material parameter survives the conversion" begin
        for _ in 1:20
            val = RGBSpectrum(0.2f0, 0.55f0, 0.75f0, 1.0f0)
            mat = Diffuse(Kd=ConstTexture(val), σ=0.0f0)
            dev = Hikari.to_device_material(nothing, mat)
            @test dev.Kd.kind == TexKind.CONST_SPECTRUM
            @test dev.Kd.rgb == val
        end
    end
end
