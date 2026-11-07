# pbrt names a spectrum or a file of (λ, value) pairs under one syntax. `killeroo-gold.pbrt` gives
# its conductor a file, and the scene stopped loading when names became the only strings accepted.

using Test, Hikari

@testset "a pbrt spectrum parameter may name a file" begin
    mktempdir() do dir
        mkpath(joinpath(dir, "spds"))
        write(joinpath(dir, "spds", "eta.spd"), "# a comment\n400.0 1.5  # trailing\n\n500.0 1.25\n600.0 1.0\n")
        write(joinpath(dir, "spds", "odd.spd"), "400.0 1.5 500.0\n")
        parse(text) = Hikari.parse_params!(Hikari.TokenStream(Hikari.tokenize(text); base_dir = dir))

        params = parse("\"spectrum eta\" [ \"spds/eta.spd\" ]")
        @test params["eta"].values == Any[400.0, 1.5, 500.0, 1.25, 600.0, 1.0]
        ior = Hikari.pbrt_get_ior(Hikari.PBRTEntity("Material", "conductor", params), "eta")
        @test ior isa Hikari.PiecewiseLinearSpectrum{3}
        @test ior == Hikari.from_interleaved(Hikari.PiecewiseLinearSpectrum{3},
                                             (400f0, 1.5f0, 500f0, 1.25f0, 600f0, 1f0))

        # A name is still a name, a missing one still refused by name, a bad file by path.
        named = parse("\"spectrum eta\" [ \"metal-Au-eta\" ]")
        @test named["eta"].values == Any["metal-Au-eta"]
        @test Hikari.pbrt_get_ior(Hikari.PBRTEntity("Material", "conductor", named), "eta") ==
              Hikari.NAMED_IOR_SPECTRA["metal-Au-eta"]
        unknown = parse("\"spectrum eta\" [ \"metal-Nope-eta\" ]")
        @test_throws ArgumentError Hikari.pbrt_get_ior(Hikari.PBRTEntity("Material", "conductor", unknown), "eta")
        @test_throws ArgumentError parse("\"spectrum eta\" [ \"spds/odd.spd\" ]")
    end
end
