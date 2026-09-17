using Test
using Hikari
using Raycore
using GeometryBasics: Point2f, Vec3f
using LinearAlgebra: normalize

# ============================================================================
# Every material shades through `get_bxdf`, and only through `get_bxdf`
# ============================================================================
#
# Surface shading has exactly one implementation: `vp_shade_material_kernel!{T}`
# (and the closest-hit shader, which calls the same two inner functions). It
# resolves the BxDF once per hit with `get_bxdf` — pbrt-v4's
# `Material::GetBxDF` — and then calls the BSDF methods on the carrier that
# comes back.
#
# A second implementation reached through `with_index`, live for exactly one
# configuration (hardware RT on a scene with media), calls
# `sample_bsdf_spectral` on the RAW material. Once a material moved to the
# GetBxDF pattern its BSDF methods lived on the carrier, so the raw material
# matched nothing but a gray-Lambertian `::Material` catch-all — silently,
# because a catch-all is a method, not an error. A specular
# `Material "dielectric" "float eta" 1.0` medium boundary therefore became a
# diffuse wall: light could not leave the box and `medium_smoke_point` rendered
# at 0.037 of its reference energy on hardware RT while software RT was fine.
#
# There is one implementation, and these tests pin the safety a second one only
# appears to give: every material class resolves through `get_bxdf`, and no
# converted material has a BSDF method on its raw type for a future `with_index`
# to find.

const TABLE  = Hikari.get_srgb_table()
const LAMBDA = Hikari.sample_wavelengths_stratified((0.11f0, 0.27f0, 0.53f0, 0.79f0))
const TFC    = Hikari.TextureFilterContext(Point2f(0.5f0, 0.5f0))
const WO     = normalize(Vec3f(0.3f0, 0.2f0, 1.0f0))
const NS     = Vec3f(0f0, 0f0, 1f0)
const DPDUS  = Vec3f(1f0, 0f0, 0f0)
const U      = Point2f(0.37f0, 0.61f0)
const RNG    = 0.42f0

# Push `mat` into a fresh CPU scene and hand back what the device would see:
# the stored (converted) material and the static set backing it.
function device_material(mat)
    scene = Hikari.Scene(; hw_accel = false)
    key = scene.media_interfaces[push!(scene, mat)].material
    static = scene.materials.static
    return static.data[key.type_idx][key.vec_idx], static
end

# Every material class that reaches the device with a BSDF of its own.
material_cases() = [
    "Diffuse"                   => Hikari.Diffuse(),
    "DiffuseTransmission"       => Hikari.DiffuseTransmission(),
    "Conductor"                 => Hikari.Conductor(),
    "Mirror"                    => Hikari.Mirror(),
    "Dielectric"                => Hikari.Dielectric(),
    "Dielectric eta=1"          => Hikari.Dielectric(index = 1.0f0),
    "Dielectric rough"          => Hikari.Dielectric(roughness = 0.3f0),
    "ThinDielectric"            => Hikari.ThinDielectric(),
    "CoatedDiffuse"             => Hikari.CoatedDiffuse(),
    "CoatedConductor"           => Hikari.CoatedConductor(),
    "CoatedDiffuseTransmission" => Hikari.CoatedDiffuseTransmission(),
]

# The argument list `vp_shade_material_kernel!` would use.
bsdf_argtypes(M, static) = Tuple{M, typeof(TABLE), typeof(static), Vec3f, Vec3f, Vec3f,
                                 Hikari.TextureFilterContext, Hikari.Wavelengths,
                                 Point2f, Float32, Bool}

@testset "BxDF dispatch" begin

    @testset "get_bxdf is the only way in" begin
        for (name, mat) in material_cases()
            @testset "$name" begin
                stored, static = device_material(mat)

                # The carrier shades.
                bxdf = Hikari.get_bxdf(stored, TABLE, static, TFC, LAMBDA, false)
                s = Hikari.sample_bsdf_spectral(bxdf, TABLE, static, WO, NS, DPDUS,
                                                TFC, LAMBDA, U, RNG, false)
                @test s isa Hikari.SpectralBSDFSample

                wi = s.pdf > 0f0 ? s.wi : normalize(Vec3f(-0.2f0, 0.1f0, 0.9f0))
                f, pdf = Hikari.evaluate_bsdf_spectral(bxdf, TABLE, static, WO, wi, NS,
                                                       DPDUS, TFC, LAMBDA, false)
                @test f isa Hikari.SpectralRadiance
                @test pdf >= 0f0

                # The raw material does not. This is the assertion that would
                # have caught the original bug: with a `::Material` catch-all in
                # scope it passes for zero materials, and any future conversion
                # that forgets a call site becomes a method error rather than a
                # dimmer render.
                @test !hasmethod(Hikari.sample_bsdf_spectral, bsdf_argtypes(typeof(stored), static))
                @test !hasmethod(Hikari.evaluate_bsdf_spectral,
                                 Tuple{typeof(stored), typeof(TABLE), typeof(static),
                                       Vec3f, Vec3f, Vec3f, Vec3f,
                                       Hikari.TextureFilterContext, Hikari.Wavelengths, Bool})
            end
        end
    end

    # The exact configuration that went black: an eta = 1 dielectric is a
    # straight-through specular boundary. The gray-Lambertian fallback returned
    # a cosine-sampled hemisphere direction instead, which is what trapped every
    # ray inside the smoke box.
    @testset "eta=1 dielectric transmits" begin
        stored, static = device_material(Hikari.Dielectric(index = 1.0f0))
        bxdf = Hikari.get_bxdf(stored, TABLE, static, TFC, LAMBDA, false)
        s = Hikari.sample_bsdf_spectral(bxdf, TABLE, static, WO, NS, DPDUS,
                                        TFC, LAMBDA, U, RNG, false)
        @test s.pdf == 1f0
        @test Hikari.is_specular(s.flags)
        @test Hikari.is_transmissive(s.flags)
        @test all(isapprox.(s.wi, -WO; atol = 1f-6))
    end

    # `MixMaterial` and `Emissive` never shade a hit — mix is re-pointed at a
    # sub-material at the push site, emissive lives in `MediumInterface.emission`
    # — but `foreach_type` still generates a kernel for every type in the set, so
    # they need methods. Those methods contribute nothing, on purpose: an
    # unreachable path that renders black is a path you can see.
    @testset "BSDF-less materials are inert" begin
        for mat in (Hikari.Emissive(Le = (1, 1, 1)),
                    Hikari.MixMaterial(Hikari.TexHandle(0.5f0), Raycore.SetKey(), Raycore.SetKey()))
            stored, static = device_material(mat)
            @test typeof(stored) <: Hikari.BSDFLessMaterial
            bxdf = Hikari.get_bxdf(stored, TABLE, static, TFC, LAMBDA, false)
            s = Hikari.sample_bsdf_spectral(bxdf, TABLE, static, WO, NS, DPDUS,
                                            TFC, LAMBDA, U, RNG, false)
            @test s.pdf == 0f0
            f, pdf = Hikari.evaluate_bsdf_spectral(bxdf, TABLE, static, WO, NS, NS, DPDUS,
                                                   TFC, LAMBDA, false)
            @test pdf == 0f0
            @test Hikari.is_black(f)
        end
    end
end
