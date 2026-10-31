module SurfaceProgramReuseCheck
using Hikari, Test, JSON3
using ..SurfaceProgramGPUProbe: compare

function run(; resolution = 48, samples = 4)
    p = Hikari.Principled(roughness = 0.3, coat_weight = 0.2)
    s, g = Hikari.Sheen(), Hikari.Glossy(roughness = 0.2)
    materials = (
        p, Hikari.Glass(roughness = 0.2), Hikari.Add(p, Hikari.Add(s, g)),
        Hikari.Mix(Hikari.Add(g, s), p; factor = 0.4),
    )
    rows = map(materials) do material
        compare(material; resolution, samples)
    end
    @test all(r -> (r.capacity, r.mask) == (first(rows).capacity, first(rows).mask), rows)
    # Native identity is stronger than merely observing equal Julia signatures.
    reference = first(rows).candidate_native
    @test reference !== nothing
    @test length(reference) == 1
    @test all(r -> r.candidate_native == reference, rows)
    println(JSON3.write(rows))
    return rows
end
end
