using Test, Hikari
using GeometryBasics: Vec3f

@testset "a shader samples every subsurface lobe" begin
    S = Hikari.SpectralRadiance
    n, t = Vec3f(0, 0, 1), Vec3f(1, 0, 0)
    walk(radius, weight) = Hikari.Weighted(
        Hikari.SubsurfaceLobe{S}(S(0.8f0), S(radius), 0.25f0, 1.4f0, 0f0, false),
        S(weight / 4f0), weight, n, t)
    diffuse = Hikari.Weighted(Hikari.LambertLobe(S), S(1f0), 4f0, n, t)
    a, b = walk(0.01f0, 1f0), walk(0.2f0, 3f0)
    lobes = (diffuse, a, b)
    @test Hikari.subsurface_weight(lobes) == 4f0
    counts = [0, 0, 0]
    for i in 1:1024
        selected = Hikari.subsurface_choice(lobes, Float32((i - 0.5) * 8 / 1024))
        counts[selected == 0 ? 1 : selected] += 1
    end
    @test counts == [512, 128, 384]
    for (selected, expected) in ((Int32(2), a), (Int32(3), b))
        p, weight = Hikari.picked_walk_params(lobes, selected, Val(3))
        @test p.radius == expected.lobe.radius
        @test p.weight == expected.weight
        @test weight == expected.sample_weight
    end
    surface = map(Hikari.without_subsurface_samples, lobes)
    @test surface[1] === diffuse
    @test surface[2].sample_weight == surface[3].sample_weight == 0f0

    # An absent last walk must not suppress an earlier, active walk.
    inactive = Hikari.without_samples(b)
    @test Hikari.subsurface_choice((diffuse, a, inactive), 0.5f0) == 2
    @test Hikari.subsurface_weight((diffuse, a, inactive)) == 1f0
    lambda = Hikari.sample_wavelengths_visible(0.5f0)
    empty_walks = Hikari.LobeBSDF((diffuse, Hikari.without_samples(a), inactive), 4f0, lambda, S(0f0), false)
    @test Hikari.subsurface_scatter(empty_walks, nothing, nothing, Int32(1)) === (empty_walks, nothing)
    @test Hikari.subsurface_slot(typeof((diffuse,))) == 0
    @test Hikari.subsurface_weight(()) == 0f0
    @test Hikari.subsurface_choice((), 0.5f0) == 0
end
