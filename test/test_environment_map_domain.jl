# A direction that is a hair longer than unit must not throw.
#
# `equal_area_sphere_to_square` computed `sqrt(1 - |d.z|)` with no guard, which
# is a `DomainError` the moment `|d.z| > 1`. pbrt gets away with the same line
# because it DCHECKs that `d` is normalised; on a GPU there is no such luxury —
# Metal turns the throw into a device-side exception that kills the render loop
# and leaves the window up, frozen, indistinguishable from a hang.
#
# Found by selecting GOLD in `Mantle/examples/isubd`: a conductor is a mirror,
# so nearly every path it spawns reaches the environment, and it died within
# seconds every time. The directions are not exotic — see the second testset.

using Test, Hikari, GeometryBasics, LinearAlgebra, Random
using GeometryBasics: Vec3f, Point2f
const H = Hikari

@testset "equal_area_sphere_to_square domain" begin
    # The exact value that first threw: 1 + eps(1f0) in z.
    @test H.equal_area_sphere_to_square(Vec3f(0, 0, 1f0 + eps(1f0))) isa Point2f
    @test H.equal_area_sphere_to_square(Vec3f(0, 0, -1f0 - eps(1f0))) isa Point2f

    # Grossly non-unit, both poles, and the exact poles.
    for z in (1f0, -1f0, 1.5f0, -1.5f0, 1.8487625f0, 2f0, 10f0)
        @test H.equal_area_sphere_to_square(Vec3f(0, 0, z)) isa Point2f
    end

    # |z| ≥ 1 is the pole, and the pole is where r = 0 puts it: clamping is the
    # right answer, not merely a guard. Both poles map to a CORNER of the square
    # in the equal-area octahedral layout, so what is pinned is that the
    # slightly-over-unit direction agrees with the exact one it rounds off.
    @test H.equal_area_sphere_to_square(Vec3f(0, 0, 1f0 + eps(1f0))) ≈
          H.equal_area_sphere_to_square(Vec3f(0, 0, 1f0))
    @test H.equal_area_sphere_to_square(Vec3f(0, 0, -1f0 - eps(1f0))) ≈
          H.equal_area_sphere_to_square(Vec3f(0, 0, -1f0))

    # A normalised direction is untouched by the clamp: round-tripping still
    # works to the mapping's own precision.
    rng = MersenneTwister(20260923)
    for _ in 1:2000
        d = normalize(Vec3f(randn(rng, Float32), randn(rng, Float32), randn(rng, Float32)))
        uv = H.equal_area_sphere_to_square(d)
        @test all(-1f-5 .<= Tuple(uv) .<= 1f0 + 1f-5)
        @test H.equal_area_square_to_sphere(uv) ≈ d atol = 1f-3
    end
end

@testset "a conductor's own reflection leaves the unit sphere" begin
    # Not a hypothetical: this is `conductor.jl`'s `wi` line for line, and the
    # rate is why GOLD died in seconds rather than intermittently.
    randunit(r) = normalize(Vec3f(randn(r, Float32), randn(r, Float32), randn(r, Float32)))
    rng = MersenneTwister(20260923)
    n_over = 0
    N = 200_000
    for _ in 1:N
        n = randunit(rng); dp = randunit(rng)
        tangent, bitangent = H.shading_frame(n, dp)
        wo = H.world_to_local(randunit(rng), n, tangent, bitangent)
        wo[3] == 0f0 && continue
        wm = H.trowbridge_reitz_sample_wm(wo, Point2f(rand(rng, Float32), rand(rng, Float32)),
                                          0.05f0, 0.05f0)
        wi = -wo + 2f0 * dot(wo, wm) * wm
        wiw = H.local_to_world(wi, n, tangent, bitangent)
        abs(wiw[3]) > 1f0 && (n_over += 1)
        # The point of the fix: whatever comes out, this call returns.
        @test H.equal_area_sphere_to_square(wiw) isa Point2f
    end
    # Measured ≈1 in 1300. Pinned loosely as "this really does happen, often",
    # so the test fails if someone later assumes normalised input again.
    @test n_over > 0
end

@testset "Henyey-Greenstein denominator is guarded" begin
    # `hg_p` had a bare `sqrt(denom)` while `hg_phase_pdf` in materials/common.jl
    # guarded the identical expression. At g → ±1 and grazing cos θ, denom is
    # (1 ∓ g)² — zero, or slightly negative in Float32.
    # Away from the degenerate corner, the density is finite and positive.
    for g in (0.9f0, -0.9f0, 0.99f0, -0.99f0, 0.5f0, -0.5f0, 0f0)
        for c in (1f0, -1f0, 0.999999f0, -0.999999f0, 0f0)
            @test isfinite(H.hg_p(g, c))
            @test H.hg_p(g, c) > 0f0
            @test H.hg_p(g, c) ≈ H.hg_phase_pdf(g, c)
        end
    end

    # As |g| → 1 the phase function becomes a Dirac delta, so at the forward
    # peak it genuinely DIVERGES — and `denom = 1 + g² - 2g·cosθ` cancels to
    # zero in Float32 well before that. Inf (or NaN, at |g| = 1, where the
    # numerator vanishes too) is the arithmetic's honest answer, so finiteness
    # is not the property to demand here.
    #
    # What the guard buys is that these RETURN instead of throwing a
    # `DomainError`, and that `hg_p` does exactly what its twin does — the
    # inconsistency being the actual bug. On a GPU the throw is not a NaN you
    # can trace; it is a device-side exception that kills the render loop.
    for g in (1f0, -1f0, 0.99999f0, -0.99999f0, 0.9999999f0, -0.9999999f0)
        for c in (1f0, -1f0, 0.999999f0, -0.999999f0, 0f0)
            a = H.hg_p(g, c)           # no DomainError: reaching here is the test
            b = H.hg_phase_pdf(g, c)
            @test (isnan(a) && isnan(b)) || a ≈ b
        end
    end
end
