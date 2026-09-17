using Test
using Hikari
using GeometryBasics: Point2f

# CheckerboardTexture is an exact port of pbrt-v4's CheckerboardTexture
# (textures.cpp `Checkerboard()`, 2D case, fused with UVMapping::Map). It
# replaced the 256² LUT rasterization in the pbrt scene builder: the LUT's
# bilinear smoothing both quantized checker-edge positions and widened the
# height-gradient bands that bump mapping differentiates, which biased
# shadow_bumpgold_dome_over_velvet 21% dark at 256 spp.
#
# These tests pin the analytic evaluation against hand-computed pbrt values so
# the LUT (or any other approximation) can't sneak back in.

# Reference implementation of pbrt-v4's helpers, straight from textures.cpp.
pbrt_d(x) = begin
    y = x / 2 - floor(x / 2) - 0.5
    x / 2 + y * (1 - 2 * abs(y))
end
pbrt_bf(x, r) = begin
    if floor(x - r) == floor(x + r)
        return 1.0 - 2.0 * (Int(floor(x)) & 1)
    end
    (pbrt_d(x + r) - 2 * pbrt_d(x) + pbrt_d(x - r)) / (r * r)
end
function pbrt_checker_weight(u, v, su, sv, du, dv, dudx, dudy, dvdx, dvdy)
    s = su * u + du
    t = sv * v + dv
    ds = 1.5 * max(abs(su * dudx), abs(su * dudy))
    dt = 1.5 * max(abs(sv * dvdx), abs(sv * dvdy))
    return 0.5 - 0.5 * pbrt_bf(s, ds) * pbrt_bf(t, dt)
end

@testset "CheckerboardTexture (pbrt-v4 analytic port)" begin

@testset "point-sample parity (zero derivatives)" begin
    # 8×8 checkerboard over [0,1]²: even cell parity → tex1.
    cb = Hikari.CheckerboardTexture(8f0, 8f0, 0f0, 0f0, 10f0, 20f0)
    for (u, v, expected) in [
        (0.01f0, 0.01f0, 10f0),   # cell (0,0) even
        (0.13f0, 0.01f0, 20f0),   # cell (1,0) odd
        (0.13f0, 0.13f0, 10f0),   # cell (1,1) even
        (0.99f0, 0.99f0, 10f0),   # cell (7,7) even
        (0.99f0, 0.87f0, 20f0),   # cell (7,6) odd
    ]
        w = Hikari.checker_bf(8f0 * u, 0f0) * Hikari.checker_bf(8f0 * v, 0f0)
        val = expected == 10f0 ? 1f0 : -1f0
        @test w == val
    end
end

@testset "checker_bf matches pbrt reference" begin
    # atol accounts for Float32 cancellation in the (d(x+r) - 2d(x) + d(x-r))/r²
    # second difference at large |x| — pbrt's own Float32 evaluation has the
    # same error; the Float64 reference here is just more precise.
    for x in Float32[-3.7, -0.2, 0.0, 0.4, 1.3, 2.5, 7.99], r in Float32[0.0, 0.01, 0.3, 1.7]
        @test Hikari.checker_bf(x, r) ≈ Float32(pbrt_bf(x, r)) atol = 2e-3
    end
end

@testset "filtered weight matches pbrt reference" begin
    for (u, v) in [(0.124f0, 0.5f0), (0.5f0, 0.126f0), (0.49f0, 0.51f0)]
        for deriv in Float32[0.0, 0.001, 0.02, 0.2]
            expected = Float32(pbrt_checker_weight(u, v, 8.0, 8.0, 0.0, 0.0,
                                                   deriv, 0.0, 0.0, deriv))
            cb = Hikari.CheckerboardTexture(8f0, 8f0, 0f0, 0f0, 0f0, 1f0)
            s = cb.su * u + cb.du
            t = cb.sv * v + cb.dv
            ds = 1.5f0 * max(abs(cb.su * deriv), abs(cb.su * 0f0))
            dt = 1.5f0 * max(abs(cb.sv * 0f0), abs(cb.sv * deriv))
            w = 0.5f0 - 0.5f0 * Hikari.checker_bf(s, ds) * Hikari.checker_bf(t, dt)
            @test w ≈ expected atol = 1e-5
            # Wide filters converge to the 0.5 average.
            deriv > 0.1f0 && @test abs(w - 0.5f0) < 0.2f0
        end
    end
end

@testset "udelta/vdelta shift the lattice" begin
    # Shifting by exactly one cell flips parity.
    w0 = Hikari.checker_bf(1f0 * 0.3f0 + 0f0, 0f0)
    w1 = Hikari.checker_bf(1f0 * 0.3f0 + 1f0, 0f0)
    @test w0 == -w1
end

end

@testset "pbrt imagemap encoding rule (PNG → sRGB default)" begin
    # pbrt-v4 textures.cpp:436: the DEFAULT encoding is extension-based —
    # `.png → sRGB, everything else → linear` — for float AND spectrum
    # classes, including 16-bit PNGs (image.cpp ReadPNG decodes 16-bit
    # through `encoding.ToFloatLinear`). sRGB-decoding only 8-bit
    # spectrum-class PNGs leaves float height maps linear and
    # scaled bump gradients wrong by the local slope of the sRGB curve
    # (~1.45× at h≈0.7) — pinned by tex_conductor_bumpmap_light_point.
    @test Hikari.srgb_to_linear(0.04045f0) ≈ 0.04045f0 / 12.92f0
    @test Hikari.srgb_to_linear(0.5f0) ≈ ((0.5f0 + 0.055f0) / 1.055f0)^2.4f0
    @test Hikari.srgb_to_linear(1f0) ≈ 1f0
end

# ── the analytic filter's own bound ───────────────────────────────────────────
#
# `checker_bf` is a triangle-filtered ±1 square wave, so |bf| <= 1 by
# construction. The expression that computes it is a second difference divided
# by `r * r`, and near a cell boundary with a small radius the numerator is all
# cancellation — at `x = 1.0f0, r = 1f-6` it evaluated to 59604.6 before the
# clamp, a factor of sixty thousand over its own bound.
#
# That is not a cosmetic overshoot. The caller mixes with it as
# `(1 - w) * tex1 + w * tex2`, so a huge `w` extrapolates far outside the two
# texture values: for the `tex_conductor_checker_rough_light_point` scene's
# 0.01/0.3 roughness pair it produced roughness = -5.15e8, and
# `roughness_to_α` is `sqrt`. C++ answers `std::sqrt(-x)` with a NaN and loses
# one pixel; Julia THROWS, and the whole GPU dispatch dies with
# `DomainError: This operation requires a complex input to return a complex
# result`. Both that scene and its dielectric twin failed that way on the
# software and hardware paths alike.
@testset "checker_bf cannot leave [-1, 1]" begin
    bf = Hikari.checker_bf

    # The exact case that killed the dispatch.
    @test abs(bf(1.0f0, 1f-6)) <= 1f0

    # Cell boundaries at every scale of radius: this is where `floor(x - r)` and
    # `floor(x + r)` disagree, so the point-sample early-out does not fire and
    # the ill-conditioned division is taken.
    for x in Float32[-4, -2, -1, 0, 1, 2, 3, 8, 17],
        r in Float32[1f-7, 1f-6, 1f-5, 1f-4, 1f-3, 1f-2, 0.1, 0.5, 1, 2]
        v = bf(x, r)
        @test isfinite(v)
        @test -1f0 <= v <= 1f0
    end

    # …and away from boundaries too, over a sweep that mixes both branches.
    for x in range(-20f0, 20f0; length = 401), r in Float32[1f-6, 1f-3, 0.25f0, 1.5f0]
        v = bf(Float32(x), r)
        @test isfinite(v) && -1f0 <= v <= 1f0
    end

    # The clamp must not disturb the well-conditioned cases the other testsets
    # pin: a radius that keeps the support inside one cell is the exact point
    # sample, and `+1` / `-1` are its only two values.
    @test bf(0.25f0, 0.1f0) == 1f0
    @test bf(1.25f0, 0.1f0) == -1f0
    @test bf(2.25f0, 0.1f0) == 1f0
end
