"""
Sphere UVs match pbrt-v4's analytic parameterization.

Hikari tessellates `Shape "sphere"` into a triangle mesh, so the UVs are ours to
get right; pbrt computes them analytically. They disagreed on `v`: ours ran 0 at
the low-z end to 1 at the high-z end, pbrt's runs the other way, so every
textured or bump-mapped sphere came out mirrored about the equator.

The trap is pbrt's naming. `thetaZMin` is `acos(min(zMin, zMax) / radius)`
(shapes.h:126) and `acos` is decreasing, so `thetaZMin` is the LARGER theta —
the bottom of the sphere, not the top. Reading those names geometrically gives
exactly `v = 1 - pbrt_v`, which is what shipped.

It cost 70% of one scene's error: `shadow_bumpgold_dome_over_velvet`, a
checkerboard bump at `uscale`/`vscale` 40 on a hemisphere, scored tile 0.0481
against a 0.07 limit and dropped to 0.0143 once `v` was flipped. The other four
sphere scenes did not move at all, because nothing in them reads `v` — which is
why a render-based test alone would not have located this. The assertions below
compare against pbrt's formula directly, per vertex.
"""

using Test, Hikari, GeometryBasics

# pbrt-v4 src/pbrt/shapes.h:126-127 (thetaZMin/thetaZMax) and :245 (v).
function pbrt_sphere_v(z, radius, zmin, zmax)
    theta_z_min = acos(clamp(min(zmin, zmax) / radius, -1, 1))
    theta_z_max = acos(clamp(max(zmin, zmax) / radius, -1, 1))
    theta       = acos(clamp(z / radius, -1, 1))
    return (theta - theta_z_min) / (theta_z_max - theta_z_min)
end

@testset "sphere UVs match pbrt" begin
    radius = 10.0f0

    # Full sphere, the lower hemisphere the dome scene carves out, the upper
    # one, and an asymmetric band that pins neither pole.
    @testset "zmin=$zmin zmax=$zmax" for (zmin, zmax) in [(-10.0f0, 10.0f0),
                                                          (-10.0f0,  0.0f0),
                                                          (  0.0f0, 10.0f0),
                                                          ( -5.0f0,  8.0f0)]
        mesh = Hikari.tessellate_sphere(radius; segments = 64,
                                        zmin = zmin, zmax = zmax, phimax_deg = 360.0f0)
        pos = GeometryBasics.coordinates(mesh)
        uv  = GeometryBasics.texturecoordinates(mesh)
        @test length(pos) == length(uv)

        worst = 0.0
        for (p, t) in zip(pos, uv)
            worst = max(worst, abs(t[2] - pbrt_sphere_v(p[3], radius, zmin, zmax)))
        end
        @test worst < 1e-4

        # v must span the full range, and increase with z. Both survive a
        # uniform flip, so they are a floor under the check above, not a
        # substitute for it.
        vs = [t[2] for t in uv]
        @test minimum(vs) ≈ 0 atol = 1e-4
        @test maximum(vs) ≈ 1 atol = 1e-4

        lo = argmin(p -> p[3], pos)
        hi = argmax(p -> p[3], pos)
        @test uv[findfirst(==(lo), pos)][2] < uv[findfirst(==(hi), pos)][2]
    end

    # The poles are pushed separately from the rings, so they can disagree with
    # them without any ring vertex noticing.
    @testset "poles" begin
        mesh = Hikari.tessellate_sphere(radius; segments = 64,
                                        zmin = -radius, zmax = radius, phimax_deg = 360.0f0)
        pos = GeometryBasics.coordinates(mesh)
        uv  = GeometryBasics.texturecoordinates(mesh)

        top    = findfirst(p -> p[3] ≈  radius, pos)
        bottom = findfirst(p -> p[3] ≈ -radius, pos)
        @test top !== nothing && bottom !== nothing
        @test uv[top][2]    ≈ 1 atol = 1e-6      # +z pole
        @test uv[bottom][2] ≈ 0 atol = 1e-6      # -z pole
    end
end
