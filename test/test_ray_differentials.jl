# Screen-space differentials at a hit point.
#
# pbrt-v4 `SurfaceInteraction::ComputeDifferentials` uses the ray's TRUE
# differentials on camera rays and only approximates deeper. Hikari approximated
# EVERYWHERE, which matters far beyond mipmap selection: the footprint sets
# `δu = 0.5(|dudx|+|dudy|)`, the finite-difference step `perturb_bump_frame` uses
# for `∂h/∂u`. A wrong footprint tilts the bumped normal, which MOVES specular
# highlights instead of merely dimming them. On crown.pbrt that showed up as a
# coherent dipole over the displaced dome panels;
# `shadow_bumpgold_dome_over_velvet` scored tile 0.0699 against a 0.07 gate and
# dropped to 0.0499 once this was right.
#
# The ground truth here is not another formula — it is literally generating the
# neighbouring pixel's ray and differencing where the two land. That is what a
# differential IS, so this test cannot drift from the definition.
#
# It also pins the constant that is easy to get wrong: `dx_camera` and
# `dy_camera` are offsets on the plane `raster_to_camera` maps the film onto, and
# that plane is at 2x the camera's `near`, not `near` itself. Reconstructing a
# camera-space point on the wrong plane made every differential exactly 2x too
# large, which rendered visibly worse (tile 0.0699 -> 0.193).
using Test
using Hikari
using LinearAlgebra
using GeometryBasics: Point2f, Point3f, Vec3f

const RES = Point2f(256, 256)
const SPP = Int32(256)

function make_camera(; lens_radius = 0f0, focal_distance = 1f6, near = 0.01f0)
    film = Hikari.Film(RES)
    window = Hikari.Bounds2(Point2f(-1, -1), Point2f(1, 1))
    view = Hikari.look_at(Point3f(0, -1.2, 0.6), Point3f(0, 0, 0.5), Vec3f(0, 0, 1))
    return Hikari.PerspectiveCamera(view, window, 0f0, 1f0, lens_radius, focal_distance,
                                    40f0, film; near)
end

sample_at(px, py; lens = Point2f(0.5f0, 0.5f0)) = Hikari.CameraSample(Point2f(px, py), lens, 0f0, 1f0)

"""Where a ray meets the plane through `p0` with normal `n`."""
function plane_hit(ray, n, p0)
    t = dot(n, Vec3f(p0 - ray.o)) / dot(n, ray.d)
    return Point3f(ray.o + ray.d * t)
end

@testset "raster plane is 2x near, and flat" begin
    # Checked at more than one `near` so this pins the RELATIONSHIP rather than a
    # value that happens to hold at the default. `near` is a camera keyword, not
    # a module constant — a global invited `approximate_dp_dxy`, in a different
    # subsystem, to assume it described this plane. It does not.
    for near in (0.01f0, 0.05f0)
        cam = make_camera(; near)
        z = Hikari.raster_plane_z(cam)
        @test z ≈ -2 * near rtol = 1e-3
        # Constant across the film — the reconstruction relies on this.
        for r in (Point3f(0, 0, 0), Point3f(255, 255, 0), Point3f(10, 200, 0))
            @test Hikari.get_raster_to_camera(cam)(r)[3] ≈ z rtol = 1e-5
        end
    end
end

@testset "differentials are independent of near" begin
    # `near` only shapes the projection; it must not change where neighbouring
    # pixels actually land. If a footprint is ever derived from `near` instead of
    # `raster_plane_z` again, these two disagree by exactly the ratio.
    n = normalize(Vec3f(0.2, -0.9, 0.35))
    p0 = Point3f(0, 0, 0.5)
    dp = map((0.01f0, 0.05f0)) do near
        cam = make_camera(; near)
        r0, _ = Hikari.generate_ray(cam, sample_at(128.5f0, 128.5f0))
        Hikari.surface_dp_dxy(cam, Point3f(r0.o), Vec3f(r0.d),
                              plane_hit(r0, n, p0), n, SPP, Int32(0))[1]
    end
    @test norm(dp[1]) / norm(dp[2]) ≈ 1 atol = 1e-2
end

@testset "differentials match neighbouring-pixel rays" begin
    n = normalize(Vec3f(0.2, -0.9, 0.35))
    p0 = Point3f(0, 0, 0.5)
    s = max(0.125f0, 1f0 / sqrt(Float32(SPP)))   # RayDifferential::ScaleDifferentials

    for (label, lens_radius, focal_distance, lens_uv) in
            (("pinhole", 0f0, 1f6, Point2f(0.5f0, 0.5f0)),
             ("thin lens", 0.1f0, 1.2f0, Point2f(0.3f0, 0.7f0)))
        cam = make_camera(; lens_radius, focal_distance)
        r0, _ = Hikari.generate_ray(cam, sample_at(128.5f0, 128.5f0; lens = lens_uv))
        rx, _ = Hikari.generate_ray(cam, sample_at(129.5f0, 128.5f0; lens = lens_uv))
        ry, _ = Hikari.generate_ray(cam, sample_at(128.5f0, 129.5f0; lens = lens_uv))

        h0 = plane_hit(r0, n, p0)
        truth_x = s .* (plane_hit(rx, n, p0) - h0)
        truth_y = s .* (plane_hit(ry, n, p0) - h0)

        dpdx, dpdy = Hikari.surface_dp_dxy(cam, Point3f(r0.o), Vec3f(r0.d), h0, n, SPP, Int32(0))
        @testset "$label" begin
            @test dpdx ≈ truth_x rtol = 5e-3
            @test dpdy ≈ truth_y rtol = 5e-3
            # Magnitude specifically — the 2x plane bug preserved DIRECTION and
            # only broke scale, so a direction-only check would have passed it.
            @test norm(dpdx) / norm(truth_x) ≈ 1 atol = 5e-3
            @test norm(dpdy) / norm(truth_y) ≈ 1 atol = 5e-3
        end
    end
end

@testset "indirect bounces keep the approximation" begin
    # pbrt does not propagate differentials through indirect rays, so anything
    # past the camera ray must take the fallback — identically, not approximately.
    cam = make_camera()
    n = normalize(Vec3f(0.2, -0.9, 0.35))
    r0, _ = Hikari.generate_ray(cam, sample_at(128.5f0, 128.5f0))
    h0 = plane_hit(r0, n, Point3f(0, 0, 0.5))
    approx = Hikari.approximate_dp_dxy(h0, n, cam, SPP)
    for depth in Int32.((1, 2, 7))
        @test Hikari.surface_dp_dxy(cam, Point3f(r0.o), Vec3f(r0.d), h0, n, SPP, depth) == approx
    end
    @test Hikari.surface_dp_dxy(cam, Point3f(r0.o), Vec3f(r0.d), h0, n, SPP, Int32(0)) != approx
end

@testset "approximate_dp_dxy is correctly scaled" begin
    # It divided by the camera's near distance while the raster plane sits at 2x
    # that, so every footprint in the renderer was 2x too wide. For a face-on
    # surface the approximation should now agree with the true differential;
    # oblique normals legitimately diverge because the tangent-plane projection
    # cannot represent an elongated footprint.
    cam = make_camera()
    p0 = Point3f(0, 0, 0.5)
    for n in (Vec3f(0, -1, 0), Vec3f(0, 0, 1))
        r0, _ = Hikari.generate_ray(cam, sample_at(128.5f0, 128.5f0))
        dot(n, r0.d) == 0 && continue
        h0 = plane_hit(r0, n, p0)
        truth, _ = Hikari.surface_dp_dxy(cam, Point3f(r0.o), Vec3f(r0.d), h0, n, SPP, Int32(0))
        approx, _ = Hikari.approximate_dp_dxy(h0, n, cam, SPP)
        @test norm(approx) / norm(truth) ≈ 1 atol = 0.05
    end
end
