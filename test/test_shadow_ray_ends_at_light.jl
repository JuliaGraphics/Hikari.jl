# A shadow ray ends at the point on the light it tests, whatever point it leaves from.
#
# On a smooth-shaded triangle shadow rays leave from `vp_shadow_point`, off the hit
# along the vertex normals, but the light was sampled from the hit. The ray kept the
# hit's direction to the light, so it ran parallel to the segment it should test and
# ended beside the light point, by the offset's part across that direction. Where the
# light faces the hit at an angle, that end lies past the light's own surface, the ray
# hits the light and counts it as a blocker: on a tessellated sphere under an area
# light every triangle went dark towards its centre, where the offset is largest, and
# the scene lost 2 % of pbrt's energy at `maxdepth 1`. pbrt aims it at the light point
# from where it leaves (`SpawnRayTo`).

using Test, Hikari, GeometryBasics
using LinearAlgebra: norm, normalize

@testset "a shadow ray ends at the light point" begin
    hit = Point3f(0, 0, 0)
    ng = Vec3f(0, 0, 1)
    p_light = Point3f(0.3, 0.1, 0.5)
    wi = normalize(Vec3f(p_light - hit))
    lambda = Hikari.sample_wavelengths_visible(0.3f0)
    white = Hikari.SpectralRadiance(1f0)
    ls = Hikari.LightSampleSpectral(white, wi, 1f0, p_light, false)
    for shadow_point in (hit, Point3f(0.004, -0.003, 0.01), Point3f(-0.01, 0.02, 0.005))
        r = Hikari.compute_direct_lighting_spectral(shadow_point, ng, ng, ng, white, white, lambda, ls,
                                                    Hikari.SpectralRadiance(0.3f0), 0.3f0)
        @test r.valid
        @test norm(r.ray_direction) ≈ 1
        # The ray stops 1e-3 short of the light, so the light itself never blocks it.
        @test norm(r.ray_origin + r.ray_direction * (r.t_max + 1f-3) - p_light) < 1f-5
    end
end
