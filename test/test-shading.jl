using GeometryBasics
using Hikari
using Hikari: RGBSpectrum, MaterialIndex, shade_material, reconstruct_surface_interaction, intersect_ray_inner
using Raycore: RayDifferentials
using LinearAlgebra

# Assuming scene is already created from scene-hikari.jl:
# scene, camera, film = create_scene()

# Get the BVH and materials from the scene
bvh = scene.aggregate.bvh
materials = scene.aggregate.materials

# Create a ray (e.g., pointing at blue sphere)
ray_origin = Point3f(0.15, 0.35, -1.6)
ray_dir = normalize(Vec3f(0.2, 0.11, -2.6) - Vec3f(ray_origin))
ray = RayDifferentials(o=ray_origin, d=ray_dir)

# Intersect ray with scene
hit_found, mat_idx, hit_p, hit_n, hit_wo, hit_uv, shading_n, shading_dpdu, shading_dpdv = intersect_ray_inner(bvh, ray)

# Reconstruct surface interaction
si = reconstruct_surface_interaction(
    hit_p, hit_n, hit_wo, hit_uv,
    shading_n, shading_dpdu, shading_dpdv, ray.time
)

# Call shade_material
beta = RGBSpectrum(1f0)  # Initial throughput
depth = Int32(0)
max_depth = Int32(5)

@code_warntype shade_material(materials, mat_idx, ray, si, scene, beta, depth, max_depth)
# => RGBSpectrum(Float32[0.069, 0.107, 0.234])

material = Hikari.get_material(materials, mat_idx)
@code_warntype Hikari.shade(material, ray, si, scene, beta, depth, max_depth)

# Test specular_bounce
bsdf = material(si, true, Hikari.Radiance)
@code_warntype Hikari.specular_bounce(Hikari.Reflect(), bsdf, ray, si, scene, beta, depth, max_depth)
@code_warntype Hikari.specular_bounce(Hikari.Transmit(), bsdf, ray, si, scene, beta, depth, max_depth)
