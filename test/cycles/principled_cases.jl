# The Principled BSDF's inputs one at a time, as #47 checked them against Blender's Cycles: five
# scenes of eight spheres on a grey floor under an area light, Z up, a 512² image. Shared by the
# test (`test_cycles_principled.jl`) and by `export_scene.jl`, which writes the scenes for
# `cycles_render.py` to render the references in Blender.
using GeometryBasics, LinearAlgebra
using GeometryBasics: Point3f, Vec3f, GLTriangleFace

const CYCLES_CASES = [
    # base
    "default" => (;), "base_red" => (base_color = (0.8, 0.1, 0.1),), "rough0" => (roughness = 0.0,),
    "rough1" => (roughness = 1.0,), "diffuse_rough" => (diffuse_roughness = 1.0,), "ior_low" => (ior = 1.1,),
    "ior_high" => (ior = 2.5,), "alpha" => (alpha = 0.5,),
    # metal and specular
    "metal" => (base_color = (0.9, 0.6, 0.2), metallic = 1.0, roughness = 0.3),
    "metal_rough" => (base_color = (0.9, 0.6, 0.2), metallic = 1.0, roughness = 0.7),
    "metal_tint" => (base_color = (0.9, 0.6, 0.2), metallic = 1.0, roughness = 0.3, specular_tint = (0.3, 0.9, 0.3)),
    "aniso" => (base_color = (0.8, 0.8, 0.8), metallic = 1.0, roughness = 0.4, anisotropic = 0.8),
    "aniso_rot" => (base_color = (0.8, 0.8, 0.8), metallic = 1.0, roughness = 0.4, anisotropic = 0.8, anisotropic_rotation = 0.25),
    "spec0" => (specular_ior_level = 0.0,), "spec1" => (specular_ior_level = 1.0,),
    "spec_tint" => (specular_tint = (1.0, 0.3, 0.3),),
    # transmission and coat
    "glass" => (base_color = (1.0, 1.0, 1.0), transmission_weight = 1.0, roughness = 0.0, ior = 1.45),
    "glass_rough" => (base_color = (1.0, 1.0, 1.0), transmission_weight = 1.0, roughness = 0.3, ior = 1.45),
    "glass_color" => (base_color = (0.8, 0.3, 0.2), transmission_weight = 1.0, roughness = 0.0, ior = 1.45),
    "glass_gray" => (base_color = (0.6, 0.6, 0.6), transmission_weight = 1.0, roughness = 0.0, ior = 1.45),
    "half_transmission" => (base_color = (0.8, 0.8, 0.8), transmission_weight = 0.5, roughness = 0.2),
    "coat" => (base_color = (0.1, 0.3, 0.8), roughness = 0.6, coat_weight = 1.0, coat_roughness = 0.05),
    "coat_rough" => (base_color = (0.1, 0.3, 0.8), roughness = 0.6, coat_weight = 1.0, coat_roughness = 0.5),
    "coat_tint" => (base_color = (0.8, 0.8, 0.8), roughness = 0.6, coat_weight = 1.0, coat_tint = (0.9, 0.5, 0.2)),
    # coat, sheen, emission, thin film
    "coat_ior" => (base_color = (0.1, 0.3, 0.8), roughness = 0.6, coat_weight = 1.0, coat_ior = 2.0),
    "coat_metal" => (base_color = (0.9, 0.6, 0.2), metallic = 1.0, roughness = 0.5, coat_weight = 1.0),
    "sheen" => (base_color = (0.2, 0.6, 0.2), roughness = 0.8, sheen_weight = 1.0),
    "sheen_rough" => (base_color = (0.2, 0.6, 0.2), roughness = 0.8, sheen_weight = 1.0, sheen_roughness = 0.1),
    "sheen_tint" => (base_color = (0.2, 0.6, 0.2), roughness = 0.8, sheen_weight = 1.0, sheen_tint = (1.0, 0.3, 0.3)),
    "emission" => (emission_color = (1.0, 0.5, 0.2), emission_strength = 1.0),
    "thin_film" => (base_color = (0.5, 0.5, 0.5), roughness = 0.2, thin_film_thickness = 500.0, thin_film_ior = 1.33),
    "thin_film_metal" => (base_color = (0.8, 0.8, 0.8), metallic = 1.0, roughness = 0.2, thin_film_thickness = 400.0,
                          thin_film_ior = 1.5),
    # subsurface (Hikari's random walk takes the specular IOR, as Cycles' Random Walk does)
    "sss" => (base_color = (0.9, 0.7, 0.6), roughness = 0.4, subsurface_weight = 1.0, subsurface_radius = (1.0, 0.2, 0.1),
              subsurface_scale = 0.1),
    "sss_gray" => (base_color = (0.8, 0.8, 0.8), roughness = 0.4, subsurface_weight = 1.0, subsurface_radius = (1.0, 1.0, 1.0),
                   subsurface_scale = 0.1),
    "sss_aniso" => (base_color = (0.8, 0.8, 0.8), roughness = 0.4, subsurface_weight = 1.0, subsurface_radius = (1.0, 1.0, 1.0),
                    subsurface_scale = 0.1, subsurface_anisotropy = 0.8),
    "sss_ior" => (base_color = (0.8, 0.8, 0.8), roughness = 0.4, subsurface_weight = 1.0, subsurface_radius = (1.0, 1.0, 1.0),
                  subsurface_scale = 0.1, subsurface_ior = 1.6),
    "sss_scale" => (base_color = (0.9, 0.7, 0.6), roughness = 0.4, subsurface_weight = 1.0, subsurface_radius = (1.0, 0.2, 0.1),
                    subsurface_scale = 0.5),
    "sss_half" => (base_color = (0.9, 0.7, 0.6), roughness = 0.4, subsurface_weight = 0.5, subsurface_radius = (1.0, 0.2, 0.1),
                   subsurface_scale = 0.1),
    "glass_spec" => (base_color = (1.0, 1.0, 1.0), transmission_weight = 1.0, roughness = 0.1, specular_ior_level = 1.0),
    "rough_spec" => (base_color = (0.5, 0.5, 0.8), roughness = 0.2, specular_ior_level = 1.0),
]
cycles_batches() = collect(Iterators.partition(CYCLES_CASES, 8))

const CYCLES_CENTERS = [Point3f(x, y, 0.5f0) for y in (1.4f0, 0f0) for x in (-2.25f0, -0.75f0, 0.75f0, 2.25f0)]
const CYCLES_SCENE = (radius = 0.5f0, res = 512, world = 0.1f0, max_depth = 8,
                      camera = (position = Point3f(0, -6.5, 3.2), lookat = Point3f(0, 0.5, 0.45), up = Vec3f(0, 0, 1), fov = 40f0),
                      light = (center = Point3f(0, 0, 4.5), size = 3f0, strength = 3f0))

cycles_sphere(c) = normal_mesh(Tesselation(Sphere(c, CYCLES_SCENE.radius), 128))
function cycles_quad(c, s, z_normal)
    h = s / 2
    ps = Point3f[c + Vec3f(-h, -h, 0), c + Vec3f(h, -h, 0), c + Vec3f(h, h, 0), c + Vec3f(-h, h, 0)]
    fs = z_normal > 0 ? GLTriangleFace[(1, 2, 3), (1, 3, 4)] : GLTriangleFace[(1, 3, 2), (1, 4, 3)]
    return GeometryBasics.Mesh(ps, fs; normal = fill(Vec3f(0, 0, z_normal), 4))
end
cycles_ground() = cycles_quad(Point3f(0, 0, 0), 20f0, 1)
cycles_light() = cycles_quad(CYCLES_SCENE.light.center, CYCLES_SCENE.light.size, -1)

"""
The pixels `(row, column)` the mean of sphere `k` is taken over: inside 0.8 of its silhouette, and
of the back row only the upper half, which the front row does not cover.
"""
function cycles_sphere_pixels(k)
    cam = CYCLES_SCENE.camera
    c = CYCLES_CENTERS[k]
    f = normalize(cam.lookat - cam.position)
    r = normalize(cross(f, cam.up))
    u = cross(r, f)
    t = tan(deg2rad(cam.fov) / 2)
    d = c - cam.position
    cx, cy = dot(d, r) / (dot(d, f) * t), dot(d, u) / (dot(d, f) * t)
    rad = tan(asin(CYCLES_SCENE.radius / norm(d))) / t * 0.8
    res = CYCLES_SCENE.res
    pix = [(row, col) for row in 1:res, col in 1:res
           if ((col - 0.5) / res * 2 - 1 - cx)^2 + (1 - (row - 0.5) / res * 2 - cy)^2 <= rad^2]
    k > 4 && return pix                  # the front row: CYCLES_CENTERS has the back row (y = 1.4) first
    middle = sum(first, pix) / length(pix)
    return filter(p -> p[1] < middle, pix)
end

"""The mean colour of `img` over `pixels`, as `(r, g, b)`."""
cycles_mean(img, pixels) =
    Tuple(sum(p -> Float64(getfield(img[p...], ch)), pixels) / length(pixels) for ch in (:r, :g, :b))
