# Materials as #47 checked them against Blender's Cycles: the Principled BSDF's inputs one at a
# time, then Blender's other BSDF nodes and their mixes and sums, in scenes of eight spheres on a
# grey floor under an area light, Z up, a 512² image. Shared by the test
# (`test_cycles_materials.jl`) and by `export_scene.jl`, which writes the scenes for
# `cycles_render.py` to render the references in Blender.
using GeometryBasics, LinearAlgebra
using GeometryBasics: Point3f, Vec3f, GLTriangleFace

const CYCLES_PRINCIPLED_CASES = [
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

"""A Blender BSDF node `node` with its `inputs` by socket name, as `cycles_render.py` builds it."""
bsdf(node; properties = (;), inputs...) =
    Dict("type" => "bsdf", "node" => node, "inputs" => Dict(String(k) => v for (k, v) in inputs),
          "properties" => Dict(String(k) => v for (k, v) in pairs(properties)))
mix_spec(a, b, factor) = Dict("type" => "mix", "a" => a, "b" => b, "factor" => factor)
add_spec(a, b) = Dict("type" => "add", "a" => a, "b" => b)

# Each case: the material in Hikari, built when called (the scene's Hikari module is the caller's), and
# the same in Blender. Anisotropic nodes are left to the Principled cases, where the tangent is wired.
diffuse_case(c, r) = (H -> H.Diffuse(color = c, roughness = r), bsdf("ShaderNodeBsdfDiffuse"; Color = collect(c), Roughness = r))
glossy_case(c, r) = (H -> H.Glossy(color = c, roughness = r), bsdf("ShaderNodeBsdfGlossy"; Color = collect(c), Roughness = r))
glass_case(c, r, ior) = (H -> H.Glass(color = c, roughness = r, ior = ior),
                         bsdf("ShaderNodeBsdfGlass"; Color = collect(c), Roughness = r, IOR = ior))
metallic_case(c, tint, r) = (H -> H.Metallic(base_color = c, edge_tint = tint, roughness = r),
                             bsdf("ShaderNodeBsdfMetallic"; var"Base Color" = collect(c), var"Edge Tint" = collect(tint), Roughness = r))
translucent_case(c) = (H -> H.Translucent(color = c), bsdf("ShaderNodeBsdfTranslucent"; Color = collect(c)))
sheen_case(c, r) = (H -> H.Sheen(color = c, roughness = r), bsdf("ShaderNodeBsdfSheen"; Color = collect(c), Roughness = r))
transparent_case(c) = (H -> H.Transparent(color = c), bsdf("ShaderNodeBsdfTransparent"; Color = collect(c)))
mix_case((ha, ba), (hb, bb), f) = (H -> H.Mix(ha(H), hb(H); factor = f), mix_spec(ba, bb, f))
add_case((ha, ba), (hb, bb)) = (H -> H.Add(ha(H), hb(H)), add_spec(ba, bb))

const CYCLES_NODE_CASES = [
    "diffuse_node" => diffuse_case((0.7, 0.5, 0.3), 0.0),
    "diffuse_node_rough" => diffuse_case((0.7, 0.5, 0.3), 0.8),
    "glossy_node" => glossy_case((0.8, 0.8, 0.8), 0.3),
    "glossy_node_tint" => glossy_case((0.9, 0.6, 0.3), 0.7),
    "glass_node" => glass_case((1.0, 1.0, 1.0), 0.2, 1.5),
    "metallic_node" => metallic_case((0.617, 0.577, 0.540), (0.695, 0.726, 0.770), 0.4),
    "translucent_node" => translucent_case((0.8, 0.6, 0.4)),
    "sheen_node" => sheen_case((0.8, 0.8, 0.8), 0.3),
    "transparent_node" => transparent_case((0.8, 0.9, 1.0)),
    "mix_diffuse_glossy" => mix_case(diffuse_case((0.8, 0.2, 0.2), 0.0), glossy_case((0.8, 0.8, 0.8), 0.2), 0.3),
    "add_diffuse_sheen" => add_case(diffuse_case((0.3, 0.3, 0.6), 0.0), sheen_case((0.8, 0.8, 0.8), 0.4)),
    "mix_glass_diffuse" => mix_case(glass_case((1.0, 1.0, 1.0), 0.0, 1.45), diffuse_case((0.6, 0.6, 0.6), 0.0), 0.5),
    "mix_metallic_glossy" => mix_case(metallic_case((0.9, 0.6, 0.2), (0.95, 0.9, 0.8), 0.3), glossy_case((0.5, 0.5, 0.9), 0.1), 0.5),
    "add_glossy_glossy" => add_case(glossy_case((0.4, 0.2, 0.1), 0.05), glossy_case((0.1, 0.2, 0.4), 0.6)),
    "metallic_rough" => metallic_case((0.9, 0.4, 0.3), (0.95, 0.8, 0.7), 0.8),
    "glass_node_tint" => glass_case((0.6, 0.8, 1.0), 0.0, 1.33),
]

# The Principled BSDF's Thin Wall: transmission and subsurface of a surface of no thickness. After the
# others, so every earlier case keeps its sphere and its reference.
const CYCLES_THIN_WALL_CASES = [
    "thin_wall_glass" => (base_color = (1.0, 1.0, 1.0), transmission_weight = 1.0, roughness = 0.1, ior = 1.45,
                          thin_wall = true),
    "thin_wall_glass_color" => (base_color = (0.8, 0.3, 0.2), transmission_weight = 1.0, roughness = 0.1, ior = 1.45,
                                thin_wall = true),
    "thin_wall_sss" => (base_color = (0.9, 0.7, 0.6), roughness = 0.4, subsurface_weight = 1.0,
                        subsurface_radius = (1.0, 0.2, 0.1), subsurface_scale = 0.1, thin_wall = true),
]

const CYCLES_FILM_NODE_CASES = [
    "glass_node_film" => (H -> H.Glass(roughness = 0.2, thin_film_thickness = 400, thin_film_ior = 1.33),
        bsdf("ShaderNodeBsdfGlass"; Roughness = 0.2, var"Thin Film Thickness" = 400, var"Thin Film IOR" = 1.33)),
    "glass_node_film_rough" => (H -> H.Glass(roughness = 0.5, thin_film_thickness = 550, thin_film_ior = 1.5),
        bsdf("ShaderNodeBsdfGlass"; Roughness = 0.5, var"Thin Film Thickness" = 550, var"Thin Film IOR" = 1.5)),
    "metallic_node_film" => (H -> H.Metallic(roughness = 0.3, thin_film_thickness = 350, thin_film_ior = 1.33),
        bsdf("ShaderNodeBsdfMetallic"; Roughness = 0.3, var"Thin Film Thickness" = 350, var"Thin Film IOR" = 1.33)),
    "metallic_node_film_tint" => (H -> H.Metallic(base_color = (0.9, 0.5, 0.2), edge_tint = (0.95, 0.8, 0.7),
                                                roughness = 0.5, thin_film_thickness = 500, thin_film_ior = 1.5),
        bsdf("ShaderNodeBsdfMetallic"; var"Base Color" = [0.9, 0.5, 0.2], var"Edge Tint" = [0.95, 0.8, 0.7],
              Roughness = 0.5, var"Thin Film Thickness" = 500, var"Thin Film IOR" = 1.5)),
]

const CYCLES_TOON_NODE_CASES = [
    "toon_diffuse" => (H -> H.Toon(), bsdf("ShaderNodeBsdfToon")),
    "toon_diffuse_smooth" => (H -> H.Toon(color = (0.2, 0.5, 0.8), size = 0.3, smooth = 0.2),
        bsdf("ShaderNodeBsdfToon"; Color = [0.2, 0.5, 0.8], Size = 0.3, Smooth = 0.2)),
    "toon_glossy" => (H -> H.Toon(component = :glossy, size = 0.15),
        bsdf("ShaderNodeBsdfToon"; properties = (component = "GLOSSY",), Size = 0.15)),
    "toon_glossy_smooth" => (H -> H.Toon(color = (0.7, 0.7, 0.7), component = :glossy, size = 0.4, smooth = 0.2),
        bsdf("ShaderNodeBsdfToon"; properties = (component = "GLOSSY",), Color = [0.7, 0.7, 0.7], Size = 0.4, Smooth = 0.2)),
]

const CYCLES_SUBSURFACE_MIX_CASES = let
    a = (base_color = (0.8, 0.8, 0.8), subsurface_weight = 1, subsurface_scale = 0.15,
         subsurface_radius = (1, 1, 1), roughness = 0.4)
    b = (base_color = (0.4, 0.4, 0.4), subsurface_weight = 1, subsurface_scale = 0.03,
         subsurface_radius = (1, 1, 1), roughness = 0.4)
    spec(inputs) = Dict("type" => "principled", "inputs" => Dict(pairs(inputs)))
    [
        "mix_subsurface_nodes" => (H -> H.Mix(H.Principled(; a...), H.Principled(; b...); factor = 0.35),
            mix_spec(spec(a), spec(b), 0.35)),
        "add_subsurface_nodes" => (H -> H.Add(H.Principled(; a...), H.Principled(; b...)),
            add_spec(spec(a), spec(b))),
    ]
end

principled_case((name, inputs)) =
    name => (H -> H.Principled(; inputs...), Dict("type" => "principled", "inputs" => Dict(pairs(inputs))))

"""Every case: its name, its material in Hikari (a function of the Hikari module) and the same in Blender."""
const CYCLES_STANDARD_CASES = [
    map(principled_case, CYCLES_PRINCIPLED_CASES);
    CYCLES_NODE_CASES;
    map(principled_case, CYCLES_THIN_WALL_CASES)
]
const CYCLES_CASES = [CYCLES_STANDARD_CASES; CYCLES_FILM_NODE_CASES; CYCLES_TOON_NODE_CASES; CYCLES_SUBSURFACE_MIX_CASES]
# Keep existing reference scenes unchanged: filling their last batch would add
# interreflections from new spheres and change the old comparison values.
cycles_batches() = [collect(Iterators.partition(CYCLES_STANDARD_CASES, 8));
                    collect(Iterators.partition(CYCLES_FILM_NODE_CASES, 8));
                    collect(Iterators.partition(CYCLES_TOON_NODE_CASES, 8));
                    collect(Iterators.partition(CYCLES_SUBSURFACE_MIX_CASES, 8))]

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
