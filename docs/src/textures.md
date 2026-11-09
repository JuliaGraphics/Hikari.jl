## Textures and Node Graphs

Any input of a material can vary over the surface. An image gives it a value at each uv coordinate.
A node graph computes it at every shading point, built from the same nodes as a Blender shader with
the same names, settings and defaults. `using Hikari.TextureNodes` brings them in scope.

```@setup textures
using GeometryBasics, LinearAlgebra, Hikari, Mantle
using Hikari.TextureNodes
backend = Mantle.defaultbackend()

sphere(x, r) = uv_normal_mesh(Tessellation(Sphere(Point3f(x, 0, r), r), 96))

function showcase(materials; resolution = (1200, 360), samples = 16)
    scene = Hikari.Scene(; backend)
    n = length(materials)
    for (i, material) in enumerate(materials)
        push!(scene, sphere(i - (n + 1) / 2, 0.42f0), material)
    end
    tiles = checker(xyz(); scale = 1, color1 = (0.5, 0.5, 0.5), color2 = (0.38, 0.38, 0.38))
    ground = normal_mesh(Rect3f(Vec3f(-50, -50, -0.01), Vec3f(100, 100, 0.01)))
    push!(scene, ground, Hikari.Diffuse(color = tiles.color))
    sky, sun = Hikari.sunsky_to_envlight(; direction = normalize(Vec3f(-0.6, -1, 0.9)), turbidity = 3f0)
    push!(scene, sky)
    push!(scene, sun)
    Hikari.sync!(scene)
    film = Hikari.Film(backend, Hikari.Film(Point2f(resolution...)))
    camera = Hikari.PerspectiveCamera(Point3f(0, -1.15n, 0.32n), Point3f(0, 0, 0.34), film;
                                      up = Vec3f(0, 0, 1), fov = 16)
    Hikari.clear!(film)
    integrator = Hikari.VolPath(; samples, max_depth = 16)
    integrator(scene, film, camera)
    Hikari.fill_aux_buffers!(film, scene, camera; has_infinite_lights = true)
    Hikari.denoise!(film; config = Hikari.OIDN())
    image = Array(Hikari.postprocess!(film; tonemap = :aces, gamma = 2.2))
    close(integrator)   # its render state takes gigabytes; free it now, not at the next GC
    return image
end
```

The renders on this page use the scene of the [Materials](@ref) page, with uv-mapped spheres.

### Images

A matrix of colours or numbers is an image: pass it as an input and it is sampled at the mesh's
uv coordinates, wrapping around. `image_texture(img, coords)` samples it at other coordinates, for
example tiled with `mapping`. Its outputs are the colour and `.alpha`.

```julia
using FileIO
img = load("planks.jpg")
Hikari.Principled(base_color = img, roughness = 0.4)
```

```@example textures
# Images are matrices of rows by columns, as Julia's images are.
pattern = [Hikari.RGBSpectrum(0.1f0 + 0.7f0 * (mod(y, 0.25f0) < 0.06f0), 0.15f0 + 0.5f0 * (mod(x, 0.5f0) < 0.1f0), 0.45f0)
           for y in range(0f0, 1f0; length = 256), x in range(0f0, 1f0; length = 128)]   # a tartan
stripes = [sinpi(8y) > 0 ? 0.55f0 : 0.05f0 for y in range(0f0, 1f0; length = 256), x in 1:4]
showcase([
    Hikari.Diffuse(color = pattern),                                                        # an image
    Hikari.Principled(base_color = image_texture(pattern, mapping(uv(); scale = (4, 2, 1))),
                      roughness = 0.3),                                                     # tiled
    Hikari.Glossy(color = (0.9, 0.9, 0.9), roughness = stripes),                           # roughness map
])
```

A mesh needs uv coordinates for this, for example from `GeometryBasics.uv_normal_mesh`.

### Node graphs

Each node is a function. Its inputs take numbers, colours or other nodes' outputs. Nodes with several
outputs return their first, and name the others: `noise(p).color`, `voronoi(p).position`,
`brick(p).fac`. Components are `.x`, `.y`, `.z` (or `.r`, `.g`, `.b`), up to three at once. Julia's
arithmetic works on nodes like Blender's Math and Vector Math nodes, so a graph reads as a formula:

```julia
using Hikari.TextureNodes
p = xyz()
rings = wave(p; type = :rings, scale = 2, distortion = 4, detail = 3).fac
wood = color_ramp(rings, 0.0 => (0.6, 0.35, 0.15), 0.8 => (0.45, 0.25, 0.1), 1.0 => (0.2, 0.1, 0.04))
Hikari.Principled(base_color = wood, roughness = 0.5 + 0.3 * rings)
```

| Kind | Nodes |
|------|-------|
| Coordinates | `uv()`, `xyz()` (world position), `normal()`, `texture_coordinate()` (`.generated`, `.object`, `.uv`, `.normal`, `.camera`, `.window`, `.reflection`), `attribute(name)` (a vertex attribute of the mesh) |
| Textures | `noise`, `voronoi`, `wave`, `magic`, `checker`, `brick`, `gradient`, `white_noise`, `image_texture` |
| Colour | `color_ramp`, `mix` (with Blender's blend modes), `hsv`, `gamma`, `bright_contrast`, `invert`, `rgb_to_bw`, `separate_color`, `combine_color` |
| Math | `+ - * / ^`, `min`, `max`, `abs`, `floor`, `sqrt`, `sin` and the rest, `clamp`, `map_range`, `smoothstep`, `fract`, `greater`, `less`, `math(op, ...)` for Blender's other operations |
| Vectors | `combine`, `mapping`, `vector_rotate`, `transform`, `dot`, `cross`, `norm`, `normalize`, `vector_math(op, ...)` |
| Normals | `bump`, `normal_map`, `radial_tangent` |

Every node's options are in the [API](@ref "Public API") reference.

```@example textures
wood = color_ramp(wave(xyz(); type = :rings, direction = :z, scale = 3, distortion = 6, detail = 3).fac,
                  0.0 => (0.55, 0.32, 0.15), 0.6 => (0.4, 0.22, 0.09), 1.0 => (0.2, 0.1, 0.04))
marble = color_ramp(wave(xyz(); scale = 2, distortion = 10, detail = 6).fac,
                    0.0 => (0.92, 0.92, 0.9), 0.8 => (0.85, 0.85, 0.83), 1.0 => (0.3, 0.3, 0.33))
wall = brick(mapping(xyz(); rotation = (π / 2, 0, 0)); scale = 4, mortar = (0.8, 0.8, 0.75),
             color1 = (0.6, 0.2, 0.1), color2 = (0.45, 0.16, 0.08))
showcase([
    Hikari.Diffuse(color = noise(xyz(); scale = 4, detail = 8).color),
    Hikari.Diffuse(color = voronoi(xyz(); scale = 6).color),
    Hikari.Diffuse(color = wood),
    Hikari.Diffuse(color = wall.color),
    Hikari.Diffuse(color = marble),
])
```

`xyz()` is the world position, so a texture built on it stays in place when an object moves through
it. `texture_coordinate().object` is the position in the mesh's own frame and moves with the mesh;
`texture_coordinate().generated` spans its bounding box from 0 to 1.

### Bump and normal maps

`bump(height; strength, distance)` turns a height graph into a shading normal for a material's
`normal` input; `distance` is the height of 1 in scene units. `normal_map(color; space, strength)`
reads a tangent-space or world-space normal map. Neither moves the geometry: silhouettes stay smooth.

```@example textures
showcase([
    Hikari.Diffuse(color = (0.8, 0.8, 0.78),
                   normal = bump(noise(xyz(); scale = 20, detail = 4).fac; distance = 0.02)),  # plaster
    Hikari.Metallic(base_color = (0.95, 0.64, 0.54), roughness = 0.15,
                    normal = bump(voronoi(xyz(); scale = 12); distance = 0.02)),              # hammered copper
    Hikari.Principled(base_color = wall.color, roughness = 0.8,
                      normal = bump(1 - wall.fac; distance = 0.02)),                         # mortar grooves
    Hikari.Glass(normal = bump(voronoi(xyz(); scale = 8); distance = 0.03)),                  # hammered glass
    Hikari.Principled(base_color = (0.25, 0.1, 0.05), roughness = 0.55,
                      normal = bump(voronoi(xyz(); scale = 40, feature = :distance_to_edge);
                                    distance = 0.005)),                                       # leather
])
```

`Principled`'s `coat_normal` takes its own bump, so a smooth varnish can sit over a rough base. The
pbrt models take a height as `displacement` instead, which they apply as pbrt does.

### Updating a graph

The values in a graph, such as a `scale` or a ramp's colours, are data: a material with the same
graph and other values, given to a mesh with `Hikari.update_material!`, renders without compiling
anything. A graph with other nodes compiles their kernel the first time it renders.
