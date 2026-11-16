# Materials

A material describes how light scatters at a surface. Hikari's materials are Blender's shader
nodes, with Blender's names, inputs and defaults, plus the material models of
[pbrt-v4](https://pbrt.org) that its scenes use. Give one to every mesh you push:

```julia
handle = push!(scene, mesh, Hikari.Principled(base_color = (0.8, 0.1, 0.1), roughness = 0.3))
```

Every numeric or colour input of a material takes any of these:

| Value | Meaning |
|:------|:--------|
| `0.5` | a constant |
| `(0.8, 0.2, 0.1)`, `RGB(0.8, 0.2, 0.1)`, `Hikari.RGBSpectrum(0.8f0, 0.2f0, 0.1f0)` | a constant linear RGB colour |
| a matrix of colours or numbers, or a `Hikari.Texture` | an image, sampled at the mesh's uv coordinates |
| a node graph from `Hikari.TextureNodes` | a procedural input, see [Textures and Node Graphs](@ref) |

Settings that choose a different scattering model, such as `distribution`, `thin_wall` or
`fresnel_type`, are constants.

## The scene behind the pictures

The renders on this page put materials on balls with the Makie logo engraved, under a sun and sky,
at 256 samples per pixel, denoised with OIDN (see [Postprocessing and Denoising](@ref)):

```@example materials
using Main: docs_image, docs_samples, logo_ball_path # hide
using GeometryBasics, LinearAlgebra, Hikari, Mantle, FileIO
using Hikari.TextureNodes
backend = Mantle.defaultbackend()

const logo_ball = load(logo_ball_path())

function ball(center, r)
    positions = [center + r * p for p in coordinates(logo_ball)]
    return GeometryBasics.Mesh(positions, faces(logo_ball); normal = normals(positions, faces(logo_ball)))
end

function showcase(materials; resolution = (1200, 360), samples = 256)
    samples = docs_samples(samples) # hide
    scene = Hikari.Scene(; backend)
    n = length(materials)
    for (i, material) in enumerate(materials)
        push!(scene, ball(Point3f(i - (n + 1) / 2, 0, 0.42), 0.42f0), material)
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
nothing # hide
```

The first render with a new kind of BSDF or node compiles a shading kernel for it, which can take a
minute. Rendering it again, or with other values, compiles nothing.

## Principled BSDF

`Principled` is the material to start from. One shader covers plastic, metal, glass, skin, cloth,
paint and anything layered from them, arranged as in Cycles: sheen and coat over the metal, and
transmission and the specular layer over the diffuse base.

| Group | Inputs | For |
|:------|:-------|:----|
| Base | `base_color`, `metallic`, `roughness`, `ior`, `alpha`, `normal`, `diffuse_roughness` | every material; `metallic = 1` is a metal |
| Specular | `specular_ior_level`, `specular_tint`, `anisotropic`, `anisotropic_rotation`, `tangent` | the strength of reflections, brushed metal |
| Subsurface | `subsurface_weight`, `subsurface_radius`, `subsurface_scale`, `subsurface_ior`, `subsurface_anisotropy` | skin, wax, marble, soap |
| Transmission | `transmission_weight`, `transmission_dispersion_scale`, `transmission_dispersion_abbe_number` | glass, water, gems |
| Coat | `coat_weight`, `coat_roughness`, `coat_ior`, `coat_tint`, `coat_normal` | car paint, varnish, lacquer |
| Sheen | `sheen_weight`, `sheen_roughness`, `sheen_tint` | velvet, cloth, dust |
| Emission | `emission_color`, `emission_strength` | screens, LEDs, glowing paint |
| Thin film | `thin_film_thickness` (nm), `thin_film_ior` | soap bubbles, oil slicks, anodised metal |

`distribution` is `:multi_ggx` (Blender's default), `:ggx` or `:beckmann`; the latter two scatter
once and lose energy on rough surfaces. `thin_wall = true` makes transmission and subsurface those of
a surface with no inside, such as a soap bubble or a leaf.

```@example materials
docs_image("materials-principled-bsdf", # hide
showcase([
    Hikari.Principled(base_color = (0.8, 0.08, 0.05), roughness = 0.25),                    # plastic
    Hikari.Principled(base_color = (0.9, 0.9, 0.9), metallic = 1, roughness = 0.3,
                      anisotropic = 0.8),                                                    # brushed steel
    Hikari.Principled(base_color = (0.02, 0.1, 0.45), metallic = 0.6, roughness = 0.4,
                      coat_weight = 1, coat_roughness = 0.02),                               # car paint
    Hikari.Principled(base_color = (0.25, 0.01, 0.08), roughness = 1, sheen_weight = 1,
                      sheen_roughness = 0.3, sheen_tint = (1, 0.6, 0.8)),                    # velvet
    Hikari.Principled(base_color = (0.9, 0.7, 0.55), roughness = 0.4, subsurface_weight = 1,
                      subsurface_radius = (1, 0.4, 0.2), subsurface_scale = 0.15),           # wax
])
) # hide
```

Without a `tangent`, anisotropic reflections run around the world's Z axis, as Blender's do around
the object's.

## Glass

`Hikari.Glass(...)` is `Principled(transmission_weight = 1, ...)`: the same material, with Principled's
`base_color`, `roughness`, `ior`, thin film, dispersion and `thin_wall`.

```@example materials
docs_image("materials-glass", # hide
showcase([
    Hikari.Glass(),                                                    # clear
    Hikari.Glass(roughness = 0.25),                                    # frosted
    Hikari.Glass(base_color = (0.35, 0.75, 0.5)),                      # tinted
    Hikari.Glass(ior = 2.4, transmission_dispersion_scale = 1),        # diamond, with dispersion
    Hikari.Glass(ior = 1.33, thin_wall = true,
                 thin_film_thickness = 450, thin_film_ior = 1.33),     # soap bubble
])
) # hide
```

Common indices of refraction are 1.33 for water, 1.5 for window glass and 2.42 for diamond.
`base_color` tints the light that passes through, by its square root at each surface, so light that
enters and leaves takes it once; `specular_tint` tints the reflection, as in Blender's Principled BSDF. For colour that deepens with
the distance light travels inside, fill the glass with an absorbing medium (see [Volumes](@ref)).

## Blender's other BSDFs

| Material | Scattering | Main inputs |
|:---------|:-----------|:------------|
| `Diffuse` | Lambert, or Oren-Nayar with `roughness` | `color`, `roughness`, `normal` |
| `Glossy` | microfacet reflection, a mirror at `roughness = 0` | `color`, `roughness`, `anisotropy`, `rotation` |
| `Metallic` | a metal by its colour at normal and grazing incidence (F82), or by complex index | `base_color`, `edge_tint`, `roughness`, `fresnel_type`, `ior`, `extinction`, thin film |
| `Refraction` | transmission without reflection | `color`, `roughness`, `ior` |
| `Translucent` | diffuse transmission | `color` |
| `Sheen` | the microfiber reflection of cloth, bright at grazing angles | `color`, `roughness` |
| `Toon` | Blender's cel shading | `color`, `size`, `smooth`, `component` |
| `SubsurfaceScattering` | random-walk subsurface scattering | `color`, `scale`, `radius`, `ior`, `anisotropy` |
| `Transparent` | light passing straight through, tinted | `color` |
| `Emission` | light emitted by the surface | `color`, `strength`, `two_sided` |

```@example materials
docs_image("materials-blender-s-other-bsdfs", # hide
showcase([
    Hikari.Diffuse(color = (0.75, 0.75, 0.72), roughness = 1),                       # chalk
    Hikari.Glossy(color = (0.9, 0.9, 0.9), roughness = 0.1),                         # polished chrome
    Hikari.Metallic(base_color = (0.95, 0.64, 0.54), edge_tint = (0.99, 0.82, 0.75),
                    roughness = 0.25),                                                # copper
    Hikari.Metallic(base_color = (0.6, 0.6, 0.62), roughness = 0.15,
                    thin_film_thickness = 380, thin_film_ior = 2.0),                  # anodised titanium
    Hikari.Refraction(color = (0.8, 0.9, 1.0), roughness = 0.05, ior = 1.45),
])
) # hide
```

```@example materials
docs_image("materials-blender-s-other-bsdfs-2", # hide
showcase([
    Hikari.Translucent(color = (0.5, 0.8, 0.3)),                                      # leaf
    Hikari.Add(Hikari.Diffuse(color = (0.12, 0.01, 0.04)),
               Hikari.Sheen(color = (1, 0.55, 0.75), roughness = 0.3)),              # velvet
    Hikari.Toon(color = (0.2, 0.5, 0.9), size = 0.4, smooth = 0.05),
    Hikari.SubsurfaceScattering(color = (0.9, 0.55, 0.4), scale = 0.2),               # skin
    Hikari.Mix(Hikari.Diffuse(color = (0.9, 0.3, 0.1)), Hikari.Transparent();
               factor = 0.6),                                                         # ghost
])
) # hide
```

`Metallic` with `fresnel_type = :physical_conductor` takes the complex index of refraction
`ior` + i`extinction` of a real metal instead of colours. `Toon(component = :glossy)` is Blender's
glossy toon shader. Sheen on its own is dark except at grazing angles, so it is added to a diffuse
base for velvet; `Principled`'s `sheen_weight` does the same in one material.

## Mixing and adding shaders

`Hikari.Mix(a, b; factor)` blends two materials: `factor = 0` is `a`, `1` is `b`, and the factor can
be a node graph, which gives each point of the surface its own blend. `Hikari.Add(a, b)` sums them,
for emission over a surface or sheen over a diffuse base. Both nest, so any shader graph Blender
builds from these nodes can be written down.

```@example materials
patina = color_ramp(noise(xyz(); scale = 3, detail = 6).fac, 0.45 => (0, 0, 0), 0.6 => (1, 1, 1))
docs_image("materials-mixing-and-adding-shaders", # hide
showcase([
    Hikari.Mix(Hikari.Diffuse(color = (0.8, 0.5, 0.1)), Hikari.Glossy(roughness = 0.05);
               factor = 0.1),                                                         # glazed clay
    Hikari.Mix(Hikari.Metallic(base_color = (0.95, 0.64, 0.54), roughness = 0.2),
               Hikari.Diffuse(color = (0.15, 0.45, 0.35)); factor = patina),         # copper with patina
    Hikari.Mix(Hikari.Glass(), Hikari.Diffuse(color = (0.9, 0.9, 0.9));
               factor = checker(xyz(); scale = 6).fac),                              # glass and plaster
    Hikari.Add(Hikari.Diffuse(color = (0.05, 0.05, 0.05)),
               Hikari.Emission(color = (1, 0.35, 0.05), strength = 0.8)),            # ember
    Hikari.Principled(base_color = (0.1, 0.1, 0.1), roughness = 0.4,
                      emission_color = (0.2, 0.6, 1), emission_strength = 0.8),      # glowing paint
])
) # hide
```

## Emission

A mesh pushed with `Hikari.Emission(; color, strength, two_sided)` is an area light: Hikari samples
it like any other light, so even a small emitter lights the scene with little noise. `two_sided`
makes both faces emit. Emission inside a `Mix` or `Add`, and `Principled`'s `emission_strength`,
make the surface glow but are only found by the paths that hit it, so they light the scene noisily.
For lamps, use `Emission` (see [Lights](@ref)).

## Volumes

`Hikari.MediumInterface(surface; inside, outside, shadows)` puts a participating medium behind a
surface: `inside` fills the mesh, `outside` surrounds it, and either is `nothing` for vacuum.

| Medium | |
|:-------|:--|
| `Hikari.HomogeneousMedium(; σ_a, σ_s, Le, g)` | constant absorption `σ_a`, scattering `σ_s` and emission `Le` per scene unit (`RGBSpectrum`), and the anisotropy `g` of its phase function |
| `Hikari.Milk(; scale, g)`, `Hikari.Coffee(; scale, g)` | measured liquids; `scale` dilutes them |
| `Hikari.Smoke(; density, albedo, g)` | grey smoke |
| `Hikari.GridMedium(density; σ_a, σ_s, g, bounds)` | a medium whose density varies, from a 3D array over a box |

A medium seen through glass is lit by sunlight only along paths that refract into it, which
are found by chance and leave the medium noisy. `shadows = ior` lets the sun's shadow rays through the
surface, weighted by the transmittance of a dielectric of that index: the caustics are lost, as in
most production renderers, and the noise goes. A medium with no visible surface uses
`Hikari.NullMaterial()`.

```@example materials
docs_image("materials-volumes", # hide
showcase([
    Hikari.MediumInterface(Hikari.Glass(); inside = Hikari.Milk(), shadows = 1.5),
    Hikari.MediumInterface(Hikari.Glass(); inside = Hikari.Coffee(scale = 0.5), shadows = 1.5),
    Hikari.MediumInterface(Hikari.Glass(ior = 1.33);                                   # whisky
        inside = Hikari.HomogeneousMedium(σ_a = Hikari.RGBSpectrum(0.2f0, 1.5f0, 3f0),
                                          σ_s = Hikari.RGBSpectrum(0f0)), shadows = 1.33),
    Hikari.MediumInterface(Hikari.Glass(ior = 1.6, roughness = 0.1);                   # jade
        inside = Hikari.HomogeneousMedium(σ_a = Hikari.RGBSpectrum(2f0, 0.3f0, 1.2f0),
                                          σ_s = Hikari.RGBSpectrum(8f0)), shadows = 1.6),
    Hikari.MediumInterface(Hikari.NullMaterial(); inside = Hikari.Smoke(density = 4, albedo = 0.9, g = 0.3)),
])
) # hide
```

A heterogeneous medium is a density grid over a box, for example a procedural cloud:

```julia
density = Hikari.generate_cloud_density(128; scale = 2.5, threshold = 0.15)
cloud = Hikari.GridMedium(density; σ_a = Hikari.RGBSpectrum(0.5f0), σ_s = Hikari.RGBSpectrum(15f0),
                          bounds = Hikari.Bounds3(Point3f(-1, -1, 0), Point3f(1, 1, 2)))
push!(scene, normal_mesh(Rect3f(Vec3f(-1, -1, 0), Vec3f(2))),
      Hikari.MediumInterface(Hikari.NullMaterial(); inside = cloud))
```

A thin fog filling a room shows light on its way through it. In this chapel the sun shines
through a hole in the ceiling, and the fog fills a box with a `NullMaterial` surface just inside
the walls. It scatters so little that most of the light crosses the room, and `g = 0.3` sends a
bit more of what it scatters onward than back.

```@raw html
<details><summary>The chapel</summary>
```

```@example materials
box(lo, hi) = normal_mesh(Rect3f(Vec3f(lo...), Vec3f((hi .- lo)...)))

const half_width, front, apse_center = 2f0, -3f0, 1f0
const back = apse_center + half_width + 0.4f0
const room_height, wall, ceiling = 4.2f0, 0.2f0, 1.2f0
const hole_center, hole_radius = Point2f(0, 1.7), 0.875f0
const step_height, step_starts = 0.15f0, (0.3f0, 0.65f0)
const platform = step_height * length(step_starts)
const niche_angles, niche_half_angle = (π / 2 - 0.75f0, π / 2, π / 2 + 0.75f0), 0.18f0
const niche_depth, niche_z = 0.25f0, (1.1f0, 3.3f0)

function strip_mesh(a, b; closed = false)
    n = length(a)
    positions = [a; b]
    quads = closed ? (1:n) : (1:n - 1)
    triangles = [t for i in quads for t in (TriangleFace(i, mod1(i + 1, n), n + mod1(i + 1, n)),
                                           TriangleFace(i, n + mod1(i + 1, n), n + i))]
    return GeometryBasics.Mesh(positions, triangles; normal = normals(positions, triangles))
end

arc(r, a0, a1; n = 16) = [Point2f(r * cos(t), apse_center + r * sin(t)) for t in range(a0, a1; length = n)]
lift(points, z) = [Point3f(p..., z) for p in points]
wall_strip(points, z0, z1) = strip_mesh(lift(points, z0), lift(points, z1))

function niche(angle)
    a0, a1 = angle - niche_half_angle, angle + niche_half_angle
    z0, z1 = niche_z
    front_edge, back_edge = arc(half_width, a0, a1), arc(half_width + niche_depth, a0, a1)
    return [wall_strip(front_edge, 0, z0), wall_strip(front_edge, z1, room_height),
            wall_strip(back_edge, z0, z1),
            wall_strip([front_edge[1], back_edge[1]], z0, z1), wall_strip([front_edge[end], back_edge[end]], z0, z1),
            strip_mesh(lift(front_edge, z0), lift(back_edge, z0)), strip_mesh(lift(front_edge, z1), lift(back_edge, z1))]
end

function apse_wall()
    edges = [0f0; [a + s * niche_half_angle for a in niche_angles for s in (-1, 1)]; Float32(π)]
    solid = [wall_strip(arc(half_width, edges[k], edges[k + 1]; n = 32), 0, room_height) for k in 1:2:length(edges)]
    return [solid; [piece for a in niche_angles for piece in niche(a)]]
end

function exit_point(direction, lo, hi)
    t = minimum(k -> direction[k] > 0 ? (hi[k] - hole_center[k]) / direction[k] :
                     direction[k] < 0 ? (lo[k] - hole_center[k]) / direction[k] : Inf32, 1:2)
    return hole_center + t * direction
end

function ceiling_pieces(; n = 128)
    lo, hi = Point2f(-half_width - wall, front - wall), Point2f(half_width + wall, back + wall)
    corners = [atan(c[2] - hole_center[2], c[1] - hole_center[1]) for c in
               (lo, Point2f(hi[1], lo[2]), hi, Point2f(lo[1], hi[2]))]
    angles = sort([collect(range(-π, π; length = n + 1)[1:n]); corners])
    directions = [Point2f(cos(t), sin(t)) for t in angles]
    rim = [hole_center + hole_radius * d for d in directions]
    edge = [exit_point(d, lo, hi) for d in directions]
    z0, z1 = room_height, room_height + ceiling
    return [strip_mesh(lift(rim, z0), lift(edge, z0); closed = true),
            strip_mesh(lift(rim, z1), lift(edge, z1); closed = true),
            strip_mesh(lift(rim, z0), lift(rim, z1); closed = true)]
end

step_pieces() = [box((-half_width, y, (k - 1) * step_height), (half_width, back, k * step_height))
                 for (k, y) in enumerate(step_starts)]

function room_pieces()
    w, h = half_width, room_height
    return [box((-w - wall, front - wall, -wall), (w + wall, back + wall, 0)),
            box((-w - wall, front - wall, 0), (-w, back + wall, h)),
            box((w, front - wall, 0), (w + wall, back + wall, h)),
            box((-w, back, 0), (w, back + wall, h)),
            box((-w, front - wall, 0), (w, front, h)),
            apse_wall()..., step_pieces()..., ceiling_pieces()...]
end

function bench_pieces(; x = 1f0, y = (1.6f0, 2f0), height = 0.45f0, thickness = 0.12f0)
    z0, z1 = platform, platform + height
    seat = box((-x, y[1], z1 - thickness), (x, y[2], z1))
    leg(x0) = box((x0, y[1], z0), (x0 + thickness, y[2], z1 - thickness))
    return [seat, leg(-x), leg(x - thickness)]
end

const room_air = box((-half_width + 0.02f0, front + 0.02f0, 0.02f0), (half_width - 0.02f0, back - 0.02f0, room_height - 0.02f0))

function chapel(; sun = Vec3f(-0.12, -0.12, 1))
    scene = Hikari.Scene(; backend)
    concrete = Hikari.Diffuse(color = (0.7, 0.67, 0.63), roughness = 1)
    foreach(piece -> push!(scene, piece, concrete), [room_pieces(); bench_pieces()])
    sky, sunlight = Hikari.sunsky_to_envlight(; direction = normalize(sun), turbidity = 3f0)
    push!(scene, sky)
    push!(scene, sunlight)
    return scene
end

function render_chapel(scene; resolution = (1200, 750), samples = 256, exposure = 17f0)
    samples = docs_samples(samples) # hide
    Hikari.sync!(scene)
    film = Hikari.Film(backend, Hikari.Film(Point2f(resolution...)))
    camera = Hikari.PerspectiveCamera(Point3f(0, -2.8, 1.1), Point3f(0, 1.5, 1.8), film;
                                      up = Vec3f(0, 0, 1), fov = 74)
    Hikari.clear!(film)
    integrator = Hikari.VolPath(; samples, max_depth = 16)
    integrator(scene, film, camera)
    Hikari.fill_aux_buffers!(film, scene, camera; has_infinite_lights = true)
    Hikari.denoise!(film; config = Hikari.OIDN())
    image = Array(Hikari.postprocess!(film; tonemap = :aces, exposure, gamma = 2.2))
    close(integrator)
    return image
end
nothing # hide
```

```@raw html
</details>
```

```@example materials
scene = chapel()
fog = Hikari.HomogeneousMedium(σ_a = Hikari.RGBSpectrum(0.005f0), σ_s = Hikari.RGBSpectrum(0.1f0), g = 0.3f0)
push!(scene, room_air, Hikari.MediumInterface(Hikari.NullMaterial(); inside = fog))
docs_image("materials-light-shafts", # hide
render_chapel(scene)
) # hide
```

Outdoors the camera can stand in the fog itself. Here a box of fog covers the bay around a
suspension bridge at night, lit by the lamps on the deck and a dim blue `AmbientLight` for the
sky. `g = 0.8` scatters most light only slightly off its way, so the fog glows around the
lamps and the far end of the bridge fades into it.

```@raw html
<details><summary>The bridge</summary>
```

```@example materials
block(center, size) = box(center .- size ./ 2, center .+ size ./ 2)
rod(a, b, r) = normal_mesh(Tessellation(Cylinder(Point3f(a), Point3f(b), Float32(r)), 8))
lamp(center, r) = normal_mesh(Tessellation(Sphere(Point3f(center), Float32(r)), 8))

const tower_x, cable_end, deck_z, tower_z, half_deck = 64, 110, 11, 34, 3.5

function cable_z(x)
    u = abs(x)
    u < tower_x && return deck_z + 2 + (tower_z - deck_z - 2) * (u / tower_x)^2
    s = (u - tower_x) / (cable_end - tower_x)
    return tower_z + (deck_z - tower_z) * s - 6s * (1 - s)
end

function bridge_pieces()
    sides, xs = (-half_deck, half_deck), range(-cable_end, cable_end; length = 221)
    cables = [rod((x0, y, cable_z(x0)), (x1, y, cable_z(x1)), 0.35) for y in sides for (x0, x1) in zip(xs, xs[2:end])]
    hangers = [rod((x, y, deck_z), (x, y, cable_z(x)), 0.06) for y in sides
               for x in -cable_end+4:4:cable_end-4 if abs(x) != tower_x]
    towers = [[block((x, y, tower_z / 2), (2.2, 1.6, tower_z + 2)) for x in (-tower_x, tower_x) for y in sides];
              [block((x, 0, z), (1.6, 2half_deck, 1.4)) for x in (-tower_x, tower_x) for z in (deck_z, 20, 27, tower_z + 0.5)]]
    piers = [block((x, 0, deck_z / 2), (2, 2half_deck, deck_z)) for x in -250:25:250 if abs(x) > cable_end]
    deck = block((0, 0, deck_z), (500, 2half_deck + 1, 1.2))
    lamps = [lamp((x, 1.1y, deck_z + 2.5), 0.35) for x in -250:9:250 for y in sides]
    beacons = [lamp((x, y, tower_z + 1.5), 0.3) for x in (-tower_x, tower_x) for y in sides]
    return [cables; hangers; towers; piers; deck], lamps, beacons
end

function render_bridge(scene; resolution = (1200, 800), samples = 512, exposure = 1.8f0)
    samples = docs_samples(samples) # hide
    Hikari.sync!(scene)
    film = Hikari.Film(backend, Hikari.Film(Point2f(resolution...)))
    camera = Hikari.PerspectiveCamera(Point3f(0, -130, 3), Point3f(0, 0, 3), film;
                                      up = Vec3f(0, 0, 1), fov = 54)
    Hikari.clear!(film)
    integrator = Hikari.VolPath(; samples, max_depth = 12)
    integrator(scene, film, camera)
    Hikari.fill_aux_buffers!(film, scene, camera; has_infinite_lights = true)
    Hikari.denoise!(film; config = Hikari.OIDN())
    image = Array(Hikari.postprocess!(film; tonemap = :aces, exposure, gamma = 2.2))
    close(integrator)
    return image
end
nothing # hide
```

```@raw html
</details>
```

```@example materials
scene = Hikari.Scene(; backend)
steel, lamps, beacons = bridge_pieces()
push!(scene, merge(steel), Hikari.Diffuse(color = (0.45, 0.1, 0.04), roughness = 1))
push!(scene, merge(lamps), Hikari.Emission(color = (1, 0.55, 0.2), strength = 150, two_sided = true))
push!(scene, merge(beacons), Hikari.Emission(color = (1, 0.05, 0.02), strength = 500, two_sided = true))

ripples = noise(mapping(xyz(); scale = (0.12, 0.6, 1)); detail = 4).fac
water = Hikari.Principled(base_color = (0.01, 0.015, 0.02), roughness = 0.015, normal = bump(ripples; distance = 0.06))
push!(scene, block((0, 0, -0.5), (4000, 4000, 1)), water)

fog = Hikari.HomogeneousMedium(σ_a = Hikari.RGBSpectrum(0.001f0), σ_s = Hikari.RGBSpectrum(0.015f0), g = 0.8f0)
push!(scene, block((0, 600, 140), (3000, 1800, 300)), Hikari.MediumInterface(Hikari.NullMaterial(); inside = fog))
push!(scene, Hikari.AmbientLight(Hikari.RGBSpectrum(0.014f0, 0.032f0, 0.084f0)))
docs_image("materials-foggy-bridge", # hide
render_bridge(scene)
) # hide
```

## glTF and OpenPBR materials

`Hikari.GLTF` and `Hikari.OpenPBR` take the parameters of those standards by their own names and
render as the `Principled` BSDF Blender's importers make of them:

```julia
Hikari.GLTF(base_color_factor = (0.8, 0.1, 0.1, 1), metallic_factor = 0, roughness_factor = 0.4,
            clearcoat_factor = 1, clearcoat_roughness_factor = 0.05)
Hikari.OpenPBR(base_color = (0.9, 0.6, 0.2), base_metalness = 1, specular_roughness = 0.2)
```

## pbrt-v4's models

These keep pbrt's parameters and conventions, so a pbrt scene renders as pbrt renders it. They mix
with Blender's BSDFs like any other material.

| Material | |
|:---------|:--|
| `Conductor(; ior, extinction, roughness)` or `Conductor(; reflectance, roughness)` | a metal; `Gold()`, `Silver()`, `Copper()`, `Aluminum()` and `Brass()` use measured spectra |
| `Dielectric(; ior, roughness)`, `ThinDielectric(; ior)` | glass, and glass with no thickness |
| `CoatedDiffuse(; reflectance, roughness, thickness, ior)` | a diffuse base under a dielectric coat; `Plastic(; color, roughness)` |
| `CoatedConductor(; interface_roughness, reflectance, conductor_roughness)` | a metal under a coat |
| `DiffuseTransmission(; reflectance, transmittance)`, `CoatedDiffuseTransmission` | paper, leaves, lamp shades |

`roughness` is a number or a `(u, v)` pair, and `displacement` takes a height image or graph as pbrt's
bump mapping does.

```@example materials
docs_image("materials-pbrt-v4-s-models", # hide
showcase([
    Hikari.Gold(roughness = 0.05),
    Hikari.Conductor(ior = (0.2, 0.92, 1.1), extinction = (3.9, 2.45, 2.14), roughness = 0.2),
    Hikari.CoatedConductor(interface_roughness = 0.05, reflectance = (0.8, 0.1, 0.1),
                           conductor_roughness = 0.3),
    Hikari.CoatedDiffuse(reflectance = (0.1, 0.25, 0.7), roughness = 0.05),
    Hikari.DiffuseTransmission(reflectance = (0.6, 0.6, 0.55), transmittance = (0.5, 0.45, 0.3)),
])
) # hide
```

## Changing a material

`push!` returns a handle. `Hikari.update_material!(scene, handle, material)` gives the mesh another
material, and the next render shows it:

```julia
handle = push!(scene, mesh, Hikari.Principled(base_color = (0.8, 0.1, 0.1)))
Hikari.sync!(scene)
integrator(scene, film, camera)

Hikari.update_material!(scene, handle, Hikari.Principled(base_color = (0.1, 0.1, 0.8), roughness = 0.2))
Hikari.clear!(film)
integrator(scene, film, camera)
```

A material of the same kind with other values, images or node parameters is written where the old
one was and compiles nothing. Another BSDF, or other kinds of nodes, compiles their kernels the first
time they render.
