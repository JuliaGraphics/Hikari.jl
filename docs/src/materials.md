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
using Main: docs_image, docs_samples # hide
using GeometryBasics, LinearAlgebra, Hikari, Mantle, FileIO
using Hikari.TextureNodes
backend = Mantle.defaultbackend()

const logo_ball = load(pkgdir(Hikari, "docs", "src", "assets", "logo_ball.obj"))

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
