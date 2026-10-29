## Materials

```@setup materials
using GeometryBasics, Hikari, ImageShow
to_mesh(prim) = normal_mesh(prim isa Sphere ? Tesselation(prim, 64) : prim)
sphere(r, x, z) = to_mesh(Sphere(Point3f(x, r, z), r))
floor_mat = Hikari.Diffuse(color = (0.82, 0.82, 0.82))

function render_scene(materials; res=768, spp=32)
    r = 0.38f0; sx = 1.0f0; sz = 1.1f0
    scene = Hikari.Scene()
    positions = [(-sx, sz), (0f0, sz), (sx, sz),
                 (-sx, 0f0), (0f0, 0f0), (sx, 0f0),
                 (-sx, -sz), (0f0, -sz), (sx, -sz)]
    for (mat, (x, z)) in zip(materials, positions)
        push!(scene, sphere(r, x, z), mat)
    end
    push!(scene, to_mesh(Rect3f(Vec3f(-3, 0, -3), Vec3f(6, 0.01, 6))), floor_mat)
    push!(scene, Hikari.PointLight(Point3f(3f0, 4f0, -3f0), Hikari.RGBSpectrum(40f0)))
    push!(scene, Hikari.PointLight(Point3f(-3f0, 3f0, -2f0), Hikari.RGBSpectrum(20f0)))
    push!(scene, Hikari.PointLight(Point3f(0f0, 3f0, 3f0), Hikari.RGBSpectrum(25f0)))
    push!(scene, Hikari.AmbientLight(Hikari.RGBSpectrum(0.5f0)))
    Hikari.sync!(scene)
    film   = Hikari.Film(Point2f(res, res))
    camera = Hikari.PerspectiveCamera(
        Point3f(0f0, 3.5f0, -5f0), Point3f(0f0, 0.2f0, 0f0), film; fov=40f0)
    Hikari.clear!(film)
    Hikari.VolPath(samples=spp, max_depth=8)(scene, film, camera)
    return Array(Hikari.postprocess!(film; tonemap=:aces, exposure=1.0f0, gamma=2.2f0))
end
```

Materials are Blender's BSDFs, with Blender's inputs and defaults, plus the pbrt-v4 models Hikari
needs to match pbrt's renders. Any input takes a number, a colour, an image (`Hikari.Texture`) or a
node graph from [`Hikari.TextureNodes`](@ref), and the BSDFs nest into one another as a Blender
shader graph does.

| Blender's BSDFs | pbrt-v4's models |
|-----------------|------------------|
| `Principled`, `Diffuse`, `Glossy`, `Metallic`, `Glass`, `Refraction`, `Translucent`, `Sheen`, `Toon`, `SubsurfaceScattering`, `Transparent`, `Emission` | `Conductor` (and `Gold()`, `Silver()`, `Copper()`, `Aluminum()`, `Brass()`), `CoatedDiffuse` (`Plastic()`), `CoatedConductor`, `CoatedDiffuseTransmission`, `DiffuseTransmission` |
| `Mix(a, b; factor)`, `Add(a, b)` combine them | a `Mix` holding one of them picks one material per hit, as pbrt's `MixMaterial` |

### Principled

Blender's Principled BSDF with all its inputs: `base_color`, `metallic`, `roughness`, `ior`,
`alpha`, the `subsurface_*`, `specular_*`, `transmission_*`, `coat_*` and `sheen_*` groups,
`emission_color`/`emission_strength`, the thin film and `thin_wall`.

```julia
Hikari.Principled(base_color = (0.8, 0.1, 0.1), roughness = 0.3, coat_weight = 1)     # car paint
Hikari.Principled(base_color = (0.9, 0.6, 0.2), metallic = 1, roughness = 0.2)        # gold
Hikari.Principled(transmission_weight = 1, roughness = 0)                             # glass
```

### Diffuse

Lambertian reflection, or with `roughness` (0 to 1) the energy-preserving Oren-Nayar lobe.

```julia
Hikari.Diffuse(color = (0.8, 0.2, 0.1))                 # red
Hikari.Diffuse(color = (0.9, 0.9, 0.9), roughness = 0.5) # rough, like chalk
```

### Glossy and Metallic

`Glossy` is a GGX reflection tinted by `color` at every angle, a mirror at `roughness = 0`.
`Metallic` is Blender's artist-friendly metal: `base_color` at normal incidence, `edge_tint` at
grazing angles.

`Metallic` and `Glass` also take `thin_film_thickness` in nanometres and
`thin_film_ior`, as their Blender nodes do. A zero thickness uses the original
shader layout. Hikari evaluates interference at the path's wavelengths; Cycles
integrates it separately for each RGB channel. Strongly coloured film-coated
metals can consequently differ substantially (about 15% in red in the coloured
metal regression scene), even when the interface equations agree.

```julia
Hikari.Glossy(color = (0.95, 0.95, 0.95), roughness = 0)                    # mirror
Hikari.Metallic(base_color = (0.9, 0.64, 0.54), roughness = 0.2)            # copper
```

### Glass

Reflection and refraction with the dielectric Fresnel, tinted by `color`.

```julia
Hikari.Glass(ior = 1.5)                    # clear glass
Hikari.Glass(ior = 1.33)                   # water
Hikari.Glass(ior = 2.42)                   # diamond
Hikari.Glass(roughness = 0.3, ior = 1.5)   # frosted glass
```

### Refraction

`Refraction` bends transmitted light without Fresnel reflection. It uses Blender's
Beckmann distribution by default; set `distribution = :ggx` for GGX. Both use
single scattering, so rough surfaces can lose energy. Total internal reflection
absorbs the path. `color`, `roughness`, `ior` and `normal` can be node graphs or images.

```julia
Hikari.Refraction(roughness = 0.3, ior = 1.45)
Hikari.Refraction(distribution = :ggx, roughness = 0.3)
```

### Toon

`Toon(; color, size, smooth)` uses Blender's diffuse toon response. Set
`component = :glossy` for its glossy response. The component is constant for a
shader; `color`, `size`, `smooth` and `normal` can be node graphs or images.

```julia
Hikari.Toon(color = (0.2, 0.5, 0.8), size = 0.3, smooth = 0.2)
Hikari.Toon(component = :glossy, size = 0.15)
```

### Subsurface Scattering

`SubsurfaceScattering` implements Blender's standalone node with its default
Random Walk model. `radius * scale` gives the scattering distance in scene
units. Channels with zero scattering distance use Lambert diffuse reflection.
`color`, `scale`, `radius`, `ior`, `roughness`, `anisotropy` and `normal` can be
node graphs or images. Other Blender subsurface models are not implemented.

```julia
Hikari.SubsurfaceScattering(color = (0.8, 0.6, 0.4), scale = 0.05)
```

### Transparent

Light passes straight through, tinted by `color`, along a path and in the shadows the surface
casts. A Principled `alpha` below 1 lets light through the same way, untinted.

```julia
Hikari.Mix(Hikari.Diffuse(color = (0.2, 0.6, 0.2)), Hikari.Transparent(color = (1, 1, 1)); factor = 0.5)  # half see-through
Hikari.Transparent(color = (0.9, 0.3, 0.3))                                                          # red filter
```

### Emission

A surface that emits `strength` times `color`. A mesh of it is an area light. As a
`MediumInterface`'s `emission` it makes any material glow, and inside a `Mix` or `Add` it is
Blender's Emission shader.

```julia
Hikari.Emission(color = (1, 0.9, 0.8), strength = 10, two_sided = true)
```

### Mixing BSDFs and node graphs

```julia
using Hikari.TextureNodes
rust = noise(xyz(); scale = 4).fac
Hikari.Mix(Hikari.Metallic(roughness = 0.2), Hikari.Diffuse(color = (0.4, 0.15, 0.05)); factor = rust)
Hikari.Principled(base_color = image_texture(img), normal = bump(noise(xyz()).fac; distance = 0.01))
```

### pbrt-v4's models

`Conductor` is a metal of complex index of refraction `ior` + i`extinction` (or a `reflectance`
instead), the coated models are pbrt's layered BSDFs, and `DiffuseTransmission` is Lambert
reflection and transmission. They render as pbrt renders them.

```julia
Hikari.Conductor(ior = (0.156, 0.424, 1.383), extinction = (3.602, 2.472, 1.916), roughness = 0.05)
Hikari.Gold(roughness = 0.05)
Hikari.Plastic(color = (0.8, 0.1, 0.1), roughness = 0.02)
Hikari.CoatedDiffuse(reflectance = (0.6, 0.1, 0.1), roughness = 0.05, ior = 1.5, thickness = 0.01)
Hikari.DiffuseTransmission(reflectance = (0.2, 0.4, 0.1), transmittance = (0.1, 0.3, 0.05))
```

### Changing a material

`Hikari.update_material!(scene, handle, material)` with a material of the same kind (other values,
images or node parameters) writes it where the old one is: the next frame renders it without
compiling anything. Another kind of material, or a shader with other node types or other BSDFs,
is a new material type: its shading kernel is compiled once, and the kernels of every other
material stay as they are.

### Materials Showcase

Nine spheres demonstrating the available materials:

```@example materials
mats = [
    Hikari.Diffuse(color = (0.8, 0.4, 0.3)),                                   # terracotta
    Hikari.Diffuse(color = (0.85, 0.85, 0.85), roughness = 0.7),               # chalk
    Hikari.Glass(ior = 1.5),                                                   # glass
    Hikari.Glossy(color = (0.95, 0.93, 0.88), roughness = 0),                  # mirror
    Hikari.Gold(roughness = 0.02),                                             # gold
    Hikari.Principled(base_color = (0.8, 0.1, 0.1), roughness = 0.4, coat_weight = 1, coat_roughness = 0.02),  # car paint
    Hikari.Glass(roughness = 0.3, ior = 1.5),                                  # frosted glass
    Hikari.Plastic(color = (0.1, 0.3, 0.8), roughness = 0.02),                 # plastic
    Hikari.DiffuseTransmission(reflectance = (0.4, 0.4, 0.4), transmittance = (0.4, 0.4, 0.4)),  # paper
]

render_scene(mats; spp=64)
```

Row 1 (back): Lambertian · Oren-Nayar · Glass
Row 2 (middle): Mirror · Gold · Car paint
Row 3 (front): Frosted glass · Plastic · DiffuseTransmission
