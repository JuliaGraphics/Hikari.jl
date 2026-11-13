# Material Recipes

Realistic materials are rarely one BSDF with constant inputs. They layer a clear coat over a
base, blend two surfaces by a mask, and break up colour, roughness and normal with noise at a few
scales. The recipes below build eight such materials from the nodes of
[Textures and Node Graphs](@ref) and the BSDFs of [Materials](@ref).

```@setup recipes
using Main: docs_image, docs_samples
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
p = xyz()
```

The recipes other than the wood use the world position `p = xyz()` as texture coordinate.

## Varnished wood

The growth rings are cylinders around the pith of a log, which leans and wanders a little.
`texture_coordinate().generated` puts the log through every object, wherever it is. Noise makes the
rings uneven in shape and spacing, each ring turns abruptly from light earlywood to darker
latewood, and broad noise mottles the tone. A clear coat is the varnish.

```@example recipes
grain_noise(x, y, z = 0; detail = 2) = 2 * noise(combine(x, y, z); detail) - 1
q = texture_coordinate().generated
along = q.z - 0.5
x = q.x - 0.54 + 0.01 * along + 0.02 * grain_noise(along / 0.9, 0; detail = 3)
y = q.y - 0.44 + 0.05 * along + 0.045 * grain_noise(along / 2.2, 3.1)
r = norm(combine(x, y, 0)) + 0.013 * grain_noise(1.3 * atan(x, y), 3 * along; detail = 4)
years = r / 0.02
years = years + 0.6 * grain_noise(0.35 * years, 7.7) + 0.07 * grain_noise(280x, 4along, 280y)
f = fract(years)
late = smoothstep(0.64, 0.73, f)
tone = mix(1 - 0.25 * f * f, 0.8, late) * (1 + 0.1 * grain_noise(5x, 1.2along, 5y; detail = 4))
varnished_wood = Hikari.Principled(base_color = combine(0.66, 0.43, 0.2) * tone, roughness = 0.5,
                                   coat_weight = 1, coat_roughness = 0.03, coat_normal = normal())
nothing # hide
```

## Rusty iron

A low-frequency noise, cut sharply by `smoothstep`, decides where the rust is. The rust is a rough
Principled with its own colour noise and bump, mixed over a metal.

```@example recipes
rust_mask = smoothstep(0.48, 0.56, noise(p; scale = 2.5, detail = 8, roughness = 0.6).fac)
rust_color = color_ramp(noise(p; scale = 12, detail = 4).fac,
                        0.3 => (0.3, 0.08, 0.02), 0.7 => (0.6, 0.25, 0.07))
rust = Hikari.Principled(base_color = rust_color, roughness = 0.9,
                         normal = bump(noise(p; scale = 40).fac; distance = 0.01))
iron = Hikari.Metallic(base_color = (0.55, 0.55, 0.56), roughness = 0.3)
rusty_iron = Hikari.Mix(iron, rust; factor = rust_mask)
nothing # hide
```

## Metallic car paint

Each cell of a fine Voronoi texture is a metal flake. Its random colour tilts the flake's normal,
so flakes catch the sun one by one. The coat keeps the true normal and stays smooth.

```@example recipes
flakes = voronoi(p; scale = 250).color
car_paint = Hikari.Principled(base_color = (0.5, 0.02, 0.03), metallic = 0.8, roughness = 0.35,
                              normal = normalize(normal() + 0.3 * (flakes - 0.5)),
                              coat_weight = 1, coat_roughness = 0.02, coat_normal = normal())
nothing # hide
```

## Polished marble

Strongly distorted bands make the veins. Some subsurface scattering lets light into the stone, and
a low roughness polishes it.

```@example recipes
veins = wave(p; scale = 2, distortion = 10, detail = 6).fac
marble = Hikari.Principled(base_color = color_ramp(veins, 0.0 => (0.92, 0.91, 0.88),
                                                   0.8 => (0.85, 0.84, 0.82), 1.0 => (0.55, 0.55, 0.58)),
                           roughness = 0.08, subsurface_weight = 0.5,
                           subsurface_radius = (1, 0.9, 0.8), subsurface_scale = 0.05)
docs_image("recipes-wood-rust-paint-marble", # hide
showcase([varnished_wood, rusty_iron, car_paint, marble])
) # hide
```

## Heat-tinted steel

A thin oxide film colours steel by its thickness, which grows where the metal was hotter. Here it
grows with height, plus some noise.

```@example recipes
thickness = 100 + 500 * map_range(p.z, 0.1, 0.8) + 60 * noise(p; scale = 6).fac
heat_tinted = Hikari.Metallic(base_color = (0.6, 0.6, 0.62), edge_tint = (0.7, 0.7, 0.72),
                              roughness = 0.2, thin_film_thickness = thickness, thin_film_ior = 2.4)
nothing # hide
```

## Cracked lava

The distance to the nearest Voronoi edge finds the cracks, and an emission strength that is only
non-zero near them makes them glow. The emission lights the ground, but only along the paths that
hit the ball; for a light source, use `Hikari.Emission`.

```@example recipes
cracks = voronoi(p; scale = 5, feature = :distance_to_edge)
lava = Hikari.Principled(base_color = (0.04, 0.035, 0.03), roughness = 0.8,
                         normal = bump(noise(p; scale = 15, detail = 6).fac; distance = 0.02),
                         emission_color = (1, 0.3, 0.04),
                         emission_strength = 8 * (1 - smoothstep(0.0, 0.06, cracks)))
nothing # hide
```

## Worn gold

Real metal is never equally rough everywhere. Noise mapped into a range of roughness gives smudges
and polished spots.

```@example recipes
worn_gold = Hikari.Metallic(base_color = (1.0, 0.77, 0.34), edge_tint = (1.0, 0.9, 0.6),
                            roughness = map_range(noise(p; scale = 25, detail = 4).fac,
                                                  0.35, 0.65, 0.08, 0.4))
nothing # hide
```

## Etched glass

The roughness of glass follows a pattern: clear where the checker is 1, frosted elsewhere.

```@example recipes
etched = Hikari.Glass(roughness = 0.35 * (1 - checker(mapping(p; rotation = (π / 2, 0, 0)); scale = 6).fac))
docs_image("recipes-steel-lava-gold-glass", # hide
showcase([heat_tinted, lava, worn_gold, etched])
) # hide
```
