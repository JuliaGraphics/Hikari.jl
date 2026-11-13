# Lights

Hikari supports delta lights (point, directional, spot), area lights created from emissive geometry, and infinite lights (ambient, environment map).

A delta light shines from a single point or along a single direction. It has no size, so it casts
shadows with sharp edges and is invisible to the camera and in mirror reflections: it lights
surfaces but cannot itself be seen. For soft shadows and a visible light source, use an area light.

All lights are added to the scene with `push!(scene, light)`.

```@setup lights
using GeometryBasics, LinearAlgebra, Hikari, Mantle, ImageShow, FileIO
using Main: docs_image, docs_samples, logo_ball_path
backend = Mantle.defaultbackend()

const logo_ball = load(logo_ball_path())

function ball(center, r)
    positions = [center + r * p for p in coordinates(logo_ball)]
    return GeometryBasics.Mesh(positions, faces(logo_ball); normal = normals(positions, faces(logo_ball)))
end

const ground_plane = normal_mesh(Rect3f(Vec3f(-5000, -5000, -0.01), Vec3f(10000, 10000, 0.01)))

function render_light(lights; ground = ground_plane, exposure = 1f0, res = 512, spp = 256,
                      eye = Point3f(0, -4.2, 2), lookat = Point3f(0, 0, 0.4))
    scene = Hikari.Scene(; backend)
    push!(scene, ball(Point3f(-0.6, 0, 0.5), 0.5f0), Hikari.Diffuse(color = (0.85, 0.83, 0.78)))
    push!(scene, ball(Point3f(0.6, 0, 0.5), 0.5f0), Hikari.Metallic(base_color = (0.9, 0.9, 0.9), roughness = 0.12))

    push!(scene, ground, Hikari.Diffuse(color = (0.2, 0.2, 0.2)))

    for light in lights
        light isa Pair ? push!(scene, light...) : push!(scene, light)
    end
    Hikari.sync!(scene)

    film = Hikari.Film(backend, Hikari.Film(Point2f(res, res)))
    camera = Hikari.PerspectiveCamera(eye, lookat, film; up = Vec3f(0, 0, 1), fov = 50f0)
    Hikari.clear!(film)
    integrator = Hikari.VolPath(samples = docs_samples(spp), max_depth = 10)
    integrator(scene, film, camera)
    Hikari.fill_aux_buffers!(film, scene, camera; has_infinite_lights = any(l -> l isa Hikari.EnvironmentLight, lights))
    Hikari.denoise!(film; config = Hikari.OIDN())
    image = Array(Hikari.postprocess!(film; tonemap = :aces, exposure, gamma = 2.2f0))
    close(integrator)
    return image
end
```

## PointLight

An omnidirectional point source. Intensity falls off as 1/r². A point light has no size, so the
shadows it casts always have sharp edges, however near to or far from the objects it is.

```julia
Hikari.PointLight(position, intensity)
Hikari.PointLight(Point3f(0f0, 2.5f0, 0f0), Hikari.RGBSpectrum(6f0))
```

```@example lights
docs_image("lights-point", # hide
render_light([Hikari.PointLight(Point3f(1.2, -0.8, 1.5), Hikari.RGBSpectrum(4f0))])
) # hide
```

Moved twice as far from the balls, a light needs four times the intensity to light them as
brightly. The further away it is, the less the ground falls off across the scene, because its
near and far parts differ less in their distance to the light. Here each light is three times as far as
the one before, and nine times as strong:

```@example lights
balls = Point3f(0, 0, 0.5)
offset = Vec3f(1.2, -0.8, 1)
renders = [render_light([Hikari.PointLight(balls + f * offset, Hikari.RGBSpectrum(4f0 * f^2))])
           for f in 0.75f0 .* (1, 3, 9, 27)]
docs_image("lights-point-distance", # hide
[renders[1] renders[2]; renders[3] renders[4]]
) # hide
```

## DirectionalLight

A parallel light source at infinity, for the sun or any distant light. Intensity does not fall off with distance.

```julia
Hikari.DirectionalLight(intensity, direction)
Hikari.DirectionalLight(Hikari.RGBSpectrum(3f0), Vec3f(0.5f0, -1f0, 0.3f0))
```

```@example lights
docs_image("lights-directional", # hide
render_light([Hikari.DirectionalLight(Hikari.RGBSpectrum(1.5f0), normalize(Vec3f(-1, 0.5, -1)))])
) # hide
```

## SpotLight

A point source with a cone-shaped beam. `total_width` is the angle from the beam's axis to its edge, half the cone's opening angle, and `falloff_start` the angle from the axis where the soft falloff begins, both in degrees.

```julia
Hikari.SpotLight(position, target, intensity, total_width, falloff_start)
Hikari.SpotLight(Point3f(0f0, 3f0, 0f0), Point3f(0f0, 0f0, 0f0),
                 Hikari.RGBSpectrum(80f0), 35f0, 25f0)
```

```@example lights
docs_image("lights-spot", # hide
render_light([Hikari.SpotLight(Point3f(1.5, -1, 3), Point3f(0.25, 0.3, 0), Hikari.RGBSpectrum(15f0), 25f0, 15f0)])
) # hide
```

## AmbientLight

A uniform infinite light that adds a constant radiance from all directions. Useful as a cheap fill light or sky approximation.

```julia
Hikari.AmbientLight(Hikari.RGBSpectrum(0.5f0))
```

## Area Lights

Area lights are created by marking a mesh as emissive using the `Hikari.Emission` material. The BVH light sampler automatically handles multi-triangle emitters.

```julia
emissive_panel = Hikari.Emission(color = (3.0, 3.0, 3.0), strength = 1f0, two_sided=true)
push!(scene, panel_mesh, emissive_panel)
```

```@example lights
function softbox(scale)
    up = scale * Vec3f(-sind(40), 0, cosd(40))
    side = scale * Vec3f(0, 1.2, 0)
    center = Point3f(1.7, 0, 1.6)
    corners = [center - up - side, center - up + side, center + up + side, center + up - side]
    return normal_mesh(GeometryBasics.Mesh(corners, [TriangleFace(1, 2, 3), TriangleFace(1, 3, 4)]))
end
glow(strength) = Hikari.Emission(; color = (1, 1, 0.95), strength, two_sided = true)
docs_image("lights-area", # hide
render_light([softbox(1) => glow(4)])
) # hide
```

The larger an area light, the softer the shadows it casts. Here the panel's area doubles from
image to image, at half the strength, so the light it gives off stays the same:

```@example lights
renders = [render_light([softbox(sqrt(area)) => glow(4 / area)]) for area in (0.5f0, 1f0, 2f0, 4f0)]
docs_image("lights-area-size", # hide
[renders[1] renders[2]; renders[3] renders[4]]
) # hide
```

Any mesh can be an area light, not only a panel. Unlike a delta light, it is visible, so its shape
shows in the camera and in reflections. A flat ring hanging over the balls appears as a loop on
the metal ball:

```@example lights
function ring(center, inner, outer; n = 128)
    angles = range(0, 2π; length = n + 1)[1:n]
    positions = [center + r * Point3f(cos(a), sin(a), 0) for a in angles for r in (inner, outer)]
    index(i, k) = 2 * mod(i, n) + k
    triangles = [face for i in 0:n-1
                 for face in (TriangleFace(index(i, 1), index(i, 2), index(i + 1, 2)),
                              TriangleFace(index(i, 1), index(i + 1, 2), index(i + 1, 1)))]
    return normal_mesh(GeometryBasics.Mesh(positions, triangles))
end
docs_image("lights-area-ring", # hide
render_light([ring(Point3f(0, 0, 1.85), 0.9f0, 1.1f0) => glow(8)])
) # hide
```

## EnvironmentLight

An HDR environment map loaded from an equirectangular `.exr` image. Provides realistic image-based lighting with importance sampling.

```julia
Hikari.EnvironmentLight("path/to/environment.exr")
Hikari.EnvironmentLight("sky.exr"; scale = Hikari.RGBSpectrum(2f0), rotation = Mat3f(I))
```

`rotation` turns the map: a `Mat3f`, applied to the directions it is looked up by.

The map lights the scene with the colours of its surroundings. Here the balls stand on a disc in a
room, a forest and a quarry, from [Poly Haven](https://polyhaven.com/hdris)'s free HDRIs:

```@example lights
using Downloads
function hdri(name)
    path = joinpath(tempdir(), "$name.exr")
    isfile(path) || Downloads.download("https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/1k/$(name)_1k.exr", path)
    return path
end
disc = normal_mesh(Tessellation(Cylinder(Point3f(0, 0.3, -0.01), Point3f(0, 0.3, 0), 1.5f0), 128))
docs_image("lights-environment", # hide
hcat((render_light([Hikari.EnvironmentLight(hdri(name))]; ground = disc)
      for name in ("small_empty_room_1", "rainforest_trail", "minedump_flats"))...)
) # hide
```

## Sun and sky

`Hikari.sunsky_to_envlight` evaluates the Hosek-Wilkie model of a clear sky, as pbrt-v4's
`makesky` does, and returns the sky as an `EnvironmentLight` together with a `SunLight` for the
sun's disk. Push both. The model's up direction is +Z.

```julia
sky, sun = Hikari.sunsky_to_envlight(; direction = normalize(Vec3f(-0.6, -1, 0.9)),
                                     intensity = 1f0, turbidity = 2.5f0,
                                     ground_albedo = Hikari.RGBSpectrum(0.3f0), ground_enabled = true)
push!(scene, sky)
push!(scene, sun)
```

`direction` points towards the sun. `turbidity` runs from 1 (very clear) to 10 (hazy), and
`ground_enabled = false` leaves the lower half of the sky black, for scenes that model their own ground.

```@example lights
function render_sky(direction; turbidity = 3f0)
    sky, sun = Hikari.sunsky_to_envlight(; direction = normalize(direction), turbidity)
    return render_light([sky, sun])
end
docs_image("lights-sun-and-sky", # hide
hcat(render_sky(Vec3f(-0.4, -0.6, 1.0)), render_sky(Vec3f(-1, 0.3, 0.3); turbidity = 5f0))
) # hide
```

At noon on the left, and in the late afternoon, with a hazier sky, on the right.
