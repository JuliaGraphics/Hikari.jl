# Lights

Hikari supports delta lights (point, directional, spot), area lights created from emissive geometry, and infinite lights (ambient, environment map).

All lights are added to the scene with `push!(scene, light)`.

```@setup lights
using GeometryBasics, LinearAlgebra, Hikari, Mantle, ImageShow
using Main: docs_image, docs_samples
backend = Mantle.defaultbackend()
to_mesh(prim) = normal_mesh(prim isa Sphere ? Tessellation(prim, 64) : prim)

function render_light(lights; res=512, spp=256, eye=Point3f(0f0, 1.5f0, -3f0), lookat=Point3f(0f0, 0.4f0, 0f0))
    scene = Hikari.Scene(; backend)
    # Three spheres: matte, mirror, glass
    push!(scene, to_mesh(Sphere(Point3f(-0.8f0, 0.3f0, 0.2f0), 0.3f0)),
          Hikari.Diffuse(color = Hikari.RGBSpectrum(0.8f0, 0.3f0, 0.2f0)))
    push!(scene, to_mesh(Sphere(Point3f(0f0, 0.5f0, 0f0), 0.5f0)),
          Hikari.Glossy(roughness = 0, color = Hikari.RGBSpectrum(0.93f0)))
    push!(scene, to_mesh(Sphere(Point3f(0.8f0, 0.3f0, 0.2f0), 0.3f0)),
          Hikari.Glass(ior = 1.5f0))
    # Room
    push!(scene, to_mesh(Rect3f(Vec3f(-2, 0, -2), Vec3f(4, 0.01, 4))),
          Hikari.Diffuse(color = Hikari.RGBSpectrum(0.88f0)))
    push!(scene, to_mesh(Rect3f(Vec3f(-2, 0, 2-0.01f0), Vec3f(4, 3, 0.01f0))),
          Hikari.Diffuse(color = Hikari.RGBSpectrum(0.85f0)))
    push!(scene, to_mesh(Rect3f(Vec3f(-2, 0, -2), Vec3f(0.01f0, 3, 4))),
          Hikari.Diffuse(color = Hikari.RGBSpectrum(0.7f0, 0.15f0, 0.15f0)))
    push!(scene, to_mesh(Rect3f(Vec3f(2-0.01f0, 0, -2), Vec3f(0.01f0, 3, 4))),
          Hikari.Diffuse(color = Hikari.RGBSpectrum(0.15f0, 0.6f0, 0.15f0)))
    for l in lights; push!(scene, l); end
    Hikari.sync!(scene)
    film   = Hikari.Film(backend, Hikari.Film(Point2f(res, res)))
    camera = Hikari.PerspectiveCamera(eye, lookat, film; fov=50f0)
    Hikari.clear!(film)
    integrator = Hikari.VolPath(samples=docs_samples(spp), max_depth=10)
    integrator(scene, film, camera)
    Hikari.fill_aux_buffers!(film, scene, camera)
    Hikari.denoise!(film; config = Hikari.OIDN())
    image = Array(Hikari.postprocess!(film; tonemap=:aces, exposure=1.0f0, gamma=2.2f0))
    close(integrator)
    return image
end
```

## PointLight

An omnidirectional point source. Intensity falls off as 1/r².

```julia
Hikari.PointLight(position, intensity)
Hikari.PointLight(Point3f(0f0, 2.5f0, 0f0), Hikari.RGBSpectrum(40f0))
```

```@example lights
docs_image("lights-point", # hide
render_light([
    Hikari.PointLight(Point3f(0f0, 2.4f0, 0f0), Hikari.RGBSpectrum(40f0)),
    Hikari.PointLight(Point3f(-1f0, 1.5f0, -1.5f0), Hikari.RGBSpectrum(15f0)),
])
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
render_light([
    Hikari.DirectionalLight(Hikari.RGBSpectrum(3f0), normalize(Vec3f(1f0, -1f0, 0.5f0))),
    Hikari.AmbientLight(Hikari.RGBSpectrum(0.5f0)),
])
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
render_light([
    Hikari.SpotLight(Point3f(0f0, 3f0, -0.5f0), Point3f(0f0, 0f0, 0f0),
                     Hikari.RGBSpectrum(80f0), 30f0, 20f0),
    Hikari.AmbientLight(Hikari.RGBSpectrum(0.5f0)),
])
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
docs_image("lights-area", # hide
let
    scene = Hikari.Scene(; backend)
    push!(scene, to_mesh(Sphere(Point3f(-0.8f0, 0.3f0, 0.2f0), 0.3f0)),
          Hikari.Diffuse(color = Hikari.RGBSpectrum(0.8f0, 0.3f0, 0.2f0)))
    push!(scene, to_mesh(Sphere(Point3f(0f0, 0.5f0, 0f0), 0.5f0)),
          Hikari.Glossy(roughness = 0, color = Hikari.RGBSpectrum(0.93f0)))
    push!(scene, to_mesh(Sphere(Point3f(0.8f0, 0.3f0, 0.2f0), 0.3f0)),
          Hikari.Glass(ior = 1.5f0))
    push!(scene, to_mesh(Rect3f(Vec3f(-2, 0, -2), Vec3f(4, 0.01, 4))),
          Hikari.Diffuse(color = Hikari.RGBSpectrum(0.88f0)))
    push!(scene, to_mesh(Rect3f(Vec3f(-2, 0, 2-0.01f0), Vec3f(4, 3, 0.01f0))),
          Hikari.Diffuse(color = Hikari.RGBSpectrum(0.85f0)))
    push!(scene, to_mesh(Rect3f(Vec3f(-2, 0, -2), Vec3f(0.01f0, 3, 4))),
          Hikari.Diffuse(color = Hikari.RGBSpectrum(0.7f0, 0.15f0, 0.15f0)))
    push!(scene, to_mesh(Rect3f(Vec3f(2-0.01f0, 0, -2), Vec3f(0.01f0, 3, 4))),
          Hikari.Diffuse(color = Hikari.RGBSpectrum(0.15f0, 0.6f0, 0.15f0)))
    # Ceiling area light panel
    panel = to_mesh(Rect3f(Vec3f(-0.5f0, 2.9f0, -0.5f0), Vec3f(1f0, 0.01f0, 1f0)))
    push!(scene, panel, Hikari.Emission(color = (3.0, 3.0, 2.85), strength = 1f0, two_sided=true))
    Hikari.sync!(scene)
    film   = Hikari.Film(backend, Hikari.Film(Point2f(512, 512)))
    camera = Hikari.PerspectiveCamera(
        Point3f(0f0, 1.5f0, -3f0), Point3f(0f0, 0.4f0, 0f0), film; fov=50f0)
    Hikari.clear!(film)
    samples = 256
    samples = docs_samples(samples) # hide
    Hikari.VolPath(; samples, max_depth=10)(scene, film, camera)
    Hikari.fill_aux_buffers!(film, scene, camera)
    Hikari.denoise!(film; config = Hikari.OIDN())
    Array(Hikari.postprocess!(film; tonemap=:aces, exposure=1.5f0, gamma=2.2f0))
end
) # hide
```

## EnvironmentLight

An HDR environment map loaded from an equirectangular `.exr` image. Provides realistic image-based lighting with importance sampling.

```julia
Hikari.EnvironmentLight("path/to/environment.exr")
Hikari.EnvironmentLight("sky.exr"; scale = Hikari.RGBSpectrum(2f0), rotation = Mat3f(I))
```

`rotation` turns the map: a `Mat3f`, applied to the directions it is looked up by.

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
function render_sky(direction; turbidity = 3f0, samples = 256)
    samples = docs_samples(samples) # hide
    scene = Hikari.Scene(; backend)
    for (i, material) in enumerate((Hikari.Diffuse(color = (0.8, 0.8, 0.8)), Hikari.Gold(roughness = 0.1),
                                    Hikari.Glass()))
        push!(scene, normal_mesh(Tessellation(Sphere(Point3f(i - 2, 0, 0.42), 0.42f0), 64)), material)
    end
    push!(scene, normal_mesh(Rect3f(Vec3f(-50, -50, -0.01), Vec3f(100, 100, 0.01))),
          Hikari.Diffuse(color = (0.5, 0.5, 0.5)))
    sky, sun = Hikari.sunsky_to_envlight(; direction = normalize(direction), turbidity)
    push!(scene, sky)
    push!(scene, sun)
    Hikari.sync!(scene)
    film = Hikari.Film(backend, Hikari.Film(Point2f(600, 300)))
    camera = Hikari.PerspectiveCamera(Point3f(0, -4.5, 1.4), Point3f(0, 0, 0.6), film; up = Vec3f(0, 0, 1), fov = 30)
    Hikari.clear!(film)
    integrator = Hikari.VolPath(; samples, max_depth = 10)
    integrator(scene, film, camera)
    Hikari.fill_aux_buffers!(film, scene, camera; has_infinite_lights = true)
    Hikari.denoise!(film; config = Hikari.OIDN())
    image = Array(Hikari.postprocess!(film; tonemap = :aces, gamma = 2.2))
    close(integrator)
    return image
end
docs_image("lights-sun-and-sky", # hide
hcat(render_sky(Vec3f(-0.4, -0.6, 1.0)), render_sky(Vec3f(-1, 0.3, 0.3); turbidity = 5f0))
) # hide
```

At noon on the left, and in the late afternoon, with a hazier sky, on the right.
