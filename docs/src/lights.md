## Lights

Hikari supports delta lights (point, directional, spot), area lights created from emissive geometry, and infinite lights (ambient, environment map).

All lights are added to the scene with `push!(scene, light)`.

```@setup lights
using GeometryBasics, LinearAlgebra, Hikari, ImageShow
to_mesh(prim) = normal_mesh(prim isa Sphere ? Tesselation(prim, 64) : prim)

function render_light(lights; res=512, spp=48, eye=Point3f(0f0, 1.5f0, -3f0), lookat=Point3f(0f0, 0.4f0, 0f0))
    scene = Hikari.Scene()
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
    film   = Hikari.Film(Point2f(res, res))
    camera = Hikari.PerspectiveCamera(eye, lookat, film; fov=50f0)
    Hikari.clear!(film)
    Hikari.VolPath(samples=spp, max_depth=10)(scene, film, camera)
    return Array(Hikari.postprocess!(film; tonemap=:aces, exposure=1.0f0, gamma=2.2f0))
end
```

### PointLight

An omnidirectional point source. Intensity falls off as 1/r².

```julia
Hikari.PointLight(position, intensity)
Hikari.PointLight(Point3f(0f0, 2.5f0, 0f0), Hikari.RGBSpectrum(40f0))
```

```@example lights
render_light([
    Hikari.PointLight(Point3f(0f0, 2.4f0, 0f0), Hikari.RGBSpectrum(40f0)),
    Hikari.PointLight(Point3f(-1f0, 1.5f0, -1.5f0), Hikari.RGBSpectrum(15f0)),
])
```

### DirectionalLight

A parallel light source at infinity — simulates the sun or any distant light. Intensity does not fall off with distance.

```julia
Hikari.DirectionalLight(intensity, direction)
Hikari.DirectionalLight(Hikari.RGBSpectrum(3f0), Vec3f(0.5f0, -1f0, 0.3f0))
```

```@example lights
render_light([
    Hikari.DirectionalLight(Hikari.RGBSpectrum(3f0), normalize(Vec3f(1f0, -1f0, 0.5f0))),
    Hikari.AmbientLight(Hikari.RGBSpectrum(0.5f0)),
])
```

### SpotLight

A point source with a cone-shaped beam. `total_width` is the full cone angle; `falloff_start` is where the soft falloff begins — both in degrees.

```julia
Hikari.SpotLight(position, target, intensity, total_width, falloff_start)
Hikari.SpotLight(Point3f(0f0, 3f0, 0f0), Point3f(0f0, 0f0, 0f0),
                 Hikari.RGBSpectrum(80f0), 35f0, 25f0)
```

```@example lights
render_light([
    Hikari.SpotLight(Point3f(0f0, 3f0, -0.5f0), Point3f(0f0, 0f0, 0f0),
                     Hikari.RGBSpectrum(80f0), 30f0, 20f0),
    Hikari.AmbientLight(Hikari.RGBSpectrum(0.5f0)),
])
```

### AmbientLight

A uniform infinite light that adds a constant radiance from all directions. Useful as a cheap fill light or sky approximation.

```julia
Hikari.AmbientLight(Hikari.RGBSpectrum(0.5f0))
```

### Area Lights

Area lights are created by marking a mesh as emissive using the `Hikari.Emission` material. The BVH light sampler automatically handles multi-triangle emitters.

```julia
emissive_panel = Hikari.Emission(color = (3.0, 3.0, 3.0), strength = 1f0, two_sided=true)
push!(scene, panel_mesh, emissive_panel)
```

```@example lights
let
    scene = Hikari.Scene()
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
    film   = Hikari.Film(Point2f(512, 512))
    camera = Hikari.PerspectiveCamera(
        Point3f(0f0, 1.5f0, -3f0), Point3f(0f0, 0.4f0, 0f0), film; fov=50f0)
    Hikari.clear!(film)
    Hikari.VolPath(samples=64, max_depth=10)(scene, film, camera)
    Array(Hikari.postprocess!(film; tonemap=:aces, exposure=1.5f0, gamma=2.2f0))
end
```

### EnvironmentLight

An HDR environment map loaded from an equirectangular `.exr` image. Provides realistic image-based lighting with importance sampling.

```julia
Hikari.EnvironmentLight("path/to/environment.exr")
Hikari.EnvironmentLight("sky.exr"; scale=Hikari.RGBSpectrum(2f0))
```
