## Postprocessing and Denoising

The integrator writes linear HDR radiance into the film, already through the camera's sensor. What
follows works on that film without tracing a path again: an optional denoiser, then `postprocess!`,
which maps the radiance to a displayable image.

```julia
integrator(scene, film, camera)                  # render
Hikari.fill_aux_buffers!(film, scene, camera)    # guides for the denoiser
Hikari.denoise!(film; config = Hikari.OIDN())    # optional, in place
img = Hikari.postprocess!(film; exposure = 1, tonemap = :aces, gamma = 2.2)
```

```@setup postprocessing
using GeometryBasics, LinearAlgebra, Hikari, Mantle
using Hikari.TextureNodes
backend = Mantle.defaultbackend()
sphere(center, r) = normal_mesh(Tessellation(Sphere(center, r), 96))

scene = Hikari.Scene(; backend)
push!(scene, sphere(Point3f(-0.55, 0, 0.42), 0.42f0), Hikari.Gold(roughness = 0.15))
squares = checker(xyz(); scale = 8, color1 = (0.8, 0.2, 0.1), color2 = (0.9, 0.85, 0.7))
push!(scene, sphere(Point3f(0.45, 0.1, 0.42), 0.42f0), Hikari.Principled(base_color = squares.color, roughness = 0.5))
push!(scene, sphere(Point3f(0.0, 0.9, 0.25), 0.25f0), Hikari.Emission(color = (1, 0.6, 0.3), strength = 8))
tiles = checker(xyz(); scale = 1, color1 = (0.5, 0.5, 0.5), color2 = (0.38, 0.38, 0.38))
push!(scene, normal_mesh(Rect3f(Vec3f(-50, -50, -0.01), Vec3f(100, 100, 0.01))), Hikari.Diffuse(color = tiles.color))
sky, sun = Hikari.sunsky_to_envlight(; direction = normalize(Vec3f(-0.6, -1, 0.9)), turbidity = 3f0)
push!(scene, sky)
push!(scene, sun)
Hikari.sync!(scene)

film = Hikari.Film(backend, Hikari.Film(Point2f(400, 300)))
camera = Hikari.PerspectiveCamera(Point3f(0.3, -3.6, 1.45), Point3f(0, 0.2, 0.35), film; up = Vec3f(0, 0, 1), fov = 28)
function render!(film, samples)
    Hikari.clear!(film)
    integrator = Hikari.VolPath(; samples, max_depth = 8)
    integrator(scene, film, camera)
    Mantle.waitidle(backend)
    close(integrator)
    return film
end
```

### Tone mapping and exposure

`postprocess!(film; exposure, tonemap, gamma, white_point, background, sky)` returns the film's
display image and can run as often as you like on one render.

| Keyword | Default | |
|---------|---------|---|
| `exposure` | `1.0` | linear factor before tone mapping |
| `tonemap` | `:aces` | `:aces`, `:reinhard`, `:reinhard_extended`, `:uncharted2`, `:filmic`, or `nothing` to clamp |
| `gamma` | `2.2` | display gamma, or `nothing` for linear output |
| `white_point` | `4.0` | the radiance that maps to white with `:reinhard_extended` |
| `background` | `nothing` | an `RGB{Float32}` or `RGBA{Float32}` for pixels whose camera ray hit nothing; its alpha is the output's alpha there |
| `sky` | `false` | whether an environment, sky or ambient light already painted those pixels; they then keep the sky's colour and take only the background's alpha |

```@example postprocessing
render!(film, 64)
hcat([Array(Hikari.postprocess!(film; tonemap)) for tonemap in (:aces, :reinhard, :uncharted2, :filmic, nothing)]...)
```

From left to right: `:aces`, `:reinhard`, `:uncharted2`, `:filmic` and no tone mapping.

### Sensor and pixel filter

The camera's sensor and pixel filter are part of the render, so changing them means rendering
again. Both are keywords of the integrator:

```julia
Hikari.VolPath(; samples = 64,
               sensor = Hikari.PixelSensor(; iso = 200, exposure_time = 1, whitebalance = 5000),
               filter = Hikari.GaussianFilter(; radius = Point2f(1.5), sigma = 0.5f0))
```

`PixelSensor` is pbrt-v4's: `iso` and `exposure_time` scale the image by `iso * exposure_time / 100`,
`whitebalance` adapts a colour temperature in kelvin to white (0 does nothing), and `sensor` names a
camera response (`"cie1931"` by default). Hikari's defaults reproduce pbrt's images, and a pbrt
scene's own `Film` settings are applied when it is loaded. The pixel filters are `BoxFilter`,
`TriangleFilter`, `GaussianFilter` (the default), `MitchellFilter` and `LanczosSincFilter`.

### Denoising

A denoiser turns a render at a few samples per pixel into a clean image. Each one runs in place on
`film.framebuffer`, before `postprocess!`, and reads guides: the albedo, normal and depth of what
each pixel sees first. `fill_aux_buffers!` traces them without noise:

```julia
Hikari.fill_aux_buffers!(film, scene, camera; has_infinite_lights = false, grid = 9)
```

Pass `has_infinite_lights = true` when an environment, sky or ambient light paints the background.
`grid` is the number of guide rays per pixel along each side (9 by default), filtered like the image;
use 1 while the camera moves. `clear!` resets the guides with the image, so fill them after it.

| Denoiser | Runs on | |
|----------|---------|---|
| `Hikari.DenoiseConfig(; iterations = 4, sigma_color = 1, sigma_normal = 64, sigma_depth = 0.1)` | the GPU | an edge-avoiding à-trous wavelet filter; fast enough for every frame of an interactive view |
| `Hikari.OIDN(; quality = :high, guides = true)` | the CPU | Intel Open Image Denoise's ray tracing network |
| `OIDNRunner.OIDNDenoiser` | the GPU | the same network, ported to Julia and Mantle in JuliaVision |

```@example postprocessing
render!(film, 4)
noisy = Array(Hikari.postprocess!(film))

Hikari.fill_aux_buffers!(film, scene, camera; has_infinite_lights = true)
Hikari.denoise!(film; config = Hikari.DenoiseConfig())
atrous = Array(Hikari.postprocess!(film))

render!(film, 4)
Hikari.fill_aux_buffers!(film, scene, camera; has_infinite_lights = true)
Hikari.denoise!(film; config = Hikari.OIDN())
oidn = Array(Hikari.postprocess!(film))

render!(film, 1024)
hcat(noisy, atrous, oidn, Array(Hikari.postprocess!(film)))
```

From left to right: 4 samples per pixel, the à-trous filter, OIDN, and 1024 samples per pixel.

**The à-trous filter** keeps edges where the normal or depth changes, so silhouettes stay sharp,
but it smooths texture detail. `iterations` doubles its radius each time. `sigma_color` is the
brightness difference in stops it still blends across, `sigma_normal` the exponent on the normals'
agreement (higher keeps more edges) and `sigma_depth` the relative depth difference it tolerates.

**OIDN** reconstructs textures and edges from the guides and gives the best images. `quality` is
`:fast`, `:balanced` or `:high`. With `guides = false` it needs only the noisy image. The library
is Intel's official Linux x86-64 build, downloaded as an artifact the first time a film is
denoised. Each call copies the frame to the host and back.

**OIDNRunner** runs the OIDN network on the GPU from the film's device arrays, with no host copy.
It lives in JuliaVision, outside Hikari. Its images are width by height, the transpose of a
film's:

```julia
using OIDNRunner
denoiser = OIDNDenoiser(; backend)
Hikari.fill_aux_buffers!(film, scene, camera)
color, albedo, normal = permutedims(film.framebuffer), permutedims(film.albedo), permutedims(film.normal)
denoised = similar(color)
OIDNRunner.denoise!(denoiser, denoised, color, albedo, normal)
permutedims!(film.framebuffer, denoised, (2, 1))
```

In RayMakie, the screen options `denoise` and `denoise_config` select a denoiser for every frame:

```julia
RayMakie.activate!(denoise = true, denoise_config = Hikari.OIDN())
```
