"""
Environment light that illuminates the scene from all directions using an HDR environment map.
Uses equirectangular (lat-long) mapping.
"""
struct EnvironmentLight{S<:Spectrum, E<:EnvironmentMap{S}} <: Light
    """HDR environment map."""
    env_map::E

    """Scale factor for the light intensity."""
    scale::S
    """Ray types its light is hidden from (see [`Visibility`](@ref))."""
    hidden::UInt8

    function EnvironmentLight(
        env_map::E,
        scale::S=RGBSpectrum(1f0),
        hidden::UInt8=0x00,
    ) where {S<:Spectrum, E<:EnvironmentMap{S}}
        new{S, E}(env_map, scale, hidden)
    end
end
EnvironmentLight(l::EnvironmentLight, hidden::UInt8) = EnvironmentLight(l.env_map, l.scale, hidden)

# Environment lights are infinite (at infinity)
is_infinite_light(::EnvironmentLight) = true
is_infinite_light(::Type{<:EnvironmentLight}) = true
# …and it is VISIBLE along an escaped ray, unlike a directional light.
paints_escaped_rays(::EnvironmentLight) = true
paints_escaped_rays(::Type{<:EnvironmentLight}) = true

"""
Convenience constructor that loads an environment map from a file.
rotation: Mat3f rotation matrix (use rotation_matrix(angle_deg, axis) to create)
"""
function EnvironmentLight(
    path::String;
    scale::RGBSpectrum=RGBSpectrum(1f0),
    rotation::Mat3f=Mat3f(I),
)
    env_map = load_environment_map(path; rotation=rotation)
    EnvironmentLight(env_map, scale)
end

# ============================================================================
# The sky without a trace
# ============================================================================

"""
    paint_sky!(film, scene, camera, sensor) -> film

What every camera ray sees when there is nothing to hit: the environment maps,
looked up along each pixel's ray (the pixel the aux buffers use) and written to
the framebuffer as the tracer would leave it there. For a scene whose geometry
is all drawn by a rasterizer, which needs a sky and no trace: one lookup a
pixel instead of a sample's plans and queues. Only where that is the same
picture, see [`paints_sky_in_rgb`](@ref).
"""
function paint_sky!(film::Film, scene::AbstractScene, camera, sensor::PixelSensor)
    # The tracer uplifts a map's RGB to an illuminant spectrum and the CIE sensor
    # integrates it back: the colour's XYZ times D65's photometric integral, then
    # the sensor's output matrix (measured against 32 traced samples: equal to
    # the sampling noise, 0.1 %).
    xyz = Mat3f(linear_srgb_to_xyz(Vec3f(1, 0, 0))..., linear_srgb_to_xyz(Vec3f(0, 1, 0))...,
                linear_srgb_to_xyz(Vec3f(0, 0, 1))...)
    toframe = sensor.output_from_sensor * (sensor.imaging_ratio * D65_PHOTOMETRIC) * xyz
    backend = KA.get_backend(film.framebuffer)
    lights = Adapt.adapt(backend, scene.lights)
    sky_kernel!(backend)(film.framebuffer, film.crop_bounds, camera, lights, toframe; ndrange = length(film.framebuffer))
    return film
end

"""
    paints_sky_in_rgb(lights, sensor) -> Bool

Whether [`paint_sky!`](@ref) gives the tracer's sky: every light an escaped ray
sees is an `EnvironmentLight` (an RGB map), and the sensor is the CIE one, whose
spectral round trip is linear in that RGB.
"""
paints_sky_in_rgb(lights::Raycore.MultiTypeSet, sensor::PixelSensor) =
    sensor.sensor_name == "cie1931" && all(T -> !paints_escaped_rays(T) || T <: EnvironmentLight, lights.data_order)

@kernel inbounds=true function sky_kernel!(framebuffer, crop_bounds, camera, lights, toframe::Mat3f)
    idx = @index(Global)
    h, _ = size(framebuffer)
    row = ((idx - 1) % h) + 1
    col = ((idx - 1) ÷ h) + 1
    # The aux buffers' pixel, so sky and depth agree.
    pixel = Point2f(Float32(col) + crop_bounds.p_min[1] - 0.5f0, Float32(h - row) + crop_bounds.p_min[2] + 0.5f0)
    ray, ω = generate_ray(camera, CameraSample(pixel, Point2f(0.5f0), 0f0))
    L = ω > 0f0 ? mapreduce(sky_rgb, +, lights, lights, normalize(Vec3f(ray.d)); init = RGBSpectrum(0f0)) : RGBSpectrum(0f0)
    c = toframe * Vec3f(L.c[1], L.c[2], L.c[3])
    framebuffer[idx] = RGB{Float32}(c[1], c[2], c[3])
end

@propagate_inbounds sky_rgb(light::EnvironmentLight, lights, d::Vec3f) =
    sees_light(light.hidden, CAMERA_RAYS) ? light.scale.c[1] * light.env_map(d, lights) : RGBSpectrum(0f0)
@propagate_inbounds sky_rgb(::Light, lights, ::Vec3f) = RGBSpectrum(0f0)
