"""
Environment light that illuminates the scene from all directions using an HDR environment map.
Uses equirectangular (lat-long) mapping.
"""
struct EnvironmentLight{S<:Spectrum, E<:EnvironmentMap{S}} <: Light
    """LightInfinite flag - environment lights are at infinity."""
    flags::LightFlags

    """HDR environment map."""
    env_map::E

    """Scale factor for the light intensity."""
    scale::S

    function EnvironmentLight(
        env_map::E,
        scale::S=RGBSpectrum(1f0);
    ) where {S<:Spectrum, E<:EnvironmentMap{S}}
        new{S, E}(LightInfinite, env_map, scale)
    end
end

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

"""
Compute radiance arriving at interaction point from the environment light.
Uses importance sampling based on environment map luminance.

# Args
- `e::EnvironmentLight`: Environment light.
- `ref::Interaction`: Interaction point for which to compute radiance.
- `u::Point2f`: Random sample for direction selection.

# Returns
Tuple of (radiance, incident direction, pdf, visibility tester)
"""
@propagate_inbounds function sample_li(e::EnvironmentLight{S}, i::Interaction, u::Point2f, scene::AbstractScene) where {S}
    # Importance sample the environment map based on luminance
    uv, map_pdf = sample_continuous(e.env_map.distribution, u)

    # Convert UV to direction using equal-area mapping
    wi = uv_to_direction(uv, e.env_map.rotation)

    # Convert PDF from image space to solid angle
    # For equal-area mapping: pdf_solidangle = pdf_image / (4π)
    # This is because equal-area mapping preserves solid angle uniformity
    pdf = map_pdf / (4f0 * Float32(π))

    # Sample the environment map
    radiance = e.scale * e.env_map(wi)

    # Create visibility tester - the light is at "infinity"
    # Use 2x scene_radius to ensure we're far enough away
    p_light = i.p + wi * (2f0 * scene.world_radius)
    visibility = VisibilityTester(
        i,
        Interaction(p_light, i.time, wi, Normal3f(0f0))
    )

    radiance, wi, pdf, visibility
end

"""
Sample a ray leaving the environment light (for photon mapping / light tracing).
Uses importance sampling based on environment map luminance.

The ray is sampled by:
1. Importance sampling a direction from the environment map
2. Placing a disk of radius scene_radius centered at scene_center,
   perpendicular to that direction
3. Sampling a point on the disk and shooting a ray inward
"""
function sample_le(
    e::EnvironmentLight{S}, u1::Point2f, u2::Point2f, ::Float32, scene::Scene
)::Tuple{S,Ray,Normal3f,Float32,Float32} where {S}
    # Importance sample direction from environment map
    # wi is the direction light comes FROM (pointing outward from scene)
    uv, map_pdf = sample_continuous(e.env_map.distribution, u1)
    wi = uv_to_direction(uv, e.env_map.rotation)

    # Convert PDF from image space to solid angle
    # For equal-area mapping: pdf_solidangle = pdf_image / (4π)
    pdf_dir = map_pdf / (4f0 * Float32(π))

    # Ray direction is -wi (pointing into the scene)
    ray_dir = -wi

    # Create coordinate frame with ray_dir as the z-axis (following pbrt)
    # This frame is used to offset the disk sampling point
    w = -ray_dir  # = wi, the outward direction
    if abs(w[2]) < 0.99f0
        v = normalize(Vec3f(0f0, 1f0, 0f0) × w)
    else
        v = normalize(Vec3f(1f0, 0f0, 0f0) × w)
    end
    u_axis = w × v

    # Sample point on disk centered at scene_center, perpendicular to ray direction
    d = concentric_sample_disk(u2)
    p_disk = scene.world_center + scene.world_radius * (d[1] * u_axis + d[2] * v)

    # Ray origin: start from disk point, go outward by scene_radius, then shoot inward
    origin = p_disk + scene.world_radius * w

    ray = Ray(o=origin, d=ray_dir)
    # Light normal points outward (toward environment)
    light_normal = Normal3f(wi)

    # PDF for position on disk
    pdf_pos = 1f0 / (Float32(π) * scene.world_radius^2)

    # Sample radiance in the direction wi (the direction light comes FROM)
    radiance = e.scale * e.env_map(wi)

    return radiance, ray, light_normal, pdf_pos, pdf_dir
end

"""
Compute emitted radiance for a ray that escapes the scene (hits no geometry).
This is called when a camera/path ray doesn't hit anything.
"""
function le(env::EnvironmentLight, ray::Union{Ray,RayDifferentials})
    # Sample environment map in ray direction
    env.scale * env.env_map(normalize(Vec3f(ray.d)))
end

"""
PDF for sampling a particular direction from the environment light.
Returns the probability density for importance sampling this direction.
"""
function pdf_li(e::EnvironmentLight, ::Interaction, wi::Vec3f)::Float32
    # Convert direction to UV using equal-area mapping
    uv = direction_to_uv(wi, e.env_map.rotation)

    # Get PDF from 2D distribution
    map_pdf = pdf(e.env_map.distribution, uv)

    # Convert from image space to solid angle
    # For equal-area mapping: pdf_solidangle = pdf_image / (4π)
    map_pdf / (4f0 * Float32(π))
end

"""
Total power emitted by the environment light.
For an environment light, this is approximated as the average radiance times
the surface area of the bounding sphere.
"""
@propagate_inbounds function power(e::EnvironmentLight{S}, scene::Scene)::S where {S<:Spectrum}
    # Approximate power as average radiance * 4π * r²
    # This is a rough approximation - more accurate would integrate over the map
    e.scale * S(4f0 * π * scene.world_radius^2)
end
