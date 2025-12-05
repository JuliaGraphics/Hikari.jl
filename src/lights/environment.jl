"""
Environment light that illuminates the scene from all directions using an HDR environment map.
Uses equirectangular (lat-long) mapping.
"""
struct EnvironmentLight{S<:Spectrum} <: Light
    """LightInfinite flag - environment lights are at infinity."""
    flags::LightFlags

    """HDR environment map."""
    env_map::EnvironmentMap{S}

    """Scale factor for the light intensity."""
    scale::S

    """World radius for positioning the light at infinity."""
    world_radius::Float32

    function EnvironmentLight(
        env_map::EnvironmentMap{S},
        scale::S=RGBSpectrum(1f0);
        world_radius::Float32=1000f0
    ) where {S<:Spectrum}
        new{S}(LightInfinite, env_map, scale, world_radius)
    end
end

"""
Convenience constructor that loads an environment map from a file.
"""
function EnvironmentLight(
    path::String;
    scale::RGBSpectrum=RGBSpectrum(1f0),
    rotation::Float32=0f0,
    world_radius::Float32=1000f0
)
    env_map = load_environment_map(path; rotation=rotation)
    EnvironmentLight(env_map, scale; world_radius=world_radius)
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
function sample_li(e::EnvironmentLight{S}, i::Interaction, u::Point2f) where {S}
    # Importance sample the environment map based on luminance
    uv, map_pdf = sample_continuous(e.env_map.distribution, u)

    # Convert UV to direction
    wi = uv_to_direction(uv, e.env_map.rotation)

    # Convert PDF from image space to solid angle
    # pdf_solidangle = pdf_image / (2π² sin(θ))
    θ = uv[2] * π
    sin_θ = sin(θ)
    pdf = sin_θ > 0f0 ? map_pdf / (2f0 * π * π * sin_θ) : 0f0

    # Sample the environment map
    radiance = e.scale * e.env_map(wi)

    # Create visibility tester - the light is at "infinity"
    p_light = i.p + wi * e.world_radius
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
2. Placing a disk of radius world_radius centered at the scene origin,
   perpendicular to that direction
3. Sampling a point on the disk and shooting a ray in direction -wi
"""
function sample_le(
    e::EnvironmentLight{S}, u1::Point2f, u2::Point2f, ::Float32
)::Tuple{S,Ray,Normal3f,Float32,Float32} where {S}
    # Importance sample direction from environment map
    uv, map_pdf = sample_continuous(e.env_map.distribution, u1)
    wi = uv_to_direction(uv, e.env_map.rotation)

    # Convert PDF from image space to solid angle
    θ = uv[2] * π
    sin_θ = sin(θ)
    pdf_dir = sin_θ > 0f0 ? map_pdf / (2f0 * π * π * sin_θ) : 0f0

    # Sample a point on a disk perpendicular to the incoming direction
    # The disk is centered at the scene origin (0,0,0) and has radius world_radius
    d = concentric_sample_disk(u2)

    # Create coordinate frame with wi as the z-axis
    w = wi
    if abs(w[1]) > 0.1f0
        v = normalize(Vec3f(0f0, 1f0, 0f0) × w)
    else
        v = normalize(Vec3f(1f0, 0f0, 0f0) × w)
    end
    u_axis = w × v

    # Disk point in world space (centered at origin, perpendicular to wi)
    p_disk = (d[1] * e.world_radius) * u_axis + (d[2] * e.world_radius) * v

    # Ray origin: start from far away (world_radius distance) along wi direction,
    # offset by the disk point
    origin = Point3f(p_disk) + e.world_radius * w

    ray = Ray(o=origin, d=-wi)
    # Light normal points outward from the light (toward environment)
    light_normal = Normal3f(wi)

    # PDF for position on disk
    pdf_pos = 1f0 / (π * e.world_radius^2)

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
    # Convert direction to UV
    uv = direction_to_uv(wi, e.env_map.rotation)

    # Get PDF from 2D distribution
    map_pdf = pdf(e.env_map.distribution, uv)

    # Convert from image space to solid angle
    θ = uv[2] * π
    sin_θ = sin(θ)
    sin_θ > 0f0 ? map_pdf / (2f0 * π * π * sin_θ) : 0f0
end

"""
Total power emitted by the environment light.
For an environment light, this is approximated as the average radiance times
the surface area of the bounding sphere.
"""
@inline function power(e::EnvironmentLight{S})::S where {S<:Spectrum}
    # Approximate power as average radiance * 4π * r²
    # For simplicity, sample a few directions and average
    # A more accurate method would integrate over the entire environment map
    e.scale * S(4f0 * π * e.world_radius^2)
end
