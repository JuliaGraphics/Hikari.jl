"""
Environment light that illuminates the scene from all directions using an HDR environment map.
Uses equirectangular (lat-long) mapping.
"""
struct EnvironmentLight{S<:Spectrum, E<:EnvironmentMap{S}} <: Light
    """HDR environment map."""
    env_map::E

    """Scale factor for the light intensity."""
    scale::S

    function EnvironmentLight(
        env_map::E,
        scale::S=RGBSpectrum(1f0);
    ) where {S<:Spectrum, E<:EnvironmentMap{S}}
        new{S, E}(env_map, scale)
    end
end

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
    p_light = i.p + wi * (2f0 * world_radius(scene))
    visibility = VisibilityTester(
        i,
        Interaction(p_light, i.time, wi, Normal3f(0f0))
    )

    radiance, wi, pdf, visibility
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

