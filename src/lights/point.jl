struct PointLight{S<:Spectrum} <: Light
    """
    Since point lights represent singularities that only emit light
    from a single position, flag is set to `LightδPosition`.
    """
    flags::LightFlags
    """
    Ligh-source is positioned at the origin of its light space.
    """
    light_to_world::Transformation
    world_to_light::Transformation

    i::S
    """
    Position in world space.
    """
    position::Point3f

    function PointLight(light_to_world::Transformation, i::S) where S<:Spectrum
        new{S}(
            LightδPosition, light_to_world, inv(light_to_world),
            i, light_to_world(Point3f(0f0)),
        )
    end
end

function PointLight(position, i::S) where S<:Spectrum
    PointLight(translate(Vec3f(position)), i)
end

"""
Compute radiance arriving at `ref.p` interaction point at `ref.time` time
due to that light, assuming there are no occluding objects between them.

# Args

- `p::PointLight`: Light which illuminates the interaction point `ref`.
- `ref::Interaction`: Interaction point for which to compute radiance.
- `u::Point2f`: Sampling point that is ignored for `PointLight`,
    since it has no area.

# Returns

`Tuple{S, Vec3f, Float32, VisibilityTester} where S <: Spectrum`:

    - `S`: Computed radiance.
    - `Vec3f`: Incident direction to the light source `wi`.
    - `Float32`: Probability density for the light sample that was taken.
        For `PointLight` it is always `1`.
    - `VisibilityTester`: Initialized visibility tester that holds the
        shadow ray that must be traced to verify that
        there are no occluding objects between the light and reference point.
"""
function sample_li(p::PointLight, i::Interaction, ::Point2f)
    wi = normalize(Vec3f(p.position - i.p))
    pdf = 1f0
    visibility = VisibilityTester(
        i, Interaction(p.position, i.time, Vec3f(0.0f0), Normal3f(0.0f0)),
    )
    radiance = p.i / distance_squared(p.position, i.p)
    radiance, wi, pdf, visibility
end

function sample_le(
    p::PointLight, u1::Point2f, ::Point2f, ::Float32,
)::Tuple{RGBSpectrum,Ray,Normal3f,Float32,Float32}
    ray = Ray(o=p.position, d=uniform_sample_sphere(u1))
    @real_assert norm(ray.d) ≈ 1f0
    light_normal = Normal3f(ray.d)
    pdf_pos = 1f0
    pdf_dir = uniform_sphere_pdf()
    return p.i, ray, light_normal, pdf_pos, pdf_dir
end

"""
Total power emitted by the light source over the entire sphere of directions.
"""
@inline function power(p::PointLight)
    4f0 * π * p.i
end
