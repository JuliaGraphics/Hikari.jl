struct EnvironmentLight{S<:Spectrum,T<:Texture{S}} <: Light
    """
    Environment lights illuminate the entire scene based on a surrounding
    texture map, typically HDR. `LightInfinite` flag is used.
    """
    flags::LightFlags

    """
    HDR environment map texture.
    """
    env_map::T

    """
    Scale factor for the light intensity.
    """
    scale::S

    function EnvironmentLight( env_map::T, scale::S) where {S<:Spectrum,T<:Texture{S}}
        new{S,T}(
            LightInfinite,
            env_map, scale
        )
    end
end

"""
Compute radiance arriving at `ref.p` interaction point at `ref.time` time
due to the environment light.

# Args

- `e::EnvironmentLight`: Environment light which illuminates the interaction point `ref`.
- `ref::Interaction`: Interaction point for which to compute radiance.
- `u::Point2f`: Sampling point used for the environment map lookup.

# Returns

`Tuple{S, Vec3f, Float32, VisibilityTester} where S <: Spectrum`:

    - `S`: Computed radiance.
    - `Vec3f`: Incident direction to the light source `wi`.
    - `Float32`: Probability density for the light sample.
    - `VisibilityTester`: Initialized visibility tester that holds the
        shadow ray that must be traced to verify that
        there are no occluding objects between the light and reference point.
"""
function sample_li(e::EnvironmentLight, i::Interaction, u::Point2f)
    wi = uniform_sample_sphere(u)
    pdf = uniform_sphere_pdf()
    radiance = e.scale * e.env_map(wi)
    visibility = VisibilityTester(
        i, Interaction(i.p + wi * large_distance, i.time, wi, Normal3f(0.0f0)),
    )
    radiance, wi, pdf, visibility
end

function sample_le(
        e::EnvironmentLight, u1::Point2f, u2::Point2f, ::Float32,
    )::Tuple{RGBSpectrum,Ray,Normal3f,Float32,Float32}

    wi = uniform_sample_sphere(u1)
    ray = Ray(o=Point3f(0.0f0), d=wi)
    @real_assert norm(ray.d) ≈ 1.0f0
    light_normal = Normal3f(ray.d)
    pdf_pos = 1.0f0
    pdf_dir = uniform_sphere_pdf()
    radiance = e.scale * e.env_map.sample(wi)
    return radiance, ray, light_normal, pdf_pos, pdf_dir
end

"""
Emmited light if ray hit an area light source.
By default light sources have no area.
"""
function le(env::EnvironmentLight, ray::AbstractRay)
    env.env_map(Vec2f(0f0))
end
