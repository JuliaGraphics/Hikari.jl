struct AmbientLight{S<:Spectrum} <: Light
    """
    Since ambient lights emit light uniformly in all directions
    from all points in the scene, `LightInfinite` flag is used.
    """
    flags::LightFlags
    i::S

    function AmbientLight(i::S) where {S<:Spectrum}
        new{S}(LightInfinite, i,)
    end
end

"""
Compute radiance arriving at `ref.p` interaction point at `ref.time` time
due to the ambient light.

# Args

- `a::AmbientLight`: Ambient light which illuminates the interaction point `ref`.
- `ref::Interaction`: Interaction point for which to compute radiance.
- `u::Point2f`: Sampling point that is ignored for `AmbientLight`,
    since it emits light uniformly.

# Returns

`Tuple{S, Vec3f, Float32, VisibilityTester} where S <: Spectrum`:

    - `S`: Computed radiance.
    - `Vec3f`: Incident direction to the light source `wi`.
    - `Float32`: Probability density for the light sample that was taken.
        For `AmbientLight` it is always `1`.
    - `VisibilityTester`: Initialized visibility tester that holds the
        shadow ray that must be traced to verify that
        there are no occluding objects between the light and reference point.
"""
function sample_li(a::AmbientLight, i::Interaction, ::Point2f, ::AbstractScene)
    pdf = 1.0f0
    radiance = a.i
    inew = Interaction()
    radiance, Vec3f(normalize(i.p)), pdf, VisibilityTester(inew, inew)
end

function sample_le(
        a::AmbientLight, u1::Point2f, ::Point2f, ::Float32,
    )::Tuple{RGBSpectrum,Ray,Normal3f,Float32,Float32}
    ray = Ray(o=Point3f(0.0f0), d=uniform_sample_sphere(u1))
    @real_assert norm(ray.d) ≈ 1.0f0
    light_normal = Normal3f(ray.d)
    pdf_pos = 1.0f0
    pdf_dir = uniform_sphere_pdf()
    return a.i, ray, light_normal, pdf_pos, pdf_dir
end

@propagate_inbounds function power(p::AmbientLight)
    p.i
end
