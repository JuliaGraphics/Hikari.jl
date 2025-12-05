struct SpotLight{S<:Spectrum} <: Light
    flags::LightFlags
    light_to_world::Transformation
    world_to_light::Transformation
    position::Point3f
    i::S
    cos_total_width::Float32
    cos_falloff_start::Float32

    function SpotLight(
        light_to_world::Transformation, i::S,
        total_width::Float32, falloff_start::Float32,
    ) where S<:Spectrum
        new{S}(
            LightδPosition, light_to_world, inv(light_to_world),
            light_to_world(Point3f(0f0)), i,
            cos(deg2rad(total_width)), cos(deg2rad(falloff_start)),
        )
    end
end

function sample_li(s::SpotLight, ref::Interaction, ::Point2f)
    wi = normalize(Vec3f(s.position - ref.p))
    pdf = 1f0
    visibility = VisibilityTester(
        ref, Interaction(s.position, ref.time, Vec3f(0f0), Normal3f(0f0)),
    )
    radiance = s.i * falloff(s, -wi) / distance_squared(s.position, ref.p)
    radiance, wi, pdf, visibility
end

function falloff(s::SpotLight, w::Vec3f)::Float32
    wl = normalize(s.world_to_light(w))
    cosθ = wl[3]
    cosθ < s.cos_total_width && return 0f0
    cosθ ≥ s.cos_falloff_start && return 1f0
    # Compute falloff inside spotlight cone.
    δ = (cosθ - s.cos_total_width) / (s.cos_falloff_start - s.cos_total_width)
    δ^4
end

@inline function power(s::SpotLight)
    s.i * 2f0 * π * (1f0 - 0.5f0 * (s.cos_falloff_start + s.cos_total_width))
end

function sample_le(
        s::SpotLight, u1::Point2f, ::Point2f, ::Float32,
    )::Tuple{RGBSpectrum,Ray,Normal3f,Float32,Float32}

    w = s.light_to_world(uniform_sample_cone(u1, s.cos_total_width))
    ray = Ray(o=s.position, d=w)
    light_normal = Normal3f(ray.d)
    pdf_pos = 1f0
    pdf_dir = uniform_cone_pdf(s.cos_total_width)
    s.i * falloff(s, ray.d), ray, light_normal, pdf_pos, pdf_dir
end
