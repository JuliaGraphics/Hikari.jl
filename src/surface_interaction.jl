struct Interaction
    """
    Intersection point in world coordinates.
    """
    p::Point3f
    """
    Time of intersection.
    """
    time::Float32
    """
    Negative direction of ray (for ray-shape interactions)
    in world coordinates.
    """
    wo::Vec3f
    """
    Surface normal at the point in world coordinates.
    """
    n::Normal3f
end

Interaction() = Interaction(Point3f(Inf), 0f0, Vec3f(0f0), Normal3f(0f0))

struct ShadingInteraction
    n::Normal3f
    ∂p∂u::Vec3f
    ∂p∂v::Vec3f
    ∂n∂u::Normal3f
    ∂n∂v::Normal3f
end

struct SurfaceInteraction
    core::Interaction
    shading::ShadingInteraction
    uv::Point2f

    ∂p∂u::Vec3f
    ∂p∂v::Vec3f
    ∂n∂u::Normal3f
    ∂n∂v::Normal3f

    ∂u∂x::Float32
    ∂u∂y::Float32
    ∂v∂x::Float32
    ∂v∂y::Float32
    ∂p∂x::Vec3f
    ∂p∂y::Vec3f

    SurfaceInteraction() = new()

    function SurfaceInteraction(
            core::Interaction, shading::ShadingInteraction, uv,
            ∂p∂u, ∂p∂v, ∂n∂u, ∂n∂v,
            ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y,
            ∂p∂x, ∂p∂y,
        )
        new(
            core, shading, uv, ∂p∂u, ∂p∂v, ∂n∂u, ∂n∂v,
            ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y, ∂p∂x, ∂p∂y
        )
    end
end

@inline function SurfaceInteraction(
        p::Point3f, time::Float32, wo::Vec3f, uv::Point2f,
        ∂p∂u::Vec3f, ∂p∂v::Vec3f, ∂n∂u::Normal3f, ∂n∂v::Normal3f, reverse_normal::Bool
    )

    n = normalize((∂p∂u × ∂p∂v))

    if reverse_normal
        n *= -1
    end

    core = Interaction(p, time, wo, n)
    shading = ShadingInteraction(n, ∂p∂u, ∂p∂v, ∂n∂u, ∂n∂v)
    return SurfaceInteraction(
        core, shading, uv, ∂p∂u, ∂p∂v, ∂n∂u, ∂n∂v,
        0f0, 0f0, 0f0, 0f0, Vec3f(0f0), Vec3f(0f0)
    )
end

@inline function SurfaceInteraction(
        normal, hitpoint::Point3f, time::Float32, wo::Vec3f, uv::Point2f,
        ∂p∂u::Vec3f, ∂p∂v::Vec3f, ∂n∂u::Normal3f, ∂n∂v::Normal3f
    )
    core = Interaction(hitpoint, time, wo, normal)
    shading = ShadingInteraction(normal, ∂p∂u, ∂p∂v, ∂n∂u, ∂n∂v)
    return SurfaceInteraction(
        core, shading, uv, ∂p∂u, ∂p∂v, ∂n∂u, ∂n∂v,
        0.0f0, 0.0f0, 0.0f0, 0.0f0, Vec3f(0.0f0), Vec3f(0.0f0)
    )
end

@inline function set_shading_geometry(
        si::SurfaceInteraction, tangent::Vec3f, bitangent::Vec3f,
        ∂n∂u::Normal3f, ∂n∂v::Normal3f, orientation_is_authoritative::Bool,
    )
    shading_n = normalize(tangent × bitangent)
    core_n = si.core.n
    if orientation_is_authoritative
        core_n = face_forward(si.core.n, si.shading.n)
    else
        shading_n = face_forward(si.shading.n, si.core.n)
    end

    shading = ShadingInteraction(shading_n, tangent, bitangent, ∂n∂u, ∂n∂v)
    core = Interaction(si.core.p, si.core.time, si.core.wo, core_n)
    return SurfaceInteraction(si; shading=shading, core=core)
end

is_surface_interaction(i::Interaction) = i.n != Normal3f(0)

@inline function SurfaceInteraction(
        si::SurfaceInteraction;
        core=si.core , shading=si.shading, uv=si.uv, ∂p∂u=si.∂p∂u, ∂p∂v=si.∂p∂v,
        ∂n∂u=si.∂n∂u, ∂n∂v=si.∂n∂v, ∂u∂x=si.∂u∂x, ∂u∂y=si.∂u∂y,
        ∂v∂x=si.∂v∂x, ∂v∂y=si.∂v∂y, ∂p∂x=si.∂p∂x, ∂p∂y=si.∂p∂y
    )
    SurfaceInteraction(
        core, shading, uv, ∂p∂u, ∂p∂v, ∂n∂u, ∂n∂v, ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y, ∂p∂x, ∂p∂y
    )
end

"""
Compute partial derivatives needed for computing sampling rates
for things like texture antialiasing.
"""
@inline function compute_differentials(si::SurfaceInteraction, ray::RayDifferentials)

    if !ray.has_differentials
        return SurfaceInteraction(si;
            ∂u∂x=0.0f0, ∂v∂x=0.0f0, ∂u∂y=0.0f0, ∂v∂y=0f0, ∂p∂x=Vec3f(0.0f0), ∂p∂y=Vec3f(0.0f0)
        )
    end

    # Estimate screen change in p and (u, v).
    # Compute auxiliary intersection points with plane.

    d = -(si.core.n ⋅ si.core.p)
    tx = (-(si.core.n ⋅ ray.rx_origin) - d) / (si.core.n ⋅ ray.rx_direction)
    ty = (-(si.core.n ⋅ ray.ry_origin) - d) / (si.core.n ⋅ ray.ry_direction)
    px = ray.rx_origin + tx * ray.rx_direction
    py = ray.ry_origin + ty * ray.ry_direction

    ∂p∂x = px - si.core.p
    ∂p∂y = py - si.core.p
    # Compute (u, v) offsets at auxiliary points.
    # Choose two dimensions for ray offset computation.
    n = abs.(si.core.n)
    if n[1] > n[2] && n[1] > n[3]
        dim = Point2(2, 3)
    elseif n[2] > n[3]
        dim = Point2(1, 3)
    else
        dim = Point2(1, 2)
    end
    # Initialization for offset computation.
    a = Mat2f(dim[1], dim[1], dim[2], dim[2])
    bx = Point2f(px[dim[1]] - si.core.p[dim[1]], px[dim[2]] - si.core.p[dim[2]])
    by = Point2f(py[dim[1]] - si.core.p[dim[1]], py[dim[2]] - si.core.p[dim[2]])
    sx = a \ bx
    sy = a \ by

    ∂u∂x, ∂v∂x = any(isnan.(sx)) ? (0f0, 0f0) : sx
    ∂u∂y, ∂v∂y = any(isnan.(sy)) ? (0f0, 0f0) : sy
    return SurfaceInteraction(si; ∂u∂x, ∂v∂x, ∂u∂y, ∂v∂y, ∂p∂x, ∂p∂y)
end

"""
If an intersection was found, it is necessary to determine, how
the surface's material scatters light.
`compute_scattering!` method evaluates texture functions to determine
surface properties and then initializing a representation of the BSDF
at the point.
"""
@inline function compute_scattering!(
        primitive, si::SurfaceInteraction, ray::RayDifferentials,
        allow_multiple_lobes::Bool = false, transport = Radiance,
    )
    si = compute_differentials(si, ray)
    return si, primitive(si, allow_multiple_lobes, transport)
end

@inline function le(::SurfaceInteraction, ::Vec3f)::RGBSpectrum
    # TODO right now return 0, since there is no area lights implemented.
    RGBSpectrum(0f0)
end

@inline function apply(t::Transformation, si::Interaction)
    return Interaction(
        t(si.p),
        si.time,
        normalize(t(si.wo)),
        normalize(t(si.n)),
    )
end

@inline function apply(t::Transformation, si::ShadingInteraction)
    n = normalize(t(si.n))
    ∂p∂u = t(si.∂p∂u)
    ∂p∂v = t(si.∂p∂v)
    ∂n∂u = t(si.∂n∂u)
    ∂n∂v = t(si.∂n∂v)
    return ShadingInteraction(n, ∂p∂u, ∂p∂v, ∂n∂u, ∂n∂v)
end

@inline function apply(t::Transformation, si::SurfaceInteraction)
    # TODO compute shading normal separately
    core = apply(t, si.core)
    shading = apply(t, si.shading)
    ∂p∂u = t(si.∂p∂u)
    ∂p∂v = t(si.∂p∂v)
    ∂n∂u = t(si.∂n∂u)
    ∂n∂v = t(si.∂n∂v)
    ∂p∂x = t(si.∂p∂x)
    ∂p∂y = t(si.∂p∂y)
    return SurfaceInteraction(
        core, shading, si.uv, ∂p∂u, ∂p∂v, ∂n∂u, ∂n∂v,
        si.∂u∂x, si.∂u∂y, si.∂v∂x, si.∂v∂y, ∂p∂x, ∂p∂y
    )
end

# Trace-specific spawn_ray functions for SurfaceInteraction
# RayCaster has spawn_ray for its Interaction type, but we need these for our SurfaceInteraction
@inline function spawn_ray(
        p0::Interaction, p1::Interaction, δ::Float32 = 1f-6,
    )::Ray
    direction = p1.p - p0.p
    origin = p0.p .+ δ .* direction
    return Ray(origin, direction, Inf32, p0.time)
end

@inline function spawn_ray(p0::SurfaceInteraction, p1::Interaction)::Ray
    spawn_ray(p0.core, p1)
end

@inline function spawn_ray(
        si::SurfaceInteraction, direction::Vec3f, δ::Float32 = 1f-6,
    )::Ray
    origin = si.core.p .+ δ .* direction
    return Ray(o=origin, d=direction, time=si.core.time)
end
