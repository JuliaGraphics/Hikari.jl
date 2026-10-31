struct Sphere <: AbstractShape
    core::ShapeCore

    radius::Float32
    # Implicit constraints.
    z_min::Float32
    z_max::Float32
    # Parametric constraints.
    θ_min::Float32
    θ_max::Float32
    ϕ_max::Float32

    function Sphere(
        core::ShapeCore, radius::Float32,
        z_min::Float32, z_max::Float32, ϕ_max::Float32,
    )
        new(
            core, radius,
            clamp(min(z_min, z_max), -radius, radius),
            clamp(max(z_min, z_max), -radius, radius),
            acos(clamp(min(z_min, z_max) / radius, -1f0, 1f0)),
            acos(clamp(max(z_min, z_max) / radius, -1f0, 1f0)),
            deg2rad(clamp(ϕ_max, 0f0, 360f0)),
        )
    end
end

function Sphere(core::ShapeCore, radius::Float32, ϕ_max::Float32)
    Sphere(core, radius, -radius, radius, ϕ_max)
end

function object_bound(s::Sphere)
    Bounds3(
        Point3f(-s.radius, -s.radius, s.z_min),
        Point3f(s.radius, s.radius, s.z_max),
    )
end

function solve_quadratic(a::Float32, b::Float32, c::Float32)
    # Find disriminant.
    d = b^2 - 4 * a * c
    if d < 0
        return false, NaN32, NaN32
    end
    d = sqrt(d)
    # Compute roots.
    q = -0.5f0 * (b + (b < 0 ? -d : d))
    t0 = q / a
    t1 = c / q
    if t0 > t1
        t0, t1 = t1, t0
    end
    true, t0, t1
end

function refine_intersection(p::Point, s::Sphere)
    p *= s.radius ./ distance(Point3f(0), p)
    p[1] ≈ 0 && p[2] ≈ 0 && (p = Point3f(1f-6 * s.radius, p[2], p[3]))
    p
end

"""
Test if hit point exceeds clipping parameters of the sphere.
"""
function test_clipping(s::Sphere, p::Point3f, ϕ::Float32)::Bool
    (s.z_min > -s.radius && p[3] < s.z_min) ||
        (s.z_max < s.radius && p[3] > s.z_max) ||
        ϕ > s.ϕ_max
end

function compute_ϕ(p::Point3f)::Float32
    ϕ = atan(p[2], p[1])
    ϕ < 0f0 && (ϕ += 2f0 * π)
    ϕ
end

function precompute_ϕ(p::Point3f)
    z_radius = sqrt(p[1] * p[1] + p[2] * p[2])
    inv_z_radius = 1f0 / z_radius
    cos_ϕ = p[1] * inv_z_radius
    sin_ϕ = p[2] * inv_z_radius
    sin_ϕ, cos_ϕ
end

"""
Compute partial derivatives of intersection point in parametric form.
"""
function ∂p(s::Sphere, p::Point3f, θ::Float32, sin_ϕ::Float32, cos_ϕ::Float32)
    ∂p∂u = Vec3f(-s.ϕ_max * p[2], s.ϕ_max * p[1], 0f0)
    ∂p∂v = (s.θ_max - s.θ_min) * Vec3f(
        p[3] * cos_ϕ, p[3] * sin_ϕ, -s.radius * sin(θ),
    )
    ∂p∂u, ∂p∂v, sin_ϕ, cos_ϕ
end

function ∂n(
    s::Sphere, p::Point3f,
    sin_ϕ::Float32, cos_ϕ::Float32,
    ∂p∂u::Vec3f, ∂p∂v::Vec3f,
)
    ∂2p∂u2 = -s.ϕ_max * s.ϕ_max * Vec3f(p[1], p[2], 0f0)
    ∂2p∂u∂v = (s.θ_max - s.θ_min) * p[3] * s.ϕ_max * Vec3f(-sin_ϕ, cos_ϕ, 0f0)
    ∂2p∂v2 = (s.θ_max - s.θ_min)^2 * -p
    # Compute coefficients for fundamental forms.
    E = ∂p∂u ⋅ ∂p∂u
    F = ∂p∂u ⋅ ∂p∂v
    G = ∂p∂v ⋅ ∂p∂v
    n = normalize(∂p∂u × ∂p∂v)
    e = n ⋅ ∂2p∂u2
    f = n ⋅ ∂2p∂u∂v
    g = n ⋅ ∂2p∂v2
    # Compute derivatives from fundamental form coefficients.
    inv_egf = 1f0 / (E * G - F * F)
    ∂n∂u = Normal3f(
        (f * F - e * G) * inv_egf * ∂p∂u +
        (e * F - f * E) * inv_egf * ∂p∂v,
    )
    ∂n∂v = Normal3f(
        (g * F - f * G) * inv_egf * ∂p∂u +
        (f * F - g * E) * inv_egf * ∂p∂v,
    )
    ∂n∂u, ∂n∂v
end

function intersect(
    s::Sphere, ray::Union{Ray,RayDifferentials}, ::Bool = false,
)
    # Transform ray to object space.
    or = s.core.world_to_object(ray)
    # Substitute ray into sphere equation.
    a = norm(or.d)^2
    b = 2 * or.o ⋅ or.d
    c = norm(or.o)^2 - s.radius^2
    # Solve quadratic equation for t.
    exists, t0, t1 = solve_quadratic(a, b, c)
    !exists && return false, nothing, nothing
    (t0 > or.t_max || t1 < 0f0) && return false, nothing, nothing
    t0 < 0 && (t0 = t1)

    shape_hit = t0
    hit_point = refine_intersection(or(t0), s)
    ϕ = compute_ϕ(hit_point)
    # Test sphere intersection against clipping parameters.
    if test_clipping(s, hit_point, ϕ)
        shape_hit = t1
        hit_point = refine_intersection(or(t1), s)
        ϕ = compute_ϕ(hit_point)
        test_clipping(s, hit_point, ϕ) && return false, nothing, nothing
    end
    # Find parametric representation of hit point.
    u = ϕ / s.ϕ_max
    θ = acos(clamp(hit_point[3] / s.radius, -1f0, 1f0))
    v = (θ - s.θ_min) / (s.θ_max - s.θ_min)

    sin_ϕ, cos_ϕ = precompute_ϕ(hit_point)
    ∂p∂u, ∂p∂v = ∂p(s, hit_point, θ, sin_ϕ, cos_ϕ)
    ∂n∂u, ∂n∂v = ∂n(s, hit_point, sin_ϕ, cos_ϕ, ∂p∂u, ∂p∂v)

    interaction = s.core.object_to_world(SurfaceInteraction(
        hit_point, ray.time, -ray.d, Point2f(u, v),
        ∂p∂u, ∂p∂v, ∂n∂u, ∂n∂v, s,
    ))
    true, shape_hit, interaction
end

function intersect_p(
    s::Sphere, ray::Union{Ray,RayDifferentials}, ::Bool = false,
)::Bool
    # Transform ray to object space.
    or = s.core.world_to_object(ray)
    # Substitute ray into sphere equation.
    a = norm(or.d)^2
    b = 2f0 * or.o ⋅ or.d
    c = norm(or.o)^2 - s.radius^2
    # Solve quadratic equation for t.
    exists, t0, t1 = solve_quadratic(a, b, c)
    !exists && return false
    (t0 > or.t_max || t1 < 0f0) && return false
    t0 < 0 && (t0 = t1)

    hit_point = refine_intersection(or(t0), s)
    ϕ = compute_ϕ(hit_point)
    # Test sphere intersection against clipping parameters.
    if test_clipping(s, hit_point, ϕ)
        shape_hit = t1
        hit_point = refine_intersection(or(t1), s)
        ϕ = compute_ϕ(hit_point)
        test_clipping(s, hit_point, ϕ) && return false
    end
    true
end

@inline area(s::Sphere) = s.ϕ_max * s.radius * (s.z_max - s.z_min)
