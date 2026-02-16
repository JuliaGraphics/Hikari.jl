# Light Bounds for BVH Light Sampler
# Following pbrt-v4's LightBounds, DirectionCone, and Importance computation
#
# References:
# - pbrt-v4/src/pbrt/lights.h: LightBounds class, Union()
# - pbrt-v4/src/pbrt/lightsamplers.h: CompactLightBounds::Importance()
# - pbrt-v4/src/pbrt/util/vecmath.h: DirectionCone, BoundSubtendedDirections
# - pbrt-v4/src/pbrt/util/vecmath.cpp: DirectionCone Union()

# ============================================================================
# DirectionCone
# ============================================================================

"""
    DirectionCone

Represents a cone of directions with axis `w` and half-angle `cosθ`.
- `cosθ = 1`: point direction (degenerate cone)
- `cosθ = -1`: entire sphere
- `cosθ = Inf32`: empty cone (no directions)

Following pbrt-v4's DirectionCone (vecmath.h:1784-1850).
"""
struct DirectionCone
    w::Vec3f        # Cone axis (normalized)
    cosθ::Float32   # Cosine of half-angle
end

# Empty cone
DirectionCone() = DirectionCone(Vec3f(0f0, 0f0, 1f0), Inf32)
# Point direction
DirectionCone(w::Vec3f) = DirectionCone(normalize(w), 1f0)
# Entire sphere
entire_sphere() = DirectionCone(Vec3f(0f0, 0f0, 1f0), -1f0)

is_empty(c::DirectionCone) = c.cosθ == Inf32

"""
    angle_between(v1, v2) -> Float32

Numerically stable angle between two vectors.
Following pbrt-v4 (vecmath.h:972-977).
"""
@inline function angle_between(v1::Vec3f, v2::Vec3f)::Float32
    if dot(v1, v2) < 0f0
        Float32(π) - 2f0 * asin(clamp(norm(v1 + v2) * 0.5f0, -1f0, 1f0))
    else
        2f0 * asin(clamp(norm(v2 - v1) * 0.5f0, -1f0, 1f0))
    end
end

"""
    Base.union(a::DirectionCone, b::DirectionCone) -> DirectionCone

Find the minimal cone containing both input cones.
Following pbrt-v4 (vecmath.cpp:56-83).
"""
function Base.union(a::DirectionCone, b::DirectionCone)
    is_empty(a) && return b
    is_empty(b) && return a

    # Check if one cone is inside the other
    θ_a = acos(clamp(a.cosθ, -1f0, 1f0))
    θ_b = acos(clamp(b.cosθ, -1f0, 1f0))
    θ_d = angle_between(a.w, b.w)

    min(θ_d + θ_b, Float32(π)) <= θ_a && return a
    min(θ_d + θ_a, Float32(π)) <= θ_b && return b

    # Compute spread angle of merged cone
    θ_o = (θ_a + θ_d + θ_b) * 0.5f0
    θ_o >= Float32(π) && return entire_sphere()

    # Find merged cone's axis by rotating a.w toward b.w
    θ_r = θ_o - θ_a
    wr = a.w × b.w
    len_sq = dot(wr, wr)
    len_sq == 0f0 && return entire_sphere()

    # Rodrigues rotation: rotate a.w by θ_r around axis wr
    axis = normalize(wr)
    sinθ = sin(θ_r)
    cosθ_rot = cos(θ_r)
    w = a.w * cosθ_rot + (axis × a.w) * sinθ + axis * dot(axis, a.w) * (1f0 - cosθ_rot)

    return DirectionCone(normalize(w), cos(θ_o))
end

"""
    bound_subtended_directions(b::Bounds3, p::Point3f) -> DirectionCone

Compute the bounding cone of directions from point `p` to bounding box `b`.
If `p` is inside the box, returns entire sphere.
Following pbrt-v4 (vecmath.h:1815-1828).
"""
@inline function bound_subtended_directions(b::Raycore.Bounds3, p::Point3f)::DirectionCone
    # Bounding sphere
    p_center = (b.p_min + b.p_max) * 0.5f0
    radius_sq = Raycore.distance_squared(b.p_max, p_center)
    d2 = Raycore.distance_squared(p, p_center)

    # Point inside sphere → entire sphere
    d2 < radius_sq && return entire_sphere()

    w = normalize(Vec3f(p_center - p))
    sin2_theta_max = radius_sq / d2
    cos_theta_max = sqrt(max(0f0, 1f0 - sin2_theta_max))
    return DirectionCone(w, cos_theta_max)
end

# ============================================================================
# LightBounds
# ============================================================================

"""
    LightBounds

Stores spatial and directional bounds for a light or group of lights.
Used by BVHLightSampler for importance-based light selection.

Following pbrt-v4's LightBounds (lights.h:104-125).
"""
struct LightBounds
    bounds::Raycore.Bounds3   # Spatial AABB
    w::Vec3f                  # Direction axis (normalized)
    phi::Float32              # Total power/flux
    cosθ_o::Float32           # Opening angle cosine
    cosθ_e::Float32           # Edge cutoff angle cosine
    two_sided::Bool
end

# Default (zero-power)
LightBounds() = LightBounds(Raycore.Bounds3(), Vec3f(0f0, 0f0, 1f0), 0f0, 1f0, 1f0, false)

centroid(lb::LightBounds) = (lb.bounds.p_min + lb.bounds.p_max) * 0.5f0

"""
    Base.union(a::LightBounds, b::LightBounds) -> LightBounds

Merge two LightBounds. Following pbrt-v4 (lights.h:137-153).
"""
function Base.union(a::LightBounds, b::LightBounds)
    a.phi == 0f0 && return b
    b.phi == 0f0 && return a

    cone = union(DirectionCone(a.w, a.cosθ_o), DirectionCone(b.w, b.cosθ_o))
    cosθ_o = cone.cosθ
    cosθ_e = min(a.cosθ_e, b.cosθ_e)

    return LightBounds(
        union(a.bounds, b.bounds),
        cone.w,
        a.phi + b.phi,
        cosθ_o,
        cosθ_e,
        a.two_sided | b.two_sided,
    )
end

# ============================================================================
# Importance Computation
# ============================================================================

# Clamped trigonometric subtraction helpers
# cos(a - b) clamped: if a < b (cosA > cosB), returns 1 (angle is 0)
@inline cos_sub_clamped(sinA::Float32, cosA::Float32, sinB::Float32, cosB::Float32) =
    cosA > cosB ? 1f0 : cosA * cosB + sinA * sinB

# sin(a - b) clamped: if a < b, returns 0
@inline sin_sub_clamped(sinA::Float32, cosA::Float32, sinB::Float32, cosB::Float32) =
    cosA > cosB ? 0f0 : sinA * cosB - cosA * sinB

"""
    importance(lb::LightBounds, p::Point3f, n::Vec3f) -> Float32

Compute the importance of a light (region) for shading point `p` with normal `n`.
Pass `n = Vec3f(0)` for medium scattering (no normal term).

Following pbrt-v4's CompactLightBounds::Importance (lightsamplers.h:144-201).
"""
@propagate_inbounds function importance(lb::LightBounds, p::Point3f, n::Vec3f)::Float32
    lb.phi == 0f0 && return 0f0

    # Clamped squared distance to center
    pc = centroid(lb)
    d2 = Raycore.distance_squared(p, pc)
    d2 = max(d2, norm(Raycore.diagonal(lb.bounds)) * 0.5f0)

    # Direction from center to point
    wi = normalize(Vec3f(p - pc))
    cosθ_w = dot(lb.w, wi)
    lb.two_sided && (cosθ_w = abs(cosθ_w))
    sinθ_w = sqrt(max(0f0, 1f0 - cosθ_w^2))

    # Bounding cone from p to bounds
    cosθ_b = bound_subtended_directions(lb.bounds, p).cosθ
    sinθ_b = sqrt(max(0f0, 1f0 - cosθ_b^2))

    # Angle arithmetic: cos(θ_w - θ_o)
    sinθ_o = sqrt(max(0f0, 1f0 - lb.cosθ_o^2))
    cosθ_x = cos_sub_clamped(sinθ_w, cosθ_w, sinθ_o, lb.cosθ_o)
    sinθ_x = sin_sub_clamped(sinθ_w, cosθ_w, sinθ_o, lb.cosθ_o)

    # cos(θ' = θ_x - θ_b) - final projected angle
    cosθp = cos_sub_clamped(sinθ_x, cosθ_x, sinθ_b, cosθ_b)

    cosθp <= lb.cosθ_e && return 0f0

    imp = lb.phi * cosθp / d2

    # Surface normal cosine factor
    if n != Vec3f(0f0)
        cosθ_i = abs(dot(wi, n))
        sinθ_i = sqrt(max(0f0, 1f0 - cosθ_i^2))
        cosθp_i = cos_sub_clamped(sinθ_i, cosθ_i, sinθ_b, cosθ_b)
        imp *= cosθp_i
    end

    return max(imp, 0f0)
end

# ============================================================================
# light_bounds() per light type
# ============================================================================

"""
    light_bounds(light) -> Union{LightBounds, Nothing}

Compute LightBounds for a light. Returns `nothing` for infinite lights.
"""
light_bounds(::Light) = nothing  # Default: infinite lights have no bounds

# PointLight: isotropic point emitter
function light_bounds(light::PointLight)
    p = light.position
    phi = 4f0 * Float32(π) * light.scale * luminance(light.i)
    LightBounds(
        Raycore.Bounds3(p, p),
        Vec3f(0f0, 0f0, 1f0),  # Arbitrary direction (isotropic)
        phi,
        Float32(cos(π)),        # cosθ_o = -1 (radiates all directions)
        Float32(cos(π / 2)),    # cosθ_e = 0
        false,
    )
end

# SpotLight: directional point emitter
function light_bounds(light::SpotLight)
    p = light.position
    # Direction in world space: +Z in local space transformed to world
    w = normalize(Vec3f(light.light_to_world(Vec3f(0f0, 0f0, 1f0))))
    phi = 4f0 * Float32(π) * light.scale * luminance(light.i)

    # cosθ_e = cos(acos(cosFalloffEnd) - acos(cosFalloffStart))
    # Following pbrt-v4 lights.cpp:1374
    cosθ_e = Float32(cos(acos(light.cos_total_width) - acos(light.cos_falloff_start)))
    # Allow slop for fp round-off (pbrt-v4 lights.cpp:1377-1378)
    if cosθ_e == 1f0 && light.cos_total_width != light.cos_falloff_start
        cosθ_e = 0.999f0
    end

    LightBounds(
        Raycore.Bounds3(p, p),
        w,
        phi,
        light.cos_falloff_start,
        cosθ_e,
        false,
    )
end

# DiffuseAreaLight: triangle emitter
function light_bounds(light::DiffuseAreaLight)
    # Bounds from vertices
    bounds = Raycore.Bounds3(light.vertices[1], light.vertices[1])
    bounds = union(bounds, Raycore.Bounds3(light.vertices[2], light.vertices[2]))
    bounds = union(bounds, Raycore.Bounds3(light.vertices[3], light.vertices[3]))

    # Power: π * (twoSided ? 2 : 1) * area * scale * luminance(Le)
    sided_factor = light.two_sided ? 2f0 : 1f0
    Le_lum = light.Le isa RGBSpectrum ? luminance(light.Le) : light.scale
    phi = Float32(π) * sided_factor * light.area * light.scale * Le_lum

    # Normal direction (triangle is flat → cosθ_o = 1)
    w = Vec3f(light.normal)

    LightBounds(
        bounds,
        w,
        phi,
        1f0,                    # cosθ_o = 1 (single normal direction)
        Float32(cos(π / 2)),    # cosθ_e = 0 (no edge cutoff)
        light.two_sided,
    )
end
