# Type-based light classification via dispatch (no flags field needed)

# Delta lights emit from a single point or direction (cannot be hit by random rays)
is_δ_light(::Light) = false

# Infinite lights are at infinity (environment maps, sun, sky)
is_infinite_light(::Light) = false
is_infinite_light(::Type{<:Light}) = false

"""
    paints_escaped_rays(light) -> Bool

Whether a ray that hits nothing comes back carrying THIS light's radiance.

Narrower than [`is_infinite_light`](@ref) and not a synonym for it. Both a
`DirectionalLight` and an `EnvironmentLight` are at infinity, but only the
environment map is *visible* along an escaped ray: a directional light is a delta
light in direction, so its `Le` is zero for every ray that misses everything.

The distinction is what decides whether a background colour is composited behind
the render. Asking `is_infinite_light` instead reported that a scene lit by one
directional light already had its background painted, and it does not — the sky
came out black whatever `backgroundcolor` said, because "the render already put
radiance there" was true of the lights' NAMES and false of their contribution.
"""
paints_escaped_rays(::Light) = false
paints_escaped_rays(::Type{<:Light}) = false

struct VisibilityTester
    p0::Interaction
    p1::Interaction
end

@propagate_inbounds function unoccluded(t::VisibilityTester, scene::AbstractScene)::Bool
    # Explicit isinf check to avoid tuple iteration in SPIR-V (any() causes PHI node errors)
    p0_inf = isinf(t.p0.p[1]) || isinf(t.p0.p[2]) || isinf(t.p0.p[3])
    p1_inf = isinf(t.p1.p[1]) || isinf(t.p1.p[2]) || isinf(t.p1.p[3])
    if p0_inf && p1_inf
        return true
    end
    !intersect_p(scene, spawn_ray(t.p0, t.p1))
end

function trace(t::VisibilityTester, scene::AbstractScene)::RGBSpectrum
    ray = spawn_ray(t.p0, t.p1)
    s = RGBSpectrum(1f0)
    while true
        hit, primitive, interaction = intersect!(scene, ray)
        # Handle opaque surface.
        if hit && primitive.material isa Nothing
            return RGBSpectrum(0f0)
        end
        # TODO update transmittance in presence of media in ray
        !hit && break
        ray = spawn_ray(interaction, t.p1)
    end
    s
end

"""
Emmited light if ray hit an area light source.
By default light sources have no area.
"""
@propagate_inbounds le(::Light, ::Union{Ray,RayDifferentials}) = RGBSpectrum(0f0)
