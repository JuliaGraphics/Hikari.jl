# Type-based light classification via dispatch (no flags field needed)

# Delta lights emit from a single point or direction (cannot be hit by random rays)
is_δ_light(::Type{<:Light}) = false

# Infinite lights are at infinity (environment maps, sun, sky)
is_infinite_light(::Light) = false
is_infinite_light(::Type{<:Light}) = false

"""
    paints_escaped_rays(light) -> Bool

Whether a ray that hits nothing comes back carrying THIS light's radiance.

Narrower than `is_infinite_light` and not a synonym for it. Both a
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

"""The ray types `light` is hidden from, as `hidden_rays(::Visibility)`."""
@inline hidden_rays(light::Light) = light.hidden
"""`light` hidden from the ray types `hidden`."""
with_hidden(light::L, hidden::UInt8) where {L <: Light} = Base.typename(L).wrapper(light, hidden)
