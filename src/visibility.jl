const CAMERA_RAYS = 0x01
const DIFFUSE_RAYS = 0x02
const GLOSSY_RAYS = 0x04
const TRANSMISSION_RAYS = 0x08
const VOLUME_SCATTER_RAYS = 0x10
const SHADOW_RAYS = 0x20

const LIGHT_SAMPLED_RAYS = DIFFUSE_RAYS | GLOSSY_RAYS | TRANSMISSION_RAYS | VOLUME_SCATTER_RAYS

"""
    Visibility(; camera = true, diffuse = true, glossy = true, transmission = true,
               volume_scatter = true, shadow = true)

Which rays see a mesh or a light, as Blender's Ray Visibility panels.

For a mesh, `camera = false` hides it from rays leaving the camera while reflections,
refractions and diffuse bounces still see it, and `shadow = false` stops it casting shadows.
A ray's type is the lobe it was sampled from: a reflection off a diffuse lobe is diffuse,
any other reflection glossy, and any transmission transmission.

For a light (`push!(scene, light; visibility)`), and for an emissive mesh, a hidden ray type
is the light that type of scattering does not receive: a surface lit by a light hidden from
`diffuse` rays reflects none of it from its diffuse lobes, whether found by light sampling or
by a bounced ray. `camera = false` on an environment light leaves the background black while
it still lights the scene, and `shadow = false` on a light lets its light through every
surface on the way.
"""
struct Visibility
    camera::Bool
    diffuse::Bool
    glossy::Bool
    transmission::Bool
    volume_scatter::Bool
    shadow::Bool
end
Visibility(; camera::Bool = true, diffuse::Bool = true, glossy::Bool = true, transmission::Bool = true,
           volume_scatter::Bool = true, shadow::Bool = true) =
    Visibility(camera, diffuse, glossy, transmission, volume_scatter, shadow)

hidden_bit(visible::Bool, bit::UInt8) = visible ? 0x00 : bit

hidden_rays(v::Visibility) =
    hidden_bit(v.camera, CAMERA_RAYS) | hidden_bit(v.diffuse, DIFFUSE_RAYS) |
    hidden_bit(v.glossy, GLOSSY_RAYS) | hidden_bit(v.transmission, TRANSMISSION_RAYS) |
    hidden_bit(v.volume_scatter, VOLUME_SCATTER_RAYS) | hidden_bit(v.shadow, SHADOW_RAYS)

instance_mask(v::Visibility) = ~hidden_rays(v)

"""The ray types an emissive mesh of visibility `v` keeps its light from, as a light's `hidden` field."""
emitter_hidden_rays(v::Visibility) = hidden_rays(v) & LIGHT_SAMPLED_RAYS

@inline function bounce_ray_type(flags::UInt8)
    is_transmissive(flags) && return TRANSMISSION_RAYS
    flags & BXDF_DIFFUSE != 0x00 && return DIFFUSE_RAYS
    return GLOSSY_RAYS
end

"""
The lobe classes a light `hidden` from some ray types is evaluated without, for light leaving
towards `wi`: all of them through the surface if transmission is hidden, else the diffuse or
glossy ones among reflections, so that light sampling leaves out what the bounced rays of
those types cannot find (`bounce_ray_type`).
"""
@inline function hidden_lobes(hidden::UInt8, transmits::Bool)
    transmits && return hidden & TRANSMISSION_RAYS != 0x00 ? DIFFUSE_RAYS | GLOSSY_RAYS : 0x00
    return hidden & (DIFFUSE_RAYS | GLOSSY_RAYS)
end

"""The cull mask of the shadow rays towards a light `hidden` from some ray types: nothing blocks a light casting no shadows."""
@inline shadow_cull_mask(hidden::UInt8) = hidden & SHADOW_RAYS != 0x00 ? 0x00 : SHADOW_RAYS

"""Whether a ray of `ray_type` sees a light `hidden` from some ray types."""
@inline sees_light(hidden::UInt8, ray_type::UInt8) = hidden & ray_type == 0x00
