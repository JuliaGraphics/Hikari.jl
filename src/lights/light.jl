@enum LightFlags::UInt8 begin
    LightδPosition = 0b1
    LightδDirection = 0b10
    LightArea = 0b100
    LightInfinite = 0b1000
end

@inline function is_δ_light(flag::LightFlags)::Bool
    flag == LightδPosition || flag == LightδDirection
end

struct VisibilityTester
    p0::Interaction
    p1::Interaction
end

@inline function unoccluded(t::VisibilityTester, scene::Scene)::Bool
    !intersect_p(scene, spawn_ray(t.p0, t.p1))
end

function trace(t::VisibilityTester, scene::Scene)::RGBSpectrum
    ray = spawn_ray(t.p0, t.p1)
    s = RGBSpectrum(1f0)
    while true
        hit, interaction = intersect!(scene, ray)
        # Handle opaque surface.
        if hit && interaction.primitive.material isa Nothing
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
@inline le(::Light, ::Union{Ray,RayDifferentials}) = RGBSpectrum(0f0)
