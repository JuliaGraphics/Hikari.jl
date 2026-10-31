struct GeometricPrimitive{T<:AbstractShape} <: Primitive
    shape::T
    material::Maybe{M} where M<:Material

    function GeometricPrimitive(
        shape::T, material::Maybe{M} = nothing,
    ) where {T<:AbstractShape,M<:Material}
        new{T}(shape, material)
    end
end

function intersect!(
    p::GeometricPrimitive{T}, ray::Union{Ray,RayDifferentials},
) where T<:AbstractShape
    intersects, t_hit, interaction = intersect(p.shape, ray)
    !intersects && return false, nothing
    ray.t_max = t_hit
    interaction.primitive = p
    true, interaction
end

@inline function intersect_p(
    p::GeometricPrimitive, ray::Union{Ray,RayDifferentials},
)
    intersect_p(p.shape, ray)
end
@inline world_bound(p::GeometricPrimitive) = world_bound(p.shape)

function compute_scattering!(
    p::GeometricPrimitive, si::SurfaceInteraction,
    allow_multiple_lobes::Bool, ::Type{T},
) where T<:TransportMode
    !(p.material isa Nothing) && p.material(si, allow_multiple_lobes, T)
    @real_assert (si.core.n ⋅ si.shading.n) ≥ 0f0
end
