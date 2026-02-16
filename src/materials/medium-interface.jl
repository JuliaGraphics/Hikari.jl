# SetKey and MediumInterface types
# Extracted to allow inclusion before spectral-eval.jl
# which needs MediumInterface for BSDF forwarding

# ============================================================================
# Medium Index Type (internal - used at runtime for GPU dispatch)
# ============================================================================

@propagate_inbounds is_vacuum(idx::SetKey) = Raycore.is_invalid(idx)
@propagate_inbounds has_medium(idx::SetKey) = Raycore.is_valid(idx)

# ============================================================================
# User-facing MediumInterface (stores actual Medium objects)
# ============================================================================

"""
    MediumInterface{M<:Material, I, O}

Material wrapper that combines a surface BSDF with medium boundaries.
Surfaces define boundaries between media.

Emission is handled separately by DiffuseAreaLight (registered per-triangle
in scene.lights, not on the material).

# Fields
- `material`: The underlying BSDF material (e.g., MatteMaterial, GlassMaterial)
- `inside`: Medium for inside the surface, or `nothing` for vacuum
- `outside`: Medium for outside the surface, or `nothing` for vacuum

# Usage
```julia
# Glass with fog inside
MediumInterface(GlassMaterial(...); inside=fog)

# Simple material with no medium transition
MediumInterface(MatteMaterial(Kd, σ))
```
"""
struct MediumInterface{M<:Material, I, O, E} <: Material
    material::M
    inside::I      # Medium or Nothing
    outside::O     # Medium or Nothing
    emission::E    # NamedTuple (Le, scale, two_sided) or Nothing
end

# Convenience constructors
function MediumInterface(material::M; inside=nothing, outside=nothing, emission=nothing) where {M<:Material}
    MediumInterface{M, typeof(inside), typeof(outside), typeof(emission)}(material, inside, outside, emission)
end

function MediumInterface(material::M, medium) where {M<:Material}
    # Same medium on both sides (e.g., object embedded in fog)
    MediumInterface{M, typeof(medium), typeof(medium), Nothing}(material, medium, medium, nothing)
end

"""Check if this interface represents a medium transition"""
@propagate_inbounds function is_medium_transition(mi::MediumInterface)
    mi.inside !== mi.outside
end

# ============================================================================
# Internal MediumInterfaceIdx (stores SetKey for GPU dispatch)
# ============================================================================

"""
    MediumInterfaceIdx

Internal index wrapper with material and medium indices
for GPU-compatible dispatch. Created during scene building from `MediumInterface`.

# Fields (all SetKey indices)
- `material`: SetKey into materials container (BSDF material)
- `inside`: SetKey for inside medium (invalid = vacuum)
- `outside`: SetKey for outside medium (invalid = vacuum)
"""
struct MediumInterfaceIdx
    material::SetKey
    inside::SetKey
    outside::SetKey
end

"""Check if this interface represents a medium transition"""
@propagate_inbounds function is_medium_transition(mi::MediumInterfaceIdx)
    # Compare both type_idx and vec_idx to check if indices point to different media
    mi.inside.type_idx != mi.outside.type_idx || mi.inside.vec_idx != mi.outside.vec_idx
end

# Accessors for medium indices
@propagate_inbounds get_inside_medium(mi::MediumInterfaceIdx) = mi.inside
@propagate_inbounds get_outside_medium(mi::MediumInterfaceIdx) = mi.outside

"""
    get_medium_index(mi::MediumInterfaceIdx, wi::Vec3f, n::Vec3f) -> SetKey

Determine which medium a ray enters based on direction and surface normal.
- If dot(wi, n) > 0: ray going in direction of normal -> outside medium
- If dot(wi, n) < 0: ray going against normal -> inside medium
"""
@propagate_inbounds function get_medium_index(mi::MediumInterfaceIdx, wi::Vec3f, n::Vec3f)
    if dot(wi, n) > 0f0
        mi.outside
    else
        mi.inside
    end
end

@propagate_inbounds function get_crossing_medium(mi::MediumInterfaceIdx, entering::Bool)
    entering ? mi.inside : mi.outside
end
