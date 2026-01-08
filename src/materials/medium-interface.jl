# MediumIndex and MediumInterface types
# Extracted to allow inclusion before spectral-eval.jl
# which needs MediumInterface for BSDF forwarding

# ============================================================================
# Medium Index Type (internal - used at runtime for GPU dispatch)
# ============================================================================

"""
    MediumIndex

Index into media tuple for runtime dispatch.
- medium_type: Which tuple slot (1-based), 0 = vacuum/no medium

This is an internal type - users should use `MediumInterface` with actual
medium objects, which gets converted to indices during scene building.
"""
struct MediumIndex
    medium_type::Int32   # Which medium type in tuple (0 = vacuum/no medium)
end

MediumIndex() = MediumIndex(Int32(0))

@inline is_vacuum(idx::MediumIndex) = idx.medium_type == Int32(0)
@inline has_medium(idx::MediumIndex) = idx.medium_type > Int32(0)

# ============================================================================
# User-facing MediumInterface (stores actual Medium objects)
# ============================================================================

"""
    MediumInterface{M<:Material, I, O}

User-facing material wrapper that combines a surface BSDF with medium objects.
This follows pbrt-v4's approach where surfaces define boundaries between media.

# Fields
- `material`: The underlying BSDF material (e.g., GlassMaterial)
- `inside`: Medium for inside the surface, or `nothing` for vacuum
- `outside`: Medium for outside the surface, or `nothing` for vacuum

# Usage
```julia
fog = HomogeneousMedium(σ_s=0.5f0, ...)
glass_with_fog = MediumInterface(glass; inside=fog, outside=nothing)
```

During scene building, this is converted to `MediumInterfaceIdx` which uses
integer indices for GPU-compatible dispatch.
"""
struct MediumInterface{M<:Material, I, O} <: Material
    material::M
    inside::I    # Medium or Nothing
    outside::O   # Medium or Nothing
end

# Convenience constructors
function MediumInterface(material::M; inside=nothing, outside=nothing) where {M<:Material}
    MediumInterface{M, typeof(inside), typeof(outside)}(material, inside, outside)
end

function MediumInterface(material::M, medium) where {M<:Material}
    # Same medium on both sides (e.g., object embedded in fog)
    MediumInterface{M, typeof(medium), typeof(medium)}(material, medium, medium)
end

"""Check if this interface represents a medium transition"""
@inline function is_medium_transition(mi::MediumInterface)
    mi.inside !== mi.outside
end

# ============================================================================
# Internal MediumInterfaceIdx (stores MediumIndex for GPU dispatch)
# ============================================================================

"""
    MediumInterfaceIdx{M<:Material}

Internal material wrapper with medium indices for GPU-compatible dispatch.
Created during scene building from `MediumInterface`.

- `material`: The underlying BSDF material
- `inside`: MediumIndex for inside medium (0 = vacuum)
- `outside`: MediumIndex for outside medium (0 = vacuum)
"""
struct MediumInterfaceIdx{M<:Material} <: Material
    material::M
    inside::MediumIndex
    outside::MediumIndex
end

"""Check if this interface represents a medium transition"""
@inline function is_medium_transition(mi::MediumInterfaceIdx)
    mi.inside.medium_type != mi.outside.medium_type
end

"""
    get_medium_index(mi::MediumInterfaceIdx, wi::Vec3f, n::Vec3f) -> MediumIndex

Determine which medium a ray enters based on direction and surface normal.
- If dot(wi, n) > 0: ray going in direction of normal -> outside medium
- If dot(wi, n) < 0: ray going against normal -> inside medium
"""
@inline function get_medium_index(mi::MediumInterfaceIdx, wi::Vec3f, n::Vec3f)
    if dot(wi, n) > 0f0
        mi.outside
    else
        mi.inside
    end
end

# Backwards compatibility alias
@inline get_medium(mi::MediumInterfaceIdx, wi::Vec3f, n::Vec3f) = get_medium_index(mi, wi, n)

# ============================================================================
# Conversion: MediumInterface -> MediumInterfaceIdx
# ============================================================================

"""
    to_indexed(mi::MediumInterface, medium_to_index::Dict) -> MediumInterfaceIdx

Convert a user-facing MediumInterface to an indexed version for GPU dispatch.
`medium_to_index` maps medium objects to their tuple indices.
"""
function to_indexed(mi::MediumInterface, medium_to_index::Dict)
    inside_idx = mi.inside === nothing ? MediumIndex() : MediumIndex(Int32(medium_to_index[mi.inside]))
    outside_idx = mi.outside === nothing ? MediumIndex() : MediumIndex(Int32(medium_to_index[mi.outside]))
    MediumInterfaceIdx(mi.material, inside_idx, outside_idx)
end

# Non-MediumInterface materials pass through unchanged
to_indexed(mat::Material, ::Dict) = mat
