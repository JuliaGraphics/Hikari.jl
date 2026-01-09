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

# Trait functions for GPU-compatible type checking (avoids `isa` runtime dispatch)
"""Check if a material is a MediumInterfaceIdx (GPU-compatible, no runtime type introspection)"""
@inline is_medium_interface_idx(::MediumInterfaceIdx) = true
@inline is_medium_interface_idx(::Material) = false

# GPU-safe accessors for medium indices (return vacuum for non-interface materials)
"""Get the inside medium index from a material (vacuum if not a MediumInterfaceIdx)"""
@inline get_inside_medium(mi::MediumInterfaceIdx) = mi.inside
@inline get_inside_medium(::Material) = MediumIndex()

"""Get the outside medium index from a material (vacuum if not a MediumInterfaceIdx)"""
@inline get_outside_medium(mi::MediumInterfaceIdx) = mi.outside
@inline get_outside_medium(::Material) = MediumIndex()

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

# GPU-safe version that works with any material type
"""Get medium index based on direction for any material (vacuum for non-interface)"""
@inline function get_crossing_medium(material, entering::Bool)
    if entering
        get_inside_medium(material)
    else
        get_outside_medium(material)
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

# ============================================================================
# Helper: Extract media and convert materials
# ============================================================================

"""
    extract_media_and_convert(materials_list) -> (converted_materials, media_tuple)

Extract all media from MediumInterface materials and convert them to indexed versions.
Returns the converted materials and a tuple of media for VolPath rendering.

This is useful for external integrations (like TraceMakie) that build scenes
without using the full Hikari.Scene constructor.

# Example
```julia
materials = [matte, glass_with_fog, ...]
converted, media = extract_media_and_convert(materials)
# converted contains MediumInterfaceIdx instead of MediumInterface
# media is a tuple of Medium objects for VolPath
```
"""
function extract_media_and_convert(materials_list::Vector{<:Material})
    # Collect unique media from MediumInterface materials
    media_list = []
    medium_to_index = Dict{Any, Int}()

    for mat in materials_list
        if mat isa MediumInterface
            for medium in (mat.inside, mat.outside)
                if medium !== nothing && !haskey(medium_to_index, medium)
                    push!(media_list, medium)
                    medium_to_index[medium] = length(media_list)
                end
            end
        end
    end

    # Convert MediumInterface -> MediumInterfaceIdx with proper indices
    converted_materials = [to_indexed(mat, medium_to_index) for mat in materials_list]

    # Build media tuple (empty if no media)
    media_tuple = isempty(media_list) ? () : Tuple(media_list)

    return converted_materials, media_tuple
end
