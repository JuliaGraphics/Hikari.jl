# Type-stable dispatch functions for heterogeneous medium tuples
# Uses @generated functions with unrolled if-branches for GPU compatibility

# ============================================================================
# with_medium - GPU-safe closure dispatch over media tuple
# ============================================================================

"""
    with_medium(f, media, idx, args...)

Execute function `f` with the medium at index `idx`, passing additional `args`.
The function is called as `f(medium, args...)` where `medium` has a concrete type.

This provides type-stable medium dispatch by using compile-time unrolled if-branches.
The closure receives the medium as its first argument plus any additional args,
avoiding variable capture issues on GPU.

# Example
```julia
# Instead of:
mp = sample_point_dispatch(table, media, idx, p, λ)  # May have GPU issues

# Use:
mp = with_medium(_sample_point_helper, media, idx, table, p, λ)
# Where: _sample_point_helper(medium, table, p, λ) = sample_point(table, medium, p, λ)
```
"""
@propagate_inbounds @generated function with_medium(f::F, media::M, idx::Int32, args...) where {F, M <: Tuple}
    N = length(M.parameters)

    if N == 0
        return :(error("with_medium: empty media tuple"))
    end

    # Build unrolled if-else chain
    expr = :(f(media[$N], args...))

    for i in (N-1):-1:1
        expr = quote
            if idx == Int32($i)
                f(media[$i], args...)
            else
                $expr
            end
        end
    end

    return expr
end

# Single medium fallback (no dispatch needed)
@propagate_inbounds function with_medium(f::F, medium::Medium, idx::Int32, args...) where F
    return f(medium, args...)
end

# ============================================================================
# Helper functions for medium dispatch (no variable capture)
# ============================================================================

"""Sample medium properties at a point - helper for with_medium"""
@propagate_inbounds function _sample_point_helper(medium, table::RGBToSpectrumTable, p::Point3f, λ::Wavelengths)
    return sample_point(table, medium, p, λ)
end

"""Get ray majorant from medium - helper for with_medium"""
@propagate_inbounds function _get_majorant_helper(medium, table::RGBToSpectrumTable, ray::Raycore.Ray, t_min::Float32, t_max::Float32, λ::Wavelengths)
    return get_majorant(table, medium, ray, t_min, t_max, λ)
end

"""Check if medium is emissive - helper for with_medium"""
@propagate_inbounds function _is_emissive_helper(medium)
    return is_emissive(medium)
end

# ============================================================================
# Medium Point Sampling Dispatch
# ============================================================================

"""
    sample_point_dispatch(table, media, idx, p, λ) -> MediumProperties

Type-stable dispatch for sampling medium properties at a point.
Uses with_medium pattern for GPU compatibility.
"""
@propagate_inbounds function sample_point_dispatch(
    table::RGBToSpectrumTable,
    media::NTuple{N,Any},
    idx::Int32,
    p::Point3f,
    λ::Wavelengths
) where {N}
    return with_medium(_sample_point_helper, media, idx, table, p, λ)
end

# ============================================================================
# Medium Majorant Dispatch
# ============================================================================

"""
    get_majorant_dispatch(table, media, idx, ray, t_min, t_max, λ) -> RayMajorantSegment

Type-stable dispatch for getting ray majorant from medium.
Uses with_medium pattern for GPU compatibility.
"""
@propagate_inbounds function get_majorant_dispatch(
    table::RGBToSpectrumTable,
    media::NTuple{N,Any},
    idx::Int32,
    ray::Raycore.Ray,
    t_min::Float32,
    t_max::Float32,
    λ::Wavelengths
) where {N}
    return with_medium(_get_majorant_helper, media, idx, table, ray, t_min, t_max, λ)
end

# ============================================================================
# Medium Emissive Check Dispatch
# ============================================================================

"""
    is_emissive_dispatch(media, idx) -> Bool

Type-stable dispatch for checking if medium is emissive.
Uses with_medium pattern for GPU compatibility.
"""
@propagate_inbounds function is_emissive_dispatch(
    media::NTuple{N,Any},
    idx::Int32
) where {N}
    return with_medium(_is_emissive_helper, media, idx)
end

# ============================================================================
# Single Medium Fallbacks (for non-tuple media)
# ============================================================================

@propagate_inbounds function sample_point_dispatch(
    table::RGBToSpectrumTable,
    medium::Medium,
    idx::Int32,
    p::Point3f,
    λ::Wavelengths
)
    return sample_point(table, medium, p, λ)
end

@propagate_inbounds function get_majorant_dispatch(
    table::RGBToSpectrumTable,
    medium::Medium,
    idx::Int32,
    ray::Raycore.Ray,
    t_min::Float32,
    t_max::Float32,
    λ::Wavelengths
)
    return get_majorant(table, medium, ray, t_min, t_max, λ)
end

@propagate_inbounds function is_emissive_dispatch(medium::Medium, idx::Int32)
    return is_emissive(medium)
end

# ============================================================================
# Ray Deflection Dispatch (for gravitational lensing / spacetime curvature)
# ============================================================================

"""Apply ray deflection - helper for with_medium"""
@propagate_inbounds function _apply_deflection_helper(medium, p::Point3f, ray_d::Vec3f, dt::Float32)
    return apply_deflection(medium, p, ray_d, dt)
end

"""
    apply_deflection_dispatch(media, idx, p, ray_d, dt) -> Vec3f

Type-stable dispatch for applying ray deflection in a medium.
Uses with_medium pattern for GPU compatibility.
"""
@propagate_inbounds function apply_deflection_dispatch(
    media::NTuple{N,Any},
    idx::Int32,
    p::Point3f,
    ray_d::Vec3f,
    dt::Float32
) where {N}
    return with_medium(_apply_deflection_helper, media, idx, p, ray_d, dt)
end

# Single medium fallback
@propagate_inbounds function apply_deflection_dispatch(
    medium::Medium,
    idx::Int32,
    p::Point3f,
    ray_d::Vec3f,
    dt::Float32
)
    return apply_deflection(medium, p, ray_d, dt)
end
