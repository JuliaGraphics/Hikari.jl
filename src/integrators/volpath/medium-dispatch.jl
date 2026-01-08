# Type-stable dispatch functions for heterogeneous medium tuples
# Uses @generated functions like material-dispatch.jl

# ============================================================================
# Medium Point Sampling Dispatch
# ============================================================================

"""
    sample_point_dispatch(table, media, idx, p, λ) -> MediumProperties

Type-stable dispatch for sampling medium properties at a point.
"""
@inline @generated function sample_point_dispatch(
    table::RGBToSpectrumTable,
    media::NTuple{N,Any},
    idx::Int32,
    p::Point3f,
    λ::Wavelengths
) where {N}
    branches = [quote
        @inbounds if idx === Int32($i)
            return @inline sample_point(table, media[$i], p, λ)
        end
    end for i in 1:N]

    quote
        $(branches...)
        return MediumProperties()
    end
end

# ============================================================================
# Medium Majorant Dispatch
# ============================================================================

"""
    get_majorant_dispatch(table, media, idx, ray, t_min, t_max, λ) -> RayMajorantSegment

Type-stable dispatch for getting ray majorant from medium.
"""
@inline @generated function get_majorant_dispatch(
    table::RGBToSpectrumTable,
    media::NTuple{N,Any},
    idx::Int32,
    ray::Raycore.Ray,
    t_min::Float32,
    t_max::Float32,
    λ::Wavelengths
) where {N}
    branches = [quote
        @inbounds if idx === Int32($i)
            return @inline get_majorant(table, media[$i], ray, t_min, t_max, λ)
        end
    end for i in 1:N]

    quote
        $(branches...)
        return RayMajorantSegment()
    end
end

# ============================================================================
# Medium Emissive Check Dispatch
# ============================================================================

"""
    is_emissive_dispatch(media, idx) -> Bool

Type-stable dispatch for checking if medium is emissive.
"""
@inline @generated function is_emissive_dispatch(
    media::NTuple{N,Any},
    idx::Int32
) where {N}
    branches = [quote
        @inbounds if idx === Int32($i)
            return @inline is_emissive(media[$i])
        end
    end for i in 1:N]

    quote
        $(branches...)
        return false
    end
end

# ============================================================================
# Single Medium Fallbacks (for non-tuple media)
# ============================================================================

@inline function sample_point_dispatch(
    table::RGBToSpectrumTable,
    medium::Medium,
    idx::Int32,
    p::Point3f,
    λ::Wavelengths
)
    return sample_point(table, medium, p, λ)
end

@inline function get_majorant_dispatch(
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

@inline function is_emissive_dispatch(medium::Medium, idx::Int32)
    return is_emissive(medium)
end

# MediumIndex is defined in media.jl
