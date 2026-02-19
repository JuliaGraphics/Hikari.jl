# Medium dispatch - uses Raycore.with_index for type-stable dispatch
# Call patterns: with_index(sample_point, media, idx, media, table, p, λ)

# ============================================================================
# Ray Deflection Dispatch (for gravitational lensing / spacetime curvature)
# ============================================================================

@propagate_inbounds function _apply_deflection_helper(medium, p::Point3f, ray_d::Vec3f, dt::Float32)
    return apply_deflection(medium, p, ray_d, dt)
end

@propagate_inbounds function apply_deflection_dispatch(
    media::Raycore.StaticMultiTypeSet,
    idx::Raycore.SetKey,
    p::Point3f,
    ray_d::Vec3f,
    dt::Float32
)
    return Raycore.with_index(_apply_deflection_helper, media, idx, p, ray_d, dt)
end

# Single medium fallback
@propagate_inbounds function apply_deflection_dispatch(
    medium::Medium,
    idx::Raycore.SetKey,
    p::Point3f,
    ray_d::Vec3f,
    dt::Float32
)
    return apply_deflection(medium, p, ray_d, dt)
end
