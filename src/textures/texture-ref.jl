# Texture reference utilities
# Uses Raycore.TextureRef for GPU-compatible texture references

# ============================================================================
# Unified Texture Evaluation - Same code path for CPU and GPU
# ============================================================================

"""
    eval_tex(ctx::StaticMultiTypeVec, tex_ref_or_val, uv::Point2f) -> T

Unified texture evaluation that works identically on CPU and GPU:
- `Raycore.TextureRef` - dereferences via context and samples
- raw value `T` - returns the value directly (constant)

Context is always StaticMultiTypeVec (from Adapt.adapt_structure on MultiTypeVec).
Materials should always use TextureRef or raw values, never Texture directly.
"""
# TextureRef path
@propagate_inbounds function eval_tex(ctx::Raycore.StaticMultiTypeVec, tref::Raycore.TextureRef, uv::Point2f)
    data = Raycore.deref(ctx, tref)
    return _sample_texture_data(data, uv)
end

# Raw value path (constant) - for Float32, Spectrum types
@propagate_inbounds function eval_tex(::Raycore.StaticMultiTypeVec, val::T, ::Point2f) where T<:Union{Float32, Spectrum}
    return val
end

# ============================================================================
# MultiTypeVec Integration
# ============================================================================

"""
    Raycore.maybe_convert_field(dhv::MultiTypeVec, tex::Texture)

Convert Hikari Texture to Raycore.TextureRef for MultiTypeVec storage.
Texture is only used for loading data - materials store TextureRef.
"""
function Raycore.maybe_convert_field(dhv::Raycore.MultiTypeVec, tex::Texture)
    return Raycore.store_texture(dhv, tex.data)
end
