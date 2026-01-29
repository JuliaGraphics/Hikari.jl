# Texture reference utilities
# Uses Raycore.TextureRef for GPU-compatible texture references

# ============================================================================
# Unified Texture Evaluation - Same code path for CPU and GPU
# ============================================================================

"""
    eval_tex(ctx::StaticMultiTypeSet, tex_ref_or_val, uv::Point2f) -> T

Unified texture evaluation that works identically on CPU and GPU:
- `Raycore.TextureRef` - dereferences via context and samples
- raw value `T` - returns the value directly (constant)

Context is always StaticMultiTypeSet (from Adapt.adapt_structure on MultiTypeSet).
Materials should always use TextureRef or raw values, never Texture directly.
"""
# TextureRef path
@propagate_inbounds function eval_tex(ctx::Raycore.StaticMultiTypeSet, tref::Raycore.TextureRef, uv::Point2f)
    data = Raycore.deref(ctx, tref)
    return _sample_texture_data(data, uv)
end

# Raw value path (constant) - for Float32, RGB, Spectrum types
@propagate_inbounds function eval_tex(::Raycore.StaticMultiTypeSet, val::T, ::Point2f) where T<:Union{Float32, RGB{Float32}, Spectrum}
    return val
end

# ============================================================================
# MultiTypeSet Integration
# ============================================================================

"""
    Raycore.maybe_convert_field(dhv::MultiTypeSet, tex::Texture)

Convert Hikari Texture to Raycore.TextureRef or raw value for MultiTypeSet storage.
- Const textures (scalars): return constval directly (no indirection needed)
- Non-const textures (arrays): convert to TextureRef
"""
function Raycore.maybe_convert_field(dhv::Raycore.MultiTypeSet, tex::Texture)
    # Const textures store the value in constval, data is uninitialized
    tex.isconst && return tex.constval
    return Raycore.store_texture(dhv, tex.data)
end

# Light skip methods are in lights/light-sampler.jl (after light types are defined)
# Distribution types are now converted via the standard texture ref infrastructure
