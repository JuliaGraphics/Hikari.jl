# Texture reference utilities
# Uses Raycore.TextureRef for GPU-compatible texture references

# ============================================================================
# Texture Filter Context (pbrt-v4 style)
# ============================================================================

"""
    TextureFilterContext

Context for texture filtering, containing UV coordinates and screen-space derivatives.
Following pbrt-v4's TextureEvalContext pattern.

Materials receive this context and pass it to eval_tex for proper texture filtering.
The UV derivatives (dudx, dudy, dvdx, dvdy) are computed from:
1. Surface partial derivatives (dpdu, dpdv) at intersection
2. Screen-space position derivatives (dpdx, dpdy) from camera

These enable mipmap level selection and anisotropic filtering.
"""
struct TextureFilterContext
    uv::Point2f
    dudx::Float32
    dudy::Float32
    dvdx::Float32
    dvdy::Float32
    face_idx::UInt32              # Triangle face index (for vertex colors)
    bary::SVector{3, Float32}     # Barycentric coordinates (for vertex colors)
end

# Default context with no filtering (derivatives = 0)
TextureFilterContext(uv::Point2f) = TextureFilterContext(uv, 0f0, 0f0, 0f0, 0f0, UInt32(0), SVector{3,Float32}(0f0, 0f0, 0f0))
TextureFilterContext(uv::Point2f, dudx, dudy, dvdx, dvdy) = TextureFilterContext(uv, dudx, dudy, dvdx, dvdy, UInt32(0), SVector{3,Float32}(0f0, 0f0, 0f0))

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
# Context-Aware Texture Evaluation (pbrt-v4 MaterialEvalContext style)
# ============================================================================

"""
    eval_tex(ctx::StaticMultiTypeSet, tex_ref_or_val, tfc::TextureFilterContext) -> T

Texture evaluation using TextureFilterContext for proper filtering.
This is the preferred interface for materials - uses bilinear filtering
and UV derivatives for future mipmap support.
"""
# TextureRef path with context
@propagate_inbounds function eval_tex(ctx::Raycore.StaticMultiTypeSet, tref::Raycore.TextureRef, tfc::TextureFilterContext)
    data = Raycore.deref(ctx, tref)
    return _sample_texture_data_filtered(data, tfc.uv, tfc.dudx, tfc.dudy, tfc.dvdx, tfc.dvdy)
end

# Raw value path (constant) - context ignored
@propagate_inbounds function eval_tex(::Raycore.StaticMultiTypeSet, val::T, ::TextureFilterContext) where T<:Union{Float32, RGB{Float32}, Spectrum}
    return val
end

# PiecewiseLinearSpectrum passthrough (not a texture, stored directly in material)
@propagate_inbounds eval_tex(::Raycore.StaticMultiTypeSet, val::PiecewiseLinearSpectrum, ::Point2f) = val
@propagate_inbounds eval_tex(::Raycore.StaticMultiTypeSet, val::PiecewiseLinearSpectrum, ::TextureFilterContext) = val

# ============================================================================
# Filtered Texture Evaluation (with UV derivatives for mipmap selection)
# ============================================================================

"""
    eval_tex_filtered(ctx::StaticMultiTypeSet, tex_ref_or_val, uv::Point2f,
                      dudx, dudy, dvdx, dvdy) -> T

Filtered texture evaluation using UV derivatives for mipmap selection.
Following pbrt-v4's texture filtering approach.

Currently falls back to unfiltered sampling (TODO: implement mipmaps).
The derivatives are passed through for future mipmap implementation.
"""
# TextureRef path with filtering
@propagate_inbounds function eval_tex_filtered(
    ctx::Raycore.StaticMultiTypeSet, tref::Raycore.TextureRef, uv::Point2f,
    dudx::Float32, dudy::Float32, dvdx::Float32, dvdy::Float32
)
    data = Raycore.deref(ctx, tref)
    return _sample_texture_data_filtered(data, uv, dudx, dudy, dvdx, dvdy)
end

# Raw value path (constant) - derivatives ignored
@propagate_inbounds function eval_tex_filtered(
    ::Raycore.StaticMultiTypeSet, val::T, ::Point2f,
    ::Float32, ::Float32, ::Float32, ::Float32
) where T<:Union{Float32, RGB{Float32}, Spectrum}
    return val
end

"""
    _sample_texture_data_filtered(data, uv, dudx, dudy, dvdx, dvdy) -> T

Sample texture with filtering based on UV derivatives.
Uses the derivatives to compute the filter footprint for mipmap selection.

TODO: Implement proper mipmap-based filtering. Currently uses bilinear sampling
as a simple improvement over point sampling.
"""
@propagate_inbounds function _sample_texture_data_filtered(
    data::AbstractArray{T,N}, uv::Point2f,
    dudx::Float32, dudy::Float32, dvdx::Float32, dvdy::Float32
)::T where {T,N}
    # For now, use bilinear sampling as a simple filter
    # TODO: Implement mipmap selection based on derivatives:
    #   width = max(sqrt(dudx^2 + dvdx^2), sqrt(dudy^2 + dvdy^2)) * tex_size
    #   level = log2(max(1, width))
    return _sample_texture_bilinear(data, uv)
end

# 0-dim arrays (scalar constants) - just return the value
@propagate_inbounds function _sample_texture_data_filtered(
    data::AbstractArray{T,0}, ::Point2f,
    ::Float32, ::Float32, ::Float32, ::Float32
)::T where T
    return data[]
end

"""
    _sample_texture_bilinear(data, uv) -> T

Bilinear texture sampling for 2D textures.
Provides smoother results than point sampling.
"""
@propagate_inbounds function _sample_texture_bilinear(data::AbstractArray{T,2}, uv::Point2f)::T where T
    # Apply UV flip (standard texture coordinate convention)
    uv_adj = Vec2f(1f0 - uv[2], uv[1])

    h, w = size(data)
    # Convert to pixel coordinates (0-indexed float)
    px = uv_adj[2] * (w - 1) + 1f0
    py = uv_adj[1] * (h - 1) + 1f0

    # Get integer coordinates
    x0 = unsafe_trunc(Int32, floor(px))
    y0 = unsafe_trunc(Int32, floor(py))
    x1 = x0 + Int32(1)
    y1 = y0 + Int32(1)

    # Clamp to valid range
    x0 = clamp(x0, Int32(1), Int32(w))
    x1 = clamp(x1, Int32(1), Int32(w))
    y0 = clamp(y0, Int32(1), Int32(h))
    y1 = clamp(y1, Int32(1), Int32(h))

    # Compute interpolation weights
    fx = px - floor(px)
    fy = py - floor(py)

    # Sample four corners
    c00 = data[y0, x0]
    c10 = data[y0, x1]
    c01 = data[y1, x0]
    c11 = data[y1, x1]

    # Bilinear interpolation
    c0 = c00 * (1f0 - fx) + c10 * fx
    c1 = c01 * (1f0 - fx) + c11 * fx
    return c0 * (1f0 - fy) + c1 * fy
end

# Fallback for non-2D textures
@propagate_inbounds function _sample_texture_bilinear(data::AbstractArray{T,N}, uv::Point2f)::T where {T,N}
    return _sample_texture_data(data, uv)
end

# ============================================================================
# MultiTypeSet Integration
# ============================================================================


# TODO remove the below function, Raycores conversion should work correctly without it
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

# ============================================================================
# VertexColorTexture MultiTypeSet conversion
# ============================================================================

function Raycore.maybe_convert_field(dhv::Raycore.MultiTypeSet, vtex::VertexColorTexture)
    vtex.face_colors isa AbstractArray || return vtex
    ref = Raycore.store_texture(dhv, vtex.face_colors)
    return VertexColorTexture(ref, vtex.n_faces)
end

# ============================================================================
# SurfaceInteraction-based Texture Evaluation
# ============================================================================

# Regular textures: extract UV from SI
@propagate_inbounds eval_tex(ctx::Raycore.StaticMultiTypeSet, tref::Raycore.TextureRef, si::SurfaceInteraction) = eval_tex(ctx, tref, si.uv)
@propagate_inbounds eval_tex(::Raycore.StaticMultiTypeSet, val::T, ::SurfaceInteraction) where T<:Union{Float32, RGB{Float32}, Spectrum} = val
@propagate_inbounds eval_tex(::Raycore.StaticMultiTypeSet, val::PiecewiseLinearSpectrum, ::SurfaceInteraction) = val

# VertexColorTexture: barycentric interpolation using face_idx and bary from SI
@propagate_inbounds function eval_tex(ctx::Raycore.StaticMultiTypeSet, vtex::VertexColorTexture, si::SurfaceInteraction)
    data = Raycore.deref(ctx, vtex.face_colors)
    fi = si.face_idx
    b = si.bary
    return data[1, fi] * b[1] + data[2, fi] * b[2] + data[3, fi] * b[3]
end

# VertexColorTexture: barycentric interpolation using face_idx and bary from TextureFilterContext
@propagate_inbounds function eval_tex(ctx::Raycore.StaticMultiTypeSet, vtex::VertexColorTexture, tfc::TextureFilterContext)
    data = Raycore.deref(ctx, vtex.face_colors)
    fi = tfc.face_idx
    b = tfc.bary
    return data[1, fi] * b[1] + data[2, fi] * b[2] + data[3, fi] * b[3]
end

# VertexColorTexture fallback for UV-only paths (fast-wavefront) - gray placeholder
@propagate_inbounds eval_tex(::Raycore.StaticMultiTypeSet, ::VertexColorTexture, ::Point2f) = RGBSpectrum(0.5f0)

# Light skip methods are in lights/light-sampler.jl (after light types are defined)
# Distribution types are now converted via the standard texture ref infrastructure
