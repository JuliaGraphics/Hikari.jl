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

# A hit's point-sampled context: no footprint, but the face and barycentrics a
# vertex-colour texture indexes with. The contexts above carry face 0, which a
# `VertexColorTexture` reads as `data[k, 0]` — in front of its buffer. That is
# what the intersection-time alpha test and `MixMaterial`'s amount did: the
# alpha of whatever lay there decided whether the ray passed through, and a quad
# in vertex colours rendered as the background on Metal (`test_colour_encoding`).
TextureFilterContext(uv::Point2f, face_idx::UInt32, bary::SVector{3, Float32}) =
    TextureFilterContext(uv, 0f0, 0f0, 0f0, 0f0, face_idx, bary)

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
    return sample_texture_data(data, uv)
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
    return sample_texture_data_filtered(data, tfc.uv, tfc.dudx, tfc.dudy, tfc.dvdx, tfc.dvdy)
end

# Raw value path (constant) - context ignored
@propagate_inbounds function eval_tex(::Raycore.StaticMultiTypeSet, val::T, ::TextureFilterContext) where T<:Union{Float32, RGB{Float32}, Spectrum}
    return val
end

# PiecewiseLinearSpectrum passthrough (not a texture, stored directly in material)
@propagate_inbounds eval_tex(::Raycore.StaticMultiTypeSet, val::PiecewiseLinearSpectrum, ::Point2f) = val
@propagate_inbounds eval_tex(::Raycore.StaticMultiTypeSet, val::PiecewiseLinearSpectrum, ::TextureFilterContext) = val

# ============================================================================
# CheckerboardTexture evaluation — pbrt-v4 textures.cpp Checkerboard(), 2D case
# ============================================================================

# pbrt-v4 textures.cpp:186 `d`: running integral of the ±1 checkerboard square
# wave, used to filter the checkerboard analytically over the pixel footprint.
@propagate_inbounds function checker_d(x::Float32)
    y = x / 2f0 - floor(x / 2f0) - 0.5f0
    return x / 2f0 + y * (1f0 - 2f0 * abs(y))
end

# pbrt-v4 textures.cpp:191 `bf`: triangle-filtered 1D checkerboard of radius r
# centred at x. When the filter support stays inside one cell this is the exact
# ±1 point sample (which also covers r == 0, i.e. no derivatives available).
#
# The result is CLAMPED to [-1, 1]. That is not a change to the filter, it is
# the filter's own invariant: `bf` is a triangle-filtered ±1 square wave, so it
# cannot leave [-1, 1] mathematically. The expression below can, because it is a
# second difference divided by `r * r` — near a cell boundary with a small
# radius the numerator is all cancellation and the division amplifies the noise.
# Measured on Float32: at `x = 1.0f0, r = 1f-6` it returns **59604.6**.
#
# pbrt has the identical expression and the identical blow-up, and gets away
# with it because C++ carries the damage as a NaN: the caller's
# `(1-w)*tex1 + w*tex2` extrapolates to a large negative roughness and
# `std::sqrt` of that is NaN, so one pixel is wrong. Julia's `sqrt` THROWS on a
# negative real, so the same value takes the whole dispatch down —
# `tex_conductor_checker_rough_light_point` and its dielectric twin died with
# `DomainError: This operation requires a complex input to return a complex
# result` on both the software and hardware paths.
#
# Clamping recovers the right answer rather than merely bounding the wrong one:
# where the division is ill-conditioned the filter support is far smaller than a
# cell, which is the point-sample case, and a point sample IS ±1.
@propagate_inbounds function checker_bf(x::Float32, r::Float32)
    if floor(x - r) == floor(x + r)
        return 1f0 - 2f0 * Float32(unsafe_trunc(Int32, floor(x)) & Int32(1))
    end
    v = (checker_d(x + r) - 2f0 * checker_d(x) + checker_d(x - r)) / (r * r)
    return clamp(v, -1f0, 1f0)
end

# Checkerboard() 2D branch + UVMapping::Map fused, then the
# (1 - w) * tex1 + w * tex2 mix from {Float,Spectrum}CheckerboardTexture::Evaluate.
@propagate_inbounds function eval_tex(ctx::Raycore.StaticMultiTypeSet, cb::CheckerboardTexture,
                                      tfc::TextureFilterContext)
    s = cb.su * tfc.uv[1] + cb.du
    t = cb.sv * tfc.uv[2] + cb.dv
    ds = 1.5f0 * max(abs(cb.su * tfc.dudx), abs(cb.su * tfc.dudy))
    dt = 1.5f0 * max(abs(cb.sv * tfc.dvdx), abs(cb.sv * tfc.dvdy))
    w = 0.5f0 - 0.5f0 * checker_bf(s, ds) * checker_bf(t, dt)
    t1 = eval_tex(ctx, cb.tex1, tfc)
    t2 = eval_tex(ctx, cb.tex2, tfc)
    return t1 * (1f0 - w) + t2 * w
end

# Derivative-free entry points point-sample (r = 0 ⇒ checker_bf hits the exact
# ±1 branch), matching pbrt with zero differentials.
@propagate_inbounds eval_tex(ctx::Raycore.StaticMultiTypeSet, cb::CheckerboardTexture, uv::Point2f) =
    eval_tex(ctx, cb, TextureFilterContext(uv))
@propagate_inbounds eval_tex(ctx::Raycore.StaticMultiTypeSet, cb::CheckerboardTexture, si::SurfaceInteraction) =
    eval_tex(ctx, cb, TextureFilterContext(si.uv))

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
    return sample_texture_data_filtered(data, uv, dudx, dudy, dvdx, dvdy)
end

# Raw value path (constant) - derivatives ignored
@propagate_inbounds function eval_tex_filtered(
    ::Raycore.StaticMultiTypeSet, val::T, ::Point2f,
    ::Float32, ::Float32, ::Float32, ::Float32
) where T<:Union{Float32, RGB{Float32}, Spectrum}
    return val
end

"""
    sample_texture_data_filtered(data, uv, dudx, dudy, dvdx, dvdy) -> T

Sample texture with filtering based on UV derivatives.
TODO: Implement proper mipmap-based filtering with EWA or box filter.
Currently uses nearest-neighbor (point) sampling, which matches pbrt-v4's
behavior for pre-rasterized textures and avoids spurious blur.
"""
@propagate_inbounds function sample_texture_data_filtered(
    data::AbstractArray{T,N}, uv::Point2f,
    dudx::Float32, dudy::Float32, dvdx::Float32, dvdy::Float32
)::T where {T,N}
    return sample_texture_data(data, uv)
end

# 0-dim arrays (scalar constants) - just return the value
@propagate_inbounds function sample_texture_data_filtered(
    data::AbstractArray{T,0}, ::Point2f,
    ::Float32, ::Float32, ::Float32, ::Float32
)::T where T
    return data[]
end

"""
    sample_texture_bilinear(data, uv) -> T

Bilinear texture sampling for 2D textures.
Provides smoother results than point sampling.
"""
@propagate_inbounds function sample_texture_bilinear(data::AbstractArray{T,2}, uv::Point2f)::T where T
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
@propagate_inbounds function sample_texture_bilinear(data::AbstractArray{T,N}, uv::Point2f)::T where {T,N}
    return sample_texture_data(data, uv)
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
# update_item overloads — let `Raycore.update!` reuse existing texture slots
# instead of pushing a new one on every update.  Without these the raw
# `Texture` / `VertexColorTexture` values coming through `update_material!`
# would hit the generic struct fallback, mis-match on field names vs the
# stored (converted) form, and leak ~hundreds of MB of GPU memory per
# rebuild in scenes with per-frame mesh updates.
# ============================================================================

# Const Texture over a scalar field: unwrap to the raw value.
function Raycore.update_item(::Raycore.MultiTypeSet, old, new::Texture)
    new.isconst && return new.constval
    error("update_item: non-const Texture cannot overwrite a scalar field of type $(typeof(old)); rebuild the scene")
end

# Non-const Texture against a stored TextureRef: copy into the existing GPU
# buffer (reallocating on size mismatch).
function Raycore.update_item(dhv::Raycore.MultiTypeSet, old::Raycore.TextureRef,
                              new::Texture)
    new.isconst && error("update_item: replacing a sampled TextureRef with a const Texture is not supported; rebuild the scene")
    Raycore.copyto_texture!(dhv, old, new.data)
    return old
end

# VertexColorTexture update: reuse the existing ref, copy face_colors in.
# `old.face_colors` is always a `TextureRef` once the item has been pushed,
# `new.face_colors` is the raw CPU array the user supplied.
function Raycore.update_item(dhv::Raycore.MultiTypeSet,
                              old::VertexColorTexture{<:Raycore.TextureRef},
                              new::VertexColorTexture{<:AbstractArray})
    Raycore.copyto_texture!(dhv, old.face_colors, new.face_colors)
    return old
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
