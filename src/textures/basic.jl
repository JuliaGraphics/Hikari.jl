const TextureType = Union{Float32,S} where S<:Spectrum

# Texture wraps actual texture data arrays. For constant values, use raw values directly
# in material fields (materials should have loose type parameters).
struct Texture{ElType, N, T<:AbstractArray{ElType, N}}
    data::T
    constval::ElType
    isconst::Bool
end
Texture(data::AbstractArray{T,N}) where {T,N} = Texture{T,N,typeof(data)}(data, zero(T), false)
function ConstTexture(val::T) where {T}
    arr = Array{T,0}(undef)
    Texture{T,0,typeof(arr)}(arr, val, true)
end

Base.zero(::Type{RGBSpectrum}) = RGBSpectrum(0.0f0, 0.0f0, 0.0f0, 1.0f0)

# Sample texture data array with bilinear interpolation. pbrt-v4 uses
# bilinear (and mipmap-trilinear) by default for imagemap textures. Without
# at least bilinear the BumpMap function sees `h(u+du) == h(u)` whenever
# du is sub-texel, which left Crown's gold dome with coherent mirror
# reflections because the bump gradient came out as exactly zero.
#
# UV convention matches the prior nearest-neighbour version (dim 1 is
# 1-v, dim 2 is u, repeat wrap via `frac`).
@propagate_inbounds function sample_texture_data(data::AbstractArray{T,N}, uv::Point2f)::T where {T,N}
    u_wrapped = uv[1] - floor(uv[1])
    v_wrapped = uv[2] - floor(uv[2])
    s = size(data)
    # Convert wrapped UV into fractional texel-space coordinates. The 0.5
    # offset puts (0,0) at the centre of texel (1,1) so the bilinear
    # weights cleanly span the four surrounding texels.
    fx = (1f0 - v_wrapped) * Float32(s[1]) - 0.5f0
    fy = u_wrapped * Float32(s[2]) - 0.5f0
    ix = floor(fx); iy = floor(fy)
    tx = fx - ix;   ty = fy - iy
    # Repeat wrap on the integer indices too (-1 → s, s → 1).
    row0 = mod(Int(ix),   s[1]) + 1
    row1 = mod(Int(ix)+1, s[1]) + 1
    col0 = mod(Int(iy),   s[2]) + 1
    col1 = mod(Int(iy)+1, s[2]) + 1
    @inbounds v00 = data[row0, col0]
    @inbounds v10 = data[row1, col0]
    @inbounds v01 = data[row0, col1]
    @inbounds v11 = data[row1, col1]
    w00 = (1f0 - tx) * (1f0 - ty)
    w10 = tx * (1f0 - ty)
    w01 = (1f0 - tx) * ty
    w11 = tx * ty
    return v00 * w00 + v10 * w10 + v01 * w01 + v11 * w11
end

# 0-dim arrays are scalar constants - just return the value, no UV sampling
@propagate_inbounds function sample_texture_data(data::AbstractArray{T,0}, ::Point2f)::T where T
    return data[]
end

# The single value of a 0-d texture, whichever field is actually holding it.
# The two constructors disagree: `Texture(arr)` puts it in `data` and zeroes
# `constval`, while `ConstTexture(val)` puts it in `constval` and leaves `data`
# an `Array{T,0}(undef)`. So `data[]` alone is uninitialised memory for every
# const texture, which is what `isconst` is here to tell us.
@inline constant_value(t::Texture{T, 0}) where {T} = t.isconst ? t.constval : t.data[]

function (c::Texture{T})(si::SurfaceInteraction)::T where {T<:TextureType}
    return sample_texture_data(c.data, si.uv)
end

# UV-only texture evaluation
@propagate_inbounds function evaluate_texture(tex::Texture{T}, uv::Point2f)::T where T
    tex.isconst && return tex.constval
    return sample_texture_data(tex.data, uv)
end

# Procedural checkerboard texture — exact port of pbrt-v4's CheckerboardTexture
# (textures.cpp `Checkerboard()`, 2D case) fused with `UVMapping::Map`.
# Evaluated analytically at shading time with pbrt's closed-form triangle-filter
# integral. This replaces the old 256² LUT rasterization in the pbrt scene
# builder: the LUT's bilinear smoothing both quantized checker-edge positions to
# the texel grid and widened the height-field gradient bands that bump mapping
# differentiates (shadow_bumpgold_dome_over_velvet pinned the resulting 21%
# energy loss).
#
# `tex1` is returned on even-parity cells, `tex2` on odd — same convention as
# pbrt (`(1 - w) * tex1 + w * tex2` with w = 0 at even parity).
struct CheckerboardTexture{T}
    su::Float32   # uscale
    sv::Float32   # vscale
    du::Float32   # udelta
    dv::Float32   # vdelta
    tex1::T
    tex2::T
end

to_texture(c::CheckerboardTexture) = c

"""
Number of entries in a field texture's colour ramp.

Inline in the struct rather than a second stored array, because a material
parameter is ONE `TexHandle` and a handle carries one `(slot, idx)`. The
coefficients get that slot — they scale with the number of cells and a real
grid has thousands — so the ramp, which does not, rides along inline.
"""
const FEM_RAMP_N = 32

"""
    FEMFieldTexture(coeffs, ramp; vmin, vmax)

The FEM solution as a texture: evaluates the element's own polynomial at the
hit's reference coordinate and looks the value up in a colour ramp.

A TEXTURE and not a shading path. Writing a colour into the hit would bypass
the BSDF, the lights, the shadows and the path continuation — an albedo viewer
rather than a renderer. Bound to any material's parameter, the field composes
with every BSDF Hikari has, and material dispatch stays `MultiTypeSet`
untouched.

`VertexColorTexture` is the precedent and the shape is the same: read
`face_idx` off the hit to pick the data, read the hit's surface coordinate to
interpolate. What differs is only that a per-face colour is interpolated
linearly and a field is evaluated — `uv` carries ξ, which the intersection put
there.
"""
struct FEMFieldTexture{C}
    coeffs::C                                   # FEM_NMONO x ncells → TextureRef
    ramp::NTuple{FEM_RAMP_N, RGBSpectrum}
    vmin::Float32
    vmax::Float32
end

# A NAME is not a ramp, and saying so beats the `MethodError` this produced.
# Hikari depends on `Colors` but on no colormap package, so there is nothing here
# that could turn `:viridis` into colours.
FEMFieldTexture(coeffs, ramp::Symbol; kw...) = error(
    "`colormap = :$ramp` is a NAME, and Hikari cannot resolve one: it has no " *
    "colormap package, only `Colors`. Pass a vector of colours instead — " *
    "`FEMFieldTexture` resamples whatever length it is to $FEM_RAMP_N. " *
    "`Hikari.FEM_DEFAULT_RAMP` is one, and with Makie loaded so is " *
    "`Makie.to_colormap(:$ramp)`.")

"""Build one from a coefficient block and any Makie-style colour ramp."""
function FEMFieldTexture(coeffs, ramp::AbstractVector; vmin = 0, vmax = 1)
    n = length(ramp)
    resampled = ntuple(FEM_RAMP_N) do i
        c = ramp[clamp(round(Int, (i - 1) / (FEM_RAMP_N - 1) * (n - 1)) + 1, 1, n)]
        c isa RGBSpectrum ? c : RGBSpectrum(Float32(red(c)), Float32(green(c)), Float32(blue(c)))
    end
    return FEMFieldTexture(coeffs, resampled, Float32(vmin), Float32(vmax))
end


# Anything that is a spatially-varying texture (as opposed to a raw constant
# scalar/spectrum value). Use this for "is it a texture?" checks; matching only
# `Texture` silently misses procedural textures.
# `FEMFieldTexture` is declared later (fem.jl) but belongs in this union, so it
# is named by a forward reference rather than by widening the union at its
# definition site — a check for "is it a texture?" that misses one silently
# treats it as a constant.
const AnyTexture = Union{Texture, CheckerboardTexture, FEMFieldTexture}

# Per-face vertex color texture: stores 3 colors per face for barycentric interpolation
struct VertexColorTexture{T}
    face_colors::T   # (3, n_faces) matrix of RGBSpectrum, becomes TextureRef via MultiTypeSet
    n_faces::Int32
end

# ============================================================================
# Auto-wrapping helpers: convert raw values to Texture
# ============================================================================

to_texture(t::Texture) = t
to_texture(v::RGBSpectrum) = ConstTexture(v)
to_texture(v::Float32) = ConstTexture(v)
to_texture(v::Real) = to_texture(Float32(v))
# For color tuples/vectors (use Tuple{Real,Real,Real} to handle mixed Int/Float)
to_texture(v::Tuple{Real,Real,Real}) = to_texture(RGBSpectrum(Float32(v[1]), Float32(v[2]), Float32(v[3])))
# Support Colors.jl RGB types (RGB, RGBA, etc.)
to_texture(c::Colorant) = to_texture(RGBSpectrum(Float32(red(c)), Float32(green(c)), Float32(blue(c))))
to_texture(c::AbstractMatrix{<: RGB}) = Texture(map(c-> RGBSpectrum(Float32(red(c)), Float32(green(c)), Float32(blue(c))), c))
# Identity passthrough for PiecewiseLinearSpectrum (not a texture, stored directly)
to_texture(s::PiecewiseLinearSpectrum) = s

