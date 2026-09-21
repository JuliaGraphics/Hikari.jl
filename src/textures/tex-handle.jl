# ============================================================================
# TexHandle — one non-parametric handle for every material parameter
# ============================================================================
#
# This removes the material type explosion.
#
# A material parameter used to be stored as whatever it happened to be: a raw
# `Float32` for a constant, a `Raycore.TextureRef` for an image map, a
# `CheckerboardTexture` for a procedural. Those are different TYPES, so
# `Conductor{…,Float32,…}` and `Conductor{…,TextureRef{…},…}` are different
# materials — and the per-material chit path compiles one closest-hit shader per
# concrete material type. Crown's 50 named materials collapse to 12 Julia types
# of which only ~5 are distinct material CLASSES; the rest is
# constant-vs-texture combinatorics and wrapper types.
#
# pbrt-v4 does not have this problem: every parameter is a `FloatTexture` /
# `SpectrumTexture` `TaggedPointer`, and a constant is a `FloatConstantTexture`
# behind the same handle. `TexHandle` is that in a form SPIR-V accepts — a
# tagged union, isbits, no pointers, no type parameters.
#
# Constants live INLINE. Out-of-line kinds carry the runtime `(slot, idx)` pair
# that `Raycore.with_texture` dispatches on; `TextureRef` cannot be used because
# it encodes its slot in its type, which is exactly what is being erased.
#
# ── Why this is affordable ──────────────────────────────────────────────────
#
# The tag switch is paid ONCE PER HIT inside `get_bxdf`, never in a BSDF inner
# loop. That ordering is what makes it work: without the `get_bxdf` split (done
# first, deliberately) the switch would be inlined at each of the ~87 texture
# reads and would likely cost more than the chits it saves.

"""Texture kinds a `TexHandle` can name. Stored as a `UInt8` tag."""
module TexKind
const NONE           = UInt8(0)   # unset optional parameter (e.g. no bump)
const CONST_FLOAT    = UInt8(1)   # payload in `f`
const CONST_SPECTRUM = UInt8(2)   # payload in `rgb`
const IMAGE          = UInt8(3)   # (slot, idx) → image array in the texture store
const CHECKER        = UInt8(4)   # (slot, idx) → 1-element CheckerboardTexture array
const VERTEX_COLOR   = UInt8(5)   # (slot, idx) → (3, n_faces) per-face vertex colours
end

"""
    TexHandle

A material parameter: an inline constant, a stored texture, or nothing. Isbits
and NON-PARAMETRIC, so a material holding these has one concrete type however
its parameters were specified.
"""
struct TexHandle
    kind::UInt8
    f::Float32          # CONST_FLOAT payload
    rgb::RGBSpectrum    # CONST_SPECTRUM payload
    slot::Int32         # 1-based texture type slot
    idx::Int32          # element index within that slot
end

const _TH_ZERO = RGBSpectrum(0f0, 0f0, 0f0)

# ── Constructors ────────────────────────────────────────────────────────────

@inline TexHandle() = TexHandle(TexKind.NONE, 0f0, _TH_ZERO, Int32(0), Int32(0))
@inline TexHandle(v::Float32) = TexHandle(TexKind.CONST_FLOAT, v, _TH_ZERO, Int32(0), Int32(0))
@inline TexHandle(v::Real) = TexHandle(Float32(v))
@inline TexHandle(v::RGBSpectrum) = TexHandle(TexKind.CONST_SPECTRUM, 0f0, v, Int32(0), Int32(0))
@inline TexHandle(v::RGB{Float32}) = TexHandle(RGBSpectrum(v.r, v.g, v.b))
@inline TexHandle(c::Colorant) = TexHandle(RGBSpectrum(Float32(red(c)), Float32(green(c)), Float32(blue(c))))
@inline TexHandle(v::Tuple{Real,Real,Real}) =
    TexHandle(RGBSpectrum(Float32(v[1]), Float32(v[2]), Float32(v[3])))
@inline TexHandle(h::TexHandle) = h
@inline TexHandle(::Nothing) = TexHandle()

# `to_texture` wraps a constant in a 0-dimensional `Texture` (`ConstTexture`),
# which is how most pbrt scalar/colour parameters arrive. Unwrap it inline —
# a constant has no business occupying a texture slot.
@inline TexHandle(t::Texture{T, 0}) where {T} = TexHandle(constant_value(t))

# The slot lives in the TextureRef's TYPE; capture it as data.
@inline TexHandle(t::Raycore.TextureRef{A, T, N, TIdx}) where {A, T, N, TIdx} =
    TexHandle(TexKind.IMAGE, 0f0, _TH_ZERO, Int32(TIdx), Int32(t.idx))

# Lets Raycore's push-time struct rebuild (`convert_to_texturerefs` calls
# `BaseT(new_fields...)`) drop a freshly-stored `TextureRef` into a
# `TexHandle`-typed field: Julia's default constructor `convert`s each argument
# to its field type.
Base.convert(::Type{TexHandle}, t::Raycore.TextureRef) = TexHandle(t)
Base.convert(::Type{TexHandle}, t::Texture{T, 0}) where {T} = TexHandle(t)
Base.convert(::Type{TexHandle}, v::Union{Real, RGBSpectrum, Colorant, Tuple{Real,Real,Real}}) = TexHandle(v)

@inline is_none(h::TexHandle) = h.kind == TexKind.NONE

# Host-side queries used by the pbrt builder to decide whether a parameter can
# still be folded into a scalar (pbrt's `uroughness` defaults to `roughness`,
# which is only meaningful when `roughness` is a constant).
#
# Total over the host forms, not just handles: when no scene exists to store
# textures in, a pbrt parameter is still whatever `build_pbrt_textures` built —
# a 0-d `ConstTexture` for a constant, an image `Texture` otherwise.
@inline is_const_float(h::TexHandle) = h.kind == TexKind.CONST_FLOAT
@inline is_const_float(::Texture{T, 0}) where {T <: Real} = true
@inline is_const_float(x) = false

"""
    const_float(h, default) -> Float32

The inline scalar of a constant handle, or `default` for out-of-line kinds.
"""
@inline const_float(h::TexHandle, default::Real = 0f0) =
    h.kind == TexKind.CONST_FLOAT ? h.f :
    h.kind == TexKind.CONST_SPECTRUM ? h.rgb.c[1] : Float32(default)
@inline const_float(t::Texture{T, 0}, default::Real = 0f0) where {T <: Real} =
    Float32(constant_value(t))
@inline const_float(x, default::Real = 0f0) = Float32(default)

"""
    const_spectrum(h) -> RGBSpectrum

The inline colour of a constant handle. Errors for out-of-line kinds: callers
are host-side paths (area-light registration) that have no texture store.
"""
@inline function const_spectrum(h::TexHandle)
    h.kind == TexKind.CONST_SPECTRUM && return h.rgb
    h.kind == TexKind.CONST_FLOAT && return RGBSpectrum(h.f, h.f, h.f)
    h.kind == TexKind.NONE && return _TH_ZERO
    error("TexHandle: kind $(h.kind) is stored out-of-line and needs the " *
          "scene's texture store to evaluate; const_spectrum only handles " *
          "inline constants.")
end

# ── Evaluation ──────────────────────────────────────────────────────────────
# Call from `get_bxdf` only. Each returns ONE type, so the runtime slot dispatch
# inside `with_texture` stays type-stable.

@inline _th_scalar(v::Real) = Float32(v)
@inline _th_scalar(v::RGBSpectrum) = v.c[1]
@inline _th_scalar(v::RGB{Float32}) = v.r
@inline _th_scalar(v) = 0f0

@inline _th_spectrum(v::RGBSpectrum) = v
@inline _th_spectrum(v::RGB{Float32}) = RGBSpectrum(v.r, v.g, v.b)
@inline _th_spectrum(v::Real) = (x = Float32(v); RGBSpectrum(x, x, x))
@inline _th_spectrum(v) = _TH_ZERO

# `with_texture` emits ONE ARM PER SLOT of the texture store and every arm is
# compiled, including the ones a given kind never selects at runtime — a
# CHECKER handle's slot switch still generates the image arm for the slot
# holding vertex colours. So each of these has to be TOTAL over every array
# type that can live in the store; the fallbacks are what the unreachable arms
# compile to. Without them the unreachable arm is a MethodError, which on the
# GPU is an unsupported call, not a nice error.
const _TH_TEXEL = Union{Float32, RGB{Float32}, Spectrum}

@propagate_inbounds _th_img_float(arr::AbstractArray{T, 2}, uv::Point2f) where {T <: _TH_TEXEL} =
    _th_scalar(sample_texture_data(arr, uv))
@propagate_inbounds _th_img_float(arr::AbstractArray{T, 0}, uv::Point2f) where {T <: _TH_TEXEL} =
    _th_scalar(@inbounds arr[])
@propagate_inbounds _th_img_float(arr, uv::Point2f) = 0f0

@propagate_inbounds _th_img_spec(arr::AbstractArray{T, 2}, uv::Point2f) where {T <: _TH_TEXEL} =
    _th_spectrum(sample_texture_data(arr, uv))
@propagate_inbounds _th_img_spec(arr::AbstractArray{T, 0}, uv::Point2f) where {T <: _TH_TEXEL} =
    _th_spectrum(@inbounds arr[])
@propagate_inbounds _th_img_spec(arr, uv::Point2f) = _TH_ZERO

# Procedurals live as a one-element array in the same store; re-read element 1
# and evaluate. `eval_tex` on a CheckerboardTexture takes the filter context so
# it can match pbrt's filtered checkerboard, so pass the whole tfc through.
@propagate_inbounds _th_proc_float(arr::AbstractArray{<:CheckerboardTexture}, textures, tfc) =
    _th_scalar(eval_tex(textures, (@inbounds arr[1]), tfc))
@propagate_inbounds _th_proc_float(arr, textures, tfc) = 0f0
@propagate_inbounds _th_proc_spec(arr::AbstractArray{<:CheckerboardTexture}, textures, tfc) =
    _th_spectrum(eval_tex(textures, (@inbounds arr[1]), tfc))
@propagate_inbounds _th_proc_spec(arr, textures, tfc) = _TH_ZERO

# Per-face vertex colours: a (3, n_faces) matrix interpolated by the hit's
# barycentrics. Same shape as the old `VertexColorTexture`, minus the wrapper
# struct that used to be a material type parameter.
@propagate_inbounds function _th_vcol_spec(data::AbstractArray{T, 2}, tfc) where {T <: _TH_TEXEL}
    fi = tfc.face_idx
    b = tfc.bary
    return _th_spectrum(@inbounds data[1, fi] * b[1] + data[2, fi] * b[2] + data[3, fi] * b[3])
end
@propagate_inbounds _th_vcol_spec(data, tfc) = _TH_ZERO
@propagate_inbounds _th_vcol_float(data, tfc) = _th_vcol_spec(data, tfc).c[1]

"""
    eval_handle(textures, h::TexHandle, tfc) -> Float32

Resolve a scalar parameter.
"""
@propagate_inbounds function eval_handle(textures, h::TexHandle, tfc::TextureFilterContext)
    k = h.kind
    if k == TexKind.CONST_FLOAT
        return h.f
    elseif k == TexKind.IMAGE
        return Raycore.with_texture(_th_img_float, textures, h.slot, h.idx, tfc.uv)
    elseif k == TexKind.CHECKER
        return Raycore.with_texture(_th_proc_float, textures, h.slot, h.idx, textures, tfc)
    elseif k == TexKind.VERTEX_COLOR
        return Raycore.with_texture(_th_vcol_float, textures, h.slot, h.idx, tfc)
    elseif k == TexKind.CONST_SPECTRUM
        return h.rgb.c[1]
    end
    return 0f0
end

"""
    eval_handle_spectrum(textures, h::TexHandle, tfc) -> RGBSpectrum

Resolve a colour parameter to RGB. Uplifting to spectral stays with the caller,
which holds the wavelength sample.
"""
@propagate_inbounds function eval_handle_spectrum(textures, h::TexHandle, tfc::TextureFilterContext)
    k = h.kind
    if k == TexKind.CONST_SPECTRUM
        return h.rgb
    elseif k == TexKind.IMAGE
        return Raycore.with_texture(_th_img_spec, textures, h.slot, h.idx, tfc.uv)
    elseif k == TexKind.CHECKER
        return Raycore.with_texture(_th_proc_spec, textures, h.slot, h.idx, textures, tfc)
    elseif k == TexKind.VERTEX_COLOR
        return Raycore.with_texture(_th_vcol_spec, textures, h.slot, h.idx, tfc)
    elseif k == TexKind.CONST_FLOAT
        return RGBSpectrum(h.f, h.f, h.f)
    end
    return _TH_ZERO
end


# ============================================================================
# Host material parameters → device handles
# ============================================================================
#
# A material can be built long before any scene exists — RayMakie constructs
# `Diffuse(color_texture, …)` inside a plot's argument-conversion node, and the
# Hikari scene is only created when the screen renders. So a texture parameter
# has to survive on the host as the texture itself, and become a `TexHandle`
# only when the material is pushed, which is the first moment a texture store
# exists to put it in.
#
# That is why the texture-carrying fields of every material are TYPE
# PARAMETERS: the host form is whatever the caller passed, and
# `to_device_material` rewrites every one of them to a `TexHandle`, so the form
# that reaches the GPU is the same concrete type for all spellings. A
# constant-coloured `Diffuse` and an image-mapped one are one closest-hit
# shader, not two.

"""
    matparam(x)

Normalize a material parameter at CONSTRUCTION time. Constants collapse
straight to an inline `TexHandle`; anything needing a texture store stays as-is
until [`to_device_material`](@ref) runs at push time.
"""
@inline matparam(x::Texture) = x
@inline matparam(x::CheckerboardTexture) = x
@inline matparam(x::VertexColorTexture) = x
@inline matparam(x::PiecewiseLinearSpectrum) = x
@inline matparam(x) = TexHandle(x)

"""
    device_param(dhv, x) -> TexHandle

Resolve one host material parameter against the set's texture store.
"""
device_param(dhv, h::TexHandle) = h
device_param(dhv, s::PiecewiseLinearSpectrum) = s   # lives inline on the device
device_param(dhv, t::Texture{T, 0}) where {T} = TexHandle(t)
device_param(dhv, t::Texture{T, N}) where {T, N} =
    TexHandle(Raycore.store_texture(dhv, t.data))
device_param(dhv, r::Raycore.TextureRef) = TexHandle(r)
device_param(dhv, v::Union{Real, RGBSpectrum, Colorant, Tuple{Real,Real,Real}}) = TexHandle(v)

# Procedurals and per-face colours go into the same store as a one-element /
# (3, n_faces) array, so a single `(slot, idx)` pair reaches all of them.
function device_param(dhv, c::CheckerboardTexture)
    ref = Raycore.store_texture(dhv, [c])
    return TexHandle(TexKind.CHECKER, 0f0, _TH_ZERO,
                     Int32(texref_slot(ref)), Int32(ref.idx))
end
function device_param(dhv, v::VertexColorTexture)
    ref = Raycore.store_texture(dhv, v.face_colors)
    return TexHandle(TexKind.VERTEX_COLOR, 0f0, _TH_ZERO,
                     Int32(texref_slot(ref)), Int32(ref.idx))
end
# Sub-materials (MediumInterface, MixMaterial) recurse; anything else keeps
# Raycore's own conversion.
device_param(dhv, m::Material) = to_device_material(dhv, m)
device_param(dhv, x) = Raycore.maybe_convert_field(dhv, x)

@inline texref_slot(::Raycore.TextureRef{A, T, N, TIdx}) where {A, T, N, TIdx} = TIdx

"""
    to_device_material(dhv, mat) -> mat′

Rewrite every texture-carrying field of `mat` into a `TexHandle`, storing
whatever needs a slot in `dhv`. A field carries a texture exactly when its
declared type is a free type parameter of the struct — concretely-typed fields
(`eta::Float32`, `remap_roughness::Bool`, `max_depth::Int32`, `SetKey`s) are
left to Raycore's own conversion.
"""
@generated function to_device_material(dhv, mat::M) where {M <: Material}
    wrapper = Base.typename(M).wrapper
    declared = fieldtypes(Base.unwrap_unionall(wrapper))
    args = Any[]
    for (i, fname) in enumerate(fieldnames(M))
        v = :(getfield(mat, $(QuoteNode(fname))))
        push!(args, declared[i] isa TypeVar ? :(device_param(dhv, $v)) :
                                              :(Raycore.maybe_convert_field(dhv, $v)))
    end
    return :($(wrapper)($(args...)))
end

# ...but a stored MATERIAL field is a `TexHandle`, not a raw scalar, and the
# overload above would unwrap a const `Texture` to its `RGBSpectrum` — which
# `setindex!` then refuses to put into a `Vector{Diffuse{TexHandle,…}}`. The
# error names two `Diffuse` parametrisations and no call site, so it reads as a
# type puzzle rather than as "this field is a handle".
#
# `device_param` is the rule for turning anything into a handle, and it is the
# same one `to_device_material` uses at push time — so an update now lands the
# item in exactly the form a push would have.
Raycore.update_item(dhv::Raycore.MultiTypeSet, ::TexHandle, new) = device_param(dhv, new)

"""
    update_item(dhv, old::TexHandle, new::Texture)

Overwrite the texture slot `old` already names, instead of storing another copy.

`Raycore.store_texture` always appends, so resolving a material a second time
used to mean a second upload and a second slot — which is why this combination
used to `error` outright rather than grow the store without bound. But a handle
of kind `IMAGE` carries exactly what is needed to avoid that: `slot` is the
index into `texture_order` (the GPU array TYPE) and `idx` is the element within
it, which together are the contents of a `TextureRef`. Rebuilding that ref lets
this take the same `copyto_texture!` path `TextureRef` fields already use, so
the data lands in the existing array and the handle is returned unchanged.

The error this replaces was not a small inconvenience. It fired whenever a plot
carrying a textured material was re-resolved — setting `pose` on a skinned mesh
re-resolves the mesh's whole compute edge, `material` among it — so animating
such a plot meant discarding the screen and rebuilding the scene, and its BVH,
for every frame. Measured on the raven at 640x360: 33.7 s to build, 0.8 s to
re-pose on the same screen.

A handle that is NOT an image (a constant, or unset) has no slot to reuse, so
that case stores once, which is a real material change rather than a repeat.
"""
function Raycore.update_item(dhv::Raycore.MultiTypeSet, old::TexHandle, new::Texture)
    (new.isconst || old.kind != TexKind.IMAGE || old.slot <= 0) &&
        return device_param(dhv, new)
    AT = dhv.texture_order[old.slot]
    ref = Raycore.TextureRef{AT, eltype(AT), ndims(AT), Int(old.slot)}(Int(old.idx))
    Raycore.copyto_texture!(dhv, ref, new.data)
    return old
end

