# ============================================================================
# Postprocessing Pipeline
# ============================================================================
#
# Reads from film.framebuffer (sensor-calibrated linear HDR sRGB),
# writes to film.postprocess (display-ready RGBA).
#
# This is purely a display mapping step. All sensor simulation
# (ISO, exposure_time, white_balance, sensor curves) is applied during
# rendering and already baked into the framebuffer.
#
# Can be called multiple times with different parameters without re-rendering.

using ImageCore: RGB

# ============================================================================
# Tonemapping
# ============================================================================

@propagate_inbounds function tonemap_reinhard(r::Float32, g::Float32, b::Float32)
    lum = 0.2126f0 * r + 0.7152f0 * g + 0.0722f0 * b
    scale = ifelse(lum > 0f0, 1f0 / (1f0 + lum), 1f0)
    clamp(r * scale, 0f0, 1f0), clamp(g * scale, 0f0, 1f0), clamp(b * scale, 0f0, 1f0)
end

@propagate_inbounds function tonemap_reinhard_extended(r::Float32, g::Float32, b::Float32, Lwhite::Float32)
    lum = 0.2126f0 * r + 0.7152f0 * g + 0.0722f0 * b
    Lwhite2 = Lwhite * Lwhite
    scale = ifelse(lum > 0f0, (1f0 + lum / Lwhite2) / (1f0 + lum), 1f0)
    clamp(r * scale, 0f0, 1f0), clamp(g * scale, 0f0, 1f0), clamp(b * scale, 0f0, 1f0)
end

@propagate_inbounds function tonemap_aces(r::Float32, g::Float32, b::Float32)
    a = 2.51f0
    b_c = 0.03f0
    c = 2.43f0
    d = 0.59f0
    e = 0.14f0
    r_out = clamp((r * (a * r + b_c)) / (r * (c * r + d) + e), 0f0, 1f0)
    g_out = clamp((g * (a * g + b_c)) / (g * (c * g + d) + e), 0f0, 1f0)
    b_out = clamp((b * (a * b + b_c)) / (b * (c * b + d) + e), 0f0, 1f0)
    r_out, g_out, b_out
end

@propagate_inbounds function uncharted2_partial(x::Float32)
    A = 0.15f0; B = 0.50f0; C = 0.10f0; D = 0.20f0; E = 0.02f0; F = 0.30f0
    ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
end

@propagate_inbounds function tonemap_uncharted2(r::Float32, g::Float32, b::Float32)
    W = 11.2f0
    exposure_bias = 2.0f0
    r_out = uncharted2_partial(r * exposure_bias)
    g_out = uncharted2_partial(g * exposure_bias)
    b_out = uncharted2_partial(b * exposure_bias)
    white_scale = 1f0 / uncharted2_partial(W)
    clamp(r_out * white_scale, 0f0, 1f0),
    clamp(g_out * white_scale, 0f0, 1f0),
    clamp(b_out * white_scale, 0f0, 1f0)
end

@propagate_inbounds function tonemap_filmic(r::Float32, g::Float32, b::Float32)
    function filmic_channel(x::Float32)
        x = max(0f0, x - 0.004f0)
        (x * (6.2f0 * x + 0.5f0)) / (x * (6.2f0 * x + 1.7f0) + 0.06f0)
    end
    filmic_channel(r), filmic_channel(g), filmic_channel(b)
end

const TONEMAP_NONE = UInt8(0)
const TONEMAP_REINHARD = UInt8(1)
const TONEMAP_REINHARD_EXT = UInt8(2)
const TONEMAP_ACES = UInt8(3)
const TONEMAP_UNCHARTED2 = UInt8(4)
const TONEMAP_FILMIC = UInt8(5)

@propagate_inbounds function apply_tonemap(r::Float32, g::Float32, b::Float32, mode::UInt8, wp::Float32)
    if mode == TONEMAP_REINHARD
        return tonemap_reinhard(r, g, b)
    elseif mode == TONEMAP_REINHARD_EXT
        return tonemap_reinhard_extended(r, g, b, wp)
    elseif mode == TONEMAP_ACES
        return tonemap_aces(r, g, b)
    elseif mode == TONEMAP_UNCHARTED2
        return tonemap_uncharted2(r, g, b)
    elseif mode == TONEMAP_FILMIC
        return tonemap_filmic(r, g, b)
    else
        return clamp(r, 0f0, 1f0), clamp(g, 0f0, 1f0), clamp(b, 0f0, 1f0)
    end
end

# ============================================================================
# Postprocess Kernel
# ============================================================================

@kernel inbounds=true function postprocess_kernel!(dst, @Const(src), @Const(depth),
                                      exposure::Float32, tonemap_mode::UInt8,
                                      inv_gamma::Float32, apply_gamma::Bool, white_point::Float32,
                                      mask_escaped::Bool, sky::Bool,
                                      bg_r::Float32, bg_g::Float32, bg_b::Float32, bg_a::Float32,
                                      depth_h::Int32, depth_w::Int32)
    i = @index(Global, Linear)
    begin
        c = src[i]
        r = c.r * exposure
        g = c.g * exposure
        b = c.b * exposure
        r, g, b = apply_tonemap(r, g, b, tonemap_mode, white_point)
        if apply_gamma
            r = r^inv_gamma
            g = g^inv_gamma
            b = b^inv_gamma
        end

        # Background compositing for escaped rays
        out_a = 1f0
        if mask_escaped
            row = Int32(((i - 1) % depth_h) + 1)
            col = Int32(((i - 1) ÷ depth_h) + 1)
            d_row = depth_h - row + Int32(1)  # Y-flip

            escaped = Int32(0)
            total = Int32(0)
            for dr in Int32(-1):Int32(1)
                for dc in Int32(-1):Int32(1)
                    nr = d_row + dr
                    nc = col + dc
                    inside = (nr >= Int32(1)) & (nr <= depth_h) & (nc >= Int32(1)) & (nc <= depth_w)
                    if inside
                        didx = (nc - Int32(1)) * depth_h + nr
                        escaped += Int32(missed(depth[didx]))
                        total += Int32(1)
                    end
                end
            end

            # `escaped / total` with a fast reciprocal (this backend lowers fdiv
            # that way: 6f0/6f0 comes out 0.99999994) must not stand in for the
            # endpoints — a fully escaped pixel has to read EXACTLY 1 here, or
            # the output alpha below is 2^-24 where it must be 0.
            alpha = escaped == total ? 1f0 : Float32(escaped) / Float32(total)
            # What fills the escaped part of the pixel. `sky` says an infinite
            # light already painted radiance there (an environment map, a sun, an
            # ambient term), and then the picture's own colour is what fills it —
            # painting the background over that is how an environment map used to
            # come back as a flat white rectangle.
            fill_r = sky ? r : bg_r
            fill_g = sky ? g : bg_g
            fill_b = sky ? b : bg_b
            # Weighted by the background's alpha, which is the one thing that
            # decides how much of the backdrop is there at all: opaque leaves the
            # sky (or the background colour) standing, transparent leaves the
            # escaped part empty and the colour premultiplied for a compositing
            # caller — including at the antialiased fringe, where `alpha` is
            # between the two.
            r = r * (1f0 - alpha) + fill_r * bg_a * alpha
            g = g * (1f0 - alpha) + fill_g * bg_a * alpha
            b = b * (1f0 - alpha) + fill_b * bg_a * alpha
            # The pixel's REAL alpha: the background's, plus what the rays hit on
            # top of it. An opaque background keeps alpha 1 everywhere (as this
            # always was); a transparent one lets the misses read as uncovered —
            # a renderer's caller then needs no separate coverage mask.
            out_a = bg_a + (1f0 - bg_a) * (1f0 - alpha)
        end

        dst[i] = RGBA{Float32}(r, g, b, out_a)
    end
end

# ============================================================================
# Public API
# ============================================================================

"""
    postprocess!(film; exposure=1.0, tonemap=:aces, gamma=2.2, white_point=4.0, background=nothing)

Convert the HDR framebuffer to a display-ready image.

This is a pure display mapping step. The framebuffer already contains
sensor-calibrated linear sRGB (ISO, exposure_time, white_balance are
applied during rendering). Call this multiple times with different
parameters without re-rendering.

# Arguments
- `exposure`: Linear brightness multiplier applied before tonemapping
- `tonemap`: Tonemapping curve. One of:
  - `:aces` (default) - ACES filmic, industry standard
  - `:reinhard` - simple Reinhard L/(1+L)
  - `:reinhard_extended` - extended Reinhard with white point
  - `:uncharted2` - Uncharted 2 filmic
  - `:filmic` - Hejl-Dawson filmic
  - `nothing` - linear clamp (no tonemapping)
- `gamma`: Gamma correction exponent (default 2.2). `nothing` to skip.
- `white_point`: White point for `:reinhard_extended` (default 4.0)
- `background`: `RGB{Float32}` or `RGBA{Float32}` color for compositing escaped rays
  (see [`missed`](@ref)). Its alpha becomes the output alpha of the escaped fraction:
  an opaque background yields alpha 1 everywhere, a transparent one makes misses
  read as uncovered — and at the antialiased fringe, where a pixel is half
  escaped, color and alpha are then already premultiplied.
- `sky`: whether an infinite light (environment map, sun, ambient) already painted
  the escaped rays. Then the escaped part of a pixel keeps the colour the render
  gave it instead of the background's — but its alpha still comes from the
  background, so a caller asking for a transparent background still gets coverage
  and not a sky.
"""
function postprocess!(film::Film;
    exposure::Real = 1.0,
    tonemap::Union{Symbol, Nothing} = :aces,
    gamma::Union{Real, Nothing} = 2.2,
    white_point::Real = 4.0,
    background::Union{RGB{Float32}, RGBA{Float32}, Nothing} = nothing,
    sky::Bool = false,
)
    src = film.framebuffer
    dst = film.postprocess

    exp_f32 = Float32(exposure)
    wp_f32 = Float32(white_point)
    inv_gamma = isnothing(gamma) ? 1.0f0 : 1f0 / Float32(gamma)
    apply_gamma = !isnothing(gamma)

    tonemap_mode = if tonemap === :reinhard
        TONEMAP_REINHARD
    elseif tonemap === :reinhard_extended
        TONEMAP_REINHARD_EXT
    elseif tonemap === :aces
        TONEMAP_ACES
    elseif tonemap === :uncharted2
        TONEMAP_UNCHARTED2
    elseif tonemap === :filmic
        TONEMAP_FILMIC
    else
        TONEMAP_NONE
    end

    mask_escaped = !isnothing(background)
    bg_r = mask_escaped ? background.r : 0f0
    bg_g = mask_escaped ? background.g : 0f0
    bg_b = mask_escaped ? background.b : 0f0
    bg_a = mask_escaped ? Float32(alpha(background)) : 1f0   # alpha(::RGB) == 1

    backend = KernelAbstractions.get_backend(src)
    kernel! = postprocess_kernel!(backend)
    kernel!(dst, src, film.depth, exp_f32, tonemap_mode, inv_gamma, apply_gamma, wp_f32,
            mask_escaped, sky, bg_r, bg_g, bg_b, bg_a,
            Int32(size(src, 1)), Int32(size(src, 2));
            ndrange=length(src))
    Mantle.waitidle(mantle_device(backend))

    return film.postprocess
end
