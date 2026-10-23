# Ported from Blender Cycles v5.2.1: intern/cycles/kernel/svm/color_util.h,
# mix.h, hsv.h, gamma.h, brightness.h, invert.h, ramp.h, ramp_util.h and
# intern/cycles/util/color.h. Copyright 2011-2022 Blender Foundation.
# SPDX-License-Identifier: Apache-2.0. Modified: rewritten in Julia.
# See LICENSES/Apache-2.0.txt.

module BlendType
const MIX         = UInt8(1)
const ADD         = UInt8(2)
const MULTIPLY    = UInt8(3)
const SCREEN      = UInt8(4)
const OVERLAY     = UInt8(5)
const SUBTRACT    = UInt8(6)
const DIVIDE      = UInt8(7)
const DIFFERENCE  = UInt8(8)
const EXCLUSION   = UInt8(9)
const DARKEN      = UInt8(10)
const LIGHTEN     = UInt8(11)
const DODGE       = UInt8(12)
const BURN        = UInt8(13)
const HUE         = UInt8(14)
const SATURATION  = UInt8(15)
const VALUE       = UInt8(16)
const COLOR       = UInt8(17)
const SOFT_LIGHT  = UInt8(18)
const LINEAR_LIGHT = UInt8(19)
end

module ColorModel
const RGB = UInt8(1)
const HSV = UInt8(2)
const HSL = UInt8(3)
end

@inline _interp(a, b, t) = a + t * (b - a)

function _rgb_to_hsv(rgb::Vec3f)
    cmax = max(rgb[1], rgb[2], rgb[3])
    cmin = min(rgb[1], rgb[2], rgb[3])
    cdelta = cmax - cmin
    v = cmax
    s = cmax != 0f0 ? cdelta / cmax : 0f0
    h = 0f0
    if s != 0f0
        c = (_splat(cmax) - rgb) / cdelta
        h = rgb[1] == cmax ? c[3] - c[2] : rgb[2] == cmax ? 2f0 + c[1] - c[3] : 4f0 + c[2] - c[1]
        h /= 6f0
        h < 0f0 && (h += 1f0)
    end
    return Vec3f(h, s, v)
end

function _hsv_to_rgb(hsv::Vec3f)
    h, s, v = hsv
    s == 0f0 && return Vec3f(v, v, v)
    h == 1f0 && (h = 0f0)
    h *= 6f0
    i = floor(h)
    f = h - i
    p = v * (1f0 - s)
    q = v * (1f0 - s * f)
    t = v * (1f0 - s * (1f0 - f))
    i == 0f0 && return Vec3f(v, t, p)
    i == 1f0 && return Vec3f(q, v, p)
    i == 2f0 && return Vec3f(p, v, t)
    i == 3f0 && return Vec3f(p, q, v)
    i == 4f0 && return Vec3f(t, p, v)
    return Vec3f(v, p, q)
end

function _rgb_to_hsl(rgb::Vec3f)
    cmax = max(rgb[1], rgb[2], rgb[3])
    cmin = min(rgb[1], rgb[2], rgb[3])
    l = min(1f0, (cmax + cmin) / 2f0)
    cmax == cmin && return Vec3f(0f0, 0f0, l)
    cdelta = cmax - cmin
    s = l > 0.5f0 ? cdelta / (2f0 - cmax - cmin) : cdelta / (cmax + cmin)
    h = if cmax == rgb[1]
        (rgb[2] - rgb[3]) / cdelta + (rgb[2] < rgb[3] ? 6f0 : 0f0)
    elseif cmax == rgb[2]
        (rgb[3] - rgb[1]) / cdelta + 2f0
    else
        (rgb[1] - rgb[2]) / cdelta + 4f0
    end
    return Vec3f(h / 6f0, s, l)
end

function _hsl_to_rgb(hsl::Vec3f)
    h, s, l = hsl
    nr = clamp(abs(h * 6f0 - 3f0) - 1f0, 0f0, 1f0)
    ng = clamp(2f0 - abs(h * 6f0 - 2f0), 0f0, 1f0)
    nb = clamp(2f0 - abs(h * 6f0 - 4f0), 0f0, 1f0)
    chroma = (1f0 - abs(2f0 * l - 1f0)) * s
    return Vec3f((nr - 0.5f0) * chroma + l, (ng - 0.5f0) * chroma + l, (nb - 0.5f0) * chroma + l)
end

@inline function _overlay(t::Float32, a::Float32, b::Float32)
    tm = 1f0 - t
    return a < 0.5f0 ? a * (tm + 2f0 * t * b) : 1f0 - (tm + 2f0 * t * (1f0 - b)) * (1f0 - a)
end
@inline _divide(t::Float32, a::Float32, b::Float32) = b != 0f0 ? (1f0 - t) * a + t * a / b : a
@inline function _dodge(t::Float32, a::Float32, b::Float32)
    a == 0f0 && return a
    tmp = 1f0 - t * b
    tmp <= 0f0 && return 1f0
    return min(a / tmp, 1f0)
end
@inline function _burn(t::Float32, a::Float32, b::Float32)
    tmp = (1f0 - t) + t * b
    tmp <= 0f0 && return 0f0
    return clamp(1f0 - (1f0 - a) / tmp, 0f0, 1f0)
end

function _mix_hue(t, a::Vec3f, b::Vec3f)
    hsv2 = _rgb_to_hsv(b)
    hsv2[2] == 0f0 && return a
    hsv = _rgb_to_hsv(a)
    return _interp(a, _hsv_to_rgb(Vec3f(hsv2[1], hsv[2], hsv[3])), t)
end
function _mix_saturation(t, a::Vec3f, b::Vec3f)
    hsv = _rgb_to_hsv(a)
    hsv[2] == 0f0 && return a
    s = (1f0 - t) * hsv[2] + t * _rgb_to_hsv(b)[2]
    return _hsv_to_rgb(Vec3f(hsv[1], s, hsv[3]))
end
function _mix_value(t, a::Vec3f, b::Vec3f)
    hsv = _rgb_to_hsv(a)
    v = (1f0 - t) * hsv[3] + t * _rgb_to_hsv(b)[3]
    return _hsv_to_rgb(Vec3f(hsv[1], hsv[2], v))
end
function _mix_color(t, a::Vec3f, b::Vec3f)
    hsv2 = _rgb_to_hsv(b)
    hsv2[2] == 0f0 && return a
    hsv = _rgb_to_hsv(a)
    return _interp(a, _hsv_to_rgb(Vec3f(hsv2[1], hsv2[2], hsv[3])), t)
end
function _mix_soft(t, a::Vec3f, b::Vec3f)
    one = Vec3f(1)
    scr = one - (one - b) .* (one - a)
    return (1f0 - t) * a + t * ((one - a) .* b .* a + a .* scr)
end

"""Blender's Mix node for colours, with a scalar factor."""
@noinline function _blend(op::UInt8, t::Float32, a::Vec3f, b::Vec3f)
    B = BlendType
    op == B.MIX          && return _interp(a, b, t)
    op == B.ADD          && return _interp(a, a + b, t)
    op == B.MULTIPLY     && return _interp(a, a .* b, t)
    op == B.SCREEN       && return Vec3f(1) - (_splat(1f0 - t) + t * (Vec3f(1) - b)) .* (Vec3f(1) - a)
    op == B.OVERLAY      && return _overlay.(t, a, b)
    op == B.SUBTRACT     && return _interp(a, a - b, t)
    op == B.DIVIDE       && return _divide.(t, a, b)
    op == B.DIFFERENCE   && return _interp(a, abs.(a - b), t)
    op == B.EXCLUSION    && return max.(_interp(a, a + b - 2f0 * a .* b, t), 0f0)
    op == B.DARKEN       && return _interp(a, min.(a, b), t)
    op == B.LIGHTEN      && return _interp(a, max.(a, b), t)
    op == B.DODGE        && return _dodge.(t, a, b)
    op == B.BURN         && return _burn.(t, a, b)
    op == B.HUE          && return _mix_hue(t, a, b)
    op == B.SATURATION   && return _mix_saturation(t, a, b)
    op == B.VALUE        && return _mix_value(t, a, b)
    op == B.COLOR        && return _mix_color(t, a, b)
    op == B.SOFT_LIGHT   && return _mix_soft(t, a, b)
    op == B.LINEAR_LIGHT && return a + t * (2f0 * b - Vec3f(1))
    return Vec3f(0)
end

"""The Mix node. A vector factor mixes per component, which only the plain blend does."""
@noinline function _mix_node(ins::NodeInstr, t::Vec3f, a::Vec3f, b::Vec3f)
    blend, clamp_factor, clamp_result = ins.mode
    t = clamp_factor == 0x01 ? clamp.(t, 0f0, 1f0) : t
    result = blend == BlendType.MIX ? (Vec3f(1) - t) .* a + t .* b : _blend(blend, t[1], a, b)
    return clamp_result == 0x01 ? clamp.(result, 0f0, 1f0) : result
end

@noinline function _hsv_node(fac::Float32, color::Vec3f, hue::Float32, sat::Float32, val::Float32)
    hsv = _rgb_to_hsv(color)
    adjusted = _hsv_to_rgb(Vec3f(_fract(hsv[1] + hue + 0.5f0), _saturate(hsv[2] * sat), hsv[3] * val))
    return max.(fac * adjusted + (1f0 - fac) * color, 0f0)
end

@inline function _gamma(color::Vec3f, gamma::Float32)
    gamma == 0f0 && return Vec3f(1)
    return Vec3f(ntuple(i -> color[i] > 0f0 ? color[i]^gamma : color[i], 3))
end

@inline function _bright_contrast(color::Vec3f, bright::Float32, contrast::Float32)
    a = 1f0 + contrast
    b = bright - contrast * 0.5f0
    return max.(a * color .+ b, 0f0)
end

@inline _invert(color::Vec3f, fac::Float32) = fac * (Vec3f(1) - color) + (1f0 - fac) * color

"""Luminance of linear Rec. 709 (sRGB) primaries, the space Hikari renders in."""
@inline _rgb_to_bw(c::Vec3f) = 0.2126f0 * c[1] + 0.7152f0 * c[2] + 0.0722f0 * c[3]

@inline function _separate_color(model::UInt8, c::Vec3f)
    model == ColorModel.HSV && return _rgb_to_hsv(c)
    model == ColorModel.HSL && return _rgb_to_hsl(c)
    return c
end
@inline function _combine_color(model::UInt8, c::Vec3f)
    model == ColorModel.HSV && return _hsv_to_rgb(c)
    model == ColorModel.HSL && return _hsl_to_rgb(c)
    return c
end

# A ramp is a table of `RAMP_TABLE_SIZE` colours stored in the `DATA`
# instructions after the `RAMP` one, three entries to an instruction.
const RAMP_TABLE_SIZE = 256
const RAMP_ENTRIES_PER_DATA = 3

@propagate_inbounds function _ramp_entry(prog, i::Int32, entry::Int32)
    data = prog[i + Int32(1) + div(entry, Int32(RAMP_ENTRIES_PER_DATA))]
    first = Int32(3) * rem(entry, Int32(RAMP_ENTRIES_PER_DATA)) + Int32(1)
    return Vec3f(_kat(data.k, first), _kat(data.k, first + Int32(1)), _kat(data.k, first + Int32(2)))
end

@propagate_inbounds function _ramp_lookup(prog, i::Int32, f::Float32, interpolate::Bool)
    n = Int32(RAMP_TABLE_SIZE)
    f = _saturate(f) * Float32(n - Int32(1))
    j = clamp(_to_int(f), Int32(0), n - Int32(1))
    t = f - Float32(j)
    result = _ramp_entry(prog, i, j)
    interpolate && t > 0f0 && (result = (1f0 - t) * result + t * _ramp_entry(prog, i, j + Int32(1)))
    return result
end
