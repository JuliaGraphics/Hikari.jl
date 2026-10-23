# Ported from Blender Cycles v5.2.1: intern/cycles/kernel/svm/math_util.h,
# map_range.h, clamp.h and intern/cycles/util/math_base.h, math_float3.h.
# Copyright 2011-2022 Blender Foundation. SPDX-License-Identifier: Apache-2.0.
# Modified: rewritten in Julia, every operation applied per component of a Vec3f.
# See LICENSES/Apache-2.0.txt.

module MathOp
const ADD            = UInt8(1)
const SUBTRACT       = UInt8(2)
const MULTIPLY       = UInt8(3)
const DIVIDE         = UInt8(4)
const MULTIPLY_ADD   = UInt8(5)
const POWER          = UInt8(6)
const LOGARITHM      = UInt8(7)
const SQRT           = UInt8(8)
const INVERSE_SQRT   = UInt8(9)
const ABSOLUTE       = UInt8(10)
const EXPONENT       = UInt8(11)
const MINIMUM        = UInt8(12)
const MAXIMUM        = UInt8(13)
const LESS_THAN      = UInt8(14)
const GREATER_THAN   = UInt8(15)
const SIGN           = UInt8(16)
const COMPARE        = UInt8(17)
const SMOOTH_MIN     = UInt8(18)
const SMOOTH_MAX     = UInt8(19)
const ROUND          = UInt8(20)
const FLOOR          = UInt8(21)
const CEIL           = UInt8(22)
const TRUNCATE       = UInt8(23)
const FRACTION       = UInt8(24)
const MODULO         = UInt8(25)
const FLOORED_MODULO = UInt8(26)
const WRAP           = UInt8(27)
const SNAP           = UInt8(28)
const PINGPONG       = UInt8(29)
const SINE           = UInt8(30)
const COSINE         = UInt8(31)
const TANGENT        = UInt8(32)
const ARCSINE        = UInt8(33)
const ARCCOSINE      = UInt8(34)
const ARCTANGENT     = UInt8(35)
const ARCTAN2        = UInt8(36)
const SINH           = UInt8(37)
const COSH           = UInt8(38)
const TANH           = UInt8(39)
const RADIANS        = UInt8(40)
const DEGREES        = UInt8(41)
end

module VectorMathOp
const CROSS       = UInt8(1)
const PROJECT     = UInt8(2)
const REFLECT     = UInt8(3)
const REFRACT     = UInt8(4)
const FACEFORWARD = UInt8(5)
const DOT         = UInt8(6)
const DISTANCE    = UInt8(7)
const LENGTH      = UInt8(8)
const SCALE       = UInt8(9)
const NORMALIZE   = UInt8(10)
end

@inline _safe_divide(a::Float32, b::Float32) = b != 0f0 ? a / b : 0f0
@inline _saturate(x::Float32) = clamp(x, 0f0, 1f0)
@inline _fract(x::Float32) = x - floor(x)
@inline _to_int(x::Float32) = unsafe_trunc(Int32, x)

@inline function _compatible_pow(x::Float32, y::Float32)
    y == 0f0 && return 1f0
    x == 0f0 && return 0f0
    if x < 0f0
        return rem(-y, 2f0) == 0f0 ? (-x)^y : -((-x)^y)
    end
    return x^y
end
@inline _safe_pow(a::Float32, b::Float32) = a < 0f0 && b != Float32(_to_int(b)) ? 0f0 : _compatible_pow(a, b)
@inline _safe_log(a::Float32, b::Float32) = a <= 0f0 || b <= 0f0 ? 0f0 : _safe_divide(log(a), log(b))
@inline _safe_sqrt(a::Float32) = sqrt(max(a, 0f0))
@inline _inverse_sqrt(a::Float32) = a > 0f0 ? 1f0 / sqrt(a) : 0f0
@inline _safe_modulo(a::Float32, b::Float32) = b != 0f0 ? rem(a, b) : 0f0
@inline _safe_floored_modulo(a::Float32, b::Float32) = b != 0f0 ? a - floor(a / b) * b : 0f0
@inline function _wrap(value::Float32, hi::Float32, lo::Float32)
    range = hi - lo
    return range != 0f0 ? value - range * floor((value - lo) / range) : lo
end
@inline _pingpong(a::Float32, b::Float32) = b != 0f0 ? abs(_fract((a - b) / (b * 2f0)) * b * 2f0 - b) : 0f0
@inline _compatible_atan2(y::Float32, x::Float32) = x == 0f0 && y == 0f0 ? 0f0 : atan(y, x)
@inline _compatible_sign(f::Float32) = f == 0f0 ? 0f0 : f > 0f0 ? 1f0 : -1f0
@inline function _smoothmin(a::Float32, b::Float32, k::Float32)
    k == 0f0 && return min(a, b)
    h = max(k - abs(a - b), 0f0) / k
    return min(a, b) - h * h * h * k * (1f0 / 6f0)
end
@inline _compare(a::Float32, b::Float32, c::Float32) = a == b || abs(a - b) <= max(c, eps(Float32)) ? 1f0 : 0f0
@inline _truncate(a::Float32) = a >= 0f0 ? floor(a) : ceil(a)
@inline _b2f(b::Bool) = ifelse(b, 1f0, 0f0)

"""Blender's Math node on scalars, or per component on vectors."""
@inline function _math(op::UInt8, a::T, b::T, c::T) where {T <: Union{Float32, Vec3f}}
    M = MathOp
    op == M.MULTIPLY       && return a .* b
    op == M.ADD            && return a + b
    op == M.SUBTRACT       && return a - b
    op == M.SINE           && return sin.(a)
    op == M.COSINE         && return cos.(a)
    op == M.DIVIDE         && return _safe_divide.(a, b)
    op == M.MULTIPLY_ADD   && return a .* b + c
    op == M.POWER          && return _safe_pow.(a, b)
    op == M.LOGARITHM      && return _safe_log.(a, b)
    op == M.SQRT           && return _safe_sqrt.(a)
    op == M.INVERSE_SQRT   && return _inverse_sqrt.(a)
    op == M.ABSOLUTE       && return abs.(a)
    op == M.EXPONENT       && return exp.(a)
    op == M.MINIMUM        && return min.(a, b)
    op == M.MAXIMUM        && return max.(a, b)
    op == M.LESS_THAN      && return _b2f.(a .< b)
    op == M.GREATER_THAN   && return _b2f.(a .> b)
    op == M.SIGN           && return _compatible_sign.(a)
    op == M.COMPARE        && return _compare.(a, b, c)
    op == M.SMOOTH_MIN     && return _smoothmin.(a, b, c)
    op == M.SMOOTH_MAX     && return -_smoothmin.(-a, -b, c)
    op == M.ROUND          && return floor.(a .+ 0.5f0)
    op == M.FLOOR          && return floor.(a)
    op == M.CEIL           && return ceil.(a)
    op == M.TRUNCATE       && return _truncate.(a)
    op == M.FRACTION       && return a - floor.(a)
    op == M.MODULO         && return _safe_modulo.(a, b)
    op == M.FLOORED_MODULO && return _safe_floored_modulo.(a, b)
    op == M.WRAP           && return _wrap.(a, b, c)
    op == M.SNAP           && return floor.(_safe_divide.(a, b)) .* b
    op == M.PINGPONG       && return _pingpong.(a, b)
    op == M.SINE           && return sin.(a)
    op == M.COSINE         && return cos.(a)
    op == M.TANGENT        && return tan.(a)
    op == M.ARCSINE        && return asin.(clamp.(a, -1f0, 1f0))
    op == M.ARCCOSINE      && return acos.(clamp.(a, -1f0, 1f0))
    op == M.ARCTANGENT     && return atan.(a)
    op == M.ARCTAN2        && return _compatible_atan2.(a, b)
    op == M.SINH           && return sinh.(a)
    op == M.COSH           && return cosh.(a)
    op == M.TANH           && return tanh.(a)
    op == M.RADIANS        && return a * Float32(π / 180)
    op == M.DEGREES        && return a * Float32(180 / π)
    return zero(a)
end

@inline function _safe_normalize(a::Vec3f)
    t = norm(a)
    return t != 0f0 ? a * (1f0 / t) : a
end
@inline function _project(v::Vec3f, onto::Vec3f)
    len_squared = dot(onto, onto)
    return len_squared != 0f0 ? (dot(v, onto) / len_squared) * onto : Vec3f(0)
end
@inline _reflect(incident::Vec3f, unit_normal::Vec3f) = incident - 2f0 * unit_normal * dot(incident, unit_normal)
@inline function _refract(incident::Vec3f, normal::Vec3f, eta::Float32)
    d = dot(normal, incident)
    k = 1f0 - eta * eta * (1f0 - d * d)
    k < 0f0 && return Vec3f(0)
    return eta * incident - (eta * d + sqrt(k)) * normal
end
@inline _faceforward(v::Vec3f, incident::Vec3f, reference::Vec3f) = dot(reference, incident) < 0f0 ? v : -v

@inline function _vector_math(op::UInt8, a::Vec3f, b::Vec3f, c::Vec3f, scale::Float32)
    V = VectorMathOp
    op == V.CROSS       && return cross(a, b)
    op == V.PROJECT     && return _project(a, b)
    op == V.REFLECT     && return _reflect(a, _safe_normalize(b))
    op == V.REFRACT     && return _refract(a, _safe_normalize(b), scale)
    op == V.FACEFORWARD && return _faceforward(a, b, c)
    op == V.DOT         && return _splat(dot(a, b))
    op == V.DISTANCE    && return _splat(norm(a - b))
    op == V.LENGTH      && return _splat(norm(a))
    op == V.SCALE       && return a * scale
    op == V.NORMALIZE   && return _safe_normalize(a)
    return Vec3f(0)
end

module RangeType
const LINEAR       = UInt8(1)
const STEPPED      = UInt8(2)
const SMOOTHSTEP   = UInt8(3)
const SMOOTHERSTEP = UInt8(4)
end

@inline _smoothstep01(t::Float32) = (3f0 - 2f0 * t) * (t * t)
@inline _smootherstep01(t::Float32) = t * t * t * (t * (t * 6f0 - 15f0) + 10f0)

@inline function _map_range(range::UInt8, use_clamp::Bool, scalar::Bool, x::Float32, from_min::Float32,
                            from_max::Float32, to_min::Float32, to_max::Float32, steps::Float32)
    scalar && from_max == from_min && return 0f0
    factor = _safe_divide(x - from_min, from_max - from_min)
    if range == RangeType.STEPPED
        factor = steps > 0f0 ? floor(factor * (steps + 1f0)) / steps : 0f0
    elseif range == RangeType.SMOOTHSTEP
        factor = _smoothstep01(_saturate(factor))
    elseif range == RangeType.SMOOTHERSTEP
        factor = _smootherstep01(_saturate(factor))
    end
    result = to_min + factor * (to_max - to_min)
    clamps = use_clamp && (range == RangeType.LINEAR || range == RangeType.STEPPED)
    clamps || return result
    return to_min > to_max ? clamp(result, to_max, to_min) : clamp(result, to_min, to_max)
end

@inline _clamp_node(range::Bool, x::Float32, lo::Float32, hi::Float32) = range && lo > hi ? clamp(x, hi, lo) : clamp(x, lo, hi)
