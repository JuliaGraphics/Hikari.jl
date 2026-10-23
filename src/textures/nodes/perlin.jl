# Ported from Blender Cycles v5.2.1: intern/cycles/kernel/svm/noise.h, which
# Cycles adapted from Open Shading Language.
# Copyright 2009-2010 Sony Pictures Imageworks Inc., et al. All Rights Reserved.
# Copyright 2011-2022 Blender Foundation.
# SPDX-License-Identifier: BSD-3-Clause. Modified: rewritten in Julia, one
# method per dimension. See LICENSES/BSD-3-Clause.txt.

@inline _fade(t::Float32) = t * t * t * (t * (t * 6f0 - 15f0) + 10f0)
@inline _negate_if(val::Float32, condition::UInt32) = condition != 0x00000000 ? -val : val
@inline _floorfrac(x::Float32) = (f = floor(x); (x - f, reinterpret(UInt32, unsafe_trunc(Int32, f))))

@inline function _grad1(hash::UInt32, x::Float32)
    h = hash & 0x0000000f
    g = Float32(1 + (h & 0x00000007))
    return _negate_if(g, h & 0x00000008) * x
end
@inline function _grad2(hash::UInt32, x::Float32, y::Float32)
    h = hash & 0x00000007
    u = h < 0x00000004 ? x : y
    v = 2f0 * (h < 0x00000004 ? y : x)
    return _negate_if(u, h & 0x00000001) + _negate_if(v, h & 0x00000002)
end
@inline function _grad3(hash::UInt32, x::Float32, y::Float32, z::Float32)
    h = hash & 0x0000000f
    u = h < 0x00000008 ? x : y
    vt = h == 0x0000000c || h == 0x0000000e ? x : z
    v = h < 0x00000004 ? y : vt
    return _negate_if(u, h & 0x00000001) + _negate_if(v, h & 0x00000002)
end
@inline function _grad4(hash::UInt32, x::Float32, y::Float32, z::Float32, w::Float32)
    h = hash & 0x0000001f
    u = h < 0x00000018 ? x : y
    v = h < 0x00000010 ? y : z
    s = h < 0x00000008 ? z : w
    return _negate_if(u, h & 0x00000001) + _negate_if(v, h & 0x00000002) + _negate_if(s, h & 0x00000004)
end

@inline _lerp(a, b, t) = (1f0 - t) * a + t * b
@inline _mix(a, b, t) = a + t * (b - a)

@inline function _perlin(x::Float32)
    fx, X = _floorfrac(x)
    return _mix(_grad1(_hash_uint(X), fx), _grad1(_hash_uint(X + 0x1), fx - 1f0), _fade(fx))
end

@inline function _perlin(p::Vec2f)
    (fx, X), (fy, Y) = _floorfrac(p[1]), _floorfrac(p[2])
    u, v = _fade(fx), _fade(fy)
    g(i, j) = _grad2(_hash_uint(X + i, Y + j), fx - Float32(i), fy - Float32(j))
    return _lerp(_lerp(g(0x0, 0x0), g(0x1, 0x0), u), _lerp(g(0x0, 0x1), g(0x1, 0x1), u), v)
end

@inline function _perlin(p::Vec3f)
    (fx, X), (fy, Y), (fz, Z) = _floorfrac(p[1]), _floorfrac(p[2]), _floorfrac(p[3])
    u, v, w = _fade(fx), _fade(fy), _fade(fz)
    g(i, j, k) = _grad3(_hash_uint(X + i, Y + j, Z + k), fx - Float32(i), fy - Float32(j), fz - Float32(k))
    return _lerp(_lerp(_lerp(g(0x0, 0x0, 0x0), g(0x1, 0x0, 0x0), u), _lerp(g(0x0, 0x1, 0x0), g(0x1, 0x1, 0x0), u), v),
                 _lerp(_lerp(g(0x0, 0x0, 0x1), g(0x1, 0x0, 0x1), u), _lerp(g(0x0, 0x1, 0x1), g(0x1, 0x1, 0x1), u), v), w)
end

@inline function _perlin(p::Vec4f)
    (fx, X), (fy, Y), (fz, Z), (fw, W) = _floorfrac(p[1]), _floorfrac(p[2]), _floorfrac(p[3]), _floorfrac(p[4])
    u, v, t, s = _fade(fx), _fade(fy), _fade(fz), _fade(fw)
    g(i, j, k, l) = _grad4(_hash_uint(X + i, Y + j, Z + k, W + l), fx - Float32(i), fy - Float32(j), fz - Float32(k), fw - Float32(l))
    cube(l) = _lerp(_lerp(_lerp(g(0x0, 0x0, 0x0, l), g(0x1, 0x0, 0x0, l), u), _lerp(g(0x0, 0x1, 0x0, l), g(0x1, 0x1, 0x0, l), u), v),
                    _lerp(_lerp(g(0x0, 0x0, 0x1, l), g(0x1, 0x0, 0x1, l), u), _lerp(g(0x0, 0x1, 0x1, l), g(0x1, 0x1, 0x1, l), u), v), t)
    return _mix(cube(0x0), cube(0x1), s)
end

@inline _noise_scale(::Float32) = 0.2500f0
@inline _noise_scale(::Vec2f) = 0.6616f0
@inline _noise_scale(::Vec3f) = 0.9820f0
@inline _noise_scale(::Vec4f) = 0.8344f0

"""`rem(x, 100000)`, skipping the division for the coordinates that need none."""
@inline _wrap_1e5(x::Float32) = abs(x) < 100000f0 ? x : rem(x, 100000f0) + ifelse(abs(x) >= 1f6, 0.5f0, 0f0)

"""Signed Perlin noise in about [-1, 1], repeating every 100000 to keep its precision."""
@inline _snoise(p) = _noise_scale(p) * _perlin(_wrap_1e5.(p))

@inline _unoise(p) = 0.5f0 * _snoise(p) + 0.5f0
