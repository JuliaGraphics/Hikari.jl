# Ported from Blender Cycles v5.2.1: intern/cycles/util/hash.h.
# Copyright 2011-2022 Blender Foundation. SPDX-License-Identifier: Apache-2.0.
# The Jenkins hash is lookup3 by Bob Jenkins (public domain), the PCG hashes are
# from "Hash Functions for GPU Rendering", JCGT 2020. Modified: rewritten in
# Julia. See LICENSES/Apache-2.0.txt.

@inline _rot(x::UInt32, k) = (x << k) | (x >> (32 - k))

@inline function _lookup3_mix(a::UInt32, b::UInt32, c::UInt32)
    a -= c; a ⊻= _rot(c, 4);  c += b
    b -= a; b ⊻= _rot(a, 6);  a += c
    c -= b; c ⊻= _rot(b, 8);  b += a
    a -= c; a ⊻= _rot(c, 16); c += b
    b -= a; b ⊻= _rot(a, 19); a += c
    c -= b; c ⊻= _rot(b, 4);  b += a
    return a, b, c
end

@inline function _lookup3_final(a::UInt32, b::UInt32, c::UInt32)
    c ⊻= b; c -= _rot(b, 14)
    a ⊻= c; a -= _rot(c, 11)
    b ⊻= a; b -= _rot(a, 25)
    c ⊻= b; c -= _rot(b, 16)
    a ⊻= c; a -= _rot(c, 4)
    b ⊻= a; b -= _rot(a, 14)
    c ⊻= b; c -= _rot(b, 24)
    return a, b, c
end

@inline _seed(n) = 0xdeadbeef + UInt32(n << 2) + UInt32(13)

@inline function _hash_uint(kx::UInt32)
    a = b = c = _seed(1)
    a += kx
    return _lookup3_final(a, b, c)[3]
end
@inline function _hash_uint(kx::UInt32, ky::UInt32)
    a = b = c = _seed(2)
    b += ky
    a += kx
    return _lookup3_final(a, b, c)[3]
end
@inline function _hash_uint(kx::UInt32, ky::UInt32, kz::UInt32)
    a = b = c = _seed(3)
    c += kz
    b += ky
    a += kx
    return _lookup3_final(a, b, c)[3]
end
@inline function _hash_uint(kx::UInt32, ky::UInt32, kz::UInt32, kw::UInt32)
    a = b = c = _seed(4)
    a += kx
    b += ky
    c += kz
    a, b, c = _lookup3_mix(a, b, c)
    a += kw
    return _lookup3_final(a, b, c)[3]
end

@inline _uint_to_float_incl(n::UInt32) = Float32(n) * (1f0 / Float32(0xFFFFFFFF))
@inline _bits(x::Float32) = reinterpret(UInt32, x)

@inline _hash_to_float(k::Float32) = _uint_to_float_incl(_hash_uint(_bits(k)))
@inline _hash_to_float(k::Vec2f) = _uint_to_float_incl(_hash_uint(_bits(k[1]), _bits(k[2])))
@inline _hash_to_float(k::Vec3f) = _uint_to_float_incl(_hash_uint(_bits(k[1]), _bits(k[2]), _bits(k[3])))
@inline _hash_to_float(k::Vec4f) = _uint_to_float_incl(_hash_uint(_bits(k[1]), _bits(k[2]), _bits(k[3]), _bits(k[4])))

@inline _hash_to_float2(k::Float32) = Vec2f(_hash_to_float(k), _hash_to_float(Vec2f(k, 1f0)))
@inline _hash_to_float2(k::Vec2f) = Vec2f(_hash_to_float(k), _hash_to_float(Vec3f(k[1], k[2], 1f0)))
@inline _hash_to_float2(k::Vec3f) = Vec2f(_hash_to_float(k), _hash_to_float(Vec3f(k[3], k[1], k[2])))
@inline _hash_to_float2(k::Vec4f) = Vec2f(_hash_to_float(k), _hash_to_float(Vec4f(k[3], k[1], k[4], k[2])))

@inline _hash_to_float3(k::Float32) = Vec3f(_hash_to_float(k), _hash_to_float(Vec2f(k, 1f0)), _hash_to_float(Vec2f(k, 2f0)))
@inline _hash_to_float3(k::Vec2f) =
    Vec3f(_hash_to_float(k), _hash_to_float(Vec3f(k[1], k[2], 1f0)), _hash_to_float(Vec3f(k[1], k[2], 2f0)))
@inline _hash_to_float3(k::Vec3f) =
    Vec3f(_hash_to_float(k), _hash_to_float(Vec4f(k[1], k[2], k[3], 1f0)), _hash_to_float(Vec4f(k[1], k[2], k[3], 2f0)))
@inline _hash_to_float3(k::Vec4f) =
    Vec3f(_hash_to_float(k), _hash_to_float(Vec4f(k[3], k[1], k[4], k[2])), _hash_to_float(Vec4f(k[4], k[3], k[2], k[1])))

@inline _hash_to_float4(k::Vec4f) =
    Vec4f(_hash_to_float(k), _hash_to_float(Vec4f(k[4], k[1], k[2], k[3])),
          _hash_to_float(Vec4f(k[3], k[4], k[1], k[2])), _hash_to_float(Vec4f(k[2], k[3], k[4], k[1])))

"""A hash of a point to a point of the same dimension, every component in [0, 1]."""
@inline _hash_to_same(k::Float32) = _hash_to_float(k)
@inline _hash_to_same(k::Vec2f) = _hash_to_float2(k)
@inline _hash_to_same(k::Vec3f) = _hash_to_float3(k)
@inline _hash_to_same(k::Vec4f) = _hash_to_float4(k)

const _PCG_INV = 1f0 / Float32(0x7FFFFFFF)

@inline function _pcg(v::NTuple{3, Int32})
    x, y, z = v .* Int32(1664525) .+ Int32(1013904223)
    x += y * z; y += z * x; z += x * y
    x ⊻= x >> 16; y ⊻= y >> 16; z ⊻= z >> 16
    x += y * z; y += z * x; z += x * y
    return (x, y, z) .& Int32(0x7FFFFFFF)
end
@inline function _pcg(v::NTuple{4, Int32})
    x, y, z, w = v .* Int32(1664525) .+ Int32(1013904223)
    x += y * w; y += z * x; z += x * y; w += y * z
    x ⊻= x >> 16; y ⊻= y >> 16; z ⊻= z >> 16; w ⊻= w >> 16
    x += y * w; y += z * x; z += x * y; w += y * z
    return (x, y, z, w) .& Int32(0x7FFFFFFF)
end
@inline function _pcg(v::NTuple{2, Int32})
    x, y = v .* Int32(1664525) .+ Int32(1013904223)
    x += y * Int32(1664525); y += x * Int32(1664525)
    x ⊻= x >> 16; y ⊻= y >> 16
    x += y * Int32(1664525); y += x * Int32(1664525)
    return (x, y) .& Int32(0x7FFFFFFF)
end

@inline _hash_int_to_float(v::NTuple{N, Int32}) where {N} = Vec{N, Float32}(Float32.(_pcg(v)) .* _PCG_INV)
