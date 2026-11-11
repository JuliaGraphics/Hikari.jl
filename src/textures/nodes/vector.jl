# Ported from Blender Cycles v5.2.1: intern/cycles/kernel/svm/mapping_util.h,
# vector_rotate.h and intern/cycles/util/transform.h, math_float3.h.
# Copyright 2011-2022 Blender Foundation. SPDX-License-Identifier: Apache-2.0.
# Modified: rewritten in Julia. See LICENSES/Apache-2.0.txt.

module MappingType
const POINT   = UInt8(1)
const TEXTURE = UInt8(2)
const VECTOR  = UInt8(3)
const NORMAL  = UInt8(4)
end

module RotateType
const AXIS_ANGLE = UInt8(1)
const X          = UInt8(2)
const Y          = UInt8(3)
const Z          = UInt8(4)
const EULER_XYZ  = UInt8(5)
end

"""The rows of the rotation by Euler angles `e` (radians) about x, then y, then z."""
@inline function _euler_rows(e::Vec3f)
    cx, cy, cz = cos(e[1]), cos(e[2]), cos(e[3])
    sx, sy, sz = sin(e[1]), sin(e[2]), sin(e[3])
    return (Vec3f(cy * cz, sy * sx * cz - cx * sz, sy * cx * cz + sx * sz),
            Vec3f(cy * sz, sy * sx * sz + cx * cz, sy * cx * sz - sx * cz),
            Vec3f(-sy, cy * sx, cy * cx))
end
@inline _rotate(rows, v::Vec3f) = Vec3f(dot(rows[1], v), dot(rows[2], v), dot(rows[3], v))
@inline _rotate_transposed(rows, v::Vec3f) = v[1] * rows[1] + v[2] * rows[2] + v[3] * rows[3]

@inline _safe_divide3(a::Vec3f, b::Vec3f) = _safe_divide.(a, b)

@noinline function _mapping(type::UInt8, v::Vec3f, location::Vec3f, rotation::Vec3f, scale::Vec3f)
    rows = _euler_rows(rotation)
    type == MappingType.POINT   && return _rotate(rows, v .* scale) + location
    type == MappingType.TEXTURE && return _safe_divide3(_rotate_transposed(rows, v - location), scale)
    type == MappingType.VECTOR  && return _rotate(rows, v .* scale)
    type == MappingType.NORMAL  && return _safe_normalize(_rotate(rows, _safe_divide3(v, scale)))
    return Vec3f(0)
end

function _rotate_around_axis(p::Vec3f, axis::Vec3f, angle::Float32)
    c, s = cos(angle), sin(angle)
    x, y, z = axis
    return Vec3f((c + (1 - c) * x * x) * p[1] + ((1 - c) * x * y - z * s) * p[2] + ((1 - c) * x * z + y * s) * p[3],
                 ((1 - c) * x * y + z * s) * p[1] + (c + (1 - c) * y * y) * p[2] + ((1 - c) * y * z - x * s) * p[3],
                 ((1 - c) * x * z - y * s) * p[1] + ((1 - c) * y * z + x * s) * p[2] + (c + (1 - c) * z * z) * p[3])
end

@noinline function _vector_rotate(type::UInt8, invert::Bool, v::Vec3f, center::Vec3f, axis::Vec3f, angle::Float32, rotation::Vec3f)
    if type == RotateType.EULER_XYZ
        rows = _euler_rows(rotation)
        return (invert ? _rotate_transposed(rows, v - center) : _rotate(rows, v - center)) + center
    end
    axis = type == RotateType.X ? Vec3f(1, 0, 0) : type == RotateType.Y ? Vec3f(0, 1, 0) :
           type == RotateType.Z ? Vec3f(0, 0, 1) : axis
    axis_length = norm(axis)
    axis_length == 0f0 && return v
    return _rotate_around_axis(v - center, axis / axis_length, invert ? -angle : angle) + center
end
