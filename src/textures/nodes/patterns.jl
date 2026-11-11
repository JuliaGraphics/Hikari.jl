# Ported from Blender Cycles v5.2.1: intern/cycles/kernel/svm/wave.h,
# gradient.h, magic.h, checker.h, brick.h and white_noise.h.
# Copyright 2011-2022 Blender Foundation. SPDX-License-Identifier: Apache-2.0.
# Modified: rewritten in Julia. See LICENSES/Apache-2.0.txt.

module WaveType
const BANDS = UInt8(1)
const RINGS = UInt8(2)
end

module WaveDirection
const X         = UInt8(1)
const Y         = UInt8(2)
const Z         = UInt8(3)
const DIAGONAL  = UInt8(4)
const SPHERICAL = UInt8(5)
end

module WaveProfile
const SINE     = UInt8(1)
const SAW      = UInt8(2)
const TRIANGLE = UInt8(3)
end

@noinline function _wave(type::UInt8, direction::UInt8, profile::UInt8, p::Vec3f, distortion::Float32, detail::Float32,
               detail_scale::Float32, detail_roughness::Float32, phase::Float32)
    p = (p .+ 0.000001f0) * 0.999999f0
    n = if type == WaveType.BANDS
        direction == WaveDirection.X ? p[1] * 20f0 :
        direction == WaveDirection.Y ? p[2] * 20f0 :
        direction == WaveDirection.Z ? p[3] * 20f0 : (p[1] + p[2] + p[3]) * 10f0
    else
        rp = direction == WaveDirection.X ? p .* Vec3f(0, 1, 1) :
             direction == WaveDirection.Y ? p .* Vec3f(1, 0, 1) :
             direction == WaveDirection.Z ? p .* Vec3f(1, 1, 0) : p
        norm(rp) * 20f0
    end
    n += phase
    if distortion != 0f0
        n += distortion * (_noise_fbm(p * detail_scale, detail, detail_roughness, 2f0, true) * 2f0 - 1f0)
    end
    profile == WaveProfile.SINE && return 0.5f0 + 0.5f0 * sin(n - Float32(π / 2))
    n /= Float32(2π)
    profile == WaveProfile.SAW && return n - floor(n)
    return abs(n - floor(n + 0.5f0)) * 2f0
end

module GradientType
const LINEAR           = UInt8(1)
const QUADRATIC        = UInt8(2)
const EASING           = UInt8(3)
const DIAGONAL         = UInt8(4)
const RADIAL           = UInt8(5)
const QUADRATIC_SPHERE = UInt8(6)
const SPHERICAL        = UInt8(7)
end

function _gradient(type::UInt8, p::Vec3f)
    x, y, z = p
    G = GradientType
    type == G.LINEAR && return x
    type == G.QUADRATIC && return (r = max(x, 0f0); r * r)
    if type == G.EASING
        r = min(max(x, 0f0), 1f0)
        t = r * r
        return 3f0 * t - 2f0 * t * r
    end
    type == G.DIAGONAL && return (x + y) * 0.5f0
    type == G.RADIAL && return atan(y, x) / Float32(2π) + 0.5f0
    r = max(0.999999f0 - sqrt(x * x + y * y + z * z), 0f0)
    type == G.QUADRATIC_SPHERE && return r * r
    type == G.SPHERICAL && return r
    return 0f0
end

@noinline function _magic(p::Vec3f, scale::Float32, n::Int32, distortion::Float32)
    px, py, pz = rem(p[1] * scale, Float32(2π)), rem(p[2] * scale, Float32(2π)), rem(p[3] * scale, Float32(2π))
    x = sin((px + py + pz) * 5f0)
    y = cos((-px + py - pz) * 5f0)
    z = -cos((-px - py + pz) * 5f0)
    if n > 0
        x *= distortion; y *= distortion; z *= distortion
        y = -cos(x - y + z) * distortion
        if n > 1
            x = cos(x - y - z) * distortion
            if n > 2
                z = sin(-x - y - z) * distortion
                if n > 3
                    x = -cos(-x + y - z) * distortion
                    if n > 4
                        y = -sin(-x + y + z) * distortion
                        if n > 5
                            y = -cos(-x + y + z) * distortion
                            if n > 6
                                x = cos(x + y + z) * distortion
                                if n > 7
                                    z = sin(x + y - z) * distortion
                                    if n > 8
                                        x = -cos(-x - y + z) * distortion
                                        n > 9 && (y = -sin(x - y + z) * distortion)
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    if distortion != 0f0
        distortion *= 2f0
        x /= distortion; y /= distortion; z /= distortion
    end
    return Vec3f(0.5f0 - x, 0.5f0 - y, 0.5f0 - z)
end

function _checker(p::Vec3f)
    p = (p .+ 0.000001f0) * 0.999999f0
    xi, yi, zi = abs.(unsafe_trunc.(Int32, floor.(p)))
    return (rem(xi, Int32(2)) == rem(yi, Int32(2))) == (rem(zi, Int32(2)) == Int32(1)) ? 1f0 : 0f0
end

@inline function _brick_noise(n::UInt32)
    n = (n + UInt32(1013)) & 0x7fffffff
    n = (n >> 13) ⊻ n
    nn = (n * (n * n * UInt32(60493) + UInt32(19990303)) + UInt32(1376312589)) & 0x7fffffff
    return 0.5f0 * (Float32(nn) / 1073741824f0)
end

@inline function _smoothstepf(f::Float32)
    f <= 0f0 && return 0f0
    f >= 1f0 && return 1f0
    ff = f * f
    return 3f0 * ff - 2f0 * ff * f
end

@noinline function _brick(p::Vec3f, mortar_size::Float32, mortar_smooth::Float32, bias::Float32, brick_width::Float32,
                row_height::Float32, offset_amount::Float32, offset_frequency::Int32, squash_amount::Float32,
                squash_frequency::Int32)
    rownum = unsafe_trunc(Int32, floor(p[2] / row_height))
    offset = 0f0
    if offset_frequency != 0 && squash_frequency != 0
        brick_width *= rem(rownum, squash_frequency) != 0 ? 1f0 : squash_amount
        offset = rem(rownum, offset_frequency) != 0 ? 0f0 : brick_width * offset_amount
    end
    bricknum = unsafe_trunc(Int32, floor((p[1] + offset) / brick_width))
    x = (p[1] + offset) - brick_width * Float32(bricknum)
    y = p[2] - row_height * Float32(rownum)
    tint = _saturate(_brick_noise(reinterpret(UInt32, (rownum << 16) + (bricknum & Int32(0xFFFF)))) + bias)
    min_dist = min(min(x, y), min(brick_width - x, row_height - y))
    mortar = if min_dist >= mortar_size
        0f0
    elseif mortar_smooth == 0f0
        1f0
    else
        _smoothstepf((1f0 - min_dist / mortar_size) / mortar_smooth)
    end
    return tint, mortar
end

@noinline function _white_noise(dims::UInt8, v::Vec3f, w::Float32, want_color::Bool)
    return _with_dims(dims, v, w) do co
        (_hash_to_float(co), want_color ? _hash_to_float3(co) : Vec3f(0))
    end
end
