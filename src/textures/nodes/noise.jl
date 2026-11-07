# Ported from Blender Cycles v5.2.1: intern/cycles/kernel/svm/fractal_noise.h
# and noisetex.h. Copyright 2011-2022 Blender Foundation.
# SPDX-License-Identifier: Apache-2.0. Modified: rewritten in Julia, one
# generic method for every dimension. See LICENSES/Apache-2.0.txt.

module NoiseType
const MULTIFRACTAL        = UInt8(1)
const FBM                 = UInt8(2)
const HYBRID_MULTIFRACTAL = UInt8(3)
const RIDGED_MULTIFRACTAL = UInt8(4)
const HETERO_TERRAIN      = UInt8(5)
end

@inline _octaves(detail::Float32) = _to_int(detail)

@inline function _noise_fbm(p, detail::Float32, roughness::Float32, lacunarity::Float32, normalize::Bool)
    fscale, amp, maxamp, sum = 1f0, 1f0, 0f0, 0f0
    for _ in Int32(0):_octaves(detail)
        sum += _snoise(fscale * p) * amp
        maxamp += amp
        amp *= roughness
        fscale *= lacunarity
    end
    rmd = detail - floor(detail)
    if rmd != 0f0
        sum2 = sum + _snoise(fscale * p) * amp
        return normalize ? _mix(0.5f0 * sum / maxamp + 0.5f0, 0.5f0 * sum2 / (maxamp + amp) + 0.5f0, rmd) :
               _mix(sum, sum2, rmd)
    end
    return normalize ? 0.5f0 * sum / maxamp + 0.5f0 : sum
end

@inline function _noise_multi_fractal(p, detail::Float32, roughness::Float32, lacunarity::Float32)
    value, pwr = 1f0, 1f0
    for _ in Int32(0):_octaves(detail)
        value *= pwr * _snoise(p) + 1f0
        pwr *= roughness
        p *= lacunarity
    end
    rmd = detail - floor(detail)
    rmd != 0f0 && (value *= rmd * pwr * _snoise(p) + 1f0)
    return value
end

@inline function _noise_hetero_terrain(p, detail::Float32, roughness::Float32, lacunarity::Float32, offset::Float32)
    pwr = roughness
    value = offset + _snoise(p)
    p *= lacunarity
    for _ in Int32(1):_octaves(detail)
        value += (_snoise(p) + offset) * pwr * value
        pwr *= roughness
        p *= lacunarity
    end
    rmd = detail - floor(detail)
    rmd != 0f0 && (value += rmd * ((_snoise(p) + offset) * pwr * value))
    return value
end

@inline function _noise_hybrid_multi_fractal(p, detail::Float32, roughness::Float32, lacunarity::Float32,
                                             offset::Float32, gain::Float32)
    pwr, value, weight = 1f0, 0f0, 1f0
    i = Int32(0)
    while weight > 0.001f0 && i <= _octaves(detail)
        weight = min(weight, 1f0)
        signal = (_snoise(p) + offset) * pwr
        pwr *= roughness
        value += weight * signal
        weight *= gain * signal
        p *= lacunarity
        i += Int32(1)
    end
    rmd = detail - floor(detail)
    if rmd != 0f0 && weight > 0.001f0
        weight = min(weight, 1f0)
        value += rmd * weight * ((_snoise(p) + offset) * pwr)
    end
    return value
end

@inline function _noise_ridged_multi_fractal(p, detail::Float32, roughness::Float32, lacunarity::Float32,
                                             offset::Float32, gain::Float32)
    pwr = roughness
    signal = offset - abs(_snoise(p))
    signal *= signal
    value = signal
    for _ in Int32(1):_octaves(detail)
        p *= lacunarity
        weight = _saturate(signal * gain)
        signal = offset - abs(_snoise(p))
        signal *= signal
        signal *= weight
        value += signal * pwr
        pwr *= roughness
    end
    return value
end

@inline function _noise_select(p, type::UInt8, normalize::Bool, detail, roughness, lacunarity, offset, gain,
                               all_types::Bool = true)
    all_types || return _noise_fbm(p, detail, roughness, lacunarity, normalize)
    type == NoiseType.MULTIFRACTAL        && return _noise_multi_fractal(p, detail, roughness, lacunarity)
    type == NoiseType.HYBRID_MULTIFRACTAL && return _noise_hybrid_multi_fractal(p, detail, roughness, lacunarity, offset, gain)
    type == NoiseType.RIDGED_MULTIFRACTAL && return _noise_ridged_multi_fractal(p, detail, roughness, lacunarity, offset, gain)
    type == NoiseType.HETERO_TERRAIN      && return _noise_hetero_terrain(p, detail, roughness, lacunarity, offset)
    return _noise_fbm(p, detail, roughness, lacunarity, normalize)
end

"""Seed offsets for distortion and the colour channels, every component in [100, 200]."""
@inline _random_offset(::Float32, seed::Float32) = 100f0 + _hash_to_float(seed) * 100f0
@inline _random_offset(::Vec{N, Float32}, seed::Float32) where {N} =
    Vec{N, Float32}(ntuple(i -> 100f0 + _hash_to_float(Vec2f(seed, Float32(i - 1))) * 100f0, Val(N)))

@inline _ndims(::Float32) = 1
@inline _ndims(::Vec{N, Float32}) where {N} = N

@inline function _distorted(p, distortion::Float32)
    distortion == 0f0 && return p
    return p + _distortion_offset(p) * distortion
end
@inline _distortion_offset(p::Float32) = _snoise(p + _random_offset(p, 0f0))
@inline _distortion_offset(p::Vec{N, Float32}) where {N} =
    Vec{N, Float32}(ntuple(i -> _snoise(p + _random_offset(p, Float32(i - 1))), Val(N)))

"""Blender's Noise Texture at `co` (already scaled): its `fac` and, if `want_color`, its colour."""
@inline function _noise_texture(co, type::UInt8, normalize::Bool, want_color::Bool,
                                detail, roughness, lacunarity, offset, gain, distortion, all_types::Bool)
    p = _distorted(co, distortion)
    value = _noise_select(p, type, normalize, detail, roughness, lacunarity, offset, gain, all_types)
    want_color || return (value, Vec3f(value, 0f0, 0f0))
    n = Float32(_ndims(co))
    c2 = _noise_select(p + _random_offset(p, n), type, normalize, detail, roughness, lacunarity, offset, gain, all_types)
    c3 = _noise_select(p + _random_offset(p, n + 1f0), type, normalize, detail, roughness, lacunarity, offset, gain, all_types)
    return (value, Vec3f(value, c2, c3))
end

"""A point of `dims` dimensions from a 3D vector and a fourth coordinate `w`; always 3D unless `all_dims`."""
@inline function _with_dims(f, dims::UInt8, v::Vec3f, w::Float32, all_dims::Bool = true)
    all_dims || return f(v)
    dims == 0x01 && return f(w)
    dims == 0x02 && return f(Vec2f(v[1], v[2]))
    dims == 0x04 && return f(Vec4f(v[1], v[2], v[3], w))
    return f(v)
end

function _noise_node(ins::NodeInstr, v::Vec3f, w::Float32, scale::Float32, detail::Float32, roughness::Float32,
                     lacunarity::Float32, offset::Float32, gain::Float32, distortion::Float32)
    dims, type, normalize = ins.mode
    detail = clamp(detail, 0f0, 15f0)
    roughness = max(roughness, 0f0)
    return _with_dims(dims, v * scale, w * scale, _has(ins, NodeFeature.NOISE_DIMS)) do co
        _noise_texture(co, type, normalize == 0x01, _wants(ins, 2), detail, roughness, lacunarity, offset, gain, distortion,
                       _has(ins, NodeFeature.NOISE_TYPES))
    end
end
