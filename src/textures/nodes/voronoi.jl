# Ported from Blender Cycles v5.2.1: intern/cycles/kernel/svm/voronoi.h.
# Copyright 2011-2022 Blender Foundation. SPDX-License-Identifier: Apache-2.0.
# Smooth F1 and distance to edge are based on code copyright (c) 2013 Inigo
# Quilez, SPDX-License-Identifier: MIT. Modified: rewritten in Julia, one
# generic method for 2 to 4 dimensions. See LICENSES/Apache-2.0.txt and
# LICENSES/MIT.txt.

module VoronoiFeature
const F1               = UInt8(1)
const F2               = UInt8(2)
const SMOOTH_F1        = UInt8(3)
const DISTANCE_TO_EDGE = UInt8(4)
const N_SPHERE_RADIUS  = UInt8(5)
end

module VoronoiMetric
const EUCLIDEAN = UInt8(1)
const MANHATTAN = UInt8(2)
const CHEBYCHEV = UInt8(3)
const MINKOWSKI = UInt8(4)
end

struct VoronoiParams
    scale::Float32
    detail::Float32
    roughness::Float32
    lacunarity::Float32
    smoothness::Float32
    exponent::Float32
    randomness::Float32
    max_distance::Float32
    normalize::Bool
    feature::UInt8
    metric::UInt8
end

@inline _with_max_distance(p::VoronoiParams, d::Float32) =
    VoronoiParams(p.scale, p.detail, p.roughness, p.lacunarity, p.smoothness, p.exponent, p.randomness, d,
                  p.normalize, p.feature, p.metric)

@inline function _voronoi_distance(a::Vec{N, Float32}, b::Vec{N, Float32}, params::VoronoiParams) where {N}
    d = abs.(a - b)
    params.metric == VoronoiMetric.EUCLIDEAN && return norm(a - b)
    params.metric == VoronoiMetric.MANHATTAN && return sum(d)
    params.metric == VoronoiMetric.CHEBYCHEV && return maximum(d)
    params.metric == VoronoiMetric.MINKOWSKI && return sum(d .^ params.exponent)^(1f0 / params.exponent)
    return 0f0
end

@inline function _voronoi_distance_bound(a::Vec{N, Float32}, b::Vec{N, Float32}, params::VoronoiParams) where {N}
    d = abs.(a - b)
    params.metric == VoronoiMetric.EUCLIDEAN && return dot(a - b, a - b)
    params.metric == VoronoiMetric.MANHATTAN && return sum(d)
    params.metric == VoronoiMetric.CHEBYCHEV && return maximum(d)
    params.metric == VoronoiMetric.MINKOWSKI && return sum(d .^ params.exponent)
    return 0f0
end

struct VoronoiOutput
    distance::Float32
    color::Vec3f
    position::Vec4f
end

const _NO_VORONOI = VoronoiOutput(0f0, Vec3f(0), Vec4f(0))

@inline _voronoi_position(c::Float32) = Vec4f(0f0, 0f0, 0f0, c)
@inline _voronoi_position(c::Vec2f) = Vec4f(c[1], c[2], 0f0, 0f0)
@inline _voronoi_position(c::Vec3f) = Vec4f(c[1], c[2], c[3], 0f0)
@inline _voronoi_position(c::Vec4f) = c

@inline _cell_int(c::Vec{N, Float32}) where {N} = ntuple(i -> unsafe_trunc(Int32, c[i]), Val(N))
@inline _cell_hash(cell::NTuple{N, Int32}) where {N} = _hash_int_to_float(cell)
@inline _cell_color(cell::NTuple{2, Int32}) = _hash_int_to_float((cell[1], cell[2], Int32(0)))
@inline _cell_color(cell::NTuple{3, Int32}) = _hash_int_to_float(cell)
@inline _cell_color(cell::NTuple{4, Int32}) = (h = _hash_int_to_float(cell); Vec3f(h[1], h[2], h[3]))

"""
A loop the compiler keeps rolled. Scanning 3^4 or 5^4 cells unrolled, for every feature and
dimension, makes the kernel too large for Metal's compiler to finish in reasonable time.
"""
macro rolled(loop)
    loop.args[2] = Expr(:block, loop.args[2], Expr(:loopinfo, (Symbol("llvm.loop.unroll.disable"),)))
    return esc(loop)
end

"""The number of cells within `r` of a cell in `N` dimensions."""
@inline _cell_count(::Val{N}, r::Int32) where {N} = (Int32(2) * r + Int32(1))^N

"""The offset of the `k`-th (from 0) of those cells, the first coordinate varying fastest, as Cycles scans them."""
@inline function _cell_offset(k::Int32, r::Int32, ::Val{N}) where {N}
    w = Int32(2) * r + Int32(1)
    return ntuple(i -> rem(div(k, w^(i - 1)), w) - r, Val(N))
end
@inline _add(a::NTuple{N, Int32}, b::NTuple{N, Int32}) where {N} = ntuple(i -> a[i] + b[i], Val(N))
@inline _point(offset::NTuple{N, Int32}, cell::NTuple{N, Int32}, params) where {N} =
    Vec{N, Float32}(Float32.(offset)) + _cell_hash(_add(cell, offset)) * params.randomness

function _voronoi_f1(params::VoronoiParams, coord::Vec{N, Float32}) where {N}
    cell_f = floor.(coord)
    local_position = coord - cell_f
    cell = _cell_int(cell_f)
    min_distance = floatmax(Float32)
    target_offset = ntuple(_ -> Int32(0), Val(N))
    target_position = Vec{N, Float32}(0)
    @rolled for k in Int32(0):_cell_count(Val(N), Int32(1)) - Int32(1)
        offset = _cell_offset(k, Int32(1), Val(N))
        point = _point(offset, cell, params)
        d = _voronoi_distance_bound(point, local_position, params)
        if d < min_distance
            target_offset, min_distance, target_position = offset, d, point
        end
    end
    return VoronoiOutput(_voronoi_distance(target_position, local_position, params),
                         _cell_color(_add(cell, target_offset)), _voronoi_position(target_position + cell_f))
end

function _voronoi_smooth_f1(params::VoronoiParams, coord::Vec{N, Float32}) where {N}
    cell_f = floor.(coord)
    local_position = coord - cell_f
    cell = _cell_int(cell_f)
    smooth_distance = 0f0
    smooth_color = Vec3f(0)
    smooth_position = Vec{N, Float32}(0)
    h = -1f0
    @rolled for k in Int32(0):_cell_count(Val(N), Int32(2)) - Int32(1)
        offset = _cell_offset(k, Int32(2), Val(N))
        point = _point(offset, cell, params)
        d = _voronoi_distance(point, local_position, params)
        h = h == -1f0 ? 1f0 : _smoothstep01(_saturate(0.5f0 + 0.5f0 * (smooth_distance - d) / params.smoothness))
        correction = params.smoothness * h * (1f0 - h)
        smooth_distance = _mix(smooth_distance, d, h) - correction
        correction /= 1f0 + 3f0 * params.smoothness
        smooth_color = _mix(smooth_color, _cell_color(_add(cell, offset)), h) .- correction
        smooth_position = _mix(smooth_position, point, h) .- correction
    end
    return VoronoiOutput(smooth_distance, smooth_color, _voronoi_position(cell_f + smooth_position))
end

function _voronoi_f2(params::VoronoiParams, coord::Vec{N, Float32}) where {N}
    cell_f = floor.(coord)
    local_position = coord - cell_f
    cell = _cell_int(cell_f)
    d1 = d2 = floatmax(Float32)
    zero_offset = ntuple(_ -> Int32(0), Val(N))
    offset1, offset2 = zero_offset, zero_offset
    position1 = position2 = Vec{N, Float32}(0)
    @rolled for k in Int32(0):_cell_count(Val(N), Int32(1)) - Int32(1)
        offset = _cell_offset(k, Int32(1), Val(N))
        point = _point(offset, cell, params)
        d = _voronoi_distance(point, local_position, params)
        if d < d1
            d2, d1 = d1, d
            offset2, offset1 = offset1, offset
            position2, position1 = position1, point
        elseif d < d2
            d2, offset2, position2 = d, offset, point
        end
    end
    return VoronoiOutput(d2, _cell_color(_add(cell, offset2)), _voronoi_position(position2 + cell_f))
end

function _voronoi_distance_to_edge(params::VoronoiParams, coord::Vec{N, Float32}) where {N}
    cell_f = floor.(coord)
    local_position = coord - cell_f
    cell = _cell_int(cell_f)
    to_closest = Vec{N, Float32}(0)
    min_distance = floatmax(Float32)
    @rolled for k in Int32(0):_cell_count(Val(N), Int32(1)) - Int32(1)
        to_point = _point(_cell_offset(k, Int32(1), Val(N)), cell, params) - local_position
        d = dot(to_point, to_point)
        if d < min_distance
            min_distance, to_closest = d, to_point
        end
    end
    min_distance = floatmax(Float32)
    @rolled for k in Int32(0):_cell_count(Val(N), Int32(1)) - Int32(1)
        to_point = _point(_cell_offset(k, Int32(1), Val(N)), cell, params) - local_position
        perpendicular = to_point - to_closest
        if dot(perpendicular, perpendicular) > 0.0001f0
            min_distance = min(min_distance, dot((to_closest + to_point) / 2f0, normalize(perpendicular)))
        end
    end
    return min_distance
end

function _voronoi_n_sphere_radius(params::VoronoiParams, coord::Vec{N, Float32}) where {N}
    cell_f = floor.(coord)
    local_position = coord - cell_f
    cell = _cell_int(cell_f)
    closest = Vec{N, Float32}(0)
    closest_offset = ntuple(_ -> Int32(0), Val(N))
    min_distance = floatmax(Float32)
    @rolled for k in Int32(0):_cell_count(Val(N), Int32(1)) - Int32(1)
        offset = _cell_offset(k, Int32(1), Val(N))
        point = _point(offset, cell, params)
        d = dot(point - local_position, point - local_position)
        if d < min_distance
            min_distance, closest, closest_offset = d, point, offset
        end
    end
    min_distance = floatmax(Float32)
    closest_to_closest = Vec{N, Float32}(0)
    @rolled for k in Int32(0):_cell_count(Val(N), Int32(1)) - Int32(1)
        neighbour = _cell_offset(k, Int32(1), Val(N))
        all(iszero, neighbour) && continue
        offset = _add(neighbour, closest_offset)
        point = _point(offset, cell, params)
        d = dot(closest - point, closest - point)
        if d < min_distance
            min_distance, closest_to_closest = d, point
        end
    end
    return norm(closest_to_closest - closest) / 2f0
end

# ── 1D: points hashed from float cell coordinates ──────────────────────────

@inline _point1(offset::Float32, cell::Float32, params) = offset + _hash_to_float(cell + offset) * params.randomness

function _voronoi_f1(params::VoronoiParams, coord::Float32)
    cell = floor(coord)
    local_position = coord - cell
    min_distance, target_offset, target_position = floatmax(Float32), 0f0, 0f0
    for i in -1:1
        offset = Float32(i)
        point = _point1(offset, cell, params)
        d = abs(point - local_position)
        if d < min_distance
            target_offset, min_distance, target_position = offset, d, point
        end
    end
    return VoronoiOutput(min_distance, _hash_to_float3(cell + target_offset), _voronoi_position(target_position + cell))
end

function _voronoi_smooth_f1(params::VoronoiParams, coord::Float32)
    cell = floor(coord)
    local_position = coord - cell
    smooth_distance, smooth_position, smooth_color = 0f0, 0f0, Vec3f(0)
    h = -1f0
    for i in -2:2
        offset = Float32(i)
        point = _point1(offset, cell, params)
        d = abs(point - local_position)
        h = h == -1f0 ? 1f0 : _smoothstep01(_saturate(0.5f0 + 0.5f0 * (smooth_distance - d) / params.smoothness))
        correction = params.smoothness * h * (1f0 - h)
        smooth_distance = _mix(smooth_distance, d, h) - correction
        correction /= 1f0 + 3f0 * params.smoothness
        smooth_color = _mix(smooth_color, _hash_to_float3(cell + offset), h) .- correction
        smooth_position = _mix(smooth_position, point, h) - correction
    end
    return VoronoiOutput(smooth_distance, smooth_color, _voronoi_position(cell + smooth_position))
end

function _voronoi_f2(params::VoronoiParams, coord::Float32)
    cell = floor(coord)
    local_position = coord - cell
    d1 = d2 = floatmax(Float32)
    offset1 = offset2 = position1 = position2 = 0f0
    for i in -1:1
        offset = Float32(i)
        point = _point1(offset, cell, params)
        d = abs(point - local_position)
        if d < d1
            d2, d1 = d1, d
            offset2, offset1 = offset1, offset
            position2, position1 = position1, point
        elseif d < d2
            d2, offset2, position2 = d, offset, point
        end
    end
    return VoronoiOutput(d2, _hash_to_float3(cell + offset2), _voronoi_position(position2 + cell))
end

function _voronoi_distance_to_edge(params::VoronoiParams, coord::Float32)
    cell = floor(coord)
    local_position = coord - cell
    mid = _hash_to_float(cell) * params.randomness
    left = -1f0 + _hash_to_float(cell - 1f0) * params.randomness
    right = 1f0 + _hash_to_float(cell + 1f0) * params.randomness
    return min(abs((mid + left) / 2f0 - local_position), abs((mid + right) / 2f0 - local_position))
end

function _voronoi_n_sphere_radius(params::VoronoiParams, coord::Float32)
    cell = floor(coord)
    local_position = coord - cell
    closest, closest_offset, min_distance = 0f0, 0f0, floatmax(Float32)
    for i in -1:1
        offset = Float32(i)
        point = _point1(offset, cell, params)
        d = abs(point - local_position)
        if d < min_distance
            min_distance, closest, closest_offset = d, point, offset
        end
    end
    min_distance, closest_to_closest = floatmax(Float32), 0f0
    for i in -1:1
        i == 0 && continue
        offset = Float32(i) + closest_offset
        point = _point1(offset, cell, params)
        d = abs(closest - point)
        if d < min_distance
            min_distance, closest_to_closest = d, point
        end
    end
    return abs(closest_to_closest - closest) / 2f0
end

# ── Fractal Voronoi ─────────────────────────────────────────────────────────

@inline function _voronoi_octave(params::VoronoiParams, coord)
    params.feature == VoronoiFeature.F2 && return _voronoi_f2(params, coord)
    params.feature == VoronoiFeature.SMOOTH_F1 && params.smoothness != 0f0 && return _voronoi_smooth_f1(params, coord)
    return _voronoi_f1(params, coord)
end

function _fractal_voronoi(params::VoronoiParams, coord)
    amplitude, max_amplitude, scale = 1f0, 0f0, 1f0
    distance, color, position = 0f0, Vec3f(0), Vec4f(0)
    zero_input = params.detail == 0f0 || params.roughness == 0f0
    for i in Int32(0):_to_int(ceil(params.detail))
        octave = _voronoi_octave(params, coord * scale)
        if zero_input
            max_amplitude = 1f0
            distance, color, position = octave.distance, octave.color, octave.position
            break
        end
        if Float32(i) <= params.detail
            max_amplitude += amplitude
            distance += octave.distance * amplitude
            color += octave.color * amplitude
            position = _mix(position, octave.position / scale, amplitude)
            scale *= params.lacunarity
            amplitude *= params.roughness
        else
            remainder = params.detail - floor(params.detail)
            if remainder != 0f0
                max_amplitude = _mix(max_amplitude, max_amplitude + amplitude, remainder)
                distance = _mix(distance, distance + octave.distance * amplitude, remainder)
                color = _mix(color, color + octave.color * amplitude, remainder)
                position = _mix(position, _mix(position, octave.position / scale, amplitude), remainder)
            end
        end
    end
    if params.normalize
        distance /= max_amplitude * params.max_distance
        color /= max_amplitude
    end
    return VoronoiOutput(distance, color, _safe_divide.(position, params.scale))
end

function _fractal_voronoi_distance_to_edge(params::VoronoiParams, coord)
    amplitude, max_amplitude, scale, distance = 1f0, params.max_distance, 1f0, 8f0
    zero_input = params.detail == 0f0 || params.roughness == 0f0
    for i in Int32(0):_to_int(ceil(params.detail))
        octave_distance = _voronoi_distance_to_edge(params, coord * scale)
        if zero_input
            distance = octave_distance
            break
        end
        if Float32(i) <= params.detail
            max_amplitude = _mix(max_amplitude, params.max_distance / scale, amplitude)
            distance = _mix(distance, min(distance, octave_distance / scale), amplitude)
            scale *= params.lacunarity
            amplitude *= params.roughness
        else
            remainder = params.detail - floor(params.detail)
            if remainder != 0f0
                lerp_amplitude = _mix(max_amplitude, params.max_distance / scale, amplitude)
                max_amplitude = _mix(max_amplitude, lerp_amplitude, remainder)
                lerp_distance = _mix(distance, min(distance, octave_distance / scale), amplitude)
                distance = _mix(distance, min(distance, lerp_distance), remainder)
            end
        end
    end
    params.normalize && (distance /= max_amplitude)
    return distance
end

@inline _max_distance(params::VoronoiParams, ::Float32) = 0.5f0 + 0.5f0 * params.randomness
@inline _max_distance(params::VoronoiParams, ::Vec{N, Float32}) where {N} =
    _voronoi_distance(Vec{N, Float32}(0), Vec{N, Float32}(0.5f0 + 0.5f0 * params.randomness), params)

"""Blender's Voronoi Texture: distance, colour, position, w and radius."""
@noinline function _voronoi_node(ins::NodeInstr, coord::Vec3f, w::Float32, scale, detail, roughness, lacunarity,
                       smoothness, exponent, randomness)
    dims, feature, packed = ins.mode
    metric, normalize = packed & 0x0f, packed >> 4 == 0x01
    params = VoronoiParams(scale, clamp(detail, 0f0, 15f0), clamp(roughness, 0f0, 1f0), lacunarity,
                           clamp(smoothness / 2f0, 0f0, 0.5f0), exponent, clamp(randomness, 0f0, 1f0), 0f0,
                           normalize, feature, metric)
    return _with_dims(dims, coord * scale, w * scale) do co
        if feature == VoronoiFeature.DISTANCE_TO_EDGE
            p = _with_max_distance(params, 0.5f0 + 0.5f0 * params.randomness)
            return (_fractal_voronoi_distance_to_edge(p, co), Vec3f(0), Vec4f(0), 0f0)
        elseif feature == VoronoiFeature.N_SPHERE_RADIUS
            return (0f0, Vec3f(0), Vec4f(0), _voronoi_n_sphere_radius(params, co))
        end
        f2 = feature == VoronoiFeature.F2 ? 2f0 : 1f0
        out = _fractal_voronoi(_with_max_distance(params, _max_distance(params, co) * f2), co)
        return (out.distance, out.color, out.position, 0f0)
    end
end
