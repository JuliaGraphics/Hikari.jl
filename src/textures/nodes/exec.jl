@inline _component(v::Vec3f, i::UInt8) = i == 0x01 ? v[1] : i == 0x02 ? v[2] : i == 0x03 ? v[3] : 0f0

@propagate_inbounds _node_image(arr::AbstractArray{T, 2}, uv::Point2f) where {T <: _TH_TEXEL} =
    _th_spectrum(sample_texture_data(arr, uv))
@propagate_inbounds _node_image(arr, uv::Point2f) = _TH_ZERO

const _RAMP_DATA = Int32(cld(RAMP_TABLE_SIZE, RAMP_ENTRIES_PER_DATA))

"""Run instruction `i` of `prog`; returns how many `DATA` instructions after it it consumed."""
@propagate_inbounds function _exec!(s, prog, i::Int32, ins::NodeInstr, textures, tfc::TextureFilterContext)
    op = ins.op
    if op == NodeOp.CONST
        _store!(s, ins, Vec3f(ins.k[1], ins.k[2], ins.k[3]))
    elseif op == NodeOp.UV
        _store!(s, ins, Vec3f(tfc.uv[1], tfc.uv[2], 0f0))
    elseif op == NodeOp.POSITION
        _store!(s, ins, Vec3f(tfc.p))
    elseif op == NodeOp.NORMAL
        _store!(s, ins, tfc.n)
    elseif op == NodeOp.MATH
        _store!(s, ins, _math(ins.mode[1], _v(s, ins, 1), _v(s, ins, 2), _v(s, ins, 3)))
    elseif op == NodeOp.VECTOR_MATH
        _store!(s, ins, _vector_math(ins.mode[1], _v(s, ins, 1), _v(s, ins, 2), _v(s, ins, 3), _f(s, ins, 4)))
    elseif op == NodeOp.SWIZZLE
        a = _v(s, ins, 1)
        _store!(s, ins, Vec3f(_component(a, ins.mode[1]), _component(a, ins.mode[2]), _component(a, ins.mode[3])))
    elseif op == NodeOp.COMBINE
        _store!(s, ins, Vec3f(_f(s, ins, 1), _f(s, ins, 2), _f(s, ins, 3)))
    elseif op == NodeOp.AFFINE_ROW
        a, k = _v(s, ins, 1), ins.k
        _store!(s, ins, _splat(k[6] * a[1] + k[7] * a[2] + k[8] * a[3] + k[9]))
    elseif op == NodeOp.MIX
        _store!(s, ins, _mix_node(ins, _v(s, ins, 1), _v(s, ins, 2), _v(s, ins, 3)))
    elseif op == NodeOp.CLAMP
        _store!(s, ins, _clamp_node.(ins.mode[1] == 0x01, _v(s, ins, 1), _v(s, ins, 2), _v(s, ins, 3)))
    elseif op == NodeOp.MAP_RANGE
        range, use_clamp = ins.mode[1], ins.mode[2] == 0x01
        scalar = !_bit(ins.outwide, 1)
        _store!(s, ins, _map_range.(range, use_clamp, scalar, _v(s, ins, 1), _v(s, ins, 2), _v(s, ins, 3),
                                    _v(s, ins, 4), _v(s, ins, 5), _v(s, ins, 6)))
    elseif op == NodeOp.RAMP
        _store!(s, ins, _ramp_lookup(prog, i, _f(s, ins, 1), ins.mode[1] == 0x01))
        return _RAMP_DATA
    elseif op == NodeOp.HSV
        _store!(s, ins, _hsv_node(_f(s, ins, 4), _v(s, ins, 5), _f(s, ins, 1), _f(s, ins, 2), _f(s, ins, 3)))
    elseif op == NodeOp.GAMMA
        _store!(s, ins, _gamma(_v(s, ins, 1), _f(s, ins, 2)))
    elseif op == NodeOp.BRIGHT_CONTRAST
        _store!(s, ins, _bright_contrast(_v(s, ins, 1), _f(s, ins, 2), _f(s, ins, 3)))
    elseif op == NodeOp.INVERT
        _store!(s, ins, _invert(_v(s, ins, 2), _f(s, ins, 1)))
    elseif op == NodeOp.RGB_TO_BW
        _store!(s, ins, _splat(_rgb_to_bw(_v(s, ins, 1))))
    elseif op == NodeOp.SEPARATE_COLOR
        _store!(s, ins, _separate_color(ins.mode[1], _v(s, ins, 1)))
    elseif op == NodeOp.COMBINE_COLOR
        _store!(s, ins, _combine_color(ins.mode[1], _v(s, ins, 1)))
    elseif op == NodeOp.MAPPING
        _store!(s, ins, _mapping(ins.mode[1], _v(s, ins, 1), _v(s, ins, 2), _v(s, ins, 3), _v(s, ins, 4)))
    elseif op == NodeOp.VECTOR_ROTATE
        _store!(s, ins, _vector_rotate(ins.mode[1], ins.mode[2] == 0x01, _v(s, ins, 1), _v(s, ins, 2),
                                       _v(s, ins, 3), _f(s, ins, 4), _v(s, ins, 5)))
    elseif op == NodeOp.NOISE
        fac, color = _noise_node(ins, _v(s, ins, 1), _f(s, ins, 2), _f(s, ins, 3), _f(s, ins, 4), _f(s, ins, 5),
                                 _f(s, ins, 6), _f(s, ins, 7), _f(s, ins, 8), _f(s, ins, 9))
        _store!(s, ins, _splat(fac), color)
    elseif op == NodeOp.VORONOI
        distance, color, position, radius = _voronoi_node(ins, _v(s, ins, 1), _f(s, ins, 2), _f(s, ins, 3),
                                                          _f(s, ins, 4), _f(s, ins, 5), _f(s, ins, 6),
                                                          _f(s, ins, 7), _f(s, ins, 8), _f(s, ins, 9))
        _store!(s, ins, _splat(distance), color, Vec3f(position[1], position[2], position[3]),
                _splat(position[4]), _splat(radius))
    elseif op == NodeOp.WHITE_NOISE
        value, color = _white_noise(ins.mode[1], _v(s, ins, 1), _f(s, ins, 2), _wants(ins, 2))
        _store!(s, ins, _splat(value), color)
    elseif op == NodeOp.WAVE
        f = _wave(ins.mode[1], ins.mode[2], ins.mode[3], _v(s, ins, 1) * _f(s, ins, 2), _f(s, ins, 3),
                  _f(s, ins, 4), _f(s, ins, 5), _f(s, ins, 6), _f(s, ins, 7))
        _store!(s, ins, _splat(f), _splat(f))
    elseif op == NodeOp.GRADIENT
        f = _saturate(_gradient(ins.mode[1], _v(s, ins, 1)))
        _store!(s, ins, _splat(f), _splat(f))
    elseif op == NodeOp.MAGIC
        color = _magic(_v(s, ins, 1), _f(s, ins, 2), Int32(ins.mode[1]), _f(s, ins, 3))
        _store!(s, ins, color, _splat((color[1] + color[2] + color[3]) / 3f0))
    elseif op == NodeOp.CHECKER
        f = _checker(_v(s, ins, 1) * _f(s, ins, 4))
        _store!(s, ins, f == 1f0 ? _v(s, ins, 2) : _v(s, ins, 3), _splat(f))
    elseif op == NodeOp.BRICK
        extra = prog[i + Int32(1)].k
        tint, f = _brick(_v(s, ins, 1) * _f(s, ins, 5), _f(s, ins, 6), _f(s, ins, 7), _f(s, ins, 8), _f(s, ins, 9),
                         _f(s, ins, 10), extra[1], Int32(ins.mode[1]), extra[2], Int32(ins.mode[2]))
        color1 = f != 1f0 ? (1f0 - tint) * _v(s, ins, 2) + tint * _v(s, ins, 3) : _v(s, ins, 2)
        _store!(s, ins, color1 * (1f0 - f) + _v(s, ins, 4) * f, _splat(f))
        return Int32(1)
    elseif op == NodeOp.IMAGE
        rgb = Raycore.with_texture(_node_image, textures, reinterpret(Int32, ins.k[9]), reinterpret(Int32, ins.k[10]),
                                   Point2f(_v(s, ins, 1)[1], _v(s, ins, 1)[2]))
        _store!(s, ins, Vec3f(rgb.c[1], rgb.c[2], rgb.c[3]))
    end
    return Int32(0)
end

"""
Run a compiled node program at a hit; the result is what its `RESULT` instruction reads.
Not inlined: every texture read of a material calls it, and one copy of the interpreter per
call site would multiply the kernel's size.
"""
@noinline function eval_nodes(prog::AbstractVector{NodeInstr}, textures, tfc::TextureFilterContext)
    s = MVector{NODE_STACK, Float32}(undef)
    i = Int32(1)
    n = Int32(length(prog))
    @inbounds while i <= n
        ins = prog[i]
        ins.op == NodeOp.RESULT && return _v(s, ins, 1)
        i += Int32(1) + _exec!(s, prog, i, ins, textures, tfc)
    end
    return Vec3f(0)
end

@propagate_inbounds _th_nodes_spec(prog::AbstractVector{NodeInstr}, textures, tfc) =
    RGBSpectrum(eval_nodes(prog, textures, tfc))
@propagate_inbounds _th_nodes_spec(prog, textures, tfc) = _TH_ZERO
@propagate_inbounds _th_nodes_float(prog::AbstractVector{NodeInstr}, textures, tfc) =
    eval_nodes(prog, textures, tfc)[1]
@propagate_inbounds _th_nodes_float(prog, textures, tfc) = 0f0
