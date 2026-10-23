# ============================================================================
# Node textures: Blender-style shader graphs over uv, position and normal
# ============================================================================
#
# A graph is built on the host from `TexNode`s (see the `TextureNodes`
# submodule) and compiled at push time into a flat program, a
# `Vector{NodeInstr}`, which lives in the texture store like any image. ONE
# interpreter evaluates every program, so any graph binds to any material
# parameter without a new kernel: the graph is data, not a type.
#
# As in Cycles' SVM, values live on a per-thread stack of floats: a scalar takes
# one slot and a vector or colour three. An instruction names the stack offset
# of each of its inputs, or `NO_SLOT` to read its constant from `k` instead, and
# the offsets its outputs go to.

module NodeOp
const CONST           = UInt8(1)
const UV              = UInt8(2)
const POSITION        = UInt8(3)
const NORMAL          = UInt8(4)
const RESULT          = UInt8(5)
const DATA            = UInt8(6)
const MATH            = UInt8(7)
const VECTOR_MATH     = UInt8(8)
const SWIZZLE         = UInt8(9)
const COMBINE         = UInt8(10)
const AFFINE_ROW      = UInt8(11)
const IMAGE           = UInt8(12)
const MIX             = UInt8(13)
const CLAMP           = UInt8(14)
const MAP_RANGE       = UInt8(15)
const RAMP            = UInt8(16)
const HSV             = UInt8(17)
const GAMMA           = UInt8(18)
const BRIGHT_CONTRAST = UInt8(19)
const INVERT          = UInt8(20)
const RGB_TO_BW       = UInt8(21)
const SEPARATE_COLOR  = UInt8(22)
const COMBINE_COLOR   = UInt8(23)
const MAPPING         = UInt8(24)
const VECTOR_ROTATE   = UInt8(25)
const NOISE           = UInt8(26)
const VORONOI         = UInt8(27)
const WHITE_NOISE     = UInt8(28)
const WAVE            = UInt8(29)
const GRADIENT        = UInt8(30)
const MAGIC           = UInt8(31)
const CHECKER         = UInt8(32)
const BRICK           = UInt8(33)
end

const NODE_STACK = 64
const NODE_ARGS = 10
const NODE_OUTS = 5
const NO_SLOT = 0x00

"""
One instruction of a compiled node program. `arg[i]` is the stack offset input
`i` is read from (`NO_SLOT`: the constant `k[i]`), and bit `i` of `wide` says
whether that offset holds a vector. `out[j]` is where output `j` is stored
(`NO_SLOT`: not needed), bit `j` of `outwide` whether it is a vector. `mode`
holds the node's settings, such as a math operation or a noise type.
"""
struct NodeInstr
    op::UInt8
    mode::NTuple{3, UInt8}
    out::NTuple{NODE_OUTS, UInt8}
    outwide::UInt8
    arg::NTuple{NODE_ARGS, UInt8}
    wide::UInt16
    k::NTuple{NODE_ARGS, Float32}
end

@inline _splat(x::Float32) = Vec3f(x, x, x)
@inline _bit(mask, i) = (mask >> (i - 1)) & one(mask) == one(mask)

"""Input `i` as a scalar: a vector source gives its first component."""
@inline function _f(s, ins::NodeInstr, i)
    o = ins.arg[i]
    return o == NO_SLOT ? ins.k[i] : @inbounds s[o]
end

"""Input `i` as a vector: a scalar source is splatted."""
@inline function _v(s, ins::NodeInstr, i)
    o = ins.arg[i]
    o == NO_SLOT && return _splat(ins.k[i])
    _bit(ins.wide, i) || return _splat(@inbounds s[o])
    return @inbounds Vec3f(s[o], s[o + 0x01], s[o + 0x02])
end

@inline function _put!(s, o::UInt8, wide::Bool, v::Vec3f)
    o == NO_SLOT && return
    @inbounds s[o] = v[1]
    if wide
        @inbounds s[o + 0x01] = v[2]
        @inbounds s[o + 0x02] = v[3]
    end
    return
end

@inline _store!(s, ins::NodeInstr, a::Vec3f) = _put!(s, ins.out[1], _bit(ins.outwide, 1), a)
@inline function _store!(s, ins::NodeInstr, a::Vec3f, b::Vec3f)
    _store!(s, ins, a)
    _put!(s, ins.out[2], _bit(ins.outwide, 2), b)
end
@inline function _store!(s, ins::NodeInstr, a::Vec3f, b::Vec3f, c::Vec3f)
    _store!(s, ins, a, b)
    _put!(s, ins.out[3], _bit(ins.outwide, 3), c)
end
@inline function _store!(s, ins::NodeInstr, a::Vec3f, b::Vec3f, c::Vec3f, d::Vec3f, e::Vec3f)
    _store!(s, ins, a, b, c)
    _put!(s, ins.out[4], _bit(ins.outwide, 4), d)
    _put!(s, ins.out[5], _bit(ins.outwide, 5), e)
end

@inline _wants(ins::NodeInstr, j) = ins.out[j] != NO_SLOT

# A constant picked by a runtime position, by selects: indexing a tuple at a
# runtime position is refused by the SPIR-V emitter.
@inline function _kat(k::NTuple{NODE_ARGS, Float32}, j::Int32)
    v = 0f0
    Base.Cartesian.@nexprs 10 i -> (v = ifelse(j == Int32(i), k[i], v))
    return v
end
