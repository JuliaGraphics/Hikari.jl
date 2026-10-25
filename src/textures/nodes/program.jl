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
const ARGS            = UInt8(34)
const CLOSURE         = UInt8(35)
const PRINCIPLED      = UInt8(36)
const BUMP            = UInt8(37)
const POSITION_DX     = UInt8(38)
const POSITION_DY     = UInt8(39)
const UV_DX           = UInt8(40)
const UV_DY           = UInt8(41)
const NORMAL_MAP      = UInt8(42)
const ATTRIBUTE       = UInt8(43)
const ATTRIBUTE_DX    = UInt8(44)
const ATTRIBUTE_DY    = UInt8(45)
const CAMERA          = UInt8(46)
const WINDOW          = UInt8(47)
const REFLECTION      = UInt8(48)
end

"""
The node families a program uses, as bits of the type parameter of its instructions. The
interpreter for a program is compiled with only those families, as Cycles compiles its kernels
for the nodes a scene uses: one function holding every node takes the registers of the heaviest
one, which slows down and enlarges every program.
"""
module NodeFeature
const BLEND          = UInt32(1) << 0
const COLOR          = UInt32(1) << 1
const RAMP           = UInt32(1) << 2
const VECTOR         = UInt32(1) << 3
const IMAGE          = UInt32(1) << 4
const NOISE          = UInt32(1) << 5
const NOISE_TYPES    = UInt32(1) << 6
const NOISE_DIMS     = UInt32(1) << 7
const VORONOI        = UInt32(1) << 8
const VORONOI_SMOOTH = UInt32(1) << 9
const VORONOI_EDGE   = UInt32(1) << 10
const VORONOI_DIMS   = UInt32(1) << 11
const WHITE_NOISE    = UInt32(1) << 12
const WAVE           = UInt32(1) << 13
const GRADIENT       = UInt32(1) << 14
const MAGIC          = UInt32(1) << 15
const CHECKER        = UInt32(1) << 16
const BRICK          = UInt32(1) << 17
const VECTOR_MATH    = UInt32(1) << 18
const RANGE          = UInt32(1) << 19
const MIX            = UInt32(1) << 20
const MATH_TRIG      = UInt32(1) << 21
const MATH_POWER     = UInt32(1) << 22
const MATH_ROUND     = UInt32(1) << 23
const MATH_COMPARE   = UInt32(1) << 24
const ATTRIBUTE      = UInt32(1) << 25
const BSDF           = UInt32(1) << 26
const PRINCIPLED     = UInt32(1) << 27
const BUMP           = UInt32(1) << 28
end

"""
The most stack floats a program may use. Each program runs with the smallest of
`NODE_STACK_SIZES` that holds what it uses, stored in bits `STACK_SHIFT` and up of `F`: the
stack takes thread memory, so a larger one than needed slows the heavier nodes down.
"""
const NODE_STACK = 128
const NODE_STACK_SIZES = (16, 32, 64, 128)
const STACK_SHIFT = 30

@inline _stack_size(F::UInt32) = 16 << Int((F >> STACK_SHIFT) & 0x3)
_stack_bits(used::Int) = UInt32(findfirst(>=(used), NODE_STACK_SIZES) - 1) << STACK_SHIFT
const NODE_ARGS = 10
const NODE_OUTS = 5
const NO_SLOT = 0x00

"""
One instruction of a compiled node program. `arg[i]` is the stack offset input
`i` is read from (`NO_SLOT`: the constant `k[i]`), and bit `i` of `wide` says
whether that offset holds a vector. `out[j]` is where output `j` is stored
(`NO_SLOT`: not needed), bit `j` of `outwide` whether it is a vector. `mode`
holds the node's settings, such as a math operation or a noise type. `F` is the
program's `NodeFeature` bits.
"""
struct NodeInstr{F}
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

@inline _new_stack(::Val{N}) where {N} = MVector{N, Float32}(undef)

@inline _get(s::MVector, o) = @inbounds s[o]
@inline _set(s::MVector, o, x::Float32) = (@inbounds s[o] = x; s)

"""Input `i` as a scalar: a vector source gives its first component."""
@inline function _f(s, ins::NodeInstr, i)
    o = ins.arg[i]
    return o == NO_SLOT ? ins.k[i] : _get(s, o)
end

"""Input `i` as a vector: a scalar source is splatted."""
@inline function _v(s, ins::NodeInstr, i)
    o = ins.arg[i]
    o == NO_SLOT && return _splat(ins.k[i])
    _bit(ins.wide, i) || return _splat(_get(s, o))
    return Vec3f(_get(s, o), _get(s, o + 0x01), _get(s, o + 0x02))
end

"""The stack with `v` stored at `o`, three floats of it if `wide`."""
@inline function _put(s, o::UInt8, wide::Bool, v::Vec3f)
    o == NO_SLOT && return s
    s = _set(s, o, v[1])
    wide || return s
    return _set(_set(s, o + 0x01, v[2]), o + 0x02, v[3])
end

@inline _store(s, ins::NodeInstr, a::Vec3f) = _put(s, ins.out[1], _bit(ins.outwide, 1), a)
@inline _store(s, ins::NodeInstr, a::Vec3f, b::Vec3f) = _put(_store(s, ins, a), ins.out[2], _bit(ins.outwide, 2), b)
@inline _store(s, ins::NodeInstr, a::Vec3f, b::Vec3f, c::Vec3f) =
    _put(_store(s, ins, a, b), ins.out[3], _bit(ins.outwide, 3), c)
@inline function _store(s, ins::NodeInstr, a::Vec3f, b::Vec3f, c::Vec3f, d::Vec3f, e::Vec3f)
    s = _store(s, ins, a, b, c)
    s = _put(s, ins.out[4], _bit(ins.outwide, 4), d)
    return _put(s, ins.out[5], _bit(ins.outwide, 5), e)
end

@inline _wants(ins::NodeInstr, j) = ins.out[j] != NO_SLOT
@inline _has(::NodeInstr{F}, feature::UInt32) where {F} = F & feature != zero(UInt32)

# A constant picked by a runtime position, by selects: indexing a tuple at a
# runtime position is refused by the SPIR-V emitter.
@inline function _kat(k::NTuple{NODE_ARGS, Float32}, j::Int32)
    v = 0f0
    Base.Cartesian.@nexprs 10 i -> (v = ifelse(j == Int32(i), k[i], v))
    return v
end
