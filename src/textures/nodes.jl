# ============================================================================
# Node textures: Blender-style shader graphs over uv, position and normal
# ============================================================================
#
# A graph is built on the host from `TexNode`s (see the `TextureNodes`
# submodule) and compiled at push time into a flat register program, a
# `Vector{NodeInstr}`, which lives in the texture store like any image. ONE
# interpreter evaluates every program, so any graph binds to any material
# parameter without a new kernel: the graph is data, not a type.
#
# Every value is a `Vec3f`. A scalar is stored in all three components, so it
# can be used wherever a vector or colour is expected, as in Blender.

module NodeOp
const CONST       = UInt8(1)
const UV          = UInt8(2)
const POSITION    = UInt8(3)
const NORMAL      = UInt8(4)
const ADD         = UInt8(5)
const SUB         = UInt8(6)
const MUL         = UInt8(7)
const DIV         = UInt8(8)
const POW         = UInt8(9)
const MIN         = UInt8(10)
const MAX         = UInt8(11)
const MOD         = UInt8(12)
const ABS         = UInt8(13)
const FLOOR       = UInt8(14)
const FRACT       = UInt8(15)
const SIN         = UInt8(16)
const COS         = UInt8(17)
const SQRT        = UInt8(18)
const EXP         = UInt8(19)
const ATAN2       = UInt8(20)
const GREATER     = UInt8(21)
const MIX         = UInt8(22)
const CLAMP       = UInt8(23)
const SMOOTHSTEP  = UInt8(24)
const DOT         = UInt8(25)
const LENGTH      = UInt8(26)
const CROSS       = UInt8(27)
const NORMALIZE   = UInt8(28)
const SWIZZLE     = UInt8(29)
const COMBINE     = UInt8(30)
const AFFINE_ROW  = UInt8(31)
const NOISE       = UInt8(32)
const VORONOI     = UInt8(33)
const WHITE_NOISE = UInt8(34)
const IMAGE       = UInt8(35)
end

"""
One instruction of a compiled node program: `dst = op(a, b, c; k)`, with `a`,
`b`, `c` and `dst` register numbers and `k` the node's constant parameters.
"""
struct NodeInstr
    op::UInt8
    dst::UInt8
    a::UInt8
    b::UInt8
    c::UInt8
    k::NTuple{4, Float32}
end

const NODE_REGISTERS = 16
const _NodeRegisters = NTuple{NODE_REGISTERS, Vec3f}

@inline _sel(c::Bool, a::Vec3f, b::Vec3f) = Vec3f(ifelse(c, a[1], b[1]), ifelse(c, a[2], b[2]), ifelse(c, a[3], b[3]))
@inline _splat(x::Float32) = Vec3f(x, x, x)

# Registers are read and written by selects over literal indices: indexing a
# tuple at a runtime position is refused by the SPIR-V emitter.
@inline function _reg(r::_NodeRegisters, i::UInt8)
    v = Vec3f(0)
    Base.Cartesian.@nexprs 16 k -> (v = _sel(i == UInt8(k), r[k], v))
    return v
end
@inline _setreg(r::_NodeRegisters, i::UInt8, v::Vec3f) =
    ntuple(k -> _sel(i == UInt8(k), v, r[k]), Val(NODE_REGISTERS))

# ── Noise ───────────────────────────────────────────────────────────────────

@inline _floor_u32(x::Float32) = reinterpret(UInt32, unsafe_trunc(Int32, floor(x)))

@inline function _hash3(x::UInt32, y::UInt32, z::UInt32)
    x = x * 0x0019660d + 0x3c6ef35f
    y = y * 0x0019660d + 0x3c6ef35f
    z = z * 0x0019660d + 0x3c6ef35f
    x += y * z; y += z * x; z += x * y
    x ⊻= x >> 16; y ⊻= y >> 16; z ⊻= z >> 16
    x += y * z; y += z * x; z += x * y
    return x, y, z
end

@inline _unit(h::UInt32) = Float32(h >> 8) * (1f0 / 16777216f0)
@inline _hash01(x::UInt32, y::UInt32, z::UInt32) = (h = _hash3(x, y, z); Vec3f(_unit(h[1]), _unit(h[2]), _unit(h[3])))

@inline function _grad(h::UInt32, x::Float32, y::Float32, z::Float32)
    h &= 0x0000000f
    u = h < 0x00000008 ? x : y
    v = h < 0x00000004 ? y : (h == 0x0000000c || h == 0x0000000e ? x : z)
    return (iszero(h & 0x00000001) ? u : -u) + (iszero(h & 0x00000002) ? v : -v)
end

@inline _fade(t::Float32) = t * t * t * (t * (t * 6f0 - 15f0) + 10f0)
@inline _lerp(a, b, t) = a + (b - a) * t

"""Signed 3D gradient noise, roughly in [-1, 1]."""
@inline function _perlin(p::Vec3f)
    fx, fy, fz = floor(p[1]), floor(p[2]), floor(p[3])
    X, Y, Z = _floor_u32(p[1]), _floor_u32(p[2]), _floor_u32(p[3])
    x, y, z = p[1] - fx, p[2] - fy, p[3] - fz
    u, v, w = _fade(x), _fade(y), _fade(z)
    g(i, j, k) = _grad(_hash3(X + i, Y + j, Z + k)[1], x - Float32(i), y - Float32(j), z - Float32(k))
    return _lerp(_lerp(_lerp(g(0x0, 0x0, 0x0), g(0x1, 0x0, 0x0), u),
                       _lerp(g(0x0, 0x1, 0x0), g(0x1, 0x1, 0x0), u), v),
                 _lerp(_lerp(g(0x0, 0x0, 0x1), g(0x1, 0x0, 0x1), u),
                       _lerp(g(0x0, 0x1, 0x1), g(0x1, 0x1, 0x1), u), v), w)
end

"""Fractal noise in [0, 1] around 0.5: `detail` octaves (a fractional last one blends in), each `lacunarity` times finer and `roughness` times weaker."""
@inline function _fbm(p::Vec3f, detail::Float32, roughness::Float32, lacunarity::Float32)
    octaves = clamp(detail, 0f0, 15f0)
    total, amplitude, norm, scale = 0f0, 1f0, 0f0, 1f0
    for _ in Int32(0):unsafe_trunc(Int32, octaves)
        total += amplitude * _perlin(p * scale)
        norm += amplitude
        amplitude *= roughness
        scale *= lacunarity
    end
    whole = 0.5f0 + 0.5f0 * total / norm
    rest = octaves - floor(octaves)
    rest == 0f0 && return whole
    finer = 0.5f0 + 0.5f0 * (total + amplitude * _perlin(p * scale)) / (norm + amplitude)
    return _lerp(whole, finer, rest)
end

@inline function _noise(p::Vec3f, k::NTuple{4, Float32})
    n1 = _fbm(p, k[1], k[2], k[3])
    k[4] == 0f0 && return _splat(n1)
    return Vec3f(n1, _fbm(p + Vec3f(31.4f0, 47.2f0, 12.9f0), k[1], k[2], k[3]),
                     _fbm(p + Vec3f(93.7f0, 11.3f0, 71.1f0), k[1], k[2], k[3]))
end

"""3D Voronoi F1. `k = (randomness, output)`: output 0 is the distance to the nearest feature point, 1 its cell's random colour, 2 its position."""
@inline function _voronoi(p::Vec3f, k::NTuple{4, Float32})
    cx, cy, cz = floor(p[1]), floor(p[2]), floor(p[3])
    X, Y, Z = _floor_u32(p[1]), _floor_u32(p[2]), _floor_u32(p[3])
    best = Inf32
    cell = (X, Y, Z)
    feature = Vec3f(0)
    for dz in Int32(-1):Int32(1), dy in Int32(-1):Int32(1), dx in Int32(-1):Int32(1)
        c = (X + reinterpret(UInt32, dx), Y + reinterpret(UInt32, dy), Z + reinterpret(UInt32, dz))
        q = Vec3f(cx + Float32(dx), cy + Float32(dy), cz + Float32(dz)) + k[1] * _hash01(c...)
        d = norm(q - p)
        if d < best
            best, cell, feature = d, c, q
        end
    end
    k[2] == 1f0 && return _hash01(cell[1] ⊻ 0x2545f491, cell[2], cell[3])
    k[2] == 2f0 && return feature
    return _splat(best)
end

@inline _white_noise(p::Vec3f) =
    _hash01(reinterpret(UInt32, p[1]), reinterpret(UInt32, p[2]), reinterpret(UInt32, p[3]) ⊻ 0x68e31da4)

# ── Interpreter ─────────────────────────────────────────────────────────────

@inline _component(v::Vec3f, i::Float32) = i == 1f0 ? v[1] : i == 2f0 ? v[2] : i == 3f0 ? v[3] : 0f0
@inline _safe_div(a::Float32, b::Float32) = b == 0f0 ? 0f0 : a / b
@inline _safe_pow(a::Float32, b::Float32) = a < 0f0 && b != floor(b) ? 0f0 : a^b
@inline _floored_mod(a::Float32, b::Float32) = b == 0f0 ? 0f0 : a - b * floor(a / b)
@inline _smoothstep(e0::Float32, e1::Float32, x::Float32) =
    (t = clamp(_safe_div(x - e0, e1 - e0), 0f0, 1f0); t * t * (3f0 - 2f0 * t))

@propagate_inbounds _node_image(arr::AbstractArray{T, 2}, uv::Point2f) where {T <: _TH_TEXEL} =
    _th_spectrum(sample_texture_data(arr, uv))
@propagate_inbounds _node_image(arr, uv::Point2f) = _TH_ZERO

@propagate_inbounds function _node_op(ins::NodeInstr, a::Vec3f, b::Vec3f, c::Vec3f, textures, tfc)
    op, k = ins.op, ins.k
    op == NodeOp.CONST       && return Vec3f(k[1], k[2], k[3])
    op == NodeOp.UV          && return Vec3f(tfc.uv[1], tfc.uv[2], 0f0)
    op == NodeOp.POSITION    && return Vec3f(tfc.p)
    op == NodeOp.NORMAL      && return tfc.n
    op == NodeOp.ADD         && return a + b
    op == NodeOp.SUB         && return a - b
    op == NodeOp.MUL         && return a .* b
    op == NodeOp.DIV         && return _safe_div.(a, b)
    op == NodeOp.POW         && return _safe_pow.(a, b)
    op == NodeOp.MIN         && return min.(a, b)
    op == NodeOp.MAX         && return max.(a, b)
    op == NodeOp.MOD         && return _floored_mod.(a, b)
    op == NodeOp.ABS         && return abs.(a)
    op == NodeOp.FLOOR       && return floor.(a)
    op == NodeOp.FRACT       && return a - floor.(a)
    op == NodeOp.SIN         && return sin.(a)
    op == NodeOp.COS         && return cos.(a)
    op == NodeOp.SQRT        && return sqrt.(max.(a, 0f0))
    op == NodeOp.EXP         && return exp.(a)
    op == NodeOp.ATAN2       && return atan.(a, b)
    op == NodeOp.GREATER     && return Vec3f(ifelse.(a .> b, 1f0, 0f0))
    op == NodeOp.MIX         && return a + (b - a) .* c
    op == NodeOp.CLAMP       && return clamp.(a, b, c)
    op == NodeOp.SMOOTHSTEP  && return _smoothstep.(a, b, c)
    op == NodeOp.DOT         && return _splat(dot(a, b))
    op == NodeOp.LENGTH      && return _splat(norm(a))
    op == NodeOp.CROSS       && return cross(a, b)
    op == NodeOp.NORMALIZE   && return (l = norm(a); l == 0f0 ? a : a / l)
    op == NodeOp.SWIZZLE     && return Vec3f(_component(a, k[1]), _component(a, k[2]), _component(a, k[3]))
    op == NodeOp.COMBINE     && return Vec3f(a[1], b[1], c[1])
    op == NodeOp.AFFINE_ROW  && return _splat(k[1] * a[1] + k[2] * a[2] + k[3] * a[3] + k[4])
    op == NodeOp.NOISE       && return _noise(a, k)
    op == NodeOp.VORONOI     && return _voronoi(a, k)
    op == NodeOp.WHITE_NOISE && return _white_noise(a)
    if op == NodeOp.IMAGE
        rgb = Raycore.with_texture(_node_image, textures, reinterpret(Int32, k[1]), reinterpret(Int32, k[2]),
                                   Point2f(a[1], a[2]))
        return Vec3f(rgb.c[1], rgb.c[2], rgb.c[3])
    end
    return Vec3f(0)
end

"""Run a compiled node program at a hit; the result is the last instruction's value."""
@propagate_inbounds function eval_nodes(prog::AbstractVector{NodeInstr}, textures, tfc::TextureFilterContext)
    regs = ntuple(_ -> Vec3f(0), Val(NODE_REGISTERS))
    out = Vec3f(0)
    for i in Int32(1):Int32(length(prog))
        ins = @inbounds prog[i]
        out = _node_op(ins, _reg(regs, ins.a), _reg(regs, ins.b), _reg(regs, ins.c), textures, tfc)
        regs = _setreg(regs, ins.dst, out)
    end
    return out
end

@propagate_inbounds _th_nodes_spec(prog::AbstractVector{NodeInstr}, textures, tfc) =
    RGBSpectrum(eval_nodes(prog, textures, tfc))
@propagate_inbounds _th_nodes_spec(prog, textures, tfc) = _TH_ZERO
@propagate_inbounds _th_nodes_float(prog::AbstractVector{NodeInstr}, textures, tfc) =
    eval_nodes(prog, textures, tfc)[1]
@propagate_inbounds _th_nodes_float(prog, textures, tfc) = 0f0

# ── Host graph ──────────────────────────────────────────────────────────────

"""
    TexNode

A node of a texture graph. Build graphs with the functions in
[`Hikari.TextureNodes`](@ref) and arithmetic on nodes; pass the result as any
material parameter.
"""
struct TexNode
    op::UInt8
    inputs::Vector{TexNode}
    k::NTuple{4, Float32}
    image::Any
end

TexNode(op::UInt8, inputs::Vector{TexNode} = TexNode[], k = (0f0, 0f0, 0f0, 0f0)) =
    TexNode(op, inputs, Float32.(k), nothing)

const _SWIZZLE_LETTERS = Dict('x' => 1f0, 'y' => 2f0, 'z' => 3f0, 'r' => 1f0, 'g' => 2f0, 'b' => 3f0)

_is_swizzle(s::Symbol) = (str = String(s); 1 <= length(str) <= 3 && all(in(keys(_SWIZZLE_LETTERS)), str))

"""
`node.x`, `node.zy`, `node.rgb`: pick components. A single component is a
scalar; missing trailing components of a shorter swizzle are zero.
"""
function Base.getproperty(n::TexNode, s::Symbol)
    hasfield(TexNode, s) && return getfield(n, s)
    _is_swizzle(s) || error("TexNode has no component `$s`; use x/y/z or r/g/b, up to three of them")
    idx = [_SWIZZLE_LETTERS[c] for c in String(s)]
    k = length(idx) == 1 ? (idx[1], idx[1], idx[1], 0f0) : (idx..., zeros(Float32, 4 - length(idx))...)
    return TexNode(NodeOp.SWIZZLE, [n], k)
end

Base.propertynames(::TexNode) = (:x, :y, :z)
Base.show(io::IO, n::TexNode) = print(io, "TexNode(op = ", n.op, ", ", length(n.inputs), " inputs)")

"""Registers a subtree needs when its costliest inputs are evaluated first (Sethi-Ullman)."""
function _register_need(n::TexNode, memo = IdDict{TexNode, Int}())
    return get!(memo, n) do
        needs = sort([_register_need(i, memo) for i in n.inputs]; rev = true)
        maximum((need + i - 1 for (i, need) in enumerate(needs)); init = 1)
    end
end

"""Inputs in topological order, each node once, the root last; the costliest input of a node goes first."""
function _topological(root::TexNode)
    order = TexNode[]
    seen = IdDict{TexNode, Nothing}()
    memo = IdDict{TexNode, Int}()
    function visit(n)
        haskey(seen, n) && return
        foreach(visit, sort(n.inputs; by = i -> _register_need(i, memo), rev = true))
        seen[n] = nothing
        push!(order, n)
    end
    visit(root)
    return order
end

"""
    compile_nodes(root, store_image) -> Vector{NodeInstr}

Linearize the graph and allocate registers by liveness. `store_image(data)`
returns the `(slot, idx)` an image input lives at.
"""
function compile_nodes(root::TexNode, store_image)
    order = _topological(root)
    position = IdDict{TexNode, Int}(n => i for (i, n) in enumerate(order))
    last_use = zeros(Int, length(order))
    for (i, n) in enumerate(order), input in n.inputs
        last_use[position[input]] = i
    end
    free = collect(UInt8(NODE_REGISTERS):-1:UInt8(1))
    register = zeros(UInt8, length(order))
    program = NodeInstr[]
    for (i, n) in enumerate(order)
        srcs = [register[position[input]] for input in n.inputs]
        for input in unique(n.inputs)
            last_use[position[input]] == i && push!(free, register[position[input]])
        end
        isempty(free) && error("texture graph needs more than $NODE_REGISTERS live values at once")
        register[i] = pop!(free)
        a, b, c = (get(srcs, j, UInt8(1)) for j in 1:3)
        k = n.op == NodeOp.IMAGE ? _image_k(store_image(n.image)) : n.k
        push!(program, NodeInstr(n.op, register[i], a, b, c, k))
    end
    return program
end

_image_k((slot, idx)) = (reinterpret(Float32, Int32(slot)), reinterpret(Float32, Int32(idx)), 0f0, 0f0)

@inline matparam(x::TexNode) = x

_compile_into(dhv, n::TexNode) =
    compile_nodes(n, data -> (ref = Raycore.store_texture(dhv, data); (texref_slot(ref), ref.idx)))

function device_param(dhv, n::TexNode)
    ref = Raycore.store_texture(dhv, _compile_into(dhv, n))
    return TexHandle(TexKind.NODES, 0f0, _TH_ZERO, Int32(texref_slot(ref)), Int32(ref.idx))
end

"""The `idx`-th stored array of type `AT`, found the way `Raycore.copyto_texture!` finds it."""
_stored_array(dhv::Raycore.MultiTypeSet, AT, idx) =
    filter(arr -> typeof(arr) === AT, dhv.texture_gpu_arrays)[idx]

"""
    update_item(dhv, old::TexHandle, new::TexNode)

Overwrite the program `old` names when the new one has the same length, which
is the case whenever only a graph's parameters change, so tweaking them does
not take a new slot on every update.
"""
function Raycore.update_item(dhv::Raycore.MultiTypeSet, old::TexHandle, new::TexNode)
    old.kind == TexKind.NODES || return device_param(dhv, new)
    AT = dhv.texture_order[old.slot]
    program = _compile_into(dhv, new)
    length(program) == length(_stored_array(dhv, AT, old.idx)) || return device_param(dhv, new)
    ref = Raycore.TextureRef{AT, eltype(AT), ndims(AT), Int(old.slot)}(Int(old.idx))
    Raycore.copyto_texture!(dhv, ref, program)
    return old
end

# ============================================================================
# User-facing graph builders
# ============================================================================

"""
Builders for node textures, after Blender's shader nodes. `using Hikari.TextureNodes`, then e.g.

    p = xyz()
    rings = fract(norm(p.xy) * 40 + noise(p * 3) * 2)
    wood = color_ramp(rings, 0.0 => (0.6, 0.35, 0.15), 0.8 => (0.45, 0.25, 0.1), 1.0 => (0.2, 0.1, 0.04))
    Hikari.Diffuse(Kd = wood)

Inputs are [`uv`](@ref), [`xyz`](@ref) (world-space position) and [`normal`](@ref).
Arithmetic (`+ - * / ^`, `min`, `max`, `mod`, `abs`, `floor`, `sin`, `cos`,
`sqrt`, `exp`, `atan`, `clamp`) works elementwise on nodes and plain numbers or
colours; `dot`, `cross`, `norm` and `normalize` come from `LinearAlgebra`.
"""
module TextureNodes

using ..Hikari: TexNode, NodeOp, Texture, RGBSpectrum, to_texture
using LinearAlgebra: LinearAlgebra
using ..Hikari: Colorant, red, green, blue

export uv, xyz, normal, constant, combine, mix, smoothstep, fract, greater, transform, mapping,
       noise, noise_color, voronoi, white_noise, image_texture, color_ramp, rotation_matrix

const NodeLike = Union{TexNode, Real, Tuple{Real, Real, Real}, Colorant, AbstractVector{<:Real}}

constant(x::Real) = TexNode(NodeOp.CONST, TexNode[], (x, x, x, 0))
constant(v::Union{Tuple{Real, Real, Real}, AbstractVector{<:Real}}) = TexNode(NodeOp.CONST, TexNode[], (v[1], v[2], v[3], 0))
constant(c::Colorant) = constant((red(c), green(c), blue(c)))
node(x::TexNode) = x
node(x) = constant(x)

"""Texture coordinates `(u, v, 0)`."""
uv() = TexNode(NodeOp.UV)
"""World-space position of the shading point."""
xyz() = TexNode(NodeOp.POSITION)
"""Shading normal (world space)."""
normal() = TexNode(NodeOp.NORMAL)

op(code, args...; k = (0, 0, 0, 0)) = TexNode(code, TexNode[node(a) for a in args], k)

for (f, code) in ((:+, :ADD), (:-, :SUB), (:*, :MUL), (:/, :DIV), (:^, :POW),
                  (:min, :MIN), (:max, :MAX), (:mod, :MOD), (:atan, :ATAN2))
    @eval begin
        Base.$f(a::TexNode, b::TexNode) = op(NodeOp.$code, a, b)
        Base.$f(a::TexNode, b::NodeLike) = op(NodeOp.$code, a, b)
        Base.$f(a::NodeLike, b::TexNode) = op(NodeOp.$code, a, b)
    end
end
Base.:^(a::TexNode, b::Integer) = op(NodeOp.POW, a, b)

for (f, code) in ((:abs, :ABS), (:floor, :FLOOR), (:sin, :SIN), (:cos, :COS), (:sqrt, :SQRT), (:exp, :EXP))
    @eval Base.$f(a::TexNode) = op(NodeOp.$code, a)
end
Base.:-(a::TexNode) = 0 - a
Base.clamp(x::TexNode, lo::NodeLike, hi::NodeLike) = op(NodeOp.CLAMP, x, lo, hi)
LinearAlgebra.dot(a::TexNode, b::NodeLike) = op(NodeOp.DOT, a, b)
LinearAlgebra.dot(a::NodeLike, b::TexNode) = op(NodeOp.DOT, a, b)
LinearAlgebra.dot(a::TexNode, b::TexNode) = op(NodeOp.DOT, a, b)
LinearAlgebra.cross(a::TexNode, b::NodeLike) = op(NodeOp.CROSS, a, b)
LinearAlgebra.cross(a::NodeLike, b::TexNode) = op(NodeOp.CROSS, a, b)
LinearAlgebra.cross(a::TexNode, b::TexNode) = op(NodeOp.CROSS, a, b)
LinearAlgebra.norm(a::TexNode) = op(NodeOp.LENGTH, a)
LinearAlgebra.normalize(a::TexNode) = op(NodeOp.NORMALIZE, a)

"""`x - floor(x)`."""
fract(x::NodeLike) = op(NodeOp.FRACT, x)
"""`a + (b - a) * t`, elementwise."""
mix(a::NodeLike, b::NodeLike, t::NodeLike) = op(NodeOp.MIX, a, b, t)
"""Hermite step from 0 at `edge0` to 1 at `edge1`."""
smoothstep(edge0::NodeLike, edge1::NodeLike, x::NodeLike) = op(NodeOp.SMOOTHSTEP, edge0, edge1, x)
"""1 where `a > b`, else 0, elementwise."""
greater(a::NodeLike, b::NodeLike) = op(NodeOp.GREATER, a, b)
"""A vector from the first components of three scalars."""
combine(x::NodeLike, y::NodeLike, z::NodeLike) = op(NodeOp.COMBINE, x, y, z)

"""
    transform(p, M)

Apply an affine map to a point: `M` is a 4×4 matrix (only its top three rows
are used) or a 3×3 one. Use the inverse of an object's model matrix to get
coordinates that stay attached to the object.
"""
function transform(p::NodeLike, M::AbstractMatrix)
    size(M) in ((3, 3), (4, 4)) || error("transform needs a 3×3 or 4×4 matrix, got $(size(M))")
    t(i) = size(M, 2) == 4 ? M[i, 4] : 0
    rows = [op(NodeOp.AFFINE_ROW, p; k = (M[i, 1], M[i, 2], M[i, 3], t(i))) for i in 1:3]
    return combine(rows...)
end

"""Rotation by Euler angles in radians, applied about x, then y, then z."""
function rotation_matrix((a, b, c))
    Rx = [1 0 0; 0 cos(a) -sin(a); 0 sin(a) cos(a)]
    Ry = [cos(b) 0 sin(b); 0 1 0; -sin(b) 0 cos(b)]
    Rz = [cos(c) -sin(c) 0; sin(c) cos(c) 0; 0 0 1]
    return Rz * Ry * Rx
end

"""
    mapping(p; location = (0, 0, 0), rotation = (0, 0, 0), scale = (1, 1, 1))

Blender's Mapping node for points: scale, then rotate (Euler xyz, radians),
then translate.
"""
function mapping(p::NodeLike; location = (0, 0, 0), rotation = (0, 0, 0), scale = (1, 1, 1))
    L = rotation_matrix(rotation) * LinearAlgebra.Diagonal(collect(Float64.(scale)))
    return transform(p, [L collect(Float64.(location)); 0 0 0 1])
end

"""
    noise(p; detail = 2, roughness = 0.5, lacunarity = 2)

Fractal gradient noise of a 3D coordinate, in [0, 1] around 0.5.
"""
noise(p::NodeLike; detail = 2, roughness = 0.5, lacunarity = 2) =
    op(NodeOp.NOISE, p; k = (detail, roughness, lacunarity, 0))

"""Three decorrelated [`noise`](@ref) channels."""
noise_color(p::NodeLike; detail = 2, roughness = 0.5, lacunarity = 2) =
    op(NodeOp.NOISE, p; k = (detail, roughness, lacunarity, 1))

"""
    voronoi(p; randomness = 1, output = :distance)

3D Voronoi (F1) of a coordinate. `output` is `:distance` to the nearest
feature point, `:color`, a random colour per cell, or `:position` of the
feature point.
"""
function voronoi(p::NodeLike; randomness = 1, output::Symbol = :distance)
    mode = Dict(:distance => 0, :color => 1, :position => 2)
    haskey(mode, output) || error("voronoi output must be :distance, :color or :position, got :$output")
    return op(NodeOp.VORONOI, p; k = (randomness, mode[output], 0, 0))
end

"""A random value in [0, 1] per channel for every distinct input, e.g. `white_noise(floor(p))` per cell."""
white_noise(p::NodeLike) = op(NodeOp.WHITE_NOISE, p)

"""
    image_texture(img, coords = uv())

Sample an image (a matrix of colours or numbers, or a `Hikari.Texture`) at the
first two components of `coords`, wrapping like a uv lookup.
"""
function image_texture(img, coords::NodeLike = uv())
    data = img isa Texture ? img.data : img isa AbstractMatrix{<:Colorant} ? to_texture(img).data : img
    ndims(data) == 2 || error("image needs a 2D image, got $(ndims(data)) dimensions")
    return TexNode(NodeOp.IMAGE, TexNode[node(coords)], (0f0, 0f0, 0f0, 0f0), data)
end

"""
    color_ramp(t, stops...)

Piecewise-linear ramp: `stops` are `position => value` pairs in increasing
position; `t` below the first or above the last stop holds the end value.
"""
function color_ramp(t::NodeLike, stops::Pair...)
    isempty(stops) && error("color_ramp needs at least one stop")
    issorted(first.(stops)) || error("color_ramp stops must be in increasing position")
    out = node(last(stops[1]))
    for ((t0, _), (t1, v1)) in zip(stops[1:end-1], stops[2:end])
        out = mix(out, v1, clamp((t - t0) / max(t1 - t0, 1e-6), 0, 1))
    end
    return out
end

end
