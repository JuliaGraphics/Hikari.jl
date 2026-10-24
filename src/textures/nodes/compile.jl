# ── Host graph ──────────────────────────────────────────────────────────────

"""
One node of a texture graph, as the interpreter runs it: an operation, its
settings, its inputs (`TexNode` outputs of other nodes, or `Float32`
constants) and the width of each output (1 for a scalar, 3 for a vector).
"""
mutable struct NodeCore
    op::UInt8
    mode::NTuple{3, UInt8}
    inputs::Vector{Any}
    widths::Vector{Int}
    data::Any
end

"""
    TexNode

An output of a node in a texture graph. Build graphs with the functions in
[`Hikari.TextureNodes`](@ref) and arithmetic on nodes; pass the result as any
material parameter.
"""
struct TexNode
    core::NodeCore
    output::Int
end

_width(n::TexNode) = getfield(n, :core).widths[getfield(n, :output)]
_width(::Real) = 1

Base.show(io::IO, n::TexNode) =
    print(io, "TexNode(op = ", getfield(n, :core).op, ", output ", getfield(n, :output), ")")

const _OUTPUT_NAMES = Dict{UInt8, Vector{Symbol}}(
    NodeOp.NOISE       => [:fac, :color],
    NodeOp.VORONOI     => [:distance, :color, :position, :w, :radius],
    NodeOp.WHITE_NOISE => [:value, :color],
    NodeOp.WAVE        => [:color, :fac],
    NodeOp.GRADIENT    => [:color, :fac],
    NodeOp.MAGIC       => [:color, :fac],
    NodeOp.CHECKER     => [:color, :fac],
    NodeOp.BRICK       => [:color, :fac],
)

const _SWIZZLE_LETTERS = Dict('x' => 0x01, 'y' => 0x02, 'z' => 0x03, 'r' => 0x01, 'g' => 0x02, 'b' => 0x03)

_is_swizzle(s::Symbol) = (str = String(s); 1 <= length(str) <= 3 && all(in(keys(_SWIZZLE_LETTERS)), str))

function _swizzle(n::TexNode, s::Symbol)
    idx = [_SWIZZLE_LETTERS[c] for c in String(s)]
    mode = length(idx) == 1 ? (idx[1], idx[1], idx[1]) : Tuple(vcat(idx, zeros(UInt8, 3 - length(idx))))
    return TexNode(NodeCore(NodeOp.SWIZZLE, mode, Any[n], [length(idx) == 1 ? 1 : 3], nothing), 1)
end

"""
`node.x`, `node.zy`, `node.rgb`: pick components; a single component is a
scalar, missing trailing components of a shorter swizzle are zero. A node with
several outputs names the others: `noise(p).color`, `voronoi(p).position`.
"""
function Base.getproperty(n::TexNode, s::Symbol)
    hasfield(TexNode, s) && return getfield(n, s)
    core = getfield(n, :core)
    names = get(_OUTPUT_NAMES, core.op, Symbol[])
    j = findfirst(==(s), names)
    j === nothing || return TexNode(core, j)
    _is_swizzle(s) && return _swizzle(n, s)
    outputs = isempty(names) ? "" : ", or one of its outputs $(join(names, ", "))"
    error("TexNode has no component `$s`; use x/y/z or r/g/b, up to three of them$outputs")
end

Base.propertynames(n::TexNode) = (:x, :y, :z, get(_OUTPUT_NAMES, getfield(n, :core).op, Symbol[])...)

# ── Compiler ────────────────────────────────────────────────────────────────

_cores(n::TexNode) = getfield(n, :core)
_input_cores(c::NodeCore) = [getfield(x, :core) for x in c.inputs if x isa TexNode]

"""Stack floats a node needs when its costliest inputs are evaluated first (Sethi-Ullman)."""
function _stack_need(c::NodeCore, memo = IdDict{NodeCore, Int}())
    return get!(memo, c) do
        inputs = unique(_input_cores(c))
        needs = sort([(_stack_need(i, memo), sum(i.widths)) for i in inputs]; rev = true)
        held = 0
        need = sum(c.widths)
        for (n, w) in needs
            need = max(need, held + n)
            held += w
        end
        max(need, held + sum(c.widths))
    end
end

"""Nodes in evaluation order, each once, every root after its inputs; the costliest input of a node goes first."""
function _topological(roots::Vector{NodeCore})
    order = NodeCore[]
    seen = IdDict{NodeCore, Nothing}()
    memo = IdDict{NodeCore, Int}()
    stack = Any[(r, false) for r in reverse(roots)]
    while !isempty(stack)
        c, expanded = pop!(stack)
        haskey(seen, c) && continue
        if expanded
            seen[c] = nothing
            push!(order, c)
        else
            push!(stack, (c, true))
            inputs = sort(unique(_input_cores(c)); by = i -> _stack_need(i, memo))
            foreach(i -> haskey(seen, i) || push!(stack, (i, false)), inputs)
        end
    end
    return order
end

"""First-fit allocation of `width` consecutive stack floats."""
function _allocate!(free::BitVector, width::Int)
    for start in 1:(length(free) - width + 1)
        if all(view(free, start:start + width - 1))
            free[start:start + width - 1] .= false
            return start
        end
    end
    error("texture graph needs more than $NODE_STACK floats of live values at once")
end

_pad(xs, n, fill) = ntuple(i -> i <= length(xs) ? xs[i] : fill, n)

const _OP_FEATURES = Dict(
    NodeOp.HSV => NodeFeature.COLOR, NodeOp.GAMMA => NodeFeature.COLOR, NodeOp.BRIGHT_CONTRAST => NodeFeature.COLOR,
    NodeOp.INVERT => NodeFeature.COLOR, NodeOp.RGB_TO_BW => NodeFeature.COLOR, NodeOp.SEPARATE_COLOR => NodeFeature.COLOR,
    NodeOp.COMBINE_COLOR => NodeFeature.COLOR, NodeOp.RAMP => NodeFeature.RAMP, NodeOp.MAPPING => NodeFeature.VECTOR,
    NodeOp.VECTOR_ROTATE => NodeFeature.VECTOR, NodeOp.IMAGE => NodeFeature.IMAGE, NodeOp.NOISE => NodeFeature.NOISE,
    NodeOp.VORONOI => NodeFeature.VORONOI, NodeOp.WHITE_NOISE => NodeFeature.WHITE_NOISE, NodeOp.WAVE => NodeFeature.WAVE,
    NodeOp.GRADIENT => NodeFeature.GRADIENT, NodeOp.MAGIC => NodeFeature.MAGIC, NodeOp.CHECKER => NodeFeature.CHECKER,
    NodeOp.BRICK => NodeFeature.BRICK, NodeOp.VECTOR_MATH => NodeFeature.VECTOR_MATH, NodeOp.AFFINE_ROW => NodeFeature.VECTOR,
    NodeOp.MIX => NodeFeature.MIX, NodeOp.CLAMP => NodeFeature.RANGE, NodeOp.MAP_RANGE => NodeFeature.RANGE)

"""The `NodeFeature` bits the interpreter needs to run node `c`."""
function _features(c::NodeCore)
    f = get(_OP_FEATURES, c.op, UInt32(0))
    dims, kind = c.mode
    if c.op == NodeOp.MATH
        f |= _math_group(c.mode[1])
    elseif c.op == NodeOp.MIX && c.mode[1] != BlendType.MIX
        f |= NodeFeature.BLEND
    elseif c.op == NodeOp.NOISE
        dims != 0x03 && (f |= NodeFeature.NOISE_DIMS)
        kind != NoiseType.FBM && (f |= NodeFeature.NOISE_TYPES)
    elseif c.op == NodeOp.VORONOI
        dims != 0x03 && (f |= NodeFeature.VORONOI_DIMS)
        kind == VoronoiFeature.SMOOTH_F1 && (f |= NodeFeature.VORONOI_SMOOTH)
        kind in (VoronoiFeature.DISTANCE_TO_EDGE, VoronoiFeature.N_SPHERE_RADIUS) && (f |= NodeFeature.VORONOI_EDGE)
    end
    return f
end

const _SHADER_OPS = (NodeOp.CLOSURE, NodeOp.PRINCIPLED, NodeOp.BUMP, NodeOp.POSITION_DX, NodeOp.POSITION_DY,
                     NodeOp.UV_DX, NodeOp.UV_DY, NodeOp.NORMAL_MAP)

"""
    compile_nodes(root, store_image) -> Vector{NodeInstr{F}}

Linearize the graph and place every value on the stack by liveness.
`store_image(data)` returns the `(slot, idx)` an image input lives at.
"""
function compile_nodes(root::TexNode, store_image)
    order = _topological([getfield(root, :core)])
    any(c -> c.op in _SHADER_OPS, order) &&
        error("a texture cannot use `bump`, `normal_map` or a shader node; give the normal to a shader's `normal` input")
    return _compile(order, root, store_image)
end

"""
    compile_shader(closures, store_image) -> Vector{NodeInstr{F}}

The program of a shader: every closure node in `closures` with what it reads, in one program
that runs until its `RESULT`.
"""
compile_shader(closures::Vector{NodeCore}, store_image) = _compile(_topological(closures), nothing, store_image)

function _compile(order::Vector{NodeCore}, root, store_image)
    features = mapreduce(_features, |, order; init = UInt32(0))
    position = IdDict{NodeCore, Int}(c => i for (i, c) in enumerate(order))
    last_use = Dict{Tuple{Int, Int}, Int}()
    for (i, c) in enumerate(order), x in c.inputs
        x isa TexNode && (last_use[(position[getfield(x, :core)], getfield(x, :output))] = i)
    end
    if root !== nothing
        root_key = (position[getfield(root, :core)], getfield(root, :output))
        last_use[root_key] = length(order) + 1
    end

    free = trues(NODE_STACK)
    offset = Dict{Tuple{Int, Int}, Int}()
    instructions = []
    used = 0
    for (i, c) in enumerate(order)
        args = UInt8[]
        wide = Bool[]
        k = Float32[]
        for x in c.inputs
            if x isa TexNode
                key = (position[getfield(x, :core)], getfield(x, :output))
                push!(args, UInt8(offset[key]))
                push!(wide, _width(x) == 3)
                push!(k, 0f0)
            else
                push!(args, NO_SLOT)
                push!(wide, false)
                push!(k, Float32(something(x, 0f0)))
            end
        end
        for x in unique(filter(x -> x isa TexNode, c.inputs))
            key = (position[getfield(x, :core)], getfield(x, :output))
            if last_use[key] == i
                free[offset[key]:offset[key] + _width(x) - 1] .= true
            end
        end
        outs = UInt8[]
        outwide = UInt8(0)
        for (j, w) in enumerate(c.widths)
            if haskey(last_use, (i, j))
                offset[(i, j)] = _allocate!(free, w)
                used = max(used, offset[(i, j)] + w - 1)
                push!(outs, UInt8(offset[(i, j)]))
                w == 3 && (outwide |= UInt8(1) << (j - 1))
            else
                push!(outs, NO_SLOT)
            end
        end
        chunks = collect(Iterators.partition(eachindex(args), NODE_ARGS))
        first_chunk = isempty(chunks) ? Int[] : collect(first(chunks))
        kt = _pad(k[first_chunk], NODE_ARGS, 0f0)
        if c.op == NodeOp.IMAGE
            slot, idx = store_image(c.data)
            kt = Base.setindex(Base.setindex(kt, reinterpret(Float32, Int32(slot)), 9), reinterpret(Float32, Int32(idx)), 10)
        end
        push!(instructions, (c.op, c.mode, _pad(outs, NODE_OUTS, NO_SLOT), outwide, _pad(args[first_chunk], NODE_ARGS, NO_SLOT),
                             _wide_bits(wide[first_chunk]), kt))
        for chunk in Iterators.drop(chunks, 1)
            push!(instructions, (NodeOp.ARGS, (0x00, 0x00, 0x00), _pad(UInt8[], NODE_OUTS, NO_SLOT), 0x00,
                                 _pad(args[chunk], NODE_ARGS, NO_SLOT), _wide_bits(wide[chunk]), _pad(k[chunk], NODE_ARGS, 0f0)))
        end
        c.data isa Vector{Float32} && append!(instructions, _data_instructions(c.data))
    end
    result_args = root === nothing ? UInt8[] : [UInt8(offset[root_key])]
    result_wide = root !== nothing && _width(root) == 3 ? UInt16(1) : UInt16(0)
    push!(instructions, (NodeOp.RESULT, (0x00, 0x00, 0x00), _pad(UInt8[], NODE_OUTS, NO_SLOT), 0x00,
                         _pad(result_args, NODE_ARGS, NO_SLOT), result_wide, _pad(Float32[], NODE_ARGS, 0f0)))
    F = features | _stack_bits(max(used, 1))
    return [NodeInstr{F}(fields...) for fields in instructions]
end

_wide_bits(wide) = reduce(|, (UInt16(w) << (j - 1) for (j, w) in enumerate(wide)); init = UInt16(0))

"""Floats carried inline after the instruction that reads them, `NODE_ARGS` to an instruction."""
_data_instructions(data::Vector{Float32}) =
    [(NodeOp.DATA, (0x00, 0x00, 0x00), _pad(UInt8[], NODE_OUTS, NO_SLOT), 0x00, _pad(UInt8[], NODE_ARGS, NO_SLOT),
      UInt16(0), _pad(chunk, NODE_ARGS, 0f0))
     for chunk in Iterators.partition(data, NODE_ARGS)]

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

Overwrite the program `old` names when the new one has the same length and node
families, which is the case whenever only a graph's parameters change, so
tweaking them does not take a new slot on every update.
"""
function Raycore.update_item(dhv::Raycore.MultiTypeSet, old::TexHandle, new::TexNode)
    old.kind == TexKind.NODES || return device_param(dhv, new)
    AT = dhv.texture_order[old.slot]
    program = _compile_into(dhv, new)
    eltype(program) === eltype(AT) && length(program) == length(_stored_array(dhv, AT, old.idx)) || return device_param(dhv, new)
    ref = Raycore.TextureRef{AT, eltype(AT), ndims(AT), Int(old.slot)}(Int(old.idx))
    Raycore.copyto_texture!(dhv, ref, program)
    return old
end
