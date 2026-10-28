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
    NodeOp.ATTRIBUTE   => [:color, :fac],
    NodeOp.IMAGE       => [:color, :alpha],
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
    NodeOp.MIX => NodeFeature.MIX, NodeOp.CLAMP => NodeFeature.RANGE, NodeOp.MAP_RANGE => NodeFeature.RANGE,
    NodeOp.ATTRIBUTE => NodeFeature.ATTRIBUTE, NodeOp.ATTRIBUTE_DX => NodeFeature.ATTRIBUTE,
    NodeOp.ATTRIBUTE_DY => NodeFeature.ATTRIBUTE,
    NodeOp.BUMP => NodeFeature.BUMP, NodeOp.POSITION_DX => NodeFeature.BUMP,
    NodeOp.POSITION_DY => NodeFeature.BUMP, NodeOp.UV_DX => NodeFeature.BUMP, NodeOp.UV_DY => NodeFeature.BUMP)

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

const _ATTRIBUTE_OPS = (NodeOp.ATTRIBUTE, NodeOp.ATTRIBUTE_DX, NodeOp.ATTRIBUTE_DY)

const _SHADER_OPS = (NodeOp.CLOSURE, NodeOp.PRINCIPLED, NodeOp.BUMP, NodeOp.POSITION_DX, NodeOp.POSITION_DY,
                     NodeOp.UV_DX, NodeOp.UV_DY, NodeOp.NORMAL_MAP, NodeOp.ATTRIBUTE, NodeOp.ATTRIBUTE_DX,
                     NodeOp.ATTRIBUTE_DY, NodeOp.CAMERA, NodeOp.WINDOW, NodeOp.REFLECTION)

"""
A compiled node program before it is linked: its instructions, each read of a stored array (an
image, a mesh attribute) holding the read's ordinal where the array's place goes, and the data of
the reads in that order. Compiling stores nothing: `link` puts in where each datum is stored, and
where that is is decided by `place!`, so a program pushed new and a program replacing another
compile the same way and differ only in where their data goes.
"""
struct CompiledProgram{F}
    instructions::Vector{NodeInstr{F}}
    data::Vector{Any}
end

"""
    compile_nodes(root) -> CompiledProgram

Linearize the graph and place every value on the stack by liveness.
"""
function compile_nodes(root::TexNode)
    root = _simplified(root, IdDict{NodeCore, NodeCore}())
    order = _topological([getfield(root, :core)])
    any(c -> c.op in _SHADER_OPS, order) &&
        error("a texture cannot use `bump`, `normal_map`, mesh attributes, texture coordinates other than uv or a " *
              "shader node; use them in a shader graph")
    return _compile(order, root, NodeFeature.TEXTURE)
end

"""
    compile_shader(closures) -> CompiledProgram

The program of a shader: every value the BSDF nodes `closures` read, then their instructions in
that order, so a material runs the values and finds every BSDF's inputs still on the stack.
"""
function compile_shader(closures::Vector{NodeCore})
    memo = IdDict{NodeCore, NodeCore}()
    cores = [folded(_simplified(c, memo)) for c in closures]
    values = _topological(unique(reduce(vcat, map(_input_cores, cores); init = NodeCore[])))
    return _compile(vcat(values, cores), nothing, UInt32(0))
end

"""Whether argument code `o` names a slot of the value stack: neither a constant nor a folded input."""
@inline reads_stack(o::UInt8) = o != NO_SLOT && o < WIDE_CONSTANT_ARG

"""
    closure_program(c::CompiledProgram) -> (instructions, inputs)

The BSDF part of the shader program `c`, from its first BSDF instruction to its `RESULT`, as the
closure kernel runs it: every input read off the value stack renumbered to where the values kernel
gathers it (`gather_inputs!`), the inputs in program order and a vector taking three floats.
`inputs` is how many floats that is. It names no node family, so a shader of other families has
the same one and its closure kernel stays compiled.
"""
function closure_program(c::CompiledProgram)
    first_bsdf = findfirst(ins -> ins.op in (NodeOp.CLOSURE, NodeOp.PRINCIPLED, NodeOp.RESULT), c.instructions)
    g = 0
    instructions = map(c.instructions[first_bsdf:end]) do ins
        args = collect(ins.arg)
        for a in 1:NODE_ARGS
            reads_stack(args[a]) || continue
            args[a] = UInt8(g + 1)
            g += _bit(ins.wide, a) ? 3 : 1
        end
        NodeInstr{NodeFeature.CLOSURES}(ins.op, ins.mode, ins.out, ins.outwide, Tuple(args), ins.wide, ins.k)
    end
    g < Int(WIDE_CONSTANT_ARG) || throw(ArgumentError(
        "a shader's BSDF nodes read $g floats of computed values; at most $(Int(WIDE_CONSTANT_ARG) - 1) are gathered"))
    return instructions, g
end

"""A BSDF input the compiler folds into the BSDF's instruction: the shading normal, as `normal()` reads it."""
struct ShadingNormal end
"""A BSDF input the compiler folds into the data after the BSDF's instructions: a constant colour."""
struct WideConstant
    value::Vec3f
end

"""The argument code of a BSDF input that is the shading normal; codes from `WIDE_CONSTANT_ARG` up are folded colours."""
const SHADING_NORMAL_ARG = 0xFF
const WIDE_CONSTANT_ARG = 0xC0

"""
`c`, a BSDF node, with the inputs that need no instruction folded into its own: the default
normal and constant colours. A BSDF of such inputs alone runs no value instruction at all.
"""
folded(c::NodeCore) = NodeCore(c.op, c.mode, Any[fold_input(x) for x in c.inputs], c.widths, c.data)
fold_input(x) = x
function fold_input(x::TexNode)
    c = getfield(x, :core)
    c.op == NodeOp.NORMAL && isempty(c.inputs) && return ShadingNormal()
    c.op == NodeOp.CONST && return WideConstant(Vec3f(c.inputs[1], c.inputs[2], c.inputs[3]))
    return x
end

input_arg(::ShadingNormal, wides) = SHADING_NORMAL_ARG
input_arg(x::WideConstant, wides) = (push!(wides, x.value); WIDE_CONSTANT_ARG + UInt8(length(wides) - 1))
input_arg(x, wides) = NO_SLOT
input_k(::Union{ShadingNormal, WideConstant}) = 0f0
input_k(x) = Float32(something(x, 0f0))

"""Folded colours packed three to a `DATA` instruction, each at a multiple of three floats."""
wide_constant_instructions(wides::Vector{Vec3f}) =
    [(NodeOp.DATA, (0x00, 0x00, 0x00), _pad(UInt8[], NODE_OUTS, NO_SLOT), 0x00, _pad(UInt8[], NODE_ARGS, NO_SLOT),
      UInt16(0), _pad(reduce(vcat, map(collect, chunk)), NODE_ARGS, 0f0))
     for chunk in Iterators.partition(wides, 3)]

"""
The graph under a node as it is run: a constant colour of three equal components read as the
scalar it splats from, and the Principled BSDF inputs that a weight of constant zero leaves unread
cut off, so neither costs an instruction.
"""
_simplified(n::TexNode, memo) = TexNode(_simplified(getfield(n, :core), memo), getfield(n, :output))
_simplified(x, memo) = x
function _simplified(c::NodeCore, memo::IdDict{NodeCore, NodeCore})
    return get!(memo, c) do
        inputs = Any[_simplified(x, memo) for x in c.inputs]
        c.op == NodeOp.PRINCIPLED && _prune_principled!(inputs)
        NodeCore(c.op, c.mode, Any[_splat_constant(x) for x in inputs], c.widths, c.data)
    end
end

_splat_constant(x) = x
function _splat_constant(n::TexNode)
    c = getfield(n, :core)
    (c.op == NodeOp.CONST && allequal(c.inputs)) || return n
    return Float32(c.inputs[1])
end

"""The Principled BSDF inputs, by position, that each weight leaves unread when it is zero."""
const _PRINCIPLED_UNREAD = (16 => (17, 18), 9 => (10, 11, 12, 13), 19 => (20, 21), 22 => (23, 24, 25),
                            27 => (28, 29), 31 => (30,), 32 => (33,))

function _prune_principled!(inputs)
    for (weight, unread) in _PRINCIPLED_UNREAD
        _constant(inputs[weight], 0) || continue
        foreach(i -> inputs[i] = 0f0, unread)
    end
    # The sheen normal blends towards the coat normal by the coat weight, so the coat normal is
    # still read: it becomes the normal, which is computed anyway.
    _constant(inputs[22], 0) && (inputs[26] = inputs[7])
    return inputs
end

function _compile(order::Vector{NodeCore}, root, kind::UInt32)
    features = mapreduce(_features, |, order; init = kind)
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
    data = Any[]
    used = 0
    for (i, c) in enumerate(order)
        args = UInt8[]
        wide = Bool[]
        k = Float32[]
        wides = Vec3f[]
        for x in c.inputs
            if x isa TexNode
                key = (position[getfield(x, :core)], getfield(x, :output))
                push!(args, UInt8(offset[key]))
                push!(wide, _width(x) == 3)
                push!(k, 0f0)
            else
                push!(args, input_arg(x, wides))
                push!(wide, false)
                push!(k, input_k(x))
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
        if c.op == NodeOp.IMAGE || (c.op in _ATTRIBUTE_OPS && c.data isa BoundAttribute)
            push!(data, c.op == NodeOp.IMAGE ? c.data : c.data.corners)
            kt = located(kt, length(data), 0)
        end
        data_count = cld(length(wides), 3)
        mode = isempty(wides) ? c.mode : (c.mode[1], c.mode[2], UInt8(data_count))
        push!(instructions, (c.op, mode, _pad(outs, NODE_OUTS, NO_SLOT), outwide, _pad(args[first_chunk], NODE_ARGS, NO_SLOT),
                             _wide_bits(wide[first_chunk]), kt))
        for chunk in Iterators.drop(chunks, 1)
            push!(instructions, (NodeOp.ARGS, (0x00, 0x00, 0x00), _pad(UInt8[], NODE_OUTS, NO_SLOT), 0x00,
                                 _pad(args[chunk], NODE_ARGS, NO_SLOT), _wide_bits(wide[chunk]), _pad(k[chunk], NODE_ARGS, 0f0)))
        end
        append!(instructions, wide_constant_instructions(wides))
        c.data isa Vector{Float32} && append!(instructions, _data_instructions(c.data))
    end
    result_args = root === nothing ? UInt8[] : [UInt8(offset[root_key])]
    result_wide = root !== nothing && _width(root) == 3 ? UInt16(1) : UInt16(0)
    push!(instructions, (NodeOp.RESULT, (0x00, 0x00, 0x00), _pad(UInt8[], NODE_OUTS, NO_SLOT), 0x00,
                         _pad(result_args, NODE_ARGS, NO_SLOT), result_wide, _pad(Float32[], NODE_ARGS, 0f0)))
    F = features | _stack_bits(max(used, 1))
    return CompiledProgram([NodeInstr{F}(fields...) for fields in instructions], data)
end

"`k` with the place of a stored array in the two constants a read of it takes it from."
located(k::NTuple{NODE_ARGS, Float32}, slot, idx) =
    Base.setindex(Base.setindex(k, reinterpret(Float32, Int32(slot)), 9), reinterpret(Float32, Int32(idx)), 10)

"Whether `ins` reads a stored array: an image, or a mesh attribute bound to one (slot 0 is none, as the device reads it)."
reads_array(ins::NodeInstr) = (ins.op == NodeOp.IMAGE || ins.op in _ATTRIBUTE_OPS) && reinterpret(Int32, ins.k[9]) > Int32(0)

"The place each read of a linked `program` takes its array from, in program order: what `link` put in."
array_places(program::AbstractVector{<:NodeInstr}) =
    [(Int(reinterpret(Int32, ins.k[9])), Int(reinterpret(Int32, ins.k[10]))) for ins in program if reads_array(ins)]

"""
    link(compiled, places) -> Vector{NodeInstr{F}}

The program a device runs: the read of the `k`-th datum taking its array from `places[k]`, its
`(slot, idx)` in the set the program is stored in.
"""
link(c::CompiledProgram{F}, places) where {F} =
    [reads_array(ins) ? NodeInstr{F}(ins.op, ins.mode, ins.out, ins.outwide, ins.arg, ins.wide,
                                     located(ins.k, places[reinterpret(Int32, ins.k[9])]...)) : ins
     for ins in c.instructions]

_wide_bits(wide) = reduce(|, (UInt16(w) << (j - 1) for (j, w) in enumerate(wide)); init = UInt16(0))

"""Floats carried inline after the instruction that reads them, `NODE_ARGS` to an instruction."""
_data_instructions(data::Vector{Float32}) =
    [(NodeOp.DATA, (0x00, 0x00, 0x00), _pad(UInt8[], NODE_OUTS, NO_SLOT), 0x00, _pad(UInt8[], NODE_ARGS, NO_SLOT),
      UInt16(0), _pad(chunk, NODE_ARGS, 0f0))
     for chunk in Iterators.partition(data, NODE_ARGS)]


"Where the data of one material's programs went, so a datum two of them read is stored once."
const Places = IdDict{Any, Tuple{Int, Int}}

"""
    place!(dhv, data, placed, over = ()) -> places

Where in `dhv` each datum of `data` goes: the `k`-th over `over[k]`, copied into that array when it
holds the datum's element type and dimensions, else into an array stored anew. A datum `placed`
already has stays where it is, and an array goes to one datum of a placement only. An image stored
already (a `TexHandle`) is read where it is, and no array is copied over it.

`over` is the places of the program a new one replaces (`array_places`), so an update keeps the
arrays its program reads, as Raycore keeps a material's texture field
(`update_item(dhv, ::TextureRef, ::AbstractArray)`): the set takes no array and its layout, and
with it every plan recorded against it, stays.
"""
function place!(dhv::Raycore.MultiTypeSet, data, placed::Places, over = ())
    for datum in data
        datum isa TexHandle && get!(placed, datum, (Int(datum.slot), Int(datum.idx)))
    end
    return map(enumerate(data)) do (k, datum)
        get!(placed, datum) do
            stored = texels(datum)
            k <= length(over) && holds(dhv, over[k], stored) && !(over[k] in values(placed)) ?
                copied!(dhv, over[k], stored) : stored!(dhv, stored)
        end
    end
end

"""A datum as a program reads it: an image of colours as `RGBSpectrum`s, one of numbers as `Float32`s."""
texels(img::AbstractMatrix{<:Colorant}) = to_texture(img).data
texels(img::AbstractMatrix{<:Real}) = convert(Matrix{Float32}, img)
texels(data) = data

"Whether the array at `place` can take `data`: of its element type and dimensions. Its size may differ."
function holds(dhv::Raycore.MultiTypeSet, (slot, idx)::Tuple{Int, Int}, data::AbstractArray)
    AT = dhv.texture_order[slot]
    return eltype(AT) === eltype(data) && ndims(AT) == ndims(data)
end

function copied!(dhv::Raycore.MultiTypeSet, (slot, idx)::Tuple{Int, Int}, data::AbstractArray)
    AT = dhv.texture_order[slot]
    Raycore.copyto_texture!(dhv, Raycore.TextureRef{AT, eltype(AT), ndims(AT), slot}(idx), data)
    return (slot, idx)
end

stored!(dhv::Raycore.MultiTypeSet, data::AbstractArray) = (ref = Raycore.store_texture(dhv, data); (texref_slot(ref), ref.idx))

"The array `dhv` holds at `place`, found the way `Raycore.copyto_texture!` finds it."
stored_array(dhv::Raycore.MultiTypeSet, (slot, idx)::Tuple{Int, Int}) =
    filter(arr -> typeof(arr) === dhv.texture_order[slot], dhv.texture_gpu_arrays)[idx]

"""
    stored_program!(dhv, compiled, placed) -> (slot, idx)

`compiled` linked to its data placed anew in `dhv`, and stored; where it is.
"""
stored_program!(dhv::Raycore.MultiTypeSet, @nospecialize(c::CompiledProgram), placed::Places) =
    stored!(dhv, link(c, place!(dhv, c.data, placed)))

"""
    updated_program!(dhv, place, compiled, placed) -> (slot, idx)

`compiled` in place of the program `dhv` holds at `place`: its data placed over that program's,
and the program placed over it by the same rule, so it keeps its place whenever it keeps its node
families, whatever its length. Else it is stored anew, where it returns.
"""
function updated_program!(dhv::Raycore.MultiTypeSet, place::Tuple{Int, Int}, c::CompiledProgram, placed::Places)
    old = Array(stored_array(dhv, place))
    program = link(c, place!(dhv, c.data, placed, array_places(old)))
    return holds(dhv, place, program) ? copied!(dhv, place, program) : stored!(dhv, program)
end

function device_param(dhv, n::TexNode)
    slot, idx = stored_program!(dhv, compile_nodes(n), Places())
    return TexHandle(TexKind.NODES, 0f0, _TH_ZERO, Int32(slot), Int32(idx))
end

"""
    update_item(dhv, old::TexHandle, new::TexNode)

The program of `new` where the one `old` names is, with the images it reads where that one's
are (`updated_program!`), so updating a graph takes no array and keeps the set's layout.
"""
function Raycore.update_item(dhv::Raycore.MultiTypeSet, old::TexHandle, new::TexNode)
    old.kind == TexKind.NODES || return device_param(dhv, new)
    slot, idx = updated_program!(dhv, (Int(old.slot), Int(old.idx)), compile_nodes(new), Places())
    return TexHandle(TexKind.NODES, 0f0, _TH_ZERO, Int32(slot), Int32(idx))
end
