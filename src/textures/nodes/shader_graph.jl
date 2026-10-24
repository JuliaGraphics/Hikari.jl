# ── Host shader graph ───────────────────────────────────────────────────────

"""
A shader in a graph: a BSDF node, or a Mix Shader or Add Shader of others. Build them with the
functions in [`Hikari.TextureNodes`](@ref) and pass the result to [`ShaderMaterial`](@ref).
"""
abstract type ShaderNode end

"""A BSDF node: the instruction `op` with its settings in `mode` and its inputs, weight excluded."""
struct BSDFNode <: ShaderNode
    op::UInt8
    mode::NTuple{3, UInt8}
    inputs::Vector{Any}
end

struct MixShaderNode <: ShaderNode
    fac::Any
    a::ShaderNode
    b::ShaderNode
end

struct AddShaderNode <: ShaderNode
    a::ShaderNode
    b::ShaderNode
end

"""Every BSDF under `s` with the weight the mixes above it give it, as Cycles' compiler flattens a shader tree."""
_weighted(s::BSDFNode, w) = [(s, w)]
_weighted(s::MixShaderNode, w) = vcat(_weighted(s.a, w * (1 - s.fac)), _weighted(s.b, w * s.fac))
_weighted(s::AddShaderNode, w) = vcat(_weighted(s.a, w), _weighted(s.b, w))

_weight_input(w::Real) = Float32(w)
_weight_input(w::TexNode) = w

"""The closure nodes of a shader, one per BSDF, each reading its weight first."""
closure_cores(shader::ShaderNode) =
    [NodeCore(b.op, b.mode, Any[_weight_input(w); b.inputs], Int[], nothing) for (b, w) in _weighted(shader, 1f0)]

"""A shader graph as a material holds it until it is compiled at push."""
struct ShaderGraph
    closures::Vector{NodeCore}
end

ShaderGraph(shader::ShaderNode) = ShaderGraph(closure_cores(shader))

_constant(x, value) = x isa Real && Float32(x) == Float32(value)

"""Whether a shader emits light: an emission closure, or a Principled BSDF whose emission strength is not zero."""
has_emission(g::ShaderGraph) =
    any(c -> (c.op == NodeOp.CLOSURE && c.mode[1] == ClosureKind.EMISSION) ||
             (c.op == NodeOp.PRINCIPLED && !_constant(c.inputs[31], 0)), g.closures)

"""Whether a shader lets light through unrefracted: a transparent closure, or a Principled BSDF whose alpha is not one."""
has_transparency(g::ShaderGraph) =
    any(c -> (c.op == NodeOp.CLOSURE && c.mode[1] == ClosureKind.TRANSPARENT) ||
             (c.op == NodeOp.PRINCIPLED && !_constant(c.inputs[6], 1)), g.closures)

function device_param(dhv, g::ShaderGraph)
    store_lookup_tables!(dhv)
    program = compile_shader(g.closures, data -> (ref = Raycore.store_texture(dhv, data); (texref_slot(ref), ref.idx)))
    ref = Raycore.store_texture(dhv, program)
    return TexHandle(TexKind.NODES, 0f0, _TH_ZERO, Int32(texref_slot(ref)), Int32(ref.idx))
end

"""
`n` with every position and uv it reads moved by `filter_width` of the screen-space differential
`axis` (`:dx` or `:dy`): the height a bump compares with the one at the hit.
"""
function shifted_graph(n::TexNode, axis::Symbol, filter_width::Float32)
    reads = IdDict{NodeCore, Bool}()
    return TexNode(_shifted(getfield(n, :core), axis, filter_width, IdDict{NodeCore, NodeCore}(), reads),
                   getfield(n, :output))
end

_reads_coordinates(c::NodeCore, memo) =
    get!(() -> c.op in (NodeOp.POSITION, NodeOp.UV) || any(i -> _reads_coordinates(i, memo), _input_cores(c)), memo, c)

function _shifted(c::NodeCore, axis::Symbol, fw::Float32, memo, reads)
    _reads_coordinates(c, reads) || return c
    return get!(memo, c) do
        if c.op == NodeOp.POSITION
            NodeCore(axis === :dx ? NodeOp.POSITION_DX : NodeOp.POSITION_DY, c.mode, Any[fw], c.widths, nothing)
        elseif c.op == NodeOp.UV
            NodeCore(axis === :dx ? NodeOp.UV_DX : NodeOp.UV_DY, c.mode, Any[fw], c.widths, nothing)
        else
            inputs = Any[x isa TexNode ? TexNode(_shifted(getfield(x, :core), axis, fw, memo, reads), getfield(x, :output)) : x
                         for x in c.inputs]
            NodeCore(c.op, c.mode, inputs, c.widths, c.data)
        end
    end
end
