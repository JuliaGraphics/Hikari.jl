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

"""Whether a shader scatters below its surface: a Principled BSDF with subsurface weight and no thin wall."""
has_subsurface(g::ShaderGraph) =
    any(c -> c.op == NodeOp.PRINCIPLED && !_constant(c.inputs[9], 0) && !_constant(c.inputs[34], 1), g.closures)

_zero(x) = _constant(x, 0)
_one(x) = _constant(x, 1)

"""The most closures one closure node of a shader can produce at a hit."""
function _closures_of(c::NodeCore)
    c.op == NodeOp.CLOSURE && return c.mode[1] in (ClosureKind.TRANSPARENT, ClosureKind.EMISSION) ? 0 : 1
    i = c.inputs
    thin_wall = !_zero(i[34])
    sheen = _zero(i[27]) ? 0 : 1
    coat = _zero(i[22]) ? 0 : 1
    metal = _zero(i[3]) ? 0 : 1
    _one(i[3]) && return sheen + coat + metal
    transmission = _zero(i[19]) ? 0 : thin_wall ? 2 : 1
    subsurface = _zero(i[9]) ? 0 : thin_wall ? 2 : 1
    return sheen + coat + metal + transmission + 1 + subsurface + 1
end

"""How many closures a shader can produce at a hit: what its closure list holds room for."""
closure_capacity(g::ShaderGraph) = clamp(sum(_closures_of, g.closures; init = 0), 1, MAX_CLOSURES)

function _store_shader(dhv, g::ShaderGraph)
    store_lookup_tables!(dhv)
    program = compile_shader(g.closures, data -> (ref = Raycore.store_texture(dhv, data); (texref_slot(ref), ref.idx)))
    return program, Raycore.store_texture(dhv, program)
end

function device_param(dhv, g::ShaderGraph)
    _, ref = _store_shader(dhv, g)
    return TexHandle(TexKind.NODES, 0f0, _TH_ZERO, Int32(texref_slot(ref)), Int32(ref.idx))
end

"""
A shader's program on the device: its index among the stored programs of its type, and in its type
the features it runs, so a kernel finds the program without branching over every type stored.
"""
struct ShaderProgram{F}
    idx::Int32
end

function shader_program(dhv, g::ShaderGraph)
    program, ref = _store_shader(dhv, g)
    return _shader_program(program, Int32(ref.idx))
end
_shader_program(::AbstractVector{NodeInstr{F}}, idx::Int32) where {F} = ShaderProgram{F}(idx)

"""The programs stored with the features of `p`, found by their element type."""
@generated function shader_programs(textures::Raycore.StaticMultiTypeSet{D, Tex}, ::ShaderProgram{F}) where {D, Tex, F}
    slot = findfirst(T -> eltype(eltype(T)) === NodeInstr{F}, Tex.parameters)
    return :(@inbounds textures.textures[$slot])
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
