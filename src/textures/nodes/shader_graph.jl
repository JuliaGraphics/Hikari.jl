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
function _weighted(s::MixShaderNode, w)
    fac = clamp(s.fac, 0f0, 1f0)
    return vcat(_weighted(s.a, w * (1 - fac)), _weighted(s.b, w * fac))
end
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

"""Whether a shader lets light through unrefracted: a transparent closure, or a Principled BSDF whose alpha is not one."""
has_transparency(g::ShaderGraph) =
    any(c -> (c.op == NodeOp.CLOSURE && c.mode[1] == ClosureKind.TRANSPARENT) ||
             (c.op == NodeOp.PRINCIPLED && !_constant(c.inputs[6], 1)), g.closures)

"""The types of a shader's BSDF nodes, in program order, as its material's type holds them."""
bsdf_nodes(g::ShaderGraph) = Tuple{map(bsdf_node, g.closures)...}

function bsdf_node(c::NodeCore)
    c.op == NodeOp.CLOSURE || return principled_node(c)
    c.mode[1] == ClosureKind.SUBSURFACE && _zero(c.inputs[3]) && return SubsurfaceBSDF{false}
    node = closure_node(c.mode[1])
    c.mode[2] == 0x00 && return node
    convention = c.mode[2] == 0x02 ? CyclesBeckmann : CyclesGGX
    return SingleScatterBSDF{node, convention}
end
function closure_node(kind::UInt8)
    kind == ClosureKind.DIFFUSE && return DiffuseBSDF
    kind == ClosureKind.TRANSLUCENT && return TranslucentBSDF
    kind == ClosureKind.GLOSSY && return GlossyBSDF
    kind == ClosureKind.METALLIC && return MetallicBSDF
    kind == ClosureKind.METALLIC_FILM && return MetallicFilmBSDF
    kind == ClosureKind.REFRACTION_BECKMANN && return RefractionBSDF{CyclesBeckmann}
    kind == ClosureKind.REFRACTION_GGX && return RefractionBSDF{CyclesGGX}
    kind == ClosureKind.GLASS && return GlassBSDF
    kind == ClosureKind.GLASS_FILM && return GlassFilmBSDF
    kind == ClosureKind.SHEEN && return SheenBSDF
    kind == ClosureKind.TOON_DIFFUSE && return ToonBSDF{false}
    kind == ClosureKind.TOON_GLOSSY && return ToonBSDF{true}
    kind == ClosureKind.SUBSURFACE && return SubsurfaceBSDF{true}
    kind == ClosureKind.TRANSPARENT && return TransparentBSDF
    kind == ClosureKind.EMISSION && return EmissionBSDF
    error("no BSDF node of closure kind $kind")
end
function principled_node(c::NodeCore)
    thin_wall = c.inputs[34]
    thin_wall isa Real || error("a Principled BSDF's thin wall is a constant, not a node graph: it decides which lobes the material has")
    return PrincipledBSDF{Float32(thin_wall) > 0.5f0, !_zero(c.inputs[9]), !_zero(c.inputs[31])}
end

_zero(x) = _constant(x, 0)
_one(x) = _constant(x, 1)

"""
The transparency of `g` alone, for alpha tests: its transparent closures, and each Principled
BSDF whose alpha is not one as the transparent closure its alpha makes of it. Its program reads
only what the transparency depends on, where the whole shader's runs every closure.
"""
function transparency_graph(g::ShaderGraph)
    cores = NodeCore[]
    for c in g.closures
        if c.op == NodeOp.CLOSURE && c.mode[1] == ClosureKind.TRANSPARENT
            push!(cores, c)
        elseif c.op == NodeOp.PRINCIPLED && !_constant(c.inputs[6], 1)
            push!(cores, NodeCore(NodeOp.CLOSURE, (ClosureKind.TRANSPARENT, 0x00, 0x00),
                                  Any[c.inputs[1], 1 - c.inputs[6]], Int[], nothing))
        end
    end
    return ShaderGraph(cores)
end

"""
A shader's program on the device: its index among the stored programs of its type, and in its type
the features it runs, so a kernel finds the program without branching over every type stored.
"""
struct ShaderProgram{F}
    idx::Int32
end

ShaderProgram(::CompiledProgram{F}, idx::Integer) where {F} = ShaderProgram{F}(Int32(idx))

"""The program of `g` stored in `dhv`, its data placed with `placed`; no program for no graph."""
function shader_program(dhv, g::ShaderGraph, placed::Places)
    store_lookup_tables!(dhv)
    c = compile_shader(g.closures)
    return ShaderProgram(c, last(stored_program!(dhv, c, placed)))
end
shader_program(dhv, ::Nothing, placed::Places) = nothing

"""The slot of the set `dhv` that holds the programs of `p`'s node families."""
program_slot(dhv::Raycore.MultiTypeSet, ::ShaderProgram{F}) where {F} =
    findfirst(AT -> eltype(AT) === NodeInstr{F}, dhv.texture_order)

"""The programs stored with the features of `p`, found by their element type."""
@generated function shader_programs(textures::Union{Raycore.StaticMultiTypeSet{D, Tex}, Raycore.SetView{<:Any, <:Any, D, Tex}},
                                     ::ShaderProgram{F}) where {D, Tex, F}
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
    get!(() -> c.op in (NodeOp.POSITION, NodeOp.UV, NodeOp.ATTRIBUTE) || any(i -> _reads_coordinates(i, memo), _input_cores(c)),
         memo, c)

function _shifted(c::NodeCore, axis::Symbol, fw::Float32, memo, reads)
    _reads_coordinates(c, reads) || return c
    return get!(memo, c) do
        if c.op == NodeOp.POSITION
            NodeCore(axis === :dx ? NodeOp.POSITION_DX : NodeOp.POSITION_DY, c.mode, Any[fw], c.widths, nothing)
        elseif c.op == NodeOp.UV
            NodeCore(axis === :dx ? NodeOp.UV_DX : NodeOp.UV_DY, c.mode, Any[fw], c.widths, nothing)
        elseif c.op == NodeOp.ATTRIBUTE
            NodeCore(axis === :dx ? NodeOp.ATTRIBUTE_DX : NodeOp.ATTRIBUTE_DY, c.mode, Any[fw], c.widths, c.data)
        else
            inputs = Any[x isa TexNode ? TexNode(_shifted(getfield(x, :core), axis, fw, memo, reads), getfield(x, :output)) : x
                         for x in c.inputs]
            NodeCore(c.op, c.mode, inputs, c.widths, c.data)
        end
    end
end

# ── Mesh attributes ─────────────────────────────────────────────────────────

"""A mesh attribute an attribute node reads, at the corners of the mesh's triangles: see `_corner_values`."""
struct BoundAttribute
    name::Symbol
    corners::Vector{Vec3f}
end

_attribute_name(name::Symbol) = name
_attribute_name(a::BoundAttribute) = a.name

_attribute_cores(g::ShaderGraph) = filter(c -> c.op in _ATTRIBUTE_OPS, _topological(g.closures))

"""
`g` with the attributes it reads taken from the triangle mesh `mesh`: each attribute node holds its
values at the corners of every triangle and the corners' uv, in face order.
"""
function bind_attributes(g::ShaderGraph, mesh::GeometryBasics.Mesh)
    cores = _attribute_cores(g)
    names = unique(_attribute_name(c.data) for c in cores)
    isempty(names) && return g
    corners = Dict(name => _corner_values(mesh, name) for name in names)
    bound = IdDict{NodeCore, NodeCore}()
    rebuild(c::NodeCore) = get!(bound, c) do
        inputs = Any[x isa TexNode ? TexNode(rebuild(getfield(x, :core)), getfield(x, :output)) : x for x in c.inputs]
        data = c.op in _ATTRIBUTE_OPS ? BoundAttribute(_attribute_name(c.data), corners[_attribute_name(c.data)]) : c.data
        NodeCore(c.op, c.mode, inputs, c.widths, data)
    end
    return ShaderGraph(map(rebuild, g.closures))
end

_linear(c::Colorant) = (rgb = RGB{Float32}(c); Vec3f(_srgb_to_linear.((red(rgb), green(rgb), blue(rgb)))...))
_srgb_to_linear(x) = x <= 0.04045f0 ? x / 12.92f0 : ((x + 0.055f0) / 1.055f0)^2.4f0
_as_vec3(x::Colorant) = _linear(x)
_as_vec3(x::Real) = Vec3f(x)
_as_vec3(x::AbstractVector{<:Real}) = Vec3f(ntuple(i -> i <= length(x) ? Float32(x[i]) : 0f0, 3))

"""Per vertex values of attribute `name` of `mesh`, `nothing` where it has none."""
function _vertex_values(mesh::GeometryBasics.Mesh, name::Symbol)
    ps = Point3f.(GeometryBasics.coordinates(mesh))
    if name === :object
        return Vec3f.(ps)
    elseif name === :generated
        lo, hi = reduce((a, b) -> min.(a, b), ps), reduce((a, b) -> max.(a, b), ps)
        size = map(d -> d > 0 ? d : 1f0, hi - lo)
        return [Vec3f((p - lo) ./ size) for p in ps]
    elseif name === :object_normal
        return Vec3f.(GeometryBasics.decompose_normals(mesh))
    end
    hasproperty(mesh, name) || return nothing
    return _as_vec3.(getproperty(mesh, name))
end

"""
The values of `name` at the corners of each triangle of `mesh` and the corners' uv (Hikari's
default uv per triangle where the mesh has none), six per triangle in face order.
"""
function _corner_values(mesh::GeometryBasics.Mesh, name::Symbol)
    triangles = GeometryBasics.faces(mesh)
    values = _vertex_values(mesh, name)
    uvs = hasproperty(mesh, :uv) ? Vec3f.(_as_vec3.(GeometryBasics.texturecoordinates(mesh))) : nothing
    default_uv = (Vec3f(0, 0, 0), Vec3f(1, 0, 0), Vec3f(1, 1, 0))
    corners = Vec3f[]
    for f in triangles
        idx = Int.(GeometryBasics.value.(f))
        append!(corners, isnothing(values) ? (Vec3f(0), Vec3f(0), Vec3f(0)) : values[idx])
        append!(corners, isnothing(uvs) ? default_uv : uvs[idx])
    end
    return corners
end
