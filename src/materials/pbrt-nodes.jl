# pbrt models in the same value program and BSDF program as the Cycles nodes.
# The graph supplies parameter readers to the existing scattering implementations;
# spectra remain spectra, and layered BSDFs retain their random-walk estimator.

struct GraphParameter{J} end
struct GraphSpectrum{N, Offset} end

struct PbrtBSDF{Model, Schema, Arguments, Data, Alpha, UnitWeight, Frame} <: ShaderBSDF end
struct PbrtNodeData{T}
    values::Vector{Float32}
end
inline_node_data(d::PbrtNodeData) = d.values
inline_before_colors(::PbrtNodeData) = true
function bsdf_node_data(::PbrtNodeData{T}, c::NodeCore) where {T}
    M, S, A, D, Alpha = T.parameters[1:5]
    Frame = T.parameters[7]
    return PbrtBSDF{M, S, A, D, Alpha, _one(c.inputs[1]), Frame}
end
instruction_count(::Type{<:PbrtBSDF{M, S, A, D}}) where {M, S, A, D} = cld(A, NODE_ARGS) + cld(D, NODE_ARGS)

const PbrtGraphMaterial = Union{
    Dielectric, ThinDielectric, Conductor, DiffuseTransmission,
    CoatedDiffuse, CoatedConductor, CoatedDiffuseTransmission,
}

# RGB and alpha must share the same image core so updates replace one image.
function graph_color_alpha(x)
    input = pbrt_node_input(x)
    if !(x isa TexNode) && input isa TexNode && getfield(input, :core).op in (NodeOp.IMAGE, NodeOp.PARAMETER)
        return input, cutout(x) ? TexNode(getfield(input, :core), 2) : 1.0f0
    end
    return input, graph_constant_alpha(x)
end
graph_constant_alpha(x) = 1.0f0
graph_constant_alpha(x::Colorant) = Float32(alpha(x))
graph_constant_alpha(x::RGBSpectrum) = get_alpha(x)
graph_constant_alpha(x::Texture{T, 0}) where {T} = graph_constant_alpha(constant_value(x))
graph_constant_alpha(x::TexHandle) = x.kind == TexKind.CONST_SPECTRUM ? get_alpha(x.rgb) : 1.0f0

function pbrt_shader_node(m::Material)
    inputs = Any[]
    data = Float32[]
    parameters = IdDict{Any, Any}()
    scalars = IdDict{Any, Any}()
    schema = map(fieldnames(typeof(m))) do field
        x = getfield(m, field)
        if field in (:normal, :displacement)
            (:nothing,)
        elseif !pbrt_isset(x)
            (:nothing,)
        elseif x isa PiecewiseLinearSpectrum
            offset = length(data)
            append!(data, x.lambdas)
            append!(data, x.values)
            (:spectrum, length(x.lambdas), offset)
        elseif field in (:roughness, :interface_roughness, :conductor_roughness)
            u, v = roughness_pair(x)
            a = pbrt_parameter!(inputs, parameters, scalars, u, true)
            b = pbrt_parameter!(inputs, parameters, scalars, v, true)
            (:pair, a, b)
        elseif field in (:remap_roughness, :max_depth, :n_samples)
            push!(data, x isa Bool ? Float32(x) : x isa Integer ? reinterpret(Float32, Int32(x)) : Float32(x))
            (x isa Bool ? :bool : x isa Integer ? :int : :float, length(data))
        else
            (:input, pbrt_parameter!(inputs, parameters, scalars, x, pbrt_scalar_field(m, field)))
        end
    end
    alpha_index = 0
    if m isa Union{Diffuse, CoatedDiffuse}
        field = m isa Diffuse ? :color : :reflectance
        color, coverage = graph_color_alpha(getfield(m, field))
        color_index = schema[findfirst(==(field), fieldnames(typeof(m)))][2] - 1
        inputs[color_index] = color
        push!(inputs, coverage)
        alpha_index = length(inputs) + 1
    end
    model = Base.typename(typeof(m)).wrapper
    frame = 0
    if pbrt_isset(m.displacement)
        hasfield(typeof(m), :normal) && !isnothing(m.normal) &&
            throw(ArgumentError("combine normal and displacement with an explicit normal graph"))
        bump = pbrt_bump_node(pbrt_node_input(m.displacement))
        frame = length(inputs) + 2
        push!(inputs, bump, TexNode(getfield(bump, :core), 2))
    elseif hasfield(typeof(m), :normal) && !isnothing(m.normal)
        frame = length(inputs) + 2
        push!(inputs, normal_input(m.normal), ShadingTangent())
    end
    if frame != 0
        normal = inputs[frame - 1]
        memo = IdDict{NodeCore, NodeCore}()
        for j in 1:(frame - 2)
            j + 1 == alpha_index && continue # Alpha tests run before bump mapping.
            inputs[j] = pbrt_parameter_normal(inputs[j], normal, memo)
        end
    end
    T = PbrtBSDF{model, schema, length(inputs) + 1, length(data), alpha_index, false, frame}
    return BSDFNode(NodeOp.CLOSURE, (ClosureKind.PBRT, 0x00, 0x00), inputs, PbrtNodeData{T}(data))
end
shader_node(m::PbrtGraphMaterial) = pbrt_shader_node(m)
nests(::PbrtGraphMaterial) = true
surface_material(m::PbrtGraphMaterial) = ShaderMaterial(shader_node(m))
pbrt_isset(x) = true
pbrt_isset(::Nothing) = false
pbrt_isset(x::TexHandle) = !is_none(x)

struct GraphParameterContext{B, S, P, C}
    stack::S
    program::P
    instruction::Int32
    shading::C
    regularize::Bool
end
@inline graph_value(c::GraphParameterContext{B}, ::GraphParameter{J}) where {B, J} =
    node_v(c.stack, c.program, c.instruction, J, instruction_count(B), c.shading)
@inline function graph_datum(c::GraphParameterContext{<:PbrtBSDF{M, S, A}}, offset::Int) where {M, S, A}
    ins = @inbounds c.program[c.instruction + Int32(cld(A, NODE_ARGS) + (offset - 1) ÷ NODE_ARGS)]
    return _kat(ins.k, Int32((offset - 1) % NODE_ARGS + 1))
end

# The schema contains layout alone. Values and spectra live in the program and
# may change without changing the BSDF kernel's type.
@generated function graph_material(c::GraphParameterContext{B}) where {B}
    Model, schema = B.parameters[1:2]
    fields = map(schema) do item
        kind = item[1]
        kind === :nothing && return :(nothing)
        kind === :input && return :(GraphParameter{$(item[2])}())
        kind === :pair && return :((GraphParameter{$(item[2])}(), GraphParameter{$(item[3])}()))
        kind === :spectrum && return :(GraphSpectrum{$(item[2]), $(item[3])}())
        value = :(graph_datum(c, $(item[2])))
        kind === :bool && return :($value != 0.0f0)
        kind === :int && return :(reinterpret(Int32, $value))
        return value
    end
    return :($Model($(fields...)))
end

@inline eval_handle(c::GraphParameterContext, p::GraphParameter, tfc) = graph_value(c, p)[1]
@inline function eval_handle_spectrum(c::GraphParameterContext, p::GraphParameter, tfc)
    v = graph_value(c, p)
    return RGBSpectrum(v[1], v[2], v[3])
end
@inline eval_handle_spectrum(c::GraphParameterContext, ::Nothing, tfc) = RGBSpectrum(0.0f0)
@inline eval_dielectric_ior(c::GraphParameterContext, p::GraphParameter, tfc, lambda) =
    (eval_handle(c, p, tfc), false)
@inline eval_ior_spectral(table, c::GraphParameterContext, p::GraphParameter, tfc, lambda) =
    uplift_rgb_unbounded(table, eval_handle_spectrum(c, p, tfc), lambda)
@inline conductor_ior(table, c::GraphParameterContext, eta, k, ::Nothing, tfc, lambda) =
    (eval_ior_spectral(table, c, eta, tfc, lambda), eval_ior_spectral(table, c, k, tfc, lambda))
@inline conductor_ior(table, c::GraphParameterContext, eta, k, r::GraphParameter, tfc, lambda) =
    reflectance_ior(table, eval_handle_spectrum(c, r, tfc), lambda)

# Identical endpoint handling, binary search and interpolation to
# PiecewiseLinearSpectrum, reading its data directly rather than copying a curve
# into every hit's local storage.
@inline function graph_spectrum(c, ::GraphSpectrum{N, O}, lambda::Float32) where {N, O}
    lambda <= graph_datum(c, O + 1) && return graph_datum(c, O + N + 1)
    lambda >= graph_datum(c, O + N) && return graph_datum(c, O + 2N)
    lo, hi = 1, N
    while lo + 1 < hi
        mid = (lo + hi) >>> 1
        if graph_datum(c, O + mid) <= lambda
            lo = mid
        else
            hi = mid
        end
    end
    t = (lambda - graph_datum(c, O + lo)) / (graph_datum(c, O + hi) - graph_datum(c, O + lo))
    return graph_datum(c, O + N + lo) * (1.0f0 - t) + graph_datum(c, O + N + hi) * t
end
@inline eval_dielectric_ior(c::GraphParameterContext, p::GraphSpectrum, tfc, lambda) =
    (graph_spectrum(c, p, lambda.lambda[1]), true)
@inline eval_ior_spectral(table, c::GraphParameterContext, p::GraphSpectrum, tfc, lambda) =
    SpectralRadiance(ntuple(i -> graph_spectrum(c, p, lambda.lambda[i]), 4))

struct PbrtTerm{B, C}
    context::C
    weight::Float32
    coverage::Float32
end
@inline function node_lobes(::Type{B}, s, prog, i::Int32, sc::ShaderContext, lut) where {B <: PbrtBSDF}
    c = GraphParameterContext{B, typeof(s), typeof(prog), typeof(sc)}(s, prog, i, sc, lut.regularize)
    weight = node_f(s, prog, i, 1, instruction_count(B), sc)
    alpha_index = B.parameters[5]
    coverage = alpha_index == 0 ? 1.0f0 : clamp(node_f(s, prog, i, alpha_index, instruction_count(B), sc), 0.0f0, 1.0f0)
    return ((PbrtTerm{B, typeof(c)}(c, weight, coverage),), Vec3f(weight * (1.0f0 - coverage)), Vec3f(0))
end
@inline function pbrt_term_bsdf(w::PbrtTerm, table, lambda)
    c = w.context
    sc = c.shading
    n, t = pbrt_graph_frame(c)
    return get_bxdf(graph_material(c), table, c, sc.tfc, lambda, c.regularize, sc.wo, n, t)
end
@inline function lift_shader_term(w::PbrtTerm, scale, table, lambda)
    b = pbrt_term_bsdf(w, table, lambda)
    factor = w.weight * w.coverage * scale
    sampling = b.total > 0.0f0 ? w.weight * w.coverage / b.total : 0.0f0
    return map(v -> Weighted(v.lobe, v.weight * factor, v.sample_weight * sampling, v.n, v.t), b.lobes)
end
@inline function lobe_bsdf(::Type{Tuple{B}}, (terms, transparency, emission), table, lambda::Wavelengths) where {B <: PbrtBSDF}
    # A single unit-weight node is exactly the original BSDF, including its
    # sampling probabilities and layered/proportional PDF semantics.
    B.parameters[6] && return pbrt_term_bsdf(first(terms), table, lambda)
    return pbrt_graph_bsdf(Tuple{B}, terms, transparency, emission, table, lambda)
end
@inline function pbrt_graph_bsdf(::Type{M}, terms, transparency, emission, table, lambda) where {M}
    selects_component(M) && return component_bsdf(M, terms, transparency, emission, table, lambda)
    alpha = 1.0f0 - _avg(transparency)
    scale = alpha > 1.0f-6 ? inv(alpha) : 0.0f0
    lobes = lift_shader_terms(terms, scale, table, lambda)
    total = reduce(+, map(w -> w.sample_weight, lobes); init = 0.0f0)
    return LobeBSDF(lobes, total, lambda, SpectralRadiance(0.0f0), disperses(lobes))
end

node_has_transparency(::PbrtNodeData{T}, c::NodeCore) where {T} =
    T.parameters[5] != 0 && !_one(c.inputs[T.parameters[5]])
node_transparency(::PbrtNodeData{T}, c::NodeCore) where {T} = 1 - c.inputs[T.parameters[5]]

@inline function shader_guide(w::PbrtTerm, depth)
    c = w.context
    value = guide_material(graph_material(c), c, c.shading.tfc, depth)
    return w.weight * Vec3f(value.r, value.g, value.b)
end

# A layered random walk supplies a proportional sampling PDF. Selecting the
# entire BSDF at the vertex preserves that estimator for both direct lighting
# and continuation. Dispersive pbrt nodes likewise keep wavelength termination
# attached to the selected model instead of terminating unrelated graph branches.
function selects_component(::Type{B}) where {M, S, B <: PbrtBSDF{M, S}}
    B.parameters[7] != 0 && return true
    M <: Union{CoatedDiffuse, CoatedConductor, CoatedDiffuseTransmission} && return true
    M <: Union{Dielectric, ThinDielectric} || return false
    return first(S)[1] === :spectrum
end

struct BSDFComponent{B, F}
    bsdf::B
    factor::Float32
    sample_weight::Float32
    frame::F
end
struct ComponentBSDF{C}
    components::C
    selected::Int32
    scale::Float32
    emission::SpectralRadiance
end
@inline function prepare_component(w::Weighted, scale, table, lambda)
    v = lift(w, scale, table, lambda)
    b = LobeBSDF((v,), v.sample_weight, lambda, SpectralRadiance(0.0f0), disperses((v,)))
    return BSDFComponent(b, 1.0f0, v.sample_weight, nothing)
end
@inline component_has_samples(b::LobeBSDF) = b.total > 0.0f0
@inline component_has_samples(b::LayeredBSDF) = true
@inline function prepare_component(w::PbrtTerm{B}, scale, table, lambda) where {B}
    b = pbrt_term_bsdf(w, table, lambda)
    weight = component_has_samples(b) ? max(w.weight * w.coverage, 0.0f0) : 0.0f0
    frame = B.parameters[7] == 0 ? nothing : pbrt_graph_frame(w.context)
    return BSDFComponent(b, w.weight * w.coverage * scale, weight, frame)
end
@inline function component_choice(parts::Tuple, target::Float32, index::Int32 = Int32(1))
    w = first(parts).sample_weight
    (target < w || length(parts) == 1) && return (index, w, first(parts).factor)
    return component_choice(Base.tail(parts), target - w, index + Int32(1))
end
@inline function shader_emission(::Type{M}, emission, table, lambda) where {M}
    return emits(M) && _max(emission) > 0.0f0 ?
        uplift_rgb_illuminant(table, RGBSpectrum(emission[1], emission[2], emission[3]), lambda) / D65_PHOTOMETRIC :
        SpectralRadiance(0.0f0)
end
@inline function component_bsdf(::Type{M}, terms, transparency, emission, table, lambda) where {M}
    alpha = 1.0f0 - _avg(transparency)
    scale = alpha > 1.0f-6 ? inv(alpha) : 0.0f0
    parts = map(w -> prepare_component(w, scale, table, lambda), terms)
    total = reduce(+, map(w -> w.sample_weight, parts); init = 0.0f0)
    c = first_native_context(terms)
    u = mix_hash_float(c.shading.tfc.p, c.shading.wo, SetKey(1, 1), SetKey(1, 2))
    selected, weight, factor = component_choice(parts, u * total)
    return ComponentBSDF(
        parts, selected, weight > 0.0f0 ? factor * total / weight : 0.0f0,
        shader_emission(M, emission, table, lambda)
    )
end
@inline first_native_context(terms::Tuple{<:PbrtTerm, Vararg}) = first(terms).context
@inline first_native_context(terms::Tuple) = first_native_context(Base.tail(terms))
@inline function selected_component(f, parts::Tuple, index::Int32, args...)
    (index == Int32(1) || length(parts) == 1) && return f(first(parts).bsdf, args...)
    return selected_component(f, Base.tail(parts), index - Int32(1), args...)
end
@inline terminates_at_vertex(b::ComponentBSDF) =
    b.scale > 0.0f0 && selected_component(terminates_at_vertex, b.components, b.selected)
@inline function evaluate_bsdf_spectral(b::ComponentBSDF, args...)
    b.scale > 0.0f0 || return (SpectralRadiance(0.0f0), 0.0f0)
    f, pdf = selected_component(evaluate_bsdf_spectral, b.components, b.selected, args...)
    return f * b.scale, pdf
end
@inline function sample_bsdf_spectral(b::ComponentBSDF, args...)
    b.scale > 0.0f0 || return SpectralBSDFSample()
    s = selected_component(sample_bsdf_spectral, b.components, b.selected, args...)
    return SpectralBSDFSample(s.f * b.scale, s.wi, s.pdf, s.flags, s.eta, s.pdf_is_proportional, s.secondary_terminated)
end
@propagate_inbounds function add_emission!(pixel_L, bsdf::ComponentBSDF, work)
    is_black(bsdf.emission) && return nothing
    L = work.beta * bsdf.emission / average(work.r_u)
    accumulate_spectrum!(pixel_L, (work.pixel_index - Int32(1)) * Int32(4), L)
    return nothing
end
@inline function component_subsurface(parts::Tuple, index::Int32, work, accel, sample_idx)
    part = first(parts)
    if index == Int32(1)
        bsdf, updated = subsurface_scatter(part.bsdf, work, accel, sample_idx)
        return ((BSDFComponent(bsdf, part.factor, part.sample_weight, part.frame), Base.tail(parts)...), updated)
    end
    rest, updated = component_subsurface(Base.tail(parts), index - Int32(1), work, accel, sample_idx)
    return ((part, rest...), updated)
end
@inline component_subsurface(parts::Tuple{}, index::Int32, work, accel, sample_idx) = ((), work)
@inline function subsurface_scatter(b::ComponentBSDF, work, accel, sample_idx::Int32)
    parts, updated = component_subsurface(b.components, b.selected, work, accel, sample_idx)
    return (ComponentBSDF(parts, b.selected, b.scale, b.emission), updated)
end

pbrt_node_input(x) = node_input(x)
pbrt_node_input(x::Colorant) = TextureNodes.constant((Float32(red(x)), Float32(green(x)), Float32(blue(x))))
pbrt_node_input(x::Texture{T, 0}) where {T} = pbrt_node_input(constant_value(x))

@inline function pbrt_graph_frame(c::GraphParameterContext{B}) where {B}
    frame = B.parameters[7]
    sc = c.shading
    frame == 0 && return sc.tfc.n, sc.dpdus
    chunks = instruction_count(B)
    return node_v(c.stack, c.program, c.instruction, frame, chunks, sc),
        node_v(c.stack, c.program, c.instruction, frame + 1, chunks, sc)
end

struct PbrtBumpGeometry{F}
    dpdu::Vec3f
    dndu::Vec3f
    dndv::Vec3f
    filter::F
end
PbrtBumpGeometry(dpdu, dndu, dndv) = PbrtBumpGeometry(dpdu, dndu, dndv, nothing)
@inline pbrt_height_context(sc) = pbrt_height_context(sc.bump.filter, sc.tfc)
@inline pbrt_height_context(::Nothing, fallback) = fallback
@inline pbrt_height_context(filter::TextureFilterContext, fallback) = filter
@inline function pbrt_bump_steps(tfc)
    du = 0.5f0 * (abs(tfc.dudx) + abs(tfc.dudy))
    dv = 0.5f0 * (abs(tfc.dvdx) + abs(tfc.dvdy))
    return du == 0.0f0 ? 5.0f-4 : du, dv == 0.0f0 ? 5.0f-4 : dv
end
@inline function pbrt_bump_tangents(sc)
    ts = cross(sc.tfc.n, sc.bump.dpdu)
    return dot(ts, ts) > 1.0f-16 ? (cross(ts, sc.tfc.n), ts) : coordinate_system(sc.tfc.n)
end
@inline function pbrt_shifted_leaf(op, sc)
    tfc = pbrt_height_context(sc)
    du, dv = pbrt_bump_steps(tfc)
    ss, ts = pbrt_bump_tangents(sc)
    op == NodeOp.POSITION_U && return Vec3f(tfc.p) + du * ss
    op == NodeOp.POSITION_V && return Vec3f(tfc.p) + dv * ts
    op == NodeOp.UV_U && return Vec3f(tfc.uv[1] + du, tfc.uv[2], 0)
    return Vec3f(tfc.uv[1], tfc.uv[2] + dv, 0)
end
function pbrt_bump_node(height)
    h = scalar_graph_input(height isa TexNode ? height : TextureNodes.constant(height))
    return TexNode(
        NodeCore(
            NodeOp.PBRT_BUMP, (0x00, 0x00, 0x00),
            Any[pbrt_height_base(h), shifted_graph(h, :pbrt_u, 1.0f0), shifted_graph(h, :pbrt_v, 1.0f0)], [3, 3], nothing
        ), 1
    )
end
@inline function graph_bump_frame(h0, hu, hv, sc)
    du, dv = pbrt_bump_steps(pbrt_height_context(sc))
    ss, ts = pbrt_bump_tangents(sc)
    n = sc.tfc.n
    dpu = ss + ((hu - h0) / du) * n + h0 * sc.bump.dndu
    dpv = ts + ((hv - h0) / dv) * n + h0 * sc.bump.dndv
    raw = cross(dpu, dpv)
    l = sqrt(dot(raw, raw))
    result = l > 1.0f-10 ? raw / l : n
    return dot(result, sc.ng) < 0.0f0 ? -result : result, dpu
end

shader_geometry(::ShaderProgram{F}, geom, primitive, filter = () -> nothing) where {F} =
    F & NodeFeature.PBRT_BUMP != 0 ? PbrtBumpGeometry(geom.dpdu, vp_compute_normal_derivatives(primitive)..., filter()) : nothing
shader_geometry(::Type, geom, primitive, filter = () -> nothing) = nothing
shader_geometry(::Type{<:ShaderMaterial{P}}, geom, primitive, filter = () -> nothing) where {P <: Union{ShaderProgram, FusedShaderProgram}} =
    shader_geometry(P(Int32(0)), geom, primitive, filter)
shader_geometry(p::FusedShaderProgram{F}, geom, primitive, filter = () -> nothing) where {F} =
    shader_geometry(ShaderProgram{F}(p.idx), geom, primitive, filter)

# Existing pbrt texture objects are graph leaves. Their filtering and mesh
# interpolation stay in their existing evaluators; their outputs can feed any
# material or value node in the graph.
struct GraphTextureDatum{T}
    value::T
end
parameter_node(value) = TexNode(NodeCore(NodeOp.PARAMETER, (0x00, 0x00, 0x00), Any[], [3, 1], GraphTextureDatum(value)), 1)
function place_datum!(dhv, datum::GraphTextureDatum, over, placed)
    if over !== nothing && eltype(dhv.texture_order[over[1]]) === TexHandle && !(over in values(placed))
        old = only(Array(stored_array(dhv, over)))
        handle = Raycore.update_item(dhv, old, datum.value)
        return copied!(dhv, over, [handle])
    end
    return stored!(dhv, [device_param(dhv, datum.value)])
end
@inline function graph_texture_context(ins, sc)
    tfc = sc.tfc
    op = ins.op
    _has(ins, NodeFeature.PBRT_BUMP) && op == NodeOp.PARAMETER_BUMP && return pbrt_height_context(sc)
    if _has(ins, NodeFeature.PBRT_BUMP) && (op == NodeOp.PARAMETER_U || op == NodeOp.PARAMETER_V)
        tfc = pbrt_height_context(sc)
        du, dv = pbrt_bump_steps(tfc)
        ss, ts = pbrt_bump_tangents(sc)
        return op == NodeOp.PARAMETER_U ? shifted(tfc, Point2f(tfc.uv[1] + du, tfc.uv[2]), tfc.p + du * ss) :
            shifted(tfc, Point2f(tfc.uv[1], tfc.uv[2] + dv), tfc.p + dv * ts)
    elseif _has(ins, NodeFeature.BUMP) && (op == NodeOp.PARAMETER_DX || op == NodeOp.PARAMETER_DY)
        dx, dy = _bump_differentials(sc)
        fw = ins.k[1]
        return op == NodeOp.PARAMETER_DX ?
            shifted(tfc, Point2f(tfc.uv[1] + fw * tfc.dudx, tfc.uv[2] + fw * tfc.dvdx), tfc.p + fw * dx) :
            shifted(tfc, Point2f(tfc.uv[1] + fw * tfc.dudy, tfc.uv[2] + fw * tfc.dvdy), tfc.p + fw * dy)
    end
    return tfc
end
@inline function graph_texture_parameter(textures, ins, sc, stack)
    tfc = graph_texture_context(ins, sc)
    if ins.mode[1] == 0x01
        normal = _v(stack, ins, 1)
        tfc = TextureFilterContext(
            tfc.uv, tfc.dudx, tfc.dudy, tfc.dvdx, tfc.dvdy,
            tfc.face_idx, tfc.bary, tfc.p, normal
        )
    end
    return Raycore.with_texture(
        graph_parameter_value, textures, reinterpret(Int32, ins.k[9]),
        reinterpret(Int32, ins.k[10]), textures, tfc
    )
end
@inline graph_parameter_value(data::AbstractVector{TexHandle}, textures, tfc) =
    eval_handle_spectrum(textures, @inbounds(data[1]), tfc)
@inline graph_parameter_value(data, textures, tfc) = RGBSpectrum(0.0f0)

has_pbrt_nodes(::Type{<:PbrtBSDF}) = true

# A single pbrt node with displacement supplies the vertex's frame, as the
# original material did. Keep this information outside the scattering BSDF so
# the integrator's hemisphere tests, continuation flags and cosines agree too.
struct FramedShaderBSDF{B}
    bsdf::B
    normal::Vec3f
    tangent::Vec3f
end
@inline function material_bsdf(::Type{Tuple{B}}, produced, table, lambda) where {B <: PbrtBSDF}
    bsdf = lobe_bsdf(Tuple{B}, produced, table, lambda)
    B.parameters[6] && B.parameters[7] != 0 || return bsdf
    normal, tangent = pbrt_graph_frame(first(first(produced)).context)
    return FramedShaderBSDF(bsdf, normal, tangent)
end
@inline material_vertex(bsdf, work) = (bsdf, work)
@inline material_vertex(bsdf::FramedShaderBSDF, work) =
    (bsdf.bsdf, with_shading_frame(work, bsdf.normal, bsdf.tangent))
@generated function with_shading_frame(work::W, normal::Vec3f, tangent::Vec3f) where {W}
    fields = map(fieldnames(W)) do f
        f === :ns && return :(normal)
        f === :dpdus && return :(tangent)
        f === :dpdvs && return :(cross(normal, tangent))
        return :(getfield(work, $(QuoteNode(f))))
    end
    return :($W($(fields...)))
end
@inline apply_component_frame(work, ::Nothing) = work
@inline apply_component_frame(work, frame::Tuple{Vec3f, Vec3f}) = with_shading_frame(work, frame...)
@inline function selected_frame(parts::Tuple, index::Int32, work)
    (index == Int32(1) || length(parts) == 1) && return apply_component_frame(work, first(parts).frame)
    return selected_frame(Base.tail(parts), index - Int32(1), work)
end
@inline material_vertex(bsdf::ComponentBSDF, work) = (bsdf, selected_frame(bsdf.components, bsdf.selected, work))

# Only pbrt's legacy filtered texture leaves need a distinct unshifted height
# context. Value graphs are unfiltered, and their position and UV are unchanged.
function pbrt_height_base(height::TexNode)
    memo = IdDict{NodeCore, NodeCore}()
    function rebuild(c)
        return get!(memo, c) do
            inputs = Any[x isa TexNode ? TexNode(rebuild(getfield(x, :core)), getfield(x, :output)) : x for x in c.inputs]
            NodeCore(c.op == NodeOp.PARAMETER ? NodeOp.PARAMETER_BUMP : c.op, c.mode, inputs, c.widths, c.data)
        end
    end
    return TexNode(rebuild(getfield(height, :core)), getfield(height, :output))
end

# Texture leaves and elementary arithmetic share one small interpreter. More
# involved node families keep their values pass instead of enlarging every
# native BSDF's fused kernel.
const NATIVE_SMALL_FEATURES = NodeFeature.IMAGE | NodeFeature.BUMP

small_native_program(c::CompiledProgram{F}) where {F} =
    F & ~(NATIVE_SMALL_FEATURES | NodeFeature.PBRT_BUMP | (UInt32(3) << STACK_SHIFT)) == 0 &&
    all(small_parameter_tree, c.data)
function canonical_node_program(::PbrtNodeData, c::CompiledProgram{F}) where {F}
    small_native_program(c) || return c
    features = NodeFeature.IMAGE | (F & (NodeFeature.BUMP | NodeFeature.PBRT_BUMP | (UInt32(3) << STACK_SHIFT)))
    instructions = [NodeInstr{features}(ins.op, ins.mode, ins.out, ins.outwide, ins.arg, ins.wide, ins.k) for ins in c.instructions]
    return CompiledProgram(instructions, c.data)
end
function device_node_program(::PbrtNodeData, c::CompiledProgram, idx)
    computed = first(c.instructions).op != NodeOp.CLOSURE
    return computed && small_native_program(c) ? FusedShaderProgram(c, idx) : ShaderProgram(c, idx)
end

pbrt_scalar_field(m, field) = endswith(string(field), "roughness") || field in (:thickness, :g, :scale, :interface_ior) ||
    (field === :ior && !(m isa Conductor))

scalar_graph_input(x) = x
scalar_graph_input(x::TexNode) = _width(x) == 3 ? x.x : x

# pbrt evaluates the other material inputs after applying displacement. Bind
# normal reads in those inputs to the graph's displaced normal; height itself
# and the pre-intersection alpha query retain their original context.
pbrt_parameter_normal(x, normal, memo) = x
function pbrt_parameter_normal(x::TexNode, normal, memo)
    c = getfield(x, :core)
    c.op == NodeOp.NORMAL && return normal
    core = get!(memo, c) do
        inputs = Any[pbrt_parameter_normal(v, normal, memo) for v in c.inputs]
        mode = c.mode
        if c.op == NodeOp.PARAMETER
            inputs = Any[normal]
            mode = (0x01, mode[2], mode[3])
        end
        NodeCore(c.op, mode, inputs, c.widths, c.data)
    end
    return TexNode(core, getfield(x, :output))
end

small_parameter_tree(x) = true
small_parameter_tree(x::GraphTextureDatum) = small_parameter_tree(x.value)
small_parameter_tree(::TexNode) = false
small_parameter_tree(h::TexHandle) = !(h.kind in (TexKind.NODES, TexKind.CHECKER))
small_parameter_tree(c::CheckerboardTexture) = small_parameter_tree(c.tex1) && small_parameter_tree(c.tex2)

function pbrt_parameter!(inputs, parameters, scalars, value, scalar::Bool)
    base = get!(() -> pbrt_node_input(value), parameters, value)
    input = scalar ? get!(() -> scalar_graph_input(base), scalars, value) : base
    push!(inputs, input)
    return length(inputs) + 1
end
