# ============================================================================
# Shader programs: node programs that end in closures, as Cycles' SVM does
# ============================================================================
#
# A shader graph's BSDF nodes, Mix Shader and Add Shader are flattened on the
# host into one closure node per BSDF, each with the product of the mix factors
# above it as its weight. The program computes those weights and every input
# like any other value on the stack, and each closure instruction appends what
# it describes to a fixed-size list. The material then samples and evaluates
# the list as one BSDF.

module ClosureKind
const DIFFUSE     = UInt8(1)
const GLOSSY      = UInt8(2)
const METALLIC    = UInt8(3)
const TRANSLUCENT = UInt8(4)
end

"""The lobes a closure list holds, each with one of the `Fresnel` models where it reflects."""
module Lobe
const DIFFUSE     = UInt8(1)
const TRANSLUCENT = UInt8(2)
const MICROFACET  = UInt8(3)
end

module Fresnel
const CONSTANT   = UInt8(0)
const F82_TINT   = UInt8(1)
const DIELECTRIC = UInt8(2)
end

"""The most closures one shader produces at a hit; past it, further closures are dropped."""
const MAX_CLOSURES = 8
const CLOSURE_WEIGHT_CUTOFF = 1f-5

"""
What a shader program reads besides the texture inputs: where the hit is seen from, the
position's screen-space differentials (for bump), the shading tangent and the geometric normal.
"""
struct ShaderContext
    tfc::TextureFilterContext
    wo::Vec3f
    dpdx::Vec3f
    dpdy::Vec3f
    dpdus::Vec3f
    ng::Vec3f
end

"""
One closure in RGB, as the program produces it. `weight` is the colour the lobe is scaled by,
`f0` and `tint` are its Fresnel parameters (`F82_TINT`: base and edge tint; `DIELECTRIC`: the
reflectance at normal incidence and `ior`), `alpha` the GGX roughness.
"""
struct RGBClosure
    lobe::UInt8
    fresnel::UInt8
    weight::Vec3f
    f0::Vec3f
    tint::Vec3f
    n::Vec3f
    alpha::Float32
    ior::Float32
end

RGBClosure() = RGBClosure(0x00, 0x00, Vec3f(0), Vec3f(0), Vec3f(0), Vec3f(0, 0, 1), 0f0, 1f0)

struct ClosureList
    count::Int32
    closures::NTuple{MAX_CLOSURES, RGBClosure}
end

ClosureList() = ClosureList(Int32(0), ntuple(_ -> RGBClosure(), Val(MAX_CLOSURES)))

# Written and read by selects: a runtime index into a tuple is refused by the SPIR-V emitter.
@inline function _push(cl::ClosureList, c::RGBClosure)
    (cl.count >= MAX_CLOSURES || _max(c.weight) <= CLOSURE_WEIGHT_CUTOFF) && return cl
    n = cl.count + Int32(1)
    return ClosureList(n, ntuple(k -> ifelse(Int32(k) == n, c, cl.closures[k]), Val(MAX_CLOSURES)))
end

@inline function _closure_at(closures::NTuple{N, T}, i::Int32) where {N, T}
    c = closures[1]
    Base.Cartesian.@nexprs 8 k -> (k <= N && (c = ifelse(i == Int32(k), closures[min(k, N)], c)))
    return c
end

@inline _max(v::Vec3f) = max(v[1], v[2], v[3])
@inline _saturate3(v::Vec3f) = clamp.(v, 0f0, 1f0)
@inline _unit(n::Vec3f, fallback::Vec3f) = (l = norm(n); l > 1f-8 ? n / l : fallback)

"""Input `j` of the `ARGS` instruction after `i`: the inputs of a node past `NODE_ARGS`."""
@inline _fx(s, prog, i::Int32, j) = _f(s, (@inbounds prog[i + Int32(1)]), j)
@inline _vx(s, prog, i::Int32, j) = _v(s, (@inbounds prog[i + Int32(1)]), j)

"""A closure's normal: unit, and on the side `wo` is on, so a bump that tilts past the view does not flip it."""
@inline function _closure_normal(n::Vec3f, sc::ShaderContext)
    nn = _unit(n, sc.tfc.n)
    return dot(nn, sc.wo) * dot(sc.tfc.n, sc.wo) > 0f0 ? nn : sc.tfc.n
end

@inline _f0_from_ior(ior::Float32) = ((ior - 1f0) / (ior + 1f0))^2
@inline function _ior_from_f0(f0::Float32)
    r = sqrt(clamp(f0, 0f0, 0.99f0))
    return (1f0 + r) / (1f0 - r)
end

@inline _dielectric_layer(weight::Vec3f, f0::Vec3f, ior::Float32, n::Vec3f, alpha::Float32) =
    RGBClosure(Lobe.MICROFACET, Fresnel.DIELECTRIC, weight, f0, Vec3f(1), n, alpha, ior)

"""
What a dielectric layer reflects of the light from `wo`, estimated by its Fresnel at the view
angle: exact for a smooth layer, more than a rough one reflects at grazing angles. Cycles reads
the rough albedo from precomputed tables instead.
"""
@inline function _dielectric_albedo(f0::Vec3f, ior::Float32, cos_o::Float32)
    s = _schlick_shape(cos_o, ior)
    return f0 .+ (1f0 .- f0) .* s
end

"""The real dielectric Fresnel remapped to go from 0 at normal incidence to 1 at grazing, as Cycles' generalized Schlick."""
@inline function _schlick_shape(cos_i::Float32, ior::Float32)
    r0 = _f0_from_ior(ior)
    return clamp((fresnel_dielectric(cos_i, ior) - r0) / max(1f0 - r0, 1f-6), 0f0, 1f0)
end

@inline function _emit_closure(cl::ClosureList, s, ins::NodeInstr, sc::ShaderContext)
    w = _f(s, ins, 1)
    kind = ins.mode[1]
    if kind == ClosureKind.DIFFUSE
        n = _closure_normal(_v(s, ins, 4), sc)
        return _push(cl, RGBClosure(Lobe.DIFFUSE, Fresnel.CONSTANT, w * _saturate3(_v(s, ins, 2)), Vec3f(1), Vec3f(1),
                                    n, 0f0, 1f0))
    elseif kind == ClosureKind.TRANSLUCENT
        n = _closure_normal(_v(s, ins, 3), sc)
        return _push(cl, RGBClosure(Lobe.TRANSLUCENT, Fresnel.CONSTANT, w * _saturate3(_v(s, ins, 2)), Vec3f(1),
                                    Vec3f(1), n, 0f0, 1f0))
    elseif kind == ClosureKind.GLOSSY
        n = _closure_normal(_v(s, ins, 4), sc)
        alpha = clamp(_f(s, ins, 3), 0f0, 1f0)^2
        return _push(cl, RGBClosure(Lobe.MICROFACET, Fresnel.CONSTANT, Vec3f(w), _saturate3(_v(s, ins, 2)), Vec3f(1),
                                    n, alpha, 1f0))
    elseif kind == ClosureKind.METALLIC
        n = _closure_normal(_v(s, ins, 5), sc)
        alpha = clamp(_f(s, ins, 4), 0f0, 1f0)^2
        return _push(cl, RGBClosure(Lobe.MICROFACET, Fresnel.F82_TINT, Vec3f(w), _saturate3(_v(s, ins, 2)),
                                    _saturate3(_v(s, ins, 3)), n, alpha, 1f0))
    end
    return cl
end

"""
Blender's Principled BSDF, as Cycles lays it out: a coat over the metal, the specular layer and
the diffuse base, each layer taking from those under it what it reflects.
"""
@inline function _principled(cl::ClosureList, s, prog, i::Int32, ins::NodeInstr, sc::ShaderContext)
    w = Vec3f(_f(s, ins, 1))
    base = max.(_v(s, ins, 2), 0f0)
    metallic = clamp(_f(s, ins, 3), 0f0, 1f0)
    alpha = clamp(_f(s, ins, 4), 0f0, 1f0)^2
    ior = max(_f(s, ins, 5), 1f-5)
    n = _closure_normal(_v(s, ins, 6), sc)
    specular_level = max(_f(s, ins, 7), 0f0)
    specular_tint = max.(_v(s, ins, 8), 0f0)
    coat_weight = max(_f(s, ins, 9), 0f0)
    coat_alpha = clamp(_f(s, ins, 10), 0f0, 1f0)^2
    coat_ior = max(_fx(s, prog, i, 1), 1f0)
    coat_tint = max.(_vx(s, prog, i, 2), 0f0)
    coat_n = _closure_normal(_vx(s, prog, i, 3), sc)

    if coat_weight > CLOSURE_WEIGHT_CUTOFF
        coat = coat_weight * w
        f0 = Vec3f(_f0_from_ior(coat_ior))
        cl = _push(cl, _dielectric_layer(coat, f0, coat_ior, coat_n, coat_alpha))
        albedo = coat .* _dielectric_albedo(f0, coat_ior, abs(dot(sc.wo, coat_n)))
        w = clamp.(w .- albedo, 0f0, w) .* (1f0 .- min(coat_weight, 1f0) .* (1f0 .- coat_tint))
    end

    base_clamped = min.(base, 1f0)
    if metallic > CLOSURE_WEIGHT_CUTOFF
        cl = _push(cl, RGBClosure(Lobe.MICROFACET, Fresnel.F82_TINT, metallic * w, base_clamped,
                                  min.(specular_tint, 1f0), n, alpha, 1f0))
        w = w * (1f0 - metallic)
    end

    f0 = _f0_from_ior(ior)
    eta = ior
    if specular_level != 0.5f0
        f0 *= 2f0 * specular_level
        eta = ior < 1f0 ? 1f0 / _ior_from_f0(f0) : _ior_from_f0(f0)
    end
    if eta != 1f0 && _max(w) > CLOSURE_WEIGHT_CUTOFF
        spec_f0 = min.(f0 * specular_tint, 1f0)
        cl = _push(cl, _dielectric_layer(w, spec_f0, eta, n, alpha))
        albedo = w .* _dielectric_albedo(spec_f0, eta, abs(dot(sc.wo, n)))
        w = clamp.(w .- albedo, 0f0, w)
    end

    return _push(cl, RGBClosure(Lobe.DIFFUSE, Fresnel.CONSTANT, base .* w, Vec3f(1), Vec3f(1), n, 0f0, 1f0))
end

"""
Blender's Bump node, as Cycles' `svm_node_set_bump`: the height at the hit and at the hit moved
by `filter_width` of each screen-space differential, turned into a normal.
"""
@inline function _bump(s, ins::NodeInstr, sc::ShaderContext)
    h_c, h_x, h_y = _f(s, ins, 1), _f(s, ins, 2), _f(s, ins, 3)
    strength = clamp(_f(s, ins, 4), 0f0, 1f0)
    scale = ins.mode[1] == 0x01 ? -_f(s, ins, 5) : _f(s, ins, 5)
    normal_in = _unit(_v(s, ins, 6), sc.tfc.n)
    filter_width = _f(s, ins, 7)
    dpdx, dpdy = _bump_differentials(sc)

    rx = cross(dpdy, normal_in)
    ry = cross(normal_in, dpdx)
    det = dot(dpdx, rx)
    surfgrad = (h_x - h_c) * rx + (h_y - h_c) * ry
    normal_out = filter_width * abs(det) * normal_in - scale * sign(det) * surfgrad
    l = norm(normal_out)
    l > 0f0 || return normal_in
    return _unit(strength * (normal_out / l) + (1f0 - strength) * normal_in, normal_in)
end

"""The differentials a bump shifts the hit by; along the shading tangents when the hit has none."""
@inline function _bump_differentials(sc::ShaderContext)
    dot(sc.dpdx, sc.dpdx) > 0f0 && dot(sc.dpdy, sc.dpdy) > 0f0 && return sc.dpdx, sc.dpdy
    t = _unit(sc.dpdus - dot(sc.dpdus, sc.tfc.n) * sc.tfc.n, coordinate_system(sc.tfc.n)[1])
    return BUMP_DEFAULT_DELTA * t, BUMP_DEFAULT_DELTA * cross(sc.tfc.n, t)
end

@inline function _shifted_leaf(op::UInt8, s, ins::NodeInstr, sc::ShaderContext)
    fw = ins.k[1]
    dpdx, dpdy = _bump_differentials(sc)
    tfc = sc.tfc
    op == NodeOp.POSITION_DX && return Vec3f(tfc.p) + fw * dpdx
    op == NodeOp.POSITION_DY && return Vec3f(tfc.p) + fw * dpdy
    op == NodeOp.UV_DX && return Vec3f(tfc.uv[1] + fw * tfc.dudx, tfc.uv[2] + fw * tfc.dvdx, 0f0)
    return Vec3f(tfc.uv[1] + fw * tfc.dudy, tfc.uv[2] + fw * tfc.dvdy, 0f0)
end

"""
Run a compiled shader program at a hit: the closures it produces. Not inlined, for the reason
`eval_nodes` is not.
"""
@noinline function eval_shader(prog::AbstractVector{NodeInstr{F}}, textures, sc::ShaderContext) where {F}
    s = _new_stack(Val(_stack_size(F)))
    cl = ClosureList()
    i = Int32(1)
    n = length(prog) % Int32
    @inbounds while i <= n
        ins = prog[i]
        op = ins.op
        skip = Int32(0)
        if op == NodeOp.RESULT
            return cl
        elseif op == NodeOp.CLOSURE
            cl = _emit_closure(cl, s, ins, sc)
        elseif op == NodeOp.PRINCIPLED
            cl = _principled(cl, s, prog, i, ins, sc)
            skip = Int32(1)
        elseif op == NodeOp.BUMP
            s = _store(s, ins, _bump(s, ins, sc))
        elseif NodeOp.POSITION_DX <= op <= NodeOp.UV_DY
            s = _store(s, ins, _shifted_leaf(op, s, ins, sc))
        else
            s, skip = _exec(s, prog, i, ins, textures, sc.tfc)
        end
        i += Int32(1) + skip
    end
    return cl
end

@propagate_inbounds _th_shader(prog::AbstractVector{<:NodeInstr}, textures, sc) = eval_shader(prog, textures, sc)
@propagate_inbounds _th_shader(prog, textures, sc) = ClosureList()
