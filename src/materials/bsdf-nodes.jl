# Ported from Blender Cycles v5.2.1: intern/cycles/kernel/svm/closure.h, copyright 2011-2022
# Blender Foundation, SPDX-License-Identifier: Apache-2.0. Modified: rewritten in Julia as typed
# BSDF nodes. See LICENSES/Apache-2.0.txt.

# ============================================================================
# BSDF nodes: what a shader graph's BSDFs are in a material's type
# ============================================================================
#
# A shader graph's BSDF nodes, Mix Shader and Add Shader are flattened on the host into one
# node per BSDF, each weighted by the product of the mix factors above it (`ShaderGraph`). The
# program computes every input and weight; its BSDF instructions come after all of that, in the
# order of the material's node types. Each node type turns its instruction's inputs into lobes
# (`node_lobes`) the way Cycles' `svm_node_closure_bsdf` sets up its closures, in RGB; the
# material lifts them to the path's wavelengths (`lift`).

abstract type ShaderBSDF end

"""
    node_lobes(::Type{<:ShaderBSDF}, s, prog, i, sc, lut) -> (lobes, transparency, emission)

The weighted lobes, in RGB, of the BSDF node whose instruction is `prog[i]`, with `s` the stack
the program's values left, and what it adds to the hit's transparency and emission.
"""
function node_lobes end

"""The instructions a BSDF node's inputs take in a program: its own and the `ARGS` after it."""
function instruction_count end

"""
Input `j` of the BSDF node whose `chunks` instructions start at `prog[i]`, as a vector: off the
stack, a constant, the shading normal, or a colour folded into the data after its instructions
(`compile_shader`).
"""
@inline function node_v(s, prog, i::Int32, j::Int, chunks::Int, sc::ShaderContext)
    ins = @inbounds prog[i + Int32((j - 1) ÷ NODE_ARGS)]
    a = (j - 1) % NODE_ARGS + 1
    o = ins.arg[a]
    o == SHADING_NORMAL_ARG && return sc.tfc.n
    if o >= WIDE_CONSTANT_ARG
        m = Int32(o - WIDE_CONSTANT_ARG)
        d = @inbounds prog[i + Int32(chunks) + m ÷ Int32(3)]
        base = (m % Int32(3)) * Int32(3)
        return Vec3f(_kat(d.k, base + Int32(1)), _kat(d.k, base + Int32(2)), _kat(d.k, base + Int32(3)))
    end
    return _v(s, ins, a)
end

"""Input `j` as a scalar: a vector's first component."""
@inline node_f(s, prog, i::Int32, j::Int, chunks::Int, sc::ShaderContext) = node_v(s, prog, i, j, chunks, sc)[1]

"""Where the BSDF node after the one at `prog[i]` starts: past its instructions and the colours folded after them."""
@inline next_node(prog, i::Int32, chunks::Int) = i + Int32(chunks) + Int32((@inbounds prog[i]).mode[3])

"""`w` kept as Cycles keeps a closure: with no share of the samples when it weighs nothing."""
@inline kept(w::Weighted) =
    (_max(w.weight) > CLOSURE_WEIGHT_CUTOFF && w.sample_weight > 0f0) ? w :
    Weighted(w.lobe, w.weight, 0f0, w.n, w.t)

# The nodes of a material, in program order: each node's lobes, then the rest's.
@inline node_lobes(::Type{Tuple{}}, s, prog, i::Int32, sc::ShaderContext, lut) = ((), Vec3f(0), Vec3f(0))
@inline function node_lobes(::Type{M}, s, prog, i::Int32, sc::ShaderContext, lut) where {M <: Tuple}
    T = Base.tuple_type_head(M)
    lobes, transparency, emission = node_lobes(T, s, prog, i, sc, lut)
    rest, rest_transparency, rest_emission =
        node_lobes(Base.tuple_type_tail(M), s, prog, next_node(prog, i, instruction_count(T)), sc, lut)
    return ((lobes..., rest...), transparency + rest_transparency, emission + rest_emission)
end

"""Blender's Diffuse BSDF: Lambert, or Cycles' energy-preserving Oren-Nayar with roughness."""
struct DiffuseBSDF <: ShaderBSDF end
instruction_count(::Type{DiffuseBSDF}) = 1
@inline function node_lobes(::Type{DiffuseBSDF}, s, prog, i::Int32, sc::ShaderContext, lut)
    color = max.(node_v(s, prog, i, 2, 1, sc), 0f0)
    n = _closure_normal(node_v(s, prog, i, 4, 1, sc), sc)
    lobe = DiffuseLobe(node_f(s, prog, i, 3, 1, sc), color, n, sc.wo)
    return ((kept(Weighted(lobe, node_f(s, prog, i, 1, 1, sc) * color, n)),), Vec3f(0), Vec3f(0))
end

"""Blender's Transparent BSDF: light passes through unrefracted, by `color`, which is no lobe but the hit's transparency."""
struct TransparentBSDF <: ShaderBSDF end
instruction_count(::Type{TransparentBSDF}) = 1
@inline node_lobes(::Type{TransparentBSDF}, s, prog, i::Int32, sc::ShaderContext, lut) =
    ((), transparency(s, prog, i, sc), Vec3f(0))
@inline transparency(s, prog, i::Int32, sc::ShaderContext) =
    node_f(s, prog, i, 1, 1, sc) * _saturate3(node_v(s, prog, i, 2, 1, sc))

"""
The transparency of a program of Transparent BSDFs alone, as an alpha test runs one
(`transparency_graph`): how many there are is the program's, so they are walked at run time.
"""
@noinline function program_transparency(prog::AbstractVector{<:NodeInstr}, textures, sc::ShaderContext)
    s, i = run_values(prog, textures, sc)
    t = Vec3f(0)
    while (@inbounds prog[i]).op != NodeOp.RESULT
        t += transparency(s, prog, i, sc)
        i = next_node(prog, i, 1)
    end
    return t
end

"""Blender's Emission: radiance `strength` times `color`, found by the paths that hit it."""
struct EmissionBSDF <: ShaderBSDF end
instruction_count(::Type{EmissionBSDF}) = 1
@inline node_lobes(::Type{EmissionBSDF}, s, prog, i::Int32, sc::ShaderContext, lut) =
    ((), Vec3f(0), node_f(s, prog, i, 1, 1, sc) * max.(node_v(s, prog, i, 2, 1, sc), 0f0) * node_f(s, prog, i, 3, 1, sc))

"""Whether a material of these nodes can emit light."""
emits(::Type{Tuple{}}) = false
emits(::Type{M}) where {M <: Tuple} = emits(Base.tuple_type_head(M)) || emits(Base.tuple_type_tail(M))
emits(::Type{<:ShaderBSDF}) = false
emits(::Type{EmissionBSDF}) = true

"""
The lobes, transparency and emission a shader program produces at a hit. The interpreter and the
nodes are in one function, out of line, so the stack is not copied out of it.
"""
@noinline function shader_lobes(::Type{M}, prog::AbstractVector{<:NodeInstr}, textures, sc::ShaderContext,
                               regularize::Bool) where {M <: Tuple}
    s, i = run_values(prog, textures, sc)
    return node_lobes(M, s, prog, i, sc, LobeSetup(lookup_tables(textures), regularize))
end

"""Whether any lobe disperses the path's wavelengths."""
@inline disperses(lobes::Tuple) = reduce(|, map(w -> disperses(w.lobe), lobes); init = false)
@inline disperses(::BSDFLobe) = false

"""The BSDF of the lobes, transparency and emission a material's nodes produced."""
@propagate_inbounds function lobe_bsdf(::Type{M}, (rgb, transparency, emission), table,
                                       lambda::Wavelengths) where {M <: Tuple}
    alpha = 1f0 - _avg(transparency)
    scale = alpha > 1f-6 ? 1f0 / alpha : 0f0
    lobes = map(w -> lift(w, scale, table, lambda), rgb)
    total = reduce(+, map(w -> w.sample_weight, lobes); init = 0f0)
    L = emits(M) && _max(emission) > 0f0 ?
        uplift_rgb_illuminant(table, RGBSpectrum(emission[1], emission[2], emission[3]), lambda) / D65_PHOTOMETRIC :
        SpectralRadiance(0f0)
    return LobeBSDF(lobes, total, lambda, L, disperses(lobes))
end

"""Blender's Translucent BSDF: Lambert through the surface."""
struct TranslucentBSDF <: ShaderBSDF end
instruction_count(::Type{TranslucentBSDF}) = 1
@inline function node_lobes(::Type{TranslucentBSDF}, s, prog, i::Int32, sc::ShaderContext, lut)
    color = max.(node_v(s, prog, i, 2, 1, sc), 0f0)
    n = _closure_normal(node_v(s, prog, i, 3, 1, sc), sc)
    return ((kept(Weighted(TranslucentLobe(), node_f(s, prog, i, 1, 1, sc) * color, n)),), Vec3f(0), Vec3f(0))
end

"""Blender's Glossy BSDF: GGX reflection of `color`, no Fresnel."""
struct GlossyBSDF <: ShaderBSDF end
"""Blender's Metallic BSDF: GGX reflection with the F82 tint of a conductor."""
struct MetallicBSDF <: ShaderBSDF end
struct MetallicFilmBSDF <: ShaderBSDF end
instruction_count(::Type{<:Union{GlossyBSDF, MetallicBSDF}}) = 1
instruction_count(::Type{MetallicFilmBSDF}) = cld(10, NODE_ARGS)
@inline node_lobes(::Type{GlossyBSDF}, s, prog, i::Int32, sc::ShaderContext, lut) = reflection_node(Val(false), s, prog, i, sc, lut)
@inline node_lobes(::Type{MetallicBSDF}, s, prog, i::Int32, sc::ShaderContext, lut) = reflection_node(Val(true), s, prog, i, sc, lut)
@inline node_lobes(::Type{MetallicFilmBSDF}, s, prog, i::Int32, sc::ShaderContext, lut) =
    reflection_node(Val(true), s, prog, i, sc, lut, Val(true))

@inline function reflection_node(::Val{Metal}, s, prog, i::Int32, sc::ShaderContext, lut,
                                 ::Val{Film} = Val(false)) where {Metal, Film}
    chunks = Film ? instruction_count(MetallicFilmBSDF) : 1
    base = 2 + Metal
    w = node_f(s, prog, i, 1, chunks, sc)
    roughness = clamp(node_f(s, prog, i, base + 1, chunks, sc), 0f0, 1f0)
    anisotropy = clamp(node_f(s, prog, i, base + 2, chunks, sc), -1f0, 1f0)
    n = _valid_normal(_closure_normal(node_v(s, prog, i, base + 4, chunks, sc), sc), sc)
    ax, ay, t = _anisotropic_alphas(roughness, anisotropy, node_v(s, prog, i, base + 5, chunks, sc), n,
                                    node_f(s, prog, i, base + 3, chunks, sc), sc)
    weighted = if Metal
        thickness = Film ? node_f(s, prog, i, 9, chunks, sc) : 0f0
        film_ior = Film ? max(node_f(s, prog, i, 10, chunks, sc), 1f-5) : 0f0
        fresnel = F82Tint(_saturate3(node_v(s, prog, i, 2, chunks, sc)),
                          _saturate3(node_v(s, prog, i, 3, chunks, sc)), thickness, film_ior)
        Weighted(MicrofacetLobe{:reflection}(fresnel; alpha_x = ax, alpha_y = ay), Vec3f(w), n; t)
    else
        color = max.(node_v(s, prog, i, 2, chunks, sc), 0f0)
        Weighted(MicrofacetLobe{:reflection}(NoFresnel(color); alpha_x = ax, alpha_y = ay), w * color, n; t)
    end
    return ((kept(first(setup_fresnel(lut, weighted, dot(n, sc.wo)))),), Vec3f(0), Vec3f(0))
end

"""Blender's Glass BSDF: GGX reflection and refraction with the real dielectric Fresnel, tinted by `color`."""
struct GlassBSDF <: ShaderBSDF end
struct GlassFilmBSDF <: ShaderBSDF end
instruction_count(::Type{<:Union{GlassBSDF, GlassFilmBSDF}}) = 1
@inline node_lobes(::Type{GlassBSDF}, s, prog, i::Int32, sc::ShaderContext, lut) =
    glass_node(Val(false), s, prog, i, sc, lut)
@inline node_lobes(::Type{GlassFilmBSDF}, s, prog, i::Int32, sc::ShaderContext, lut) =
    glass_node(Val(true), s, prog, i, sc, lut)
@inline function glass_node(::Val{Film}, s, prog, i::Int32, sc::ShaderContext, lut) where {Film}
    color = max.(node_v(s, prog, i, 2, 1, sc), 0f0)
    alpha = clamp(node_f(s, prog, i, 3, 1, sc), 0f0, 1f0)^2
    ior = max(node_f(s, prog, i, 4, 1, sc), 1f-5)
    n = _closure_normal(node_v(s, prog, i, 5, 1, sc), sc)
    back = _backfacing(sc)
    fresnel = if Film
        film_ior = max(node_f(s, prog, i, 7, 1, sc), 1f-5)
        SchlickFresnel(Vec3f(_f0_from_ior(ior)), color, color, node_f(s, prog, i, 6, 1, sc),
                       back ? film_ior / ior : film_ior)
    else
        DielectricFresnel(color, color)
    end
    lobe = MicrofacetLobe{:glass}(fresnel; alpha_x = alpha, ior = back ? 1f0 / ior : ior)
    weighted = Weighted(lobe, Vec3f(node_f(s, prog, i, 1, 1, sc)), n)
    return ((kept(first(setup_fresnel(lut, weighted, dot(n, sc.wo)))),), Vec3f(0), Vec3f(0))
end

"""Blender's Sheen BSDF with the microfiber distribution."""
struct SheenBSDF <: ShaderBSDF end
instruction_count(::Type{SheenBSDF}) = 1
@inline function node_lobes(::Type{SheenBSDF}, s, prog, i::Int32, sc::ShaderContext, lut)
    n = _closure_normal(node_v(s, prog, i, 4, 1, sc), sc)
    weight = node_f(s, prog, i, 1, 1, sc) * max.(node_v(s, prog, i, 2, 1, sc), 0f0)
    return ((kept(sheen(lut, weight, n, node_f(s, prog, i, 3, 1, sc), sc.wo)),), Vec3f(0), Vec3f(0))
end

# ── Toon ────────────────────────────────────────────────────────────────────

"""Blender's diffuse or glossy Toon BSDF, with its component fixed in the shader's type."""
struct ToonBSDF{Glossy} <: ShaderBSDF end
instruction_count(::Type{<:ToonBSDF}) = 1
@inline function node_lobes(::Type{ToonBSDF{Glossy}}, s, prog, i::Int32, sc::ShaderContext, lut) where {Glossy}
    n = _closure_normal(node_v(s, prog, i, 5, 1, sc), sc)
    weight = node_f(s, prog, i, 1, 1, sc) * max.(node_v(s, prog, i, 2, 1, sc), 0f0)
    lobe = ToonLobe{Glossy}(node_f(s, prog, i, 3, 1, sc), node_f(s, prog, i, 4, 1, sc))
    return ((kept(Weighted(lobe, weight, n)),), Vec3f(0), Vec3f(0))
end

# ── Principled ──────────────────────────────────────────────────────────────

"""
Blender's Principled BSDF, as Cycles' `svm_node_closure_bsdf` lays it out: transparency first, then
sheen, coat and emission, which take from what is under them what they reflect, then the metal,
transmission and the specular layer over subsurface and diffuse. `ThinWall` is its thin wall input,
a constant; `Subsurface` and `Emissive` whether its subsurface weight and emission strength are not
the constant zero, so a shader without them carries no code for them.
"""
struct PrincipledBSDF{ThinWall, Subsurface, Emissive} <: ShaderBSDF end
instruction_count(::Type{<:PrincipledBSDF}) = cld(34, NODE_ARGS)
emits(::Type{PrincipledBSDF{T, S, E}}) where {T, S, E} = E

@inline function principled_inputs(s, prog, i::Int32, sc::ShaderContext)
    c = instruction_count(PrincipledBSDF)
    return PrincipledInputs(
        node_f(s, prog, i, 1, c, sc), node_v(s, prog, i, 2, c, sc), node_f(s, prog, i, 3, c, sc), node_f(s, prog, i, 4, c, sc),
        node_f(s, prog, i, 5, c, sc), node_f(s, prog, i, 6, c, sc), node_v(s, prog, i, 7, c, sc), node_f(s, prog, i, 8, c, sc),
        node_f(s, prog, i, 9, c, sc), node_f(s, prog, i, 14, c, sc), node_v(s, prog, i, 15, c, sc), node_f(s, prog, i, 16, c, sc),
        node_f(s, prog, i, 17, c, sc), node_v(s, prog, i, 18, c, sc), node_f(s, prog, i, 19, c, sc), node_f(s, prog, i, 20, c, sc),
        node_f(s, prog, i, 21, c, sc), node_f(s, prog, i, 22, c, sc), node_f(s, prog, i, 23, c, sc), node_f(s, prog, i, 24, c, sc),
        node_v(s, prog, i, 25, c, sc), node_v(s, prog, i, 26, c, sc), node_f(s, prog, i, 27, c, sc), node_f(s, prog, i, 28, c, sc),
        node_v(s, prog, i, 29, c, sc), node_v(s, prog, i, 30, c, sc) * node_f(s, prog, i, 31, c, sc),
        node_f(s, prog, i, 32, c, sc), node_f(s, prog, i, 33, c, sc), node_f(s, prog, i, 13, c, sc),
        node_f(s, prog, i, 34, c, sc), node_v(s, prog, i, 10, c, sc), node_f(s, prog, i, 11, c, sc))
end

@inline node_lobes(::Type{PrincipledBSDF{T, S, E}}, s, prog, i::Int32, sc::ShaderContext, lut) where {T, S, E} =
    principled_lobes(Val(T), Val(S), principled_inputs(s, prog, i, sc), sc, lut)

"""A lobe no layer produced at this hit: the type a slot holds, with no weight and no share of the samples."""
@inline absent(lobe::BSDFLobe, n::Vec3f) = Weighted(lobe, Vec3f(0), 0f0, n, Vec3f(0))

@inline function principled_lobes(::Val{Thin}, ::Val{SS}, p::PrincipledInputs, sc::ShaderContext, lut) where {Thin, SS}
    wo = sc.wo
    w = Vec3f(p.weight)
    alpha = clamp(p.alpha, 0f0, 1f0)
    transparency = alpha < 1f0 ? w * (1f0 - alpha) : Vec3f(0)
    alpha < 1f0 && (w = w * alpha)
    base = max.(p.base, 0f0)
    base_clamped = min.(base, 1f0)
    metallic = clamp(p.metallic, 0f0, 1f0)
    roughness = clamp(p.roughness, 0f0, 1f0)
    ior = max(p.ior, 1f-5)
    n = _closure_normal(p.n, sc)
    diffuse_roughness = clamp(p.diffuse_roughness, 0f0, 1f0)
    subsurface = SS ? clamp(p.subsurface, 0f0, 1f0) : 0f0
    specular_tint = max.(p.specular_tint, 0f0)
    transmission = clamp(p.transmission, 0f0, 1f0)
    coat_weight = max(p.coat_weight, 0f0)
    coat_n = _closure_normal(p.coat_n, sc)
    film_thickness = p.film_thickness
    film_ior = film_thickness > THIN_FILM_THICKNESS_CUTOFF ? max(p.film_ior, 1f-5) : 0f0

    # Sheen
    sheen_weight = max(p.sheen_weight, 0f0)
    l_sheen = if sheen_weight > CLOSURE_WEIGHT_CUTOFF
        sheen_n = _unit(n + clamp(coat_weight, 0f0, 1f0) * (coat_n - n), n)
        sheen(lut, sheen_weight * max.(p.sheen_tint, 0f0) .* w, sheen_n, clamp(p.sheen_roughness, 0f0, 1f0), wo)
    else
        absent(SheenLobe(1f0, 0f0), n)
    end
    w = _layering(l_sheen.weight, w)

    # Coat
    coat_ior = max(p.coat_ior, 1f0)
    coat_vn = _valid_normal(coat_n, sc)
    coat_lobe = MicrofacetLobe{:reflection}(DielectricFresnel(Vec3f(1), Vec3f(1));
                                            alpha_x = clamp(p.coat_roughness, 0f0, 1f0)^2, ior = coat_ior)
    l_coat = if coat_weight > CLOSURE_WEIGHT_CUTOFF
        coat, refl = setup_fresnel(lut, Weighted(coat_lobe, coat_weight * w, coat_vn), dot(coat_vn, wo))
        w = _layering(coat.weight .* refl, w)
        tint = max.(p.coat_tint, 0f0)
        if tint != Vec3f(1)
            cos_nt = sqrt(max(1f0 - (1f0 - dot(wo, coat_vn)^2) / coat_ior^2, 1f-8))
            w = w .* (1f0 .+ coat_weight .* (_pow3(tint, 1f0 / cos_nt) .- 1f0))
        end
        coat
    else
        absent(coat_lobe, coat_vn)
    end
    emission = _max(p.emission) > 0f0 ? max.(p.emission, 0f0) .* w : Vec3f(0)

    vn = _valid_normal(n, sc)
    ax, ay, t = _anisotropic_alphas(roughness, clamp(p.anisotropic, 0f0, 1f0), p.tangent, vn, p.rotation, sc)

    # Metal
    metal_lobe = MicrofacetLobe{:reflection}(F82Tint(base_clamped, min.(specular_tint, 1f0), film_thickness, film_ior);
                                             alpha_x = ax, alpha_y = ay)
    l_metal = if metallic > CLOSURE_WEIGHT_CUTOFF
        metal = first(setup_fresnel(lut, Weighted(metal_lobe, metallic * w, vn; t), dot(vn, wo)))
        w = w * (1f0 - metallic)
        metal
    else
        absent(metal_lobe, vn)
    end

    # Transmission
    transmission_lobes, w = principled_transmission(Val(Thin), transmission, w, vn, roughness, ior, specular_tint,
                                                    base_clamped, film_thickness, film_ior, p, sc, lut)

    # Specular
    eta = _specular_eta(ior, p.specular_level)
    f0 = _specular_f0(ior, p.specular_level)
    specular_lobe = MicrofacetLobe{:reflection}(SchlickFresnel(min.(f0 * specular_tint, 1f0), Vec3f(1), Vec3f(1),
                                                               film_thickness, film_ior); alpha_x = ax, alpha_y = ay, ior = eta)
    l_specular = if eta != 1f0 || film_thickness > THIN_FILM_THICKNESS_CUTOFF
        specular, refl = setup_fresnel(lut, Weighted(specular_lobe, w, vn; t), dot(vn, wo))
        w = _layering(specular.weight .* refl, w)
        specular
    else
        absent(specular_lobe, vn)
    end

    # Subsurface
    subsurface_lobes = principled_subsurface(Val(Thin), Val(SS), subsurface, w, n, roughness, ior, diffuse_roughness,
                                             base_clamped, p, wo)

    # Diffuse
    l_diffuse = Weighted(DiffuseLobe(diffuse_roughness, base, n, wo), base .* (1f0 - subsurface) .* w, n)

    lobes = map(kept, (l_sheen, l_coat, l_metal, transmission_lobes..., l_specular, subsurface_lobes..., l_diffuse))
    return (lobes, transparency, emission)
end

"""The Principled BSDF's glass: one glass lobe, or for a thin wall a thin glass's reflection and transmission."""
@inline function principled_transmission(::Val{false}, transmission::Float32, w::Vec3f, vn::Vec3f, roughness::Float32,
                                         ior::Float32, specular_tint::Vec3f, base_clamped::Vec3f, film_thickness::Float32,
                                         film_ior::Float32, p::PrincipledInputs, sc::ShaderContext, lut)
    abbe = max(p.abbe, 0f0)
    lobe = MicrofacetLobe{:glass}(SchlickFresnel(min.(_f0_from_ior(ior) * specular_tint, 1f0), Vec3f(1), sqrt.(base_clamped),
                                                 film_thickness, film_ior);
                                  alpha_x = roughness^2, ior = _backfacing(sc) ? 1f0 / ior : ior,
                                  inv_abbe = abbe > 0f0 ? clamp(p.dispersion, 0f0, 1f0) / abbe : 0f0)
    transmission > CLOSURE_WEIGHT_CUTOFF || return ((absent(lobe, vn),), w)
    glass = first(setup_fresnel(lut, Weighted(lobe, transmission * w, vn), dot(vn, sc.wo)))
    return ((glass,), w * (1f0 - transmission))
end

@inline function principled_transmission(::Val{true}, transmission::Float32, w::Vec3f, vn::Vec3f, roughness::Float32,
                                         ior::Float32, specular_tint::Vec3f, base_clamped::Vec3f, film_thickness::Float32,
                                         film_ior::Float32, p::PrincipledInputs, sc::ShaderContext, lut)
    reflection, through = thin_glass(lut, transmission * w, vn, roughness^2, ior, specular_tint, base_clamped, sc.wo,
                                     transmission > CLOSURE_WEIGHT_CUTOFF)
    return ((reflection, through), transmission > CLOSURE_WEIGHT_CUTOFF ? w * (1f0 - transmission) : w)
end

"""
Cycles' `bsdf_thin_glass_setup`: a slab of glass of no thickness, which reflects what its two
interfaces and the light between them reflect, and lets the rest through, tinted by `color` along
its path in the slab, as a reflection mirrored to the far side and roughened for two refractions.
"""
@inline function thin_glass(lut, w::Vec3f, n::Vec3f, alpha::Float32, ior::Float32, reflection_tint::Vec3f, color::Vec3f,
                            wo::Vec3f, present::Bool)
    cos_i = dot(n, wo)
    sin2_t = (1f0 - cos_i^2) / ior^2
    transmission_alpha = clamp(alpha * 3.4f0 * (ior - 1f0) * (ior - 0.5f0)^2 / ior^3, 0f0, 1f0)
    reflection_lobe = MicrofacetLobe{:reflection}(NoFresnel(reflection_tint); alpha_x = alpha)
    through_lobe = MicrofacetLobe{:thin}(NoFresnel(Vec3f(1)); alpha_x = transmission_alpha)
    (present && sin2_t < 1f0) || return (absent(reflection_lobe, n), absent(through_lobe, n))
    cos_t = sqrt(1f0 - sin2_t)
    f0 = min.(_f0_from_ior(ior) .* reflection_tint, 1f0)
    r1 = f0 .+ (1f0 .- f0) .* _schlick_shape(cos_i, ior)
    t1 = 1f0 .- r1
    c = _pow3(max.(color, 0f0), 1f0 / max(cos_t, 1f-6))
    rc = r1 .* c
    T = c .* t1 .* t1 ./ max.(1f0 .- rc .* rc, 1f-6)
    R = r1 .+ T .* r1 .* c
    reflection = first(setup_fresnel(lut, Weighted(reflection_lobe, R .* w, n), cos_i))
    through = first(setup_fresnel(lut, Weighted(through_lobe, T .* w, n), cos_i))
    return (reflection, through)
end

"""
The Principled BSDF's subsurface: none where its weight is the constant zero; for a thin wall a
diffuse reflection and transmission split by the anisotropy; else the random walk, or a diffuse
lobe where the radius is zero.
"""
@inline principled_subsurface(::Val{Thin}, ::Val{false}, subsurface, w, n, roughness, ior, diffuse_roughness,
                              base_clamped, p, wo) where {Thin} = ()

@inline function principled_subsurface(::Val{true}, ::Val{true}, subsurface::Float32, w::Vec3f, n::Vec3f, roughness::Float32,
                                       ior::Float32, diffuse_roughness::Float32, base_clamped::Vec3f, p::PrincipledInputs,
                                       wo::Vec3f)
    lobe = DiffuseLobe(diffuse_roughness, base_clamped, n, wo)
    subsurface > CLOSURE_WEIGHT_CUTOFF || return (absent(lobe, n), absent(lobe, -n))
    ss = base_clamped .* subsurface .* w
    a = p.subsurface_anisotropy
    return (Weighted(lobe, clamp(0.5f0 * (1f0 - a), 0f0, 1f0) * ss, n), Weighted(lobe, clamp(0.5f0 * (1f0 + a), 0f0, 1f0) * ss, -n))
end

@inline function principled_subsurface(::Val{false}, ::Val{true}, subsurface::Float32, w::Vec3f, n::Vec3f, roughness::Float32,
                                       ior::Float32, diffuse_roughness::Float32, base_clamped::Vec3f, p::PrincipledInputs,
                                       wo::Vec3f)
    radius = max.(p.subsurface_radius, 0f0) * max(p.subsurface_scale, 0f0)
    walk = SubsurfaceLobe(base_clamped, radius, roughness^2, _specular_eta(ior, p.specular_level), p.subsurface_anisotropy)
    diffuse = DiffuseLobe(diffuse_roughness, base_clamped, n, wo)
    subsurface > CLOSURE_WEIGHT_CUTOFF || return (absent(walk, n), absent(diffuse, n))
    ss = base_clamped .* subsurface .* w
    _max(radius) < 1f-8 && return (absent(walk, n), Weighted(diffuse, ss, n))
    return (Weighted(walk, ss, n), absent(diffuse, n))
end
