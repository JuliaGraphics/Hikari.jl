# ============================================================================
# Blender's BSDFs, and what Hikari renders a material with
# ============================================================================
#
# The BSDFs below are Blender's, with its inputs and defaults; with `Diffuse`, `Emission` and
# `Mix` they nest into one another as a Blender shader graph does. A material a user writes is a
# description: when it is pushed, `surface_material` turns it into what Hikari renders it with, one
# of its own models (the pbrt ones, and `Diffuse` alone) or a shader compiled from its BSDF nodes.

"""
    Principled(; base_color = (0.8, 0.8, 0.8), metallic = 0, roughness = 0.5, ior = 1.5, alpha = 1,
               normal = nothing, diffuse_roughness = 0, subsurface_weight = 0, subsurface_radius = (1, 0.2, 0.1),
               subsurface_scale = 0.05, subsurface_ior = 1.4, subsurface_anisotropy = 0, specular_ior_level = 0.5,
               specular_tint = (1, 1, 1), anisotropic = 0, anisotropic_rotation = 0, tangent = nothing,
               transmission_weight = 0, transmission_dispersion_scale = 0, transmission_dispersion_abbe_number = 20,
               coat_weight = 0, coat_roughness = 0.03, coat_ior = 1.5, coat_tint = (1, 1, 1), coat_normal = nothing,
               sheen_weight = 0, sheen_roughness = 0.5, sheen_tint = (1, 1, 1), emission_color = (1, 1, 1),
               emission_strength = 0, thin_film_thickness = 0, thin_film_ior = 1.33, thin_wall = false)

Blender's Principled BSDF with every input it has, laid out as Cycles does: transparency, sheen and
coat over the metal, transmission and the specular layer over the diffuse base. Any input but
`thin_wall` takes a number, a colour, an image or a node graph.

`normal = nothing` is the surface's shading normal and `coat_normal = nothing` is `normal`; the
default `tangent` is Blender's, radial around the Z axis (here the world's, Blender's is the
object's). With `thin_wall`, a constant, transmission and subsurface are those of a surface of no
thickness. Emission is found by the paths that hit it; for a light, use [`Emission`](@ref).
"""
@description struct Principled <: Material
    base_color = (0.8, 0.8, 0.8)
    metallic = 0
    roughness = 0.5
    ior = 1.5
    alpha = 1
    normal = nothing
    diffuse_roughness = 0
    subsurface_weight = 0
    subsurface_radius = (1, 0.2, 0.1)
    subsurface_scale = 0.05
    subsurface_ior = 1.4
    subsurface_anisotropy = 0
    specular_ior_level = 0.5
    specular_tint = (1, 1, 1)
    anisotropic = 0
    anisotropic_rotation = 0
    tangent = nothing
    transmission_weight = 0
    transmission_dispersion_scale = 0
    transmission_dispersion_abbe_number = 20
    coat_weight = 0
    coat_roughness = 0.03
    coat_ior = 1.5
    coat_tint = (1, 1, 1)
    coat_normal = nothing
    sheen_weight = 0
    sheen_roughness = 0.5
    sheen_tint = (1, 1, 1)
    emission_color = (1, 1, 1)
    emission_strength = 0
    thin_film_thickness = 0
    thin_film_ior = 1.33
    thin_wall = false
end

"""
    Glass(; color = (1, 1, 1), roughness = 0, ior = 1.5, normal = nothing,
          thin_film_thickness = 0, thin_film_ior = 1.33)

Blender's Glass BSDF: GGX reflection and refraction with the dielectric Fresnel, both tinted by
`color`, with the multiscatter energy compensation of Cycles' default distribution. `roughness` is
Blender's, the square root of the GGX alpha.
"""
@description struct Glass <: Material
    color = (1, 1, 1)
    roughness = 0
    ior = 1.5
    normal = nothing
    thin_film_thickness = 0
    thin_film_ior = 1.33
end

"""
    Glossy(; color = (0.8, 0.8, 0.8), roughness = 0.5, anisotropy = 0, rotation = 0, normal = nothing,
           tangent = nothing)

Blender's Glossy BSDF with the multiscatter GGX distribution: a microfacet reflection tinted by
`color` at every angle, a mirror at no `roughness`.
"""
@description struct Glossy <: Material
    color = (0.8, 0.8, 0.8)
    roughness = 0.5
    anisotropy = 0
    rotation = 0
    normal = nothing
    tangent = nothing
end

"""
    Metallic(; base_color = (0.617, 0.577, 0.540), edge_tint = (0.695, 0.726, 0.770), roughness = 0.5,
             anisotropy = 0, rotation = 0, normal = nothing, tangent = nothing,
             thin_film_thickness = 0, thin_film_ior = 1.33)

Blender's Metallic BSDF with the F82 tint Fresnel: `base_color` at normal incidence, `edge_tint`
towards grazing angles. For a metal of measured complex IOR, as pbrt renders one, use
[`Conductor`](@ref).
"""
@description struct Metallic <: Material
    base_color = (0.617, 0.577, 0.540)
    edge_tint = (0.695, 0.726, 0.770)
    roughness = 0.5
    anisotropy = 0
    rotation = 0
    normal = nothing
    tangent = nothing
    thin_film_thickness = 0
    thin_film_ior = 1.33
end

"""
    Translucent(; color = (0.8, 0.8, 0.8), normal = nothing)

Blender's Translucent BSDF: Lambert through the surface.
"""
@description struct Translucent <: Material
    color = (0.8, 0.8, 0.8)
    normal = nothing
end

"""
    Sheen(; color = (0.8, 0.8, 0.8), roughness = 0.5, normal = nothing)

Blender's Sheen BSDF with the microfiber distribution.
"""
@description struct Sheen <: Material
    color = (0.8, 0.8, 0.8)
    roughness = 0.5
    normal = nothing
end

"""
    Transparent(; color = (1, 1, 1))

Blender's Transparent BSDF: light passes through unrefracted, by `color`, averaged to a coverage.
"""
@description struct Transparent <: Material
    color = (1, 1, 1)
end

"""
    Add(a, b)

Blender's Add Shader: the lobes of both, which is not energy conserving in general. Both must be
BSDFs a shader graph holds (`Principled`, `Diffuse`, `Glass`, `Glossy`, `Metallic`, `Translucent`,
`Sheen`, `Transparent`, `Emission`, and mixes and sums of them).
"""
struct Add{A <: Material, B <: Material} <: Material
    a::A
    b::B
end

# ── Into a shader graph ─────────────────────────────────────────────────────

"""
    nests(material) -> Bool

Whether a material is a BSDF node of a shader graph: Blender's BSDFs, `Diffuse` (without a
displacement) and `Emission`, and mixes and sums of them. pbrt's models are not; a `Mix` of one
picks between materials.
"""
nests(::Material) = false
nests(::Union{Principled, Glass, Glossy, Metallic, Translucent, Sheen, Transparent, Emission}) = true
nests(m::Diffuse) = isnothing(m.displacement)
nests(m::Union{Mix, Add}) = nests(m.a) && nests(m.b)

"""
    node_input(x)

A material field as a shader graph's input reads it: a number, a constant colour, a node's output
as it is, and an image sampled at the uv.
"""
node_input(x::TexNode) = x
node_input(x::Real) = Float32(x)
node_input(x::Union{Tuple{Real, Real, Real}, Colorant, AbstractVector{<:Real}}) = TextureNodes.constant(x)
node_input(x::RGBSpectrum) = TextureNodes.constant((x.c[1], x.c[2], x.c[3]))
node_input(t::Texture{T, 0}) where {T} = node_input(constant_value(t))
node_input(t::Texture) = TextureNodes.image_texture(t)
node_input(img::AbstractMatrix) = TextureNodes.image_texture(img)
function node_input(h::TexHandle)
    h.kind == TexKind.IMAGE && return TextureNodes.image_texture(h)
    h.kind == TexKind.CONST_FLOAT && return h.f
    h.kind == TexKind.CONST_SPECTRUM && return node_input(h.rgb)
    throw(ArgumentError("a shader input takes a stored image, a constant, or a node graph, not a texture handle of kind $(h.kind)"))
end
node_input(x) = throw(ArgumentError("a shader input takes a number, a colour, an image or a node graph, not a $(typeof(x))"))

"""The shading normal, where a normal input is `nothing`."""
normal_input(::Nothing) = TextureNodes.normal()
normal_input(n) = node_input(n)
"""Blender's radial tangent, where a tangent input is `nothing`."""
tangent_input(::Nothing) = TextureNodes.radial_tangent()
tangent_input(t) = node_input(t)

closure(kind::UInt8, inputs...) = BSDFNode(NodeOp.CLOSURE, (kind, 0x00, 0x00), Any[inputs...])

"""
    shader_node(material) -> ShaderNode

The node a material is in a shader graph (`nests`).
"""
function shader_node end

function shader_node(m::Diffuse)
    isnothing(m.displacement) ||
        throw(ArgumentError("a Diffuse in a shader graph takes its bump as its `normal` (a `bump` node), not a displacement"))
    return closure(ClosureKind.DIFFUSE, node_input(m.color), node_input(m.roughness), normal_input(m.normal))
end
function shader_node(m::Glass)
    inputs = (node_input(m.color), node_input(m.roughness), node_input(m.ior), normal_input(m.normal))
    _constant(m.thin_film_thickness, 0) && return closure(ClosureKind.GLASS, inputs...)
    return closure(ClosureKind.GLASS_FILM, inputs..., node_input(m.thin_film_thickness), node_input(m.thin_film_ior))
end
shader_node(m::Glossy) =
    closure(ClosureKind.GLOSSY, node_input(m.color), node_input(m.roughness), node_input(m.anisotropy),
            node_input(m.rotation), normal_input(m.normal), tangent_input(m.tangent))
function shader_node(m::Metallic)
    inputs = (node_input(m.base_color), node_input(m.edge_tint), node_input(m.roughness),
              node_input(m.anisotropy), node_input(m.rotation), normal_input(m.normal), tangent_input(m.tangent))
    _constant(m.thin_film_thickness, 0) && return closure(ClosureKind.METALLIC, inputs...)
    return closure(ClosureKind.METALLIC_FILM, inputs..., node_input(m.thin_film_thickness), node_input(m.thin_film_ior))
end
shader_node(m::Translucent) = closure(ClosureKind.TRANSLUCENT, node_input(m.color), normal_input(m.normal))
shader_node(m::Sheen) = closure(ClosureKind.SHEEN, node_input(m.color), node_input(m.roughness), normal_input(m.normal))
shader_node(m::Transparent) = closure(ClosureKind.TRANSPARENT, node_input(m.color))
shader_node(m::Emission) = closure(ClosureKind.EMISSION, node_input(m.color), node_input(m.strength))
shader_node(m::Mix) = MixShaderNode(node_input(m.factor), shader_node(m.a), shader_node(m.b))
shader_node(m::Add) = AddShaderNode(shader_node(m.a), shader_node(m.b))
shader_node(m::Material) =
    throw(ArgumentError("a $(nameof(typeof(m))) is not a BSDF of a shader graph: only Blender's BSDFs, Diffuse, Emission, " *
                        "and mixes and sums of them are"))

function shader_node(p::Principled)
    p.thin_wall isa Bool ||
        throw(ArgumentError("a Principled's `thin_wall` is a constant, not a $(typeof(p.thin_wall)): it decides which lobes the material has"))
    normal = normal_input(p.normal)
    coat_normal = isnothing(p.coat_normal) ? normal : node_input(p.coat_normal)
    inputs = Any[node_input(p.base_color), node_input(p.metallic), node_input(p.roughness), node_input(p.ior),
                 node_input(p.alpha), normal, node_input(p.diffuse_roughness), node_input(p.subsurface_weight),
                 node_input(p.subsurface_radius), node_input(p.subsurface_scale), node_input(p.subsurface_ior),
                 node_input(p.subsurface_anisotropy), node_input(p.specular_ior_level), node_input(p.specular_tint),
                 node_input(p.anisotropic), node_input(p.anisotropic_rotation), tangent_input(p.tangent),
                 node_input(p.transmission_weight), node_input(p.transmission_dispersion_scale),
                 node_input(p.transmission_dispersion_abbe_number), node_input(p.coat_weight),
                 node_input(p.coat_roughness), node_input(p.coat_ior), node_input(p.coat_tint), coat_normal,
                 node_input(p.sheen_weight), node_input(p.sheen_roughness), node_input(p.sheen_tint),
                 node_input(p.emission_color), node_input(p.emission_strength), node_input(p.thin_film_thickness),
                 node_input(p.thin_film_ior), Float32(p.thin_wall)]
    return BSDFNode(NodeOp.PRINCIPLED, (0x00, 0x00, 0x00), inputs)
end

# ── What Hikari renders a material with ─────────────────────────────────────

"""
    surface_material(material) -> material

What Hikari renders `material` with: a shader compiled from the BSDF nodes of Blender's BSDFs and
their mixes and sums, a `Diffuse` without a `normal` and pbrt's models as they are, and a mix that
holds one of pbrt's models as the material that picks one of its own at each hit. `push!` and
`update_material!` call it.
"""
surface_material(m::Material) = m
surface_material(m::Union{Principled, Glass, Glossy, Metallic, Translucent, Sheen, Transparent, Add}) =
    ShaderMaterial(shader_node(m))
surface_material(m::Diffuse) = isnothing(m.normal) ? m : ShaderMaterial(shader_node(m))
surface_material(m::Mix) = nests(m) ? ShaderMaterial(shader_node(m)) :
    Mix(surface_material(m.a), surface_material(m.b), m.factor)
surface_material(m::MediumInterface) =
    MediumInterface(surface_material(m.material); inside = m.inside, outside = m.outside, emission = m.emission,
                    shadows = m.shadows)
