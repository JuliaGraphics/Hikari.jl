# ============================================================================
# glTF's and OpenPBR's materials, as Blender's Principled BSDF
# ============================================================================
#
# Two standards' material models with their own names, each a front end of `Principled`: Blender's
# Principled BSDF was designed close to OpenPBR, and Blender's glTF importer maps glTF's materials
# onto it the way `Principled(::GLTF)` does.

"""
    GLTF(; base_color_factor = (1, 1, 1, 1), base_color_texture = nothing, metallic_factor = 1,
         roughness_factor = 1, metallic_roughness_texture = nothing, normal_texture = nothing, normal_scale = 1,
         emissive_factor = (0, 0, 0), emissive_texture = nothing, emissive_strength = 1, alpha_mode = :opaque,
         alpha_cutoff = 0.5, ior = 1.5, transmission_factor = 0, transmission_texture = nothing,
         clearcoat_factor = 0, clearcoat_texture = nothing, clearcoat_roughness_factor = 0,
         clearcoat_roughness_texture = nothing, clearcoat_normal_texture = nothing, sheen_color_factor = (0, 0, 0),
         sheen_color_texture = nothing, sheen_roughness_factor = 0, sheen_roughness_texture = nothing,
         specular_factor = 1, specular_texture = nothing, specular_color_factor = (1, 1, 1),
         specular_color_texture = nothing, iridescence_factor = 0, iridescence_ior = 1.3,
         iridescence_thickness = 400, anisotropy_strength = 0, anisotropy_rotation = 0)

A glTF 2.0 material, its metallic-roughness model and the `KHR_materials_*` extensions (ior,
transmission, clearcoat, sheen, specular, emissive strength, iridescence, anisotropy), with glTF's
names. Each factor multiplies its texture, an image sampled at the uv, as glTF defines them:
metallic is the blue channel of `metallic_roughness_texture` and roughness its green; the colour
textures are taken as linear, decoded from sRGB. `alpha_mode` is `:opaque`, `:mask` (cut out below
`alpha_cutoff`) or `:blend`. It renders as the `Principled` BSDF Blender's glTF importer makes of it.
"""
@description struct GLTF <: Material
    base_color_factor = (1, 1, 1, 1)
    base_color_texture = nothing
    metallic_factor = 1
    roughness_factor = 1
    metallic_roughness_texture = nothing
    normal_texture = nothing
    normal_scale = 1
    emissive_factor = (0, 0, 0)
    emissive_texture = nothing
    emissive_strength = 1
    alpha_mode = :opaque
    alpha_cutoff = 0.5
    ior = 1.5
    transmission_factor = 0
    transmission_texture = nothing
    clearcoat_factor = 0
    clearcoat_texture = nothing
    clearcoat_roughness_factor = 0
    clearcoat_roughness_texture = nothing
    clearcoat_normal_texture = nothing
    sheen_color_factor = (0, 0, 0)
    sheen_color_texture = nothing
    sheen_roughness_factor = 0
    sheen_roughness_texture = nothing
    specular_factor = 1
    specular_texture = nothing
    specular_color_factor = (1, 1, 1)
    specular_color_texture = nothing
    iridescence_factor = 0
    iridescence_ior = 1.3
    iridescence_thickness = 400
    anisotropy_strength = 0
    anisotropy_rotation = 0
end

"""
    OpenPBR(; base_weight = 1, base_color = (0.8, 0.8, 0.8), base_metalness = 0, base_diffuse_roughness = 0,
            specular_weight = 1, specular_color = (1, 1, 1), specular_roughness = 0.3,
            specular_roughness_anisotropy = 0, specular_ior = 1.5, transmission_weight = 0,
            transmission_color = (1, 1, 1), transmission_dispersion_scale = 0,
            transmission_dispersion_abbe_number = 20, subsurface_weight = 0, subsurface_color = (0.8, 0.8, 0.8),
            subsurface_radius = 1, subsurface_radius_scale = (1, 0.5, 0.25), subsurface_scatter_anisotropy = 0,
            fuzz_weight = 0, fuzz_color = (1, 1, 1), fuzz_roughness = 0.5, coat_weight = 0, coat_color = (1, 1, 1),
            coat_roughness = 0, coat_ior = 1.6, thin_film_weight = 0, thin_film_thickness = 0.5,
            thin_film_ior = 1.4, emission_luminance = 0, emission_color = (1, 1, 1), geometry_opacity = 1,
            geometry_thin_walled = false, geometry_normal = nothing, geometry_coat_normal = nothing,
            geometry_tangent = nothing)

An OpenPBR Surface, with OpenPBR's names and defaults, rendered as the `Principled` BSDF it
corresponds to in Blender. What Principled has no input for is approximated: the transmission and
subsurface colours tint the base where their weight is the constant one, fuzz is sheen, and coat
darkening, coat anisotropy and the transmission depth and scattering are not modelled.
`thin_film_thickness` is in micrometres, `emission_luminance` in nits.
"""
@description struct OpenPBR <: Material
    base_weight = 1
    base_color = (0.8, 0.8, 0.8)
    base_metalness = 0
    base_diffuse_roughness = 0
    specular_weight = 1
    specular_color = (1, 1, 1)
    specular_roughness = 0.3
    specular_roughness_anisotropy = 0
    specular_ior = 1.5
    transmission_weight = 0
    transmission_color = (1, 1, 1)
    transmission_dispersion_scale = 0
    transmission_dispersion_abbe_number = 20
    subsurface_weight = 0
    subsurface_color = (0.8, 0.8, 0.8)
    subsurface_radius = 1
    subsurface_radius_scale = (1, 0.5, 0.25)
    subsurface_scatter_anisotropy = 0
    fuzz_weight = 0
    fuzz_color = (1, 1, 1)
    fuzz_roughness = 0.5
    coat_weight = 0
    coat_color = (1, 1, 1)
    coat_roughness = 0
    coat_ior = 1.6
    thin_film_weight = 0
    thin_film_thickness = 0.5
    thin_film_ior = 1.4
    emission_luminance = 0
    emission_color = (1, 1, 1)
    geometry_opacity = 1
    geometry_thin_walled = false
    geometry_normal = nothing
    geometry_coat_normal = nothing
    geometry_tangent = nothing
end

const Standard = Union{GLTF, OpenPBR}

nests(::Standard) = true
shader_node(m::Standard) = shader_node(Principled(m))
surface_material(m::Standard) = surface_material(Principled(m))

# ── Inputs combined before they are Principled's ────────────────────────────

"""`a` times `b`: as numbers or colours where both are constants, else as a node graph."""
times(a::Real, b::Real) = a * b
times(a::Real, b::NTuple{3, Real}) = a .* b
times(a::NTuple{3, Real}, b::Real) = a .* b
times(a::NTuple{3, Real}, b::NTuple{3, Real}) = a .* b
times(a, ::Nothing) = a
times(a, b) = node_input(a) * node_input(b)

"""The sampled image of a glTF texture, `nothing` where there is none."""
sampled(::Nothing) = nothing
sampled(t) = TextureNodes.image_texture(t)

"""One channel of a glTF texture (`:x`, `:y`, `:z`, `:alpha`), `nothing` where there is none."""
channel(::Nothing, c::Symbol) = nothing
channel(t, c::Symbol) = getproperty(sampled(t), c)

rgb(c::Tuple{Real, Real, Real, Real}) = (c[1], c[2], c[3])
rgb(c) = c

"""Whether a colour is the constant black: a layer of it is no layer, and its code is left out."""
none(c::NTuple{3, Real}) = all(iszero, c)
none(c) = false

# glTF's alpha: one where the material is opaque, cut at the cutoff where it masks.
gltf_alpha(mode::Symbol, a, cutoff) =
    mode === :opaque ? 1 : mode === :blend ? a : mode === :mask ? cut(a, cutoff) :
    throw(ArgumentError("a glTF alpha mode is :opaque, :mask or :blend, not :$mode"))
cut(a::Real, cutoff::Real) = a >= cutoff ? 1 : 0
cut(a, cutoff) = TextureNodes.greater(node_input(a), cutoff)

"""
    Principled(m::GLTF)

The Principled BSDF of a glTF material, as Blender's glTF importer makes it.
"""
function Principled(g::GLTF)
    base = sampled(g.base_color_texture)
    alpha = times(g.base_color_factor[4], isnothing(base) ? nothing : base.alpha)
    normal = isnothing(g.normal_texture) ? nothing :
             TextureNodes.normal_map(sampled(g.normal_texture); strength = node_input(g.normal_scale))
    coat_normal = isnothing(g.clearcoat_normal_texture) ? nothing : TextureNodes.normal_map(sampled(g.clearcoat_normal_texture))
    sheen = rgb(g.sheen_color_factor)
    emits = !(isnothing(g.emissive_texture) && none(rgb(g.emissive_factor)))
    return Principled(;
        base_color = times(rgb(g.base_color_factor), base),
        metallic = times(g.metallic_factor, channel(g.metallic_roughness_texture, :z)),
        roughness = times(g.roughness_factor, channel(g.metallic_roughness_texture, :y)),
        ior = g.ior,
        alpha = gltf_alpha(g.alpha_mode, alpha, g.alpha_cutoff),
        normal,
        specular_ior_level = times(0.5 * g.specular_factor, channel(g.specular_texture, :alpha)),
        specular_tint = times(rgb(g.specular_color_factor), sampled(g.specular_color_texture)),
        anisotropic = g.anisotropy_strength,
        anisotropic_rotation = g.anisotropy_rotation / 2π,
        transmission_weight = times(g.transmission_factor, channel(g.transmission_texture, :x)),
        coat_weight = times(g.clearcoat_factor, channel(g.clearcoat_texture, :x)),
        coat_roughness = times(g.clearcoat_roughness_factor, channel(g.clearcoat_roughness_texture, :y)),
        coat_normal,
        sheen_weight = isnothing(g.sheen_color_texture) && none(sheen) ? 0 : 1,
        sheen_tint = times(sheen, sampled(g.sheen_color_texture)),
        sheen_roughness = times(g.sheen_roughness_factor, channel(g.sheen_roughness_texture, :alpha)),
        emission_color = times(rgb(g.emissive_factor), sampled(g.emissive_texture)),
        emission_strength = emits ? g.emissive_strength : 0,
        thin_film_thickness = g.iridescence_factor isa Real && g.iridescence_factor == 0 ? 0 : g.iridescence_thickness,
        thin_film_ior = g.iridescence_ior)
end

"""
    Principled(m::OpenPBR)

The Principled BSDF of an OpenPBR Surface, as Blender maps one onto the other.
"""
Principled(o::OpenPBR) = Principled(;
    base_color = constant_one(o.subsurface_weight) ? o.subsurface_color :
                 constant_one(o.transmission_weight) ? o.transmission_color : times(o.base_weight, o.base_color),
    metallic = o.base_metalness,
    roughness = o.specular_roughness,
    ior = o.specular_ior,
    alpha = o.geometry_opacity,
    normal = o.geometry_normal,
    diffuse_roughness = o.base_diffuse_roughness,
    subsurface_weight = o.subsurface_weight,
    subsurface_radius = o.subsurface_radius_scale,
    subsurface_scale = o.subsurface_radius,
    subsurface_anisotropy = o.subsurface_scatter_anisotropy,
    specular_ior_level = times(0.5, o.specular_weight),
    specular_tint = o.specular_color,
    anisotropic = o.specular_roughness_anisotropy,
    tangent = o.geometry_tangent,
    transmission_weight = o.transmission_weight,
    transmission_dispersion_scale = o.transmission_dispersion_scale,
    transmission_dispersion_abbe_number = o.transmission_dispersion_abbe_number,
    coat_weight = o.coat_weight,
    coat_roughness = o.coat_roughness,
    coat_ior = o.coat_ior,
    coat_tint = o.coat_color,
    coat_normal = o.geometry_coat_normal,
    sheen_weight = o.fuzz_weight,
    sheen_roughness = o.fuzz_roughness,
    sheen_tint = o.fuzz_color,
    emission_color = o.emission_color,
    emission_strength = o.emission_luminance,
    thin_film_thickness = o.thin_film_weight isa Real && o.thin_film_weight == 0 ? 0 : times(1000, o.thin_film_thickness),
    thin_film_ior = o.thin_film_ior,
    thin_wall = o.geometry_thin_walled)

"""Whether a weight is the constant one: a colour only Principled's base carries replaces it then."""
constant_one(w::Real) = w == 1
constant_one(w) = false
