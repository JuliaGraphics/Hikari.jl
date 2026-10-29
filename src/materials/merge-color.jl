# ── Putting a colour on a material ──────────────────────────────────────────
#
# Replace whatever a material uses as its colour with `color_tex`, leaving everything else about it
# alone. This is how a plot's `color` reaches a user-supplied `material`, and how `FEMMaterial`
# puts a field on one. Which field is the colour is this package's business, so it is said here,
# once per material (`color_field`).

"""The field of a material that is its colour; `nothing` for a material that has none."""
color_field(::Material) = nothing
color_field(::Union{Diffuse, Glass, Glossy, Translucent, Sheen, Toon, SubsurfaceScattering, Transparent, Emission}) = :color
color_field(::Union{Principled, Metallic, OpenPBR}) = :base_color
# glTF's base colour texture multiplies its factor, so a colour put there is the colour.
color_field(::GLTF) = :base_color_texture
# A conductor's colour is its reflectance, as pbrt-v4 gives a conductor one instead of its IOR.
color_field(::Union{Conductor, CoatedDiffuse, DiffuseTransmission, CoatedDiffuseTransmission}) = :reflectance
color_field(::CoatedConductor) = :conductor_reflectance

function merge_color_with_material(color_tex, material::Material)
    name = color_field(material)
    if name === nothing
        @warn "a $(nameof(typeof(material))) has no colour; the plot's colour is ignored"
        return material
    end
    return with_field(material, name, color_tex)
end

# No surface, so nothing for a plot colour to tint.
merge_color_with_material(color_tex, material::NullMaterial) = material

# A mix or sum is tinted where it is made: both of what it combines.
merge_color_with_material(color_tex, m::Union{Mix, Add}) =
    with_field(with_field(m, :a, merge_color_with_material(color_tex, m.a)), :b, merge_color_with_material(color_tex, m.b))

merge_color_with_material(color_tex, m::MediumInterface) =
    MediumInterface(merge_color_with_material(color_tex, m.material); inside = m.inside, outside = m.outside,
                    emission = m.emission, shadows = m.shadows)
