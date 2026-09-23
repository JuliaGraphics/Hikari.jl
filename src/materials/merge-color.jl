# ── Putting a colour on a material ──────────────────────────────────────────
#
# Replace whatever a material uses as its colour with `color_tex`, leaving
# everything else about it alone. This is how a plot's `color` reaches a
# user-supplied `material`, and how `FEMMaterial` puts a field on one.
#
# It lives here and not in a frontend because it is a statement about Hikari's
# materials: which of a `Conductor`'s five fields is the colour is this
# package's business, and a second copy of that knowledge anywhere else goes
# stale the moment a material gains a field.

# Every one of these passes the material's fields positionally, so a field ADDED
# to a Hikari material silently turns the call into a `MethodError` — and one
# raised here surfaces as "failed to resolve trace_renderobject" from the compute
# graph, several layers from the cause. `displacement` was added to all six and
# none of them were updated, so every plot carrying a colour texture failed.
function merge_color_with_material(color_tex, material::Diffuse)
    Diffuse(color_tex, material.σ, material.displacement)
end

function merge_color_with_material(color_tex, material::Mirror)
    Mirror(color_tex, material.displacement)
end

function merge_color_with_material(color_tex, material::Dielectric)
    Dielectric(
        material.Kr, color_tex,
        material.u_roughness, material.v_roughness,
        material.index, material.remap_roughness, material.displacement
    )
end

function merge_color_with_material(color_tex, material::Conductor)
    Conductor(material.eta, material.k, material.roughness, color_tex,
                     material.remap_roughness, material.displacement)
end

function merge_color_with_material(color_tex, material::CoatedDiffuse)
    CoatedDiffuse(
        color_tex, material.u_roughness, material.v_roughness, material.thickness,
        material.eta, material.albedo, material.g, material.max_depth, material.n_samples,
        material.remap_roughness, material.displacement
    )
end

function merge_color_with_material(color_tex, material::ThinDielectric)
    material
end

function merge_color_with_material(color_tex, material::DiffuseTransmission)
    DiffuseTransmission(
        color_tex, material.transmittance, material.scale, material.displacement
    )
end

function merge_color_with_material(color_tex, material::CoatedDiffuseTransmission)
    CoatedDiffuseTransmission(
        color_tex, material.transmittance, material.u_roughness, material.v_roughness, material.thickness,
        material.eta, material.albedo, material.g, material.max_depth, material.n_samples,
        material.remap_roughness, material.displacement
    )
end

function merge_color_with_material(color_tex, material::CoatedConductor)
    material
end

function merge_color_with_material(color_tex, material::MediumInterface)
    merged_inner = merge_color_with_material(color_tex, material.material)
    MediumInterface(merged_inner; inside=material.inside, outside=material.outside, emission=material.emission)
end

# Fallback for unknown material types
function merge_color_with_material(color_tex, material::Material)
    @warn "Unknown material type $(typeof(material)), ignoring color"
    material
end

