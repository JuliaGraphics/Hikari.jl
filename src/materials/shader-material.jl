# Ported from Blender Cycles v5.2.1: intern/cycles/kernel/closure/bsdf.h, bsdf_sheen.h,
# bsdf_oren_nayar.h, copyright 2011-2022 Blender Foundation, SPDX-License-Identifier: Apache-2.0;
# and bsdf_microfacet.h, adapted by Cycles from Open Shading Language, copyright 2009-2010 Sony
# Pictures Imageworks Inc., et al. All Rights Reserved, copyright 2011-2022 Blender Foundation,
# SPDX-License-Identifier: BSD-3-Clause. Modified: rewritten in Julia, spectral. See
# LICENSES/Apache-2.0.txt and LICENSES/BSD-3-Clause.txt.

# ============================================================================
# ShaderMaterial: a material that is a shader graph
# ============================================================================

"""
    ShaderMaterial(shader)

A material made of a shader graph, as Blender's Material Output takes one: BSDF nodes such as
[`principled_bsdf`](@ref TextureNodes.principled_bsdf), combined by `mix_shader` and
`add_shader`, with node textures and `bump` at any of their inputs. The graph's values are
compiled to one program, and its BSDF nodes are the material's type (`bsdf_nodes`): a kernel holds
the lobes those nodes produce and nothing else, and graphs that differ in their values alone share
it. A shader with transparency holds a second program, of its transparency alone
(`transparency_graph`), which is all an alpha test runs.
"""
struct ShaderMaterial{ProgT, AlphaT, Nodes} <: Material
    program::ProgT
    alpha::AlphaT
end

ShaderMaterial(shader::ShaderNode) = ShaderMaterial(ShaderGraph(shader))

function ShaderMaterial(graph::ShaderGraph)
    alpha = has_transparency(graph) ? transparency_graph(graph) : nothing
    return ShaderMaterial{ShaderGraph, typeof(alpha), bsdf_nodes(graph)}(graph, alpha)
end

"""
    bind_mesh(mesh, material) -> (mesh, material)

The mesh and material to push for `material` on `mesh`: for a shader that reads mesh attributes,
the mesh in triangles and the shader with the attributes of its corners; otherwise both as they
are. `push!(scene, mesh, material)` does this; call it yourself to update a material in a slot.
"""
bind_mesh(mesh::GeometryBasics.Mesh, material::Material) = (mesh, material)
function bind_mesh(mesh::GeometryBasics.Mesh, mat::ShaderMaterial{ShaderGraph})
    isempty(_attribute_cores(mat.program)) && return (mesh, mat)
    triangles = GeometryBasics.mesh(GeometryBasics.expand_faceviews(mesh); facetype = GLTriangleFace)
    return (triangles, ShaderMaterial(bind_attributes(mat.program, triangles)))
end

# Both programs are placed with one `Places`, so the images and attributes the transparency reads
# are stored once.
function to_device_material(dhv, mat::ShaderMaterial{ShaderGraph, A, M}) where {A, M}
    placed = Places()
    program = shader_program(dhv, mat.program, placed)
    alpha = shader_program(dhv, mat.alpha, placed)
    return ShaderMaterial{typeof(program), typeof(alpha), M}(program, alpha)
end

"""
    update_item(dhv, old::ShaderMaterial, new::ShaderMaterial)

A stored shader material updated to a new graph: each program where `old`'s is, with the images
and attributes it reads where that one's are (`updated_program!`). The set takes no array and
keeps its layout, so the scene keeps its recorded plans.

A graph of other node families or other BSDF nodes is another material type, compiled into
other kernels; it is pushed, not updated.
"""
function Raycore.update_item(dhv::Raycore.MultiTypeSet, old::ShaderMaterial{P, A, M},
                             new::ShaderMaterial{ShaderGraph, AG, M}) where {P <: ShaderProgram, A, AG, M}
    # Both compiled, and so checked, before anything is placed: compiling stores nothing.
    program, alpha = compiled_over(old.program, new.program), compiled_over(old.alpha, new.alpha)
    placed = Places()
    return ShaderMaterial{P, A, M}(updated_program!(dhv, old.program, program, placed),
                                   updated_program!(dhv, old.alpha, alpha, placed))
end

function Raycore.update_item(::Raycore.MultiTypeSet, ::ShaderMaterial{<:ShaderProgram, A, M},
                             ::ShaderMaterial{ShaderGraph, A2, M2}) where {A, M, A2, M2}
    error("the updated shader has other BSDF nodes than the one in its slot, so it is another material type: " *
          "push it as a new material")
end

"""`g` compiled to replace the program `old`: of `old`'s node families, or the material would change type."""
function compiled_over(::ShaderProgram{F}, g::ShaderGraph) where {F}
    c = compile_shader(g.closures)
    c isa CompiledProgram{F} ||
        error("the updated shader uses other node families than the one in its slot, so it is another material " *
              "type: push it as a new material")
    return c
end
compiled_over(::Nothing, ::Nothing) = nothing
compiled_over(old, new) =
    error("the updated shader differs from the one in its slot in transparency, so it is another material type: " *
          "push it as a new material")

updated_program!(dhv::Raycore.MultiTypeSet, old::ShaderProgram{F}, c::CompiledProgram{F}, placed::Places) where {F} =
    ShaderProgram(c, last(updated_program!(dhv, (program_slot(dhv, old), Int(old.idx)), c, placed)))
updated_program!(::Raycore.MultiTypeSet, ::Nothing, ::Nothing, ::Places) = nothing

@propagate_inbounds function shader_context(work, camera, samples_per_pixel::Int32)
    dpdx, dpdy = approximate_dp_dxy(work.pi, work.n, camera, samples_per_pixel)
    dudx, dudy, dvdx, dvdy = compute_uv_derivatives(work.dpdu, work.dpdv, dpdx, dpdy)
    tfc = TextureFilterContext(work.uv, dudx, dudy, dvdx, dvdy, work.face_idx, work.bary, work.pi, work.ns)
    tangent_sign = dot(cross(work.ns, work.dpdus), work.dpdv) < 0f0 ? -1f0 : 1f0
    p_camera, window = camera_coordinates(camera, work.pi)
    return ShaderContext(tfc, work.wo, dpdx, dpdy, work.dpdus, work.n, tangent_sign, p_camera, window)
end

"""
    material_bxdf(mat, table, textures, work, camera, samples_per_pixel, lambda, regularize)

The BSDF of `mat` at the hit `work`: `get_bxdf` with the hit's texture filter context and shading
frame, and for a shader the lobes its BSDF nodes produce there from what its program computes.
"""
@propagate_inbounds function material_bxdf(mat, table, textures, work, camera, samples_per_pixel::Int32,
                                           lambda::Wavelengths, regularize::Bool)
    tfc = compute_texture_filter_context(work, camera, samples_per_pixel)
    return get_bxdf(mat, table, textures, tfc, lambda, regularize, work.wo, work.ns, work.dpdus)
end

@propagate_inbounds material_bxdf(mi::MediumInterface, table, textures, work, camera, samples_per_pixel::Int32,
                                  lambda::Wavelengths, regularize::Bool) =
    material_bxdf(mi.material, table, textures, work, camera, samples_per_pixel, lambda, regularize)

@propagate_inbounds function material_bxdf(mat::ShaderMaterial{P, T, M}, table, textures, work, camera,
                                           samples_per_pixel::Int32, lambda::Wavelengths,
                                           regularize::Bool) where {P, T, M <: Tuple}
    sc = shader_context(work, camera, samples_per_pixel)
    produced = shader_lobes(M, shader_programs(textures, mat.program)[mat.program.idx], textures, sc)
    return lobe_bsdf(M, produced, table, lambda, regularize)
end

"""
    add_emission!(pixel_L, bsdf, work)

Add what the surface at `work` emits towards the path. Only a shader emits this way: the other
emitters are area lights, which the emitter kernel handles.
"""
@inline add_emission!(pixel_L, bsdf, work) = nothing
"""Whether a hit on a material can add what it emits (`add_emission!`): a shader with an Emission node."""
emits(mat::Material) = emits(typeof(mat))
emits(::Type{<:Material}) = false
emits(::Type{<:MediumInterface{M}}) where {M} = emits(M)
emits(::Type{<:ShaderMaterial{P, T, M}}) where {P, T, M} = emits(M)
@propagate_inbounds function add_emission!(pixel_L, bsdf::LobeBSDF, work)
    is_black(bsdf.emission) && return nothing
    L = work.beta * bsdf.emission / average(work.r_u)
    accumulate_spectrum!(pixel_L, (work.pixel_index - Int32(1)) * Int32(4), L)
    return nothing
end
@inline _uplift(table, c::Vec3f, lambda) = uplift_rgb(table, RGBSpectrum(c[1], c[2], c[3]), lambda)
"""A positive colour that is not a reflectance, such as a mean free path, lifted by its shape and scaled back."""
@inline function _uplift_unbounded(table, c::Vec3f, lambda)
    m = _max(c)
    m > 0f0 || return SpectralRadiance(0f0)
    return _uplift(table, c / m, lambda) * m
end

"""OpenPBR's dispersion: the Cauchy IOR at `lambda` nm through the d line's `ior` and the Abbe number."""
@inline function _dispersed_ior(ior::Float32, inv_abbe::Float32, lambda::Float32)
    inside = ior < 1f0
    n = inside ? 1f0 / ior : ior
    fac = 1f0 / (1f0 / 0.4861f0^2 - 1f0 / 0.6563f0^2)
    B = (n - 1f0) * inv_abbe * fac
    A = n - B / 0.5876f0^2
    n = A + B / (lambda * 1f-3)^2
    return inside ? 1f0 / n : n
end

# ── Alpha and guides ────────────────────────────────────────────────────────

@propagate_inbounds get_surface_alpha(::ShaderMaterial{P, Nothing}, textures, tfc::TextureFilterContext) where {P} = 1f0
@propagate_inbounds function get_surface_alpha(mat::ShaderMaterial{P, <:ShaderProgram}, textures,
                                               tfc::TextureFilterContext) where {P}
    sc = ShaderContext(tfc, tfc.n, Vec3f(0), Vec3f(0), Vec3f(0), tfc.n, 1f0, Vec3f(0), Point2f(0))
    return clamp(1f0 - _avg(program_transparency(shader_programs(textures, mat.alpha)[mat.alpha.idx], textures, sc)),
                 0f0, 1f0)
end

@propagate_inbounds function guide_material(mat::ShaderMaterial{P, A, M}, materials, tfc, ::Val{D}) where {P, A, M <: Tuple, D}
    sc = ShaderContext(tfc, tfc.n, Vec3f(0), Vec3f(0), Vec3f(0), tfc.n, 1f0, Vec3f(0), Point2f(0))
    lobes, _, _ = shader_lobes(M, shader_programs(materials, mat.program)[mat.program.idx], materials, sc)
    a = reduce(+, map(w -> w.sample_weight > 0f0 ? guide_albedo(w.lobe, w.weight) : Vec3f(0), lobes); init = Vec3f(0))
    return guide_rgb(RGBSpectrum(a[1], a[2], a[3]))
end
