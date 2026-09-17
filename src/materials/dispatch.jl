# ============================================================================
# Spectral Material Dispatch (type-stable dispatch over StaticMultiTypeSet)
# ============================================================================


"""
    get_perturbed_shading_frame(materials, idx, ns, dpdus, tfc) -> (ns', dpdus')

Type-stable dispatch that returns the bump-perturbed shading frame for a
material. Materials without a `displacement` height field (or with an unset
one) return the frame untouched.

Here and not inside the material's `sample_bsdf_spectral`, where the
perturbation never reaches the path-integrator's `cos_theta = dot(wi,
work.ns)` factor, so bumps on Conductor surfaces vanish (Crown's gold
dome rendered smooth even with bump mapping active). Hoisting it to the
intersection point lets `work.ns` itself be the perturbed normal — the
BSDF, MIS, and direct-lighting paths all see the same shading frame.
"""
@propagate_inbounds function get_perturbed_shading_frame(
    materials::StaticMultiTypeSet, idx::SetKey,
    ns::Vec3f, dpdus::Vec3f, dpdu::Vec3f, dpdv::Vec3f,
    dndu::Vec3f, dndv::Vec3f,
    ng::Vec3f, tfc::TextureFilterContext
)::Tuple{Vec3f, Vec3f}
    # A surface whose material is not in the set has no displacement to read, so
    # the frame is the geometric one. Same guard, same reason as
    # `get_surface_alpha_dispatch`: a null-material medium boundary carries an
    # INVALID index, and on a scene whose only object is a volume the material
    # set is EMPTY, where `with_index` expands to `error(...)` — unraisable in a
    # GPU kernel, so the ray simply disappeared. This is the call on the medium
    # side of the trace: it is what killed a ray that had already crossed INTO
    # the volume, on the far wall of the same cube.
    Raycore.is_valid(idx) || return (ns, dpdus)
    return with_index(perturb_shading_frame_impl, materials, idx,
                      materials, ns, dpdus, dpdu, dpdv, dndu, dndv, ng, tfc)
end

# One implementation for every material: read the `displacement` handle (a
# compile-time-NONE constant for material types without the field, so the whole
# body folds away there) and perturb only when it names something.
#
# Un-bumped materials want the normalized shading tangent `dpdus`; the bumped
# branch consumes the unnormalized geometric `dpdu` / `dpdv`, per pbrt-v4
# shapes.h:959.
@propagate_inbounds function perturb_shading_frame_impl(
    mat::Material, materials,
    ns::Vec3f, dpdus::Vec3f,
    dpdu::Vec3f, dpdv::Vec3f,
    dndu::Vec3f, dndv::Vec3f,
    ng::Vec3f, tfc::TextureFilterContext,
)
    h = displacement(mat)
    is_none(h) && return (ns, dpdus)
    return perturb_bump_frame(h, materials, ns, dpdu, dpdv, dndu, dndv, ng, tfc)
end


"""
    russian_roulette_spectral(beta, r_u, eta_scale, depth, rr_sample, min_depth=1)

Apply Russian roulette for path termination. Follows pbrt-v4:
  rrBeta = beta * etaScale / r_u.Average()
  q = max(0, 1 - rrBeta.MaxComponentValue())
Returns (should_continue::Bool, new_beta::SpectralRadiance).
"""
@propagate_inbounds function russian_roulette_spectral(
    beta::SpectralRadiance,
    r_u::SpectralRadiance,
    eta_scale::Float32,
    depth::Int32,
    rr_sample::Float32,
    min_depth::Int32=Int32(1)
)
    if depth <= min_depth
        return (true, beta)
    end
    # pbrt-v4: rrBeta = beta * etaScale / r_u.Average()
    r_u_avg = average(r_u)
    rr_beta = if r_u_avg > 1f-10
        beta * eta_scale / r_u_avg
    else
        beta * eta_scale
    end
    max_comp = max_component(rr_beta)
    if max_comp >= 1f0
        return (true, beta)
    end
    q = 1f0 - max_comp
    if rr_sample < q
        return (false, beta)
    else
        return (true, beta / (1f0 - q))
    end
end

"""
    get_surface_alpha_dispatch(materials::StaticMultiTypeSet, idx::SetKey, uv::Point2f) -> Float32

Type-stable dispatch for evaluating surface alpha at a UV point.
Returns alpha ∈ [0, 1] where 0 = fully transparent, 1 = fully opaque.
"""
@propagate_inbounds function get_surface_alpha_dispatch(
    materials::StaticMultiTypeSet, idx::SetKey, uv::Point2f
)::Float32
    # A surface with no material in the set is not alpha-tested: opaque, so the
    # caller proceeds to the null-material branch that owns this case.
    #
    # The guard is the fix for a volume being the only thing in a scene. A
    # volume's bounding cube is `MediumInterface(NullMaterial(); inside = m)`,
    # and a `NullMaterial` is never pushed, so `idx` here is INVALID — and if
    # the volume is the only object, `materials` is EMPTY. `with_index` expands
    # to `error("with_index: empty StaticMultiTypeSet")` for an empty set, which
    # a GPU kernel cannot raise: every ray that hit the cube died right here,
    # producing neither a continuation ray nor an escape, and the volume
    # rendered as a solid black box. With any other material in the scene the
    # same invalid index instead fell into `with_index`'s else-branch and read
    # some ARBITRARY material's alpha — opaque by luck, which is why adding an
    # unrelated sphere "fixed" it.
    Raycore.is_valid(idx) || return 1.0f0
    return with_index(get_surface_alpha, materials, idx, materials, uv)
end


# ============================================================================
# The materials that have no BSDF of their own
# ============================================================================
#
# The shading pass dispatches a `vp_shade_material_kernel!{T}` per concrete type
# in the scene's material set, so every one of them needs BSDF methods to
# compile — including the types that never actually shade a hit:
#
#   * `MixMaterial` is re-pointed at one of its sub-materials at the PUSH site
#     (`resolve_mix_material` in `vp_trace_and_shade_kernel!` and in delta
#     tracking), so a `TypedHitRef{MixMaterial}` is never produced.
#   * `Emissive` lives in `MediumInterface.emission`, not in the material slot;
#     it only reaches the material set if something pushes it directly.
#
# These return no sample and no contribution rather than something plausible.
# The previous version was a gray Lambertian with albedo 0.5, dispatched on
# `::Material` — as a catch-all it silently absorbed every material converted to
# the pbrt-v4 `Material::GetBxDF` pattern, whose BSDF methods had moved onto the
# BxDF carrier, and turned a specular dielectric medium boundary into a diffuse
# wall. That took two months to notice precisely because a gray surface looks
# like a surface. A black one does not.
const BSDFLessMaterial = Union{Emissive, MixMaterial}

@propagate_inbounds sample_bsdf_spectral(
    ::BSDFLessMaterial, ::RGBToSpectrumTable, textures,
    wo::Vec3f, n::Vec3f, dpdus::Vec3f, ::TextureFilterContext,
    ::Wavelengths, sample_u::Point2f, rng::Float32,
    regularize::Bool = false,
) = SpectralBSDFSample()

@propagate_inbounds evaluate_bsdf_spectral(
    ::BSDFLessMaterial, ::RGBToSpectrumTable, textures,
    wo::Vec3f, wi::Vec3f, n::Vec3f, dpdus::Vec3f, ::TextureFilterContext,
    ::Wavelengths, regularize::Bool = false,
) = (SpectralRadiance(), 0f0)

# ============================================================================
# Push-time conversion hook
# ============================================================================
#
# `Raycore.MultiTypeSet` calls `maybe_convert_field` on every item it stores.
# For materials Hikari takes over completely: `to_device_material` rewrites the
# texture-carrying fields into `TexHandle`s (storing whatever needs a slot),
# which is what collapses "same material, different parameter spelling" into a
# single concrete type and therefore a single closest-hit shader.
Raycore.maybe_convert_field(dhv::Raycore.MultiTypeSet, m::Material) =
    to_device_material(dhv, m)
