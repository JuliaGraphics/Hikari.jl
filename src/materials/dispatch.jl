# ============================================================================
# Spectral Material Dispatch (type-stable dispatch over StaticMultiTypeSet)
# ============================================================================


"""
    get_perturbed_shading_frame(materials, idx, ns, dpdus, dpdu, dpdv, dndu, dndv, ng, tfc) -> (ns', dpdus')
    get_perturbed_shading_frame(materials, idx, ns, dpdus, dpdu, dpdv, ng, bump_inputs) -> (ns', dpdus')

Type-stable dispatch that returns the bump-perturbed shading frame for a
material. Materials without a `displacement` height field (or with an unset
one) return the frame untouched. `bump_inputs()` returns `(dndu, dndv, tfc)`
and is called only for a material with a height field, so a hit on one without
pays nothing for the screen-space differentials.

Here and not inside the material's `sample_bsdf_spectral`, where the
perturbation never reaches the path-integrator's `cos_theta = dot(wi,
work.ns)` factor, so bumps on Conductor surfaces vanish (Crown's gold
dome rendered smooth even with bump mapping active). Hoisting it to the
intersection point lets `work.ns` itself be the perturbed normal — the
BSDF, MIS, and direct-lighting paths all see the same shading frame.
"""
@propagate_inbounds get_perturbed_shading_frame(
    materials::DeviceSet, idx::SetKey,
    ns::Vec3f, dpdus::Vec3f, dpdu::Vec3f, dpdv::Vec3f,
    dndu::Vec3f, dndv::Vec3f,
    ng::Vec3f, tfc::TextureFilterContext
) = get_perturbed_shading_frame(materials, idx, ns, dpdus, dpdu, dpdv, ng, () -> (dndu, dndv, tfc))

@propagate_inbounds function get_perturbed_shading_frame(
    materials::DeviceSet, idx::SetKey,
    ns::Vec3f, dpdus::Vec3f, dpdu::Vec3f, dpdv::Vec3f,
    ng::Vec3f, bump_inputs::F
)::Tuple{Vec3f, Vec3f} where {F}
    # A surface whose material is not in the set has no displacement to read, so
    # the frame is the geometric one. Same guard, same reason as
    # `surface_transparency_dispatch`: a null-material medium boundary carries an
    # INVALID index, and on a scene whose only object is a volume the material
    # set is EMPTY, where `with_index` expands to `error(...)` — unraisable in a
    # GPU kernel, so the ray simply disappeared. This is the call on the medium
    # side of the trace: it is what killed a ray that had already crossed INTO
    # the volume, on the far wall of the same cube.
    Raycore.is_valid(idx) || return (ns, dpdus)
    return with_index(perturb_shading_frame_impl, materials, idx,
                      materials, ns, dpdus, dpdu, dpdv, ng, bump_inputs)
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
    ng::Vec3f, bump_inputs::F,
) where {F}
    h = displacement(mat)
    is_none(h) && return (ns, dpdus)
    dndu, dndv, tfc = bump_inputs()
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
    surface_transparency_dispatch(materials, idx::SetKey, tfc) -> Vec3f

What the material `idx` lets straight through at a hit (`surface_transparency`). `tfc` is the
hit's point-sampled `TextureFilterContext(uv, face_idx, bary)`: a vertex-colour alpha needs the
face, not only the UV. A view of the materials (`shade_set`) holds the types that can be cut out;
any other lets nothing through.
"""
@propagate_inbounds function surface_transparency_dispatch(
    materials::DeviceSet, idx::SetKey, tfc::TextureFilterContext
)::Vec3f
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
    Raycore.is_valid(idx) || return Vec3f(0)
    return with_index(surface_transparency, materials, idx, materials, tfc)
end

"""The coverage of the material `idx` at a hit: 1 opaque, 0 letting everything through."""
@propagate_inbounds surface_alpha_dispatch(materials::DeviceSet, idx::SetKey, tfc::TextureFilterContext) =
    1f0 - _avg(surface_transparency_dispatch(materials, idx, tfc))


"""
    may_cut(T) -> Bool

Whether a material of device type `T` can let a ray through where it is hit: one with a
`surface_transparency` of its own. Every other type is opaque wherever it is, so a shadow ray needs
no code for it (`shade_set`).
"""
may_cut(T::Type) = which(surface_transparency, Tuple{T, Any, TextureFilterContext}) !==
                   which(surface_transparency, Tuple{Material, Any, TextureFilterContext})

# ============================================================================
# The materials that have no BSDF of their own
# ============================================================================
#
# The shading pass dispatches a `vp_shade_kernel!` per concrete type in the
# scene's material set, and a mix's kernel dispatches its component over all of
# them, so every one needs BSDF methods to compile — including the types that
# never actually shade a hit:
#
#   * `MixMaterial` is re-pointed at one of its components in its own kernel
#     (`resolve_mix_material` in `vp_shade_surface_hit!`), so the mix itself is
#     never shaded.
#   * `Emission` lives in `MediumInterface.emission`, not in the material slot;
#     it only reaches the material set if something pushes it directly.
#
# These return no sample and no contribution rather than something plausible.
# The previous version was a gray Lambertian with albedo 0.5, dispatched on
# `::Material` — as a catch-all it silently absorbed every material converted to
# the pbrt-v4 `Material::GetBxDF` pattern, whose BSDF methods had moved onto the
# BxDF carrier, and turned a specular dielectric medium boundary into a diffuse
# wall. That took two months to notice precisely because a gray surface looks
# like a surface. A black one does not.
const BSDFLessMaterial = Union{Emission, MixMaterial}

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
# single concrete type and therefore a single shading kernel.
Raycore.maybe_convert_field(dhv::Raycore.MultiTypeSet, m::Material) =
    to_device_material(dhv, m)
