# ============================================================================
# Bump mapping — perturbs the shading frame from a height texture.
#
# Used for pbrt-v4's `displacement` / `bumpmap` parameter (and indirectly for
# `normalmap`, which is just a different way of saying "use this texture to
# perturb the shading normal"). Without it, Hikari rendered scenes like Crown
# as smooth surfaces; with it, displacement textures on gold and pearl
# materials show through.
#
# NOT a `BumpMapped{M, T}` WRAPPER material: such a wrapper doubles the
# concrete material type count of any scene that bump-maps some of its
# surfaces — `Conductor{…}` and `BumpMapped{Conductor{…}, TextureRef{…}}` are
# different types, so the per-material closest-hit path compiled a separate
# shader for each. pbrt-v4 does not wrap: `displacement` is a plain field on
# every `Material`, and `Material::GetBxDF` is oblivious to it. Hikari now does
# the same — `displacement(mat)` is a `TexHandle` field, NONE when absent, and
# the perturbation is one predictable branch per hit.
#
# Algorithm follows pbrt-v4 §9.3 "Bump Mapping":
#   1. Sample the height texture h at (u, v), (u+ε, v), (u, v+ε).
#   2. Compute ∂h/∂u, ∂h/∂v by finite differences.
#   3. Perturb the tangent vectors:
#        dpdu' = dpdu + (∂h/∂u) * ns
#        dpdv' = dpdv + (∂h/∂v) * ns
#   4. New shading normal n' = normalize(dpdu' × dpdv'); preserve sign.
#   5. Re-orthonormalize dpdu' against n'.
#
# `dpdvs` is reconstructed at evaluation time as cross(ns, dpdus), mirroring
# how `vp_compute_shading_tangents` builds it in volpath/intersection.jl.
# ============================================================================

# Floor on the finite-difference step when the texture-filter screen-space
# derivatives are zero (e.g. when the integrator does not propagate ray
# differentials). Matches pbrt-v4 materials.h::BumpMap: `if (du == 0) du = .0005f`.
# This is sub-texel on Crown's 1024×1024 bump maps; the earlier 1/256 default
# was 4× the texel size and averaged out the high-frequency ornamental
# engraving on the gold dome (mitra_right_back conductor + bump).
const BUMP_DEFAULT_DELTA = 5f-4

"""
    displacement(mat::Material) -> TexHandle

The material's height-field (`displacement` / `bumpmap` / `normalmap`) handle,
or a NONE handle when the material has no such field. Mirrors pbrt-v4's
`Material::GetDisplacement`.
"""
@generated function displacement(mat::M) where {M <: Material}
    return :displacement in fieldnames(M) ? :(mat.displacement) : :(TexHandle())
end

"""
    set_displacement(mat, h::TexHandle) -> mat′

Rebuild `mat` with its `displacement` field replaced. Host-side only (the pbrt
builder constructs the material first, then attaches the height field). Returns
`mat` unchanged for material types that have no displacement field — pbrt-v4
likewise has no displacement on `MixMaterial` (materials.h: the mix resolves to
a sub-material before `GetDisplacement` is consulted).
"""
@generated function set_displacement(mat::M, h) where {M <: Material}
    fs = fieldnames(M)
    :displacement in fs || return :(mat)
    args = [f === :displacement ? :h : :(getfield(mat, $(QuoteNode(f)))) for f in fs]
    # Through the UnionAll: rebuilding through the CONCRETE type would convert
    # `h` to the existing field's type, and for an unstored texture that
    # conversion is exactly what cannot exist yet.
    return :($(Base.typename(M).wrapper)($(args...)))
end

# ─────────────────────────────────────────────────────────────────────────────
# Height-field gradient → perturbed shading frame
# ─────────────────────────────────────────────────────────────────────────────

# Sample the height (float) at a TextureFilterContext. The bump texture is a
# float texture (the scalar height field); spectrum bump textures fall back to
# luminance via the float branch of imagemap loading in build_pbrt_textures.
# Takes the full filter context, not just uv: pbrt-v4's BumpMap evaluates the
# shifted samples through `shiftedCtx`, which keeps the original dudx/dudy —
# procedural textures (CheckerboardTexture) filter over that footprint.
@propagate_inbounds _bump_height(bump::TexHandle, materials, tfc::TextureFilterContext) =
    eval_handle(materials, bump, tfc)

"""
    perturb_bump_frame(bump, materials, ns, dpdu, dpdv, dndu, dndv, ng, tfc)
        -> (ns_perturbed, dpdus_perturbed)

Implements pbrt-v4 materials.h:BumpMap exactly:
    dpdu' = dpdu + (∂h/∂u) * n + h * dndu
    dpdv' = dpdv + (∂h/∂v) * n + h * dndv

`dpdu` and `dpdv` are the UNNORMALIZED surface partial derivatives.
pbrt-v4 (shapes.h:959 `ss = isect.dpdu`) feeds the BumpMap formula the
raw geometric ∂p/∂u; the tilt of n_p depends on the ratio |dhdu|/|dpdu|.
Passing the unit-length shading tangent over-tilted n_p by 2-60× — that
was the bumped-gold leak into the crown interior pinned by
`shadow_bumpgold_dome_over_velvet`.

`ng` is the geometric (face) normal — pbrt-v4 `SetShadingGeometry` flips the
new bumped normal against `ng`, not against the original interpolated `ns`.
"""
@propagate_inbounds function perturb_bump_frame(bump::TexHandle, materials,
                                                ns::Vec3f,
                                                dpdu::Vec3f, dpdv::Vec3f,
                                                dndu::Vec3f, dndv::Vec3f,
                                                ng::Vec3f,
                                                tfc::TextureFilterContext)
    uv = tfc.uv
    δu = 0.5f0 * (abs(tfc.dudx) + abs(tfc.dudy))
    δu == 0f0 && (δu = BUMP_DEFAULT_DELTA)
    δv = 0.5f0 * (abs(tfc.dvdx) + abs(tfc.dvdy))
    δv == 0f0 && (δv = BUMP_DEFAULT_DELTA)
    # pbrt-v4 materials.h:BumpMap shiftedCtx — the shifted evaluations keep the
    # original screen-space derivatives so filtered procedural textures see the
    # same footprint at all three sample points.
    h0 = _bump_height(bump, materials, tfc)
    hu = _bump_height(bump, materials,
        TextureFilterContext(Point2f(uv[1] + δu, uv[2]), tfc.dudx, tfc.dudy, tfc.dvdx, tfc.dvdy))
    hv = _bump_height(bump, materials,
        TextureFilterContext(Point2f(uv[1], uv[2] + δv), tfc.dudx, tfc.dudy, tfc.dvdx, tfc.dvdy))
    dhdu = (hu - h0) / δu
    dhdv = (hv - h0) / δv

    # pbrt-v4 materials.h:BumpMap reads `ctx.shading.dpdu`/`dpdv`, NOT the raw
    # geometric partials. For a mesh without vertex tangents those come from
    # shapes.h:962-966 + SetShadingGeometry:
    #
    #     ts = Cross(ns, dpdu);  ss = Cross(ts, ns)
    #
    # i.e. `ss` is dpdu PROJECTED perpendicular to the shading normal and `ts` is
    # perpendicular to both — each UNNORMALIZED, so eq. 9.20 keeps the
    # |dhdu|/|dpdu| ratio that sets the tilt.
    #
    # On a flat surface the interpolated ns is already perpendicular to dpdu, so
    # the projection is a no-op and raw partials happen to be right. On a CURVED
    # surface it is not, and the error grows with curvature. Measured with the
    # same bump map, amplitude and material on both: a flat quad matched pbrt at
    # tile 0.0008 / energy 0.9999 while the sphere sat at 0.4091 / 0.918.
    #
    # Note `dpdv` is not used: pbrt builds the shading frame from dpdu alone.
    ts = cross(ns, dpdu)
    ts_len_sq = dot(ts, ts)
    if ts_len_sq > 1f-16
        ss_s = cross(ts, ns)
        ts_s = ts
    else
        # Degenerate (dpdu parallel to ns): pbrt falls back to CoordinateSystem.
        ss_s, ts_s = coordinate_system(ns)
    end

    # pbrt-v4 materials.h:BumpMap eq. 9.20
    dpdu_p = ss_s + dhdu * ns + h0 * dndu
    dpdv_p = ts_s + dhdv * ns + h0 * dndv

    # pbrt-v4 interaction.cpp:184-187 — `Normal3f ns(Normalize(Cross(dpdu, dpdv)))`
    # then `SetShadingGeometry(ns, dpdu, dpdv, ..., false)`, which inside
    # SetShadingGeometry runs `shading.n = FaceForward(shading.n, n)` — i.e.
    # flips the new bumped normal so it sits on the same side of the surface
    # as the GEOMETRIC normal (`n`), not the interpolated `shading.n`.
    n_p_raw = cross(dpdu_p, dpdv_p)
    n_len = sqrt(dot(n_p_raw, n_p_raw))
    n_p = n_len > 1f-10 ? n_p_raw / n_len : ns
    if dot(n_p, ng) < 0f0
        n_p = -n_p   # FaceForward(bumped_ns, geometric_n)
    end
    # Hikari's BSDFs build their local frame from (ns, dpdus) via
    # shading_frame, which Gram-Schmidts internally; pbrt-v4 does the same
    # in BSDFFrame::FromXZ. So we hand back the raw bumped tangent — no
    # extra orthonormalization, matching pbrt-v4 exactly.
    return n_p, dpdu_p
end

# The bump perturbation is applied at intersection time by
# `get_perturbed_shading_frame` (see dispatch.jl), NOT inside the BSDF: the
# path integrator's own `cos_theta = dot(wi, work.ns)` factor has to see the
# perturbed normal too, or bumps on Conductor surfaces vanish: perturbing
# inside the BSDF renders Crown's gold dome smooth.
