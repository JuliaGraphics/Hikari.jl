# Isoparametric FEM elements as PROCEDURAL geometry.
#
# A quadratic element is a curved surface. Every tool that draws one linearises
# it to its nodes first, and the silhouette then lies — visibly, at any zoom.
# Here the element is not tessellated at all: one AABB goes into the
# acceleration structure, and a ray that enters the box is solved against the
# isoparametric map directly,
#
#     F(ξ₁, ξ₂, t) = S(ξ₁, ξ₂) − o − t·d = 0,
#
# so the silhouette is exact and the normal is the analytic one.
#
# ── How this reaches the renderer ───────────────────────────────────────────
#
# Two interfaces, and neither one is new machinery:
#
# * **Mantle's procedural protocol** (`procedural_candidate` and friends) — the
#   traversal offers a box, `FEMElements` answers it. Triangles are untouched;
#   a scene without FEM geometry carries `nothing` and compiles to the loop it
#   always had.
# * **Hikari's existing triangle record** — a procedural hit comes back as the
#   surface's exact osculating tangent plane, so `vp_compute_surface_geometry`
#   returns analytic answers with no method added to it and no second type in
#   what `closest_hit` returns.
#
# What it deliberately is NOT is a sink the shading writes a colour into. That
# was the first design and it was wrong: writing a colour bypasses the BSDF,
# the lights, the shadows and the path continuation, which makes an albedo
# viewer rather than a renderer. The field reaches the image as a TEXTURE
# (`FEMFieldTexture`), bound to any material's albedo, and every material
# Hikari has composes with it. Material dispatch stays `MultiTypeSet`, untouched.
#
# ── Why `SurfaceInteraction` fits without being bent ────────────────────────
#
#     uv::Point2f      <- ξ, the element's reference coordinate
#     ∂p∂u, ∂p∂v       <- ∂x/∂ξ, the isoparametric Jacobian Newton already has
#     face_idx::UInt32 <- the cell id, selecting the coefficient block
#
# That is not a coincidence. A parametric surface is what `SurfaceInteraction`
# was designed around; a triangle is the degenerate case of one.

using StaticArrays: SVector
import Mantle
import Mantle: procedural_miss, procedural_candidate, procedural_commit, procedural_bary,
               candidate_primitive_index, candidate_object_ray, commit_intersection!
import Raycore

# A quadratic element in 2D has nine monomials, `[(i,j) for j in 0:2 for i in 0:2]`.
const FEM_NMONO = 9

# ── The isoparametric map ───────────────────────────────────────────────────
#
# Moved here from `Mantle/examples/isubd`, which now imports it. It was written
# twice for a while — once for the mesh shader and once for the ray path — and
# the whole point of the example is that both pipelines evaluate the SAME
# element, so it has exactly one home. `test_isubd_mesh.jl` compares it against
# an unedited CPU reference and is the gate on this file.

# Monomial order is the reference's `[(i, j) for j in 0:2 for i in 0:2]`:
# (0,0) (1,0) (2,0) (0,1) (1,1) (2,1) (0,2) (1,2) (2,2).
#
# Written as products of precomputed powers rather than `ξ^e`: `^` with a
# runtime exponent is a library call, and there are only three powers of each
# coordinate to begin with.
@inline function monomials(a, b)
    a0 = one(a); a1 = a; a2 = a * a
    b0 = one(b); b1 = b; b2 = b * b
    return (a0 * b0, a1 * b0, a2 * b0,
            a0 * b1, a1 * b1, a2 * b1,
            a0 * b2, a1 * b2, a2 * b2)
end

"""
    eval_poly(coeffs, cell, ξ₁, ξ₂)

One polynomial of the element, from a flat `FEM_NMONO x ncells` coefficient
block. Indexed flat so the same function serves a `Matrix`, a `LavaArray` and a
raw device pointer in a fragment shader.
"""
@inline function eval_poly(coeffs, cell::Integer, a, b)
    mono = monomials(a, b)
    base = (cell - 1) * FEM_NMONO
    # `ntuple(…, Val(FEM_NMONO))`, not `for m in 1:FEM_NMONO`: `mono` is a TUPLE, and
    # indexing one with a runtime `m` is the same
    # `Unsupported ConstantExpr opcode: LLVMICmp` the base corners hit. `Val`
    # makes every `m` a literal, so each `mono[m]` is a field access.
    return sum(ntuple(m -> coeffs[base + m] * mono[m], Val(FEM_NMONO)))
end
# The monomial basis differentiated, in the same order. `∂/∂a` drops the terms
# with no `a`, `∂/∂b` those with no `b` — written out for the same reason the
# basis is, and because Newton needs them every iteration.
@inline function monomials_da(a, b)
    a0 = one(a); a1 = a
    b0 = one(b); b1 = b; b2 = b * b
    z = zero(a)
    return (z, a0 * b0, 2 * a1 * b0,
            z, a0 * b1, 2 * a1 * b1,
            z, a0 * b2, 2 * a1 * b2)
end

@inline function monomials_db(a, b)
    a0 = one(a); a1 = a; a2 = a * a
    b0 = one(b); b1 = b
    z = zero(b)
    return (z, z, z,
            a0 * b0, a1 * b0, a2 * b0,
            2 * a0 * b1, 2 * a1 * b1, 2 * a2 * b1)
end

"""`(∂u/∂ξ₁, ∂u/∂ξ₂)` of the same polynomial [`eval_poly`](@ref) evaluates."""
@inline function eval_poly_grad(coeffs, cell::Integer, a, b)
    da = monomials_da(a, b); db = monomials_db(a, b)
    base = (cell - 1) * FEM_NMONO
    return (sum(ntuple(m -> coeffs[base + m] * da[m], Val(FEM_NMONO))),
            sum(ntuple(m -> coeffs[base + m] * db[m], Val(FEM_NMONO))))
end

@inline eval_position(cx, cy, cell, a, b) =
    (eval_poly(cx, cell, a, b), eval_poly(cy, cell, a, b))

# ── The error estimator ─────────────────────────────────────────────────────
#
# Sample the deviation of the linear interpolation over the triangle, at all
# three EDGE midpoints and the centroid, for geometry and field, and take
# whichever is worse relative to its tolerance.
#
# Double precision throughout, deliberately: it measures deviations far below
# Float32 noise, and rounding them makes the criterion decide on garbage. The
# positions it feeds into rendering are Float32; the decision is not.

"""The element's surface point: the isoparametric map, displaced by the field."""
@inline function surfacepoint(cx, cy, cf, cell::Integer, a, b, warp)
    return (eval_poly(cx, cell, a, b),
            eval_poly(cy, cell, a, b),
            warp * eval_poly(cf, cell, a, b))
end

"""`∂S/∂ξ₁` and `∂S/∂ξ₂` — the surface's tangents, and Newton's first two columns."""
@inline function surfacetangents(cx, cy, cf, cell::Integer, a, b, warp)
    xa, xb = eval_poly_grad(cx, cell, a, b)
    ya, yb = eval_poly_grad(cy, cell, a, b)
    fa, fb = eval_poly_grad(cf, cell, a, b)
    return ((xa, ya, warp * fa), (xb, yb, warp * fb))
end

"""
The surface's unit normal — the cross product of its two tangents.

Shared, like everything else here: the mesh stage emits it per vertex and the
ray path evaluates it at the exact hit, so the two shade the same surface and
differ only in how finely one of them samples it.
"""
@inline function surfacenormal(cx, cy, cf, cell::Integer, a, b, warp)
    ta, tb = surfacetangents(cx, cy, cf, cell, a, b, warp)
    nx = ta[2]*tb[3] - ta[3]*tb[2]
    ny = ta[3]*tb[1] - ta[1]*tb[3]
    nz = ta[1]*tb[2] - ta[2]*tb[1]
    l = sqrt(nx*nx + ny*ny + nz*nz)
    l < 1e-20 && return Vec3f(0, 0, 1)
    return Vec3f(Float32(nx/l), Float32(ny/l), Float32(nz/l))
end

"""3x3 solve by Cramer's rule — no pivoting, no allocation, and the matrix is tiny."""
@inline function solve3(c1, c2, c3, r)
    F = typeof(c1[1])
    det = c1[1]*(c2[2]*c3[3] - c2[3]*c3[2]) -
          c2[1]*(c1[2]*c3[3] - c1[3]*c3[2]) +
          c3[1]*(c1[2]*c2[3] - c1[3]*c2[2])
    # Singular threshold in the working precision: `1e-20` is below Float32's
    # smallest normal and would never fire, so a degenerate Jacobian would
    # divide instead of bailing.
    abs(det) < (F === Float32 ? F(1e-12) : F(1e-20)) &&
        return (false, zero(F), zero(F), zero(F))
    d1 = r[1]*(c2[2]*c3[3] - c2[3]*c3[2]) -
         c2[1]*(r[2]*c3[3] - r[3]*c3[2]) +
         c3[1]*(r[2]*c2[3] - r[3]*c2[2])
    d2 = c1[1]*(r[2]*c3[3] - r[3]*c3[2]) -
         r[1]*(c1[2]*c3[3] - c1[3]*c3[2]) +
         c3[1]*(c1[2]*r[3] - c1[3]*r[2])
    d3 = c1[1]*(c2[2]*r[3] - c2[3]*r[2]) -
         c2[1]*(c1[2]*r[3] - c1[3]*r[2]) +
         r[1]*(c1[2]*c2[3] - c1[3]*c2[2])
    return (true, d1 / det, d2 / det, d3 / det)
end

"""
    intersect_element(cx, cy, cf, cell, ox,oy,oz, dx,dy,dz, t0, warp)
        -> (hit::Bool, t, ξ₁, ξ₂)

Newton from the element's centre. Eight iterations is generous for a quadratic
map; it converges in three or four wherever it converges at all, and a ray that
misses the element diverges or leaves `[-1,1]²` rather than stalling.

Double precision on purpose. The same reason the estimator is
([[the deviations are below Float32 noise]]) applies here for a different cause:
Newton's correction gets small by construction, and in Float32 the last two
iterations stop meaning anything.
"""
@inline function newton_from(cx, cy, cf, cell::Integer,
                             ox, oy, oz, dx, dy, dz, t0, warp, a0, b0)
    # Everything below is in the precision of the inputs, tolerances included.
    # A single `0.0` or `1e-18` written as a Float64 literal promotes the whole
    # iteration back to double, which on a GPU that runs FP64 at 1/16 rate is
    # not a rounding question but a 16x one.
    F = typeof(t0)
    a = F(a0); b = F(b0); t = F(t0)
    zero3 = (false, zero(F), zero(F), zero(F))
    # Loose enough for a grazing hit, far below any iterate still moving, and
    # scaled to what the precision can actually resolve: a converged Float32
    # residual does not reach 1e-18.
    tol = F === Float32 ? F(1e-10) : F(1e-18)
    for _ in 1:8
        sx, sy, sz = surfacepoint(cx, cy, cf, cell, a, b, warp)
        ta, tb = surfacetangents(cx, cy, cf, cell, a, b, warp)
        # F = S − o − t·d, and J = [∂S/∂ξ₁, ∂S/∂ξ₂, −d]
        r = (-(sx - ox - t * dx), -(sy - oy - t * dy), -(sz - oz - t * dz))
        ok, da, db, dt = solve3(ta, tb, (-dx, -dy, -dz), r)
        ok || return zero3
        a += da; b += db; t += dt
        # Diverged: no root worth chasing outside a box this much bigger than
        # the element. Bailing keeps a missing ray cheap.
        (abs(a) > 4 || abs(b) > 4) && return zero3
    end
    # CONVERGED, not merely in range. Without this an iterate that has not
    # converged but happens to land inside [-1,1]^2 reports a hit, and the
    # picture fills with speckle along every grazing edge — which is what it did.
    # A real root satisfies this to ~1e-16; 1e-9 is loose enough for a grazing
    # hit and far below any iterate that is still moving.
    sx, sy, sz = surfacepoint(cx, cy, cf, cell, a, b, warp)
    rx = sx - ox - t * dx; ry = sy - oy - t * dy; rz = sz - oz - t * dz
    converged = (rx * rx + ry * ry + rz * rz) < tol
    inside = abs(a) <= F(1.0000001) && abs(b) <= F(1.0000001)
    return (converged && inside, t, a, b)
end

# Where Newton starts from, in reference coordinates.
#
# ONE seed is not enough, and the failure is visible rather than subtle: from
# the element's centre alone, rays near the silhouette and at grazing angles
# either diverge (holes) or land on a spurious root (speckle). A quadratic map
# converges in three or four iterations, so nine seeds cost less than they look
# like they do, and the element is small enough that one of them is always in
# the right basin.
const NEWTON_SEEDS = ((-0.6, -0.6), (0.0, -0.6), (0.6, -0.6),
                      (-0.6,  0.0), (0.0,  0.0), (0.6,  0.0),
                      (-0.6,  0.6), (0.0,  0.6), (0.6,  0.6))

"""
    intersect_element(cx, cy, cf, cell, o…, d…, t0, warp) -> (hit, t, ξ₁, ξ₂)

The NEAREST intersection of the ray with the element, over all seeds.

Nearest, not first: a ray can cross a warped element more than once, and which
root a single Newton lands on depends on where it started. Taking the smallest
`t` over the seeds is what makes the answer independent of that.
"""
@inline function intersect_element(cx, cy, cf, cell::Integer,
                                   ox, oy, oz, dx, dy, dz, t0, warp)
    F = typeof(t0)
    besthit = false; bestt = zero(F); besta = zero(F); bestb = zero(F)
    # `@nexprs`, not `for … in NEWTON_SEEDS`: iterating a tuple indexes it at a
    # runtime position, which is the `Unsupported ConstantExpr opcode: LLVMICmp`
    # this file has hit three times already. `si` is a literal in each expansion.
    Base.Cartesian.@nexprs 9 si -> begin
        seed_si = NEWTON_SEEDS[si]
        h_si, t_si, a_si, b_si = newton_from(cx, cy, cf, cell, ox, oy, oz, dx, dy, dz,
                                             t0, warp, seed_si[1], seed_si[2])
        if h_si && t_si > F(1e-4) && (!besthit || t_si < bestt)
            besthit = true; bestt = t_si; besta = a_si; bestb = b_si
        end
    end
    return (besthit, bestt, besta, bestb)
end

"""
    element_aabb(cx, cy, cf, cell; warp, n, pad) -> Mantle.AABB

Conservative bounding box for one element — the whole of its geometry as far as
the acceleration structure is concerned.

Sampled on a grid and padded, which is honest but crude. A BERNSTEIN basis would
give it exactly and for free — the convex hull of the control points bounds the
surface — and is also the better-conditioned basis above roughly cubic. That is
one change serving both pipelines, and it is the right next step here.
"""
function element_aabb(cx, cy, cf, cell::Integer; warp, n = 24, pad = 0.02f0)
    lo = fill(Inf32, 3); hi = fill(-Inf32, 3)
    for i in 0:n, j in 0:n
        a = 2i / n - 1; b = 2j / n - 1
        p = surfacepoint(cx, cy, cf, cell, a, b, warp)
        for k in 1:3
            lo[k] = min(lo[k], Float32(p[k])); hi[k] = max(hi[k], Float32(p[k]))
        end
    end
    return Mantle.AABB(Point3f(lo...) .- pad, Point3f(hi...) .+ pad)
end

# ── The procedural payload ──────────────────────────────────────────────────

"""
    FEMElements(cx, cy, cf, warp, metadata)

The scene's FEM geometry, as the traversal sees it: three coefficient blocks
(`x`, `y`, and the field), the out-of-plane scale the field is drawn at, and
per-cell `metadata` carrying the material.

This is the value that goes on `VulkanTLAS.procedural`. It reaches the kernel
by `Adapt`, like every other device array on the accel.
"""
struct FEMElements{C, M}
    cx::C
    cy::C
    cf::C
    warp::Float32
    metadata::M
end

Adapt.adapt_structure(to, e::FEMElements) = FEMElements(
    Adapt.adapt(to, e.cx), Adapt.adapt(to, e.cy), Adapt.adapt(to, e.cf),
    e.warp, Adapt.adapt(to, e.metadata))

"""
The traversal's running best hit.

The loop carries this ITSELF rather than reading it back, because an inline ray
query carries no hit attributes for a generated intersection —
`get_barycentrics` is triangles only. Generating only an IMPROVING `t` is what
makes this local best the one traversal finally commits.
"""
struct FEMHit
    t::Float32
    ξ₁::Float32
    ξ₂::Float32
    cell::UInt32
end

@inline procedural_miss(::FEMElements) = FEMHit(1f30, 0f0, 0f0, UInt32(0))

"""
The precision the solve runs in: the coefficients'.

A property of the data rather than a constant, and it decides the frame time
here. This GPU runs FP64 at 1/16 of FP32, and the Newton is almost entirely
arithmetic — nine seeds of eight 3x3 Cramer solves per candidate box, per ray,
per bounce. `Float64` coefficients are what the reference comparison wants;
`Float32` ones are what an interactive viewport wants, and asking for it is
`Float32.(coefficients)` at the call site instead of a flag here.
"""
@inline femfloat(e::FEMElements{<:AbstractArray{T}}) where {T} = T

@inline function procedural_candidate(e::FEMElements, best::FEMHit)
    F = femfloat(e)
    cell = candidate_primitive_index()
    o, d = candidate_object_ray()
    ox = F(o[1]); oy = F(o[2]); oz = F(o[3])
    dx = F(d[1]); dy = F(d[2]); dz = F(d[3])
    warp = F(e.warp)
    # A `t` to start Newton from: the element centre projected onto the ray.
    c1, c2, c3 = surfacepoint(e.cx, e.cy, e.cf, cell, zero(F), zero(F), warp)
    t0 = (c1 - ox) * dx + (c2 - oy) * dy + (c3 - oz) * dz
    h, t, a, b = intersect_element(e.cx, e.cy, e.cf, cell,
                                   ox, oy, oz, dx, dy, dz, t0, warp)
    if h && Float32(t) < best.t
        commit_intersection!(t)
        return FEMHit(Float32(t), Float32(a), Float32(b), UInt32(cell))
    end
    return best
end

# ── The committed primitive ─────────────────────────────────────────────────
#
# A procedural hit comes back as a TRIANGLE — and not as a stand-in for one.
# It is the surface's exact osculating tangent plane at the hit:
#
#     vertices = (p, p + ∂p/∂u, p + ∂p/∂v)
#     uv       = (ξ, ξ + (1,0), ξ + (0,1))
#     normals  = the analytic normal, at all three
#
# Run Hikari's own helpers on that and every one of them returns the exact
# analytic answer, because the record was built from the analytic quantities:
#
#   `vp_compute_partial_derivatives` solves Cramer on (δuv, δp). With the layout
#     above δuv is the identity, so it returns ∂p/∂u and ∂p/∂v unchanged.
#   `vp_compute_geometric_normal` takes the cross product of the two edges,
#     which are those same two tangents.
#   `vp_compute_shading_normal` interpolates three identical normals.
#   `vp_compute_uv_barycentric` at barycentric (1,0,0) returns uv₁ = ξ.
#
# So there is no `FEMPrimitive`, no method added to any of those helpers, and no
# `Union` in what `closest_hit` returns. Nothing downstream can tell this ray
# met a box. That is what the design called for — the intersection reports
# `(t, ξ)` into the slots every other primitive reports uv into — and the
# separate primitive type it briefly grew instead was the mistake: it made every
# consumer of `closest_hit` infer a union, in triangles-only scenes too.

"""
ξ mapped to `[0,1]²`.

The element's own coordinate is `[-1,1]²`, but `uv` is what the rest of the
renderer samples textures with, so an image or a checkerboard bound to this
surface has to behave. [`FEMFieldTexture`](@ref) maps it back.
"""
@inline fem_uv(ξ₁, ξ₂) = Point2f(0.5f0 * (ξ₁ + 1f0), 0.5f0 * (ξ₂ + 1f0))

@inline function procedural_commit(e::FEMElements, best::FEMHit, prim_idx, t, empty)
    # `cell == 0` means the traversal reports a generated commit that this
    # shader never recorded. It should be unreachable — only `procedural_candidate`
    # generates one, and it always records what it generated — but the read below
    # indexes a device buffer, so being wrong about that is an out-of-bounds GPU
    # read rather than a wrong pixel. The miss primitive is the honest answer.
    best.cell == UInt32(0) && return empty
    cell = Int(best.cell)
    F = femfloat(e)
    a = F(best.ξ₁); b = F(best.ξ₂); warp = F(e.warp)
    px, py, pz = surfacepoint(e.cx, e.cy, e.cf, cell, a, b, warp)
    ta, tb = surfacetangents(e.cx, e.cy, e.cf, cell, a, b, warp)
    n = surfacenormal(e.cx, e.cy, e.cf, cell, a, b, warp)

    p = Point3f(px, py, pz)
    # Chain rule for the ξ → uv remap: ξ = 2u − 1, so ∂p/∂u = 2 ∂p/∂ξ.
    du = Vec3f(2f0 * Float32(ta[1]), 2f0 * Float32(ta[2]), 2f0 * Float32(ta[3]))
    dv = Vec3f(2f0 * Float32(tb[1]), 2f0 * Float32(tb[2]), 2f0 * Float32(tb[3]))
    uv0 = fem_uv(best.ξ₁, best.ξ₂)

    @inbounds md = e.metadata[cell]
    return typeof(empty)(
        SVector{3, Point3f}(p, p + du, p + dv),
        SVector{3, Normal3f}(n, n, n),
        SVector{3, Vec3f}(du, du, du),
        SVector{3, Point2f}(uv0, uv0 + Point2f(1, 0), uv0 + Point2f(0, 1)),
        md)
end

# The hit sits at the primitive's FIRST vertex by construction, so the
# barycentric is `(1,0,0)` and ξ rides in `uv` — see above.
@inline procedural_bary(::FEMElements, ::FEMHit) = SVector{3, Float32}(1f0, 0f0, 0f0)

# ── The field, as a texture ─────────────────────────────────────────────────

# The TYPE lives in `textures/basic.jl`, beside the other texture types, because
# `AnyTexture` is a `const Union` evaluated at load time and has to name it. Its
# behaviour — how it reaches the device and how it evaluates — is here.

# ── Reaching the device ─────────────────────────────────────────────────────
#
# Same two steps every procedural texture takes: survive construction before any
# scene exists (`matparam`), then resolve against the scene's texture store at
# push time (`device_param`). The stored value is a ONE-ELEMENT array holding
# the texture itself, with its coefficients already turned into a `TextureRef` —
# which is how `CheckerboardTexture` does it, minus the nesting.

@inline matparam(x::FEMFieldTexture) = x

function device_param(dhv, t::FEMFieldTexture)
    stored = FEMFieldTexture(Raycore.store_texture(dhv, t.coeffs), t.ramp, t.vmin, t.vmax)
    ref = Raycore.store_texture(dhv, [stored])
    return TexHandle(TexKind.FEM_FIELD, 0f0, _TH_ZERO,
                     Int32(texref_slot(ref)), Int32(ref.idx))
end

"""
The normalized field value at a hit — `uv` carries ξ, `face_idx` carries the cell.
"""
@propagate_inbounds function fem_field_value(textures, t::FEMFieldTexture, uv::Point2f,
                                             cell::Integer)
    coeffs = Raycore.deref(textures, t.coeffs)
    # CLAMPED, and this is not belt-and-braces. A texture is evaluated in every
    # context the renderer has one, and not all of them come from an FEM hit —
    # a bump-filter context, or any surface that is not this one, arrives with
    # `face_idx = 0`. `eval_poly` turns that into `base = -FEM_NMONO` and reads
    # the coefficient buffer at a negative index, which on a device is not a
    # wrong colour but a page fault: VK_ERROR_DEVICE_LOST, with the failure
    # surfacing at whatever is submitted next.
    c = clamp(Int(cell), 1, size(coeffs, 2))
    # Back out of the [0,1] uv to the element's own [-1,1]² — see `fem_uv`.
    a = Float64(2f0 * uv[1] - 1f0)
    b = Float64(2f0 * uv[2] - 1f0)
    v = Float32(eval_poly(coeffs, c, a, b))
    return clamp((v - t.vmin) / (t.vmax - t.vmin + 1f-20), 0f0, 1f0)
end

@propagate_inbounds function fem_ramp(t::FEMFieldTexture, s::Float32)
    # INTERPOLATED between the two neighbouring entries, not the nearest one.
    # Nearest quantises a smooth field into `FEM_RAMP_N` flat bands, and on a
    # curved surface those bands are contours — they read as concentric rings
    # in the shading, which looks like an artifact of the intersection and is
    # not: it is the colour ramp being sampled with a floor().
    f = clamp(s, 0f0, 1f0) * (FEM_RAMP_N - 1)
    i0 = clamp(unsafe_trunc(Int32, f), Int32(0), Int32(FEM_RAMP_N - 2))
    w = f - Float32(i0)
    # `Val`, not a runtime tuple index: indexing a tuple at a runtime position
    # puts a comparison inside an LLVM constant expression and the SPIR-V
    # emitter refuses it outright. Weighting every entry and summing is the
    # same answer with every index a literal.
    return sum(ntuple(Val(FEM_RAMP_N)) do k
        kk = Int32(k)
        wk = kk == i0 + Int32(1) ? (1f0 - w) : (kk == i0 + Int32(2) ? w : 0f0)
        t.ramp[k] * wk
    end)
end

@propagate_inbounds function eval_tex(textures, t::FEMFieldTexture,
                                      tfc::TextureFilterContext)
    return fem_ramp(t, fem_field_value(textures, t, tfc.uv, tfc.face_idx))
end

@propagate_inbounds function eval_tex(textures, t::FEMFieldTexture,
                                      si::SurfaceInteraction)
    return fem_ramp(t, fem_field_value(textures, t, si.uv, si.face_idx))
end

# The one-element store re-read, then evaluated — the `CheckerboardTexture`
# path, which is why `textures` is threaded through: the nested coefficient ref
# is resolved inside.
@propagate_inbounds _th_fem_spec(arr::AbstractArray{<:FEMFieldTexture}, textures, tfc) =
    _th_spectrum(eval_tex(textures, (@inbounds arr[1]), tfc))
@propagate_inbounds _th_fem_spec(arr, textures, tfc) = _TH_ZERO
@propagate_inbounds _th_fem_float(arr, textures, tfc) = _th_fem_spec(arr, textures, tfc).c[1]

# ── The material ────────────────────────────────────────────────────────────
#
# One type decides how a curved element is DRAWN, on every path, and it is the
# material — not a plot type and not a geometry type:
#
#   traced   `push!(scene, mesh, ::FEMMaterial)` puts one AABB per element into
#            the acceleration structure and the solve on `accel.procedural`, so
#            the tracer intersects the isoparametric map exactly.
#   raster   RayMakie keys its overlay pipeline on the material's type, so this
#            one selects the MESH SHADER that subdivides the element adaptively
#            instead of the plain vertex/fragment pair.
#   anywhere else — a backend that knows neither just draws the triangles the
#            plot was given, which is the coarse approximation both of the above
#            exist to improve on.
#
# On the DEVICE it is a `Diffuse`. `to_device_material` unwraps to `surface`, so
# no new BSDF, no per-material closest-hit shader, and Hikari's material
# dispatch (`MultiTypeSet`) is untouched — the field reaches the image as that
# Diffuse's albedo, through [`FEMFieldTexture`](@ref).

"""
    FEMMaterial(surface::Material, cx, cy, cf; warp, colormap, colorrange)
    FEMMaterial(cx, cy, cf; warp, colormap, colorrange)

Curved finite elements: a material that carries its own geometry.

`cx`, `cy`, `cf` are `FEM_NMONO x ncells` coefficient blocks — the element's `x`
and `y` maps and the solution — and `warp` is how far the solution displaces the
element out of plane.

`surface` is **any Hikari material**, and the field is merged into whatever that
material uses as its colour ([`merge_color_with_material`](@ref)). That is the
entire argument for the field being a texture rather than a shading path: it
composes, so a solution can be matte, lacquered, or polished metal without any
of those knowing what an element is.

```julia
FEMMaterial(cx, cy, cf)                          # matte
FEMMaterial(CoatedDiffuse(), cx, cy, cf)         # the field under a clear coat
FEMMaterial(Conductor(roughness = 0.08), cx, cy, cf)   # tinted metal
```

Defaults to `Diffuse()`, which is the material a colour with no material means
everywhere else in the renderer.
"""
struct FEMMaterial{E <: FEMElements, T <: FEMFieldTexture} <: GeneratedGeometry
    "The BSDF, exactly as the caller gave it."
    surface::Material
    "The elements: coefficients and the warp, for the boxes and the solve."
    elements::E
    "The solution, as a texture — the ramp and the range both renderers read."
    field::T
    "Error a RASTERISER may approximate the elements to. The tracer solves them."
    tolerance::Float32
end

# `surface` is ABSTRACTLY typed, deliberately, and the only place in this file
# where that is the right call. Parameterising on it makes `FEMMaterial{Diffuse{…}}`
# and `FEMMaterial{Conductor{…}}` different types, and a frontend that types an
# attribute slot from the first value it sees — Makie does — can then never be
# handed the second. Swapping the material on a live plot is the whole point of
# wrapping one. It costs nothing: the field is read on the HOST, by
# [`femsurface`](@ref); the device never sees a `FEMMaterial`.
#
# `tolerance` is on the material and not on a renderer's own state because the
# material is what both renderers are handed, and it is a property of the
# ELEMENT being drawn: a coarse field and a sharp one do not want the same
# number.

"""
    femsurface(m::FEMMaterial) -> Material

`m`'s BSDF with the field merged into whatever it uses as its colour.

DERIVED rather than stored. The merged material and the texture inside it used
to be two fields that had to agree — the tracer read one, the rasteriser the
other — and two fields that must agree are one field and a function.
"""
femsurface(m::FEMMaterial) = merge_color_with_material(m.field, m.surface)

"""
The coefficients in the precision the solve runs in.

Already-`Float32` data is passed THROUGH rather than copied, so a caller that
rebuilds a material around the same coefficients — a menu swapping the surface,
a slider moving the raster tolerance — hands the renderers the same arrays and
whatever they cached against them still matches.
"""
femcoeffs(c::AbstractArray{Float32}) = c
femcoeffs(c::AbstractArray) = Float32.(c)

function FEMMaterial(surface::Material, cx, cy, cf; warp = 0.6, colormap = :viridis,
                     colorrange = extrema(cf), tolerance = 2.0f-2)
    # Float32 deliberately: `femfloat` takes the solve's precision from the
    # coefficients, and a GPU that runs FP64 at 1/16 rate spends the frame in
    # Newton rather than anywhere interesting.
    c32 = (femcoeffs(cx), femcoeffs(cy), femcoeffs(cf))
    field = FEMFieldTexture(c32[3], colormap; vmin = colorrange[1], vmax = colorrange[2])
    return FEMMaterial(surface, FEMElements(c32..., Float32(warp), nothing),
                       field, Float32(tolerance))
end

FEMMaterial(cx, cy, cf; kw...) = FEMMaterial(Diffuse(), cx, cy, cf; kw...)

# On the device this IS its surface material, field and all — see above.
to_device_material(dhv, m::FEMMaterial) = to_device_material(dhv, femsurface(m))

"""
    tessellate(m::FEMMaterial) -> GeometryBasics.Mesh

The elements as triangles, for a renderer that cannot solve them.

UNIFORM, `n x n` per element, with `n` from the material's `tolerance`: a
quadratic's deviation from its chord falls as `h²`, so a quarter of the
tolerance is twice the segments per side. Uniform because this is the answer
that always works and needs nothing but the element — an ADAPTIVE one splits
where the curvature is, which is better and is what `Mantle/examples/isubd`
does by overriding the raster path with a mesh shader.

The path tracer does not call this: `push!` below puts one box per element into
the acceleration structure and solves the ray, so the traced silhouette is exact
however coarse this is. That difference is the whole demo.
"""
function tessellate(m::FEMMaterial)
    e = m.elements
    warp = Float64(e.warp)
    n = clamp(ceil(Int, sqrt(0.25 / m.tolerance)), 2, 64)
    pos = Point3f[]; nrm = Vec3f[]; uvs = Point2f[]; col = RGB{Float32}[]
    faces = GeometryBasics.GLTriangleFace[]
    span = max(m.field.vmax - m.field.vmin, 1f-8)
    for c in 1:size(e.cf, 2)
        base = length(pos)
        for j in 0:n, i in 0:n
            a = 2i / n - 1; b = 2j / n - 1
            p = surfacepoint(e.cx, e.cy, e.cf, c, a, b, warp)
            push!(pos, Point3f(p[1], p[2], p[3]))
            push!(nrm, surfacenormal(e.cx, e.cy, e.cf, c, a, b, warp))
            push!(uvs, fem_uv(a, b))
            # The field, through the SAME ramp the tracer reads — a colour
            # attribute on the mesh, which is how geometry carries its own
            # colours. Per VERTEX here and per PIXEL there, which is the whole
            # difference between approximating a field and evaluating it.
            v = (Float32(eval_poly(e.cf, c, a, b)) - m.field.vmin) / span
            sp = fem_ramp(m.field, clamp(v, 0f0, 1f0))
            push!(col, RGB{Float32}(sp.c[1], sp.c[2], sp.c[3]))
        end
        at(i, j) = base + j * (n + 1) + i + 1
        for j in 0:(n - 1), i in 0:(n - 1)
            push!(faces, GeometryBasics.GLTriangleFace(at(i, j), at(i + 1, j), at(i + 1, j + 1)))
            push!(faces, GeometryBasics.GLTriangleFace(at(i, j), at(i + 1, j + 1), at(i, j + 1)))
        end
    end
    return GeometryBasics.Mesh(pos, faces; normal = nrm, uv = uvs, color = col)
end

# ── Putting elements in a scene ─────────────────────────────────────────────

"""
    push!(scene, mesh, material::FEMMaterial; transform) -> SceneHandle

Add curved elements to a scene, as procedural geometry.

**The mesh is ignored**, deliberately and not silently: the material carries the
exact geometry, so whatever triangles the plot supplied are the approximation
this exists to replace. A backend that does not know `FEMMaterial` draws them
instead, which is the point — one plot, and the fidelity is whatever the
renderer can do.

The counterpart of `push!(scene, ::Mesh, ::Material)`, through the same door:
one `MediumInterface` per material, metadata baked per primitive, one BLAS
pushed with a transform. What differs is that the BLAS holds boxes.
"""
# Both spellings, because the generic mesh method is equally specific on its
# middle argument and Julia calls that a tie.
Base.push!(scene::Scene, ::GeometryBasics.Mesh, material::FEMMaterial; kw...) =
    push!(scene, material; kw...)
Base.push!(scene::Scene, ::Any, material::FEMMaterial; kw...) =
    push!(scene, material; kw...)

function Base.push!(scene::Scene, material::FEMMaterial;
                    transform = Mat4f(LinearAlgebra.I))
    e = material.elements
    cx, cy, cf = e.cx, e.cy, e.cf
    warp = Float64(e.warp)
    ncells = size(cf, 2)
    mi = push!(scene, MediumInterface(femsurface(material)))
    accel = scene.accel

    # Per-primitive metadata, the same record a triangle carries.
    # `primitive_index` is the cell, which is what selects the coefficient
    # block — in `FEMFieldTexture`, and in anything else reading `face_idx`.
    meta = [TriangleMeta(UInt32(mi), UInt32(c), UInt32(0)) for c in 1:ncells]

    aabbs = [element_aabb(cx, cy, cf, c; warp) for c in 1:ncells]
    blas = Mantle.build_accel!(Mantle.batchqueue(Mantle.Device())) do c
        Mantle.build_blas_aabb(c, aabbs)
    end
    # `instance_id = 0` means "inherit", so `resolve_mi_idx` falls through to
    # the per-cell metadata above instead of one material for the whole batch.
    handle = push!(accel, blas, transform; instance_id = UInt32(0))

    todev(x) = Mantle.devicearray(accel.backend, x)
    accel.procedural = FEMElements(todev(cx), todev(cy), todev(cf), e.warp, todev(meta))
    return SceneHandle(scene, mi, handle)
end
