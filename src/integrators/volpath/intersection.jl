# Ray tracing and intersection handling for VolPath
# Handles ray-scene intersection and classifies results into work queues

"Offset to the outgoing side, including Float32 position error at large coordinates."
@inline function offset_surface_origin(p::Point3f,n::Vec3f,d::Vec3f)
    normal=dot(d,n)>0f0 ? n : -n
    distance=max(1f-4,8f0*eps(Float32)*dot(abs.(p),abs.(normal)))
    q=p+normal*distance
    # Round outwards, so an offset never rounds back onto the boundary.
    Point3f(ntuple(3) do a
        x=q[a]; s=normal[a]
        s==0f0 && return x
        x==0f0 && return s>0f0 ? floatmin(Float32) : -floatmin(Float32)
        bits=reinterpret(UInt32,x)
        delta=(x<0f0)==(s<0f0) ? UInt32(1) : typemax(UInt32)
        reinterpret(Float32,bits+delta)
    end)
end

"""
    resolve_mi_idx(instance_id, primitive) -> UInt32

Pick the medium_interface_idx for a hit. `instance_id` is the 5th value
`Raycore.closest_hit` / `Raycore.any_hit` return, the instance's custom index,
on every acceleration structure: `Raycore.TLAS` and the hardware ones alike
(`gl_InstanceCustomIndexEXT` on Vulkan, `user_instance_id` on Metal).

A nonzero index replaces the triangle's per-face `medium_interface_idx`; zero
means "inherit". This is the single place where the per-instance interface
override resolves.
"""
@inline function resolve_mi_idx(instance_id::UInt32, primitive)
    instance_id != UInt32(0) ? instance_id : primitive.metadata.medium_interface_idx
end

# ============================================================================
# Geometry Helpers (shared with PhysicalWavefront)
# ============================================================================

"""
    compute_geometric_normal(primitive) -> Vec3f

Compute geometric normal from a triangle primitive.
"""
@propagate_inbounds function vp_compute_geometric_normal(primitive)
    v0 = primitive.vertices[1]
    v1 = primitive.vertices[2]
    v2 = primitive.vertices[3]
    e1 = v1 - v0
    e2 = v2 - v0
    n = Vec3f(cross(e1, e2)...)
    return normalize(n)
end

"""
    vp_compute_uv_barycentric(primitive, barycentric) -> Point2f

Compute UV coordinates using barycentric coordinates from ray intersection.
"""
@propagate_inbounds function vp_compute_uv_barycentric(primitive, barycentric)
    w, u, v = barycentric[1], barycentric[2], barycentric[3]
    uv0 = primitive.uv[1]
    uv1 = primitive.uv[2]
    uv2 = primitive.uv[3]
    return Point2f(
        w * uv0[1] + u * uv1[1] + v * uv2[1],
        w * uv0[2] + u * uv1[2] + v * uv2[2]
    )
end

"""
    alpha_filter_context(primitive, barycentric, ray_o, ray_d, t_hit) -> TextureFilterContext

What an alpha query at a hit reads: the hit's uv, face and barycentrics, and its position and
shading normal. A shader material's alpha may depend on those (a node program over `xyz()`);
built from uv alone, every shadow ray read its alpha at the origin and a checker cutout cast
no shadow. Point-sampled, as every alpha query is.
"""
@propagate_inbounds function alpha_filter_context(primitive, barycentric, ray_o, ray_d, t_hit)
    ns = vp_compute_shading_normal(primitive, barycentric, vp_compute_geometric_normal(primitive))
    return TextureFilterContext(vp_compute_uv_barycentric(primitive, barycentric), 0f0, 0f0, 0f0, 0f0,
                                primitive.metadata.primitive_index, SVector{3, Float32}(barycentric),
                                Point3f(ray_o + ray_d * t_hit), ns)
end

"""
    vp_compute_partial_derivatives(primitive) -> (dpdu, dpdv)

Compute position partial derivatives (∂p/∂u, ∂p/∂v) from triangle vertices and UVs.
These are used for texture filtering and tangent space construction.
Following pbrt-v4's Triangle::InteractionFromIntersection.
"""
@propagate_inbounds function vp_compute_partial_derivatives(primitive)
    v0 = primitive.vertices[1]
    v1 = primitive.vertices[2]
    v2 = primitive.vertices[3]
    uv0 = primitive.uv[1]
    uv1 = primitive.uv[2]
    uv2 = primitive.uv[3]

    # Compute deltas for partial derivative matrix
    δuv_10 = uv1 - uv0
    δuv_20 = uv2 - uv0
    δp_10 = Vec3f(v1 - v0)
    δp_20 = Vec3f(v2 - v0)

    # Solve for dpdu, dpdv using Cramer's rule
    det = δuv_10[1] * δuv_20[2] - δuv_10[2] * δuv_20[1]

    if abs(det) < 1f-8
        # Degenerate UV mapping - create orthonormal basis from edge
        e1 = normalize(δp_10)
        n = normalize(cross(δp_10, δp_20))
        e2 = cross(n, e1)
        return e1, e2
    end

    inv_det = 1f0 / det
    dpdu = Vec3f(δuv_20[2] * δp_10 - δuv_10[2] * δp_20) * inv_det
    dpdv = Vec3f(-δuv_20[1] * δp_10 + δuv_10[1] * δp_20) * inv_det
    return dpdu, dpdv
end

"""
    vp_compute_normal_derivatives(primitive) -> (dndu, dndv)

Per-triangle shading-normal derivatives — pbrt-v4 BumpMap needs these to
correctly perturb the shading frame on curved surfaces. The bump formula
adds a `displace * dndu` term (eq. 9.20); skipping it makes the perturbation
cancel out across surface curvature, which leaves the Crown's gold dome
looking flat-smooth instead of engraved.

Same Cramer's-rule layout as `vp_compute_partial_derivatives` but on vertex
normals rather than vertex positions. Returns zero vectors if the triangle's
UV mapping is degenerate.
"""
@propagate_inbounds function vp_compute_normal_derivatives(primitive)
    n0 = primitive.normals[1]
    n1 = primitive.normals[2]
    n2 = primitive.normals[3]
    uv0 = primitive.uv[1]
    uv1 = primitive.uv[2]
    uv2 = primitive.uv[3]

    if isnan(n0[1]) || isnan(n1[1]) || isnan(n2[1])
        return Vec3f(0f0, 0f0, 0f0), Vec3f(0f0, 0f0, 0f0)
    end

    δuv_10 = uv1 - uv0
    δuv_20 = uv2 - uv0
    δn_10 = Vec3f(n1[1] - n0[1], n1[2] - n0[2], n1[3] - n0[3])
    δn_20 = Vec3f(n2[1] - n0[1], n2[2] - n0[2], n2[3] - n0[3])

    det = δuv_10[1] * δuv_20[2] - δuv_10[2] * δuv_20[1]
    if abs(det) < 1f-8
        return Vec3f(0f0, 0f0, 0f0), Vec3f(0f0, 0f0, 0f0)
    end

    inv_det = 1f0 / det
    dndu = (δuv_20[2] * δn_10 - δuv_10[2] * δn_20) * inv_det
    dndv = (-δuv_20[1] * δn_10 + δuv_10[1] * δn_20) * inv_det
    return dndu, dndv
end

"""
    vp_compute_shading_tangents(primitive, barycentric, ns, dpdu, dpdv) -> (dpdus, dpdvs)

Compute shading tangent vectors from vertex tangents or geometric derivatives.
Following pbrt-v4's approach: use vertex tangents if available, otherwise
orthonormalize geometric dpdu/dpdv to the shading normal.
"""
@propagate_inbounds function vp_compute_shading_tangents(primitive, barycentric, ns::Vec3f, dpdu::Vec3f, dpdv::Vec3f)
    # Try to get vertex tangents
    t0 = primitive.tangents[1]
    t1 = primitive.tangents[2]
    t2 = primitive.tangents[3]

    # Check if tangents are valid (not NaN)
    has_tangents = !isnan(t0[1]) && !isnan(t1[1]) && !isnan(t2[1])

    if has_tangents
        # Interpolate vertex tangents
        w, u, v = barycentric[1], barycentric[2], barycentric[3]
        dpdus = Vec3f(
            w * t0[1] + u * t1[1] + v * t2[1],
            w * t0[2] + u * t1[2] + v * t2[2],
            w * t0[3] + u * t1[3] + v * t2[3]
        )
        dpdus = normalize(dpdus)
    else
        # Use geometric dpdu, projected onto shading tangent plane
        dpdus = dpdu - ns * dot(ns, dpdu)
        len_sq = dot(dpdus, dpdus)
        if len_sq > 1f-10
            dpdus = dpdus / sqrt(len_sq)
        else
            # Fallback: create orthonormal basis from shading normal
            if abs(ns[1]) > abs(ns[2])
                dpdus = Vec3f(-ns[3], 0f0, ns[1]) / sqrt(ns[1]*ns[1] + ns[3]*ns[3])
            else
                dpdus = Vec3f(0f0, ns[3], -ns[2]) / sqrt(ns[2]*ns[2] + ns[3]*ns[3])
            end
        end
    end

    # Compute dpdvs perpendicular to both ns and dpdus
    dpdvs = cross(ns, dpdus)

    return dpdus, dpdvs
end

"""
    vp_compute_shading_normal(primitive, barycentric, geometric_normal) -> Vec3f

Compute interpolated shading normal from vertex normals.
"""
@propagate_inbounds function vp_compute_shading_normal(primitive, barycentric, geometric_normal::Vec3f)
    n0 = primitive.normals[1]
    n1 = primitive.normals[2]
    n2 = primitive.normals[3]

    # Check if normals are valid (not NaN)
    if isnan(n0[1]) || isnan(n1[1]) || isnan(n2[1])
        return geometric_normal
    end

    w, u, v = barycentric[1], barycentric[2], barycentric[3]
    ns_x = w * n0[1] + u * n1[1] + v * n2[1]
    ns_y = w * n0[2] + u * n1[2] + v * n2[2]
    ns_z = w * n0[3] + u * n1[3] + v * n2[3]

    ns = normalize(Vec3f(ns_x, ns_y, ns_z))

    # Return raw interpolated shading normal WITHOUT flipping.
    # Following pbrt-v4: the shading normal is authoritative for smooth surfaces.
    # The geometric normal is flipped to face the shading normal at the call site.
    return ns
end

"""
    vp_shadow_point(primitive, barycentric, p) -> Point3f

Where shadow rays leave a smooth-shaded triangle: `p` pushed out onto the
tangent planes of the three vertex normals, weighted by the barycentrics
(Hanika, "Hacking the Shadow Terminator", Ray Tracing Gems II, ch. 4; what
Cycles does too).

On a coarse mesh with interpolated normals, the shading normal near the
terminator still faces a light that the flat triangle is already turned away
from. A shadow ray from `p` itself then runs into the mesh's own neighbouring
triangles and the whole triangle turns black, in a sawtooth along the
tessellation. Returns `p` where the triangle has no vertex normals, and where
`p` already lies above every vertex's tangent plane (concave regions).
"""
@propagate_inbounds function vp_shadow_point(primitive, barycentric, p::Point3f)
    n0 = Vec3f(primitive.normals[1])
    n1 = Vec3f(primitive.normals[2])
    n2 = Vec3f(primitive.normals[3])
    isnan(n0[1]) && return p
    h0 = max(dot(Vec3f(primitive.vertices[1] - p), n0), 0f0)
    h1 = max(dot(Vec3f(primitive.vertices[2] - p), n1), 0f0)
    h2 = max(dot(Vec3f(primitive.vertices[3] - p), n2), 0f0)
    return Point3f(p + barycentric[1] * h0 * n0 + barycentric[2] * h1 * n1 + barycentric[3] * h2 * n2)
end

"""
    vp_compute_surface_geometry(primitive, barycentric, ray) -> NamedTuple

Compute all surface geometry needed for material evaluation.
Returns (pi, n, dpdu, dpdv, ns, dpdus, dpdvs, uv).
"""
@propagate_inbounds function vp_compute_surface_geometry(primitive, barycentric, ray_o, ray_d, t_hit)
    # Intersection point
    pi = Point3f(ray_o + ray_d * t_hit)

    # Geometric normal
    n = vp_compute_geometric_normal(primitive)

    # UV coordinates
    uv = vp_compute_uv_barycentric(primitive, barycentric)

    # Position partial derivatives (for texture filtering)
    dpdu, dpdv = vp_compute_partial_derivatives(primitive)

    # Shading normal
    ns = vp_compute_shading_normal(primitive, barycentric, n)
    # pbrt-v4 FaceForward: flip geometric normal to face shading normal.
    # The shading normal from vertex interpolation is authoritative for
    # smooth surfaces; the geometric normal must agree with it.
    n = dot(n, ns) < 0f0 ? -n : n

    # Shading tangent vectors
    dpdus, dpdvs = vp_compute_shading_tangents(primitive, barycentric, ns, dpdu, dpdv)

    return (pi=pi, ps=vp_shadow_point(primitive, barycentric, pi), n=n, dpdu=dpdu, dpdv=dpdv, ns=ns, dpdus=dpdus, dpdvs=dpdvs, uv=uv)
end

# ============================================================================
# Trace Kernel
# ============================================================================
#
# One thread per ray of the round's queue, on any traversal (software BVH, Vulkan
# ray query, Metal intersector). Each thread:
#   1. A ray inside a medium: its hit record goes with it to the medium pipeline,
#      which delivers what survives to the surface as any other hit.
#   2. Otherwise the alpha-test loop, then either the null-material crossing,
#      handled here because a surface without a material has no shading kernel
#      to go to, or a `VPSurfaceHit` into the queue of the hit mesh's material.
#
# `materials` is the scene's `cut_set`: the types a ray may pass, whose
# transparency the alpha test reads, and nothing of the others, so that a material
# of another type does not compile this kernel again.
#
# Nothing else is computed here: the shading kernel expands the record into the
# surface it shades (`vp_shade_surface_hit!`, surface-eval.jl), so the hit is
# written in 40 bytes and the surface computed once. Writing the expanded surface
# here and reading it back there cost more than the shading on an RTX 4000 Ada.

@kernel inbounds = true function vp_trace_kernel!(
    cur, next_ray_queue, escaped_queue, medium_sample_queue,
    material_queues, surface_hits,
    pixel_L, accel, media_interfaces, media, materials, lights, rgb2spec_table,
    bvh_nodes, light_to_bit_trail,
    num_infinite_lights::Int32, num_bvh_lights::Int32, max_depth::Int32,
)
    i = @index(Global)
    if i <= cur.size[1]
        vp_trace!(cur.items[i], Int32(i), cur.items, next_ray_queue, escaped_queue, medium_sample_queue,
                  material_queues, surface_hits,
                  pixel_L, accel, media_interfaces, media, materials, lights, rgb2spec_table,
                  bvh_nodes, light_to_bit_trail, num_infinite_lights, num_bvh_lights, max_depth)
    end
end

@propagate_inbounds function vp_trace!(
    work::VPRayWorkItem, path::Int32, rays, next_ray_queue, escaped_queue, medium_sample_queue,
    material_queues, surface_hits,
    pixel_L, accel, media_interfaces, media, materials, lights, rgb2spec_table,
    bvh_nodes, light_to_bit_trail,
    num_infinite_lights::Int32, num_bvh_lights::Int32, max_depth::Int32,
)
    # A medium ray: delta tracking is bounded by the hit and delivers the surface.
    if has_medium(work.medium_idx)
        work = prepare_medium_work(work, media)
        push!(medium_sample_queue, VPMediumSampleWorkItem(work, Raycore.closest_hit_record(accel, work.ray)))
        return
    end

    # Every surface is opaque to this ray; a cutout is tested after the hit and,
    # where the ray passes, it goes on along the same line past the hit (Cycles
    # moves `tmin` the same way), so the record stays in the queued ray's parameter.
    # What it passed tints it, the throughput the shading kernel reads with it too.
    ray = work.ray
    tinted = false
    for _ in 1:Int32(16)
        rec = Raycore.closest_hit_record(accel, ray)
        if rec.kind == Raycore.HIT_MISS
            push!(escaped_queue, VPEscapedRayWorkItem(work))
            return
        end
        primitive = Raycore.hit_primitive(accel, rec)
        bary = Raycore.hit_barycentric(accel, rec)
        mi = media_interfaces[resolve_mi_idx(rec.custom, primitive)]
        mat_idx = mi.material

        tfc = alpha_filter_context(primitive, bary, ray.o, ray.d, rec.t)
        transparency = surface_transparency_dispatch(materials, mat_idx, tfc)
        through = _avg(transparency)
        if through > 0f0
            # Hashed from where the ray meets the surface: each layer it passes
            # decides apart from the last. It passes with the chance of what the
            # surface lets through, and is tinted by that over the chance, as
            # Cycles' transparent closure tints it.
            rng = pcg32_init(pbrt_hash(Point3f(ray.o + ray.d * rec.t)), pbrt_hash(ray.d))
            alpha_u, _ = pcg32_uniform_f32(rng)
            if alpha_u > 1f0 - through
                work = with_beta(work, work.beta * _uplift(rgb2spec_table, transparency, work.lambda) / through)
                tinted = true
                ray = Raycore.Ray(ray; t_min = nextfloat(rec.t))
                continue
            end
        end

        # Null-material surface (pbrt `Material "interface"` / nullptr): no BSDF,
        # no direct lighting, so no shading kernel to go to. Its emission (a
        # see-through glowing sheet), then the ray goes on at the same depth with
        # the medium swapped, as pbrt-v4's VolPath adds `isect.Le` before it skips
        # a surface without a BSDF.
        if !Raycore.is_valid(mat_idx)
            geom = vp_compute_surface_geometry(primitive, bary, ray.o, ray.d, rec.t)
            wo = -ray.d
            accumulate_arealight_hit!(pixel_L, lights, rgb2spec_table,
                bvh_nodes, light_to_bit_trail, num_infinite_lights, num_bvh_lights,
                primitive.metadata.arealight_id, geom.n, geom.uv, wo,
                work, work.beta, work.r_u, work.r_l, Raycore.area(primitive), rec.t)
            ray_d = -wo
            new_medium = null_crossing_medium(mi, work.medium_idx, ray_d, geom.n)
            new_ray = Raycore.Ray(o=offset_surface_origin(geom.pi, geom.n, ray_d), d=ray_d,
                                  t_max=Inf32, time=0f0)
            push!(next_ray_queue, VPRayWorkItem(
                new_ray, work.depth,
                work.lambda, work.pixel_index,
                work.beta, work.r_u, work.r_l,
                work.prev_intr_p, work.prev_intr_n,
                work.eta_scale,
                work.specular_bounce, work.any_non_specular_bounces,
                new_medium))
            return
        end

        wants_shading(work.depth, max_depth, primitive.metadata.arealight_id, material_queues, mat_idx) || return
        tinted && (rays.beta[path] = work.beta)
        enqueue_surface_hit!(material_queues, surface_hits, mat_idx, VPSurfaceHit(path, rec))
        return
    end
    # 16 alpha-bounces exhausted (extremely unlikely): ray absorbed
    return
end


# ============================================================================
# Shadow Ray Tracing Kernel (with medium transmittance)
# ============================================================================

"""
    trace_shadow_transmittance(accel, media_interfaces, media, materials, rgb2spec_table, origin, dir, t_max, lambda, medium_idx, refract)

Trace a shadow ray computing transmittance through media and the surfaces that let
it through; with `refract`, bending at surfaces that pass shadow rays (see
`snells_window`). Returns (T_ray, r_u, r_l, visible, dir) where `dir` is the
direction it leaves in and:
- T_ray: spectral transmittance
- r_u, r_l: MIS weight accumulators for combining with path weights
- visible: false if ray hits an opaque surface

Following pbrt-v4's TraceTransmittance: transmissive surfaces (MediumInterface) let the ray through,
while opaque surfaces block it. The final contribution is computed as:
    Ld * T_ray / average(path_r_u * r_u + path_r_l * r_l)

The traversal stops at opaque geometry itself and hands back the nearest hits on
the rest (`Raycore.transparent_hits`), which are shaded here in order, as Cycles'
`integrate_transparent_shadow` does: a cutout passes its transparency, a surface
with `shadows` what a dielectric of that index transmits, a null material
everything, changing the medium. Past the hits kept, the ray is traced on from the
last of them.
"""
# Inlined into the shading kernel. Outlined (it was, for an NVIDIA register budget
# that no longer shows: the PR #47 scenes time the same either way on an RTX 4000
# Ada, a Strix Halo and an M5), Apple's GPU compiler crashed linking a CoatedConductor
# kernel under a distant light once a cutout's colour was uplifted in here, as it
# did for the outlined subsurface walk.
@inline function trace_shadow_transmittance(
    accel, media_interfaces, media, materials, rgb2spec_table,
    origin::Point3f, dir::Vec3f, t_max::Float32, lambda::Wavelengths, medium_idx::SetKey,
    refract::Bool = false
)
    # Transmittance and MIS weights following pbrt-v4
    T_ray = SpectralRadiance(1f0)
    r_u = SpectralRadiance(1f0)
    r_l = SpectralRadiance(1f0)
    blocked = (SpectralRadiance(0f0), SpectralRadiance(1f0), SpectralRadiance(1f0), false, dir)

    current_medium = medium_idx
    # The ray traced: from `ray_o` along `dir`, the part past `t_min` up to `t_end`.
    ray_o = origin
    t_min = 0f0
    t_end = t_max
    bent = false
    # Surfaces passed, of at most `max_crossed`: Cycles' transparent bounces.
    crossed = Int32(0)
    max_crossed = Int32(10)

    while true
        th = Raycore.transparent_hits(accel, Raycore.Ray(o = ray_o, d = dir, t_min = t_min, t_max = t_end),
                                      0xff, Val(4), max_crossed - crossed)
        th.blocked && return blocked
        # Nearest first, each taken from the front of the tuple turned by one: an
        # index that varies would put the tuple in scratch memory on a GPU.
        hits = th.hits
        t_prev = t_min
        bent_here = false
        new_dir = dir
        for _ in UInt32(1):min(th.count, UInt32(4))
            rec = hits[1]
            hits = (Base.tail(hits)..., rec)
            # The medium up to the surface
            if has_medium(current_medium)
                seg_T, seg_r_u, seg_r_l = compute_transmittance_ratio_tracking(
                    rgb2spec_table, media, current_medium,
                    Point3f(ray_o + dir * t_prev), dir, rec.t - t_prev, lambda
                )
                T_ray = T_ray * seg_T
                r_u = r_u * seg_r_u
                r_l = r_l * seg_r_l
            end
            t_prev = rec.t
            crossed += Int32(1)

            primitive = Raycore.hit_primitive(accel, rec)
            barycentric = Raycore.hit_barycentric(accel, rec)
            mi = @inbounds media_interfaces[resolve_mi_idx(rec.custom, primitive)]
            n = vp_compute_geometric_normal(primitive)

            if Raycore.is_valid(mi.material) && !(mi.shadows > 0f0)
                # A cutout: what its alpha lets through, the medium unchanged.
                tfc = alpha_filter_context(primitive, barycentric, ray_o, dir, rec.t)
                T_ray = T_ray * _uplift(rgb2spec_table, surface_transparency_dispatch(materials, mi.material, tfc), lambda)
                is_black(T_ray) && return blocked
                continue
            end

            # A surface that passes shadow rays (see `MediumInterface`'s `shadows`): what a
            # dielectric of that index transmits. With `refract`, the ray bends there as
            # light does (see `snells_window`), and the radiance it brings back changes
            # by the square of the indices; without, the refraction is ignored.
            if Raycore.is_valid(mi.material)
                if refract
                    # Through Snell's window light crosses one surface; what crosses more is
                    # the paths' (see `snells_window`).
                    bent && return blocked
                    ns = vp_compute_shading_normal(primitive, barycentric, n)
                    ns = dot(ns, n) < 0f0 ? -ns : ns
                    leaving = dot(dir, ns) > 0f0            # from the inside out
                    η = leaving ? mi.shadows : 1f0 / mi.shadows
                    nf = leaving ? -ns : ns                  # facing the way the ray comes from
                    cos_i = -dot(dir, nf)
                    sin2_t = η * η * max(0f0, 1f0 - cos_i * cos_i)
                    sin2_t >= 1f0 && return blocked
                    new_dir = normalize(η * dir + (η * cos_i - sqrt(1f0 - sin2_t)) * nf)
                    T_ray = T_ray * ((1f0 - fresnel_dielectric(dot(-dir, ns), mi.shadows)) * η * η)
                    bent_here = true
                else
                    T_ray = T_ray * (1f0 - fresnel_dielectric(dot(dir, n), mi.shadows))
                end
            end
            is_black(T_ray) && return blocked

            # Update medium based on crossing direction, where the surface
            # separates two media at all
            if is_medium_transition(mi)
                current_medium = get_crossing_medium(mi, dot(dir, n) < 0f0)
            end
            # Bent, the hits after this one lie on the old line: trace anew from here.
            if bent_here
                hit_p = Point3f(ray_o + dir * rec.t)
                ray_o = offset_surface_origin(hit_p, n, new_dir)
                t_end = t_end - rec.t - 1f-4
                t_min = 0f0
                dir = new_dir
                bent = true
                break
            end
        end
        bent_here && continue
        # More hits than were kept: trace on past the last of them.
        if th.count >= UInt32(4)
            t_min = nextfloat(th.hits[4].t)
            continue
        end
        # No more surfaces - compute transmittance for remaining distance
        if has_medium(current_medium)
            seg_T, seg_r_u, seg_r_l = compute_transmittance_ratio_tracking(
                rgb2spec_table, media, current_medium,
                Point3f(ray_o + dir * t_prev), dir, t_end - t_prev, lambda
            )
            T_ray = T_ray * seg_T
            r_u = r_u * seg_r_u
            r_l = r_l * seg_r_l
        end
        return (T_ray, r_u, r_l, true, dir)
    end
end

"""
    compute_transmittance_ratio_tracking(table, media, medium_idx, origin, dir, t_max, lambda)

Compute transmittance through heterogeneous medium using ratio tracking.
Following pbrt-v4's TraceTransmittance / SampleT_maj implementation.

Uses DDA-based per-voxel majorant bounds (via RayMajorantIterator) for tight
bounds in sparse heterogeneous media, matching the primary path's delta tracking.

Returns (T_ray, r_u, r_l) where:
- T_ray: spectral transmittance estimate
- r_u, r_l: MIS weight accumulators for combining with path weights
"""
# Scene with no participating media. `media` is then a statically EMPTY
# `StaticMultiTypeSet`, and `Raycore.with_index` on one expands to a literal
# `error("with_index: empty StaticMultiTypeSet")` — a throw that can never run
# on a GPU, but which the compiler cannot prove dead, so the whole exception
# path gets emitted into every shader that can reach here. Measured in the
# bump-gold closest-hit shader: this single call was the last 6 error sites
# left after the conductor fixes.
#
# Answering it by dispatch instead is not a workaround, it is the exact result:
# with no media there is nothing to attenuate, so transmittance is 1 and both
# MIS accumulators are 1 — precisely the values `ratio_tracking_dda` initialises
# and would return after iterating zero segments.
@propagate_inbounds compute_transmittance_ratio_tracking(
    rgb2spec_table, media::Raycore.StaticMultiTypeSet{Tuple{}}, medium_idx::SetKey,
    origin::Point3f, dir::Vec3f, t_max::Float32, lambda::Wavelengths
) = (SpectralRadiance(1f0), SpectralRadiance(1f0), SpectralRadiance(1f0))

@propagate_inbounds function compute_transmittance_ratio_tracking(
    rgb2spec_table, media, medium_idx::SetKey,
    origin::Point3f, dir::Vec3f, t_max::Float32, lambda::Wavelengths
)
    template_grid = get_template_grid_from_tuple(media)
    ray = Raycore.Ray(o=origin, d=dir)
    return Raycore.with_index(
        transmittance_dda_helper,
        media, medium_idx,
        rgb2spec_table, ray, t_max, lambda, origin, dir, media, medium_idx, template_grid
    )
end

"""Helper dispatched via with_index to get concrete medium type for DDA iterator."""
@propagate_inbounds function transmittance_dda_helper(
    medium, rgb2spec_table, ray, t_max, lambda, origin, dir, media, medium_idx, template_grid
)
    iter = create_majorant_iterator(medium, rgb2spec_table, ray, t_max, lambda, template_grid)
    return ratio_tracking_dda(iter, origin, dir, media, medium_idx, rgb2spec_table, lambda,
                              control(medium, media, rgb2spec_table, lambda), controlmajorant(medium))
end

"""
Ratio tracking using DDA majorant iterator segments.
Iterates over per-voxel majorant bounds, doing ratio tracking within each segment.
"""
@propagate_inbounds function ratio_tracking_dda(
    iter::RayMajorantIterator,
    origin::Point3f, dir::Vec3f,
    media, medium_idx::SetKey, rgb2spec_table, lambda::Wavelengths,
    mc::MediumProperties, c0::Float32
)
    iter.mode == Int32(0) && return (SpectralRadiance(1f0), SpectralRadiance(1f0), SpectralRadiance(1f0))
    # The constant control spans the complete clipped ray. Evaluate its exact
    # attenuation once: multiplying a polynomial approximation per cell made
    # transmission depend on grid resolution (6% error over 1024 thin cells).
    T_ray = exp(-(iter.t_max - iter.t_min) * σ_t(mc))
    r_u = SpectralRadiance(1f0)
    r_l = SpectralRadiance(1f0)

    rng = pcg32_init(pbrt_hash(origin), pbrt_hash(dir))

    current_iter = iter
    for _ in Int32(1):majorant_segment_limit(iter)
        seg, new_iter, valid = ray_majorant_next(current_iter, media)
        if !valid
            break
        end
        current_iter = new_iter

        # Only the residual is tracked; the control was integrated above.
        σ_maj = SpectralRadiance(max(seg.σ_maj[1] - c0, 0f0), max(seg.σ_maj[2] - c0, 0f0),
                                 max(seg.σ_maj[3] - c0, 0f0), max(seg.σ_maj[4] - c0, 0f0))
        σ_maj_0 = σ_maj[1]

        # No residual in this voxel — nothing to track
        if σ_maj_0 < 1f-6
            continue
        end

        t = seg.t_min
        t_max_seg = seg.t_max

        # Ratio tracking within this DDA segment
        while t < t_max_seg
            u, rng = pcg32_uniform_f32(rng)
            dt = -log(max(1f-10, 1f0 - u)) / σ_maj_0
            t_sample = t + dt

            if t_sample >= t_max_seg
                # Past segment end — apply remaining transmittance
                dt_remain = t_max_seg - t
                T_maj = fast_exp(-dt_remain * σ_maj)
                T_maj_0 = T_maj[1]
                if T_maj_0 > 1f-10
                    T_ray = T_ray * T_maj / T_maj_0
                    r_l = r_l * T_maj / T_maj_0
                    r_u = r_u * T_maj / T_maj_0
                end
                break
            end

            # Sample the residual at the interaction point
            p = Point3f(origin + dir * t_sample)
            mp = Raycore.with_index(residual, media, medium_idx, media, rgb2spec_table, p, lambda)

            # Null-scattering coefficient (clamped non-negative)
            σ_n = σ_maj - mp.σ_a - mp.σ_s
            σ_n = SpectralRadiance(max(σ_n[1], 0f0), max(σ_n[2], 0f0), max(σ_n[3], 0f0), max(σ_n[4], 0f0))

            T_maj = fast_exp(-dt * σ_maj)

            # Ratio tracking update (null-scattering only for transmittance)
            pr = T_maj[1] * σ_maj_0
            if pr > 1f-10
                T_ray = T_ray * T_maj * σ_n / pr
                r_l = r_l * T_maj * σ_maj / pr
                r_u = r_u * T_maj * σ_n / pr
            else
                T_ray = SpectralRadiance(0f0)
                return (T_ray, r_u, r_l)
            end

            # Russian roulette termination (following pbrt-v4)
            Tr_estimate = T_ray / max(1f-10, average(r_l + r_u))
            if max_component(Tr_estimate) < 0.05f0
                q = 0.75f0
                rr_sample, rng = pcg32_uniform_f32(rng)
                if rr_sample < q
                    T_ray = SpectralRadiance(0f0)
                    return (T_ray, r_u, r_l)
                else
                    T_ray = T_ray / (1f0 - q)
                end
            end

            if is_black(T_ray)
                return (T_ray, r_u, r_l)
            end

            t = t_sample
        end

        if is_black(T_ray)
            break
        end
    end

    return (T_ray, r_u, r_l)
end

"""
    compute_transmittance_simple(table, media, medium_idx, origin, dir, t_max, lambda)

Simple wrapper that returns only the transmittance (for compatibility).
"""
@propagate_inbounds function compute_transmittance_simple(
    rgb2spec_table, media, medium_idx::SetKey,
    origin::Point3f, dir::Vec3f, t_max::Float32, lambda::Wavelengths
)
    T_ray, _, _ = compute_transmittance_ratio_tracking(rgb2spec_table, media, medium_idx, origin, dir, t_max, lambda)
    return T_ray
end


"""
    vp_trace_shadow_rays_kernel!(...)

Trace shadow rays and accumulate unoccluded contributions.
For rays through media, computes transmittance along the ray.
Handles transmissive boundaries (MediumInterface) by tracing through them.
"""
@propagate_inbounds function vp_trace_shadow_rays_kernel!(
        work,
        pixel_L,
        rgb2spec_table,
        accel,
        media_interfaces,
        media,
        materials,
        lights,
    )
    # Trace shadow ray, handling transmissive boundaries and alpha pass-through
    # Returns (T_ray, tr_r_u, tr_r_l, visible) following pbrt-v4's TraceTransmittance
    T_ray, tr_r_u, tr_r_l, visible, out = trace_shadow_transmittance(
        accel, media_interfaces, media, materials, rgb2spec_table,
        work.ray.o, work.ray.d, work.t_max, work.lambda, work.medium_idx, work.refract
    )

    if visible && !is_black(T_ray)
        # Following pbrt-v4 TraceTransmittance (intersect.h line 266):
        # Ld *= T_ray / (sr.r_u * r_u + sr.r_l * r_l).Average()
        # This combines path MIS weights (work.r_u, work.r_l) with transmittance
        # MIS weights (tr_r_u, tr_r_l)
        mis_weight = work.r_u * tr_r_u + work.r_l * tr_r_l
        mis_denom = average(mis_weight)

        if mis_denom > 1f-10
            # Through Snell's window the sky is where the ray left (see `snells_window`).
            sky = work.refract ? evaluate_escaped_ray_spectral(rgb2spec_table, lights, out, work.lambda) : SpectralRadiance(1f0)
            final_L = work.Ld * sky * T_ray / mis_denom
            if !is_black(final_L)
                # Add to pixel
                pixel_idx = work.pixel_index
                base_idx = (pixel_idx - Int32(1)) * Int32(4)

                accumulate_spectrum!(pixel_L, base_idx, final_L)
            end
        end
    end
end


# ============================================================================
# Escaped Ray Handling (Environment Light)
# ============================================================================

@propagate_inbounds function vp_handle_escaped_rays_kernel!(
    work,
    pixel_L,
    rgb2spec_table,
    lights,
    bvh_nodes,
    light_to_bit_trail,
    infinite_light_indices,
    num_infinite_lights::Int32,
    num_bvh_lights::Int32
)
    # Evaluate environment lights
    Le = evaluate_escaped_ray_spectral(rgb2spec_table, lights, work.ray_d, work.lambda)

    # Apply path throughput
    contribution = work.beta * Le

    if !is_black(contribution)
        # MIS weighting following pbrt-v4 (integrator.cpp HandleEscapedRays)
        # depth=0 or specular bounce: L = beta * Le / r_u.Average()
        # Otherwise: L = beta * Le / (r_u + r_l).Average()
        #   where r_l = sum over infinite lights of: work.r_l * bvh_pmf(light) * light.PDF_Li(wi)
        final_contrib = if windowed(work.prev_intr_n)
            # Out through Snell's window: `r_l` already holds the light sampling's
            # density for this path (see `WINDOWED`).
            contribution / average(work.r_u + work.r_l)
        elseif work.depth == Int32(0) || work.specular_bounce
            contribution / average(work.r_u)
        else
            # Full MIS: for each infinite light, accumulate r_l weighted by BVH PMF
            # Following pbrt-v4 HandleEscapedRay: iterate infinite lights, use bvh_pmf per light
            r_l = SpectralRadiance(0f0)
            for i in Int32(1):num_infinite_lights
                light_flat_idx = infinite_light_indices[i]
                light_choice_pdf = bvh_pmf(
                    bvh_nodes, light_to_bit_trail,
                    num_infinite_lights, num_bvh_lights,
                    work.prev_intr_p, work.prev_intr_n, light_flat_idx
                )
                if light_choice_pdf > 0f0
                    # Get PDF_Li for this specific light
                    # env_light_pdf_single(light, lights, wi) is with_index-compatible (element first)
                    light_idx = flat_to_light_index(lights, light_flat_idx)
                    light_pdf_li = with_index(env_light_pdf_single, lights, light_idx,
                        lights, work.ray_d
                    )
                    r_l = r_l + work.r_l * light_choice_pdf * light_pdf_li
                end
            end

            # Combine r_u and r_l
            r_sum = work.r_u + r_l
            mis_denom = average(r_sum)

            if mis_denom > 1f-10
                contribution / mis_denom
            else
                contribution / average(work.r_u)
            end
        end

        # Add to pixel
        pixel_idx = work.pixel_index
        base_idx = (pixel_idx - Int32(1)) * Int32(4)

        accumulate_spectrum!(pixel_L, base_idx, final_contrib)
    end
end

# ============================================================================
# Camera Medium Detection
# ============================================================================

"""
    detect_camera_medium_kernel!(result, accel, media_interfaces, camera_pos)

Single-workitem kernel that traces a ray from the camera position to determine
which medium the camera is inside. Writes a SetKey to result[1].
"""
@kernel inbounds=true function detect_camera_medium_kernel!(
    result,
    accel,
    media_interfaces,
    @Const(camera_pos::Point3f)
)
    # Arbitrary direction, normalized (1,1,1)/√3 — avoids axis-alignment
    d = Vec3f(0.57735027f0, 0.57735027f0, 0.57735027f0)
    ray = Raycore.Ray(o=camera_pos, d=d)
    found = false

    for _ in 1:Int32(16)
        if !found
            hit, primitive, t_hit, barycentric, instance_id = Raycore.closest_hit(accel, ray)

            if !hit
                result[1] = SetKey()
                found = true
            else
                mi_idx = resolve_mi_idx(instance_id, primitive)
                mi = media_interfaces[mi_idx]

                if is_medium_transition(mi)
                    # Medium transition boundary — determine which medium camera is in
                    n = vp_compute_geometric_normal(primitive)
                    # get_medium_index with -d gives the medium the ray was traveling through
                    medium = get_medium_index(mi, -d, n)
                    result[1] = medium
                    found = true
                else
                    # Not a medium boundary — skip past and continue
                    pi = Point3f(ray.o + d * t_hit)
                    n = vp_compute_geometric_normal(primitive)
                    offset = if dot(d, n) > 0f0; n else; -n end
                    ray = Raycore.Ray(o=Point3f(pi + offset * 1f-4), d=d)
                end
            end
        end
    end

    if !found
        # Max iterations exceeded — assume vacuum
        result[1] = SetKey()
    end
end

"""
    detect_camera_medium(mem, accel, media_interfaces, camera_pos) -> SetKey

Detect which medium the camera is inside by tracing a single ray from the camera
position. Returns a SetKey identifying the medium, or SetKey() for vacuum. The probe's
scratch comes from `mem` and goes back to it (`free!`) before this returns.
"""
function detect_camera_medium(mem::DeviceMemory, accel, media_interfaces, camera_pos::Point3f)
    # No interfaces, no medium to be inside — and the answer is not a guess,
    # it is the same `SetKey()` the kernel writes when its probe ray escapes.
    #
    # Worth short-circuiting for two reasons. It is a traced probe per distinct
    # camera position in every scene that has no media, which is most of them.
    # And on an acceleration structure nothing has been pushed into yet it does
    # not COMPILE: an untyped one carries `Triangle{UInt32}`, and
    # `resolve_mi_idx` reads `primitive.metadata.medium_interface_idx`, which a
    # `UInt32` metadata does not have. That is reachable — an empty plot, a
    # scene whose geometry has been handed to the rasteriser — and it surfaced
    # as a shader compile error naming a kernel the scene has no use for.
    isempty(media_interfaces) && return SetKey()
    # From the pool and given back: an array of its own was a dedicated device
    # allocation that nothing released, one for every camera position probed.
    result = alloc!(mem, SetKey, 1)
    kernel! = detect_camera_medium_kernel!(mem.backend)
    kernel!(result, accel, media_interfaces, camera_pos; ndrange=1)
    key = @allowscalar result[1]
    free!(mem)
    return key
end
