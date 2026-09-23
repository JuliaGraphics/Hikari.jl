# Ray tracing and intersection handling for VolPath
# Handles ray-scene intersection and classifies results into work queues

"""
    resolve_mi_idx(accel, inst_slot, primitive) -> UInt32

Pick the medium_interface_idx for a hit.  `inst_slot` is the 5th value
returned by `Raycore.closest_hit` / `Raycore.any_hit`; its meaning
depends on the accelerator type:

- `StaticTLAS` (software traversal): 1-based instance array index.  The
  override is `accel.instances[inst_slot].instance_id`.
- Hardware-adapted accelerators (`AdaptedAccel`): the override value
  itself, forwarded straight from `gl_InstanceCustomIndexEXT` via the
  inline ray query's `lava_ray_query_get_instance_custom_index`.

Either way, a nonzero override replaces the triangle's per-face
`medium_interface_idx`; zero means "inherit".  This is the single place
where the per-instance interface override resolves.
"""
@inline function resolve_mi_idx(accel::Raycore.StaticTLAS, inst_idx::UInt32, primitive)
    if inst_idx != UInt32(0)
        @inbounds override = accel.instances[inst_idx].instance_id
        if override != UInt32(0)
            return override
        end
    end
    return primitive.metadata.medium_interface_idx
end

# Hardware-adapted accelerators: the 5th closest_hit return is already the
# resolved override (carried through `gl_InstanceCustomIndexEXT`), so no
# `.instances` lookup is needed.
@inline function resolve_mi_idx(::Any, override::UInt32, primitive)
    override != UInt32(0) ? override : primitive.metadata.medium_interface_idx
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

    return (pi=pi, n=n, dpdu=dpdu, dpdv=dpdv, ns=ns, dpdus=dpdus, dpdvs=dpdvs, uv=uv)
end

# ============================================================================
# Trace + Enqueue Kernel
# ============================================================================
#
# Software-BVH counterpart of the hardware closest-hit shaders. Each thread:
#   1. If the ray is currently inside a medium, push it to
#      `medium_sample_queue` and let the medium pipeline handle it. Delta
#      tracking enqueues whatever survives to a surface, through the same
#      `enqueue_after_intersection!` used below.
#   2. Otherwise, run the alpha-test ray-query loop, then resolve MixMaterial
#      and the null-material medium swap HERE — at the push site, so the
#      per-material queue routes to a concrete type and the shading kernels
#      never see a material that has no BSDF.
#   3. Enqueue the hit: its payload once into `hit_surface_queue`, a 4-byte
#      `TypedHitRef{T}` into the matching per-material queue, and a
#      `VPHitAreaLightWorkItem` if the triangle emits.
#
# Shading itself is `vp_shade_material_kernel!` (surface-eval.jl), the same
# implementation the closest-hit shaders call.

@propagate_inbounds function vp_trace_and_shade_kernel!(
    work,
    next_ray_queue, escaped_queue, medium_sample_queue,
    per_material_queue,                   # MultiTypeMaterialQueue — routed by material type (4-byte index items)
    hit_surface_queue,                    # shared hit store the typed queues index into
    hit_area_light_queue,                 # parallel push for emission-MIS (pbrt-v4 hitAreaLightQueue)
    pixel_L,
    accel, media_interfaces, media, materials, lights,
    rgb2spec_table,
    bvh_nodes, infinite_light_indices, light_to_bit_trail,
    num_infinite_lights::Int32, num_bvh_lights::Int32, num_lights::Int32,
    max_depth::Int32, do_regularize::Bool,
    sobol_rng, sample_idx_ref,
    camera_ref,
    samples_per_pixel::Int32,
    rr_depth::Int32,
)
    # The sample index is a `Mantle.GPURef`, so it arrives as a one-element
    # device array and the recorded commands hold its ADDRESS. That is what lets
    # one recording render every sample: the plan is written once and the host
    # writes this number with one `cmd_update_buffer` per run. It was an `Int32`
    # packed into the arguments of every dispatch that reads it, which is why a
    # recorded plan needed the whole per-run repack.
    sample_idx = @inbounds sample_idx_ref[Int32(1)]
    # The camera is a `GPURef` for the same reason: a moved camera is an update
    # in the run's own submission, not a new plan.
    camera = @inbounds camera_ref[Int32(1)]
    # ─── Medium ray: trace and defer all shading to the medium pipeline ───
    if has_medium(work.medium_idx)
        hit, primitive, t_hit, barycentric, inst_idx = Raycore.closest_hit(accel, work.ray)

        if hit
            mi_idx = resolve_mi_idx(accel, inst_idx, primitive)
            mi = media_interfaces[mi_idx]
            mat_idx = mi.material

            geom = vp_compute_surface_geometry(primitive, barycentric, work.ray.o, work.ray.d, t_hit)

            tfc_bump = bump_filter_context(camera, work, geom, samples_per_pixel)
            dndu, dndv = vp_compute_normal_derivatives(primitive)
            ns_b, dpdus_b = get_perturbed_shading_frame(materials, mat_idx,
                                                       geom.ns, geom.dpdus,
                                                       geom.dpdu, geom.dpdv,
                                                       dndu, dndv, geom.n, tfc_bump)

            dpdvs_b = cross(ns_b, dpdus_b)

            push!(medium_sample_queue, VPMediumSampleWorkItem(
                work, t_hit,
                geom.pi, geom.n, geom.dpdu, geom.dpdv,
                ns_b, dpdus_b, dpdvs_b,
                geom.uv, mat_idx, mi,
                primitive.metadata.primitive_index, SVector{3,Float32}(barycentric),
                primitive.metadata.arealight_flat_idx, Raycore.area(primitive)
            ))
        else
            push!(medium_sample_queue, VPMediumSampleWorkItem(work))
        end
        return
    end

    # ─── Non-medium: trace + alpha-test + inline shade ───
    ray = work.ray
    for _ in 1:Int32(16)
        hit, primitive, t_hit, barycentric, inst_idx = Raycore.closest_hit(accel, ray)

        if !hit
            push!(escaped_queue, VPEscapedRayWorkItem(work))
            return
        end

        mi_idx = resolve_mi_idx(accel, inst_idx, primitive)
        mi = media_interfaces[mi_idx]
        mat_idx = mi.material

        uv = vp_compute_uv_barycentric(primitive, barycentric)
        alpha = get_surface_alpha_dispatch(materials, mat_idx, uv)

        if alpha < 1f0
            rng = pcg32_init(pbrt_hash(ray.o), pbrt_hash(ray.d))
            alpha_u, _ = pcg32_uniform_f32(rng)
            if alpha_u > alpha
                pi_pt = Point3f(ray.o + ray.d * t_hit)
                n = vp_compute_geometric_normal(primitive)
                offset = if dot(ray.d, n) > 0f0; n else; -n end
                ray = Raycore.Ray(o=Point3f(pi_pt + offset * 1f-4), d=ray.d)
                continue
            end
        end

        # Valid surface hit — compute geometry + bump perturbation, resolve
        # MixMaterial (must happen at the push site, not in the shading
        # kernel, so the per-material queue routes to the right concrete
        # type), and either push to the typed shading queue or handle the
        # null-material boundary inline.
        geom = vp_compute_surface_geometry(primitive, barycentric, ray.o, ray.d, t_hit)
        wo = -ray.d
        resolved_mat_idx = resolve_mix_material(materials, mat_idx, geom.pi, wo, geom.uv)

        # Null-material boundary (pbrt `Material "interface"` / nullptr): no
        # BSDF, no direct lighting, no emission.  Push a continuation ray
        # with the medium swap inline — depth NOT incremented (same as
        # `evaluate_material_inner!`'s null-material skip).
        if !Raycore.is_valid(resolved_mat_idx) && is_medium_transition(mi)
            ray_d = -wo
            new_medium = get_medium_index(mi, ray_d, geom.n)
            offset_dir = if dot(ray_d, geom.n) > 0f0; geom.n; else; -geom.n; end
            ray_origin = Point3f(geom.pi + offset_dir * 1f-4)
            new_ray = Raycore.Ray(o=ray_origin, d=ray_d, t_max=Inf32, time=0f0)
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

        # Camera-approximated differentials for every hit, matching pbrt-v4
        # ComputeDifferentials (see the trace-kernel comment above).
        tfc_bump = bump_filter_context(camera, work, geom, samples_per_pixel)
        dndu, dndv = vp_compute_normal_derivatives(primitive)
        ns_b, dpdus_b = get_perturbed_shading_frame(materials, resolved_mat_idx,
                                                   geom.ns, geom.dpdus,
                                                   geom.dpdu, geom.dpdv,
                                                   dndu, dndv, geom.n, tfc_bump)

        dpdvs_b = cross(ns_b, dpdus_b)

        hit_work = VPHitSurfaceWorkItem(
            work,
            geom.pi, geom.n, geom.dpdu, geom.dpdv,
            ns_b, dpdus_b, dpdvs_b,
            geom.uv, resolved_mat_idx, mi,
            primitive.metadata.primitive_index, SVector{3,Float32}(barycentric),
        )

        # Route into the matching per-material queue + emission queue.  The
        # push site uses `with_index(materials, ...)` ONCE per hit to pick
        # the concrete type; the per-type shading kernels never touch
        # `with_index` on the material axis (vp_shade_material_kernel uses
        # `material_of_type` — a compile-time slot lookup). Emission MIS
        # runs in its own kernel via `hit_area_light_queue`.
        enqueue_after_intersection!(per_material_queue, hit_area_light_queue, materials,
            hit_surface_queue, hit_work,
            primitive.metadata.arealight_flat_idx, Raycore.area(primitive), t_hit)
        return
    end
    # 16 alpha-bounces exhausted (extremely unlikely): ray absorbed
    return
end


# ============================================================================
# Shadow Ray Tracing Kernel (with medium transmittance)
# ============================================================================

"""
    trace_shadow_transmittance(accel, media_interfaces, media, rgb2spec_table, origin, dir, t_max, lambda, medium_idx)

Trace a shadow ray computing transmittance through media and transmissive boundaries.
Returns (T_ray, r_u, r_l, visible) where:
- T_ray: spectral transmittance
- r_u, r_l: MIS weight accumulators for combining with path weights
- visible: false if ray hits an opaque surface

Following pbrt-v4's TraceTransmittance: transmissive surfaces (MediumInterface) let the ray through,
while opaque surfaces block it. The final contribution is computed as:
    Ld * T_ray / average(path_r_u * r_u + path_r_l * r_l)
"""
# @noinline: this is a self-contained ray-query lifecycle (init+proceed+get
# all internal via Raycore.closest_hit). Keeping it as a separate OpFunction
# in SPIR-V gives the call site its own register frame so the surrounding
# shading kernel (vp_shade_material_kernel!) doesn't drag the shadow trace's
# state through its own register budget — this is what kept Conductor
# specializations pinned at 255 regs / ~1 KB spill on RTX 4000 Ada.
# Lava's refined `incomplete_rayquery` rule (compilation.jl, 2026-06-04)
# recognises this function's complete lifecycle and lets the @noinline stand.
@noinline function trace_shadow_transmittance(
    accel, media_interfaces, media, materials, rgb2spec_table,
    origin::Point3f, dir::Vec3f, t_max::Float32, lambda::Wavelengths, medium_idx::SetKey
)
    # Transmittance and MIS weights following pbrt-v4
    T_ray = SpectralRadiance(1f0)
    r_u = SpectralRadiance(1f0)
    r_l = SpectralRadiance(1f0)

    current_medium = medium_idx
    ray_o = origin
    t_remaining = t_max

    # Trace through up to N transmissive/alpha boundaries
    max_bounces = Int32(10)
    for _ in 1:max_bounces
        if t_remaining < 1f-6
            break
        end

        ray = Raycore.Ray(o=ray_o, d=dir, t_max=t_remaining)
        hit, primitive, t_hit, barycentric, inst_idx = Raycore.closest_hit(accel, ray)

        if !hit
            # No more surfaces - compute transmittance for remaining distance
            if has_medium(current_medium)
                seg_T, seg_r_u, seg_r_l = compute_transmittance_ratio_tracking(
                    rgb2spec_table, media, current_medium,
                    ray_o, dir, t_remaining, lambda
                )
                T_ray = T_ray * seg_T
                r_u = r_u * seg_r_u
                r_l = r_l * seg_r_l
            end
            return (T_ray, r_u, r_l, true)  # Visible
        end

        # Hit a surface - look up MediumInterfaceIdx (per-instance override takes priority)
        mi_idx = resolve_mi_idx(accel, inst_idx, primitive)
        mi = @inbounds media_interfaces[mi_idx]
        n = vp_compute_geometric_normal(primitive)
        entering = dot(dir, n) < 0f0

        # A surface only acts as a transparent boundary for shadow rays if it's a
        # pure medium transition with no BSDF material. Surfaces with a material
        # (e.g. dielectric glass) block shadow rays — pbrt-v4: result.hit && result.material → T_ray=0.
        # Refracted-light contributions are captured by explicit path bouncing through the BSDF.
        is_transmissive = is_medium_transition(mi) && !Raycore.is_valid(mi.material)

        if !is_transmissive
            # Check alpha for stochastic pass-through (e.g. GLTF BLEND mode foliage)
            # Following pbrt-v4: use deterministic hash of ray origin+direction
            uv = vp_compute_uv_barycentric(primitive, barycentric)
            mat_idx = mi.material
            alpha = get_surface_alpha_dispatch(materials, mat_idx, uv)

            if alpha < 1f0
                # Deterministic stochastic test (same ray always gets same decision)
                rng = pcg32_init(pbrt_hash(ray_o), pbrt_hash(dir))
                alpha_u, _ = pcg32_uniform_f32(rng)
                if alpha_u > alpha
                    # Alpha pass-through: compute medium transmittance up to surface, then continue
                    if has_medium(current_medium)
                        seg_T, seg_r_u, seg_r_l = compute_transmittance_ratio_tracking(
                            rgb2spec_table, media, current_medium,
                            ray_o, dir, t_hit, lambda
                        )
                        T_ray = T_ray * seg_T
                        r_u = r_u * seg_r_u
                        r_l = r_l * seg_r_l
                    end
                    # Move past this surface (medium unchanged — not a medium transition)
                    ray_o = Point3f(ray_o + dir * (t_hit + 1f-4))
                    t_remaining = t_remaining - t_hit - 1f-4
                    continue
                end
            end

            # Opaque surface blocks the ray
            return (SpectralRadiance(0f0), SpectralRadiance(1f0), SpectralRadiance(1f0), false)
        end

        # Transmissive boundary - compute transmittance up to this point
        if has_medium(current_medium)
            seg_T, seg_r_u, seg_r_l = compute_transmittance_ratio_tracking(
                rgb2spec_table, media, current_medium,
                ray_o, dir, t_hit, lambda
            )
            T_ray = T_ray * seg_T
            r_u = r_u * seg_r_u
            r_l = r_l * seg_r_l
        end

        if is_black(T_ray)
            return (T_ray, r_u, r_l, true)  # Transmittance is zero, no point continuing
        end

        # Update medium based on crossing direction
        current_medium = get_crossing_medium(mi, entering)

        # Move past this surface
        ray_o = Point3f(ray_o + dir * (t_hit + 1f-4))  # Small offset to avoid self-intersection
        t_remaining = t_remaining - t_hit - 1f-4
    end

    # Exceeded max bounces - treat as occluded
    return (SpectralRadiance(0f0), SpectralRadiance(1f0), SpectralRadiance(1f0), false)
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
    return ratio_tracking_dda(iter, origin, dir, media, medium_idx, rgb2spec_table, lambda)
end

"""
Ratio tracking using DDA majorant iterator segments.
Iterates over per-voxel majorant bounds, doing ratio tracking within each segment.
"""
@propagate_inbounds function ratio_tracking_dda(
    iter::RayMajorantIterator,
    origin::Point3f, dir::Vec3f,
    media, medium_idx::SetKey, rgb2spec_table, lambda::Wavelengths
)
    T_ray = SpectralRadiance(1f0)
    r_u = SpectralRadiance(1f0)
    r_l = SpectralRadiance(1f0)

    rng = pcg32_init(pbrt_hash(origin), pbrt_hash(dir))

    current_iter = iter
    for _ in Int32(1):Int32(256)  # max DDA segments
        seg, new_iter, valid = ray_majorant_next(current_iter, media)
        if !valid
            break
        end
        current_iter = new_iter

        σ_maj = seg.σ_maj
        σ_maj_0 = σ_maj[1]

        # Empty voxel — transmittance is 1, skip
        if σ_maj_0 < 1f-10
            continue
        end

        t = seg.t_min
        t_max_seg = seg.t_max

        # Ratio tracking within this DDA segment
        for _ in Int32(1):Int32(100)
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

            # Sample medium properties at interaction point
            p = Point3f(origin + dir * t_sample)
            mp = Raycore.with_index(sample_point, media, medium_idx, media, rgb2spec_table, p, lambda)

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
    )
    # Trace shadow ray, handling transmissive boundaries and alpha pass-through
    # Returns (T_ray, tr_r_u, tr_r_l, visible) following pbrt-v4's TraceTransmittance
    T_ray, tr_r_u, tr_r_l, visible = trace_shadow_transmittance(
        accel, media_interfaces, media, materials, rgb2spec_table,
        work.ray.o, work.ray.d, work.t_max, work.lambda, work.medium_idx
    )

    if visible && !is_black(T_ray)
        # Following pbrt-v4 TraceTransmittance (intersect.h line 266):
        # Ld *= T_ray / (sr.r_u * r_u + sr.r_l * r_l).Average()
        # This combines path MIS weights (work.r_u, work.r_l) with transmittance
        # MIS weights (tr_r_u, tr_r_l)
        mis_weight = work.r_u * tr_r_u + work.r_l * tr_r_l
        mis_denom = average(mis_weight)

        if mis_denom > 1f-10
            final_L = work.Ld * T_ray / mis_denom
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
        final_contrib = if work.depth == Int32(0) || work.specular_bounce
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
            hit, primitive, t_hit, barycentric, inst_idx = Raycore.closest_hit(accel, ray)

            if !hit
                result[1] = SetKey()
                found = true
            else
                mi_idx = resolve_mi_idx(accel, inst_idx, primitive)
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
    detect_camera_medium(backend, accel, media_interfaces, camera_pos) -> SetKey

Detect which medium the camera is inside by tracing a single ray from the camera
position. Returns a SetKey identifying the medium, or SetKey() for vacuum.
"""
function detect_camera_medium(backend, accel, media_interfaces, camera_pos::Point3f)
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
    result = Mantle.devicearray(backend, SetKey, (1,))
    kernel! = detect_camera_medium_kernel!(backend)
    kernel!(result, accel, media_interfaces, camera_pos; ndrange=1)
    return @allowscalar result[1]
end
