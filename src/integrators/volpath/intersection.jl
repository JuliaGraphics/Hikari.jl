# Ray tracing and intersection handling for VolPath
# Handles ray-scene intersection and classifies results into work queues

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

    # Ensure shading normal is on same side as geometric normal
    if dot(ns, geometric_normal) < 0f0
        return -ns
    else
        return ns
    end
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

    # Shading tangent vectors
    dpdus, dpdvs = vp_compute_shading_tangents(primitive, barycentric, ns, dpdu, dpdv)

    return (pi=pi, n=n, dpdu=dpdu, dpdv=dpdv, ns=ns, dpdus=dpdus, dpdvs=dpdvs, uv=uv)
end

# ============================================================================
# Primary Ray Intersection Kernel
# ============================================================================

@propagate_inbounds function vp_trace_rays_kernel!(
    work,
    medium_sample_queue,
    escaped_queue,
    hit_surface_queue,
    accel,
    media_interfaces,
    materials
)
    # Check if ray is currently traveling through a medium
    if has_medium(work.medium_idx)
        # Medium case: trace once, push to medium_sample_queue (alpha not handled here yet)
        hit, primitive, t_hit, barycentric = Raycore.closest_hit(accel, work.ray)

        if hit
            mi_idx = primitive.metadata.medium_interface_idx
            mi = media_interfaces[mi_idx]
            mat_idx = mi.material

            geom = vp_compute_surface_geometry(primitive, barycentric, work.ray.o, work.ray.d, t_hit)

            push!(medium_sample_queue, VPMediumSampleWorkItem(
                work, t_hit,
                geom.pi, geom.n, geom.dpdu, geom.dpdv,
                geom.ns, geom.dpdus, geom.dpdvs,
                geom.uv, mat_idx, mi,
                primitive.metadata.primitive_index, SVector{3,Float32}(barycentric),
                primitive.metadata.arealight_flat_idx, Raycore.area(primitive)
            ))
        else
            push!(medium_sample_queue, VPMediumSampleWorkItem(work))
        end
    else
        # Non-medium case: alpha testing loop at intersection level
        # Following pbrt-v4: alpha-killed surfaces are skipped without consuming depth
        ray = work.ray
        for _ in 1:Int32(16)
            hit, primitive, t_hit, barycentric = Raycore.closest_hit(accel, ray)

            if !hit
                push!(escaped_queue, VPEscapedRayWorkItem(work))
                return
            end

            mi_idx = primitive.metadata.medium_interface_idx
            mi = media_interfaces[mi_idx]
            mat_idx = mi.material

            # Stochastic alpha test (deterministic hash, same as shadow rays)
            uv = vp_compute_uv_barycentric(primitive, barycentric)
            alpha = get_surface_alpha_dispatch(materials, mat_idx, uv)

            if alpha < 1f0
                rng = pcg32_init(pbrt_hash(ray.o), pbrt_hash(ray.d))
                alpha_u, _ = pcg32_uniform_f32(rng)
                if alpha_u > alpha
                    # Alpha pass-through: skip this surface, no depth consumed
                    pi = Point3f(ray.o + ray.d * t_hit)
                    n = vp_compute_geometric_normal(primitive)
                    offset = if dot(ray.d, n) > 0f0; n else; -n end
                    ray = Raycore.Ray(o=Point3f(pi + offset * 1f-4), d=ray.d)
                    continue
                end
            end

            # Valid surface hit - compute geometry and push to queue
            geom = vp_compute_surface_geometry(primitive, barycentric, ray.o, ray.d, t_hit)

            push!(hit_surface_queue, VPHitSurfaceWorkItem(
                work,
                geom.pi, geom.n, geom.dpdu, geom.dpdv,
                geom.ns, geom.dpdus, geom.dpdvs,
                geom.uv, mat_idx, mi,
                primitive.metadata.primitive_index, SVector{3,Float32}(barycentric),
                primitive.metadata.arealight_flat_idx, Raycore.area(primitive),
                t_hit
            ))
            return
        end
        # Max alpha bounces exceeded (extremely unlikely) - ray absorbed
    end
end

function vp_trace_rays!(state::VolPathState, accel, media_interfaces, materials)
    input_queue = current_ray_queue(state)
    foreach(vp_trace_rays_kernel!,
        input_queue,
        state.medium_sample_queue,
        state.escaped_queue,
        state.hit_surface_queue,
        accel,
        media_interfaces,
        materials,
    )
    return nothing
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
@propagate_inbounds function trace_shadow_transmittance(
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
        hit, primitive, t_hit, barycentric = Raycore.closest_hit(accel, ray)

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

        # Hit a surface - look up MediumInterfaceIdx
        mi_idx = primitive.metadata.medium_interface_idx
        mi = media_interfaces[mi_idx]
        n = vp_compute_geometric_normal(primitive)
        entering = dot(dir, n) < 0f0

        # Check if surface is a medium transition (transmissive boundary)
        is_transmissive = is_medium_transition(mi)

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
@propagate_inbounds function compute_transmittance_ratio_tracking(
    rgb2spec_table, media, medium_idx::SetKey,
    origin::Point3f, dir::Vec3f, t_max::Float32, lambda::Wavelengths
)
    template_grid = get_template_grid_from_tuple(media)
    ray = Raycore.Ray(o=origin, d=dir)
    return Raycore.with_index(
        _transmittance_dda_helper,
        media, medium_idx,
        rgb2spec_table, ray, t_max, lambda, origin, dir, media, medium_idx, template_grid
    )
end

"""Helper dispatched via with_index to get concrete medium type for DDA iterator."""
@propagate_inbounds function _transmittance_dda_helper(
    medium, rgb2spec_table, ray, t_max, lambda, origin, dir, media, medium_idx, template_grid
)
    iter = create_majorant_iterator(medium, rgb2spec_table, ray, t_max, lambda, template_grid)
    return _ratio_tracking_dda(iter, origin, dir, media, medium_idx, rgb2spec_table, lambda)
end

"""
Ratio tracking using DDA majorant iterator segments.
Iterates over per-voxel majorant bounds, doing ratio tracking within each segment.
"""
@propagate_inbounds function _ratio_tracking_dda(
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
                T_maj = exp(-dt_remain * σ_maj)
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

            T_maj = exp(-dt * σ_maj)

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

function vp_trace_shadow_rays!(
        state::VolPathState,
        accel,
        media_interfaces,
        media,
        materials
    )
    foreach(vp_trace_shadow_rays_kernel!,
        state.shadow_queue,
        state.pixel_L,
        state.rgb2spec_table,
        accel, media_interfaces, media, materials,
    )
    return nothing
end

# ============================================================================
# Escaped Ray Handling (Environment Light)
# ============================================================================

@propagate_inbounds function vp_handle_escaped_rays_kernel!(
    work,
    pixel_L,
    rgb2spec_table,
    lights
)
    # Evaluate environment lights
    Le = evaluate_escaped_ray_spectral(rgb2spec_table, lights, work.ray_d, work.lambda)

    # Apply path throughput
    contribution = work.beta * Le

    if !is_black(contribution)
        # MIS weighting following pbrt-v4 (integrator.cpp HandleEscapedRays)
        # depth=0 or specular bounce: L = beta * Le / r_u.Average()
        # Otherwise: L = beta * Le / (r_u + r_l).Average()
        #   where r_l = work.r_l * lightChoicePDF * light.PDF_Li(ctx, wi)
        final_contrib = if work.depth == Int32(0) || work.specular_bounce
            contribution / average(work.r_u)
        else
            # Full MIS: compute light sampling PDF and combine with BSDF PDF
            # r_l = work.r_l * lightChoicePDF * light.PDF_Li
            num_lights = Int32(length(lights))
            light_choice_pdf = num_lights > 0 ? 1f0 / Float32(num_lights) : 0f0

            # Compute PDF from environment light for this direction
            light_pdf = compute_env_light_pdf(lights, work.ray_d)
            r_l = work.r_l * light_choice_pdf * light_pdf

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

function vp_handle_escaped_rays!(state::VolPathState, lights)
    foreach(vp_handle_escaped_rays_kernel!,
        state.escaped_queue,
        state.pixel_L,
        state.rgb2spec_table,
        lights,
    )
    return nothing
end
