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

# ============================================================================
# Primary Ray Intersection Kernel
# ============================================================================

@propagate_inbounds function vp_trace_rays_kernel!(
    work,
    medium_sample_queue,
    escaped_queue,
    hit_surface_queue,
    accel
)
    # Trace ray to find closest intersection
    hit, primitive, t_hit, barycentric = Raycore.closest_hit(accel, work.ray)

    # Check if ray is currently traveling through a medium
    if has_medium(work.medium_idx)
        # Ray is in medium - route to medium_sample_queue with intersection info
        # Delta tracking will use t_hit as t_max

        if hit
            # Have surface hit info for when ray survives delta tracking
            mat_idx = primitive.metadata::MaterialIndex
            pi = Point3f(work.ray.o + work.ray.d * t_hit)
            n = vp_compute_geometric_normal(primitive)
            uv = vp_compute_uv_barycentric(primitive, barycentric)
            ns = vp_compute_shading_normal(primitive, barycentric, n)

            sample_item = VPMediumSampleWorkItem(
                work.ray,
                work.depth,
                t_hit,
                work.lambda,
                work.beta,
                work.r_u,
                work.r_l,
                work.pixel_index,
                work.eta_scale,
                work.specular_bounce,
                work.any_non_specular_bounces,
                work.prev_intr_p,
                work.prev_intr_n,
                work.medium_idx,
                true,   # has_surface_hit
                pi, n, ns, uv, mat_idx
            )
            push!(medium_sample_queue, sample_item)
        else
            # Ray in medium but escaped scene - t_max = Infinity
            sample_item = VPMediumSampleWorkItem(
                work.ray,
                work.depth,
                Inf32,
                work.lambda,
                work.beta,
                work.r_u,
                work.r_l,
                work.pixel_index,
                work.eta_scale,
                work.specular_bounce,
                work.any_non_specular_bounces,
                work.prev_intr_p,
                work.prev_intr_n,
                work.medium_idx,
                false,  # no surface hit
                Point3f(0f0), Vec3f(0f0, 0f0, 1f0), Vec3f(0f0, 0f0, 1f0),
                Point2f(0f0, 0f0), MaterialIndex()
            )
            push!(medium_sample_queue, sample_item)
        end
    else
        # Ray NOT in medium - route directly
        if !hit
            # Ray escaped - push to escaped queue
            escaped_item = VPEscapedRayWorkItem(
                work.ray.d,
                work.lambda,
                work.pixel_index,
                work.beta,
                work.r_u,
                work.r_l,
                work.depth,
                work.specular_bounce,
                work.prev_intr_p,
                work.prev_intr_n
            )
            push!(escaped_queue, escaped_item)
        else
            # Hit surface - extract geometry
            mat_idx = primitive.metadata::MaterialIndex
            pi = Point3f(work.ray.o + work.ray.d * t_hit)
            n = vp_compute_geometric_normal(primitive)
            uv = vp_compute_uv_barycentric(primitive, barycentric)
            ns = vp_compute_shading_normal(primitive, barycentric, n)

            hit_item = VPHitSurfaceWorkItem(
                work.ray,
                pi, n, ns, uv, mat_idx,
                work.lambda,
                work.pixel_index,
                work.beta,
                work.r_u,
                work.r_l,
                work.depth,
                work.eta_scale,
                work.specular_bounce,
                work.any_non_specular_bounces,
                work.prev_intr_p,
                work.prev_intr_n,
                work.medium_idx,
                t_hit
            )
            push!(hit_surface_queue, hit_item)
        end
    end
end

function vp_trace_rays!(state::VolPathState, accel)
    input_queue = current_ray_queue(state)
    foreach(vp_trace_rays_kernel!,
        input_queue,
        state.medium_sample_queue,
        state.escaped_queue,
        state.hit_surface_queue,
        accel,
    )
    return nothing
end

# ============================================================================
# Shadow Ray Tracing Kernel (with medium transmittance)
# ============================================================================


# ============================================================================
# Helper functions for with_material dispatch (no variable capture)
# ============================================================================

"""
Get the medium index when crossing a material boundary.
Used with with_material for GPU-safe dispatch.
Returns (is_transmissive, new_medium_idx).
"""
@propagate_inbounds function _get_crossing_info(material, entering::Bool)
    is_trans = is_medium_interface_idx(material)
    new_medium = get_crossing_medium(material, entering)
    return (is_trans, new_medium)
end

"""
    trace_shadow_transmittance(accel, materials, media, rgb2spec_table, origin, dir, t_max, lambda, medium_idx)

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
    accel, materials, media, rgb2spec_table,
    origin::Point3f, dir::Vec3f, t_max::Float32, lambda::Wavelengths, medium_idx::MediumIndex
)
    # Transmittance and MIS weights following pbrt-v4
    T_ray = SpectralRadiance(1f0)
    r_u = SpectralRadiance(1f0)
    r_l = SpectralRadiance(1f0)

    current_medium = medium_idx
    ray_o = origin
    t_remaining = t_max

    # Trace through up to N transmissive boundaries
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

        # Hit a surface - check if it's transmissive using with_material for type stability
        mat_idx = primitive.metadata::MaterialIndex

        # Use with_material to get crossing info with concrete material type
        # This avoids union type issues on GPU by calling helper with concrete type
        n = vp_compute_geometric_normal(primitive)
        entering = dot(dir, n) < 0f0
        is_transmissive, new_medium = with_material(_get_crossing_info, materials, mat_idx, entering)

        if !is_transmissive
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

        # Update medium based on crossing direction (already computed above)
        current_medium = new_medium

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
Following pbrt-v4's TraceTransmittance implementation.

Ratio tracking uses delta tracking but only considers null-scattering events,
providing an unbiased estimate of transmittance through heterogeneous media.

Returns (T_ray, r_u, r_l) where:
- T_ray: spectral transmittance estimate
- r_u, r_l: MIS weight accumulators for combining with path weights
"""
@propagate_inbounds function compute_transmittance_ratio_tracking(
    rgb2spec_table, media, medium_idx::MediumIndex,
    origin::Point3f, dir::Vec3f, t_max::Float32, lambda::Wavelengths
)
    medium_type_idx = medium_idx.medium_type

    # Get majorant for this ray segment
    ray = Raycore.Ray(o=origin, d=dir)
    majorant = get_majorant_dispatch(rgb2spec_table, media, medium_type_idx, ray, 0f0, t_max, lambda)

    σ_maj_0 = majorant.σ_maj[1]  # First wavelength for sampling

    # Handle zero majorant (empty medium)
    if σ_maj_0 < 1f-10
        return (SpectralRadiance(1f0), SpectralRadiance(1f0), SpectralRadiance(1f0))
    end

    # Initialize PCG32 RNG seeded with ray origin and direction (matching pbrt-v4)
    # See pbrt-v4/src/pbrt/wavefront/intersect.h: RNG rng(Hash(ray.o), Hash(ray.d))
    rng = pcg32_init(pbrt_hash(origin), pbrt_hash(dir))

    # Ratio tracking state
    T_ray = SpectralRadiance(1f0)
    r_u = SpectralRadiance(1f0)
    r_l = SpectralRadiance(1f0)
    t = 0f0

    # Step through medium using ratio tracking
    max_iterations = Int32(256)
    for _ in 1:max_iterations
        # Sample exponential distance using PCG32
        u, rng = pcg32_uniform_f32(rng)
        dt = -log(max(1f-10, 1f0 - u)) / σ_maj_0
        t_sample = t + dt

        if t_sample >= t_max
            # Reached end of segment - apply remaining transmittance
            dt_remain = t_max - t
            T_maj = exp(-dt_remain * majorant.σ_maj)
            T_maj_0 = T_maj[1]
            if T_maj_0 > 1f-10
                T_ray = T_ray * T_maj / T_maj_0
                r_l = r_l * T_maj / T_maj_0
                r_u = r_u * T_maj / T_maj_0
            end
            break
        end

        # Sample medium properties at this point
        p = Point3f(origin + dir * t_sample)
        mp = sample_point_dispatch(rgb2spec_table, media, medium_type_idx, p, lambda)

        # Compute null-scattering coefficient
        σ_n = majorant.σ_maj - mp.σ_a - mp.σ_s
        # Clamp negative values
        σ_n = SpectralRadiance(max(σ_n[1], 0f0), max(σ_n[2], 0f0), max(σ_n[3], 0f0), max(σ_n[4], 0f0))

        # Compute transmittance for this segment
        T_maj = exp(-dt * majorant.σ_maj)

        # Ratio tracking update (only null-scattering for transmittance)
        # Following pbrt-v4: T_ray *= T_maj * sigma_n / pr
        pr = T_maj[1] * σ_maj_0
        if pr > 1f-10
            T_ray = T_ray * T_maj * σ_n / pr
            r_l = r_l * T_maj * majorant.σ_maj / pr
            r_u = r_u * T_maj * σ_n / pr
        else
            T_ray = SpectralRadiance(0f0)
            break
        end

        # Russian roulette termination (following pbrt-v4)
        Tr_estimate = T_ray / max(1f-10, average(r_l + r_u))
        if max_component(Tr_estimate) < 0.05f0
            q = 0.75f0
            rr_sample, rng = pcg32_uniform_f32(rng)
            if rr_sample < q
                T_ray = SpectralRadiance(0f0)
                break
            else
                T_ray = T_ray / (1f0 - q)
            end
        end

        if is_black(T_ray)
            break
        end

        t = t_sample
    end

    return (T_ray, r_u, r_l)
end

"""
    compute_transmittance_simple(table, media, medium_idx, origin, dir, t_max, lambda)

Simple wrapper that returns only the transmittance (for compatibility).
"""
@propagate_inbounds function compute_transmittance_simple(
    rgb2spec_table, media, medium_idx::MediumIndex,
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
        materials,
        media,
    )
    # Trace shadow ray, handling transmissive boundaries
    # Returns (T_ray, tr_r_u, tr_r_l, visible) following pbrt-v4's TraceTransmittance
    T_ray, tr_r_u, tr_r_l, visible = trace_shadow_transmittance(
        accel, materials, media, rgb2spec_table,
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
        materials,
        media
    )
    foreach(vp_trace_shadow_rays_kernel!,
        state.shadow_queue,
        state.pixel_L,
        state.rgb2spec_table,
        accel, materials, media,
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
