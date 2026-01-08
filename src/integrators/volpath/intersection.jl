# Ray tracing and intersection handling for VolPath
# Handles ray-scene intersection and classifies results into work queues

# ============================================================================
# Geometry Helpers (shared with PhysicalWavefront)
# ============================================================================

"""
    compute_geometric_normal(primitive) -> Vec3f

Compute geometric normal from a triangle primitive.
"""
@inline function vp_compute_geometric_normal(primitive)
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
@inline function vp_compute_uv_barycentric(primitive, barycentric)
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
@inline function vp_compute_shading_normal(primitive, barycentric, geometric_normal::Vec3f)
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

"""
    vp_trace_rays_kernel!(...)

Trace rays and classify into queues following pbrt-v4's architecture:
- Rays in medium -> medium_sample_queue (delta tracking with known t_max)
- Rays NOT in medium that miss -> escaped_queue
- Rays NOT in medium that hit -> hit_surface_queue

This implements the key pbrt-v4 pattern: intersection FIRST, then medium sampling.
For rays in media, we store both the ray and the surface hit info together,
so delta tracking can run with bounded t_max and process the hit if ray survives.
"""
@kernel function vp_trace_rays_kernel!(
    # Output queues
    medium_sample_items, medium_sample_size,   # Rays in medium with t_max
    escaped_items, escaped_size,                # Rays that missed (not in medium)
    hit_surface_items, hit_surface_size,        # Rays that hit (not in medium)
    # Input
    @Const(ray_items), @Const(ray_size),
    @Const(accel),
    @Const(max_queued::Int32)
)
    idx = @index(Global)

    @inbounds if idx <= max_queued
        current_size = ray_size[1]
        if idx <= current_size
            work = ray_items[idx]

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
                    new_idx = @atomic medium_sample_size[1] += Int32(1)
                    if new_idx <= max_queued
                        medium_sample_items[new_idx] = sample_item
                    end
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
                    new_idx = @atomic medium_sample_size[1] += Int32(1)
                    if new_idx <= max_queued
                        medium_sample_items[new_idx] = sample_item
                    end
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
                    new_idx = @atomic escaped_size[1] += Int32(1)
                    if new_idx <= max_queued
                        escaped_items[new_idx] = escaped_item
                    end
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
                    new_idx = @atomic hit_surface_size[1] += Int32(1)
                    if new_idx <= max_queued
                        hit_surface_items[new_idx] = hit_item
                    end
                end
            end
        end
    end
end

"""
    vp_trace_rays!(backend, state, accel)

Trace all rays in the current ray queue and populate output queues.
Following pbrt-v4: rays in medium go to medium_sample_queue with t_hit info.
"""
function vp_trace_rays!(
    backend,
    state::VolPathState,
    accel
)
    input_queue = current_ray_queue(state)
    n = queue_size(input_queue)
    n == 0 && return nothing

    kernel! = vp_trace_rays_kernel!(backend)
    kernel!(
        state.medium_sample_queue.items, state.medium_sample_queue.size,
        state.escaped_queue.items, state.escaped_queue.size,
        state.hit_surface_queue.items, state.hit_surface_queue.size,
        input_queue.items, input_queue.size,
        accel, state.escaped_queue.capacity;
        ndrange=Int(n)
    )

    KernelAbstractions.synchronize(backend)
    return nothing
end

# ============================================================================
# Shadow Ray Tracing Kernel (with medium transmittance)
# ============================================================================

"""
    vp_trace_shadow_rays_kernel!(...)

Trace shadow rays and accumulate unoccluded contributions.
For rays through media, computes transmittance along the ray.
Handles transmissive boundaries (MediumInterface) by tracing through them.
"""
@kernel function vp_trace_shadow_rays_kernel!(
    pixel_L,
    @Const(shadow_items), @Const(shadow_size),
    @Const(accel),
    @Const(materials),
    @Const(media),
    @Const(rgb2spec_scale), @Const(rgb2spec_coeffs), @Const(rgb2spec_res::Int32),
    @Const(max_queued::Int32)
)
    idx = @index(Global)

    rgb2spec_table = RGBToSpectrumTable(rgb2spec_res, rgb2spec_scale, rgb2spec_coeffs)

    @inbounds if idx <= max_queued
        current_size = shadow_size[1]
        if idx <= current_size
            work = shadow_items[idx]

            # Trace shadow ray, handling transmissive boundaries
            Tr, visible = trace_shadow_transmittance(
                accel, materials, media, rgb2spec_table,
                work.ray.o, work.ray.d, work.t_max, work.lambda, work.medium_idx
            )

            if visible && !is_black(Tr)
                # Apply transmittance to contribution
                contribution = work.Ld * Tr

                if !is_black(contribution)
                    # MIS weight
                    w = 1f0 / average(work.r_u + work.r_l)

                    # Add to pixel
                    pixel_idx = work.pixel_index
                    base_idx = (pixel_idx - Int32(1)) * Int32(4)

                    final_L = contribution * w
                    Atomix.@atomic pixel_L[base_idx + Int32(1)] += final_L[1]
                    Atomix.@atomic pixel_L[base_idx + Int32(2)] += final_L[2]
                    Atomix.@atomic pixel_L[base_idx + Int32(3)] += final_L[3]
                    Atomix.@atomic pixel_L[base_idx + Int32(4)] += final_L[4]
                end
            end
        end
    end
end

"""
    trace_shadow_transmittance(accel, materials, media, rgb2spec_table, origin, dir, t_max, lambda, medium_idx)

Trace a shadow ray computing transmittance through media and transmissive boundaries.
Returns (transmittance, visible) where visible is false if ray hits an opaque surface.

This follows pbrt-v4's approach: transmissive surfaces (MediumInterface) let the ray through,
while opaque surfaces block it.
"""
@inline function trace_shadow_transmittance(
    accel, materials, media, rgb2spec_table,
    origin::Point3f, dir::Vec3f, t_max::Float32, lambda::Wavelengths, medium_idx::MediumIndex
)
    Tr = SpectralRadiance(1f0)
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
                Tr = Tr * compute_transmittance_simple(
                    rgb2spec_table, media, current_medium,
                    ray_o, dir, t_remaining, lambda
                )
            end
            return (Tr, true)  # Visible
        end

        # Hit a surface - check if it's transmissive
        mat_idx = primitive.metadata::MaterialIndex
        material = get_material(materials, mat_idx)

        # Check if material is a MediumInterfaceIdx (transmissive boundary)
        # Note: MediumInterface is converted to MediumInterfaceIdx during scene building
        is_transmissive = material isa MediumInterfaceIdx

        if !is_transmissive
            # Opaque surface blocks the ray
            return (SpectralRadiance(0f0), false)
        end

        # Transmissive boundary - compute transmittance up to this point
        if has_medium(current_medium)
            Tr = Tr * compute_transmittance_simple(
                rgb2spec_table, media, current_medium,
                ray_o, dir, t_hit, lambda
            )
        end

        if is_black(Tr)
            return (Tr, true)  # Transmittance is zero, no point continuing
        end

        # Update medium based on crossing direction
        n = vp_compute_geometric_normal(primitive)
        entering = dot(dir, n) < 0f0
        current_medium = entering ? material.inside : material.outside

        # Move past this surface
        ray_o = Point3f(ray_o + dir * (t_hit + 1f-4))  # Small offset to avoid self-intersection
        t_remaining = t_remaining - t_hit - 1f-4
    end

    # Exceeded max bounces - treat as occluded
    return (SpectralRadiance(0f0), false)
end

"""
    compute_transmittance_simple(table, media, medium_idx, origin, dir, t_max, lambda)

Simple transmittance computation using Beer-Lambert law.
Assumes homogeneous medium properties along the ray.
"""
@inline function compute_transmittance_simple(
    rgb2spec_table, media, medium_idx::MediumIndex,
    origin::Point3f, dir::Vec3f, t_max::Float32, lambda::Wavelengths
)
    # Sample medium at midpoint
    mid_p = Point3f(origin + dir * (t_max * 0.5f0))
    mp = sample_point_dispatch(rgb2spec_table, media, medium_idx.medium_type, mid_p, lambda)

    # Beer-Lambert: T = exp(-σ_t * distance)
    σ_t = mp.σ_a + mp.σ_s
    return exp(-σ_t * t_max)
end

"""
    vp_trace_shadow_rays!(backend, state, accel, materials, media)

Trace shadow rays and accumulate contributions.
Handles transmissive boundaries (MediumInterface) by tracing through them.
"""
function vp_trace_shadow_rays!(
    backend,
    state::VolPathState,
    accel,
    materials,
    media
)
    n = queue_size(state.shadow_queue)
    n == 0 && return nothing

    kernel! = vp_trace_shadow_rays_kernel!(backend)
    kernel!(
        state.pixel_L,
        state.shadow_queue.items, state.shadow_queue.size,
        accel, materials, media,
        state.rgb2spec_table.scale, state.rgb2spec_table.coeffs, state.rgb2spec_table.res,
        state.shadow_queue.capacity;
        ndrange=Int(n)
    )

    KernelAbstractions.synchronize(backend)
    return nothing
end

# ============================================================================
# Escaped Ray Handling (Environment Light)
# ============================================================================

"""
    vp_handle_escaped_rays_kernel!(...)

Handle rays that escaped the scene by evaluating environment lights.
"""
@kernel function vp_handle_escaped_rays_kernel!(
    pixel_L,
    @Const(escaped_items), @Const(escaped_size),
    @Const(lights),
    @Const(rgb2spec_scale), @Const(rgb2spec_coeffs), @Const(rgb2spec_res::Int32),
    @Const(max_queued::Int32)
)
    idx = @index(Global)

    rgb2spec_table = RGBToSpectrumTable(rgb2spec_res, rgb2spec_scale, rgb2spec_coeffs)

    @inbounds if idx <= max_queued
        current_size = escaped_size[1]
        if idx <= current_size
            work = escaped_items[idx]

            # Evaluate environment lights
            Le = evaluate_escaped_ray_spectral(rgb2spec_table, lights, work.ray_d, work.lambda)

            # Apply path throughput
            contribution = work.beta * Le

            if !is_black(contribution)
                # MIS weight
                # If first bounce or specular, no MIS
                final_contrib = if work.depth == Int32(0) || work.specular_bounce
                    contribution / average(work.r_u)
                else
                    # Full MIS: 1 / (r_u + r_l).Average()
                    # For environment light, r_l would need light PDF
                    # Simplified for now
                    contribution / average(work.r_u)
                end

                # Add to pixel
                pixel_idx = work.pixel_index
                base_idx = (pixel_idx - Int32(1)) * Int32(4)

                Atomix.@atomic pixel_L[base_idx + Int32(1)] += final_contrib[1]
                Atomix.@atomic pixel_L[base_idx + Int32(2)] += final_contrib[2]
                Atomix.@atomic pixel_L[base_idx + Int32(3)] += final_contrib[3]
                Atomix.@atomic pixel_L[base_idx + Int32(4)] += final_contrib[4]
            end
        end
    end
end

"""
    vp_handle_escaped_rays!(backend, state, lights)

Evaluate environment lights for escaped rays.
"""
function vp_handle_escaped_rays!(
    backend,
    state::VolPathState,
    lights
)
    n = queue_size(state.escaped_queue)
    n == 0 && return nothing

    kernel! = vp_handle_escaped_rays_kernel!(backend)
    kernel!(
        state.pixel_L,
        state.escaped_queue.items, state.escaped_queue.size,
        lights,
        state.rgb2spec_table.scale, state.rgb2spec_table.coeffs, state.rgb2spec_table.res,
        state.escaped_queue.capacity;
        ndrange=Int(n)
    )

    KernelAbstractions.synchronize(backend)
    return nothing
end
