# Ray tracing and intersection handling for PhysicalWavefront
# Uses Raycore for BVH traversal

# ============================================================================
# Primary/Bounce Ray Intersection Kernel
# ============================================================================

"""
    pw_trace_rays_kernel!(...)

Trace rays from ray_queue, handle hits and misses:
- Misses -> push to escaped_ray_queue (for environment light)
- Hits on emissive -> push to hit_area_light_queue
- Hits on non-emissive -> push to material_eval_queue

This kernel does NOT generate shadow rays - that happens in direct lighting.
"""
@kernel inbounds=true function pw_trace_rays_kernel!(
    # Output queues
    escaped_queue,
    hit_light_queue,
    material_queue,
    # Input
    @Const(ray_queue),
    # Scene data
    @Const(accel),           # BVH/TLAS accelerator
    @Const(materials),       # Tuple of material arrays
    @Const(max_queued::Int32)
)
    idx = @index(Global)

     if idx <= max_queued
        current_size = ray_queue.size[1]
        if idx <= current_size
            work = ray_queue.items[idx]

            # Trace ray using Raycore - capture barycentric coordinates for interpolation
            hit, primitive, t_hit, barycentric = Raycore.closest_hit(accel, work.ray)

            if !hit
                # Ray escaped - push to escaped queue for environment light
                escaped_item = PWEscapedRayWorkItem(work)
                push!(escaped_queue, escaped_item)
            else
                # Got a hit - extract surface info
                raw_mat_idx = primitive.metadata.medium_interface_idx

                # Compute intersection point
                pi = Point3f(work.ray.o + work.ray.d * t_hit)

                # Get geometric normal from primitive
                n = compute_geometric_normal(primitive)
                wo = -work.ray.d

                # Compute UV coordinates using barycentric interpolation
                uv = compute_uv_barycentric(primitive, barycentric)

                # Compute shading normal using barycentric interpolation of vertex normals
                ns = compute_shading_normal(primitive, barycentric, n)
                # pbrt-v4 FaceForward: flip geometric normal to face shading normal.
                # The shading normal from vertex interpolation is authoritative for
                # smooth surfaces; the geometric normal must agree with it.
                n = dot(n, ns) < 0f0 ? -n : n
                dpdu, dpdv = compute_tangent_frame(primitive)
                dpdus, dpdvs = dpdu, dpdv

                # Resolve MixMaterial to get the actual material index
                # Following pbrt-v4: MixMaterial is resolved at intersection time
                # Note: textures parameter not available here, pass nothing (CPU textures are self-contained)
                mat_idx = resolve_mix_material(materials, nothing, raw_mat_idx, pi, wo, uv)

                # TODO: DiffuseAreaLight hit detection via arealight_flat_idx
                # (will be added when PhysicalWavefront gets DiffuseAreaLight support)

                # Push to material queue for BSDF evaluation
                if true
                    mat_item = PWMaterialEvalWorkItem(
                        pi, n, dpdu, dpdv,
                        ns, dpdus, dpdvs, uv,
                        work.depth, work.lambda, work.pixel_index,
                        work.any_non_specular_bounces,
                        wo, work.beta, work.r_u, work.eta_scale,
                        mat_idx
                    )
                    push!(material_queue, mat_item)
                end
            end
        end
    end
end

# ============================================================================
# Geometry Helpers
# ============================================================================

"""
    compute_geometric_normal(primitive) -> Vec3f

Compute geometric normal from a primitive (triangle).
"""
@propagate_inbounds function compute_geometric_normal(primitive)
    # For Raycore.Triangle, vertices is an SVector{3, Point3f}
    v0 = primitive.vertices[1]
    v1 = primitive.vertices[2]
    v2 = primitive.vertices[3]
    e1 = v1 - v0
    e2 = v2 - v0
    n = Vec3f(cross(e1, e2)...)
    return normalize(n)
end

"""
    compute_uv(primitive, p::Point3f) -> Point2f

Compute UV coordinates for a point on a primitive.
Uses barycentric interpolation for triangles.
"""
@propagate_inbounds function compute_uv(primitive, p::Point3f)
    # Compute barycentric coordinates
    v0 = primitive.vertices[1]
    v1 = primitive.vertices[2]
    v2 = primitive.vertices[3]

    e1 = v1 - v0
    e2 = v2 - v0
    vp = p - v0

    # Compute barycentric via dot products
    d11 = dot(e1, e1)
    d12 = dot(e1, e2)
    d22 = dot(e2, e2)
    dp1 = dot(vp, e1)
    dp2 = dot(vp, e2)

    denom = d11 * d22 - d12 * d12
    if abs(denom) < 1f-10
        return Point2f(0f0, 0f0)
    end

    v = (d22 * dp1 - d12 * dp2) / denom
    w = (d11 * dp2 - d12 * dp1) / denom
    u = 1f0 - v - w

    # Interpolate UVs from primitive.uv (SVector{3, Point2f})
    uv0 = primitive.uv[1]
    uv1 = primitive.uv[2]
    uv2 = primitive.uv[3]
    return Point2f(
        u * uv0[1] + v * uv1[1] + w * uv2[1],
        u * uv0[2] + v * uv1[2] + w * uv2[2]
    )
end

"""
    compute_tangent_frame(primitive) -> (dpdu::Vec3f, dpdv::Vec3f)

Compute tangent vectors for a primitive.
"""
@propagate_inbounds function compute_tangent_frame(primitive)
    v0 = primitive.vertices[1]
    v1 = primitive.vertices[2]
    v2 = primitive.vertices[3]

    e1 = Vec3f((v1 - v0)...)
    e2 = Vec3f((v2 - v0)...)

    # Use edges as tangent frame (could be improved with UV derivatives)
    dpdu = normalize(e1)
    n = normalize(Vec3f(cross(e1, e2)...))
    dpdv = normalize(Vec3f(cross(Vec3f(n...), dpdu)...))

    return (dpdu, dpdv)
end

"""
    compute_uv_barycentric(primitive, barycentric) -> Point2f

Compute UV coordinates using barycentric coordinates from ray intersection.
More accurate than recomputing barycentric from position.
"""
@propagate_inbounds function compute_uv_barycentric(primitive, barycentric)
    # Barycentric coordinates: (w, u, v) where w + u + v = 1
    w, u, v = barycentric[1], barycentric[2], barycentric[3]

    # Interpolate UVs from primitive.uv (SVector{3, Point2f})
    uv0 = primitive.uv[1]
    uv1 = primitive.uv[2]
    uv2 = primitive.uv[3]

    return Point2f(
        w * uv0[1] + u * uv1[1] + v * uv2[1],
        w * uv0[2] + u * uv1[2] + v * uv2[2]
    )
end

"""
    compute_shading_normal(primitive, barycentric, geometric_normal) -> Vec3f

Compute interpolated shading normal from vertex normals using barycentric coordinates.
Falls back to geometric normal if vertex normals are not available (NaN).
"""
@propagate_inbounds function compute_shading_normal(primitive, barycentric, geometric_normal::Vec3f)
    # Get vertex normals from primitive
    n0 = primitive.normals[1]
    n1 = primitive.normals[2]
    n2 = primitive.normals[3]

    # Check if normals are valid (not NaN - used as sentinel for missing normals)
    if isnan(n0[1]) || isnan(n1[1]) || isnan(n2[1])
        return geometric_normal
    end

    # Barycentric interpolation of normals
    w, u, v = barycentric[1], barycentric[2], barycentric[3]
    ns_x = w * n0[1] + u * n1[1] + v * n2[1]
    ns_y = w * n0[2] + u * n1[2] + v * n2[2]
    ns_z = w * n0[3] + u * n1[3] + v * n2[3]

    # Normalize the interpolated normal
    ns = normalize(Vec3f(ns_x, ns_y, ns_z))

    # Return raw interpolated shading normal WITHOUT flipping.
    # Following pbrt-v4: the shading normal is authoritative for smooth surfaces.
    # The geometric normal should be flipped to face the shading normal (FaceForward),
    # not the other way around. This is done at the call site.
    return ns
end

# ============================================================================
# Shadow Ray Intersection Kernel
# ============================================================================

"""
    pw_trace_shadow_rays_kernel!(pixel_L, shadow_queue, accel, max_queued)

Trace shadow rays and accumulate unoccluded contributions to pixel buffer.
"""
@kernel inbounds=true function pw_trace_shadow_rays_kernel!(
    pixel_L,  # Flat array: 4 floats per pixel (spectral)
    @Const(shadow_queue),
    @Const(accel),
    @Const(max_queued::Int32)
)
    idx = @index(Global)

     if idx <= max_queued
        current_size = shadow_queue.size[1]
        if idx <= current_size
            work = shadow_queue.items[idx]

            # Check for occlusion using any_hit
            # Create ray with limited t_max
            shadow_ray = Raycore.Ray(o=work.ray.o, d=work.ray.d, t_max=work.t_max)
            hit_result = Raycore.any_hit(accel, shadow_ray)
            occluded = hit_result[1]  # First element is the hit boolean

            if !occluded
                # Add contribution to pixel with MIS weighting
                # Following pbrt-v4 (intersect.h RecordShadowRayResult):
                # Ld = w.Ld / (w.r_u + w.r_l).Average()
                pixel_idx = work.pixel_index
                base_idx = (pixel_idx - Int32(1)) * Int32(4)

                # Compute MIS weight: 1 / average(r_u + r_l)
                # This is the key step that was missing!
                r_sum = work.r_u + work.r_l
                mis_weight_inv = (r_sum.data[1] + r_sum.data[2] + r_sum.data[3] + r_sum.data[4]) * 0.25f0

                # Apply MIS weight to Ld
                if mis_weight_inv > 0f0
                    mis_weight = 1f0 / mis_weight_inv
                    Ld_weighted = work.Ld * mis_weight
                    accumulate_spectrum!(pixel_L, base_idx, Ld_weighted)
                end
            end
        end
    end
end

# ============================================================================
# Escaped Ray Handling (Environment Light)
# ============================================================================

"""
    pw_handle_escaped_rays_kernel!(pixel_L, escaped_queue, lights, rgb2spec_table, max_queued)

Handle rays that escaped the scene by evaluating environment lights.
"""
@kernel inbounds=true function pw_handle_escaped_rays_kernel!(
    pixel_L,
    @Const(escaped_queue),
    @Const(lights),  # Tuple of lights
    @Const(rgb2spec_table),
    @Const(max_queued::Int32)
)
    idx = @index(Global)

     if idx <= max_queued
        current_size = escaped_queue.size[1]
        if idx <= current_size
            work = escaped_queue.items[idx]

            # Evaluate environment lights for this direction
            Le = evaluate_escaped_ray_spectral(rgb2spec_table, lights, work.ray_d, work.lambda)

            # Apply path throughput
            contribution = work.beta * Le

            if !is_black(contribution)
                # MIS weighting following pbrt-v4 (integrator.cpp HandleEscapedRays)
                # depth=0 or specular bounce: L = beta * Le / r_u.Average()
                # Otherwise: L = beta * Le / (r_u + r_l).Average()
                #   where r_l = work.r_l * lightChoicePDF * light.PDF_Li(ctx, wi)

                final_contrib = if work.depth == 0 || work.specular_bounce
                    # No MIS needed - divide by r_u only
                    r_u_avg = (work.r_u.data[1] + work.r_u.data[2] + work.r_u.data[3] + work.r_u.data[4]) * 0.25f0
                    if r_u_avg > 0f0
                        contribution / r_u_avg
                    else
                        contribution
                    end
                else
                    # Full MIS: compute light sampling PDF and combine with BSDF PDF
                    # r_l = work.r_l * lightChoicePDF * light.PDF_Li
                    # For uniform light selection: lightChoicePDF = 1/num_lights
                    # (but we handle multiple lights, so compute PDF for each that contributes)
                    num_lights = Int32(length(lights))
                    light_choice_pdf = if num_lights > 0
                        1f0 / Float32(num_lights)
                    else
                        0f0
                    end

                    # Compute PDF from environment light for this direction
                    light_pdf = compute_env_light_pdf(lights, work.ray_d)
                    r_l = work.r_l * light_choice_pdf * light_pdf

                    # Combine r_u and r_l
                    r_sum = work.r_u + r_l
                    r_avg = (r_sum.data[1] + r_sum.data[2] + r_sum.data[3] + r_sum.data[4]) * 0.25f0

                    if r_avg > 0f0
                        contribution / r_avg
                    else
                        contribution
                    end
                end

                # Add to pixel
                pixel_idx = work.pixel_index
                base_idx = (pixel_idx - Int32(1)) * Int32(4)

                accumulate_spectrum!(pixel_L, base_idx, final_contrib)
            end
        end
    end
end

# ============================================================================
# Hit Area Light Handling
# ============================================================================

"""
    pw_handle_hit_area_lights_kernel!(pixel_L, hit_light_queue, rgb2spec_table, materials, max_queued)

Handle rays that hit emissive surfaces.
"""
@kernel inbounds=true function pw_handle_hit_area_lights_kernel!(
    pixel_L,
    @Const(hit_light_queue),
    @Const(rgb2spec_table),
    @Const(materials),
    @Const(max_queued::Int32)
)
    idx = @index(Global)

     if idx <= max_queued
        current_size = hit_light_queue.size[1]
        if idx <= current_size
            work = hit_light_queue.items[idx]

            # Get emission from material
            Le = with_index(get_emission_spectral, materials, work.material_idx,
                rgb2spec_table, materials, work.wo, work.n, work.uv, work.lambda
            )

            # Apply path throughput
            contribution = work.beta * Le

            if !is_black(contribution)
                # MIS weight for area light
                # If first bounce or specular bounce, no MIS
                final_contrib = if work.depth == Int32(0) || work.specular_bounce
                    contribution
                else
                    # For MIS, would need light PDF and compare with BSDF PDF
                    # For now, use full contribution
                    contribution
                end

                # Add to pixel
                pixel_idx = work.pixel_index
                base_idx = (pixel_idx - Int32(1)) * Int32(4)

                accumulate_spectrum!(pixel_L, base_idx, final_contrib)
            end
        end
    end
end

# ============================================================================
# High-Level Trace Functions
# ============================================================================

"""
    pw_trace_rays!(backend, escaped_queue, hit_light_queue, material_queue,
                   ray_queue, accel, materials)

Trace all rays in ray_queue and populate output queues.
"""
function pw_trace_rays!(
    backend,
    escaped_queue::WorkQueue{PWEscapedRayWorkItem},
    hit_light_queue::WorkQueue{PWHitAreaLightWorkItem},
    material_queue::WorkQueue{PWMaterialEvalWorkItem},
    ray_queue::WorkQueue{PWRayWorkItem},
    accel,
    materials
)
    n = length(ray_queue)
    n == 0 && return nothing

    # Reset output queues
    empty!(escaped_queue)
    empty!(hit_light_queue)
    empty!(material_queue)

    kernel! = pw_trace_rays_kernel!(backend)
    kernel!(
        escaped_queue,
        hit_light_queue,
        material_queue,
        ray_queue,
        accel, materials, ray_queue.capacity;
        ndrange=Int(ray_queue.capacity)
    )

    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    pw_trace_shadow_rays!(backend, pixel_L, shadow_queue, accel)

Trace shadow rays and accumulate unoccluded contributions.
"""
function pw_trace_shadow_rays!(
    backend,
    pixel_L::AbstractVector{Float32},
    shadow_queue::WorkQueue{PWShadowRayWorkItem},
    accel
)
    n = length(shadow_queue)
    n == 0 && return nothing

    kernel! = pw_trace_shadow_rays_kernel!(backend)
    kernel!(pixel_L, shadow_queue, accel, shadow_queue.capacity; ndrange=Int(shadow_queue.capacity))

    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    pw_handle_escaped_rays!(backend, pixel_L, escaped_queue, lights, rgb2spec_table)

Evaluate environment lights for escaped rays.
"""
function pw_handle_escaped_rays!(
    backend,
    pixel_L::AbstractVector{Float32},
    escaped_queue::WorkQueue{PWEscapedRayWorkItem},
    lights,
    rgb2spec_table::RGBToSpectrumTable
)
    n = length(escaped_queue)
    n == 0 && return nothing

    kernel! = pw_handle_escaped_rays_kernel!(backend)
    kernel!(
        pixel_L, escaped_queue, lights,
        rgb2spec_table,
        escaped_queue.capacity;
        ndrange=Int(escaped_queue.capacity)
    )

    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    pw_handle_hit_area_lights!(backend, pixel_L, hit_light_queue, rgb2spec_table, materials)

Evaluate emission for rays that hit area lights.
"""
function pw_handle_hit_area_lights!(
    backend,
    pixel_L::AbstractVector{Float32},
    hit_light_queue::WorkQueue{PWHitAreaLightWorkItem},
    rgb2spec_table,
    materials
)
    n = length(hit_light_queue)
    n == 0 && return nothing

    kernel! = pw_handle_hit_area_lights_kernel!(backend)
    kernel!(pixel_L, hit_light_queue, rgb2spec_table, materials, hit_light_queue.capacity; ndrange=Int(hit_light_queue.capacity))

    KernelAbstractions.synchronize(backend)
    return nothing
end
