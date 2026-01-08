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
@kernel function pw_trace_rays_kernel!(
    # Output queues
    escaped_queue_items, escaped_queue_size,
    hit_light_queue_items, hit_light_queue_size,
    material_queue_items, material_queue_size,
    # Input
    @Const(ray_queue_items), @Const(ray_queue_size),
    # Scene data
    @Const(accel),           # BVH/TLAS accelerator
    @Const(materials),       # Tuple of material arrays
    @Const(max_queued::Int32)
)
    idx = @index(Global)

    @inbounds if idx <= max_queued
        current_size = ray_queue_size[1]
        if idx <= current_size
            work = ray_queue_items[idx]

            # Trace ray using Raycore - capture barycentric coordinates for interpolation
            hit, primitive, t_hit, barycentric = Raycore.closest_hit(accel, work.ray)

            if !hit
                # Ray escaped - push to escaped queue for environment light
                escaped_item = PWEscapedRayWorkItem(work)
                new_idx = @atomic escaped_queue_size[1] += Int32(1)
                escaped_queue_items[new_idx] = escaped_item
            else
                # Got a hit - extract surface info
                mat_idx = primitive.metadata::MaterialIndex

                # Compute intersection point
                pi = Point3f(work.ray.o + work.ray.d * t_hit)

                # Get geometric normal from primitive
                n = compute_geometric_normal(primitive)
                wo = -work.ray.d

                # Compute UV coordinates using barycentric interpolation
                uv = compute_uv_barycentric(primitive, barycentric)

                # Compute shading normal using barycentric interpolation of vertex normals
                ns = compute_shading_normal(primitive, barycentric, n)
                dpdu, dpdv = compute_tangent_frame(primitive)
                dpdus, dpdvs = dpdu, dpdv

                # Check if material is emissive
                is_em = is_emissive_dispatch(materials, mat_idx)

                if is_em
                    # Hit an area light - push to hit_light queue
                    hit_light_item = PWHitAreaLightWorkItem(
                        pi, n, uv, wo,
                        work.lambda, work.depth, work.beta,
                        work.r_u, work.r_l, work.prev_intr_ctx,
                        work.specular_bounce, work.pixel_index,
                        mat_idx
                    )
                    new_idx = @atomic hit_light_queue_size[1] += Int32(1)
                    hit_light_queue_items[new_idx] = hit_light_item
                end

                # Always push to material queue for BSDF evaluation
                # (emissive materials have no BSDF, but we check there)
                if !is_em
                    mat_item = PWMaterialEvalWorkItem(
                        pi, n, dpdu, dpdv,
                        ns, dpdus, dpdvs, uv,
                        work.depth, work.lambda, work.pixel_index,
                        work.any_non_specular_bounces,
                        wo, work.beta, work.r_u, work.eta_scale,
                        mat_idx
                    )
                    new_idx = @atomic material_queue_size[1] += Int32(1)
                    material_queue_items[new_idx] = mat_item
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
@inline function compute_geometric_normal(primitive)
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
@inline function compute_uv(primitive, p::Point3f)
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
@inline function compute_tangent_frame(primitive)
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
@inline function compute_uv_barycentric(primitive, barycentric)
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
@inline function compute_shading_normal(primitive, barycentric, geometric_normal::Vec3f)
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

    # Ensure shading normal is on the same side as geometric normal
    if dot(ns, geometric_normal) < 0f0
        return -ns
    else
        return ns
    end
end

# ============================================================================
# Shadow Ray Intersection Kernel
# ============================================================================

"""
    pw_trace_shadow_rays_kernel!(pixel_L, shadow_queue_items, shadow_queue_size,
                                  accel, max_queued)

Trace shadow rays and accumulate unoccluded contributions to pixel buffer.
"""
@kernel function pw_trace_shadow_rays_kernel!(
    pixel_L,  # Flat array: 4 floats per pixel (spectral)
    @Const(shadow_queue_items), @Const(shadow_queue_size),
    @Const(accel),
    @Const(max_queued::Int32)
)
    idx = @index(Global)

    @inbounds if idx <= max_queued
        current_size = shadow_queue_size[1]
        if idx <= current_size
            work = shadow_queue_items[idx]

            # Check for occlusion using any_hit
            # Create ray with limited t_max
            shadow_ray = Raycore.Ray(o=work.ray.o, d=work.ray.d, t_max=work.t_max)
            hit_result = Raycore.any_hit(accel, shadow_ray)
            occluded = hit_result[1]  # First element is the hit boolean

            if !occluded
                # Add contribution to pixel
                pixel_idx = work.pixel_index
                base_idx = (pixel_idx - Int32(1)) * Int32(4)

                # Atomic add for each spectral channel
                Atomix.@atomic pixel_L[base_idx + Int32(1)] += work.Ld.data[1]
                Atomix.@atomic pixel_L[base_idx + Int32(2)] += work.Ld.data[2]
                Atomix.@atomic pixel_L[base_idx + Int32(3)] += work.Ld.data[3]
                Atomix.@atomic pixel_L[base_idx + Int32(4)] += work.Ld.data[4]
            end
        end
    end
end

# ============================================================================
# Escaped Ray Handling (Environment Light)
# ============================================================================

"""
    pw_handle_escaped_rays_kernel!(pixel_L, escaped_queue_items, escaped_queue_size,
                                    lights, rgb2spec_table, max_queued)

Handle rays that escaped the scene by evaluating environment lights.
"""
@kernel function pw_handle_escaped_rays_kernel!(
    pixel_L,
    @Const(escaped_queue_items), @Const(escaped_queue_size),
    @Const(lights),  # Tuple of lights
    @Const(rgb2spec_scale),  # RGB to spectrum table scale array
    @Const(rgb2spec_coeffs), # RGB to spectrum table coefficients
    @Const(rgb2spec_res::Int32),  # RGB to spectrum table resolution
    @Const(max_queued::Int32)
)
    idx = @index(Global)

    # Reconstruct table struct from components for GPU compatibility
    rgb2spec_table = RGBToSpectrumTable(rgb2spec_res, rgb2spec_scale, rgb2spec_coeffs)

    @inbounds if idx <= max_queued
        current_size = escaped_queue_size[1]
        if idx <= current_size
            work = escaped_queue_items[idx]

            # Evaluate environment lights for this direction
            Le = evaluate_escaped_ray_spectral(rgb2spec_table, lights, work.ray_d, work.lambda)

            # Apply path throughput
            contribution = work.beta * Le

            if !is_black(contribution)
                # MIS weight for environment light
                # If this was a specular bounce, no MIS needed
                # Otherwise, compute weight
                final_contrib = if work.specular_bounce
                    contribution
                else
                    # For now, use contribution directly
                    # Full MIS would compare with BSDF sampling probability
                    contribution
                end

                # Add to pixel
                pixel_idx = work.pixel_index
                base_idx = (pixel_idx - Int32(1)) * Int32(4)

                Atomix.@atomic pixel_L[base_idx + Int32(1)] += final_contrib.data[1]
                Atomix.@atomic pixel_L[base_idx + Int32(2)] += final_contrib.data[2]
                Atomix.@atomic pixel_L[base_idx + Int32(3)] += final_contrib.data[3]
                Atomix.@atomic pixel_L[base_idx + Int32(4)] += final_contrib.data[4]
            end
        end
    end
end

# ============================================================================
# Hit Area Light Handling
# ============================================================================

"""
    pw_handle_hit_area_lights_kernel!(pixel_L, hit_light_queue_items, hit_light_queue_size,
                                       materials, max_queued)

Handle rays that hit emissive surfaces.
"""
@kernel function pw_handle_hit_area_lights_kernel!(
    pixel_L,
    @Const(hit_light_queue_items), @Const(hit_light_queue_size),
    @Const(materials),
    @Const(max_queued::Int32)
)
    idx = @index(Global)

    @inbounds if idx <= max_queued
        current_size = hit_light_queue_size[1]
        if idx <= current_size
            work = hit_light_queue_items[idx]

            # Get emission from material
            Le = get_emission_spectral_dispatch(
                materials, work.material_idx,
                work.wo, work.n, work.uv, work.lambda
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

                Atomix.@atomic pixel_L[base_idx + Int32(1)] += final_contrib.data[1]
                Atomix.@atomic pixel_L[base_idx + Int32(2)] += final_contrib.data[2]
                Atomix.@atomic pixel_L[base_idx + Int32(3)] += final_contrib.data[3]
                Atomix.@atomic pixel_L[base_idx + Int32(4)] += final_contrib.data[4]
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
    escaped_queue::PWWorkQueue{PWEscapedRayWorkItem},
    hit_light_queue::PWWorkQueue{PWHitAreaLightWorkItem},
    material_queue::PWWorkQueue{PWMaterialEvalWorkItem},
    ray_queue::PWWorkQueue{PWRayWorkItem},
    accel,
    materials
)
    n = queue_size(ray_queue)
    n == 0 && return nothing

    # Reset output queues
    reset_queue!(backend, escaped_queue)
    reset_queue!(backend, hit_light_queue)
    reset_queue!(backend, material_queue)

    kernel! = pw_trace_rays_kernel!(backend)
    kernel!(
        escaped_queue.items, escaped_queue.size,
        hit_light_queue.items, hit_light_queue.size,
        material_queue.items, material_queue.size,
        ray_queue.items, ray_queue.size,
        accel, materials, Int32(n);
        ndrange=Int(n)
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
    shadow_queue::PWWorkQueue{PWShadowRayWorkItem},
    accel
)
    n = queue_size(shadow_queue)
    n == 0 && return nothing

    kernel! = pw_trace_shadow_rays_kernel!(backend)
    kernel!(pixel_L, shadow_queue.items, shadow_queue.size, accel, Int32(n); ndrange=Int(n))

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
    escaped_queue::PWWorkQueue{PWEscapedRayWorkItem},
    lights,
    rgb2spec_table::RGBToSpectrumTable
)
    n = queue_size(escaped_queue)
    n == 0 && return nothing

    kernel! = pw_handle_escaped_rays_kernel!(backend)
    kernel!(
        pixel_L, escaped_queue.items, escaped_queue.size, lights,
        rgb2spec_table.scale, rgb2spec_table.coeffs, rgb2spec_table.res,
        Int32(n);
        ndrange=Int(n)
    )

    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    pw_handle_hit_area_lights!(backend, pixel_L, hit_light_queue, materials)

Evaluate emission for rays that hit area lights.
"""
function pw_handle_hit_area_lights!(
    backend,
    pixel_L::AbstractVector{Float32},
    hit_light_queue::PWWorkQueue{PWHitAreaLightWorkItem},
    materials
)
    n = queue_size(hit_light_queue)
    n == 0 && return nothing

    kernel! = pw_handle_hit_area_lights_kernel!(backend)
    kernel!(pixel_L, hit_light_queue.items, hit_light_queue.size, materials, Int32(n); ndrange=Int(n))

    KernelAbstractions.synchronize(backend)
    return nothing
end
