# Surface material evaluation for VolPath
# Handles BSDF sampling, direct lighting, and path continuation at surface hits
#
# NOTE: `materials` is a StaticMultiTypeSet containing both materials and textures.
# Use eval_tex(materials, field, uv) to sample textures via TextureRef.

# ============================================================================
# Texture Filtering Derivatives (pbrt-v4 style)
# ============================================================================

"""
    approximate_dp_dxy(pi, n, camera, samples_per_pixel) -> (dpdx, dpdy)

Approximate screen-space position derivatives at intersection point.
Following pbrt-v4's Camera::Approximate_dp_dxy method.

This estimates how much the surface position changes per pixel, which is used
for texture filtering (mipmap level selection). The approximation assumes the
surface is locally planar near the intersection point.

For a perspective camera, this is approximately:
    dp/dscreen ≈ distance * tan(fov/2) / (resolution/2)

Arguments:
- `pi`: Intersection point in world space
- `n`: Surface normal at intersection
- `camera`: Camera with dx_camera, dy_camera precomputed
- `samples_per_pixel`: Number of samples per pixel (for scaling)

Returns (dpdx, dpdy) - approximate change in position per screen pixel.
"""
@propagate_inbounds function approximate_dp_dxy(
    pi::Point3f, n::Vec3f, camera, samples_per_pixel::Int32
)
    # Transform intersection point to camera space
    # camera.core.core.camera_to_world is world->camera transform (inverted from usual)
    # We need camera->world inverse = world->camera
    camera_to_world = camera.core.core.camera_to_world

    # For perspective camera, approximate dpdx/dpdy based on distance and fov
    # The camera has precomputed dx_camera and dy_camera (change per pixel in camera space)

    # Get camera position (world space origin)
    camera_pos = camera_to_world(Point3f(0f0))

    # Distance from camera to intersection point
    to_point = Vec3f(pi - camera_pos)
    dist = sqrt(dot(to_point, to_point))

    # Scale factor based on distance and samples per pixel
    # Following pbrt-v4: scale by sqrt(samples) for antialiasing
    scale = dist / sqrt(Float32(max(1, samples_per_pixel)))

    # Transform dx_camera and dy_camera to world space
    # These represent how the ray direction changes per pixel
    dx_world = camera_to_world(camera.dx_camera)
    dy_world = camera_to_world(camera.dy_camera)

    # Project onto the tangent plane at the intersection
    # dpdx ≈ scale * (dx_world - n * dot(n, dx_world))
    dpdx = scale * (dx_world - n * dot(n, dx_world))
    dpdy = scale * (dy_world - n * dot(n, dy_world))

    return dpdx, dpdy
end

"""
    compute_uv_derivatives(dpdu, dpdv, dpdx, dpdy) -> (dudx, dudy, dvdx, dvdy)

Compute UV derivatives from position derivatives using least-squares solve.
Following pbrt-v4's SurfaceInteraction::ComputeDifferentials.

Given:
- dpdu, dpdv: How position changes with UV (∂p/∂u, ∂p/∂v)
- dpdx, dpdy: How position changes with screen pixel (∂p/∂x, ∂p/∂y)

Solve for:
- dudx, dudy: How u changes with screen pixel (∂u/∂x, ∂u/∂y)
- dvdx, dvdy: How v changes with screen pixel (∂v/∂x, ∂v/∂y)

Uses the normal equations: (A^T A) [du/dx; dv/dx]^T = A^T [dpdx]
where A = [dpdu | dpdv] is a 3x2 matrix.
"""
@propagate_inbounds function compute_uv_derivatives(
    dpdu::Vec3f, dpdv::Vec3f, dpdx::Vec3f, dpdy::Vec3f
)
    # Compute A^T A (2x2 matrix)
    ata00 = dot(dpdu, dpdu)
    ata01 = dot(dpdu, dpdv)
    ata11 = dot(dpdv, dpdv)

    # Compute determinant and check for degeneracy
    det = ata00 * ata11 - ata01 * ata01
    if abs(det) < 1f-10
        return (0f0, 0f0, 0f0, 0f0)
    end
    inv_det = 1f0 / det

    # Compute A^T b for x direction
    atb0x = dot(dpdu, dpdx)
    atb1x = dot(dpdv, dpdx)

    # Compute A^T b for y direction
    atb0y = dot(dpdu, dpdy)
    atb1y = dot(dpdv, dpdy)

    # Solve using Cramer's rule (2x2 system)
    # [dudx]   [ata11  -ata01] [atb0x]
    # [dvdx] = [-ata01  ata00] [atb1x] * inv_det
    dudx = (ata11 * atb0x - ata01 * atb1x) * inv_det
    dvdx = (ata00 * atb1x - ata01 * atb0x) * inv_det
    dudy = (ata11 * atb0y - ata01 * atb1y) * inv_det
    dvdy = (ata00 * atb1y - ata01 * atb0y) * inv_det

    # Clamp to reasonable values (following pbrt-v4)
    clamp_val = 1f8
    dudx = clamp(dudx, -clamp_val, clamp_val)
    dvdx = clamp(dvdx, -clamp_val, clamp_val)
    dudy = clamp(dudy, -clamp_val, clamp_val)
    dvdy = clamp(dvdy, -clamp_val, clamp_val)

    return (dudx, dudy, dvdx, dvdy)
end

# TextureFilterContext is defined in textures/texture-ref.jl

"""
    compute_texture_filter_context(work, camera, samples_per_pixel) -> TextureFilterContext

Compute texture filtering context from material evaluation work item.
Uses approximate screen-space derivatives for proper mipmap selection.
"""
@propagate_inbounds function compute_texture_filter_context(
    work::VPMaterialEvalWorkItem, camera, samples_per_pixel::Int32
)
    # Compute screen-space position derivatives
    dpdx, dpdy = approximate_dp_dxy(work.pi, work.n, camera, samples_per_pixel)

    # Compute UV derivatives from position derivatives
    dudx, dudy, dvdx, dvdy = compute_uv_derivatives(work.dpdu, work.dpdv, dpdx, dpdy)

    return TextureFilterContext(work.uv, dudx, dudy, dvdx, dvdy)
end

# ============================================================================
# Process Surface Hits - Emission and Material Queue Setup
# ============================================================================

@propagate_inbounds function vp_process_surface_hits_kernel!(
    work,
    material_queue,
    pixel_L,
    materials,
    rgb2spec_table
)
    wo = -work.ray.d

    # Resolve MixMaterial to get the actual material index
    # Following pbrt-v4: MixMaterial is resolved at intersection time
    # using stochastic selection based on the amount texture and a hash
    material_idx = resolve_mix_material(
        materials, work.material_idx,
        work.pi, wo, work.uv
    )

    # Check if surface is emissive
    if is_emissive(materials, material_idx)
        # Get emission (use simple TextureFilterContext for emission - no derivatives needed)
        tfc = TextureFilterContext(work.uv)
        Le = get_emission_spectral_dispatch(
            rgb2spec_table, materials, material_idx,
            wo, work.n, tfc, work.lambda
        )

        if !is_black(Le)
            # Apply path throughput
            contribution = work.beta * Le

            # MIS weight: on first bounce or specular, no MIS
            final_contrib = if work.depth == Int32(0) || work.specular_bounce
                contribution / average(work.r_u)
            else
                # Full MIS weight
                contribution / average(work.r_u + work.r_l)
            end

            # Add to pixel
            pixel_idx = work.pixel_index
            base_idx = (pixel_idx - Int32(1)) * Int32(4)

            accumulate_spectrum!(pixel_L, base_idx, final_contrib)
        end
    end

    # Create material evaluation work item (for non-emissive or mixed materials)
    # Skip pure emissive materials (using resolved material_idx)
    if !is_pure_emissive_dispatch(materials, material_idx)
        push!(material_queue, VPMaterialEvalWorkItem(work, wo, material_idx))
    end
end

function vp_process_surface_hits!(state::VolPathState, materials)
    foreach(vp_process_surface_hits_kernel!,
        state.hit_surface_queue,
        state.material_queue,
        state.pixel_L,
        materials,
        state.rgb2spec_table,
    )
    return nothing
end

# ============================================================================
# Direct Lighting Inner Function
# ============================================================================

"""Inner function for surface direct lighting - can use return statements.

Uses power-weighted light sampling via alias table for better importance sampling
in scenes with lights of varying intensities (pbrt-v4's PowerLightSampler approach).

Now uses pre-computed Sobol samples from pixel_samples (pbrt-v4 RaySamples style).
"""
@propagate_inbounds function surface_direct_lighting_inner!(
    shadow_queue,
    work::VPMaterialEvalWorkItem,
    materials,
    lights,
    rgb2spec_table,
    light_sampler_p,
    light_sampler_q,
    light_sampler_alias,
    num_lights::Int32,
    # Pre-computed Sobol samples (SOA layout)
    pixel_samples_direct_uc,
    pixel_samples_direct_u,
    # Camera for texture filtering (pbrt-v4 style)
    camera,
    samples_per_pixel::Int32
)
    # Skip if no lights
    num_lights < Int32(1) && return

    # Use pre-computed Sobol samples for light sampling (pbrt-v4 RaySamples.direct)
    pixel_idx = work.pixel_index
    u_light = pixel_samples_direct_u[pixel_idx]
    light_select = pixel_samples_direct_uc[pixel_idx]

    # Select light using power-weighted alias table sampling
    # Returns (1-based index, PMF for that light)
    light_idx, light_pmf = sample_light_sampler(
        light_sampler_p, light_sampler_q, light_sampler_alias, light_select
    )

    # Validate index (should always be valid if sampler was built correctly)
    if light_idx < Int32(1) || light_idx > num_lights || light_pmf <= 0f0
        return
    end

    # Sample the light (works with both Tuple and StaticMultiTypeSet)
    light_sample = sample_light_spectral(
        rgb2spec_table, lights, light_idx, work.pi, work.lambda, u_light
    )

    if light_sample.pdf > 0f0 && !is_black(light_sample.Li)
        # Compute texture filter context with proper screen-space derivatives (pbrt-v4 style)
        tfc = compute_texture_filter_context(work, camera, samples_per_pixel)

        # Evaluate BSDF for light direction
        bsdf_f, bsdf_pdf = evaluate_spectral_material(
            rgb2spec_table, materials, work.material_idx,
            work.wo, light_sample.wi, work.ns, tfc, work.lambda
        )

        if !is_black(bsdf_f)
            # Compute direct lighting contribution with MIS
            result = compute_direct_lighting_spectral(
                work.pi, work.ns, work.wo, work.beta, work.r_u, work.lambda,
                light_sample, bsdf_f, bsdf_pdf
            )

            if result.valid
                # Following pbrt-v4 (surfscatter.cpp lines 299-315):
                # - Ld = beta * f * Li (no PDF division - that happens at shadow ray resolution)
                # - r_l = r_u * lightPDF where lightPDF = ls.pdf * light_pmf
                # compute_direct_lighting_spectral already sets r_l = r_u * ls.pdf
                # So we multiply by light_pmf to get the full light PDF in r_l
                scaled_r_l = result.r_l * light_pmf

                # Create shadow ray
                shadow_ray = Raycore.Ray(
                    o = result.ray_origin,
                    d = result.ray_direction,
                    t_max = result.t_max
                )

                shadow_item = VPShadowRayWorkItem(
                    shadow_ray,
                    result.t_max,
                    work.lambda,
                    result.Ld,  # NOT divided by light_pmf - MIS handles this
                    result.r_u,
                    scaled_r_l,
                    work.pixel_index,
                    work.current_medium  # Shadow ray starts in same medium
                )

                push!(shadow_queue, shadow_item)
            end
        end
    end
    return
end

# ============================================================================
# Direct Lighting at Surface Hits
# ============================================================================

@propagate_inbounds function vp_sample_surface_direct_lighting_kernel!(
    work,
    shadow_queue,
    materials,
    lights,
    rgb2spec_table,
    light_sampler_p, light_sampler_q, light_sampler_alias,
    num_lights::Int32,
    pixel_samples_direct_uc, pixel_samples_direct_u,
    camera, samples_per_pixel::Int32
)
    surface_direct_lighting_inner!(
        shadow_queue,
        work, materials, lights, rgb2spec_table,
        light_sampler_p, light_sampler_q, light_sampler_alias,
        num_lights,
        pixel_samples_direct_uc, pixel_samples_direct_u,
        camera, samples_per_pixel
    )
end

function vp_sample_surface_direct_lighting!(state::VolPathState, materials, lights, camera, samples_per_pixel::Int32)
    pixel_samples = state.pixel_samples
    foreach(vp_sample_surface_direct_lighting_kernel!,
        state.material_queue,
        state.shadow_queue,
        materials,
        lights,
        state.rgb2spec_table,
        state.light_sampler_p, state.light_sampler_q, state.light_sampler_alias,
        state.num_lights,
        pixel_samples.direct_uc, pixel_samples.direct_u,
        camera, samples_per_pixel,
    )
    return nothing
end

# ============================================================================
# BSDF Sampling Inner Function
# ============================================================================

"""Inner function for material evaluation - can use return statements.

Now uses pre-computed Sobol samples from pixel_samples (pbrt-v4 RaySamples style).
"""
@propagate_inbounds function evaluate_material_inner!(
    next_ray_queue,
    work::VPMaterialEvalWorkItem,
    materials,
    rgb2spec_table,
    max_depth::Int32,
    do_regularize::Bool,
    # Pre-computed Sobol samples (SOA layout)
    pixel_samples_indirect_uc,
    pixel_samples_indirect_u,
    pixel_samples_indirect_rr,
    # Camera for texture filtering (pbrt-v4 style)
    camera,
    samples_per_pixel::Int32
)
    # Check depth limit
    new_depth = work.depth + Int32(1)
    if new_depth >= max_depth
        return
    end

    # Use pre-computed Sobol samples for BSDF sampling (pbrt-v4 RaySamples.indirect)
    pixel_idx = work.pixel_index
    u = pixel_samples_indirect_u[pixel_idx]
    rng = pixel_samples_indirect_uc[pixel_idx]
    rr_sample = pixel_samples_indirect_rr[pixel_idx]

    # Apply regularization if enabled and we've had a non-specular bounce
    # (pbrt-v4: regularize && anyNonSpecularBounces)
    regularize = do_regularize && work.any_non_specular_bounces

    # Compute texture filter context with proper screen-space derivatives (pbrt-v4 style)
    tfc = compute_texture_filter_context(work, camera, samples_per_pixel)

    # Sample BSDF
    sample = sample_spectral_material(
        rgb2spec_table, materials, work.material_idx,
        work.wo, work.ns, tfc, work.lambda, u, rng, regularize
    )

    # Check if valid sample
    if sample.pdf > 0f0 && !is_black(sample.f)
        # Compute new throughput
        cos_theta = abs(dot(sample.wi, work.ns))
        new_beta = if sample.is_specular
            work.beta * sample.f
        else
            work.beta * sample.f * cos_theta / sample.pdf
        end

        # Update eta scale for refraction
        new_eta_scale = work.eta_scale * sample.eta_scale

        # Update MIS weights
        new_r_l = if sample.is_specular
            work.r_u  # No MIS for specular
        else
            work.r_u / sample.pdf
        end

        # Russian roulette
        should_continue, final_beta = russian_roulette_spectral(
            new_beta, new_depth, rr_sample
        )

        if should_continue
            # Determine medium for continuation ray using MediumInterfaceIdx
            # Following pbrt-v4: use ray direction relative to surface normal
            # to determine which medium the ray enters
            new_medium = if is_medium_transition(work.interface)
                # Surface defines a medium boundary - get medium based on ray direction
                # If wi · n > 0, ray goes "outside" the surface
                # If wi · n < 0, ray goes "inside" the surface
                get_medium_index(work.interface, sample.wi, work.n)
            else
                # No medium transition at this surface
                # Stay in current medium (reflection or transmission through regular material)
                work.current_medium
            end

            # Create continuation ray
            # Offset origin slightly to avoid self-intersection
            offset_dir = if dot(sample.wi, work.n) > 0f0
                work.n
            else
                -work.n
            end
            ray_origin = Point3f(work.pi + offset_dir * 0.0001f0)

            new_ray = Raycore.Ray(
                o = ray_origin,
                d = sample.wi,
                t_max = Inf32,
                time = 0f0
            )

            ray_item = VPRayWorkItem(
                new_ray,
                new_depth,
                work.lambda,
                work.pixel_index,
                final_beta,
                work.r_u,  # r_u unchanged
                new_r_l,
                work.pi,   # prev_intr_p
                work.ns,   # prev_intr_n
                new_eta_scale,
                sample.is_specular,
                work.any_non_specular_bounces || !sample.is_specular,
                new_medium
            )

            push!(next_ray_queue, ray_item)
        end
    end
    return
end

# ============================================================================
# BSDF Sampling and Path Continuation
# ============================================================================

@propagate_inbounds function vp_evaluate_materials_kernel!(
    work,
    next_ray_queue,
    materials,
    rgb2spec_table,
    max_depth::Int32,
    do_regularize::Bool,
    pixel_samples_indirect_uc, pixel_samples_indirect_u, pixel_samples_indirect_rr,
    camera, samples_per_pixel::Int32
)
    evaluate_material_inner!(
        next_ray_queue,
        work, materials, rgb2spec_table, max_depth,
        do_regularize,
        pixel_samples_indirect_uc, pixel_samples_indirect_u, pixel_samples_indirect_rr,
        camera, samples_per_pixel
    )
end

function vp_evaluate_materials!(state::VolPathState, materials, camera, samples_per_pixel::Int32, regularize::Bool = true)
    output_queue = next_ray_queue(state)
    pixel_samples = state.pixel_samples
    foreach(vp_evaluate_materials_kernel!,
        state.material_queue,
        output_queue,
        materials,
        state.rgb2spec_table,
        state.max_depth,
        regularize,
        pixel_samples.indirect_uc, pixel_samples.indirect_u, pixel_samples.indirect_rr,
        camera, samples_per_pixel,
    )
    return nothing
end

# ============================================================================
# Helper: Check for pure emissive material
# ============================================================================

"""
    is_pure_emissive_dispatch(materials, mat_idx)

Check if material is purely emissive (no BSDF).
"""
@propagate_inbounds function is_pure_emissive_dispatch(materials, mat_idx::SetKey)
    # Default: assume materials with emission also have BSDF
    # Override for EmissiveMaterial which has no BSDF
    return is_emissive(materials, mat_idx) &&
           !has_bsdf_dispatch(materials, mat_idx)
end

"""
    has_bsdf_dispatch(materials, mat_idx)

Check if material has a BSDF component.
"""
@propagate_inbounds function has_bsdf_dispatch(materials, mat_idx::SetKey)
    # Most materials have BSDF
    # EmissiveMaterial does not
    type_idx = mat_idx.type_idx
    # This would need proper dispatch based on material type
    # For now, return true (most materials have BSDF)
    return true
end
