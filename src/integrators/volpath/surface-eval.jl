# Surface material evaluation for VolPath
# Handles BSDF sampling, direct lighting, and path continuation at surface hits
#
# NOTE: `materials` is a StaticMultiTypeVec containing both materials and textures.
# Use eval_tex(materials, field, uv) to sample textures via TextureRef.

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
        # Get emission
        Le = get_emission_spectral_dispatch(
            rgb2spec_table, materials, material_idx,
            wo, work.n, work.uv, work.lambda
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
    pixel_samples_direct_u
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

    # Sample the light (works with both Tuple and StaticMultiTypeVec)
    light_sample = sample_light_spectral(
        rgb2spec_table, lights, light_idx, work.pi, work.lambda, u_light
    )

    if light_sample.pdf > 0f0 && !is_black(light_sample.Li)
        # Evaluate BSDF for light direction
        bsdf_f, bsdf_pdf = evaluate_spectral_material(
            rgb2spec_table, materials, work.material_idx,
            work.wo, light_sample.wi, work.ns, work.uv, work.lambda
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
    pixel_samples_direct_uc, pixel_samples_direct_u
)
    surface_direct_lighting_inner!(
        shadow_queue,
        work, materials, lights, rgb2spec_table,
        light_sampler_p, light_sampler_q, light_sampler_alias,
        num_lights,
        pixel_samples_direct_uc, pixel_samples_direct_u
    )
end

function vp_sample_surface_direct_lighting!(state::VolPathState, materials, lights)
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
    pixel_samples_indirect_rr
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

    # Sample BSDF
    sample = sample_spectral_material(
        rgb2spec_table, materials, work.material_idx,
        work.wo, work.ns, work.uv, work.lambda, u, rng, regularize
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
            # Determine medium for continuation ray
            # Following pbrt-v4: use ray direction relative to surface normal
            # to determine which medium the ray enters
            new_medium = if has_medium_interface_dispatch(materials, work.material_idx)
                # Material has MediumInterface - get medium based on ray direction
                # If wi · n > 0, ray goes "outside" the surface
                # If wi · n < 0, ray goes "inside" the surface
                get_medium_index_for_direction_dispatch(materials, work.material_idx, sample.wi, work.n)
            else
                # No MediumInterface on this material
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
    pixel_samples_indirect_uc, pixel_samples_indirect_u, pixel_samples_indirect_rr
)
    evaluate_material_inner!(
        next_ray_queue,
        work, materials, rgb2spec_table, max_depth,
        do_regularize,
        pixel_samples_indirect_uc, pixel_samples_indirect_u, pixel_samples_indirect_rr
    )
end

function vp_evaluate_materials!(state::VolPathState, materials, regularize::Bool = true)
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
@propagate_inbounds function is_pure_emissive_dispatch(materials, mat_idx::MaterialIndex)
    # Default: assume materials with emission also have BSDF
    # Override for EmissiveMaterial which has no BSDF
    return is_emissive(materials, mat_idx) &&
           !has_bsdf_dispatch(materials, mat_idx)
end

"""
    has_bsdf_dispatch(materials, mat_idx)

Check if material has a BSDF component.
"""
@propagate_inbounds function has_bsdf_dispatch(materials, mat_idx::MaterialIndex)
    # Most materials have BSDF
    # EmissiveMaterial does not
    type_idx = mat_idx.type_idx
    # This would need proper dispatch based on material type
    # For now, return true (most materials have BSDF)
    return true
end
