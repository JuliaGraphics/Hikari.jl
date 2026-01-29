# Medium scattering kernel for volumetric path tracing
# Handles real scattering events: direct lighting + phase function sampling

# ============================================================================
# Medium Direct Lighting Inner Function
# ============================================================================

"""Inner function for medium direct lighting - can use return statements.

Uses power-weighted light sampling via alias table for better importance sampling
in scenes with lights of varying intensities (pbrt-v4's PowerLightSampler approach).

Now uses pre-computed Sobol samples from pixel_samples (pbrt-v4 RaySamples style).
"""
@propagate_inbounds function medium_direct_lighting_inner!(
    shadow_queue,
    work::VPMediumScatterWorkItem,
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
    light_idx, light_pmf = sample_light_sampler(
        light_sampler_p, light_sampler_q, light_sampler_alias, light_select
    )

    # Validate index
    if light_idx < Int32(1) || light_idx > num_lights || light_pmf <= 0f0
        return
    end

    # Sample the light (works with both Tuple and StaticMultiTypeSet)
    light_sample = sample_light_spectral(
        rgb2spec_table, lights, light_idx, work.p, work.lambda, u_light
    )

    if light_sample.pdf > 0f0 && !is_black(light_sample.Li)
        # Evaluate phase function for light direction
        cos_θ = dot(work.wo, light_sample.wi)
        phase_val = hg_p(work.g, cos_θ)

        if phase_val > 0f0
            # Compute direct lighting contribution following pbrt-v4 (media.cpp lines 287-300):
            # Ld = beta * phase * Li
            # NO PDF division here - that happens at shadow ray resolution via MIS weights
            Ld = work.beta * phase_val * light_sample.Li

            # MIS weights following pbrt-v4 (media.cpp lines 293-297):
            # lightPDF = ls->pdf * sampledLight->p (light PDF including selection probability)
            # phasePDF = 0 for delta lights, else phase->PDF(wo, wi)
            # r_u = w.r_u * phasePDF
            # r_l = w.r_u * lightPDF
            # Now using power-weighted PMF instead of uniform 1/num_lights
            light_pdf = light_sample.pdf * light_pmf  # Include light selection probability
            phase_pdf = if light_sample.is_delta
                0f0  # Delta lights have no MIS with phase sampling
            else
                phase_val  # HG PDF equals value for importance sampling
            end
            r_u = work.r_u * phase_pdf
            r_l = work.r_u * light_pdf

            # Shadow ray from scatter point to light
            shadow_origin = work.p
            shadow_dir = light_sample.wi
            t_max = if light_sample.is_delta
                # Point/directional light - go to light position
                norm(light_sample.p_light - work.p) - 0.001f0
            else
                # Area light - use large distance
                1f6
            end

            shadow_ray = Raycore.Ray(
                o = shadow_origin,
                d = shadow_dir,
                t_max = t_max,
                time = work.time
            )

            shadow_item = VPShadowRayWorkItem(
                shadow_ray,
                t_max,
                work.lambda,
                Ld,
                r_u,
                r_l,
                work.pixel_index,
                work.medium_idx  # Shadow ray travels through same medium
            )

            push!(shadow_queue, shadow_item)
        end
    end
    return
end

# ============================================================================
# Medium Scatter Kernel (Direct Lighting)
# ============================================================================

@propagate_inbounds function vp_medium_direct_lighting_kernel!(
    work,
    shadow_queue,
    lights,
    rgb2spec_table,
    light_sampler_p, light_sampler_q, light_sampler_alias,
    num_lights::Int32,
    pixel_samples_direct_uc, pixel_samples_direct_u
)
    medium_direct_lighting_inner!(
        shadow_queue,
        work, lights, rgb2spec_table,
        light_sampler_p, light_sampler_q, light_sampler_alias,
        num_lights,
        pixel_samples_direct_uc, pixel_samples_direct_u
    )
end

# ============================================================================
# Medium Scatter Inner Function
# ============================================================================

"""Inner function for medium scatter - can use return statements.

Now uses pre-computed Sobol samples from pixel_samples (pbrt-v4 RaySamples style).
"""
@propagate_inbounds function medium_scatter_inner!(
    ray_queue,
    work::VPMediumScatterWorkItem,
    max_depth::Int32,
    # Pre-computed Sobol samples (SOA layout)
    pixel_samples_indirect_u
)
    # Check depth limit
    new_depth = work.depth + Int32(1)
    if new_depth >= max_depth
        return
    end

    # Use pre-computed Sobol samples for phase function sampling (pbrt-v4 RaySamples.indirect)
    pixel_idx = work.pixel_index
    u = pixel_samples_indirect_u[pixel_idx]
    wi, phase_pdf = sample_hg(work.g, work.wo, u)

    if phase_pdf > 0f0
        # Update throughput
        # For phase functions: f = phase, pdf = phase, so ratio = 1
        # But we need f/pdf for path throughput
        new_beta = work.beta  # phase/pdf = 1 for importance-sampled HG

        # Update MIS weights
        new_r_u = work.r_u
        new_r_l = work.r_u / phase_pdf

        # Create continuation ray
        new_ray = Raycore.Ray(
            o = work.p,
            d = wi,
            t_max = Inf32,
            time = work.time
        )

        ray_item = VPRayWorkItem(
            new_ray,
            new_depth,
            work.lambda,
            work.pixel_index,
            new_beta,
            new_r_u,
            new_r_l,
            work.p,           # prev_intr_p
            work.wo,          # prev_intr_n (use wo as pseudo-normal for MIS)
            1f0,              # eta_scale (no refraction in medium)
            false,            # specular_bounce
            true,             # any_non_specular_bounces
            work.medium_idx   # Stay in same medium
        )

        push!(ray_queue, ray_item)
    end
    return
end

# ============================================================================
# Medium Scatter Kernel (Phase Function Sampling)
# ============================================================================

@propagate_inbounds function vp_medium_scatter_kernel!(
    work,
    ray_queue,
    max_depth::Int32,
    pixel_samples_indirect_u
)
    medium_scatter_inner!(ray_queue, work, max_depth, pixel_samples_indirect_u)
end

# ============================================================================
# High-Level Functions
# ============================================================================

function vp_sample_medium_direct_lighting!(state::VolPathState, lights)
    pixel_samples = state.pixel_samples
    foreach(vp_medium_direct_lighting_kernel!,
        state.medium_scatter_queue,
        state.shadow_queue,
        lights,
        state.rgb2spec_table,
        state.light_sampler_p, state.light_sampler_q, state.light_sampler_alias,
        state.num_lights,
        pixel_samples.direct_uc, pixel_samples.direct_u,
    )
    return nothing
end

function vp_sample_medium_scatter!(state::VolPathState)
    output_queue = next_ray_queue(state)
    pixel_samples = state.pixel_samples
    foreach(vp_medium_scatter_kernel!,
        state.medium_scatter_queue,
        output_queue,
        state.max_depth,
        pixel_samples.indirect_u,
    )
    return nothing
end
