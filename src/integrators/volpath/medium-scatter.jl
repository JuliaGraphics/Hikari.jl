# Medium scattering kernel for volumetric path tracing
# Handles real scattering events: direct lighting + phase function sampling

# ============================================================================
# Medium Direct Lighting Inner Function
# ============================================================================

"""Inner function for medium direct lighting - can use return statements."""
@propagate_inbounds function medium_direct_lighting_inner!(
    shadow_items, shadow_size,
    work::VPMediumScatterWorkItem,
    lights,
    rgb2spec_table,
    num_lights::Int32,
    max_queued::Int32
)
    # Skip if no lights
    num_lights < Int32(1) && return

    # Random numbers for light sampling
    u_light = rand(Point2f)
    light_select = rand(Float32)

    # Select light uniformly
    light_idx = floor_int32(light_select * Float32(num_lights)) + Int32(1)
    light_idx = min(light_idx, num_lights)

    # Sample the light
    light_sample = sample_light_from_tuple(
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
            light_pdf = light_sample.pdf / Float32(num_lights)  # Include light selection probability
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

            @inbounds begin
                new_idx = @atomic shadow_size[1] += Int32(1)
                if new_idx <= max_queued
                    shadow_items[new_idx] = shadow_item
                end
            end
        end
    end
    return
end

# ============================================================================
# Medium Scatter Kernel (Direct Lighting)
# ============================================================================

"""
    vp_medium_direct_lighting_kernel!(...)

Sample direct lighting at medium scattering events.
Creates shadow rays for each scatter point.
"""
@kernel inbounds=true function vp_medium_direct_lighting_kernel!(
    # Output
    shadow_items, shadow_size,
    # Input
    @Const(scatter_items), @Const(scatter_size),
    @Const(lights),
    @Const(media),
    @Const(rgb2spec_scale), @Const(rgb2spec_coeffs), @Const(rgb2spec_res::Int32),
    @Const(num_lights::Int32), @Const(max_queued::Int32)
)
    idx = @index(Global)

    rgb2spec_table = RGBToSpectrumTable(rgb2spec_res, rgb2spec_scale, rgb2spec_coeffs)

    @inbounds if idx <= max_queued
        current_size = scatter_size[1]
        if idx <= current_size
            work = scatter_items[idx]
            medium_direct_lighting_inner!(
                shadow_items, shadow_size,
                work, lights, rgb2spec_table, num_lights, max_queued
            )
        end
    end
end

# ============================================================================
# Medium Scatter Inner Function
# ============================================================================

"""Inner function for medium scatter - can use return statements."""
@propagate_inbounds function medium_scatter_inner!(
    ray_items, ray_size,
    work::VPMediumScatterWorkItem,
    max_depth::Int32,
    max_queued::Int32
)
    # Check depth limit
    new_depth = work.depth + Int32(1)
    if new_depth >= max_depth
        return
    end

    # Sample phase function
    u = rand(Point2f)
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

        @inbounds begin
            new_idx = @atomic ray_size[1] += Int32(1)
            if new_idx <= max_queued
                ray_items[new_idx] = ray_item
            end
        end
    end
    return
end

# ============================================================================
# Medium Scatter Kernel (Phase Function Sampling)
# ============================================================================

"""
    vp_medium_scatter_kernel!(...)

Sample phase function at scatter points to generate continuation rays.
"""
@kernel inbounds=true function vp_medium_scatter_kernel!(
    # Output
    ray_items, ray_size,
    # Input
    @Const(scatter_items), @Const(scatter_size),
    @Const(max_depth::Int32), @Const(max_queued::Int32)
)
    idx = @index(Global)

    @inbounds if idx <= max_queued
        current_size = scatter_size[1]
        if idx <= current_size
            work = scatter_items[idx]
            medium_scatter_inner!(ray_items, ray_size, work, max_depth, max_queued)
        end
    end
end

# ============================================================================
# High-Level Functions
# ============================================================================

"""
    vp_sample_medium_direct_lighting!(backend, state, scene, media)

Sample direct lighting at all medium scatter points.
"""
function vp_sample_medium_direct_lighting!(
    backend,
    state::VolPathState,
    lights,
    media
)
    n = queue_size(state.medium_scatter_queue)
    n == 0 && return nothing

    num_lights = count_lights(lights)

    kernel! = vp_medium_direct_lighting_kernel!(backend)
    kernel!(
        state.shadow_queue.items, state.shadow_queue.size,
        state.medium_scatter_queue.items, state.medium_scatter_queue.size,
        lights,
        media,
        state.rgb2spec_table.scale, state.rgb2spec_table.coeffs, state.rgb2spec_table.res,
        num_lights, state.shadow_queue.capacity;
        ndrange=Int(state.medium_scatter_queue.capacity)  # Fixed ndrange to avoid OpenCL recompilation
    )

    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    vp_sample_medium_scatter!(backend, state)

Sample phase functions at scatter points to generate continuation rays.
"""
function vp_sample_medium_scatter!(
    backend,
    state::VolPathState
)
    n = queue_size(state.medium_scatter_queue)
    n == 0 && return nothing

    output_queue = next_ray_queue(state)

    kernel! = vp_medium_scatter_kernel!(backend)
    kernel!(
        output_queue.items, output_queue.size,
        state.medium_scatter_queue.items, state.medium_scatter_queue.size,
        state.max_depth, output_queue.capacity;
        ndrange=Int(state.medium_scatter_queue.capacity)  # Fixed ndrange to avoid OpenCL recompilation
    )

    KernelAbstractions.synchronize(backend)
    return nothing
end
