# Medium scattering kernel for volumetric path tracing
# Handles real scattering events: direct lighting + phase function sampling

# ============================================================================
# Medium Direct Lighting Inner Function
# ============================================================================

# Light-sampling roulette: retain all early NEE estimates and one quarter at
# high scattering orders. The participation probability belongs in BOTH MIS
# densities (including the phase continuation), not just a brightness multiplier.
@inline medium_light_probability(depth::Int32) = depth < Int32(8) ? 1f0 : 0.25f0

"""Inner function for medium direct lighting - can use return statements.

Uses power-weighted light sampling via alias table for better importance sampling
in scenes with lights of varying intensities (pbrt-v4's PowerLightSampler approach).

Sobol samples are generated inline rather than read from a pre-populated
per-pixel buffer; see `surface_direct_lighting_inner!` in surface-eval.jl
for the rationale.
"""
@propagate_inbounds function medium_direct_lighting_inner!(
    shadow_queue,
    work::VPMediumScatterWorkItem,
    lights,
    media,
    rgb2spec_table,
    bvh_nodes,
    infinite_light_indices,
    num_infinite_lights::Int32,
    num_bvh_lights::Int32,
    num_lights::Int32,
    sobol_rng,           # SobolRNG — samples generated on demand inline
    sample_idx::Int32,
)
    # Skip if no lights
    num_lights < Int32(1) && return

    # Inline Sobol sample generation (matches dimension layout used by the
    # surface path's `surface_direct_lighting_inner!`).
    px, py, sample_idx = path_pixel_sample(sobol_rng, work.pixel_index, sample_idx)
    base_dim = Int32(6) + Int32(7) * work.depth
    light_select = sample_1d(sobol_rng, px, py, sample_idx, base_dim + Int32(1))
    participation = medium_light_probability(work.depth)
    light_select >= participation && return
    light_select /= participation
    u_light_x, u_light_y = sample_2d(sobol_rng, px, py, sample_idx, base_dim + Int32(3))
    u_light = Point2f(u_light_x, u_light_y)

    # Select light using BVH light sampler (spatially-aware importance sampling)
    # Medium scattering has no surface normal → pass Vec3f(0f0)
    light_idx, light_pmf = bvh_sample_light(
        bvh_nodes, infinite_light_indices,
        num_infinite_lights, num_bvh_lights,
        work.p, Vec3f(0f0), light_select
    )

    # Validate index
    if light_idx < Int32(1) || light_idx > num_lights || light_pmf <= 0f0
        return
    end

    # Sample the light (works with both Tuple and StaticMultiTypeSet)
    light_sample = sample_light_spectral(
        rgb2spec_table, lights, light_idx, work.p, work.lambda, u_light
    )

    # Under a refracting surface the sky is sampled through Snell's window (see
    # `snells_window`): the shadow ray bends at the surface, and finds the sky's
    # radiance in the direction it leaves in.
    η = Raycore.with_index(refracting, media, work.medium_idx)
    if η > 0f0 && light_sample.pdf > 0f0 && !light_sample.is_delta &&
       Raycore.with_index(is_infinite_light, lights, flat_to_light_index(lights, light_idx))
        ω_in, jacobian = snells_window(light_sample.wi, η)
        jacobian > 0f0 || return
        phase_val = hg_p(work.g, dot(-work.wo, ω_in))
        phase_val > 0f0 || return
        shadow_ray = Raycore.Ray(o = work.p, d = ω_in, t_max = 1f6, time = work.time)
        push!(shadow_queue, VPShadowRayWorkItem(
            shadow_ray, 1f6, work.lambda,
            work.beta * phase_val,                       # times the sky, found at the end
            work.r_u * phase_val,                        # MIS with the paths into the window
            # The sky all the infinite lights make, over the density of their mixture.
            work.r_u * (compute_env_light_pdf(lights, light_sample.wi) * light_pmf * jacobian * participation),
            work.pixel_index, work.medium_idx, true))
        return
    end

    if light_sample.pdf > 0f0 && !is_black(light_sample.Li)
        # Evaluate phase function for light direction
        # Between the ways the light goes before and after (see `hg_p`): `wo` points back.
        cos_θ = dot(-work.wo, light_sample.wi)
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
            light_pdf = light_sample.pdf * light_pmf * participation
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
                work.medium_idx,  # Shadow ray travels through same medium
                false
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
    media,
    rgb2spec_table,
    bvh_nodes, infinite_light_indices,
    num_infinite_lights::Int32, num_bvh_lights::Int32,
    num_lights::Int32,
    sobol_rng, sample_idx_ref,
)
    sample_idx = @inbounds sample_idx_ref[Int32(1)]
    medium_direct_lighting_inner!(
        shadow_queue,
        work, lights, media, rgb2spec_table,
        bvh_nodes, infinite_light_indices,
        num_infinite_lights, num_bvh_lights,
        num_lights,
        sobol_rng, sample_idx,
    )
end

# ============================================================================
# Medium Scatter Inner Function
# ============================================================================

"""Inner function for medium scatter - can use return statements.

Sobol samples are generated inline; see `surface_direct_lighting_inner!`.
"""
@propagate_inbounds function medium_scatter_inner!(
    ray_queue,
    work::VPMediumScatterWorkItem,
    sobol_rng,
    sample_idx::Int32,
    rr_depth::Int32 = Int32(1),
    media = nothing,
)
    # A real scattering is only queued below the depth limit (`sample_T_maj_loop!`).
    new_depth = work.depth + Int32(1)

    # Inline Sobol sample generation for phase function direction (dim+6 = 2D).
    px, py, sample_idx = path_pixel_sample(sobol_rng, work.pixel_index, sample_idx)
    base_dim = Int32(6) + Int32(7) * work.depth
    u_x, u_y = sample_2d(sobol_rng, px, py, sample_idx, base_dim + Int32(6))
    u = Point2f(u_x, u_y)
    wi, phase_pdf = sample_hg(work.g, work.wo, u)

    if phase_pdf > 0f0
        # Update throughput
        # For phase functions: f = phase, pdf = phase, so ratio = 1
        # But we need f/pdf for path throughput
        # Use the same throughput-based roulette as surface paths. Surviving
        # paths are reweighted, preserving the expectation at high depth limits.
        rr_sample = sample_1d(sobol_rng, px, py, sample_idx, base_dim + Int32(5))
        keep, new_beta = russian_roulette_spectral(work.beta, work.r_u, work.eta_scale,
                                                   new_depth, rr_sample, rr_depth)
        keep || return

        # Update MIS weights
        new_r_u = work.r_u
        new_r_l = work.r_u * (medium_light_probability(work.depth) / phase_pdf)

        # Create continuation ray
        new_ray = Raycore.Ray(
            o = work.p,
            d = wi,
            t_max = media === nothing ? Inf32 :
                Raycore.with_index(medium_trace_distance, media, work.medium_idx, media, work.p),
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
            work.eta_scale,   # eta_scale (carry through from path state)
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
    sobol_rng, sample_idx_ref,
    rr_depth::Int32,
    media,
)
    sample_idx = @inbounds sample_idx_ref[Int32(1)]
    medium_scatter_inner!(ray_queue, work, sobol_rng, sample_idx, rr_depth, media)
end
