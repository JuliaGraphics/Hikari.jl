# Delta tracking kernel for volumetric path tracing
# Implements null-scattering method from pbrt-v4
#
# Key architecture (following pbrt-v4):
# 1. Rays are traced FIRST to find intersection distance (t_max)
# 2. Rays in medium are routed to medium_sample_queue with t_max
# 3. This kernel runs delta tracking up to t_max
# 4. If ray survives to t_max, it processes the stored surface hit

# ============================================================================
# Helper: Add contribution to pixel
# ============================================================================

"""Add spectral contribution to film pixel"""
@propagate_inbounds function add_to_pixel!(
    pixel_L::AbstractVector{Float32},
    pixel_index::Int32,
    L::SpectralRadiance,
    lambda::Wavelengths
)
    base = (pixel_index - Int32(1)) * Int32(4)
    @inbounds for i in 1:4
        # Atomic add for thread safety
        Atomix.@atomic pixel_L[base + Int32(i)] += L[i]
    end
end

# ============================================================================
# Delta Tracking Kernel (processes VPMediumSampleWorkItem)
# ============================================================================

"""
    vp_sample_medium_kernel!(...)

Process rays in medium using delta tracking with known t_max from intersection.
This follows pbrt-v4's architecture where intersection happens FIRST.

For each ray:
1. Run delta tracking up to t_max
2. If absorption: terminate path
3. If real scatter: push to medium_scatter_queue
4. If escape (reach t_max): process stored surface hit or escaped ray
"""
@kernel inbounds=true function vp_sample_medium_kernel!(
    # Output queues
    scatter_items, scatter_size,                       # Real scatter events -> phase sampling
    hit_surface_items, hit_surface_size,              # Surface hits (ray survived medium)
    escaped_items, escaped_size,                       # Escaped rays (no surface hit)
    pixel_L,                                           # Film (for medium emission)
    # Input
    @Const(medium_sample_items), @Const(medium_sample_size),
    @Const(media),                                    # Media tuple
    @Const(rgb2spec_scale), @Const(rgb2spec_coeffs), @Const(rgb2spec_res::Int32),
    @Const(max_depth::Int32), @Const(max_queued::Int32)
)
    idx = @index(Global)

    # Reconstruct RGB to spectrum table
    rgb2spec_table = RGBToSpectrumTable(rgb2spec_res, rgb2spec_scale, rgb2spec_coeffs)

    @inbounds if idx <= max_queued
        current_size = medium_sample_size[1]
        if idx <= current_size
            work = medium_sample_items[idx]

            # Run delta tracking for this ray
            sample_medium_interaction!(
                scatter_items, scatter_size,
                hit_surface_items, hit_surface_size,
                escaped_items, escaped_size,
                pixel_L,
                work, media, rgb2spec_table, max_depth, max_queued
            )
        end
    end
end

"""
    sample_medium_interaction!(...)

Inner function for medium sampling with bounded t_max.
Implements delta tracking following pbrt-v4's SampleMediumInteraction.
"""
@propagate_inbounds function sample_medium_interaction!(
    # Output queues
    scatter_items, scatter_size,
    hit_surface_items, hit_surface_size,
    escaped_items, escaped_size,
    pixel_L,
    # Input
    work::VPMediumSampleWorkItem,
    media,
    rgb2spec_table,
    max_depth::Int32,
    max_queued::Int32
)
    medium_type_idx = work.medium_idx.medium_type
    t_max = work.t_max

    # Get majorant for this ray segment
    majorant = get_majorant_dispatch(
        rgb2spec_table, media, medium_type_idx,
        work.ray, 0f0, t_max, work.lambda
    )

    # Delta tracking state
    t = majorant.t_min
    beta = work.beta
    r_u = work.r_u
    r_l = work.r_l
    scattered = false

    # Tracking loop (bounded iterations for GPU)
    max_iterations = Int32(1000)
    for _ in 1:max_iterations
        # Sample exponential distance
        u = rand(Float32)
        σ_maj_0 = majorant.σ_maj[1]  # Use first wavelength for sampling

        if σ_maj_0 < 1f-10
            # No extinction - ray passes through entire segment
            t = majorant.t_max
            break
        end

        # Sample distance: t' = t - ln(1-u) / σ_maj
        dt = -log(max(1f-10, 1f0 - u)) / σ_maj_0
        t_sample = t + dt

        if t_sample >= majorant.t_max
            # Reached end of medium segment without interaction
            # Update throughput for remaining distance
            # Following pbrt-v4: apply T_maj and normalize by T_maj[0]
            dt_remain = majorant.t_max - t
            T_maj = exp(-dt_remain * majorant.σ_maj)
            T_maj_0 = T_maj[1]  # First wavelength (index 1 in Julia)
            if T_maj_0 > 1f-10
                beta = beta * T_maj / T_maj_0
                r_u = r_u * T_maj / T_maj_0
                r_l = r_l * T_maj / T_maj_0
            end
            t = majorant.t_max
            break
        end

        # Sample medium properties at interaction point
        p = Point3f(work.ray.o + work.ray.d * t_sample)
        mp = sample_point_dispatch(rgb2spec_table, media, medium_type_idx, p, work.lambda)

        # Compute transmittance for this segment (from t to t_sample)
        # Following pbrt-v4: T_maj is computed fresh for each segment
        T_maj = exp(-dt * majorant.σ_maj)

        # Add emission if present (always, scaled by sigma_a/sigma_maj)
        if !is_black(mp.Le) && work.depth < max_depth
            pr = σ_maj_0 * T_maj[1]
            if pr > 1f-10
                r_e = r_u * majorant.σ_maj * T_maj / pr
                if !is_black(r_e)
                    Le_contrib = beta * mp.σ_a * T_maj * mp.Le / (pr * average(r_e))
                    add_to_pixel!(pixel_L, work.pixel_index, Le_contrib, work.lambda)
                end
            end
        end

        # Compute event probabilities
        p_absorb = mp.σ_a[1] / σ_maj_0
        p_scatter = mp.σ_s[1] / σ_maj_0
        p_null = max(0f0, 1f0 - p_absorb - p_scatter)

        # Sample event type
        u_event = rand(Float32)

        if u_event < p_absorb
            # === ABSORPTION ===
            # Path terminates
            beta = SpectralRadiance(0f0)
            return  # Done

        elseif u_event < p_absorb + p_scatter
            # === REAL SCATTERING ===
            if work.depth >= max_depth
                return  # Max depth reached
            end

            # Update beta and r_u for scattering
            # Following pbrt-v4: beta *= T_maj * sigma_s / pdf
            pdf = T_maj[1] * mp.σ_s[1]
            if pdf > 1f-10
                beta = beta * T_maj * mp.σ_s / pdf
                r_u = r_u * T_maj * mp.σ_s / pdf
            end

            # Push to scatter queue for phase function sampling
            scatter_item = VPMediumScatterWorkItem(
                p,
                -work.ray.d,  # wo points back toward camera
                work.ray.time,
                work.lambda,
                work.pixel_index,
                beta,
                r_u,
                work.depth,
                work.medium_idx,
                mp.g
            )

            @inbounds begin
                new_idx = @atomic scatter_size[1] += Int32(1)
                if new_idx <= max_queued
                    scatter_items[new_idx] = scatter_item
                end
            end
            scattered = true
            return

        else
            # === NULL SCATTERING ===
            # Continue tracking with updated throughput
            # Following pbrt-v4: beta *= T_maj * sigma_n / pdf
            σ_n = majorant.σ_maj - mp.σ_a - mp.σ_s
            # Clamp negative values element-wise
            σ_n = SpectralRadiance(max(σ_n[1], 0f0), max(σ_n[2], 0f0), max(σ_n[3], 0f0), max(σ_n[4], 0f0))

            pdf = T_maj[1] * σ_n[1]
            if pdf > 1f-10
                beta = beta * T_maj * σ_n / pdf
                r_u = r_u * T_maj * σ_n / pdf
                r_l = r_l * T_maj * majorant.σ_maj / pdf
            else
                beta = SpectralRadiance(0f0)
            end

            t = t_sample

            # Check if throughput is too low
            if is_black(beta) || is_black(r_u)
                return
            end
        end
    end

    # Ray survived to t_max - process what's at the end
    if is_black(beta) || is_black(r_u) || work.depth >= max_depth
        return
    end

    # Finalize throughput
    # (T_maj already applied in the loop when we reach t_max)

    if !work.has_surface_hit
        # Ray escaped scene (t_max was Infinity)
        escaped_item = VPEscapedRayWorkItem(
            work.ray.d,
            work.lambda,
            work.pixel_index,
            beta,
            r_u,
            r_l,
            work.depth,
            work.specular_bounce,
            work.prev_intr_p,
            work.prev_intr_n
        )
        @inbounds begin
            new_idx = @atomic escaped_size[1] += Int32(1)
            if new_idx <= max_queued
                escaped_items[new_idx] = escaped_item
            end
        end
    else
        # Ray reached surface - push to hit_surface_queue for material eval
        hit_item = VPHitSurfaceWorkItem(
            work.ray,
            work.hit_pi,
            work.hit_n,
            work.hit_ns,
            work.hit_uv,
            work.hit_material_idx,
            work.lambda,
            work.pixel_index,
            beta,      # Updated throughput after medium traversal
            r_u,       # Updated r_u
            r_l,       # Updated r_l
            work.depth,
            work.eta_scale,
            work.specular_bounce,
            work.any_non_specular_bounces,
            work.prev_intr_p,
            work.prev_intr_n,
            work.medium_idx,
            work.t_max
        )
        @inbounds begin
            new_idx = @atomic hit_surface_size[1] += Int32(1)
            if new_idx <= max_queued
                hit_surface_items[new_idx] = hit_item
            end
        end
    end
    return
end

# ============================================================================
# High-Level Medium Sampling Function
# ============================================================================

"""
    vp_sample_medium_interaction!(backend, state, media)

Run delta tracking on rays in medium_sample_queue.
These rays already have t_max from intersection testing.
"""
function vp_sample_medium_interaction!(
    backend,
    state::VolPathState,
    media
)
    n = queue_size(state.medium_sample_queue)
    n == 0 && return nothing

    kernel! = vp_sample_medium_kernel!(backend)
    kernel!(
        state.medium_scatter_queue.items, state.medium_scatter_queue.size,
        state.hit_surface_queue.items, state.hit_surface_queue.size,
        state.escaped_queue.items, state.escaped_queue.size,
        state.pixel_L,
        state.medium_sample_queue.items, state.medium_sample_queue.size,
        media,
        state.rgb2spec_table.scale, state.rgb2spec_table.coeffs, state.rgb2spec_table.res,
        state.max_depth, state.medium_sample_queue.capacity;
        ndrange=Int(state.medium_sample_queue.capacity)  # Fixed ndrange to avoid OpenCL recompilation
    )

    KernelAbstractions.synchronize(backend)
    return nothing
end
