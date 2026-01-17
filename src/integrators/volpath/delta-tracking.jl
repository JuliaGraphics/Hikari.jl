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
Implements delta tracking following pbrt-v4's SampleT_maj pattern.

Now uses DDA majorant iterator for GridMedium to get per-voxel majorant bounds,
significantly reducing null scattering events in sparse heterogeneous media.
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

    # Create majorant iterator (DDA for GridMedium, single-segment for HomogeneousMedium)
    iter = create_majorant_iterator_dispatch(
        rgb2spec_table, media, medium_type_idx,
        work.ray, t_max, work.lambda
    )

    # Delta tracking state
    beta = work.beta
    r_u = work.r_u
    r_l = work.r_l

    # Accumulated transmittance across segments (reset after each interaction)
    T_maj_accum = SpectralRadiance(1f0)

    # Dispatch to type-specific iteration
    result = sample_T_maj_loop!(
        iter, T_maj_accum, beta, r_u, r_l,
        scatter_items, scatter_size,
        pixel_L,
        work, media, medium_type_idx, rgb2spec_table, max_depth, max_queued
    )

    # Unpack result
    beta = result.beta
    r_u = result.r_u
    r_l = result.r_l
    done = result.done

    # If we terminated early (absorption or scatter), we're done
    if done
        return
    end

    # Ray survived to t_max - process what's at the end
    if is_black(beta) || is_black(r_u) || work.depth >= max_depth
        return
    end

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
            beta,
            r_u,
            r_l,
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

# Result type for sample_T_maj_loop!
struct SampleTMajResult
    beta::SpectralRadiance
    r_u::SpectralRadiance
    r_l::SpectralRadiance
    done::Bool  # true if terminated (absorption/scatter), false if reached end
end

"""
    sample_T_maj_loop!(iter::HomogeneousMajorantIterator, ...) -> SampleTMajResult

Delta tracking loop for homogeneous medium (single segment).
"""
@propagate_inbounds function sample_T_maj_loop!(
    iter::HomogeneousMajorantIterator,
    T_maj_accum::SpectralRadiance,
    beta::SpectralRadiance,
    r_u::SpectralRadiance,
    r_l::SpectralRadiance,
    scatter_items, scatter_size,
    pixel_L,
    work::VPMediumSampleWorkItem,
    media,
    medium_type_idx::Int32,
    rgb2spec_table,
    max_depth::Int32,
    max_queued::Int32
)
    # Get the single segment
    seg_result = homogeneous_next(iter)
    if seg_result === nothing
        return SampleTMajResult(beta, r_u, r_l, false)
    end
    seg, _ = seg_result

    # Sample within segment
    return sample_segment!(
        seg, T_maj_accum, beta, r_u, r_l,
        scatter_items, scatter_size,
        pixel_L,
        work, media, medium_type_idx, rgb2spec_table, max_depth, max_queued
    )
end

"""
    sample_T_maj_loop!(iter::DDAMajorantIterator, ...) -> SampleTMajResult

Delta tracking loop for heterogeneous medium using DDA majorant iterator.
Iterates over voxel segments with per-voxel majorant bounds.
"""
@propagate_inbounds function sample_T_maj_loop!(
    iter::DDAMajorantIterator,
    T_maj_accum::SpectralRadiance,
    beta::SpectralRadiance,
    r_u::SpectralRadiance,
    r_l::SpectralRadiance,
    scatter_items, scatter_size,
    pixel_L,
    work::VPMediumSampleWorkItem,
    media,
    medium_type_idx::Int32,
    rgb2spec_table,
    max_depth::Int32,
    max_queued::Int32
)
    # Iterate over DDA segments (bounded for GPU)
    max_segments = Int32(256)  # Reasonable upper bound for majorant grid traversal

    current_iter = iter
    for _ in 1:max_segments
        seg_result = dda_next(current_iter)
        if seg_result === nothing
            # No more segments - ray survived
            break
        end
        seg, new_iter = seg_result
        current_iter = new_iter

        # Sample within this segment
        result = sample_segment!(
            seg, T_maj_accum, beta, r_u, r_l,
            scatter_items, scatter_size,
            pixel_L,
            work, media, medium_type_idx, rgb2spec_table, max_depth, max_queued
        )

        if result.done
            # Terminated (absorption or scatter)
            return result
        end

        # Update state for next segment
        beta = result.beta
        r_u = result.r_u
        r_l = result.r_l
        # T_maj_accum is reset to 1.0 inside sample_segment! after interactions
        T_maj_accum = SpectralRadiance(1f0)
    end

    return SampleTMajResult(beta, r_u, r_l, false)
end

"""
    sample_segment!(seg, T_maj_accum, beta, r_u, r_l, ...) -> SampleTMajResult

Sample interactions within a single majorant segment.
Following pbrt-v4's inner loop of SampleT_maj.
"""
@propagate_inbounds function sample_segment!(
    seg::RayMajorantSegment,
    T_maj_accum::SpectralRadiance,
    beta::SpectralRadiance,
    r_u::SpectralRadiance,
    r_l::SpectralRadiance,
    scatter_items, scatter_size,
    pixel_L,
    work::VPMediumSampleWorkItem,
    media,
    medium_type_idx::Int32,
    rgb2spec_table,
    max_depth::Int32,
    max_queued::Int32
)
    σ_maj = seg.σ_maj
    σ_maj_0 = σ_maj[1]

    # Handle zero-majorant segment (empty voxel)
    if σ_maj_0 < 1f-10
        # Just apply transmittance for this segment (which is 1.0 for zero extinction)
        # No need to sample - ray passes through
        return SampleTMajResult(beta, r_u, r_l, false)
    end

    t = seg.t_min
    t_max_seg = seg.t_max

    # Inner sampling loop (bounded iterations)
    max_samples = Int32(100)
    for _ in 1:max_samples
        # Sample exponential distance
        u = rand(Float32)
        dt = -log(max(1f-10, 1f0 - u)) / σ_maj_0
        t_sample = t + dt

        if t_sample >= t_max_seg
            # Passed end of segment without interaction
            # Apply transmittance for remaining distance
            dt_remain = t_max_seg - t
            T_maj = exp(-dt_remain * σ_maj)
            T_maj_0 = T_maj[1]
            if T_maj_0 > 1f-10
                beta = beta * T_maj / T_maj_0
                r_u = r_u * T_maj / T_maj_0
                r_l = r_l * T_maj / T_maj_0
            end
            # Continue to next segment
            return SampleTMajResult(beta, r_u, r_l, false)
        end

        # Compute transmittance for this step
        T_maj = exp(-dt * σ_maj)

        # Sample medium properties at interaction point
        p = Point3f(work.ray.o + work.ray.d * t_sample)
        mp = sample_point_dispatch(rgb2spec_table, media, medium_type_idx, p, work.lambda)

        # Add emission if present
        if !is_black(mp.Le) && work.depth < max_depth
            pr = σ_maj_0 * T_maj[1]
            if pr > 1f-10
                r_e = r_u * σ_maj * T_maj / pr
                if !is_black(r_e)
                    Le_contrib = beta * mp.σ_a * T_maj * mp.Le / (pr * average(r_e))
                    add_to_pixel!(pixel_L, work.pixel_index, Le_contrib, work.lambda)
                end
            end
        end

        # Compute event probabilities
        p_absorb = mp.σ_a[1] / σ_maj_0
        p_scatter = mp.σ_s[1] / σ_maj_0

        # Sample event type
        u_event = rand(Float32)

        if u_event < p_absorb
            # === ABSORPTION ===
            return SampleTMajResult(SpectralRadiance(0f0), r_u, r_l, true)

        elseif u_event < p_absorb + p_scatter
            # === REAL SCATTERING ===
            if work.depth >= max_depth
                return SampleTMajResult(beta, r_u, r_l, true)
            end

            # Update beta and r_u for scattering
            pdf = T_maj[1] * mp.σ_s[1]
            if pdf > 1f-10
                beta = beta * T_maj * mp.σ_s / pdf
                r_u = r_u * T_maj * mp.σ_s / pdf
            end

            # Push to scatter queue
            scatter_item = VPMediumScatterWorkItem(
                p,
                -work.ray.d,
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
            return SampleTMajResult(beta, r_u, r_l, true)

        else
            # === NULL SCATTERING ===
            σ_n = σ_maj - mp.σ_a - mp.σ_s
            σ_n = SpectralRadiance(max(σ_n[1], 0f0), max(σ_n[2], 0f0), max(σ_n[3], 0f0), max(σ_n[4], 0f0))

            pdf = T_maj[1] * σ_n[1]
            if pdf > 1f-10
                beta = beta * T_maj * σ_n / pdf
                r_u = r_u * T_maj * σ_n / pdf
                r_l = r_l * T_maj * σ_maj / pdf
            else
                return SampleTMajResult(SpectralRadiance(0f0), r_u, r_l, true)
            end

            t = t_sample

            # Check throughput
            if is_black(beta) || is_black(r_u)
                return SampleTMajResult(beta, r_u, r_l, true)
            end
        end
    end

    # Exceeded max samples within segment
    return SampleTMajResult(beta, r_u, r_l, false)
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
