# Delta tracking kernel for volumetric path tracing
# Implements null-scattering method from pbrt-v4
#
# Key architecture (following pbrt-v4):
# 1. Rays are traced FIRST to find intersection distance (t_max)
# 2. Rays in medium are routed to medium_sample_queue with t_max
# 3. This kernel runs delta tracking up to t_max
# 4. If ray survives to t_max, it processes the stored surface hit
#
# Medium sampling uses deterministic RNG seeded from ray geometry (pbrt-v4 pattern)
# because delta tracking requires unbounded samples (not suitable for Sobol).

# ============================================================================
# GPU-compatible LCG RNG for Medium Sampling
# ============================================================================

# LCG constants (same as pbrt-v4)
const LCG_MULTIPLIER = UInt64(0x5DEECE66D)
const LCG_INCREMENT = UInt64(11)
const FLOAT32_SCALE = 2.3283064365386963f-10  # 1/(2^32)

"""
    lcg_init(ray_o, ray_d, t_max) -> UInt64

Initialize LCG state from ray geometry for deterministic medium sampling.
Following pbrt-v4's RNG initialization: Hash(ray.o, tMax), Hash(ray.d)
"""
@inline function lcg_init(ray_o::Point3f, ray_d::Vec3f, t_max::Float32)::UInt64
    # Hash ray origin and t_max
    ox = reinterpret(UInt32, ray_o[1])
    oy = reinterpret(UInt32, ray_o[2])
    oz = reinterpret(UInt32, ray_o[3])
    tm = reinterpret(UInt32, t_max)

    # Hash ray direction
    dx = reinterpret(UInt32, ray_d[1])
    dy = reinterpret(UInt32, ray_d[2])
    dz = reinterpret(UInt32, ray_d[3])

    # Combine using mix_bits (GPU-safe hash function from spectral-eval.jl)
    seed1 = mix_bits(u_uint64(ox) ⊻ (u_uint64(oy) << 16) ⊻ (u_uint64(oz) << 32) ⊻ u_uint64(tm))
    seed2 = mix_bits(u_uint64(dx) ⊻ (u_uint64(dy) << 16) ⊻ (u_uint64(dz) << 32))

    return seed1 ⊻ seed2
end

"""
    lcg_next(state) -> (UInt64, Float32)

Generate next random Float32 in [0,1) and return new state.
GPU-compatible LCG.
"""
@inline function lcg_next(state::UInt64)::Tuple{UInt64, Float32}
    new_state = state * LCG_MULTIPLIER + LCG_INCREMENT
    # Use upper 32 bits for random value (better quality than lower bits)
    r = Float32(u_uint32(new_state >> 32)) * FLOAT32_SCALE
    return (new_state, min(r, ONE_MINUS_EPSILON))
end

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
    accumulate_spectrum!(pixel_L, base, L)
end

# ============================================================================
# Delta Tracking Kernel (processes VPMediumSampleWorkItem)
# ============================================================================

@propagate_inbounds function vp_sample_medium_kernel!(
    work,
    scatter_queue,
    hit_surface_queue,
    escaped_queue,
    pixel_L,
    media,
    rgb2spec_table,
    max_depth::Int32
)
    # Run delta tracking for this ray
    sample_medium_interaction!(
        scatter_queue,
        hit_surface_queue,
        escaped_queue,
        pixel_L,
        work, media, rgb2spec_table, max_depth, Int32(0)  # max_queued not needed anymore
    )
end

"""
    sample_medium_interaction!(...)

Inner function for medium sampling with bounded t_max.
Implements delta tracking following pbrt-v4's SampleT_maj pattern.

Now uses DDA majorant iterator for GridMedium to get per-voxel majorant bounds,
significantly reducing null scattering events in sparse heterogeneous media.
"""

"""Helper to call sample_T_maj_loop! with created iterator (no capture)"""
@propagate_inbounds function _sample_with_iterator_helper(
    medium,
    table::RGBToSpectrumTable,
    ray::Raycore.Ray,
    t_max::Float32,
    λ::Wavelengths,
    T_maj_accum::SpectralRadiance,
    beta::SpectralRadiance,
    r_u::SpectralRadiance,
    r_l::SpectralRadiance,
    rng_state::UInt64,
    scatter_queue,
    pixel_L,
    work::VPMediumSampleWorkItem,
    media,
    medium_idx::MediumIndex,
    max_depth::Int32,
    max_queued::Int32,
    template_grid::MajorantGrid
)
    # Create iterator using template grid for type consistency across all media
    iter = create_majorant_iterator(table, medium, ray, t_max, λ, template_grid)

    # Call type-stable iteration (dispatches on iter type)
    return sample_T_maj_loop!(
        iter, T_maj_accum, beta, r_u, r_l, rng_state,
        scatter_queue,
        pixel_L,
        work, media, medium_idx, table, max_depth, max_queued
    )
end

@propagate_inbounds function sample_medium_interaction!(
    # Output queues
    scatter_queue,
    hit_surface_queue,
    escaped_queue,
    pixel_L,
    # Input
    work::VPMediumSampleWorkItem,
    media,
    rgb2spec_table,
    max_depth::Int32,
    max_queued::Int32
)
    medium_idx = work.medium_idx
    t_max = work.t_max

    # Delta tracking state
    beta = work.beta
    r_u = work.r_u
    r_l = work.r_l

    # Initialize deterministic RNG from ray geometry (pbrt-v4 pattern)
    # Medium sampling uses LCG because delta tracking requires unbounded samples
    rng_state = lcg_init(work.ray.o, work.ray.d, t_max)

    # Accumulated transmittance across segments (reset after each interaction)
    T_maj_accum = SpectralRadiance(1f0)

    # Extract template grid from media tuple for type consistency
    # This ensures all RayMajorantIterator instances have the same type parameter
    template_grid = get_template_grid_from_tuple(media)

    # Use with_medium pattern to avoid Union types (GPU-safe)
    result = with_medium(
        _sample_with_iterator_helper,
        media, medium_idx,
        rgb2spec_table, work.ray, t_max, work.lambda,
        T_maj_accum, beta, r_u, r_l, rng_state,
        scatter_queue,
        pixel_L,
        work, media, medium_idx, max_depth, max_queued,
        template_grid
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
        push!(escaped_queue, VPEscapedRayWorkItem(work, beta, r_u, r_l))
    else
        # Ray reached surface - push to hit_surface_queue for material eval
        push!(hit_surface_queue, VPHitSurfaceWorkItem(work, beta, r_u, r_l))
    end
    return
end

# Result type for sample_T_maj_loop!
struct SampleTMajResult
    beta::SpectralRadiance
    r_u::SpectralRadiance
    r_l::SpectralRadiance
    rng_state::UInt64  # Updated RNG state for continuation
    done::Bool  # true if terminated (absorption/scatter), false if reached end
end

"""
    sample_T_maj_loop!(iter::RayMajorantIterator, ...) -> SampleTMajResult

Delta tracking loop using the unified RayMajorantIterator.
Handles both homogeneous media (single segment) and heterogeneous media (DDA traversal).
Uses deterministic LCG RNG for medium sampling (pbrt-v4 pattern).
"""
@propagate_inbounds function sample_T_maj_loop!(
    iter::RayMajorantIterator,
    T_maj_accum::SpectralRadiance,
    beta::SpectralRadiance,
    r_u::SpectralRadiance,
    r_l::SpectralRadiance,
    rng_state::UInt64,
    scatter_queue,
    pixel_L,
    work::VPMediumSampleWorkItem,
    media,
    medium_idx::MediumIndex,
    rgb2spec_table,
    max_depth::Int32,
    max_queued::Int32
)
    # Iterate over majorant segments (bounded for GPU)
    # For homogeneous media this loops once, for DDA it traverses voxels
    max_segments = Int32(256)

    current_iter = iter
    current_rng = rng_state
    for _ in Int32(1):max_segments
        seg, new_iter, valid = ray_majorant_next(current_iter)
        if !valid
            # No more segments - ray survived
            break
        end
        current_iter = new_iter

        # Sample within this segment
        result = sample_segment!(
            seg, T_maj_accum, beta, r_u, r_l, current_rng,
            scatter_queue,
            pixel_L,
            work, media, medium_idx, rgb2spec_table, max_depth, max_queued
        )

        if result.done
            # Terminated (absorption or scatter)
            return result
        end

        # Update state for next segment
        beta = result.beta
        r_u = result.r_u
        r_l = result.r_l
        current_rng = result.rng_state
        # T_maj_accum is reset to 1.0 inside sample_segment! after interactions
        T_maj_accum = SpectralRadiance(1f0)
    end

    return SampleTMajResult(beta, r_u, r_l, current_rng, false)
end

"""
    sample_segment!(seg, T_maj_accum, beta, r_u, r_l, rng_state, ...) -> SampleTMajResult

Sample interactions within a single majorant segment.
Following pbrt-v4's inner loop of SampleT_maj.
Uses deterministic LCG RNG for medium sampling (pbrt-v4 pattern).
"""
@propagate_inbounds function sample_segment!(
    seg::RayMajorantSegment,
    T_maj_accum::SpectralRadiance,
    beta::SpectralRadiance,
    r_u::SpectralRadiance,
    r_l::SpectralRadiance,
    rng_state::UInt64,
    scatter_queue,
    pixel_L,
    work::VPMediumSampleWorkItem,
    media,
    medium_idx::MediumIndex,
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
        return SampleTMajResult(beta, r_u, r_l, rng_state, false)
    end

    t = seg.t_min
    t_max_seg = seg.t_max

    # Current RNG state (will be updated in loop)
    current_rng = rng_state

    # Inner sampling loop (bounded iterations)
    max_samples = Int32(100)
    for _ in Int32(1):max_samples
        # Sample exponential distance using LCG
        current_rng, u = lcg_next(current_rng)
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
            return SampleTMajResult(beta, r_u, r_l, current_rng, false)
        end

        # Compute transmittance for this step
        T_maj = exp(-dt * σ_maj)

        # Sample medium properties at interaction point
        p = Point3f(work.ray.o + work.ray.d * t_sample)
        mp = sample_point_dispatch(rgb2spec_table, media, medium_idx, p, work.lambda)

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

        # Sample event type using LCG
        current_rng, u_event = lcg_next(current_rng)

        if u_event < p_absorb
            # === ABSORPTION ===
            return SampleTMajResult(SpectralRadiance(0f0), r_u, r_l, current_rng, true)

        elseif u_event < p_absorb + p_scatter
            # === REAL SCATTERING ===
            if work.depth >= max_depth
                return SampleTMajResult(beta, r_u, r_l, current_rng, true)
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

            push!(scatter_queue, scatter_item)
            return SampleTMajResult(beta, r_u, r_l, current_rng, true)

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
                return SampleTMajResult(SpectralRadiance(0f0), r_u, r_l, current_rng, true)
            end

            t = t_sample

            # Check throughput
            if is_black(beta) || is_black(r_u)
                return SampleTMajResult(beta, r_u, r_l, current_rng, true)
            end
        end
    end

    # Exceeded max samples within segment
    return SampleTMajResult(beta, r_u, r_l, current_rng, false)
end

# ============================================================================
# High-Level Medium Sampling Function
# ============================================================================

function vp_sample_medium_interaction!(state::VolPathState, media)
    foreach(vp_sample_medium_kernel!,
        state.medium_sample_queue,
        state.medium_scatter_queue,
        state.hit_surface_queue,
        state.escaped_queue,
        state.pixel_L,
        media,
        state.rgb2spec_table,
        state.max_depth,
    )
    return nothing
end
