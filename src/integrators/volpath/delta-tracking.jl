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
    per_material_queue,
    hit_surface_queue,
    hit_area_light_queue,
    next_ray_queue,
    escaped_queue,
    pixel_L,
    media,
    materials,
    rgb2spec_table,
    max_depth::Int32
)
    # Run delta tracking for this ray
    sample_medium_interaction!(
        scatter_queue,
        per_material_queue,
        hit_surface_queue,
        hit_area_light_queue,
        next_ray_queue,
        escaped_queue,
        pixel_L,
        work, media, materials, rgb2spec_table, max_depth
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
@propagate_inbounds function sample_with_iterator_helper(
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
    medium_idx::SetKey,
    max_depth::Int32,
    template_grid::MajorantGrid
)
    # Create iterator using template grid for type consistency across all media
    iter = create_majorant_iterator(medium, table, ray, t_max, λ, template_grid)

    # Call type-stable iteration (dispatches on iter type)
    return sample_T_maj_loop!(
        iter, T_maj_accum, beta, r_u, r_l, rng_state,
        scatter_queue,
        pixel_L,
        work, media, medium_idx, table, max_depth
    )
end

@propagate_inbounds function sample_medium_interaction!(
    # Output queues
    scatter_queue,
    per_material_queue,
    hit_surface_queue,
    hit_area_light_queue,
    next_ray_queue,
    escaped_queue,
    pixel_L,
    # Input
    work::VPMediumSampleWorkItem,
    media,
    materials,
    rgb2spec_table,
    max_depth::Int32
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

    # Use with_index pattern to avoid Union types (GPU-safe)
    result = Raycore.with_index(
        sample_with_iterator_helper,
        media, medium_idx,
        rgb2spec_table, work.ray, t_max, work.lambda,
        T_maj_accum, beta, r_u, r_l, rng_state,
        scatter_queue,
        pixel_L,
        work, media, medium_idx, max_depth,
        template_grid
    )

    # Unpack result
    beta = result.beta
    r_u = result.r_u
    r_l = result.r_l
    done = result.done
    ray_d = result.ray_d

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
        # Use deflected direction for environment lookup
        escaped = VPEscapedRayWorkItem(
            ray_d, work.lambda, work.pixel_index,
            beta, r_u, r_l,
            work.depth, work.specular_bounce,
            work.prev_intr_p, work.prev_intr_n
        )
        push!(escaped_queue, escaped)
    else
        # Ray reached surface — resolve MixMaterial and route into the
        # matching per-material typed queue.  This must happen at the push
        # site (not inside the shading kernel), so the typed queue picks
        # the concrete material type.  Mirrors the non-medium path in
        # `vp_trace_and_shade_kernel!`.
        wo = -work.ray.d
        resolved_mat_idx = resolve_mix_material(
            materials, work.hit_material_idx, work.hit_pi, wo, work.hit_uv,
        )

        # Null-material surface (pbrt `Material "interface"` / nullptr):
        # no BSDF, no DL.  Push a continuation ray with the medium swap
        # inline (depth NOT incremented), same as the non-medium trace
        # kernel's null-material handling, emission of an emissive one first.
        if !Raycore.is_valid(resolved_mat_idx)
            push_arealight_hit!(hit_area_light_queue, work.hit_arealight_id,
                work.hit_pi, work.hit_n, work.hit_uv, wo, work, beta, r_u, r_l,
                work.hit_triangle_area, work.t_max)
            ray_d = -wo
            new_medium = null_crossing_medium(work.hit_interface, work.medium_idx, ray_d, work.hit_n)
            offset_dir = if dot(ray_d, work.hit_n) > 0f0; work.hit_n; else; -work.hit_n; end
            ray_origin = Point3f(work.hit_pi + offset_dir * 1f-4)
            new_ray = Raycore.Ray(o=ray_origin, d=ray_d, t_max=Inf32, time=0f0)
            push!(next_ray_queue, VPRayWorkItem(
                new_ray, work.depth,
                work.lambda, work.pixel_index,
                beta, r_u, r_l,
                work.prev_intr_p, work.prev_intr_n,
                work.eta_scale,
                work.specular_bounce, work.any_non_specular_bounces,
                new_medium))
            return
        end

        # Build VPHitSurfaceWorkItem manually so we can stamp the RESOLVED
        # material index (the VPMediumSampleWorkItem constructor would keep
        # the unresolved one, which would mis-route a MixMaterial hit).
        # Bump-perturbed shading frame is kept as computed in
        # `vp_trace_and_shade_kernel!` (using the unresolved material's bump
        # map) — same behaviour as before the per-material split.
        hit_work = VPHitSurfaceWorkItem(
            work.ray,
            work.hit_pi, work.hit_ps, work.hit_n, work.hit_dpdu, work.hit_dpdv,
            work.hit_ns, work.hit_dpdus, work.hit_dpdvs,
            work.hit_uv, resolved_mat_idx, work.hit_interface,
            work.hit_face_idx, work.hit_bary,
            work.lambda, work.pixel_index,
            beta, r_u, r_l,
            work.depth, work.eta_scale,
            work.specular_bounce, work.any_non_specular_bounces,
            work.prev_intr_p, work.prev_intr_n,
            work.medium_idx,
        )
        enqueue_after_intersection!(per_material_queue, hit_area_light_queue, materials,
            hit_surface_queue, hit_work,
            work.hit_arealight_id, work.hit_triangle_area, work.t_max)
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
    ray_d::Vec3f    # Deflected ray direction (for gravitational lensing)
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
    medium_idx::SetKey,
    rgb2spec_table,
    max_depth::Int32
)
    # Iterate over majorant segments (bounded for GPU)
    # For homogeneous media this loops once, for DDA it traverses voxels
    max_segments = Int32(256)

    # Ray state for deflection (mutable during tracking)
    ray_d = work.ray.d

    current_iter = iter
    current_rng = rng_state
    for _ in Int32(1):max_segments
        seg, new_iter, valid = ray_majorant_next(current_iter, media)
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
            work, media, medium_idx, rgb2spec_table, max_depth,
            ray_d
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
        ray_d = result.ray_d
        # T_maj_accum is reset to 1.0 inside sample_segment! after interactions
        T_maj_accum = SpectralRadiance(1f0)
    end

    return SampleTMajResult(beta, r_u, r_l, current_rng, false, ray_d)
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
    medium_idx::SetKey,
    rgb2spec_table,
    max_depth::Int32,
    ray_d::Vec3f
)
    σ_maj = seg.σ_maj
    σ_maj_0 = σ_maj[1]

    # Handle zero-majorant segment (empty voxel)
    if σ_maj_0 < 1f-10
        # Just apply transmittance for this segment (which is 1.0 for zero extinction)
        # No need to sample - ray passes through
        return SampleTMajResult(beta, r_u, r_l, rng_state, false, ray_d)
    end

    t = seg.t_min
    t_max_seg = seg.t_max

    # Incremental ray position tracking.
    # For straight rays this reproduces work.ray.o + work.ray.d * t exactly.
    # For deflecting media (spacetime), the ray follows a curved path.
    ray_o = Point3f(work.ray.o + ray_d * t)

    # Current RNG state (will be updated in loop)
    current_rng = rng_state

    # Inner sampling loop (bounded iterations)
    max_samples = Int32(1024)
    for _ in Int32(1):max_samples
        # Sample exponential distance using LCG
        current_rng, u = lcg_next(current_rng)
        dt = -log(max(1f-10, 1f0 - u)) / σ_maj_0
        t_sample = t + dt

        if t_sample >= t_max_seg
            # Passed end of segment without interaction
            # Apply transmittance for remaining distance
            dt_remain = t_max_seg - t
            T_maj = fast_exp(-dt_remain * σ_maj)
            T_maj_0 = T_maj[1]
            if T_maj_0 > 1f-10
                beta = beta * T_maj / T_maj_0
                r_u = r_u * T_maj / T_maj_0
                r_l = r_l * T_maj / T_maj_0
            end
            # Continue to next segment
            return SampleTMajResult(beta, r_u, r_l, current_rng, false, ray_d)
        end

        # Compute transmittance for this step
        T_maj = fast_exp(-dt * σ_maj)

        # Sample medium properties at interaction point (incremental position)
        p = Point3f(ray_o + ray_d * dt)
        mp = Raycore.with_index(sample_point, media, medium_idx, media, rgb2spec_table, p, work.lambda)

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
            return SampleTMajResult(SpectralRadiance(0f0), r_u, r_l, current_rng, true, ray_d)

        elseif u_event < p_absorb + p_scatter
            # === REAL SCATTERING ===
            if work.depth >= max_depth
                return SampleTMajResult(beta, r_u, r_l, current_rng, true, ray_d)
            end

            # Update beta and r_u for scattering
            pdf = T_maj[1] * mp.σ_s[1]
            if pdf > 1f-10
                beta = beta * T_maj * mp.σ_s / pdf
                r_u = r_u * T_maj * mp.σ_s / pdf
            end

            # Push to scatter queue (use deflected direction)
            scatter_item = VPMediumScatterWorkItem(
                p,
                -ray_d,  # wo points back toward camera (use deflected direction)
                work.ray.time,
                work.lambda,
                work.pixel_index,
                beta,
                r_u,
                work.depth,
                work.medium_idx,
                mp.g,
                work.eta_scale
            )

            push!(scatter_queue, scatter_item)
            return SampleTMajResult(beta, r_u, r_l, current_rng, true, ray_d)

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
                return SampleTMajResult(SpectralRadiance(0f0), r_u, r_l, current_rng, true, ray_d)
            end

            t = t_sample
            # Advance ray origin to interaction point and apply deflection
            ray_o = p
            ray_d = Raycore.with_index(apply_deflection, media, medium_idx, p, ray_d, dt)

            # Check throughput
            if is_black(beta) || is_black(r_u)
                return SampleTMajResult(beta, r_u, r_l, current_rng, true, ray_d)
            end
        end
    end

    # Exceeded max samples within segment
    return SampleTMajResult(beta, r_u, r_l, current_rng, false, ray_d)
end

