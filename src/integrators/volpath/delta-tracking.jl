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
    surface_hits,
    arrivals,
    next_ray_queue,
    escaped_queue,
    pixel_L,
    accel,
    media_interfaces,
    media,
    materials,
    lights,
    rgb2spec_table,
    bvh_nodes,
    light_to_bit_trail,
    num_infinite_lights::Int32,
    num_bvh_lights::Int32,
    max_depth::Int32
)
    # Run delta tracking for this ray
    sample_medium_interaction!(
        scatter_queue,
        per_material_queue,
        surface_hits,
        arrivals,
        next_ray_queue,
        escaped_queue,
        pixel_L,
        work, accel, media_interfaces, media, materials, lights, rgb2spec_table,
        bvh_nodes, light_to_bit_trail, num_infinite_lights, num_bvh_lights, max_depth
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
    step_limit = medium_step_limit(medium, ray.o, ray.d)
    iter = create_majorant_iterator(medium, table, ray, min(t_max, step_limit), λ, template_grid)

    # Call type-stable iteration (dispatches on iter type)
    result = sample_T_maj_loop!(
        iter, T_maj_accum, beta, r_u, r_l, rng_state,
        scatter_queue,
        pixel_L,
        work, media, medium_idx, table, max_depth,
        control(medium, media, table, λ), controlmajorant(medium)
    )
    # An empty majorant iterator still spans vacuum along a curved path.
    # Advance it, otherwise a capped continuation retraces the same origin
    # forever when its step lies outside the density grid.
    if isfinite(step_limit) && !result.done && result.ray_o == ray.o
        distance = min(t_max, step_limit)
        endpoint = Point3f(ray.o + distance * ray.d)
        direction = medium_end_direction(medium, endpoint, ray.d, step_limit, distance)
        result = SampleTMajResult(result.beta, result.r_u, result.r_l,
            result.rng_state, false, direction, endpoint, result.retrace)
    end
    if !result.done && step_limit < t_max
        return SampleTMajResult(result.beta, result.r_u, result.r_l, result.rng_state,
                                false, result.ray_d, result.ray_o, true)
    end
    return result
end

@propagate_inbounds function sample_medium_interaction!(
    # Output queues
    scatter_queue,
    per_material_queue,
    surface_hits,
    arrivals,
    next_ray_queue,
    escaped_queue,
    pixel_L,
    # Input
    work::VPMediumSampleWorkItem,
    accel,
    media_interfaces,
    media,
    materials,
    lights,
    rgb2spec_table,
    bvh_nodes,
    light_to_bit_trail,
    num_infinite_lights::Int32,
    num_bvh_lights::Int32,
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

    # A bent ray cannot reuse a hit computed before the bend. Re-enter the
    # existing intersection queue from its actual position, without counting
    # transport subdivisions as scattering/BSDF bounces.
    if result.retrace
        if !is_black(beta) && !is_black(r_u)
            ray = Raycore.Ray(o=result.ray_o, d=ray_d, t_max=Inf32, time=work.ray.time)
            push!(next_ray_queue, VPRayWorkItem(
                ray, work.depth, work.lambda, work.pixel_index, beta, r_u, r_l,
                work.prev_intr_p, work.prev_intr_n, work.eta_scale,
                work.specular_bounce, work.any_non_specular_bounces, work.medium_idx))
        end
        return
    end

    # Ray survived to t_max - process what's at the end. At the depth limit too: what
    # it reaches may emit, and the shading of a surface it hits ends the path there.
    if is_black(beta) || is_black(r_u)
        return
    end

    if !has_surface_hit(work)
        # A finite tracing horizon is a segment boundary, not an escape.
        # Dense media can restrict geometry queries to a few mean free paths;
        # surviving flights continue without adding a scattering vertex.
        if isfinite(work.ray.t_max)
            ray = Raycore.Ray(o=Point3f(work.ray.o + work.ray.t_max * ray_d),
                              d=ray_d, t_max=Inf32, time=work.ray.time)
            push!(next_ray_queue, VPRayWorkItem(ray, work.depth, work.lambda,
                work.pixel_index, beta, r_u, r_l, work.prev_intr_p, work.prev_intr_n,
                work.eta_scale, work.specular_bounce, work.any_non_specular_bounces,
                work.medium_idx))
            return
        end
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
        # The ray reached its surface: the surface hit goes to its material's
        # shading kernel as the trace's do, with the path state delta tracking left
        # (the transmittance in `beta`, `r_u` and `r_l`) as an arrival.
        rec = work.hit
        primitive = Raycore.hit_primitive(accel, rec)
        mi = media_interfaces[resolve_mi_idx(rec.custom, primitive)]
        # Null-material surface (pbrt `Material "interface"` / nullptr): no BSDF,
        # so no shading kernel to go to. Its emission, then the ray goes on at the
        # same depth with the medium swapped, as in the trace.
        if !Raycore.is_valid(mi.material)
            bary = Raycore.hit_barycentric(accel, rec)
            geom = vp_compute_surface_geometry(primitive, bary, work.ray.o, ray_d, work.t_max)
            wo = -ray_d
            accumulate_arealight_hit!(pixel_L, lights, rgb2spec_table,
                bvh_nodes, light_to_bit_trail, num_infinite_lights, num_bvh_lights,
                primitive.metadata.arealight_id, geom.n, geom.uv, wo,
                work, beta, r_u, r_l, Raycore.area(primitive), work.t_max)
            new_medium = null_crossing_medium(mi, work.medium_idx, ray_d, geom.n)
            new_ray = Raycore.Ray(o=offset_surface_origin(geom.pi, geom.n, ray_d), d=ray_d,
                                  t_max=Inf32, time=0f0)
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
        wants_shading(work.depth, max_depth, primitive.metadata.arealight_id, materials, mi.material) || return
        arrival = VPRayWorkItem(
            Raycore.Ray(o=work.ray.o, d=ray_d, t_max=work.ray.t_max, time=work.ray.time),
            work.depth, work.lambda, work.pixel_index,
            beta, r_u, r_l,
            work.prev_intr_p, work.prev_intr_n,
            work.eta_scale,
            work.specular_bounce, work.any_non_specular_bounces,
            work.medium_idx)
        a = push!(arrivals, arrival)
        a <= arrivals.capacity || return
        enqueue_surface_hit!(per_material_queue, materials, surface_hits, mi.material,
                             VPSurfaceHit(-Int32(a), rec))
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
    ray_o::Point3f  # Actual endpoint for transport continuations
    retrace::Bool  # Must recompute geometry rather than reuse work's stored hit
end

# Existing straight-medium returns do not need a continuation position.
SampleTMajResult(beta, r_u, r_l, rng, done, ray_d) =
    SampleTMajResult(beta, r_u, r_l, rng, done, ray_d, Point3f(0), false)

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
    max_depth::Int32,
    mc::MediumProperties,
    c0::Float32
)
    # Bound by the actual grid dimensions. A fixed cap silently skipped the
    # remainder of long rays through fine whitewater grids.
    max_segments = majorant_segment_limit(iter)

    # Ray state for deflection (mutable during tracking)
    ray_d = work.ray.d
    ray_o = work.ray.o
    # A curved medium's chord is ONE straight line, bent once at its end: every
    # segment samples along the chord's direction, and only the last segment's
    # end point and direction leave the loop. Carrying an intermediate segment's
    # end direction into the next one bent the chord at every voxel boundary of
    # a majorant grid and moved its later samples off the chord.
    curved = Raycore.with_index(medium_step_limit, media, medium_idx, work.ray.o, work.ray.d) < Inf32
    end_d = ray_d

    current_iter = iter
    current_rng = rng_state
    # The stored control bound covers the entire visible spectrum. A path
    # only carries four wavelengths; tighten that part to those wavelengths.
    # Residual bounds and the sampled physical coefficients are unchanged.
    path_control = max_component(σ_t(mc))
    for _ in Int32(1):max_segments
        seg, new_iter, valid = ray_majorant_next(current_iter, media)
        if !valid
            # No more segments - ray survived
            break
        end
        current_iter = new_iter

        if c0 > 0f0
            residual_bound = max(seg.σ_maj[1]-c0,0f0)
            seg = RayMajorantSegment(seg.t_min,seg.t_max,SpectralRadiance(residual_bound+path_control))
        end

        # Sample within this segment
        result = sample_segment!(
            seg, T_maj_accum, beta, r_u, r_l, current_rng,
            scatter_queue,
            pixel_L,
            work, media, medium_idx, rgb2spec_table, max_depth,
            ray_d, mc, c0 > 0f0 ? path_control : c0
        )

        if result.done || result.retrace
            # Terminated (absorption or scatter)
            return result
        end

        # Update state for next segment
        beta = result.beta
        r_u = result.r_u
        r_l = result.r_l
        current_rng = result.rng_state
        end_d = result.ray_d
        ray_d = curved ? work.ray.d : result.ray_d
        ray_o = result.ray_o
        # T_maj_accum is reset to 1.0 inside sample_segment! after interactions
        T_maj_accum = SpectralRadiance(1f0)
    end

    return SampleTMajResult(beta, r_u, r_l, current_rng, false, end_d, ray_o, false)
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
    ray_d::Vec3f,
    mc::MediumProperties,
    c0::Float32
)
    σ_maj = seg.σ_maj
    σ_maj_0 = σ_maj[1]
    # Whether this segment's majorant has room for any residual (see `control`).
    has_residual = σ_maj_0 > c0 * 1.0001f0 + 1f-6
    planned_step = Raycore.with_index(medium_step_limit, media, medium_idx, work.ray.o, ray_d)
    curved = planned_step < Inf32

    # Handle zero-majorant segment (empty voxel)
    if σ_maj_0 < 1f-10
        # Just apply transmittance for this segment (which is 1.0 for zero extinction)
        # No need to sample - ray passes through
        curved || return SampleTMajResult(beta, r_u, r_l, rng_state, false, ray_d)
        endpoint = Point3f(work.ray.o + ray_d * seg.t_max)
        direction = Raycore.with_index(medium_end_direction, media, medium_idx,
                                       endpoint, ray_d, planned_step, seg.t_max)
        return SampleTMajResult(beta, r_u, r_l, rng_state, false, direction, endpoint, false)
    end

    t = seg.t_min
    t_max_seg = seg.t_max

    # Incremental ray position tracking.
    # For straight rays this reproduces work.ray.o + work.ray.d * t exactly.
    # For deflecting media (spacetime), the ray follows a curved path.
    ray_o = Point3f(work.ray.o + ray_d * t)

    # Current RNG state (will be updated in loop)
    current_rng = rng_state

    # Null events do not advance path depth. Continue until the sampled flight
    # leaves the segment or a real interaction occurs; dropping its tail biases
    # dense volumes when the majorant is loose.
    while t < t_max_seg
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
            endpoint = Point3f(ray_o + ray_d * dt_remain)
            direction = curved ? Raycore.with_index(medium_end_direction, media, medium_idx,
                                                    endpoint, ray_d, planned_step, seg.t_max) : ray_d
            return SampleTMajResult(beta, r_u, r_l, current_rng, false, direction, endpoint, false)
        end

        # Compute transmittance for this step
        T_maj = fast_exp(-dt * σ_maj)

        # Sample medium properties at interaction point (incremental position)
        p = Point3f(ray_o + ray_d * dt)
        # The physical direction at the sample: a curved chord's own, bent part
        # way along it; a straight ray's, unchanged.
        local_d = Raycore.with_index(medium_point_direction, media, medium_idx,
                                    p, ray_d, planned_step, t_sample)

        # Sample event type using LCG. Decomposition tracking (see `control`): the
        # control's absorption and scattering come first and need no lookup; the
        # residual is looked up only past them, and only where it can be.
        current_rng, u_event = lcg_next(current_rng)
        # `mp` is the part the event falls in, `σ_known` all of the medium looked at.
        p_control = (mc.σ_a[1] + mc.σ_s[1]) / σ_maj_0
        mp, σ_known = if u_event < p_control
            mc, σ_t(mc)
        elseif has_residual
            mr = Raycore.with_index(residual, media, medium_idx, media, rgb2spec_table, p, work.lambda, local_d)
            # The residual's events follow the control's in [0, 1).
            u_event -= p_control
            mr, σ_t(mc) + σ_t(mr)
        else
            u_event -= p_control
            MediumProperties(), σ_t(mc)
        end

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

        # Compute event probabilities (of the part `mp` is, the control's or the residual's)
        p_absorb = mp.σ_a[1] / σ_maj_0
        p_scatter = mp.σ_s[1] / σ_maj_0

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
                -local_d,  # physical direction at the scattering vertex
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
            σ_n = σ_maj - σ_known
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
            # A null collision is an emission/extinction SAMPLE, not a bend
            # or a new geometry segment. Keep sampling the same bounded chord
            # and bend once at its end. Retracing at every null event made
            # geodesics depend on the random majorant samples and paid another
            # complete wavefront round for each sample in empty space.
            ray_o = p
            if !curved
                ray_d = Raycore.with_index(apply_deflection, media, medium_idx, p, ray_d, dt)
            end

            # Check throughput
            if is_black(beta) || is_black(r_u)
                return SampleTMajResult(beta, r_u, r_l, current_rng, true, ray_d)
            end
        end
    end

    return SampleTMajResult(beta, r_u, r_l, current_rng, false, ray_d)
end
