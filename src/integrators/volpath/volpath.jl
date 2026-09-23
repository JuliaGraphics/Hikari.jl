# VolPath Integrator - Volumetric Path Tracing with Participating Media
# Wavefront implementation following pbrt-v4's VolPathIntegrator

using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
import KernelAbstractions as KA


# ============================================================================
# VolPath Integrator
# ============================================================================

"""
    VolPath <: Integrator

Volumetric path tracer using hero wavelength sampling and wavefront architecture.

Extends path tracing with support for participating media (fog, smoke, clouds).
Uses delta tracking with null scattering for efficient medium traversal.

The main loop follows pbrt-v4's structure:
1. If ray is in medium: run delta tracking
   - Absorption → terminate
   - Real scatter → sample direct light + phase function → continue
   - Null scatter → continue tracking
   - Escape medium → proceed to intersection
2. Intersect with scene
3. Handle surface (emission + BSDF) or environment light
"""
mutable struct VolPath <: Integrator
    max_depth::Int32
    samples_per_pixel::Int32
    russian_roulette_depth::Int32
    regularize::Bool  # Apply BSDF regularization after first non-specular bounce
    max_component_value::Float32  # Firefly suppression: clamp RGB components (pbrt-v4 style)
    filter_params::GPUFilterParams  # Filter for pixel reconstruction (pbrt-v4 style)
    filter_sampler_data::Union{Nothing, GPUFilterSamplerData}  # Tabulated data for importance sampling
    accumulation_eltype::DataType  # Element type for RGB/weight accumulators (Float64 or Float32)

    # Hardware ray tracing flag — when true, RayMakie creates HWTLAS instead of software TLAS
    hw_accel::Bool

    # Pixel sensor: spectral → RGB conversion with ISO, white balance, exposure
    sensor::PixelSensor

    # Cached render state
    state::Union{Nothing, VolPathState}

    # Cached initial medium detection (avoids per-sample GPU alloc + flush).
    # `initial_medium_camera_pos === nothing` means invalid.
    initial_medium_camera_pos::Any  # Point3f or nothing
    initial_medium_key::Any         # SetKey or nothing

    # Cached GPU-adapted filter sampler data (avoids re-uploading every sample).
    # nothing = not yet adapted.
    filter_sampler_gpu::Any

    # Cull mask passed to trace_closest_hits! / trace_closest_hits_indirect!.
    # ANDed against each TLAS instance's instance_mask; only instances where
    # (cull_mask & instance_mask) != 0 are visible to rays from this integrator.
    # Default 0xFF matches all instances (identical to old hard-coded behavior).
    cull_mask::UInt32

    # The scene this integrator's plans listen to: `sync!` and the scene
    # mutators drop the plans of every listening integrator, which is what lets
    # a render compare nothing. `nothing` until the first `render!` registers.
    listening_to::Any

    # The adapted scene behind those plans. `Adapt.adapt(backend, scene)` is a
    # bag of VIEWS over the scene's device arrays, so a mutation that writes
    # in place (a refit, a surgical same-type `push!`) is visible through it,
    # and one that changes shape announces itself through `notify_scene_changed`
    # — which drops this via `invalidate!`. It used to be rebuilt on every
    # sample: a few hundred bytes of wrapper construction per sample, for
    # nothing.
    adapted::Any

    # The per-run refs of the current plans, typed (see `PerRun`): `nothing`
    # until the first `render!` builds plans, and again when they are dropped.
    # The plans own the object; this is the typed handle to it.
    perrun::Union{Nothing, PerRun{PerspectiveCamera}, PerRun{MatrixCamera}}
end

"""
    VolPath(; max_depth=8, samples=64, russian_roulette_depth=3, regularize=true,
            max_component_value=10, filter=GaussianFilter(),
            accumulation_eltype=Float32, hw_accel=nothing)

Create a VolPath integrator for volumetric path tracing.

# Arguments
- `max_depth`: Maximum path depth
- `samples`: Number of samples per pixel
- `russian_roulette_depth`: Depth at which to start Russian roulette
- `regularize`: Apply BSDF regularization to reduce fireflies (default: true).
  When enabled, near-specular BSDFs are roughened after the first non-specular
  bounce, following pbrt-v4's approach.
- `max_component_value`: Maximum RGB component value before clamping (default: 10).
  When set to a finite value, RGB values are scaled down if any component exceeds this.
  This is pbrt-v4's firefly suppression mechanism. Try values like 10.0 or 100.0.
- `filter`: Pixel reconstruction filter (default: GaussianFilter with radius 1.5, sigma 0.5).
  Supports BoxFilter, TriangleFilter, GaussianFilter, MitchellFilter, LanczosSincFilter.
  All filters use importance sampling with weight≈1 (pbrt-v4 compatible).
- `accumulation_eltype`: Element type for RGB/weight accumulators (default: Float32).
  Use Float64 on backends that support double precision for higher accumulation precision.
- `hw_accel`: Enable hardware ray tracing (default: false).
  When true and using LavaBackend, RayMakie creates an HWTLAS for hardware BVH traversal
  instead of software BVH. Dispatch happens on the accel type, not this flag.

Sensor simulation (ISO, exposure_time, white_balance) is applied during rendering
via the PixelSensor on the integrator state, matching pbrt-v4. The framebuffer
contains sensor-calibrated linear sRGB. Postprocessing is purely display mapping
(exposure, tonemapping, gamma).
"""
function VolPath(;
    max_depth::Int = 8,
    samples::Int = 64,
    russian_roulette_depth::Int = 3,
    regularize::Bool = true,
    max_component_value::Real = 10f0,
    filter::AbstractFilter = GaussianFilter(),  # pbrt-v4 default: Gaussian(1.5, 0.5)
    accumulation_eltype::DataType = Float32,
    hw_accel::Bool = false,
    sensor::PixelSensor = PixelSensor(),
    cull_mask::UInt32 = UInt32(0xFF)
)
    @assert accumulation_eltype in (Float32, Float64) "accumulation_eltype must be Float32 or Float64"
    # Build filter sampler data for importance sampling (nothing for Box/Triangle)
    sampler_data = GPUFilterSamplerData(filter)
    vp = VolPath(
        Int32(max_depth),
        Int32(samples),
        Int32(russian_roulette_depth),
        regularize,
        Float32(max_component_value),
        GPUFilterParams(filter),
        sampler_data,
        accumulation_eltype,
        hw_accel,
        sensor,
        nothing,           # state
        nothing,           # initial_medium_camera_pos
        nothing,           # initial_medium_key
        nothing,           # filter_sampler_gpu
        cull_mask,
        nothing,           # listening_to
        nothing,           # adapted
        nothing,           # perrun
    )
    # No GC finalizer — the VolPathState is shared with scene_state.integrator_state,
    # so proactive free! from a finalizer would free arrays still in use by the render loop.
    # GPU memory is freed by: explicit close(screen), or LavaArray's own DataRef finalizers.
    return vp
end

"""
    invalidate!(vp::VolPath)

Drop the plans `vp` recorded — and the adapted scene they were built from —
so the next `render!` adapts and builds again. Scene edits do this themselves
— `sync!(scene)` and the scene mutators notify every integrator listening to
the scene. The adaptation goes WITH the plans: a software TLAS is adapted as a
snapshot of its node and instance arrays, and an edit that rebuilds it makes
new ones, so plans rebuilt from the cached adaptation traced the geometry from
before the edit (nine spheres after the scene was down to one) and the
accel's dirty flag was never consumed. What has no verb is the integrator's own
fields: `max_depth`, `regularize`, `samples_per_pixel`, `max_component_value`,
`filter_params` and `filter_sampler_data` are baked into the plans at
`record!`, so a caller who reassigns them between renders calls this once.
(The per-run values — camera, initial medium, filter params, sample index —
are `GPURef`s and never need this: they are written per run, not baked.)
"""
function invalidate!(vp::VolPath)
    # The adaptation first, whether or not there are plans: it is what an edit
    # made stale, and `render!` re-adapts (which is the `sync!` that consumes
    # the accel's flag) when it finds none. The medium the camera starts in was
    # detected through it, so that goes too.
    vp.adapted = nothing
    vp.initial_medium_key = nothing
    st = vp.state
    (st === nothing || st.plans === nothing) && return nothing
    # Same contract as `rebuild_plans!`: the plans go back
    # without a wait; their regions are retired, not released.
    free!(st.plans)
    st.plans = nothing
    vp.perrun = nothing
    return nothing
end

# The scene-side verb (see `notify_scene_changed` in scene.jl).
invalidate_plans!(vp::VolPath) = invalidate!(vp)

"""
    Base.close(vp::VolPath)

Release all GPU memory held by the integrator's cached render state and adapted scene.
"""
function Base.close(vp::VolPath)
    # Proactively free the GPU buffers backing vp.state — bare `vp.state = nothing`
    # relies on Julia GC to run finalizers, which races the next render's
    # allocation (peak memory becomes 2× state and can OOM on big scenes).
    # Symmetric with the integrator-change branch in RayMakie's
    # apply_screen_config!.  Same precondition: caller ensures GPU is idle.
    if vp.state !== nothing
        free!(vp.state)
    end
    vp.filter_sampler_gpu = nothing
    vp.state = nothing
    vp.initial_medium_camera_pos = nothing
    vp.initial_medium_key = nothing
    vp.listening_to = nothing
    vp.adapted = nothing
    vp.perrun = nothing
    return nothing
end

# Whether the accel carries a topology change nothing has announced yet. Scene
# mutators notify themselves; this reads the flag Raycore's accel-level
# `push!`/`delete!` set, for callers who mutated below the scene API.
# `transforms_dirty` is deliberately excluded: a refit writes the same buffers
# in place, and a recorded plan survives it.
scene_dirty(scene::Scene) = scene.accel.dirty

# Scene adaptation dispatch — overridden for AdaptedAccel in hikari_integration.jl
adapt_scene_for_render(backend, scene, ::VolPath) = Adapt.adapt(backend, scene)

# Camera medium detection dispatch — overridden for AdaptedAccel in hikari_integration.jl
detect_initial_medium(backend, accel, media_interfaces, camera_pos, ::VolPath) =
    detect_camera_medium(backend, accel, media_interfaces, camera_pos)

"""
    shades_surfaces_inline(accel) -> Bool

Does tracing already shade the surfaces it hits? The hardware RT path runs
per-material closest-hit shaders, so a hit is fully shaded by the time the
trace returns; the software BVH only records hits and leaves shading to the
post-hoc kernels. Only true for rays that never enter a medium — see
`chit_owns_surface` in `render!`.

The `AdaptedAccel` method lives in `hw-rt.jl`, with the other overrides on
the accel axis.
"""
shades_surfaces_inline(accel) = false

"""
    clear!(integrator::VolPath)

Clear the integrator's internal state (RGB and weight accumulators) for restarting progressive rendering.
"""
function Hikari.clear!(vp::VolPath)
    if vp.state !== nothing
        KernelAbstractions.fill!(vp.state.pixel_rgb, zero(eltype(vp.state.pixel_rgb)))
        KernelAbstractions.fill!(vp.state.pixel_weight_sum, zero(eltype(vp.state.pixel_weight_sum)))
    end
end

# ============================================================================
# Camera Ray Generation
# ============================================================================

"""
    vp_generate_camera_rays_kernel!(...)

Generate camera rays with per-pixel wavelength sampling and filter weight computation.
Following pbrt-v4's GetCameraSample: samples the filter, computes offset and weight.
"""
@kernel inbounds=true function vp_generate_camera_rays_kernel!(
    ray_queue,  # WorkQueue passed via Adapt
    wavelengths_per_pixel,
    pdf_per_pixel,
    filter_weight_per_pixel,
    @Const(height::Int32),
    @Const(camera_ref),         # GPURef: one-element device array holding the camera
    @Const(sample_idx_ref),
    @Const(initial_medium_ref), # GPURef{SetKey}: the medium the camera starts in
    @Const(filter_params_ref),  # GPURef{GPUFilterParams}
    filter_sampler_data,  # GPUFilterSamplerData or nothing for Box/Triangle
    rng  # SobolRNG passed via Adapt
)
    idx = @index(Global)
    # See `vp_trace_and_shade_kernel!`: the sample index is a `Mantle.GPURef`.
    # So are the camera, the initial medium and the filter params — per-run
    # values the recorded commands hold the ADDRESS of, written by an `Update`
    # in the run's own submission. What a run changes never rebuilds the plan.
    sample_idx = sample_idx_ref[Int32(1)]
    camera = camera_ref[Int32(1)]
    initial_medium_idx = initial_medium_ref[Int32(1)]
    filter_params = filter_params_ref[Int32(1)]
    # The effects flags were computed CPU-side and passed as arguments; with the
    # camera behind a ref they would need refs of their own. They are two field
    # reads on a value already in registers, so the kernel computes them.
    camera_needs_time = camera_uses_motion_blur(camera)  # shutter_open != shutter_close
    camera_needs_lens = camera_uses_lens(camera)         # lens_radius > 0
    num_pixels = rng.width * height

    @inbounds if idx <= num_pixels
        pixel_idx = idx - Int32(1)
        x = u_int32(mod(pixel_idx, rng.width)) + Int32(1)
        y = u_int32(div(pixel_idx, rng.width)) + Int32(1)

        # ZSobol sampling: deterministic low-discrepancy samples based on (pixel, sample_index)
        # Matches PBRT-v4's ZSobolSampler for better convergence.  Skip the
        # lens 2D sample for pinhole cameras and the time 1D sample for
        # shutter-stopped cameras — both are pure waste on those configs
        # and the camera-ray kernel is Sobol-bound.  Effects flags come
        # from the kernel arguments (computed CPU-side) so the GPU code
        # path is just two uniform branches, not a getproperty chain.
        pixel_sample = compute_pixel_sample(rng, Int32(x), Int32(y), sample_idx,
                                            camera_needs_time, camera_needs_lens)
        wavelength_u = pixel_sample.wavelength_u

        # Sample the filter to get offset and weight (pbrt-v4 style)
        # The filter sample uses the pixel jitter values as the 2D random input
        filter_u = Point2f(pixel_sample.jitter_x, pixel_sample.jitter_y)
        fs = filter_sample(filter_params, filter_sampler_data, filter_u)

        # Store filter weight for this sample (used in accumulation)
        filter_weight_per_pixel[idx] = fs.weight

        # Sample wavelengths for this pixel
        lambda = sample_wavelengths_visible(wavelength_u)

        # Store wavelengths and PDFs
        base = pixel_idx * Int32(4)
        wavelengths_per_pixel[base + Int32(1)] = lambda.lambda[1]
        wavelengths_per_pixel[base + Int32(2)] = lambda.lambda[2]
        wavelengths_per_pixel[base + Int32(3)] = lambda.lambda[3]
        wavelengths_per_pixel[base + Int32(4)] = lambda.lambda[4]
        pdf_per_pixel[base + Int32(1)] = lambda.pdf[1]
        pdf_per_pixel[base + Int32(2)] = lambda.pdf[2]
        pdf_per_pixel[base + Int32(3)] = lambda.pdf[3]
        pdf_per_pixel[base + Int32(4)] = lambda.pdf[4]

        # Film position with filter offset (flip Y for camera convention)
        # pbrt-v4: cs.pFilm = pPixel + fs.p + Vector2f(0.5f, 0.5f)
        # x,y are 1-based; convert to 0-based pixel center: (x-1)+0.5 = x-0.5
        p_film = Point2f(
            Float32(x) - 0.5f0 + fs.p[1],
            Float32(height) - Float32(y) + 0.5f0 + fs.p[2]
        )

        camera_sample = CameraSample(p_film, Point2f(pixel_sample.lens_u, pixel_sample.lens_v), pixel_sample.time)
        ray, weight = generate_ray(camera, camera_sample)

        if weight > 0f0
            raycore_ray = Raycore.Ray(o=ray.o, d=ray.d, t_max=ray.t_max)

            work_item = VPRayWorkItem(
                raycore_ray,
                Int32(0),  # depth
                lambda,    # Already a SampledWavelengths from sample_wavelengths_visible
                Int32(idx),
                SpectralRadiance(1f0),  # beta
                SpectralRadiance(1f0),  # r_u
                SpectralRadiance(1f0),  # r_l
                Point3f(0f0),           # prev_intr_p
                Vec3f(0f0, 0f0, 1f0),   # prev_intr_n
                1f0,                    # eta_scale
                false,                  # specular_bounce
                false,                  # any_non_specular_bounces
                initial_medium_idx
            )

            push!(ray_queue, work_item)
        end
    end
end

# ============================================================================
# Inline Sobol samples
# ============================================================================
#
# Previously, a separate `vp_generate_ray_samples_kernel!` ran once per bounce
# to pre-fill five `pixel_samples_*` SOA buffers (direct_uc, direct_u,
# indirect_uc, indirect_u, indirect_rr) from Sobol.  Per-kernel GPU timing
# (Lava.with_dispatch_timing, 2026-06-03) showed that kernel was 58 % of GPU
# time on killeroo native HW RT — the actual path-tracing kernels were
# 0.1 %.  The fix: each consumer (surface_direct_lighting_inner!,
# evaluate_material_inner!, medium_direct_lighting_inner!,
# medium_scatter_inner!) now generates only the Sobol dimensions it needs,
# inline from `(work.pixel_index, sample_idx, work.depth)`.  Eliminates the
# kernel dispatch + barrier per bounce, plus the SOA write of ~13 floats per
# pixel and the SOA read of those floats in the next dispatch (~150 MB of
# memory traffic per bounce on a 1.4 Mpx scene).


# ============================================================================
# Film Accumulation
# ============================================================================

"""
    vp_accumulate_to_rgb_kernel!(...)

Accumulate spectral radiance to RGB with filter weight support.
Following pbrt-v4's film accumulation pattern:
- Convert spectral L to XYZ using CIE matching functions
- Convert XYZ to linear sRGB
- Apply maxComponentValue clamping for firefly suppression
- Accumulate weighted RGB: rgbSum += weight * rgb
- Accumulate weight: weightSum += weight

Sensor imaging_ratio is applied here during spectral-to-RGB conversion, matching
pbrt-v4's PixelSensor::ToSensorRGB. White balance is handled via the output_matrix.
"""
@kernel inbounds=true function vp_accumulate_to_rgb_kernel!(
    pixel_rgb,
    pixel_weight_sum,
    @Const(pixel_L),
    @Const(wavelengths_per_pixel),
    @Const(pdf_per_pixel),
    @Const(filter_weight_per_pixel),
    @Const(cie_x), @Const(cie_y), @Const(cie_z),
    @Const(n_pixels::Int32),
    @Const(max_component_value::Float32),
    @Const(output_matrix::Mat3f),
    @Const(imaging_ratio::Float32),
)
    pixel_idx = @index(Global)

    # Reconstruct CIE table from passed arrays
    cie_table = CIEXYZTable(cie_x, cie_y, cie_z)

    @inbounds if pixel_idx <= n_pixels
        base = (pixel_idx - Int32(1)) * Int32(4)

        L = load(pixel_L, base + Int32(1), SpectralRadiance)

        lambda_tuple = load(wavelengths_per_pixel, base + Int32(1), NTuple{4, Float32})
        pdf_tuple = load(pdf_per_pixel, base + Int32(1), NTuple{4, Float32})

        lambda = Wavelengths(lambda_tuple, pdf_tuple)

        # Convert spectral to response (CIE XYZ or sensor RGB)
        response = spectral_to_xyz(cie_table, L, lambda)
        sensor_rgb = max.(0f0, response) * imaging_ratio

        # Apply maxComponentValue clamping in sensor space (pbrt-v4: before output matrix)
        m = maximum(sensor_rgb)
        if m > max_component_value
            sensor_rgb = sensor_rgb * (max_component_value / m)
        end

        # Convert sensor RGB → output sRGB
        rgb = output_matrix * sensor_rgb

        # Get filter weight for this sample (pbrt-v4 style weighted accumulation)
        weight = filter_weight_per_pixel[pixel_idx]

        # Accumulate weighted RGB (pbrt-v4: pixel.rgbSum[c] += weight * rgb[c])
        rgb_base = (pixel_idx - Int32(1)) * Int32(3)
        Atomix.@atomic pixel_rgb[rgb_base + Int32(1)] += weight * rgb[1]
        Atomix.@atomic pixel_rgb[rgb_base + Int32(2)] += weight * rgb[2]
        Atomix.@atomic pixel_rgb[rgb_base + Int32(3)] += weight * rgb[3]

        # Accumulate weight (pbrt-v4: pixel.weightSum += weight)
        Atomix.@atomic pixel_weight_sum[pixel_idx] += weight
    end
end

"""
    vp_finalize_film_kernel!(...)

Finalize film by dividing weighted RGB sum by weight sum.
Following pbrt-v4's RGBFilm::GetPixelRGB:
- rgb = rgbSum / weightSum (if weightSum != 0)
"""
@kernel inbounds=true function vp_finalize_film_kernel!(
    framebuffer,
    @Const(pixel_rgb),
    @Const(pixel_weight_sum),
    @Const(width::Int32), @Const(height::Int32)
)
    pixel_idx = @index(Global)

    @inbounds if pixel_idx <= width * height
        px = ((pixel_idx - Int32(1)) % width) + Int32(1)
        py = ((pixel_idx - Int32(1)) ÷ width) + Int32(1)

        rgb_base = (pixel_idx - Int32(1)) * Int32(3)
        weight_sum = pixel_weight_sum[pixel_idx]

        # Normalize by weight sum (pbrt-v4 style).
        # Use zero/one relative to weight_sum's type so the kernel stays in Float32
        # when accumulation_eltype=Float32 (the GPU-compatible default).
        if weight_sum > zero(weight_sum)
            inv_weight = one(weight_sum) / weight_sum
            r = Float32(pixel_rgb[rgb_base + Int32(1)] * inv_weight)
            g = Float32(pixel_rgb[rgb_base + Int32(2)] * inv_weight)
            b = Float32(pixel_rgb[rgb_base + Int32(3)] * inv_weight)
        else
            r = 0f0
            g = 0f0
            b = 0f0
        end

        # Store linear HDR values (no clamping, no gamma)
        # Postprocessing will handle tonemapping and gamma correction
        framebuffer[py, px] = RGB{Float32}(r, g, b)
    end
end

# ============================================================================
# Main Render Function
# ============================================================================

"""
    finalize_film!(vp::VolPath, film::Film)

Run the per-pixel `pixel_rgb / pixel_weight_sum` divide and write the result
to `film.framebuffer`.  Called automatically at the end of `render!` unless
`finalize_framebuffer=false` is passed (used by batched callers that finalize
once after all samples).
"""
function finalize_film!(vp::VolPath, film::Film)
    state = vp.state
    state === nothing && return nothing
    plans = state.plans
    plans === nothing && throw(ArgumentError(
        "finalize_film!: no plans yet — the divide is a graph pass, and the graph " *
        "is built by the first `render!`. Render a sample before finalizing."))
    run_finalize!(plans, film.framebuffer)
    return nothing
end

# The barrier `runsample!` is, for the finalize plan: `VolPathState.plans` is
# `Any`, so the framebuffer check and the `run!` would otherwise be dynamic
# reads at this level. Inside, the plan is concrete.
function run_finalize!(plans::VPPlans, fb)
    plans.framebuffer === fb || throw(ArgumentError(
        "finalize_film!: this film is not the one the plans were built for. A pass " *
        "names its target when the plan is compiled, so finalizing into another " *
        "film would write the one that was rendered. Render this film first."))
    Mantle.run!(plans.finalize)
    return nothing
end

"""
    render!(vp::VolPath, scene, film, camera)

Render one iteration/sample using volumetric spectral wavefront path tracing.

This function is allocation-free and renders a single sample, accumulating
results in the film. The iteration index is read from and incremented in
`film.iteration_index`.

For progressive rendering, call this repeatedly. For complete rendering,
use the main call function `(vp::VolPath)(scene, film, camera)` which wraps
this in a loop.
"""
function render!(
    vp::VolPath,
    scene::AbstractScene,
    film::Film,
    camera::Camera;
    # When `false`, skip the per-sample `vp_finalize_film_kernel!` dispatch
    # (which divides accumulated RGB by weight and writes the framebuffer).
    # The caller must run `finalize_film!(vp, film)` itself after the last
    # sample.  For batched offline rendering (RayMakie's `colorbuffer` loop)
    # the intermediate framebuffers are never observed, so skipping all but
    # the last call saves one full-resolution dispatch per sample —
    # measured ~17 % of captured GPU time on killeroo 32 spp.  Defaults to
    # `true` so live-preview callers (Makie's renderloop) keep getting
    # per-sample updates.
    finalize_framebuffer::Bool = true,
)
    img = film.framebuffer
    height, width = size(img)
    backend = KA.get_backend(img)

    # Scene edits announce themselves: the mutators and `sync!` call
    # `notify_scene_changed`, which drops this integrator's plans. The one
    # channel that cannot notify is a mutation of the accel BELOW the scene API
    # (Raycore's `push!`/`delete!` set a flag and know nothing of Hikari), so
    # that flag is read here, once, before `adapt`'s internal `sync!` consumes
    # it. Two field reads; no value is compared.
    scene_dirty(scene) && notify_scene_changed(scene)
    # Registration makes this integrator a listener, so the notifications above
    # reach its plans. One pointer compare on the hit path. A DIFFERENT scene
    # drops the cached adaptation with the registration.
    if vp.listening_to !== scene
        vp.adapted = nothing
        listen_scene!(scene, vp)
        vp.listening_to = scene
    end

    # The adapted scene views, cached on the integrator. One nothing-check on
    # the hit path; the medium detect below reads through it, and a scene
    # mutation drops it with the plans.
    adapted = vp.adapted
    if adapted === nothing
        adapted = adapt_scene_for_render(backend, scene, vp)
        vp.adapted = adapted
    end

    # The medium the camera starts in can change with the camera — a call
    # argument, not an event. One compare on the hit; the detect runs once per
    # distinct position. Before the branch: the build needs the key.
    camera_pos = get_camera_position(camera)
    initial_medium = if vp.initial_medium_camera_pos === camera_pos && vp.initial_medium_key !== nothing
        vp.initial_medium_key::SetKey
    else
        key = detect_initial_medium(backend, adapted.accel, adapted.media_interfaces, camera_pos, vp)::SetKey
        vp.initial_medium_camera_pos = camera_pos
        vp.initial_medium_key = key
        key
    end

    # The reads that remain: plans built for THIS film, holding a ref of THIS
    # camera's type. Everything else — the state, the queues, the build itself
    # — hangs off the rebuild branch, which an event takes (`invalidate!` drops
    # the state) or a first call. The film and the camera are call ARGUMENTS,
    # which cannot announce themselves; scene mutation never comes through here
    # at all.
    plans = vp.state === nothing ? nothing : vp.state.plans
    if plans === nothing || plans.framebuffer !== img || !holdscamera(vp.perrun, camera)
        rebuild_render_state!(vp, scene, film, camera, backend, initial_medium)
        plans = (vp.state::VolPathState).plans
    end

    # The film's count of samples taken is the caller's; the device keeps its
    # own, incremented by the sample plan's first pass (see `resetsamples!`).
    # Zero here is the one thing the host has to say about it: a film that was
    # cleared starts the device's count over.
    film.iteration_index[] == 0 && resetsamples!(vp.perrun)
    film.iteration_index[] += Int32(1)

    # A sample's whole host input to the recorded plans: the per-run values
    # that CHANGED, stored through their refs — nothing at all for a still
    # scene. Through `vp.perrun`, typed as a union over the camera type, so
    # the compare is static and boxes nothing; `plans` is `Any`.
    storechanged!(vp.perrun, camera, initial_medium, vp.filter_params)
    # A sample is one plan: setup, the bounce loop, accumulate. The loop's exit
    # test runs on the device — `Mantle.repeat!` gated on the ray queue's live
    # count — so there is no chunking, no host read of that count, and nothing
    # to wait for between rounds.
    runsample!(plans)

    # Update film: divide weighted sum by weight sum (pbrt-v4 style).
    # The kwarg lets batched callers (RayMakie's colorbuffer loop) skip the
    # intermediate framebuffer writes between samples and only finalize once
    # at the end — saves a 1.4 Mpx dispatch per sample on killeroo.
    if finalize_framebuffer
        finalize_film!(vp, film)
    end

    return nothing
end

"""
    rebuild_render_state!(vp, scene, film, camera, backend, initial_medium)

Build everything a sample needs that does not change between samples: the
`VolPathState`, the per-material queues, the filter sampler's device copy and
the plans themselves. Reached from `render!` on the first sample and after an
invalidation event — never on the reuse path, so there is nothing to check
here; the state-rebuild condition answers the film the plans were just dropped
for.
"""
function rebuild_render_state!(vp::VolPath, scene::AbstractScene, film::Film,
                               camera::Camera, backend, initial_medium)
    img = film.framebuffer
    height, width = size(img)
    adapted = vp.adapted::Scene  # set by render! just above
    accel = adapted.accel
    materials = adapted.materials
    media = adapted.media
    media_interfaces = adapted.media_interfaces
    lights = adapted.lights

    # Note: Rebuild state if lights changed (num_lights mismatch) to update light sampler
    n_lights = length(lights)
    # `hit_area_light_queue` and the per-material typed queues are vestigial
    # ONLY when the HW per-material chit fully owns the surface shading.  On
    # medium-bearing scenes the wavefront medium-survives-to-surface path
    # (`delta-tracking.jl`'s `vp_sample_medium_kernel!`) still uses
    # `enqueue_after_intersection!` to push hits into both queues, so they
    # must be full-capacity even on HW.
    # `!isempty(materials)` for the same reason `trace_pass!` asks it: with no
    # materials there are no closest-hit shaders, so that path is not taken and
    # the passes it would have replaced have to stay in the graph.
    chit_owns_surface = shades_surfaces_inline(accel) && isempty(media) && !isempty(materials)

    has_media = !isempty(media)
    if vp.state === nothing ||
       vp.state.width != width ||
       vp.state.height != height ||
       vp.state.num_lights != n_lights ||
       # Queues allocated as capacity-1 placeholders (no-media medium queues,
       # chit-owns-surface hit/area-light queues) must grow if a scene that
       # needs them shows up on a reused integrator.
       (has_media && Int(vp.state.medium_sample_queue.capacity) < width * height) ||
       (!chit_owns_surface && Int(vp.state.hit_surface_queue.capacity) < width * height)
        # The previous state's regions go back before the new ones are taken, so
        # a resize does not hold two states' worth at once. The comment here
        # used to read "a bare reassign races GC finalizers and doubles peak
        # memory" and the line above it was a `KA.synchronize`; neither is true
        # any more — nothing finalizes, and `free!` retires.
        vp.state === nothing || free!(vp.state)
        # Use original scene.lights (MultiTypeSet) for PowerLightSampler (needs .backend)
        # Ensure SobolRNG has enough bits for progressive rendering:
        # When samples_per_pixel=1 (interactive mode), log2_spp=0 causes sample_idx
        # bits to collide with pixel morton code bits, making every sample identical.
        # Use at least 4096 so log2_spp=12, supporting up to 4096 progressive samples.
        sobol_spp = max(Int(vp.samples_per_pixel), 4096)
        vp.state = VolPathState(backend, width, height, scene.lights;
                                max_depth=vp.max_depth,
                                rr_depth=vp.russian_roulette_depth,
                                scene_radius=world_radius(scene),
                                samples_per_pixel=sobol_spp,
                                sampler_seed=UInt32(0),
                                accumulation_eltype=vp.accumulation_eltype,
                                sensor=vp.sensor,
                                hw_accel=chit_owns_surface,
                                has_media=has_media)
    end
    state = vp.state

    n_pixels = width * height

    # Build (or refresh) the per-material typed queues from the adapted
    # materials set.  On the HW per-material chit path the chit shaders fully
    # shade hits inline and the typed queues are never drained, so we use a
    # capacity=1 placeholder.  SW BVH and HW-with-media still route surface
    # hits through these to avoid the monolithic with_index BSDF switch.
    pmq_capacity = chit_owns_surface ? 1 : n_pixels
    ensure_per_material_queue!(state, materials, pmq_capacity)

    # Adapt filter sampler data to GPU — cache on struct to avoid re-uploading
    if vp.filter_sampler_gpu === nothing
        vp.filter_sampler_gpu = Adapt.adapt(backend, vp.filter_sampler_data)
    end

    # The camera effects flags the kernel used to get as arguments are computed
    # on the device now — with the camera behind a `GPURef` they would need refs
    # of their own, and they are two field reads on a value already in registers.
    rebuild_plans!(state, film, backend,
                   accel, media_interfaces, media, materials, lights,
                   camera,
                   initial_medium, vp.filter_params, vp.filter_sampler_gpu,
                   vp.regularize, Int32(vp.samples_per_pixel),
                   vp.max_component_value, vp.max_depth;
                   chit_owns_surface = chit_owns_surface,
                   has_media = has_media,
                   has_lights = length(lights) > 0)
    # The typed handle to what the plans just made. The union names the camera
    # types a sample can be rendered with; another is a plan the field cannot
    # hold, said here rather than as a TypeError.
    pr = (state.plans::VPPlans).perrun
    pr isa Union{PerRun{PerspectiveCamera}, PerRun{MatrixCamera}} || throw(ArgumentError(
        "VolPath renders with a PerspectiveCamera or a MatrixCamera, not a $(typeof(camera))"))
    vp.perrun = pr
    return nothing
end

"""
    (vp::VolPath)(scene, film, camera)

Render using volumetric spectral wavefront path tracing.

Media are automatically extracted from `MediumInterface` materials during scene
construction and stored in `scene.media`.

The render loop follows pbrt-v4's VolPathIntegrator:
1. Generate camera rays (possibly in a medium)
2. For each bounce:
   a. Delta tracking for rays in media → scatter or pass through
   b. Trace rays to find intersections
   c. Handle escaped rays (environment light)
   d. Handle surface hits (emission + direct lighting + BSDF sampling)
3. Accumulate spectral samples to RGB
"""
function (vp::VolPath)(
    scene::AbstractScene,
    film::Film,
    camera::Camera
)
    # Reset iteration counter for fresh render
    film.iteration_index[] = Int32(0)
    # Clear RGB and weight accumulators for fresh render
    clear!(vp)
    # Render all samples by calling render! repeatedly
    # Flush GPU between samples to avoid driver timeout on long renders.
    # Each sample generates ~100 dispatches; without flushing, N samples
    # accumulate into a single massive GPU submission that can exceed
    # the driver's preemption timeout (DEVICE_LOST).
    backend = KA.get_backend(film.framebuffer)
    dev = mantle_device(backend)
    for _ in 1:vp.samples_per_pixel
        render!(vp, scene, film, camera)
        # Device-wide, and meant to be: this bounds how much work one submission
        # carries, which is a statement about the queue rather than about any one
        # plan. Through Mantle rather than `KA.synchronize` so the wait is one
        # Mantle can see — same reason as the early exit above.
        Mantle.waitidle(dev)
    end

    postprocess!(film)
    return film.postprocess
end
