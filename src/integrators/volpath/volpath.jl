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
    material_coherence::Symbol  # Material evaluation mode: :none, :sorted, :per_type
    max_component_value::Float32  # Firefly suppression: clamp RGB components (pbrt-v4 style)
    filter_params::GPUFilterParams  # Filter for pixel reconstruction (pbrt-v4 style)

    # Cached render state
    state::Union{Nothing, VolPathState}
end

"""
    VolPath(; max_depth=8, samples=64, russian_roulette_depth=3, regularize=true,
            material_coherence=:none, max_component_value=Inf32, filter=BoxFilter())

Create a VolPath integrator for volumetric path tracing.

# Arguments
- `max_depth`: Maximum path depth
- `samples`: Number of samples per pixel
- `russian_roulette_depth`: Depth at which to start Russian roulette
- `regularize`: Apply BSDF regularization to reduce fireflies (default: true).
  When enabled, near-specular BSDFs are roughened after the first non-specular
  bounce, following pbrt-v4's approach.
- `material_coherence`: Material evaluation strategy for GPU coherence (default: :none):
  - `:none`: Standard evaluation (baseline)
  - `:sorted`: Sort work items by material type before evaluation
  - `:per_type`: Launch separate kernels per material type (pbrt-v4 style)
- `max_component_value`: Maximum RGB component value before clamping (default: Inf32).
  When set to a finite value, RGB values are scaled down if any component exceeds this.
  This is pbrt-v4's firefly suppression mechanism. Try values like 10.0 or 100.0.
- `filter`: Pixel reconstruction filter (default: BoxFilter()).
  Supports BoxFilter, TriangleFilter, GaussianFilter, MitchellFilter, LanczosSincFilter.
  Filter determines how samples contribute to pixel values based on their position.

Note: Sensor simulation (ISO, exposure_time, white_balance) is handled in postprocessing
via `FilmSensor`, not in the integrator. This matches pbrt-v4's architecture where the
film stores raw linear HDR values and sensor conversion happens at output time.
"""
function VolPath(;
    max_depth::Int = 8,
    samples::Int = 64,
    russian_roulette_depth::Int = 3,
    regularize::Bool = true,
    material_coherence::Symbol = :none,
    max_component_value::Real = Inf32,
    filter::AbstractFilter = BoxFilter()
)
    @assert material_coherence in (:none, :sorted, :per_type) "material_coherence must be :none, :sorted, :per_type"
    return VolPath(
        Int32(max_depth),
        Int32(samples),
        Int32(russian_roulette_depth),
        regularize,
        material_coherence,
        Float32(max_component_value),
        GPUFilterParams(filter),
        nothing
    )
end

"""
    clear!(integrator::VolPath)

Clear the integrator's internal state (RGB and weight accumulators) for restarting progressive rendering.
"""
function Hikari.clear!(vp::VolPath)
    if vp.state !== nothing
        KernelAbstractions.fill!(vp.state.pixel_rgb, 0f0)
        KernelAbstractions.fill!(vp.state.pixel_weight_sum, 0f0)
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
    ray_items, ray_size,
    wavelengths_per_pixel,
    pdf_per_pixel,
    filter_weight_per_pixel,
    @Const(width::Int32), @Const(height::Int32),
    @Const(camera),
    @Const(sample_idx::Int32),
    @Const(initial_medium_idx),
    @Const(filter_params::GPUFilterParams),
    @Const(log2_spp::Int32),
    @Const(n_base4_digits::Int32),
    @Const(sampler_seed::UInt32),
    @Const(sobol_matrices)
)
    idx = @index(Global)
    num_pixels = width * height

    @inbounds if idx <= num_pixels
        pixel_idx = idx - Int32(1)
        x = u_int32(mod(pixel_idx, width)) + Int32(1)
        y = u_int32(div(pixel_idx, width)) + Int32(1)

        # ZSobol sampling: deterministic low-discrepancy samples based on (pixel, sample_index)
        # Matches PBRT-v4's ZSobolSampler for better convergence
        pixel_sample = compute_pixel_sample_sobol(Int32(x), Int32(y), sample_idx, log2_spp, n_base4_digits, sampler_seed, sobol_matrices)
        wavelength_u = pixel_sample.wavelength_u

        # Sample the filter to get offset and weight (pbrt-v4 style)
        # The filter sample uses the pixel jitter values as the 2D random input
        filter_u = Point2f(pixel_sample.jitter_x, pixel_sample.jitter_y)
        filter_sample = gpu_filter_sample(filter_params, filter_u)

        # Store filter weight for this sample (used in accumulation)
        filter_weight_per_pixel[idx] = filter_sample.weight

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
        p_film = Point2f(
            Float32(x) + filter_sample.p[1],
            Float32(height) - Float32(y) + 1f0 + filter_sample.p[2]
        )

        camera_sample = CameraSample(p_film, Point2f(0.5f0, 0.5f0), 0f0)
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

            new_idx = @atomic ray_size[1] += Int32(1)
            ray_items[new_idx] = work_item
        end
    end
end

# ============================================================================
# Ray Sample Generation (pbrt-v4 RaySamples / PixelSampleState)
# ============================================================================

"""
    vp_generate_ray_samples_kernel!(...)

Generate pre-computed Sobol samples for the current bounce.
Following pbrt-v4's WavefrontPathIntegrator::GenerateRaySamples:
- For each active ray in the queue, generate 7 samples for this bounce
- Samples are stored in pixel_samples indexed by pixel_index
- Dimension allocation: 6 (camera) + 7 * depth

This replaces rand() calls with correlated low-discrepancy samples.
"""
@kernel inbounds=true function vp_generate_ray_samples_kernel!(
    # Output: pixel samples (SOA layout)
    pixel_samples_direct_uc,
    pixel_samples_direct_u,
    pixel_samples_indirect_uc,
    pixel_samples_indirect_u,
    pixel_samples_indirect_rr,
    # Input: ray queue (AOS - read pixel_index from each work item)
    @Const(ray_queue_items),
    @Const(ray_queue_size),
    # Sampler parameters
    @Const(sample_idx::Int32),
    @Const(depth::Int32),
    @Const(width::Int32),
    @Const(sobol_matrices),
    @Const(sampler_seed::UInt32),
    @Const(log2_spp::Int32),
    @Const(n_base4_digits::Int32)
)
    i = @index(Global)
    n_rays = ray_queue_size[1]

    @inbounds if i <= n_rays
        # Read pixel_index from work item (AOS layout)
        work = ray_queue_items[i]
        pixel_index = work.pixel_index

        # Recover pixel coordinates from pixel_index
        # pixel_index is 1-based, formula: pixel_index = (y-1) * width + x
        pixel_idx_0 = pixel_index - Int32(1)
        px = u_int32(mod(pixel_idx_0, width)) + Int32(1)
        py = u_int32(div(pixel_idx_0, width)) + Int32(1)

        # Base dimension: 6 (camera) + 7 * depth
        # Camera uses dims 0-5: wavelength(1), film_x(1), film_y(1), lens_x(1), lens_y(1), filter(1)
        base_dim = Int32(6) + Int32(7) * depth

        # Generate 7 samples for this bounce using ZSobol sampler
        # Direct lighting: light selection (1D) + light position (2D)
        direct_uc = zsobol_sample_1d(px, py, sample_idx, base_dim, log2_spp, n_base4_digits, sampler_seed, sobol_matrices)
        direct_u = zsobol_sample_2d(px, py, sample_idx, base_dim + Int32(1), log2_spp, n_base4_digits, sampler_seed, sobol_matrices)

        # Indirect: BSDF component (1D) + direction (2D) + Russian roulette (1D)
        indirect_uc = zsobol_sample_1d(px, py, sample_idx, base_dim + Int32(3), log2_spp, n_base4_digits, sampler_seed, sobol_matrices)
        indirect_u = zsobol_sample_2d(px, py, sample_idx, base_dim + Int32(4), log2_spp, n_base4_digits, sampler_seed, sobol_matrices)
        indirect_rr = zsobol_sample_1d(px, py, sample_idx, base_dim + Int32(6), log2_spp, n_base4_digits, sampler_seed, sobol_matrices)

        # Store in pixel samples buffer (SOA layout)
        pixel_samples_direct_uc[pixel_index] = direct_uc
        pixel_samples_direct_u[pixel_index] = Point2f(direct_u[1], direct_u[2])
        pixel_samples_indirect_uc[pixel_index] = indirect_uc
        pixel_samples_indirect_u[pixel_index] = Point2f(indirect_u[1], indirect_u[2])
        pixel_samples_indirect_rr[pixel_index] = indirect_rr
    end
end

"""
    vp_generate_ray_samples!(backend, state, sample_idx, depth, sobol_matrices, log2_spp, n_base4_digits, sampler_seed)

Generate pre-computed Sobol samples for all active rays at the current depth.
"""
function vp_generate_ray_samples!(
    backend,
    state::VolPathState,
    sample_idx::Int32,
    depth::Int32,
    sobol_matrices,
    log2_spp::Int32,
    n_base4_digits::Int32,
    sampler_seed::UInt32
)
    ray_queue = current_ray_queue(state)
    n_rays = queue_size(ray_queue)
    n_rays == 0 && return

    # Access SOA components of pixel_samples
    pixel_samples = state.pixel_samples

    kernel! = vp_generate_ray_samples_kernel!(backend)
    kernel!(
        pixel_samples.direct_uc,
        pixel_samples.direct_u,
        pixel_samples.indirect_uc,
        pixel_samples.indirect_u,
        pixel_samples.indirect_rr,
        ray_queue.items,  # Pass full items array (AOS), kernel will read pixel_index
        ray_queue.size,
        sample_idx,
        depth,
        state.width,
        sobol_matrices,
        sampler_seed,
        log2_spp,
        n_base4_digits;
        ndrange=n_rays
    )
end

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

Note: Sensor simulation (imaging_ratio, white balance) is applied in postprocessing,
not here. This matches pbrt-v4's architecture where the film stores raw linear HDR
values and the sensor conversion happens at output time.
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
    @Const(max_component_value::Float32)
)
    pixel_idx = @index(Global)

    # Reconstruct CIE table from passed arrays
    cie_table = CIEXYZTable(cie_x, cie_y, cie_z)

    @inbounds if pixel_idx <= n_pixels
        base = (pixel_idx - Int32(1)) * Int32(4)

        L = SpectralRadiance(
            pixel_L[base + Int32(1)],
            pixel_L[base + Int32(2)],
            pixel_L[base + Int32(3)],
            pixel_L[base + Int32(4)]
        )

        lambda_tuple = (
            wavelengths_per_pixel[base + Int32(1)],
            wavelengths_per_pixel[base + Int32(2)],
            wavelengths_per_pixel[base + Int32(3)],
            wavelengths_per_pixel[base + Int32(4)]
        )

        pdf_tuple = (
            pdf_per_pixel[base + Int32(1)],
            pdf_per_pixel[base + Int32(2)],
            pdf_per_pixel[base + Int32(3)],
            pdf_per_pixel[base + Int32(4)]
        )

        lambda = Wavelengths(lambda_tuple, pdf_tuple)

        # Convert spectral to XYZ
        X, Y, Z = spectral_to_xyz(cie_table, L, lambda)

        # Convert XYZ to linear sRGB (no sensor white balance - that's in postprocessing)
        R, G, B = xyz_to_linear_srgb(X, Y, Z)

        # Clamp negative values
        R = max(0f0, R)
        G = max(0f0, G)
        B = max(0f0, B)

        # Apply maxComponentValue clamping (pbrt-v4 firefly suppression)
        # This preserves color hue while preventing extremely bright values
        m = max(R, max(G, B))
        if m > max_component_value
            scale = max_component_value / m
            R *= scale
            G *= scale
            B *= scale
        end

        # Get filter weight for this sample (pbrt-v4 style weighted accumulation)
        weight = filter_weight_per_pixel[pixel_idx]

        # Accumulate weighted RGB (pbrt-v4: pixel.rgbSum[c] += weight * rgb[c])
        rgb_base = (pixel_idx - Int32(1)) * Int32(3)
        Atomix.@atomic pixel_rgb[rgb_base + Int32(1)] += weight * R
        Atomix.@atomic pixel_rgb[rgb_base + Int32(2)] += weight * G
        Atomix.@atomic pixel_rgb[rgb_base + Int32(3)] += weight * B

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

        # Normalize by weight sum (pbrt-v4 style)
        if weight_sum > 0f0
            inv_weight = 1f0 / weight_sum
            r = pixel_rgb[rgb_base + Int32(1)] * inv_weight
            g = pixel_rgb[rgb_base + Int32(2)] * inv_weight
            b = pixel_rgb[rgb_base + Int32(3)] * inv_weight
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

@propagate_inbounds function linear_to_srgb(x::Float32)
    x = clamp(x, 0f0, 1f0)
    if x <= 0.0031308f0
        return 12.92f0 * x
    else
        return 1.055f0 * x^0.41666666f0 - 0.055f0
    end
end

# ============================================================================
# Main Render Function
# ============================================================================

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
    camera::Camera
)
    img = film.framebuffer
    accel = scene.aggregate.accel
    materials = scene.aggregate.materials
    textures = get_textures(scene.aggregate)
    media = scene.aggregate.media
    lights = scene.lights

    height, width = size(img)
    backend = KA.get_backend(img)

    # Allocate or validate state
    # Note: Rebuild state if lights changed (num_lights mismatch) to update light sampler
    n_lights = count_lights(lights)
    if vp.state === nothing ||
       vp.state.width != width ||
       vp.state.height != height ||
       vp.state.num_lights != n_lights
        # Pass scene_radius for proper light power estimation (pbrt-v4 style)
        vp.state = VolPathState(backend, width, height, lights;
                                max_depth=vp.max_depth,
                                scene_radius=scene.world_radius)
    end
    state = vp.state

    n_pixels = width * height

    # Get current iteration index and increment
    sample_idx = film.iteration_index[] + Int32(1)
    film.iteration_index[] = sample_idx

    # Compute ZSobol sampler parameters (matching PBRT-v4)
    log2_spp, n_base4_digits = compute_zsobol_params(Int(vp.samples_per_pixel), width, height)
    sampler_seed = UInt32(0)  # Can be varied for different render seeds

    sobol_matrices_gpu = KernelAbstractions.allocate(backend, UInt32, length(SobolMatrices32))
    KernelAbstractions.copyto!(backend, sobol_matrices_gpu, SobolMatrices32)

    # Get accumulators from state (allocation-free)
    pixel_rgb = state.pixel_rgb
    pixel_weight_sum = state.pixel_weight_sum
    wavelengths_per_pixel = state.wavelengths_per_pixel
    pdf_per_pixel = state.pdf_per_pixel
    filter_weight_per_pixel = state.filter_weight_per_pixel

    # Initial medium (vacuum unless camera is inside medium)
    initial_medium = MediumIndex()

    # Clear spectral buffer (pixel_L) for this sample iteration
    # This is per-sample, not per-render - pixel_rgb accumulates across all samples
    reset_film!(state)

    # Reset ray queue
    reset_queue!(backend, current_ray_queue(state))

    # Generate camera rays with filter sampling (pbrt-v4 style) and ZSobol sampler
    kernel! = vp_generate_camera_rays_kernel!(backend)
    kernel!(
        current_ray_queue(state).items, current_ray_queue(state).size,
        wavelengths_per_pixel, pdf_per_pixel, filter_weight_per_pixel,
        Int32(width), Int32(height),
        camera, sample_idx, initial_medium, vp.filter_params,
        log2_spp, n_base4_digits, sampler_seed, sobol_matrices_gpu;
        ndrange=Int(n_pixels)
    )

    # Ensure multi-material queue is allocated in state for :per_type mode
    if vp.material_coherence == :per_type
        N = length(materials)
        if state.multi_material_queue === nothing ||
           length(state.multi_material_queue.queues) != N
            state.multi_material_queue = MultiMaterialQueue{N}(backend, state.material_queue.capacity)
        end
    end
    multi_queue = state.multi_material_queue

    # Path tracing loop - following pbrt-v4 wavefront architecture
    for depth in 0:(vp.max_depth - 1)
        n_rays = queue_size(current_ray_queue(state))
        n_rays == 0 && break

        # Generate pre-computed Sobol samples for this bounce (pbrt-v4 RaySamples pattern)
        # Must be called BEFORE any kernel that uses pixel_samples
        vp_generate_ray_samples!(backend, state, sample_idx, Int32(depth),
                                 sobol_matrices_gpu, log2_spp, n_base4_digits, sampler_seed)

        reset_iteration_queues!(state)
        vp_trace_rays!(backend, state, accel)

        if !isempty(media)
            n_medium = queue_size(state.medium_sample_queue)
            if n_medium > 0
                vp_sample_medium_interaction!(backend, state, media)
            end
        end

        if !isempty(media)
            n_scatter = queue_size(state.medium_scatter_queue)
            if n_scatter > 0
                if count_lights(lights) > 0
                    vp_sample_medium_direct_lighting!(backend, state, lights, media)
                end
                vp_sample_medium_scatter!(backend, state)
            end
        end

        n_escaped = queue_size(state.escaped_queue)
        if n_escaped > 0 && count_lights(lights) > 0
            vp_handle_escaped_rays!(backend, state, lights)
        end

        n_hits = queue_size(state.hit_surface_queue)
        if n_hits > 0
            if vp.material_coherence == :per_type && multi_queue !== nothing
                reset_queues!(backend, multi_queue)
                vp_process_surface_hits_coherent!(backend, state, multi_queue, materials, textures)

                # Following pbrt-v4: launch kernels unconditionally, let them check size internally
                # This avoids expensive GPU→CPU sync from queue_size() / total_size() calls
                if count_lights(lights) > 0
                    vp_sample_direct_lighting_coherent!(backend, state, multi_queue, materials, textures, lights)
                end

                # Shadow rays still need size check since it's a different queue
                # But we can batch this with other checks later
                vp_trace_shadow_rays!(backend, state, accel, materials, media)

                vp_evaluate_materials_coherent!(backend, state, multi_queue, materials, textures, media, vp.regularize)
            else
                vp_process_surface_hits!(backend, state, materials, textures)

                n_material = queue_size(state.material_queue)
                if n_material > 0 && count_lights(lights) > 0
                    vp_sample_surface_direct_lighting!(backend, state, materials, textures, lights)
                end

                n_shadow = queue_size(state.shadow_queue)
                if n_shadow > 0
                    vp_trace_shadow_rays!(backend, state, accel, materials, media)
                end

                if n_material > 0
                    if vp.material_coherence == :sorted
                        vp_evaluate_materials_sorted!(backend, state, materials, textures, media, vp.regularize)
                    else
                        vp_evaluate_materials!(backend, state, materials, textures, media, vp.regularize)
                    end
                end
            end
        end

        swap_ray_queues!(state)
    end

    # Accumulate this sample's spectral radiance to RGB with filter weights (pbrt-v4 style)
    # Note: Sensor simulation (imaging_ratio, white balance) is applied in postprocessing,
    # not here. The integrator outputs raw linear HDR values.
    kernel! = vp_accumulate_to_rgb_kernel!(backend)
    kernel!(
        pixel_rgb, pixel_weight_sum, state.pixel_L,
        wavelengths_per_pixel, pdf_per_pixel, filter_weight_per_pixel,
        state.cie_table.cie_x, state.cie_table.cie_y, state.cie_table.cie_z,
        Int32(n_pixels),
        vp.max_component_value;
        ndrange=Int(n_pixels)
    )

    # Update film: divide weighted sum by weight sum (pbrt-v4 style)
    kernel! = vp_finalize_film_kernel!(backend)
    kernel!(
        img, pixel_rgb, pixel_weight_sum,
        Int32(width), Int32(height);
        ndrange=Int(n_pixels)
    )

    return nothing
end

"""
    (vp::VolPath)(scene, film, camera)

Render using volumetric spectral wavefront path tracing.

Media are automatically extracted from `MediumInterface` materials during scene
construction and stored in `scene.aggregate.media`.

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
    if vp.state !== nothing
        backend = KA.get_backend(film.framebuffer)
        KA.fill!(vp.state.pixel_rgb, 0f0)
        KA.fill!(vp.state.pixel_weight_sum, 0f0)
    end

    # Render all samples by calling render! repeatedly
    for _ in 1:vp.samples_per_pixel
        render!(vp, scene, film, camera)
    end

    return film.postprocess
end
