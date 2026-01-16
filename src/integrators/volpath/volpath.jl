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

    # Film sensor parameters (pbrt-v4 style PixelSensor)
    iso::Float32  # ISO sensitivity (100 = baseline)
    exposure_time::Float32  # Exposure time in seconds (default 1.0)
    white_balance::Float32  # Color temperature in Kelvin (0 = use E illuminant, else adapt to D65)

    # Cached render state
    state::Union{Nothing, VolPathState}
end

"""
    VolPath(; max_depth=8, samples=64, russian_roulette_depth=3, regularize=true,
            material_coherence=:none, max_component_value=Inf32, filter=BoxFilter(),
            iso=100, exposure_time=1.0, white_balance=0)

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
- `iso`: ISO sensitivity (default: 100). Higher values = brighter image.
  Combined with exposure_time to compute imagingRatio = exposure_time * iso / 100
  following pbrt-v4's PixelSensor.
- `exposure_time`: Exposure time in seconds (default: 1.0).
- `white_balance`: Color temperature in Kelvin for white balance (default: 0).
  When 0, uses Equal-Energy (E) illuminant → D65 adaptation.
  When > 0, adapts from specified color temperature → D65 using Bradford chromatic adaptation.
  Common values: 5000 (daylight), 6500 (D65), 3200 (tungsten).
"""
function VolPath(;
    max_depth::Int = 8,
    samples::Int = 64,
    russian_roulette_depth::Int = 3,
    regularize::Bool = true,
    material_coherence::Symbol = :none,
    max_component_value::Real = Inf32,
    filter::AbstractFilter = BoxFilter(),
    iso::Real = 100,
    exposure_time::Real = 1.0,
    white_balance::Real = 0
)
    @assert material_coherence in (:none, :sorted, :per_type) "material_coherence must be :none, :sorted, or :per_type"
    return VolPath(
        Int32(max_depth),
        Int32(samples),
        Int32(russian_roulette_depth),
        regularize,
        material_coherence,
        Float32(max_component_value),
        GPUFilterParams(filter),
        Float32(iso),
        Float32(exposure_time),
        Float32(white_balance),
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
    @Const(filter_params::GPUFilterParams)
)
    idx = @index(Global)
    num_pixels = width * height

    @inbounds if idx <= num_pixels
        pixel_idx = idx - Int32(1)
        x = u_int32(mod(pixel_idx, width)) + Int32(1)
        y = u_int32(div(pixel_idx, width)) + Int32(1)

        # Stratified sampling: deterministic samples based on (pixel, sample_index)
        # Uses R2 quasi-random sequence with per-pixel Cranley-Patterson rotation
        pixel_sample = compute_pixel_sample(Int32(x), Int32(y), sample_idx)
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
# Film Accumulation
# ============================================================================

"""
    vp_accumulate_to_rgb_kernel!(...)

Accumulate spectral radiance to RGB with filter weight support.
Following pbrt-v4's RGBFilm::AddSample and PixelSensor::ToSensorRGB:
- Convert spectral L to XYZ using CIE matching functions
- Apply white balance (Bradford chromatic adaptation from source → D65)
- Convert XYZ to linear sRGB
- Apply imagingRatio = exposure_time * iso / 100 (pbrt-v4 style)
- Apply maxComponentValue clamping for firefly suppression
- Accumulate weighted RGB: rgbSum += weight * rgb
- Accumulate weight: weightSum += weight
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
    @Const(imaging_ratio::Float32),
    # White balance matrix (3x3, row-major: m11,m12,m13,m21,m22,m23,m31,m32,m33)
    @Const(wb_m11::Float32), @Const(wb_m12::Float32), @Const(wb_m13::Float32),
    @Const(wb_m21::Float32), @Const(wb_m22::Float32), @Const(wb_m23::Float32),
    @Const(wb_m31::Float32), @Const(wb_m32::Float32), @Const(wb_m33::Float32)
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

        # Apply white balance (Bradford chromatic adaptation matrix: XYZ → XYZ')
        # This transforms from source illuminant to D65
        X_wb = wb_m11 * X + wb_m12 * Y + wb_m13 * Z
        Y_wb = wb_m21 * X + wb_m22 * Y + wb_m23 * Z
        Z_wb = wb_m31 * X + wb_m32 * Y + wb_m33 * Z

        # Convert XYZ (now in D65 space) to linear sRGB
        R, G, B = xyz_to_linear_srgb(X_wb, Y_wb, Z_wb)

        # Clamp negative values
        R = max(0f0, R)
        G = max(0f0, G)
        B = max(0f0, B)

        # Apply imagingRatio (pbrt-v4: exposure_time * iso / 100)
        # This is the key brightness/ISO scaling
        R *= imaging_ratio
        G *= imaging_ratio
        B *= imaging_ratio

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

    # Generate camera rays with filter sampling (pbrt-v4 style)
    kernel! = vp_generate_camera_rays_kernel!(backend)
    kernel!(
        current_ray_queue(state).items, current_ray_queue(state).size,
        wavelengths_per_pixel, pdf_per_pixel, filter_weight_per_pixel,
        Int32(width), Int32(height),
        camera, sample_idx, initial_medium, vp.filter_params;
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

    # Compute sensor parameters for spectral-to-RGB conversion (pbrt-v4 style PixelSensor)
    # imagingRatio = exposure_time * iso / 100 (pbrt-v4 convention)
    imaging_ratio = vp.exposure_time * vp.iso / 100f0

    # White balance matrix (Bradford chromatic adaptation from source illuminant to D65)
    # If white_balance == 0, use identity (E illuminant, no adaptation needed for E→D65 since
    # our spectral→XYZ already uses CIE standard observer which assumes E illuminant)
    wb_matrix = if vp.white_balance > 0f0
        compute_white_balance_matrix(vp.white_balance)
    else
        # Identity matrix for E illuminant (already D65-referenced via CIE observer)
        @SMatrix Float32[1 0 0; 0 1 0; 0 0 1]
    end

    # Accumulate this sample's spectral radiance to RGB with filter weights (pbrt-v4 style)
    kernel! = vp_accumulate_to_rgb_kernel!(backend)
    kernel!(
        pixel_rgb, pixel_weight_sum, state.pixel_L,
        wavelengths_per_pixel, pdf_per_pixel, filter_weight_per_pixel,
        state.cie_table.cie_x, state.cie_table.cie_y, state.cie_table.cie_z,
        Int32(n_pixels),
        vp.max_component_value,
        imaging_ratio,
        # White balance matrix (3x3, row-major)
        wb_matrix[1,1], wb_matrix[1,2], wb_matrix[1,3],
        wb_matrix[2,1], wb_matrix[2,2], wb_matrix[2,3],
        wb_matrix[3,1], wb_matrix[3,2], wb_matrix[3,3];
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
