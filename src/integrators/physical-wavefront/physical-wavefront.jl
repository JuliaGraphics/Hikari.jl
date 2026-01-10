# PhysicalWavefront Integrator - Full Spectral Path Tracing
# A physically-based wavefront renderer with hero wavelength sampling
# Uses the same interface as WhittedIntegrator: (scene, film, camera)

using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
import KernelAbstractions as KA

# ============================================================================
# Render State
# ============================================================================

"""
    PWRenderState{Arr, Table}

Holds all allocated buffers for PhysicalWavefront rendering.
Reused across frames to avoid allocation overhead.
"""
mutable struct PWRenderState{Arr, Table <: RGBToSpectrumTable}
    # Dimensions
    width::Int32
    height::Int32
    num_pixels::Int32
    max_rays::Int32

    # Spectral accumulation for CURRENT SAMPLE ONLY (4 floats per pixel)
    # Cleared at the start of each sample
    pixel_L::Arr

    # RGB accumulation across ALL SAMPLES (3 floats per pixel)
    # Spectral-to-RGB conversion happens after each sample (like pbrt-v4)
    pixel_rgb::Arr

    # Per-pixel wavelength storage (pbrt-v4 style)
    # Each pixel gets independently sampled wavelengths for proper decorrelation
    # 4 wavelengths + 4 PDFs per pixel = 8 floats per pixel
    wavelengths_per_pixel::Arr  # 4 floats per pixel (lambda values)
    pdf_per_pixel::Arr          # 4 floats per pixel (PDF values)

    # Work queues (double-buffered for ping-pong between bounces)
    ray_queue::PWWorkQueue{PWRayWorkItem}
    ray_queue_next::PWWorkQueue{PWRayWorkItem}

    # Processing queues
    escaped_queue::PWWorkQueue{PWEscapedRayWorkItem}
    hit_light_queue::PWWorkQueue{PWHitAreaLightWorkItem}
    material_queue::PWWorkQueue{PWMaterialEvalWorkItem}
    shadow_queue::PWWorkQueue{PWShadowRayWorkItem}

    # Current wavelengths (kept for compatibility, but per-pixel is preferred)
    current_lambda::Wavelengths

    # RGB to spectrum lookup table (GPU-compatible)
    rgb2spec_table::Table
end

"""
Create render state for PhysicalWavefront.
"""
function create_render_state(
    ArrayType, width::Int32, height::Int32, max_depth::Int32
)
    num_pixels = width * height
    # Allow for path expansion (conservative estimate)
    max_rays = num_pixels * Int32(2)

    # Spectral accumulation for current sample: 4 floats per pixel
    pixel_L = ArrayType(zeros(Float32, num_pixels * Int32(4)))

    # RGB accumulation across all samples: 3 floats per pixel
    pixel_rgb = ArrayType(zeros(Float32, num_pixels * Int32(3)))

    # Per-pixel wavelength storage (pbrt-v4 style): 4 floats per pixel each
    wavelengths_per_pixel = ArrayType(zeros(Float32, num_pixels * Int32(4)))
    pdf_per_pixel = ArrayType(zeros(Float32, num_pixels * Int32(4)))

    # Work queues
    ray_queue = PWWorkQueue{PWRayWorkItem}(max_rays, ArrayType)
    ray_queue_next = PWWorkQueue{PWRayWorkItem}(max_rays, ArrayType)
    escaped_queue = PWWorkQueue{PWEscapedRayWorkItem}(max_rays, ArrayType)
    hit_light_queue = PWWorkQueue{PWHitAreaLightWorkItem}(max_rays, ArrayType)
    material_queue = PWWorkQueue{PWMaterialEvalWorkItem}(max_rays, ArrayType)
    shadow_queue = PWWorkQueue{PWShadowRayWorkItem}(max_rays, ArrayType)

    # Default wavelengths (kept for compatibility)
    lambda = sample_wavelengths_uniform(0.5f0)

    # Load RGB to spectrum table and convert to GPU if needed
    rgb2spec_table_cpu = get_srgb_table()
    rgb2spec_table = to_gpu(ArrayType, rgb2spec_table_cpu)

    return PWRenderState(
        width, height, num_pixels, max_rays,
        pixel_L, pixel_rgb,
        wavelengths_per_pixel, pdf_per_pixel,
        ray_queue, ray_queue_next,
        escaped_queue, hit_light_queue,
        material_queue, shadow_queue,
        lambda,
        rgb2spec_table
    )
end

# ============================================================================
# PhysicalWavefront Integrator
# ============================================================================

"""
    PhysicalWavefront <: Integrator

Full spectral path tracer using hero wavelength sampling and wavefront architecture.

Uses 4 wavelength samples per path for efficient GPU execution. Converts
spectral radiance to RGB at the end using CIE XYZ color matching.

# Fields
- `max_depth`: Maximum path depth (bounces)
- `samples_per_pixel`: Number of samples per pixel
- `use_denoising`: Whether to run denoiser post-process
- `russian_roulette_depth`: Depth at which to start Russian roulette
"""
mutable struct PhysicalWavefront <: Integrator
    max_depth::Int32
    samples_per_pixel::Int32
    use_denoising::Bool
    russian_roulette_depth::Int32

    # Cached render state (lazily allocated)
    state::Union{Nothing, PWRenderState}
end

"""
    PhysicalWavefront(; max_depth=8, samples_per_pixel=64, use_denoising=false)

Create a PhysicalWavefront integrator.
"""
function PhysicalWavefront(;
    max_depth::Int = 8,
    samples_per_pixel::Int = 64,
    use_denoising::Bool = false,
    russian_roulette_depth::Int = 3
)
    return PhysicalWavefront(
        Int32(max_depth),
        Int32(samples_per_pixel),
        use_denoising,
        Int32(russian_roulette_depth),
        nothing
    )
end

# ============================================================================
# Main Rendering Loop
# ============================================================================

"""
    (pw::PhysicalWavefront)(scene::Scene, film::Film, camera::Camera)

Render a scene using spectral wavefront path tracing.
"""
function (pw::PhysicalWavefront)(scene::AbstractScene, film::Film, camera::Camera)
    img = film.framebuffer
    accel = scene.aggregate.accel
    materials = scene.aggregate.materials
    lights = scene.lights
    num_lights = length(lights)

    height, width = size(img)
    backend = KA.get_backend(img)

    # Get array type for allocations (use base type, not fully parameterized)
    ArrayType = typeof(similar(img, Float32, 1)).name.wrapper

    # Allocate or validate render state
    if pw.state === nothing ||
       pw.state.width != width ||
       pw.state.height != height
        pw.state = create_render_state(ArrayType, Int32(width), Int32(height), pw.max_depth)
    end

    state = pw.state

    # Clear RGB accumulator (accumulated across all samples)
    fill!(state.pixel_rgb, 0.0f0)

    # Render each sample
    for sample_idx in 1:pw.samples_per_pixel
        # Clear spectral accumulator for THIS sample only
        pw_clear_film!(backend, state.pixel_L, state.num_pixels)

        # Generate RNG seed for this sample
        rng_seed = UInt32(sample_idx) ⊻ UInt32(0x12345678)

        # Reset work queues
        reset_queue!(backend, state.ray_queue)
        reset_queue!(backend, state.ray_queue_next)
        KA.synchronize(backend)

        # Generate camera rays with PER-PIXEL wavelength sampling (pbrt-v4 style)
        # Each pixel samples its own wavelengths for decorrelated color noise
        pw_generate_camera_rays!(
            backend, state.ray_queue,
            state.wavelengths_per_pixel, state.pdf_per_pixel,
            state.width, state.height,
            camera,
            Int32(sample_idx), rng_seed
        )

        # Path tracing loop
        for depth in 0:(pw.max_depth - 1)
            n_rays = queue_size(state.ray_queue)
            n_rays == 0 && break

            # Reset processing queues
            reset_queue!(backend, state.escaped_queue)
            reset_queue!(backend, state.hit_light_queue)
            reset_queue!(backend, state.material_queue)
            reset_queue!(backend, state.shadow_queue)
            reset_queue!(backend, state.ray_queue_next)
            KA.synchronize(backend)

            # Trace rays and sort into queues
            pw_trace_rays!(
                backend,
                state.escaped_queue, state.hit_light_queue, state.material_queue,
                state.ray_queue,
                accel, materials
            )

            # Populate auxiliary buffers on first bounce (for denoising)
            if depth == 0 && pw.use_denoising
                pw_update_aux_from_material_queue!(
                    backend, film, state.material_queue, materials, state.rgb2spec_table
                )
            end

            # Handle escaped rays (environment light)
            if !isempty_tuple(lights)
                pw_handle_escaped_rays!(
                    backend, state.pixel_L,
                    state.escaped_queue, lights,
                    state.rgb2spec_table
                )
            end

            # Handle rays that hit area lights
            pw_handle_hit_area_lights!(
                backend, state.pixel_L,
                state.hit_light_queue, materials
            )

            # Sample direct lighting (creates shadow rays)
            if !isempty_tuple(lights)
                pw_sample_direct_lighting!(
                    backend, state.shadow_queue, state.material_queue,
                    materials, lights,
                    state.rgb2spec_table
                )

                # Trace shadow rays
                pw_trace_shadow_rays!(
                    backend, state.pixel_L,
                    state.shadow_queue, accel
                )
            end

            # Evaluate materials and spawn continuation rays
            pw_evaluate_materials!(
                backend, state.ray_queue_next, state.pixel_L,
                state.material_queue, materials, state.rgb2spec_table, pw.max_depth
            )

            # Swap ray queues for next bounce
            state.ray_queue, state.ray_queue_next = state.ray_queue_next, state.ray_queue
        end

        # CRITICAL: Convert this sample's spectral radiance to RGB and accumulate
        # This must happen AFTER each sample, using EACH PIXEL's wavelengths
        # (This is how pbrt-v4 does it: film.AddSample converts spectral->RGB per pixel)
        pw_accumulate_sample_to_rgb!(
            backend, state.pixel_rgb, state.pixel_L,
            state.wavelengths_per_pixel, state.pdf_per_pixel,
            state.num_pixels
        )
    end

    # Copy accumulated RGB to film framebuffer, dividing by sample count
    pw_finalize_film!(
        backend, film, state.pixel_rgb, pw.samples_per_pixel
    )

    # Apply sRGB gamma for display
    pw_apply_srgb_gamma!(backend, film)

    # Run denoiser if enabled
    if pw.use_denoising
        # Denoising would be called here
        # denoise!(film, config)
    end

    KA.synchronize(backend)
    return film.postprocess
end

# ============================================================================
# Helper for empty tuple check
# ============================================================================

@propagate_inbounds isempty_tuple(t::Tuple) = length(t) == 0
@propagate_inbounds isempty_tuple(::Tuple{}) = true

# ============================================================================
# Alternative: Single-Sample Render for Interactive Preview
# ============================================================================

"""
    render_single_sample!(pw::PhysicalWavefront, scene, film, camera, sample_idx)

Render a single sample and accumulate to film.
Useful for progressive/interactive rendering.
"""
function render_single_sample!(
    pw::PhysicalWavefront,
    scene::Scene,
    film::Film,
    camera::Camera,
    sample_idx::Int32
)
    img = film.framebuffer
    accel = scene.aggregate.accel
    materials = scene.aggregate.materials
    lights = scene.lights

    height, width = size(img)
    backend = KA.get_backend(img)

    # Ensure state exists
    if pw.state === nothing
        ArrayType = typeof(similar(img, Float32, 1))
        pw.state = create_render_state(ArrayType, Int32(width), Int32(height), pw.max_depth)
    end

    state = pw.state

    # Sample wavelengths using importance sampling
    u_lambda = Float32(sample_idx - 1 + rand()) / Float32(pw.samples_per_pixel)
    state.current_lambda = sample_wavelengths_visible(u_lambda)

    rng_seed = UInt32(sample_idx) ⊻ UInt32(0x12345678)

    # Reset queues
    reset_queue!(backend, state.ray_queue)
    KA.synchronize(backend)

    # Generate camera rays
    pw_generate_camera_rays!(
        backend, state.ray_queue,
        camera, film, state.current_lambda,
        rng_seed, sample_idx
    )

    # Path tracing loop
    for depth in 0:(pw.max_depth - 1)
        n_rays = queue_size(state.ray_queue)
        n_rays == 0 && break

        reset_queue!(backend, state.escaped_queue)
        reset_queue!(backend, state.hit_light_queue)
        reset_queue!(backend, state.material_queue)
        reset_queue!(backend, state.shadow_queue)
        reset_queue!(backend, state.ray_queue_next)
        KA.synchronize(backend)

        pw_trace_rays!(
            backend, state.ray_queue,
            state.escaped_queue, state.hit_light_queue, state.material_queue,
            accel, materials
        )

        if !isempty_tuple(lights)
            pw_handle_escaped_rays!(
                backend, state.pixel_L,
                state.escaped_queue, lights,
                state.rgb2spec_table
            )
        end

        pw_handle_hit_area_lights!(
            backend, state.pixel_L,
            state.hit_light_queue, materials, Int32(depth)
        )

        if !isempty_tuple(lights)
            pw_sample_direct_lighting!(
                backend, state.shadow_queue, state.material_queue,
                materials, lights,
                state.rgb2spec_table
            )

            pw_trace_shadow_rays!(
                backend, state.pixel_L,
                state.shadow_queue, accel
            )
        end

        pw_evaluate_materials!(
            backend, state.ray_queue_next, state.pixel_L,
            state.material_queue, materials, pw.max_depth
        )

        state.ray_queue, state.ray_queue_next = state.ray_queue_next, state.ray_queue
    end

    KA.synchronize(backend)
    return nothing
end
