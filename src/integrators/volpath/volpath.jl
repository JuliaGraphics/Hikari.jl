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

    # Cached render state
    state::Union{Nothing, VolPathState}
end

"""
    VolPath(; max_depth=8, samples_per_pixel=64, russian_roulette_depth=3)

Create a VolPath integrator for volumetric path tracing.
"""
function VolPath(;
    max_depth::Int = 8,
    samples_per_pixel::Int = 64,
    russian_roulette_depth::Int = 3
)
    return VolPath(
        Int32(max_depth),
        Int32(samples_per_pixel),
        Int32(russian_roulette_depth),
        nothing
    )
end

"""
    clear!(integrator::VolPath)

Clear the integrator's internal state (RGB accumulator) for restarting progressive rendering.
"""
function Hikari.clear!(vp::VolPath)
    if vp.state !== nothing
        KernelAbstractions.fill!(vp.state.pixel_rgb, 0f0)
    end
end

# ============================================================================
# Camera Ray Generation
# ============================================================================

"""
    vp_generate_camera_rays_kernel!(...)

Generate camera rays with per-pixel wavelength sampling.
"""
@kernel inbounds=true function vp_generate_camera_rays_kernel!(
    ray_items, ray_size,
    wavelengths_per_pixel,
    pdf_per_pixel,
    @Const(width::Int32), @Const(height::Int32),
    @Const(camera),
    @Const(sample_idx::Int32),
    @Const(initial_medium_idx)
)
    idx = @index(Global)
    num_pixels = width * height

    @inbounds if idx <= num_pixels
        pixel_idx = idx - Int32(1)
        x = u_int32(mod(pixel_idx, width)) + Int32(1)
        y = u_int32(div(pixel_idx, width)) + Int32(1)

        # Random jitter and wavelength sample
        jitter_x = rand(Float32)
        jitter_y = rand(Float32)
        wavelength_u = rand(Float32)

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

        # Film position with jitter (flip Y for camera convention)
        p_film = Point2f(
            Float32(x) - 0.5f0 + jitter_x,
            Float32(height) - Float32(y) + 0.5f0 + jitter_y
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

@kernel inbounds=true function vp_accumulate_to_rgb_kernel!(
    pixel_rgb,
    @Const(pixel_L),
    @Const(wavelengths_per_pixel),
    @Const(pdf_per_pixel),
    @Const(cie_x), @Const(cie_y), @Const(cie_z),
    @Const(n_pixels::Int32)
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

        # Convert spectral to linear RGB
        R, G, B = spectral_to_linear_rgb(cie_table, L, lambda)

        # Accumulate
        rgb_base = (pixel_idx - Int32(1)) * Int32(3)
        Atomix.@atomic pixel_rgb[rgb_base + Int32(1)] += R
        Atomix.@atomic pixel_rgb[rgb_base + Int32(2)] += G
        Atomix.@atomic pixel_rgb[rgb_base + Int32(3)] += B
    end
end

@kernel inbounds=true function vp_finalize_film_kernel!(
    framebuffer,
    @Const(pixel_rgb),
    @Const(samples::Int32),
    @Const(width::Int32), @Const(height::Int32)
)
    pixel_idx = @index(Global)

    @inbounds if pixel_idx <= width * height
        px = ((pixel_idx - Int32(1)) % width) + Int32(1)
        py = ((pixel_idx - Int32(1)) ÷ width) + Int32(1)

        rgb_base = (pixel_idx - Int32(1)) * Int32(3)
        r = pixel_rgb[rgb_base + Int32(1)] / Float32(samples)
        g = pixel_rgb[rgb_base + Int32(2)] / Float32(samples)
        b = pixel_rgb[rgb_base + Int32(3)] / Float32(samples)

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
    if vp.state === nothing ||
       vp.state.width != width ||
       vp.state.height != height
        vp.state = VolPathState(backend, width, height; max_depth=vp.max_depth)
    end
    state = vp.state

    n_pixels = width * height

    # Get current iteration index and increment
    sample_idx = film.iteration_index[] + Int32(1)
    film.iteration_index[] = sample_idx

    # Get accumulators from state (allocation-free)
    pixel_rgb = state.pixel_rgb
    wavelengths_per_pixel = state.wavelengths_per_pixel
    pdf_per_pixel = state.pdf_per_pixel

    # Initial medium (vacuum unless camera is inside medium)
    initial_medium = MediumIndex()

    # Clear spectral buffer (pixel_L) for this sample iteration
    # This is per-sample, not per-render - pixel_rgb accumulates across all samples
    reset_film!(state)

    # Reset ray queue
    reset_queue!(backend, current_ray_queue(state))

    # Generate camera rays
    kernel! = vp_generate_camera_rays_kernel!(backend)
    kernel!(
        current_ray_queue(state).items, current_ray_queue(state).size,
        wavelengths_per_pixel, pdf_per_pixel,
        Int32(width), Int32(height),
        camera, sample_idx, initial_medium;
        ndrange=Int(n_pixels)
    )

    # Path tracing loop - following pbrt-v4 wavefront architecture
    for depth in 0:(vp.max_depth - 1)
        n_rays = queue_size(current_ray_queue(state))
        n_rays == 0 && break

        # Reset work queues for this iteration
        reset_iteration_queues!(state)

        # Step 1: Trace rays to find intersections
        vp_trace_rays!(backend, state, accel)

        # Step 2: Sample medium interactions (delta tracking)
        if !isempty(media)
            n_medium = queue_size(state.medium_sample_queue)
            if n_medium > 0
                vp_sample_medium_interaction!(backend, state, media)
            end
        end

        # Step 3: Handle medium scattering events
        if !isempty(media)
            n_scatter = queue_size(state.medium_scatter_queue)
            if n_scatter > 0
                if count_lights(lights) > 0
                    vp_sample_medium_direct_lighting!(backend, state, lights, media)
                end
                vp_sample_medium_scatter!(backend, state)
            end
        end

        # Step 4: Handle escaped rays (environment light)
        n_escaped = queue_size(state.escaped_queue)
        if n_escaped > 0 && count_lights(lights) > 0
            vp_handle_escaped_rays!(backend, state, lights)
        end

        # Step 5: Process surface hits (emission + material eval)
        n_hits = queue_size(state.hit_surface_queue)
        if n_hits > 0
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
                vp_evaluate_materials!(backend, state, materials, textures, media)
            end
        end

        # Swap ray queues for next iteration
        swap_ray_queues!(state)
    end

    # Accumulate this sample's spectral radiance to RGB
    kernel! = vp_accumulate_to_rgb_kernel!(backend)
    kernel!(
        pixel_rgb, state.pixel_L,
        wavelengths_per_pixel, pdf_per_pixel,
        state.cie_table.cie_x, state.cie_table.cie_y, state.cie_table.cie_z,
        Int32(n_pixels);
        ndrange=Int(n_pixels)
    )

    # Update film: divide by current sample count
    kernel! = vp_finalize_film_kernel!(backend)
    kernel!(
        img, pixel_rgb,
        sample_idx,  # Current number of samples
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

    # Clear RGB accumulator for fresh render
    if vp.state !== nothing
        backend = KA.get_backend(film.framebuffer)
        KA.fill!(vp.state.pixel_rgb, 0f0)
    end

    # Render all samples by calling render! repeatedly
    for _ in 1:vp.samples_per_pixel
        render!(vp, scene, film, camera)
    end

    return film.postprocess
end

# Old implementation (kept for reference, can be removed):
function _old_volpath_implementation(vp::VolPath, scene::AbstractScene, film::Film, camera::Camera)
    img = film.framebuffer
    accel = scene.aggregate.accel
    materials = scene.aggregate.materials
    # Textures tuple for GPU compatibility (empty on CPU where Texture structs hold data)
    textures = get_textures(scene.aggregate)
    media = scene.aggregate.media  # Get media from scene (extracted from MediumInterface materials)
    lights = scene.lights

    height, width = size(img)
    backend = KA.get_backend(img)

    # Allocate or validate state
    if vp.state === nothing ||
       vp.state.width != width ||
       vp.state.height != height
        vp.state = VolPathState(backend, width, height; max_depth=vp.max_depth)
    end
    state = vp.state

    n_pixels = width * height

    # Allocate accumulators
    pixel_rgb = KA.allocate(backend, Float32, n_pixels * 3)
    KA.fill!(pixel_rgb, 0f0)
    wavelengths_per_pixel = KA.allocate(backend, Float32, n_pixels * 4)
    pdf_per_pixel = KA.allocate(backend, Float32, n_pixels * 4)

    # Initial medium (vacuum unless camera is inside medium)
    initial_medium = MediumIndex()

    # Render samples
    for sample_idx in 1:vp.samples_per_pixel
        # Clear spectral film for this sample
        reset_film!(state)

        # Reset ray queue
        reset_queue!(backend, current_ray_queue(state))
        KA.synchronize(backend)

        # Generate camera rays
        kernel! = vp_generate_camera_rays_kernel!(backend)
        kernel!(
            current_ray_queue(state).items, current_ray_queue(state).size,
            wavelengths_per_pixel, pdf_per_pixel,
            Int32(width), Int32(height),
            camera, Int32(sample_idx), initial_medium;
            ndrange=Int(n_pixels)
        )
        KA.synchronize(backend)

        # Path tracing loop - following pbrt-v4 wavefront architecture
        for depth in 0:(vp.max_depth - 1)
            n_rays = queue_size(current_ray_queue(state))
            n_rays == 0 && break

            # Reset work queues for this iteration
            reset_iteration_queues!(state)
            KA.synchronize(backend)

            # ================================================================
            # Step 1: Trace rays to find intersections
            # Routes rays in medium -> medium_sample_queue (with t_max)
            # Routes rays NOT in medium -> hit_surface_queue or escaped_queue
            # ================================================================
            vp_trace_rays!(backend, state, accel)

            # ================================================================
            # Step 2: Sample medium interactions (delta tracking)
            # Processes medium_sample_queue with bounded t_max
            # Outputs: medium_scatter_queue, hit_surface_queue, escaped_queue
            # ================================================================
            if !isempty(media)
                n_medium = queue_size(state.medium_sample_queue)
                if n_medium > 0
                    vp_sample_medium_interaction!(backend, state, media)
                end
            end

            # ================================================================
            # Step 3: Handle medium scattering events
            # Process real scatter events from delta tracking
            # ================================================================
            if !isempty(media)
                n_scatter = queue_size(state.medium_scatter_queue)
                if n_scatter > 0
                    # Direct lighting at scatter points
                    if count_lights(lights) > 0
                        vp_sample_medium_direct_lighting!(backend, state, lights, media)
                    end
                    # Phase function sampling for continuation
                    vp_sample_medium_scatter!(backend, state)
                end
            end

            # ================================================================
            # Step 4: Handle escaped rays (environment light)
            # ================================================================
            n_escaped = queue_size(state.escaped_queue)
            if n_escaped > 0 && count_lights(lights) > 0
                vp_handle_escaped_rays!(backend, state, lights)
            end

            # ================================================================
            # Step 5: Process surface hits (emission + material eval)
            # ================================================================
            n_hits = queue_size(state.hit_surface_queue)
            if n_hits > 0
                vp_process_surface_hits!(backend, state, materials, textures)

                # Step 5a: Direct lighting at surfaces
                n_material = queue_size(state.material_queue)
                if n_material > 0 && count_lights(lights) > 0
                    vp_sample_surface_direct_lighting!(backend, state, materials, textures, lights)
                end

                # Step 5b: Trace shadow rays
                n_shadow = queue_size(state.shadow_queue)
                if n_shadow > 0
                    vp_trace_shadow_rays!(backend, state, accel, materials, media)
                end

                # Step 5c: BSDF sampling for continuation rays
                if n_material > 0
                    vp_evaluate_materials!(backend, state, materials, textures, media)
                end
            end

            # Swap ray queues for next iteration
            swap_ray_queues!(state)
        end

        # Accumulate this sample's spectral radiance to RGB
        kernel! = vp_accumulate_to_rgb_kernel!(backend)
        kernel!(
            pixel_rgb, state.pixel_L,
            wavelengths_per_pixel, pdf_per_pixel,
            state.cie_table.cie_x, state.cie_table.cie_y, state.cie_table.cie_z,
            Int32(n_pixels);
            ndrange=Int(n_pixels)
        )
        KA.synchronize(backend)
    end

    # Finalize: divide by samples and gamma correct
    kernel! = vp_finalize_film_kernel!(backend)
    kernel!(
        img, pixel_rgb,
        vp.samples_per_pixel,
        Int32(width), Int32(height);
        ndrange=Int(n_pixels)
    )

    KA.synchronize(backend)
    return film.postprocess
end

# ============================================================================
# Utilities
# ============================================================================

# isempty_tuple is defined in physical-wavefront.jl
