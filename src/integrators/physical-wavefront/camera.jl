# Camera ray generation for PhysicalWavefront
# Generates primary rays with wavelength sampling

# ============================================================================
# Random Number Generation (GPU-compatible)
# ============================================================================

"""
    xorshift32(x::UInt32) -> UInt32

Simple xorshift PRNG for GPU usage.
"""
@inline function xorshift32(x::UInt32)::UInt32
    x = x ⊻ (x << 13)
    x = x ⊻ (x >> 17)
    x = x ⊻ (x << 5)
    return x
end

"""
    random_float(x::UInt32) -> Float32

Convert random UInt32 to Float32 in [0, 1).
"""
@inline function random_float(x::UInt32)::Float32
    return Float32(x) / Float32(typemax(UInt32))
end

# ============================================================================
# Camera Ray Generation Kernel
# ============================================================================

"""
    pw_generate_camera_rays_kernel!(ray_queue_items, ray_queue_size, width, height,
                                     camera, sample_idx, rng_base)

Generate camera rays for all pixels with wavelength sampling.
Each thread generates one camera ray for one pixel.
"""
@kernel function pw_generate_camera_rays_kernel!(
    ray_queue_items,
    ray_queue_size,
    @Const(width::Int32),
    @Const(height::Int32),
    @Const(camera),
    @Const(sample_idx::Int32),
    @Const(rng_base::UInt32),
    @Const(shared_lambda::Wavelengths)  # Shared wavelengths for all rays in this sample
)
    idx = @index(Global)
    num_pixels = width * height

    @inbounds if idx <= num_pixels
        # Convert linear index to pixel coordinates
        # idx-1 to make 0-indexed, then compute x,y
        pixel_idx = idx - Int32(1)
        x = Int32(mod(pixel_idx, width)) + Int32(1)
        y = Int32(div(pixel_idx, width)) + Int32(1)

        # Generate random numbers for this pixel
        seed = xorshift32(rng_base ⊻ UInt32(idx) ⊻ UInt32(sample_idx))
        jitter_x = random_float(seed)
        seed = xorshift32(seed)
        jitter_y = random_float(seed)

        # Compute film position with jitter
        # Flip Y to match Hikari's camera convention (y increases downward in camera space)
        p_film = Point2f(
            Float32(x) - 0.5f0 + jitter_x,
            Float32(height) - Float32(y) + 0.5f0 + jitter_y
        )

        # Create camera sample (no lens sampling for simplicity, could add later)
        camera_sample = CameraSample(p_film, Point2f(0.5f0, 0.5f0), 0f0)

        # Generate ray using Hikari's camera interface
        ray, weight = generate_ray(camera, camera_sample)

        # Use shared wavelengths for all rays in this sample iteration
        # This ensures consistent spectral-to-RGB conversion at film update
        lambda = shared_lambda

        # Only add valid rays
        if weight > 0f0
            # Convert to Raycore.Ray
            raycore_ray = Raycore.Ray(o=ray.o, d=ray.d, t_max=ray.t_max)

            # Create work item
            work_item = PWRayWorkItem(raycore_ray, lambda, Int32(idx))

            # Push to queue atomically
            new_idx = @atomic ray_queue_size[1] += Int32(1)
            ray_queue_items[new_idx] = work_item
        end
    end
end

"""
    pw_generate_camera_rays!(backend, ray_queue, width, height, camera, sample_idx, rng_base, lambda)

Generate camera rays for the current sample pass.
All rays in this sample use the same wavelengths for consistent spectral-to-RGB conversion.
"""
function pw_generate_camera_rays!(
    backend,
    ray_queue::PWWorkQueue{PWRayWorkItem},
    width::Int32,
    height::Int32,
    camera,
    sample_idx::Int32,
    rng_base::UInt32,
    lambda::Wavelengths
)
    # Reset queue
    reset_queue!(backend, ray_queue)

    num_pixels = Int(width * height)
    kernel! = pw_generate_camera_rays_kernel!(backend)

    kernel!(
        ray_queue.items, ray_queue.size,
        width, height,
        camera,
        sample_idx,
        rng_base,
        lambda;
        ndrange=num_pixels
    )

    KernelAbstractions.synchronize(backend)
    return nothing
end

# ============================================================================
# Alternative: Generate rays with stratified wavelength sampling
# ============================================================================

"""
    pw_generate_camera_rays_stratified_kernel!(...)

Generate camera rays with stratified wavelength sampling across samples.
For sample i out of N total, uses stratum i/N of the wavelength space.
"""
@kernel function pw_generate_camera_rays_stratified_kernel!(
    ray_queue_items,
    ray_queue_size,
    @Const(width::Int32),
    @Const(height::Int32),
    @Const(camera),
    @Const(sample_idx::Int32),
    @Const(total_samples::Int32),
    @Const(rng_base::UInt32)
)
    idx = @index(Global)
    num_pixels = width * height

    @inbounds if idx <= num_pixels
        # Convert linear index to pixel coordinates
        pixel_idx = idx - Int32(1)
        x = Int32(mod(pixel_idx, width)) + Int32(1)
        y = Int32(div(pixel_idx, width)) + Int32(1)

        # Generate random numbers
        seed = xorshift32(rng_base ⊻ UInt32(idx) ⊻ UInt32(sample_idx))
        jitter_x = random_float(seed)
        seed = xorshift32(seed)
        jitter_y = random_float(seed)
        seed = xorshift32(seed)
        wavelength_jitter = random_float(seed)

        # Compute film position with jitter
        p_film = Point2f(
            Float32(x) - 0.5f0 + jitter_x,
            Float32(height) - Float32(y) + 0.5f0 + jitter_y
        )

        camera_sample = CameraSample(p_film, Point2f(0.5f0, 0.5f0), 0f0)
        ray, weight = generate_ray(camera, camera_sample)

        # Stratified wavelength sampling
        # Divide wavelength range into strata, sample within current stratum
        stratum_size = 1f0 / Float32(total_samples)
        wavelength_sample = Float32(sample_idx - Int32(1)) * stratum_size + wavelength_jitter * stratum_size
        lambda = sample_wavelengths_uniform(wavelength_sample)

        if weight > 0f0
            raycore_ray = Raycore.Ray(o=ray.o, d=ray.d, t_max=ray.t_max)
            work_item = PWRayWorkItem(raycore_ray, lambda, Int32(idx))

            new_idx = @atomic ray_queue_size[1] += Int32(1)
            ray_queue_items[new_idx] = work_item
        end
    end
end

# ============================================================================
# Pixel Index Utilities
# ============================================================================

"""
    pixel_coords_from_index(idx::Int32, width::Int32) -> (x::Int32, y::Int32)

Convert linear pixel index to x,y coordinates (1-based).
"""
@inline function pixel_coords_from_index(idx::Int32, width::Int32)
    pixel_idx = idx - Int32(1)
    x = Int32(mod(pixel_idx, width)) + Int32(1)
    y = Int32(div(pixel_idx, width)) + Int32(1)
    return (x, y)
end

"""
    pixel_index_from_coords(x::Int32, y::Int32, width::Int32) -> Int32

Convert x,y coordinates to linear pixel index (1-based).
"""
@inline function pixel_index_from_coords(x::Int32, y::Int32, width::Int32)
    return (y - Int32(1)) * width + x
end
