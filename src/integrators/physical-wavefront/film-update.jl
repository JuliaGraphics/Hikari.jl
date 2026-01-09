# Film update for PhysicalWavefront path tracing
# Converts accumulated spectral radiance to RGB for Film output

# ============================================================================
# Spectral to RGB Conversion Kernel
# ============================================================================

"""
    pw_update_film_spectral_kernel!(framebuffer, pixel_L, wavelengths_per_pixel,
                                     samples_accumulated, width, height)

Convert accumulated spectral radiance to RGB and update film framebuffer.

Each pixel stores 4 spectral values in pixel_L (interleaved), and wavelengths_per_pixel
stores the sampled wavelengths. Uses CIE XYZ color matching for accurate conversion.

Arguments:
- `framebuffer`: Output RGB image (height × width)
- `pixel_L`: Accumulated spectral radiance (4 × num_pixels, interleaved)
- `wavelengths_per_pixel`: Wavelengths for each pixel's samples (4 × num_pixels)
- `pdf_per_pixel`: PDF for each wavelength sample (4 × num_pixels)
- `samples_accumulated`: Number of samples per pixel for averaging
- `width`: Image width
- `height`: Image height
"""
@kernel function pw_update_film_spectral_kernel!(
    framebuffer,
    @Const(pixel_L),
    @Const(wavelengths_per_pixel),
    @Const(pdf_per_pixel),
    @Const(samples_accumulated::Int32),
    @Const(width::Int32),
    @Const(height::Int32)
)
    idx = @index(Global)
    num_pixels = width * height

    @_inbounds if idx <= num_pixels
        # Extract spectral radiance for this pixel
        base = (idx - Int32(1)) * Int32(4)
        L = SpectralRadiance((
            pixel_L[base + Int32(1)],
            pixel_L[base + Int32(2)],
            pixel_L[base + Int32(3)],
            pixel_L[base + Int32(4)]
        ))

        # Extract wavelengths and PDFs for this pixel
        lambda = Wavelengths((
            wavelengths_per_pixel[base + Int32(1)],
            wavelengths_per_pixel[base + Int32(2)],
            wavelengths_per_pixel[base + Int32(3)],
            wavelengths_per_pixel[base + Int32(4)]
        ), (
            pdf_per_pixel[base + Int32(1)],
            pdf_per_pixel[base + Int32(2)],
            pdf_per_pixel[base + Int32(3)],
            pdf_per_pixel[base + Int32(4)]
        ))

        # Convert to linear RGB via XYZ
        R, G, B = spectral_to_linear_rgb(L, lambda)

        # Average by number of samples
        if samples_accumulated > Int32(0)
            inv_samples = 1.0f0 / Float32(samples_accumulated)
            R *= inv_samples
            G *= inv_samples
            B *= inv_samples
        end

        # Clamp negative values and NaN/Inf
        R = ifelse(isfinite(R), max(0.0f0, R), 0.0f0)
        G = ifelse(isfinite(G), max(0.0f0, G), 0.0f0)
        B = ifelse(isfinite(B), max(0.0f0, B), 0.0f0)

        # Convert linear index to 2D coordinates
        # pixel_index uses row-major: idx = (y-1)*width + x, so x varies fastest
        # framebuffer is (height, width) = (rows, cols)
        # So framebuffer[row, col] = framebuffer[y, x]
        pixel_idx = idx - Int32(1)
        x = u_int32(mod(pixel_idx, width)) + Int32(1)   # column (1 to width)
        y = u_int32(div(pixel_idx, width)) + Int32(1)   # row (1 to height)

        framebuffer[y, x] = RGB{Float32}(R, G, B)
    end
end

# ============================================================================
# Simplified Spectral Update (Uniform Wavelength Sampling)
# ============================================================================

"""
    pw_update_film_uniform_kernel!(framebuffer, pixel_L, lambda,
                                    samples_accumulated, width, height)

Convert accumulated spectral radiance to RGB using uniform wavelength sampling.

When using stratified wavelength sampling, all pixels share the same wavelengths
within a sample iteration. This kernel uses a single Wavelengths value for all pixels.
"""
@kernel function pw_update_film_uniform_kernel!(
    framebuffer,
    @Const(pixel_L),
    @Const(lambda::Wavelengths),
    @Const(samples_accumulated::Int32),
    @Const(width::Int32),
    @Const(height::Int32)
)
    idx = @index(Global)
    num_pixels = width * height

    @_inbounds if idx <= num_pixels
        # Extract spectral radiance for this pixel
        base = (idx - Int32(1)) * Int32(4)
        L = SpectralRadiance((
            pixel_L[base + Int32(1)],
            pixel_L[base + Int32(2)],
            pixel_L[base + Int32(3)],
            pixel_L[base + Int32(4)]
        ))

        # Convert to linear RGB via XYZ
        R, G, B = spectral_to_linear_rgb(L, lambda)

        # Average by number of samples
        if samples_accumulated > Int32(0)
            inv_samples = 1.0f0 / Float32(samples_accumulated)
            R *= inv_samples
            G *= inv_samples
            B *= inv_samples
        end

        # Clamp negative values and NaN/Inf
        R = ifelse(isfinite(R), max(0.0f0, R), 0.0f0)
        G = ifelse(isfinite(G), max(0.0f0, G), 0.0f0)
        B = ifelse(isfinite(B), max(0.0f0, B), 0.0f0)

        # Convert linear index to 2D coordinates
        # pixel_index uses row-major: idx = (y-1)*width + x, so x varies fastest
        # framebuffer is (height, width) = (rows, cols)
        # So framebuffer[row, col] = framebuffer[y, x]
        pixel_idx = idx - Int32(1)
        x = u_int32(mod(pixel_idx, width)) + Int32(1)   # column (1 to width)
        y = u_int32(div(pixel_idx, width)) + Int32(1)   # row (1 to height)

        framebuffer[y, x] = RGB{Float32}(R, G, B)
    end
end

# ============================================================================
# Accumulative Film Update (for progressive rendering)
# ============================================================================

"""
    pw_accumulate_to_film_kernel!(pixel_L_accum, pixel_L_sample, num_pixels)

Accumulate sample results into the film accumulator.
Adds new sample's spectral values to existing accumulated values.
"""
@kernel function pw_accumulate_to_film_kernel!(
    pixel_L_accum,
    @Const(pixel_L_sample),
    @Const(num_pixels::Int32)
)
    idx = @index(Global)

    @_inbounds if idx <= num_pixels * Int32(4)
        pixel_L_accum[idx] += pixel_L_sample[idx]
    end
end

# ============================================================================
# Clear Film Kernel
# ============================================================================

"""
    pw_clear_film_kernel!(pixel_L, num_pixels)

Clear accumulated spectral radiance to zero for new render.
"""
@kernel function pw_clear_film_kernel!(
    pixel_L,
    @Const(num_pixels::Int32)
)
    idx = @index(Global)

    @_inbounds if idx <= num_pixels * Int32(4)
        pixel_L[idx] = 0.0f0
    end
end

# ============================================================================
# Per-Sample Spectral-to-RGB Accumulation (pbrt-v4 style with per-pixel wavelengths)
# ============================================================================

"""
    pw_accumulate_sample_to_rgb!(backend, pixel_rgb, pixel_L,
                                  wavelengths_per_pixel, pdf_per_pixel, num_pixels)

Convert this sample's spectral radiance to RGB using PER-PIXEL wavelengths
and accumulate into pixel_rgb.

This is the key operation that pbrt-v4 does: spectral-to-RGB conversion happens
IMMEDIATELY after each sample, using THAT PIXEL's wavelengths. The RGB values
are then accumulated across samples.

Each pixel has independently sampled wavelengths, which decorrelates color noise
and results in much faster convergence than using shared wavelengths.
"""
function pw_accumulate_sample_to_rgb!(
    backend,
    pixel_rgb::AbstractVector{Float32},
    pixel_L::AbstractVector{Float32},
    wavelengths_per_pixel::AbstractVector{Float32},
    pdf_per_pixel::AbstractVector{Float32},
    num_pixels::Int32
)
    # CPU implementation (GPU kernel would be similar)
    pixel_L_cpu = Array(pixel_L)
    pixel_rgb_cpu = Array(pixel_rgb)
    wavelengths_cpu = Array(wavelengths_per_pixel)
    pdf_cpu = Array(pdf_per_pixel)

    @_inbounds for idx in 1:Int(num_pixels)
        # Extract spectral radiance for this pixel
        base = (idx - 1) * 4
        L = SpectralRadiance((
            pixel_L_cpu[base + 1],
            pixel_L_cpu[base + 2],
            pixel_L_cpu[base + 3],
            pixel_L_cpu[base + 4]
        ))

        # Reconstruct THIS PIXEL's wavelengths from stored values
        lambda = Wavelengths(
            (wavelengths_cpu[base + 1], wavelengths_cpu[base + 2],
             wavelengths_cpu[base + 3], wavelengths_cpu[base + 4]),
            (pdf_cpu[base + 1], pdf_cpu[base + 2],
             pdf_cpu[base + 3], pdf_cpu[base + 4])
        )

        # Convert spectral to RGB using THIS PIXEL's wavelengths
        R, G, B = spectral_to_linear_rgb(L, lambda)

        # Clamp negative values and NaN/Inf
        R = ifelse(isfinite(R), max(0.0f0, R), 0.0f0)
        G = ifelse(isfinite(G), max(0.0f0, G), 0.0f0)
        B = ifelse(isfinite(B), max(0.0f0, B), 0.0f0)

        # Accumulate into RGB buffer
        base_rgb = (idx - 1) * 3
        pixel_rgb_cpu[base_rgb + 1] += R
        pixel_rgb_cpu[base_rgb + 2] += G
        pixel_rgb_cpu[base_rgb + 3] += B
    end

    # Copy back (for GPU, this would be done in the kernel)
    copyto!(pixel_rgb, pixel_rgb_cpu)
    return nothing
end

"""
    pw_finalize_film!(backend, film, pixel_rgb, samples_per_pixel)

Copy accumulated RGB values to film framebuffer, dividing by sample count.
"""
function pw_finalize_film!(
    backend,
    film::Film,
    pixel_rgb::AbstractVector{Float32},
    samples_per_pixel::Int32
)
    width = Int32(size(film.framebuffer, 2))
    height = Int32(size(film.framebuffer, 1))
    num_pixels = width * height

    pixel_rgb_cpu = Array(pixel_rgb)
    inv_samples = 1.0f0 / Float32(samples_per_pixel)

    # Build flat RGB output for reshape
    rgb_output = Vector{Float32}(undef, Int(num_pixels) * 3)

    @_inbounds for idx in 1:Int(num_pixels)
        base = (idx - 1) * 3
        R = pixel_rgb_cpu[base + 1] * inv_samples
        G = pixel_rgb_cpu[base + 2] * inv_samples
        B = pixel_rgb_cpu[base + 3] * inv_samples

        # Clamp
        R = ifelse(isfinite(R), max(0.0f0, R), 0.0f0)
        G = ifelse(isfinite(G), max(0.0f0, G), 0.0f0)
        B = ifelse(isfinite(B), max(0.0f0, B), 0.0f0)

        rgb_output[base + 1] = R
        rgb_output[base + 2] = G
        rgb_output[base + 3] = B
    end

    # Reshape like PbrtWavefront does
    rgb_pixels = reinterpret(RGB{Float32}, rgb_output)
    reshaped = reshape(rgb_pixels, Int(width), Int(height))
    final_img = permutedims(reshaped, (2, 1))

    copyto!(film.framebuffer, final_img)
    return nothing
end

# ============================================================================
# High-Level Functions (legacy, kept for compatibility)
# ============================================================================

"""
    pw_update_film!(backend, film::Film, pixel_L, lambda, samples_accumulated)

Update film framebuffer from accumulated spectral radiance.

NOTE: This function assumes all samples used the SAME wavelengths, which is
only correct for single-sample rendering. For multi-sample rendering, use
pw_accumulate_sample_to_rgb! after each sample instead.
"""
function pw_update_film!(
    backend,
    film::Film,
    pixel_L::AbstractVector{Float32},
    lambda::Wavelengths,
    samples_accumulated::Int32
)
    width = Int32(size(film.framebuffer, 2))
    height = Int32(size(film.framebuffer, 1))
    num_pixels = width * height

    # Use flat RGB array approach (like PbrtWavefront) to avoid kernel caching issues
    pixel_L_cpu = Array(pixel_L)
    rgb_output = Vector{Float32}(undef, Int(num_pixels) * 3)

    inv_samples = samples_accumulated > Int32(0) ? 1.0f0 / Float32(samples_accumulated) : 0.0f0

    @_inbounds for idx in 1:Int(num_pixels)
        base = (idx - 1) * 4
        L = SpectralRadiance((
            pixel_L_cpu[base + 1],
            pixel_L_cpu[base + 2],
            pixel_L_cpu[base + 3],
            pixel_L_cpu[base + 4]
        ))

        R, G, B = spectral_to_linear_rgb(L, lambda)
        R *= inv_samples
        G *= inv_samples
        B *= inv_samples

        R = ifelse(isfinite(R), max(0.0f0, R), 0.0f0)
        G = ifelse(isfinite(G), max(0.0f0, G), 0.0f0)
        B = ifelse(isfinite(B), max(0.0f0, B), 0.0f0)

        out_idx = (idx - 1) * 3
        rgb_output[out_idx + 1] = R
        rgb_output[out_idx + 2] = G
        rgb_output[out_idx + 3] = B
    end

    # Reshape like PbrtWavefront does
    rgb_pixels = reinterpret(RGB{Float32}, rgb_output)
    reshaped = reshape(rgb_pixels, Int(width), Int(height))
    final_img = permutedims(reshaped, (2, 1))

    copyto!(film.framebuffer, final_img)
    return nothing
end

"""
    pw_update_film_perPixel!(backend, film::Film, pixel_L, wavelengths_per_pixel,
                              pdf_per_pixel, samples_accumulated)

Update film framebuffer with per-pixel wavelength data.

Used when each pixel has independently sampled wavelengths.
"""
function pw_update_film_perPixel!(
    backend,
    film::Film,
    pixel_L::AbstractVector{Float32},
    wavelengths_per_pixel::AbstractVector{Float32},
    pdf_per_pixel::AbstractVector{Float32},
    samples_accumulated::Int32
)
    width = Int32(size(film.framebuffer, 2))
    height = Int32(size(film.framebuffer, 1))
    num_pixels = width * height

    kernel! = pw_update_film_spectral_kernel!(backend)
    kernel!(
        film.framebuffer,
        pixel_L,
        wavelengths_per_pixel,
        pdf_per_pixel,
        samples_accumulated,
        width, height;
        ndrange=Int(num_pixels)
    )

    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    pw_accumulate_samples!(backend, pixel_L_accum, pixel_L_sample, num_pixels)

Add new sample results to accumulated film values.
"""
function pw_accumulate_samples!(
    backend,
    pixel_L_accum::AbstractVector{Float32},
    pixel_L_sample::AbstractVector{Float32},
    num_pixels::Int32
)
    kernel! = pw_accumulate_to_film_kernel!(backend)
    kernel!(
        pixel_L_accum,
        pixel_L_sample,
        num_pixels;
        ndrange=Int(num_pixels * Int32(4))
    )

    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    pw_clear_film!(backend, pixel_L, num_pixels)

Clear film accumulator to zero for a new render.
"""
function pw_clear_film!(
    backend,
    pixel_L::AbstractVector{Float32},
    num_pixels::Int32
)
    kernel! = pw_clear_film_kernel!(backend)
    kernel!(
        pixel_L,
        num_pixels;
        ndrange=Int(num_pixels * Int32(4))
    )

    KernelAbstractions.synchronize(backend)
    return nothing
end

# ============================================================================
# Auxiliary Buffer Update from Material Queue
# ============================================================================

"""
    pw_update_aux_from_material_queue!(backend, film, material_queue, materials, rgb2spec_table)

Update film auxiliary buffers (albedo, normal, depth) from first-bounce material hits.

This extracts albedo from materials during the wavefront pipeline,
unlike the separate fill_aux_buffers! which traces primary rays again.
"""
function pw_update_aux_from_material_queue!(
    backend,
    film::Film,
    material_queue::PWWorkQueue{PWMaterialEvalWorkItem},
    materials,
    rgb2spec_table::RGBToSpectrumTable
)
    n = queue_size(material_queue)
    n == 0 && return nothing

    width = Int32(size(film.framebuffer, 2))
    height = Int32(size(film.framebuffer, 1))

    kernel! = pw_update_aux_kernel!(backend)
    kernel!(
        film.albedo,
        film.normal,
        film.depth,
        material_queue.items, material_queue.size,
        materials,
        rgb2spec_table.scale, rgb2spec_table.coeffs, rgb2spec_table.res,
        width, height, Int32(n);
        ndrange=Int(n)
    )

    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    pw_update_aux_kernel!(aux_albedo, aux_normal, aux_depth,
                           material_queue_items, material_queue_size,
                           materials, rgb2spec_scale, rgb2spec_coeffs, rgb2spec_res,
                           width, height, max_queued)

Kernel to update auxiliary buffers from depth=0 material queue items.
"""
@kernel function pw_update_aux_kernel!(
    aux_albedo,    # RGB{Float32} matrix (height × width)
    aux_normal,    # Vec3f matrix (height × width)
    aux_depth,     # Float32 matrix (height × width)
    @Const(material_queue_items), @Const(material_queue_size),
    @Const(materials),
    @Const(rgb2spec_scale),  # RGB to spectrum table scale array
    @Const(rgb2spec_coeffs), # RGB to spectrum table coefficients
    @Const(rgb2spec_res::Int32),  # RGB to spectrum table resolution
    @Const(width::Int32), @Const(height::Int32),
    @Const(max_queued::Int32)
)
    idx = @index(Global)

    # Reconstruct table struct from components for GPU compatibility
    rgb2spec_table = RGBToSpectrumTable(rgb2spec_res, rgb2spec_scale, rgb2spec_coeffs)

    @_inbounds if idx <= max_queued
        current_size = material_queue_size[1]
        if idx <= current_size
            work = material_queue_items[idx]

            # Only update from first bounce hits
            if work.depth == Int32(0)
                pixel_idx = work.pixel_index

                # Get material albedo (spectral, convert to RGB)
                albedo_spec = get_albedo_spectral_dispatch(
                    rgb2spec_table, materials, work.material_idx, work.uv, work.lambda
                )
                # Simple average of spectral values as grayscale approximation
                # Could use hero wavelength for better color, but this is for denoising
                albedo_avg = average(albedo_spec)

                # Convert linear pixel index to 2D
                # pixel_index uses row-major: idx = (y-1)*width + x
                # aux buffers are (height, width), so use [y, x]
                p_idx = pixel_idx - Int32(1)
                x = u_int32(mod(p_idx, width)) + Int32(1)   # column
                y = u_int32(div(p_idx, width)) + Int32(1)   # row

                # Store albedo (grayscale for now)
                aux_albedo[y, x] = RGB{Float32}(albedo_avg, albedo_avg, albedo_avg)

                # Store normal (world space)
                aux_normal[y, x] = work.ns

                # Store depth (z-coordinate of hit point as proxy)
                aux_depth[y, x] = work.pi[3]
            end
        end
    end
end

# ============================================================================
# Tonemapping and Postprocessing
# ============================================================================

"""
    pw_apply_exposure_kernel!(framebuffer, exposure)

Apply exposure adjustment to framebuffer (in-place).
"""
@kernel function pw_apply_exposure_kernel!(
    framebuffer,
    @Const(exposure::Float32)
)
    idx = @index(Global)

    @_inbounds begin
        pixel = framebuffer[idx]
        r = pixel.r * exposure
        g = pixel.g * exposure
        b = pixel.b * exposure
        framebuffer[idx] = RGB{Float32}(r, g, b)
    end
end

"""
    pw_apply_exposure!(backend, film::Film, exposure::Float32)

Apply exposure adjustment to film framebuffer.
"""
function pw_apply_exposure!(backend, film::Film, exposure::Float32)
    kernel! = pw_apply_exposure_kernel!(backend)
    kernel!(
        film.framebuffer,
        exposure;
        ndrange=length(film.framebuffer)
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    pw_apply_srgb_gamma_kernel!(output, input)

Apply sRGB gamma curve to convert from linear to display sRGB.
"""
@kernel function pw_apply_srgb_gamma_kernel!(
    output,
    @Const(input)
)
    idx = @index(Global)

    @_inbounds begin
        pixel = input[idx]
        r = linear_to_srgb_gamma(pixel.r)
        g = linear_to_srgb_gamma(pixel.g)
        b = linear_to_srgb_gamma(pixel.b)
        output[idx] = RGB{Float32}(r, g, b)
    end
end

"""
    pw_apply_srgb_gamma!(backend, film::Film)

Apply sRGB gamma to framebuffer and store in postprocess buffer.
"""
function pw_apply_srgb_gamma!(backend, film::Film)
    kernel! = pw_apply_srgb_gamma_kernel!(backend)
    kernel!(
        film.postprocess,
        film.framebuffer;
        ndrange=length(film.framebuffer)
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end
