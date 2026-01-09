# Edge-Avoiding À-Trous Wavelet Denoiser for Path Tracing
# Based on SVGF (Schied et al. 2017) and EAW (Dammertz et al. 2010)
#
# Uses auxiliary buffers (albedo, normals, depth) from Film for edge-stopping
# to preserve sharp features while removing Monte Carlo noise.

using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
import KernelAbstractions as KA
using StaticArrays

# =============================================================================
# Denoiser Configuration
# =============================================================================

"""
    DenoiseConfig

Configuration parameters for the à-trous wavelet denoiser.

# Fields
- `iterations`: Number of filter passes (each doubles the filter radius)
- `sigma_color`: Color edge-stopping threshold (luminance sensitivity)
- `sigma_normal`: Normal edge-stopping threshold (angular sensitivity)
- `sigma_depth`: Depth edge-stopping threshold (distance sensitivity)
- `use_variance`: Whether to use per-pixel variance to guide filtering
"""
struct DenoiseConfig
    iterations::Int32       # Number of à-trous iterations (typically 4-5)
    sigma_color::Float32    # Color/luminance edge-stopping
    sigma_normal::Float32   # Normal edge-stopping (higher = more blur across normals)
    sigma_depth::Float32    # Depth edge-stopping (higher = more blur across depth)
    use_variance::Bool      # Use variance-guided filtering (SVGF style)
end

"""
    DenoiseConfig(; iterations=5, sigma_color=4.0, sigma_normal=128.0, sigma_depth=1.0, use_variance=true)

Create a denoiser configuration with sensible defaults.
"""
function DenoiseConfig(;
    iterations::Int=5,
    sigma_color::Real=4.0f0,
    sigma_normal::Real=128.0f0,
    sigma_depth::Real=1.0f0,
    use_variance::Bool=true
)
    return DenoiseConfig(
        Int32(iterations),
        Float32(sigma_color),
        Float32(sigma_normal),
        Float32(sigma_depth),
        use_variance
    )
end

# =============================================================================
# Edge-Stopping Weight Functions
# =============================================================================

"""
    denoise_luminance(r, g, b) -> Float32

Compute luminance from RGB using Rec. 709 coefficients.
"""
@inline function denoise_luminance(r::Float32, g::Float32, b::Float32)::Float32
    return 0.2126f0 * r + 0.7152f0 * g + 0.0722f0 * b
end

"""
    weight_color(lum_p, lum_q, sigma, variance) -> Float32

Color/luminance edge-stopping weight.
If variance > 0, scales by sqrt(variance) for variance-guided filtering.
"""
@inline function weight_color(
    lum_p::Float32, lum_q::Float32,
    sigma::Float32, variance::Float32
)::Float32
    diff = abs(lum_p - lum_q)
    # Variance-guided: larger variance allows more blur
    effective_sigma = if variance > 0.0f0
        sigma * sqrt(variance) + 1.0f-4
    else
        sigma
    end
    return exp(-diff / effective_sigma)
end

"""
    weight_normal(n_p, n_q, sigma) -> Float32

Normal edge-stopping weight using dot product.
High sigma means more tolerance for normal differences.
"""
@inline function weight_normal(
    n_p::Vec3f, n_q::Vec3f, sigma::Float32
)::Float32
    dot_val = dot(n_p, n_q)
    # Clamp to [0, 1] and apply power
    dot_clamped = max(0.0f0, dot_val)
    return dot_clamped ^ sigma
end

"""
    weight_depth(d_p, d_q, sigma, step_size) -> Float32

Depth edge-stopping weight.
Uses step size to adapt to increasing filter radius.
"""
@inline function weight_depth(
    d_p::Float32, d_q::Float32,
    sigma::Float32, step_size::Float32
)::Float32
    diff = abs(d_p - d_q)
    # Scale by step size to handle increasing filter radius
    return exp(-diff / (sigma * step_size + 1.0f-4))
end

# =============================================================================
# À-Trous Wavelet Kernel
# =============================================================================

# 5x5 B-spline wavelet kernel weights
# h = [1/16, 1/4, 3/8, 1/4, 1/16]
# These are the 1D weights; 2D weights are outer product
const ATROUS_KERNEL_1D = @SVector Float32[1/16, 1/4, 3/8, 1/4, 1/16]

"""
    atrous_denoise_kernel!(output, input, normals, depth, variance,
                           width, height, step_size, config)

Single pass of the à-trous wavelet filter.
Applies a 5x5 filter with edge-stopping weights.
"""
@kernel function atrous_denoise_kernel!(
    output,  # RGB{Float32} matrix (height × width)
    @Const(input),   # RGB{Float32} matrix
    @Const(normals), # Vec3f matrix
    @Const(depth),   # Float32 matrix
    @Const(variance), # Float32 matrix (can be nothing-like placeholder)
    @Const(width::Int32), @Const(height::Int32),
    @Const(step_size::Int32),
    @Const(sigma_color::Float32),
    @Const(sigma_normal::Float32),
    @Const(sigma_depth::Float32),
    @Const(use_variance::Bool)
)
    idx = @index(Global)
    num_pixels = width * height

    @_inbounds if idx <= num_pixels
        # Convert linear index to 2D coordinates (row, col) for Julia matrices
        row = ((idx - Int32(1)) % height) + Int32(1)
        col = ((idx - Int32(1)) ÷ height) + Int32(1)

        # Get center pixel data
        pixel_p = input[row, col]
        r_p, g_p, b_p = pixel_p.r, pixel_p.g, pixel_p.b
        lum_p = denoise_luminance(r_p, g_p, b_p)

        n_p = normals[row, col]
        d_p = depth[row, col]
        var_p = use_variance ? variance[row, col] : 0.0f0

        # Accumulate filtered result
        sum_r = 0.0f0
        sum_g = 0.0f0
        sum_b = 0.0f0
        sum_weight = 0.0f0

        # 5x5 filter with step_size spacing
        for dy_i in Int32(1):Int32(5)
            for dx_i in Int32(1):Int32(5)
                dy = dy_i - Int32(3)  # -2 to 2
                dx = dx_i - Int32(3)

                # Neighbor coordinates
                q_row = row + dy * step_size
                q_col = col + dx * step_size

                # Boundary check (clamp to edge)
                q_row = clamp(q_row, Int32(1), height)
                q_col = clamp(q_col, Int32(1), width)

                # Get neighbor data
                pixel_q = input[q_row, q_col]
                r_q, g_q, b_q = pixel_q.r, pixel_q.g, pixel_q.b
                lum_q = denoise_luminance(r_q, g_q, b_q)

                n_q = normals[q_row, q_col]
                d_q = depth[q_row, q_col]

                # Compute spatial kernel weight (2D separable B-spline)
                k_x = ATROUS_KERNEL_1D[dx_i]
                k_y = ATROUS_KERNEL_1D[dy_i]
                w_spatial = k_x * k_y

                # Edge-stopping weights
                w_color = weight_color(lum_p, lum_q, sigma_color, var_p)
                w_norm = weight_normal(n_p, n_q, sigma_normal)
                w_depth = weight_depth(d_p, d_q, sigma_depth, Float32(step_size))

                # Combined weight
                weight = w_spatial * w_color * w_norm * w_depth

                # Accumulate
                sum_r += r_q * weight
                sum_g += g_q * weight
                sum_b += b_q * weight
                sum_weight += weight
            end
        end

        # Normalize and store result
        if sum_weight > 1.0f-6
            inv_w = 1.0f0 / sum_weight
            output[row, col] = RGB{Float32}(sum_r * inv_w, sum_g * inv_w, sum_b * inv_w)
        else
            # No valid neighbors - keep original
            output[row, col] = RGB{Float32}(r_p, g_p, b_p)
        end
    end
end

# =============================================================================
# Variance Computation
# =============================================================================

"""
    compute_variance_kernel!(variance, input, width, height)

Compute per-pixel variance from RGB framebuffer.
Uses spatial 3x3 neighborhood for variance estimation.
"""
@kernel function compute_variance_kernel!(
    variance,  # Float32 matrix
    @Const(input),  # RGB{Float32} matrix
    @Const(width::Int32), @Const(height::Int32)
)
    idx = @index(Global)
    num_pixels = width * height

    @_inbounds if idx <= num_pixels
        # Convert linear index to 2D
        row = ((idx - Int32(1)) % height) + Int32(1)
        col = ((idx - Int32(1)) ÷ height) + Int32(1)

        # Compute spatial variance over 3x3 neighborhood
        sum_lum = 0.0f0
        sum_lum_sq = 0.0f0
        count = Int32(0)

        for dy in Int32(-1):Int32(1)
            for dx in Int32(-1):Int32(1)
                q_row = row + dy
                q_col = col + dx

                if q_row >= Int32(1) && q_row <= height && q_col >= Int32(1) && q_col <= width
                    pixel = input[q_row, q_col]
                    lum = denoise_luminance(pixel.r, pixel.g, pixel.b)
                    sum_lum += lum
                    sum_lum_sq += lum * lum
                    count += Int32(1)
                end
            end
        end

        # Variance = E[X²] - E[X]²
        if count > Int32(0)
            mean = sum_lum / Float32(count)
            mean_sq = sum_lum_sq / Float32(count)
            var = max(0.0f0, mean_sq - mean * mean)
            variance[row, col] = var
        else
            variance[row, col] = 0.0f0
        end
    end
end

# =============================================================================
# High-Level Denoising API (works with Film)
# =============================================================================

"""
    denoise!(film::Film; config=DenoiseConfig())

Apply edge-avoiding à-trous wavelet denoising to film's framebuffer.
Uses auxiliary buffers (normal, depth) from film for edge-stopping.
Result is stored in film.postprocess.

# Arguments
- `film`: Film with framebuffer and auxiliary buffers populated
- `config`: DenoiseConfig with filter parameters

# Notes
- Requires film.normal and film.depth to be populated (e.g., via fill_aux_buffers!)
- Modifies film.postprocess in place
- For PhysicalWavefront, aux buffers are populated during first bounce
"""
function denoise!(film::Film; config::DenoiseConfig=DenoiseConfig())
    height, width = size(film.framebuffer)
    num_pixels = width * height

    backend = KA.get_backend(film.framebuffer)

    # Allocate variance buffer if needed
    variance = if config.use_variance
        similar(film.depth)
    else
        # Placeholder - won't be used
        similar(film.depth)
    end

    # Compute variance if using variance-guided filtering
    if config.use_variance
        var_kernel! = compute_variance_kernel!(backend)
        var_kernel!(
            variance, film.framebuffer,
            Int32(width), Int32(height);
            ndrange=num_pixels
        )
        KA.synchronize(backend)
    end

    # Allocate ping-pong buffer
    buffer_a = film.framebuffer
    buffer_b = similar(film.framebuffer)

    denoise_kernel! = atrous_denoise_kernel!(backend)

    # À-trous iterations with exponentially increasing step size
    for i in 1:config.iterations
        step_size = Int32(1 << (i - 1))  # 1, 2, 4, 8, 16...

        # Alternate between buffers
        if i % 2 == 1
            # Read from buffer_a, write to buffer_b
            denoise_kernel!(
                buffer_b,
                buffer_a, film.normal, film.depth, variance,
                Int32(width), Int32(height), step_size,
                config.sigma_color, config.sigma_normal, config.sigma_depth,
                config.use_variance;
                ndrange=num_pixels
            )
        else
            # Read from buffer_b, write to buffer_a
            denoise_kernel!(
                buffer_a,
                buffer_b, film.normal, film.depth, variance,
                Int32(width), Int32(height), step_size,
                config.sigma_color, config.sigma_normal, config.sigma_depth,
                config.use_variance;
                ndrange=num_pixels
            )
        end

        KA.synchronize(backend)
    end

    # Copy result to postprocess buffer
    if config.iterations % 2 == 1
        # Result is in buffer_b
        copyto!(film.postprocess, buffer_b)
    else
        # Result is in buffer_a (framebuffer)
        copyto!(film.postprocess, buffer_a)
    end

    return nothing
end

"""
    denoise_inplace!(film::Film; config=DenoiseConfig())

Like denoise!, but modifies framebuffer directly instead of using postprocess.
"""
function denoise_inplace!(film::Film; config::DenoiseConfig=DenoiseConfig())
    denoise!(film; config=config)
    # Copy postprocess back to framebuffer
    copyto!(film.framebuffer, film.postprocess)
    return nothing
end
