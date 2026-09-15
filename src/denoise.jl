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
"""
struct DenoiseConfig
    iterations::Int32       # Number of à-trous iterations (typically 4-5)
    sigma_color::Float32    # Color/luminance edge-stopping
    sigma_normal::Float32   # Normal edge-stopping (higher = more blur across normals)
    sigma_depth::Float32    # Depth edge-stopping (higher = more blur across depth)
end

"""
    DenoiseConfig(; iterations=4, sigma_color=1.0, sigma_normal=64.0,
                    sigma_depth=0.1)

Create a denoiser configuration with sensible defaults.

Units (post log-luminance / relative-depth fix):
- `sigma_color` is in **stops** (log2 luminance diff) — 1.0 means a 2× ratio
  weighted at `exp(-1) ≈ 0.37`. Lower = sharper edges.
- `sigma_normal` is the exponent on `clamp(dot(n_p, n_q), 0, 1)`. Higher =
  more sensitive to normal differences.
- `sigma_depth` is **relative** depth tolerance (fraction of local depth) —
  0.1 means a 10% depth diff per filter step weighted at `exp(-1)`.

There is no `use_variance`: real SVGF needs *temporal* variance from sample
history, which a `Film` does not store, and the spatial substitute was actively
harmful — it mistakes texture for noise. See `weight_color`. It used to be a
field that did nothing, plus a kernel argument that had to be a zero-length
device array to keep the signature; that array was freed under the running
dispatch, so the GPU denoiser could not run at all.
"""
function DenoiseConfig(;
    iterations::Int=4,
    sigma_color::Real=1.0f0,
    sigma_normal::Real=64.0f0,
    sigma_depth::Real=0.1f0,
)
    return DenoiseConfig(
        Int32(iterations),
        Float32(sigma_color),
        Float32(sigma_normal),
        Float32(sigma_depth),
    )
end

# =============================================================================
# Edge-Stopping Weight Functions
# =============================================================================

"""
    denoise_luminance(r, g, b) -> Float32

Compute luminance from RGB using Rec. 709 coefficients.
"""
@propagate_inbounds function denoise_luminance(r::Float32, g::Float32, b::Float32)::Float32
    return 0.2126f0 * r + 0.7152f0 * g + 0.0722f0 * b
end

"""
    weight_color(lum_p, lum_q, sigma) -> Float32

Color/luminance edge-stopping weight in **log-luminance space** (HDR-safe).
`sigma` is in *stops* (log2 luminance ratio) — a 2^sigma ratio between p and q
weights to `exp(-1) ≈ 0.37`.

**Asymmetric:** when `q` is dimmer than the center `p` we treat the diff as
`FIREFLY_RATIO ≈ 0.3×` of its actual magnitude, so bright outliers pool down
into their dim surroundings (firefly suppression). When `q` is brighter we
stay strict so genuine edges viewed from the dim side are preserved. This is
the standard fix for à-trous's intrinsic inability to remove fireflies via a
purely symmetric bilateral weight — a firefly otherwise looks like an edge.

There is deliberately no variance term: real SVGF needs *temporal* variance from
sample history, which this `Film` does not store, and the spatial substitute was
actively harmful — it mistakes texture for noise.
"""
const FIREFLY_RATIO = 0.1f0

@propagate_inbounds function weight_color(
    lum_p::Float32, lum_q::Float32,
    sigma::Float32
)::Float32
    log_p = log2(max(lum_p, 0.0f0) + 1.0f-4)
    log_q = log2(max(lum_q, 0.0f0) + 1.0f-4)
    signed_diff = log_p - log_q   # > 0 ⇒ q is dimmer than p
    diff = signed_diff > 0.0f0 ? signed_diff * FIREFLY_RATIO : -signed_diff
    return exp(-diff / max(sigma, 1.0f-4))
end

"""
    weight_normal(n_p, n_q, sigma) -> Float32

Normal edge-stopping weight using dot product.
High sigma means more tolerance for normal differences.
"""
@propagate_inbounds function weight_normal(
    n_p::Vec3f, n_q::Vec3f, sigma::Float32
)::Float32
    dot_val = dot(n_p, n_q)
    # Clamp to [0, 1] and apply power
    dot_clamped = max(0.0f0, dot_val)
    return dot_clamped ^ sigma
end

"""
    weight_depth(d_p, d_q, sigma, step_size) -> Float32

Depth edge-stopping weight using a **relative** depth difference so the same
`sigma` works at any scene scale. `sigma` is the fractional depth tolerance
(e.g. `0.1` = 10% of local depth). Scales with `step_size` because the
expected per-step depth delta on an angled surface grows with filter radius.

Standard SVGF/EAW would use the actual screen-space depth gradient `∇d`
here; we approximate it by `max(d_p, d_q) * step_size`, which is correct up
to a constant for perspective surfaces and never blows up for misses.
"""
@propagate_inbounds function weight_depth(
    d_p::Float32, d_q::Float32,
    sigma::Float32, step_size::Float32
)::Float32
    diff = abs(d_p - d_q)
    scale = max(d_p, d_q, 1.0f0)
    return exp(-diff / (sigma * step_size * scale + 1.0f-4))
end

# =============================================================================
# À-Trous Wavelet Kernel
# =============================================================================

# 5x5 B-spline wavelet kernel weights: h = [1/16, 1/4, 3/8, 1/4, 1/16].
@inline function atrous_kernel_1d(i::Int32)
    kern = SVector{5,Float32}(0.0625f0, 0.25f0, 0.375f0, 0.25f0, 0.0625f0)
    return kern[i]
end

"""
    atrous_denoise_kernel!(output, input, normals, depth,
                           width, height, step_size,
                           sigma_color, sigma_normal, sigma_depth)

Single pass of the à-trous wavelet filter.
Applies a 5x5 filter with edge-stopping weights.
"""
@kernel inbounds=true function atrous_denoise_kernel!(
    output,  # RGB{Float32} matrix (height × width)
    @Const(input),   # RGB{Float32} matrix
    @Const(normals), # Vec3f matrix
    @Const(depth),   # Float32 matrix
    @Const(width::Int32), @Const(height::Int32),
    @Const(step_size::Int32),
    # One-element device arrays (`GPURef`s): the plan is recorded once, and a
    # `DenoiseConfig` that changes between runs reaches it through these.
    @Const(sigma_color_ref),
    @Const(sigma_normal_ref),
    @Const(sigma_depth_ref)
)
    idx = @index(Global)
    num_pixels = width * height
    sigma_color = sigma_color_ref[1]
    sigma_normal = sigma_normal_ref[1]
    sigma_depth = sigma_depth_ref[1]

     if idx <= num_pixels
        # Convert linear index to 2D coordinates (row, col) for Julia matrices
        row = ((idx - Int32(1)) % height) + Int32(1)
        col = ((idx - Int32(1)) ÷ height) + Int32(1)

        # Get center pixel data
        pixel_p = input[row, col]
        r_p, g_p, b_p = pixel_p.r, pixel_p.g, pixel_p.b
        lum_p = denoise_luminance(r_p, g_p, b_p)

        n_p = normals[row, col]
        d_p = depth[row, col]

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
                k_x = atrous_kernel_1d(dx_i)
                k_y = atrous_kernel_1d(dy_i)
                w_spatial = k_x * k_y

                # Edge-stopping weights
                w_color = weight_color(lum_p, lum_q, sigma_color)
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

# `compute_variance_kernel!` used to live here: a 3x3 spatial variance nobody
# dispatched, for a `weight_color` argument that ignored it. Deleted with the
# `use_variance` field and the zero-length placeholder array that had to be
# passed to keep the kernel signature — see `weight_color` for why spatial
# variance is the wrong quantity in the first place.

# =============================================================================
# High-Level Denoising API (works with Film)
# =============================================================================

# One pass of the filter writes what the next reads, so the chain is exactly what
# a graph derives — and it used to be spelled `KA.synchronize(backend)` after
# every iteration, which drains the whole device to express a dependency between
# two dispatches. The scratch buffer is the other half: `similar(framebuffer)` is
# a full-resolution allocation per CALL, i.e. per frame in a live preview. Here it
# is allocated once and kept on the film beside the other scratch.

"""
What `denoise!` keeps between calls: the ping-pong partner of the framebuffer,
and the compiled plan that filters into it.

Keyed on the film's framebuffer and the iteration count, because both are baked
when the plan is compiled — the passes name the buffers, and how many there are
is `config.iterations`. The sigmas are not: they ride `Ref`s and are read at
record time, so turning the filter up mid-session costs nothing.
"""
struct DenoisePlan{P,S,R}
    framebuffer::Any
    iterations::Int32
    scratch::S
    refs::R
    plan::P
    # The scratch's, and only the scratch's. Its own rather than the film's
    # because it has its own lifetime: a plan is rebuilt when the framebuffer or
    # the iteration count changes, and a rebuild that added to the film's memory
    # would accumulate one scratch per rebuild for the life of the film.
    memory::DeviceMemory
end

"""
    free!(dp::DenoisePlan)

Give back the compiled plan's regions and the scratch. Called on a rebuild and
from `free!(::Film)`, and safe at any time — the regions are retired.
"""
free!(dp::DenoisePlan) = (Mantle.free!(dp.plan); free!(dp.memory); nothing)

"""The plan for this film and iteration count, compiled if there is not one."""
function denoise_plan!(film::Film, config::DenoiseConfig)
    cached = film.denoise_plan[]
    if cached isa DenoisePlan &&
       cached.framebuffer === film.framebuffer &&
       cached.iterations == config.iterations
        cached.refs.sigma_color[] = config.sigma_color
        cached.refs.sigma_normal[] = config.sigma_normal
        cached.refs.sigma_depth[] = config.sigma_depth
        return cached
    end
    backend = KA.get_backend(film.framebuffer)
    # The one this replaces goes back before the new one is taken, so a session
    # that changes the iteration count does not leave a plan and a full-frame
    # scratch behind each time. No wait needed: retiring is safe with its passes
    # still in flight.
    if cached isa DenoisePlan
        free!(cached)
        film.denoise_plan[] = nothing
    end
    height, width = size(film.framebuffer)
    n_pixels = width * height
    mem = DeviceMemory(backend)
    scratch = alloc!(mem, eltype(film.framebuffer), size(film.framebuffer))
    # `GPURef`s, not `Ref`s: a `Ref` is read once, at `record!`, and a plan
    # records once — Mantle refuses one in a dispatch for exactly that reason.
    # A `GPURef` holds the value on the device and `r[] = x` stores into it
    # before the next run, which is what `denoise_plan!` does above when the
    # config changes. Owned by `mem`, so `free!` takes them with the scratch.
    refs = (sigma_color = Mantle.GPURef(mem.dev, config.sigma_color),
            sigma_normal = Mantle.GPURef(mem.dev, config.sigma_normal),
            sigma_depth = Mantle.GPURef(mem.dev, config.sigma_depth))
    for r in refs
        own!(mem, r)
    end

    g = Mantle.Graph(mantle_device(backend))
    for i in 1:config.iterations
        # 1, 2, 4, 8, 16… — the à-trous hole size, fixed per pass, so it is a
        # constant of the plan rather than an argument.
        step = Int32(1) << (i - 1)
        src, dst = isodd(i) ? (film.framebuffer, scratch) : (scratch, film.framebuffer)
        Mantle.dispatch!(g, atrous_denoise_kernel!,
                         (dst, src, film.normal, film.depth,
                          Int32(width), Int32(height), step,
                          refs.sigma_color, refs.sigma_normal, refs.sigma_depth),
                         n_pixels; name = "atrous-$i")
    end
    # An odd iteration count leaves the result in the scratch. A copy pass rather
    # than a broadcast afterwards, so it is ordered by the barrier the graph
    # derives like everything else.
    if isodd(config.iterations)
        Mantle.dispatch!(g, denoise_copy_kernel!,
                         (film.framebuffer, scratch, Int32(n_pixels)), n_pixels;
                         name = "writeback")
    end
    made = DenoisePlan(film.framebuffer, config.iterations, scratch, refs,
                       Mantle.record!(Mantle.Plan(g)), mem)
    film.denoise_plan[] = made
    return made
end

@kernel inbounds = true function denoise_copy_kernel!(dst, @Const(src), @Const(n::Int32))
    i = @index(Global)
    if i <= n
        dst[i] = src[i]
    end
end

"""
    denoise!(film::Film; config=DenoiseConfig())

Apply edge-avoiding à-trous wavelet denoising to `film.framebuffer` in place.
Uses `film.normal` and `film.depth` as edge-stopping guides.

# Arguments
- `film`: Film with framebuffer + aux buffers populated
- `config`: DenoiseConfig with filter parameters

# Notes
- Requires `film.normal` and `film.depth` populated (e.g. via `fill_aux_buffers!`).
- Mutates `film.framebuffer` (downstream `postprocess!` reads it).
- The plan and its scratch buffer are cached on the film; changing
  `config.iterations` or rendering to a different film rebuilds them.
"""
function denoise!(film::Film; config::DenoiseConfig=DenoiseConfig())
    config.iterations < Int32(1) && return nothing
    Mantle.run!(denoise_plan!(film, config).plan)
    return nothing
end
