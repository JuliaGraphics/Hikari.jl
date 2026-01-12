# ============================================================================
# Postprocessing Pipeline
# ============================================================================
# Non-destructive postprocessing: reads from film.framebuffer, writes to film.postprocess
# Can be called multiple times with different parameters without re-rendering.

using ImageCore: RGB

# ============================================================================
# Film Sensor Simulation (pbrt-v4 style)
# ============================================================================
# Implements ISO scaling and white balance chromatic adaptation.
# Based on pbrt-v4's PixelSensor class.

"""
    FilmSensor

Film sensor parameters for physically-based image formation.
Matches pbrt-v4's film sensor simulation.

# Fields
- `iso`: ISO sensitivity (default 100). Higher = brighter. Scales output linearly.
- `white_balance`: Color temperature in Kelvin for white balance (default 0 = disabled).
  When set, applies Bradford chromatic adaptation from the illuminant to D65.

# Example
```julia
# Nikon D850 settings from pbrt bunny-cloud scene
sensor = FilmSensor(iso=90, white_balance=5000)
postprocess!(film; sensor=sensor, tonemap=:aces)
```
"""
struct FilmSensor
    iso::Float32
    white_balance::Float32  # Color temperature in Kelvin, 0 = disabled
end

FilmSensor(; iso::Real=100, white_balance::Real=0) = FilmSensor(Float32(iso), Float32(white_balance))

# Default sensor (no adjustment)
const DEFAULT_SENSOR = FilmSensor(iso=100f0, white_balance=0f0)

# ============================================================================
# White Balance - Bradford Chromatic Adaptation
# ============================================================================
# Converts from source illuminant white point to D65 (sRGB white point)

# Bradford transformation matrices (XYZ <-> LMS cone response)
const LMS_FROM_XYZ = @SMatrix Float32[
     0.8951  0.2664 -0.1614;
    -0.7502  1.7135  0.0367;
     0.0389 -0.0685  1.0296
]

const XYZ_FROM_LMS = @SMatrix Float32[
    0.9870 -0.1471  0.1600;
    0.4323  0.5184  0.0493;
   -0.0085  0.0400  0.9685
]

# D65 white point in xy chromaticity
const D65_WHITE_XY = (0.31272f0, 0.32903f0)

"""
    planckian_xy(T::Float32) -> (x, y)

Compute CIE xy chromaticity coordinates for a Planckian (blackbody) radiator
at temperature T in Kelvin. Valid for 1667K to 25000K.
"""
@propagate_inbounds function planckian_xy(T::Float32)
    # Approximation from CIE 015:2004
    T2 = T * T
    T3 = T2 * T

    # x chromaticity
    x = if T <= 4000f0
        -0.2661239f9 / T3 - 0.2343589f6 / T2 + 0.8776956f3 / T + 0.179910f0
    else
        -3.0258469f9 / T3 + 2.1070379f6 / T2 + 0.2226347f3 / T + 0.240390f0
    end

    # y chromaticity
    x2 = x * x
    x3 = x2 * x
    y = if T <= 2222f0
        -1.1063814f0 * x3 - 1.34811020f0 * x2 + 2.18555832f0 * x - 0.20219683f0
    elseif T <= 4000f0
        -0.9549476f0 * x3 - 1.37418593f0 * x2 + 2.09137015f0 * x - 0.16748867f0
    else
        3.0817580f0 * x3 - 5.87338670f0 * x2 + 3.75112997f0 * x - 0.37001483f0
    end

    return (x, y)
end

"""
    xy_to_XYZ(x, y) -> (X, Y, Z)

Convert CIE xy chromaticity to XYZ with Y=1.
"""
@propagate_inbounds function xy_to_XYZ(x::Float32, y::Float32)
    Y = 1f0
    X = x / y
    Z = (1f0 - x - y) / y
    return (X, Y, Z)
end

"""
    compute_white_balance_matrix(src_temp::Float32) -> SMatrix{3,3,Float32}

Compute the Bradford chromatic adaptation matrix to transform
from source illuminant (at color temperature src_temp) to D65.
"""
function compute_white_balance_matrix(src_temp::Float32)
    # Get source and destination white points
    src_xy = planckian_xy(src_temp)
    dst_xy = D65_WHITE_XY

    # Convert to XYZ
    src_XYZ = xy_to_XYZ(src_xy[1], src_xy[2])
    dst_XYZ = xy_to_XYZ(dst_xy[1], dst_xy[2])

    # Transform to LMS cone response space
    src_LMS = LMS_FROM_XYZ * SVector{3,Float32}(src_XYZ...)
    dst_LMS = LMS_FROM_XYZ * SVector{3,Float32}(dst_XYZ...)

    # Compute diagonal scaling matrix in LMS space
    scale_L = dst_LMS[1] / src_LMS[1]
    scale_M = dst_LMS[2] / src_LMS[2]
    scale_S = dst_LMS[3] / src_LMS[3]
    LMS_scale = @SMatrix Float32[
        scale_L 0 0;
        0 scale_M 0;
        0 0 scale_S
    ]

    # Full Bradford transform: XYZ -> LMS -> scale -> XYZ
    return XYZ_FROM_LMS * LMS_scale * LMS_FROM_XYZ
end

"""
    apply_white_balance(r, g, b, wb_matrix) -> (r', g', b')

Apply white balance transformation using precomputed Bradford matrix.
Input/output are in linear RGB (assumed sRGB primaries).
"""
@propagate_inbounds function apply_white_balance(r::Float32, g::Float32, b::Float32,
                                                  m11::Float32, m12::Float32, m13::Float32,
                                                  m21::Float32, m22::Float32, m23::Float32,
                                                  m31::Float32, m32::Float32, m33::Float32)
    # Note: For simplicity, we apply the XYZ transform directly to RGB
    # This is an approximation but works well for typical color temperatures
    r_out = m11 * r + m12 * g + m13 * b
    g_out = m21 * r + m22 * g + m23 * b
    b_out = m31 * r + m32 * g + m33 * b
    return (max(0f0, r_out), max(0f0, g_out), max(0f0, b_out))
end

# ============================================================================
# Tonemapping Implementations
# ============================================================================

"""
Simple Reinhard tonemapping: L / (1 + L)
"""
@propagate_inbounds function _tonemap_reinhard(r::Float32, g::Float32, b::Float32)
    lum = 0.2126f0 * r + 0.7152f0 * g + 0.0722f0 * b
    scale = ifelse(lum > 0f0, 1f0 / (1f0 + lum), 1f0)
    clamp(r * scale, 0f0, 1f0), clamp(g * scale, 0f0, 1f0), clamp(b * scale, 0f0, 1f0)
end

"""
Extended Reinhard with white point control.
"""
@propagate_inbounds function _tonemap_reinhard_extended(r::Float32, g::Float32, b::Float32, Lwhite::Float32)
    lum = 0.2126f0 * r + 0.7152f0 * g + 0.0722f0 * b
    Lwhite2 = Lwhite * Lwhite
    scale = ifelse(lum > 0f0, (1f0 + lum / Lwhite2) / (1f0 + lum), 1f0)
    clamp(r * scale, 0f0, 1f0), clamp(g * scale, 0f0, 1f0), clamp(b * scale, 0f0, 1f0)
end

"""
ACES filmic approximation (Narkowicz).
Industry-standard filmic curve used in games and film.
"""
@propagate_inbounds function _tonemap_aces(r::Float32, g::Float32, b::Float32)
    a = 2.51f0
    b_c = 0.03f0
    c = 2.43f0
    d = 0.59f0
    e = 0.14f0

    r_out = clamp((r * (a * r + b_c)) / (r * (c * r + d) + e), 0f0, 1f0)
    g_out = clamp((g * (a * g + b_c)) / (g * (c * g + d) + e), 0f0, 1f0)
    b_out = clamp((b * (a * b + b_c)) / (b * (c * b + d) + e), 0f0, 1f0)
    r_out, g_out, b_out
end

"""
Uncharted 2 filmic curve helper.
"""
@propagate_inbounds function _uncharted2_partial(x::Float32)
    A = 0.15f0
    B = 0.50f0
    C = 0.10f0
    D = 0.20f0
    E = 0.02f0
    F = 0.30f0
    ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
end

"""
Uncharted 2 filmic tonemapping.
Good for high-contrast scenes, preserves detail in shadows.
"""
@propagate_inbounds function _tonemap_uncharted2(r::Float32, g::Float32, b::Float32)
    W = 11.2f0
    exposure_bias = 2.0f0

    r_out = _uncharted2_partial(r * exposure_bias)
    g_out = _uncharted2_partial(g * exposure_bias)
    b_out = _uncharted2_partial(b * exposure_bias)

    white_scale = 1f0 / _uncharted2_partial(W)

    clamp(r_out * white_scale, 0f0, 1f0),
    clamp(g_out * white_scale, 0f0, 1f0),
    clamp(b_out * white_scale, 0f0, 1f0)
end

"""
Filmic tonemapping (Hejl-Dawson).
Alternative filmic curve with good highlight rolloff.
"""
@propagate_inbounds function _tonemap_filmic(r::Float32, g::Float32, b::Float32)
    # Attempt to preserve some color saturation
    function filmic_channel(x::Float32)
        x = max(0f0, x - 0.004f0)
        (x * (6.2f0 * x + 0.5f0)) / (x * (6.2f0 * x + 1.7f0) + 0.06f0)
    end
    filmic_channel(r), filmic_channel(g), filmic_channel(b)
end

# ============================================================================
# Main Postprocessing Function (KernelAbstractions for CPU/GPU)
# ============================================================================

# Tonemap mode constants for kernel dispatch
const TONEMAP_NONE = UInt8(0)
const TONEMAP_REINHARD = UInt8(1)
const TONEMAP_REINHARD_EXT = UInt8(2)
const TONEMAP_ACES = UInt8(3)
const TONEMAP_UNCHARTED2 = UInt8(4)
const TONEMAP_FILMIC = UInt8(5)

@propagate_inbounds function _apply_tonemap(r::Float32, g::Float32, b::Float32, mode::UInt8, wp::Float32)
    if mode == TONEMAP_REINHARD
        return _tonemap_reinhard(r, g, b)
    elseif mode == TONEMAP_REINHARD_EXT
        return _tonemap_reinhard_extended(r, g, b, wp)
    elseif mode == TONEMAP_ACES
        return _tonemap_aces(r, g, b)
    elseif mode == TONEMAP_UNCHARTED2
        return _tonemap_uncharted2(r, g, b)
    elseif mode == TONEMAP_FILMIC
        return _tonemap_filmic(r, g, b)
    else
        # Linear clamp
        return clamp(r, 0f0, 1f0), clamp(g, 0f0, 1f0), clamp(b, 0f0, 1f0)
    end
end

@kernel inbounds=true function postprocess_kernel!(dst, @Const(src), exposure::Float32, tonemap_mode::UInt8,
                                      inv_gamma::Float32, apply_gamma::Bool, white_point::Float32,
                                      iso_scale::Float32, apply_wb::Bool,
                                      wb11::Float32, wb12::Float32, wb13::Float32,
                                      wb21::Float32, wb22::Float32, wb23::Float32,
                                      wb31::Float32, wb32::Float32, wb33::Float32)
    i = @index(Global, Linear)
    begin
        c = src[i]

        # Apply exposure
        r = c.r * exposure
        g = c.g * exposure
        b = c.b * exposure

        # Apply white balance (Bradford chromatic adaptation)
        if apply_wb
            r, g, b = apply_white_balance(r, g, b,
                                          wb11, wb12, wb13,
                                          wb21, wb22, wb23,
                                          wb31, wb32, wb33)
        end

        # Apply ISO scaling (relative to ISO 100 baseline)
        r = r * iso_scale
        g = g * iso_scale
        b = b * iso_scale

        # Apply tonemapping
        r, g, b = _apply_tonemap(r, g, b, tonemap_mode, white_point)

        # Apply gamma correction
        if apply_gamma
            r = r^inv_gamma
            g = g^inv_gamma
            b = b^inv_gamma
        end

        dst[i] = RGB{Float32}(r, g, b)
    end
end

"""
    postprocess!(film::Film; exposure=1.0f0, tonemap=:aces, gamma=2.2f0, white_point=4.0f0, sensor=nothing)

Apply postprocessing to film.framebuffer and write result to film.postprocess.

This function is non-destructive: the original framebuffer is preserved, allowing
you to call postprocess! multiple times with different parameters.

Works on both CPU and GPU arrays via KernelAbstractions.

# Arguments
- `film`: The Film containing rendered data
- `exposure`: Exposure multiplier applied before tonemapping (default: 1.0)
- `tonemap`: Tonemapping method (default: :aces)
  - `:reinhard` - Simple Reinhard L/(1+L)
  - `:reinhard_extended` - Extended Reinhard with white point
  - `:aces` - ACES filmic (industry standard)
  - `:uncharted2` - Uncharted 2 filmic
  - `:filmic` - Hejl-Dawson filmic
  - `nothing` - No tonemapping (linear clamp)
- `gamma`: Gamma correction value (default: 2.2, use `nothing` to skip)
- `white_point`: White point for extended Reinhard (default: 4.0)
- `sensor`: FilmSensor for pbrt-style sensor simulation (ISO, white balance)

# Example
```julia
# Render once
integrator(scene, film, camera)
to_framebuffer!(film)

# Try different postprocessing settings
postprocess!(film; exposure=1.0, tonemap=:aces)
display(film.postprocess)

# With pbrt-style sensor (bunny-cloud scene settings)
sensor = FilmSensor(iso=90, white_balance=5000)
postprocess!(film; sensor=sensor, tonemap=:aces)
display(film.postprocess)
```
"""
function postprocess!(film::Film;
    exposure::Real = 1.0,
    tonemap::Union{Symbol, Nothing} = :aces,
    gamma::Union{Real, Nothing} = 2.2,
    white_point::Real = 4.0,
    sensor::Union{FilmSensor, Nothing} = nothing,
)
    src = film.framebuffer
    dst = film.postprocess

    # Convert parameters
    exp_f32 = Float32(exposure)
    wp_f32 = Float32(white_point)
    inv_gamma = isnothing(gamma) ? 1.0f0 : 1f0 / Float32(gamma)
    apply_gamma = !isnothing(gamma)

    # Sensor parameters
    actual_sensor = isnothing(sensor) ? DEFAULT_SENSOR : sensor
    # ISO scaling: pbrt uses imagingRatio = exposureTime * ISO / 100
    # Since exposure is already a multiplier, we just scale by ISO/100
    iso_scale = actual_sensor.iso / 100f0

    # White balance matrix
    apply_wb = actual_sensor.white_balance > 0f0
    wb_matrix = if apply_wb
        compute_white_balance_matrix(actual_sensor.white_balance)
    else
        @SMatrix Float32[1 0 0; 0 1 0; 0 0 1]  # Identity
    end

    # Map tonemap symbol to mode constant
    tonemap_mode = if tonemap === :reinhard
        TONEMAP_REINHARD
    elseif tonemap === :reinhard_extended
        TONEMAP_REINHARD_EXT
    elseif tonemap === :aces
        TONEMAP_ACES
    elseif tonemap === :uncharted2
        TONEMAP_UNCHARTED2
    elseif tonemap === :filmic
        TONEMAP_FILMIC
    else
        TONEMAP_NONE
    end

    # Get backend from array type and launch kernel
    backend = KernelAbstractions.get_backend(src)
    kernel! = postprocess_kernel!(backend)
    kernel!(dst, src, exp_f32, tonemap_mode, inv_gamma, apply_gamma, wp_f32,
            iso_scale, apply_wb,
            wb_matrix[1,1], wb_matrix[1,2], wb_matrix[1,3],
            wb_matrix[2,1], wb_matrix[2,2], wb_matrix[2,3],
            wb_matrix[3,1], wb_matrix[3,2], wb_matrix[3,3];
            ndrange=length(src))
    KernelAbstractions.synchronize(backend)

    return film.postprocess
end
