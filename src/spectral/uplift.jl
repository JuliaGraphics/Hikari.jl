# RGB to Spectral Uplift
# Convert RGB colors to sampled spectral values at specific wavelengths

# =============================================================================
# RGB to Spectral Conversion
# =============================================================================

# Spectral primaries for sRGB (simplified model)
# These define wavelength ranges where each RGB component dominates
const RED_WAVELENGTH_CENTER = 650.0f0    # nm
const GREEN_WAVELENGTH_CENTER = 550.0f0  # nm
const BLUE_WAVELENGTH_CENTER = 450.0f0   # nm

"""
    rgb_to_spectral_simple(r::Float32, g::Float32, b::Float32, lambda::Wavelengths) -> SpectralRadiance

Convert RGB to spectral radiance using a simple piecewise linear model.
This is a simplified uplift that treats RGB as spectral bands.

For each wavelength λ:
- Blue region (380-490nm): primarily B channel
- Green region (490-580nm): primarily G channel
- Red region (580-780nm): primarily R channel

With smooth transitions between regions.
"""
@propagate_inbounds function rgb_to_spectral_simple(r::Float32, g::Float32, b::Float32, lambda::Wavelengths)
    # Manually unrolled to avoid closure allocations
    @inbounds begin
        v1 = rgb_to_spectral_at_wavelength(r, g, b, lambda.lambda[1])
        v2 = rgb_to_spectral_at_wavelength(r, g, b, lambda.lambda[2])
        v3 = rgb_to_spectral_at_wavelength(r, g, b, lambda.lambda[3])
        v4 = rgb_to_spectral_at_wavelength(r, g, b, lambda.lambda[4])
    end
    return SpectralRadiance((v1, v2, v3, v4))
end

"""
    rgb_to_spectral_at_wavelength(r, g, b, λ) -> Float32

Get spectral value at a single wavelength from RGB.
Uses smooth blending between spectral bands.
"""
@propagate_inbounds function rgb_to_spectral_at_wavelength(r::Float32, g::Float32, b::Float32, λ::Float32)::Float32
    # Transition wavelengths
    λ_blue_to_green = 490.0f0
    λ_green_to_red = 580.0f0

    # Transition widths (for smooth blending)
    transition_width = 40.0f0

    if λ < λ_blue_to_green - transition_width
        # Pure blue region
        return b
    elseif λ < λ_blue_to_green + transition_width
        # Blue-green transition
        t = (λ - (λ_blue_to_green - transition_width)) / (2.0f0 * transition_width)
        return (1.0f0 - t) * b + t * g
    elseif λ < λ_green_to_red - transition_width
        # Pure green region
        return g
    elseif λ < λ_green_to_red + transition_width
        # Green-red transition
        t = (λ - (λ_green_to_red - transition_width)) / (2.0f0 * transition_width)
        return (1.0f0 - t) * g + t * r
    else
        # Pure red region
        return r
    end
end

"""
    rgb_to_spectral(rgb::RGBSpectrum, lambda::Wavelengths) -> SpectralRadiance

Convert Hikari's RGBSpectrum to spectral radiance at given wavelengths.
"""
@propagate_inbounds function rgb_to_spectral(rgb::RGBSpectrum, lambda::Wavelengths)
    @inbounds return rgb_to_spectral_simple(rgb.c[1], rgb.c[2], rgb.c[3], lambda)
end

# =============================================================================
# Smits' Method (more accurate RGB to spectrum)
# =============================================================================

# Smits' spectral basis functions sampled at specific wavelengths
# Based on "An RGB to Spectrum Conversion for Reflectances" by Smits 1999

"""
    SmitsSpectralBasis

Precomputed basis spectra for Smits' RGB to spectrum conversion.
Contains white, cyan, magenta, yellow, red, green, blue basis spectra.
"""
struct SmitsSpectralBasis
    # Wavelength sample points (nm)
    wavelengths::NTuple{10, Float32}
    # Basis spectra values at each wavelength
    white::NTuple{10, Float32}
    cyan::NTuple{10, Float32}
    magenta::NTuple{10, Float32}
    yellow::NTuple{10, Float32}
    red::NTuple{10, Float32}
    green::NTuple{10, Float32}
    blue::NTuple{10, Float32}
end

# Smits' basis spectra (sampled at 380, 420, 460, 500, 540, 580, 620, 660, 700, 740 nm)
const SMITS_BASIS = SmitsSpectralBasis(
    # Wavelengths
    (380f0, 420f0, 460f0, 500f0, 540f0, 580f0, 620f0, 660f0, 700f0, 740f0),
    # White
    (1.0000f0, 1.0000f0, 0.9999f0, 0.9993f0, 0.9992f0, 0.9998f0, 1.0000f0, 1.0000f0, 1.0000f0, 1.0000f0),
    # Cyan
    (0.9710f0, 0.9426f0, 1.0007f0, 1.0007f0, 1.0007f0, 1.0007f0, 0.1564f0, 0.0000f0, 0.0000f0, 0.0000f0),
    # Magenta
    (1.0000f0, 1.0000f0, 0.9685f0, 0.2229f0, 0.0000f0, 0.0458f0, 0.8369f0, 1.0000f0, 1.0000f0, 0.9959f0),
    # Yellow
    (0.0001f0, 0.0000f0, 0.1088f0, 0.6651f0, 1.0000f0, 1.0000f0, 0.9996f0, 0.9586f0, 0.9685f0, 0.9840f0),
    # Red
    (0.1012f0, 0.0515f0, 0.0000f0, 0.0000f0, 0.0000f0, 0.0000f0, 0.8325f0, 1.0149f0, 1.0149f0, 1.0149f0),
    # Green
    (0.0000f0, 0.0000f0, 0.0273f0, 0.7937f0, 1.0000f0, 0.9418f0, 0.1719f0, 0.0000f0, 0.0000f0, 0.0025f0),
    # Blue
    (1.0000f0, 1.0000f0, 0.8916f0, 0.3323f0, 0.0000f0, 0.0000f0, 0.0003f0, 0.0369f0, 0.0483f0, 0.0496f0)
)

"""
    lerp_smits_basis(basis::NTuple{10, Float32}, λ::Float32) -> Float32

Linearly interpolate a Smits basis spectrum at wavelength λ.
"""
@propagate_inbounds function lerp_smits_basis(basis::NTuple{10, Float32}, λ::Float32)::Float32
    wavelengths = SMITS_BASIS.wavelengths

    # Clamp to range
    @inbounds begin
        if λ <= wavelengths[1]
            return basis[1]
        elseif λ >= wavelengths[10]
            return basis[10]
        end

        # Find interval
        for i in 1:9
            if λ < wavelengths[i+1]
                t = (λ - wavelengths[i]) / (wavelengths[i+1] - wavelengths[i])
                return (1.0f0 - t) * basis[i] + t * basis[i+1]
            end
        end

        return basis[10]
    end
end

"""
    rgb_to_spectral_smits(r::Float32, g::Float32, b::Float32, lambda::Wavelengths) -> SpectralRadiance

Convert RGB to spectral radiance using Smits' method.
More accurate than the simple piecewise linear approach.
"""
@propagate_inbounds function rgb_to_spectral_smits(r::Float32, g::Float32, b::Float32, lambda::Wavelengths)
    # Manually unrolled to avoid closure allocations
    @inbounds begin
        v1 = rgb_to_spectral_smits_at_wavelength(r, g, b, lambda.lambda[1])
        v2 = rgb_to_spectral_smits_at_wavelength(r, g, b, lambda.lambda[2])
        v3 = rgb_to_spectral_smits_at_wavelength(r, g, b, lambda.lambda[3])
        v4 = rgb_to_spectral_smits_at_wavelength(r, g, b, lambda.lambda[4])
    end
    return SpectralRadiance((v1, v2, v3, v4))
end

"""
    rgb_to_spectral_smits_at_wavelength(r, g, b, λ) -> Float32

Compute spectral value at wavelength λ using Smits' method.
"""
@propagate_inbounds function rgb_to_spectral_smits_at_wavelength(r::Float32, g::Float32, b::Float32, λ::Float32)::Float32
    # Decompose RGB into white + primary components
    spectrum = 0.0f0

    if r <= g && r <= b
        # Red is minimum - add white, then cyan or magenta/yellow
        spectrum += r * lerp_smits_basis(SMITS_BASIS.white, λ)
        if g <= b
            spectrum += (g - r) * lerp_smits_basis(SMITS_BASIS.cyan, λ)
            spectrum += (b - g) * lerp_smits_basis(SMITS_BASIS.blue, λ)
        else
            spectrum += (b - r) * lerp_smits_basis(SMITS_BASIS.cyan, λ)
            spectrum += (g - b) * lerp_smits_basis(SMITS_BASIS.green, λ)
        end
    elseif g <= r && g <= b
        # Green is minimum
        spectrum += g * lerp_smits_basis(SMITS_BASIS.white, λ)
        if r <= b
            spectrum += (r - g) * lerp_smits_basis(SMITS_BASIS.magenta, λ)
            spectrum += (b - r) * lerp_smits_basis(SMITS_BASIS.blue, λ)
        else
            spectrum += (b - g) * lerp_smits_basis(SMITS_BASIS.magenta, λ)
            spectrum += (r - b) * lerp_smits_basis(SMITS_BASIS.red, λ)
        end
    else
        # Blue is minimum
        spectrum += b * lerp_smits_basis(SMITS_BASIS.white, λ)
        if r <= g
            spectrum += (r - b) * lerp_smits_basis(SMITS_BASIS.yellow, λ)
            spectrum += (g - r) * lerp_smits_basis(SMITS_BASIS.green, λ)
        else
            spectrum += (g - b) * lerp_smits_basis(SMITS_BASIS.yellow, λ)
            spectrum += (r - g) * lerp_smits_basis(SMITS_BASIS.red, λ)
        end
    end

    return max(0.0f0, spectrum)
end

# =============================================================================
# Sigmoid Polynomial Method (pbrt-v4 compatible, lowest variance)
# =============================================================================

# Include the lookup table and functions
include("rgb2spec.jl")

# Global table - initialized at module load time to avoid type instability
# Will be set by __init__ or on first access
const _RGB2SPEC_TABLE_REF = Ref{RGBToSpectrumTable}()
const _RGB2SPEC_TABLE_LOADED = Ref{Bool}(false)

"""Get the global sRGB to spectrum table (loads on first access)"""
@propagate_inbounds function _get_rgb2spec_table()::RGBToSpectrumTable
    if !_RGB2SPEC_TABLE_LOADED[]
        _RGB2SPEC_TABLE_REF[] = get_srgb_table()
        _RGB2SPEC_TABLE_LOADED[] = true
    end
    return _RGB2SPEC_TABLE_REF[]
end

"""
    rgb_to_spectral_sigmoid(r::Float32, g::Float32, b::Float32, lambda::Wavelengths) -> SpectralRadiance

Convert RGB to spectral radiance using sigmoid polynomial method (pbrt-v4 style).
This provides the smoothest spectra and lowest variance for spectral rendering.

Note: Uses global table, not GPU-compatible. Use the version with explicit table for GPU kernels.
"""
@propagate_inbounds function rgb_to_spectral_sigmoid(r::Float32, g::Float32, b::Float32, lambda::Wavelengths)
    table = _get_rgb2spec_table()
    return rgb_to_spectral_sigmoid(table, r, g, b, lambda)
end

"""
    rgb_to_spectral_sigmoid(table::RGBToSpectrumTable, r, g, b, lambda) -> SpectralRadiance

GPU-compatible version that takes an explicit table parameter.
"""
@propagate_inbounds function rgb_to_spectral_sigmoid(table::RGBToSpectrumTable, r::Float32, g::Float32, b::Float32, lambda::Wavelengths)
    poly = rgb_to_spectrum(table, r, g, b)

    # Manually unrolled to avoid closure allocations
    @inbounds begin
        v1 = poly(lambda.lambda[1])
        v2 = poly(lambda.lambda[2])
        v3 = poly(lambda.lambda[3])
        v4 = poly(lambda.lambda[4])
        return SpectralRadiance((v1, v2, v3, v4))
    end
end

"""
    rgb_to_spectral_sigmoid_unbounded(r::Float32, g::Float32, b::Float32, lambda::Wavelengths) -> SpectralRadiance

Convert RGB to spectral radiance for unbounded values (emission/illumination).
Scales the spectrum to preserve the maximum RGB component.

Note: Uses global table, not GPU-compatible. Use the version with explicit table for GPU kernels.
"""
@propagate_inbounds function rgb_to_spectral_sigmoid_unbounded(r::Float32, g::Float32, b::Float32, lambda::Wavelengths)
    table = _get_rgb2spec_table()
    return rgb_to_spectral_sigmoid_unbounded(table, r, g, b, lambda)
end

"""
    rgb_to_spectral_sigmoid_unbounded(table::RGBToSpectrumTable, r, g, b, lambda) -> SpectralRadiance

GPU-compatible version that takes an explicit table parameter.
"""
@propagate_inbounds function rgb_to_spectral_sigmoid_unbounded(table::RGBToSpectrumTable, r::Float32, g::Float32, b::Float32, lambda::Wavelengths)
    # Find scale factor
    m = max(r, g, b)
    if m <= 0.0f0
        return SpectralRadiance(0.0f0)
    end

    # Normalize and get polynomial for unit-scale color
    poly = rgb_to_spectrum(table, r / m, g / m, b / m)

    # Scale to match original intensity
    max_poly = max_value(poly)
    scale = m / max_poly

    # Manually unrolled to avoid closure allocations
    @inbounds begin
        v1 = scale * poly(lambda.lambda[1])
        v2 = scale * poly(lambda.lambda[2])
        v3 = scale * poly(lambda.lambda[3])
        v4 = scale * poly(lambda.lambda[4])
    end
    return SpectralRadiance((v1, v2, v3, v4))
end

# =============================================================================
# Convenience functions for Hikari types
# =============================================================================

"""
    uplift_rgb(rgb::RGBSpectrum, lambda::Wavelengths; method=:sigmoid) -> SpectralRadiance

Convert Hikari RGBSpectrum to spectral radiance at given wavelengths.
Uses global table - not GPU-compatible. Use the version with explicit table for GPU kernels.

Methods:
- `:sigmoid` - Smooth sigmoid polynomial (pbrt-v4 style, lowest variance, default)
- `:simple` - Fast piecewise linear
- `:smits` - Smits' method
- `:passthrough` - Store RGB directly as first 3 spectral channels (pseudo-spectral, fastest)
"""
@propagate_inbounds function uplift_rgb(rgb::RGBSpectrum, lambda::Wavelengths; method::Symbol=:sigmoid)
    @inbounds begin
        if method === :sigmoid
            return rgb_to_spectral_sigmoid(rgb.c[1], rgb.c[2], rgb.c[3], lambda)
        elseif method === :smits
            return rgb_to_spectral_smits(rgb.c[1], rgb.c[2], rgb.c[3], lambda)
        elseif method === :passthrough
            # Pseudo-spectral: store RGB directly (matches PbrtWavefront behavior)
            # Channel 4 uses average of RGB for luminance
            return SpectralRadiance((rgb.c[1], rgb.c[2], rgb.c[3], (rgb.c[1] + rgb.c[2] + rgb.c[3]) / 3f0))
        else
            return rgb_to_spectral_simple(rgb.c[1], rgb.c[2], rgb.c[3], lambda)
        end
    end
end

"""
    uplift_rgb(table::RGBToSpectrumTable, rgb::RGBSpectrum, lambda::Wavelengths) -> SpectralRadiance

GPU-compatible version that takes an explicit table parameter.
Uses sigmoid polynomial method (pbrt-v4 style, lowest variance).
"""
@propagate_inbounds function uplift_rgb(table::RGBToSpectrumTable, rgb::RGBSpectrum, lambda::Wavelengths)
    @inbounds return rgb_to_spectral_sigmoid(table, rgb.c[1], rgb.c[2], rgb.c[3], lambda)
end

"""
    uplift_rgb_unbounded(rgb::RGBSpectrum, lambda::Wavelengths) -> SpectralRadiance

Convert Hikari RGBSpectrum to spectral radiance for emission/illumination.
Uses sigmoid polynomial method with scaling for unbounded values.
Uses global table - not GPU-compatible. Use the version with explicit table for GPU kernels.
"""
@propagate_inbounds function uplift_rgb_unbounded(rgb::RGBSpectrum, lambda::Wavelengths)
    @inbounds return rgb_to_spectral_sigmoid_unbounded(rgb.c[1], rgb.c[2], rgb.c[3], lambda)
end

"""
    uplift_rgb_unbounded(table::RGBToSpectrumTable, rgb::RGBSpectrum, lambda::Wavelengths) -> SpectralRadiance

GPU-compatible version that takes an explicit table parameter.
"""
@propagate_inbounds function uplift_rgb_unbounded(table::RGBToSpectrumTable, rgb::RGBSpectrum, lambda::Wavelengths)
    @inbounds return rgb_to_spectral_sigmoid_unbounded(table, rgb.c[1], rgb.c[2], rgb.c[3], lambda)
end

"""
    uplift_scalar(value::Float32, lambda::Wavelengths) -> SpectralRadiance

Convert a scalar value to uniform spectral radiance.
"""
@propagate_inbounds function uplift_scalar(value::Float32, lambda::Wavelengths)
    return SpectralRadiance(value)
end

# =============================================================================
# D65 Illuminant Spectrum (CIE Standard Illuminant D65)
# =============================================================================
# Data from pbrt-v4's CIE_Illum_D6500 - interleaved wavelength,value pairs
# The D65 illuminant represents average daylight with CCT ~6500K
# Values are normalized so that Y=100 at 560nm (CIE convention)

"""
    D65_ILLUMINANT_WAVELENGTHS

Wavelength sample points for D65 illuminant spectrum (300-830nm, 5nm intervals).
"""
const D65_ILLUMINANT_WAVELENGTHS = (
    300f0, 305f0, 310f0, 315f0, 320f0, 325f0, 330f0, 335f0, 340f0, 345f0,
    350f0, 355f0, 360f0, 365f0, 370f0, 375f0, 380f0, 385f0, 390f0, 395f0,
    400f0, 405f0, 410f0, 415f0, 420f0, 425f0, 430f0, 435f0, 440f0, 445f0,
    450f0, 455f0, 460f0, 465f0, 470f0, 475f0, 480f0, 485f0, 490f0, 495f0,
    500f0, 505f0, 510f0, 515f0, 520f0, 525f0, 530f0, 535f0, 540f0, 545f0,
    550f0, 555f0, 560f0, 565f0, 570f0, 575f0, 580f0, 585f0, 590f0, 595f0,
    600f0, 605f0, 610f0, 615f0, 620f0, 625f0, 630f0, 635f0, 640f0, 645f0,
    650f0, 655f0, 660f0, 665f0, 670f0, 675f0, 680f0, 685f0, 690f0, 695f0,
    700f0, 705f0, 710f0, 715f0, 720f0, 725f0, 730f0, 735f0, 740f0, 745f0,
    750f0, 755f0, 760f0, 765f0, 770f0, 775f0, 780f0, 785f0, 790f0, 795f0,
    800f0, 805f0, 810f0, 815f0, 820f0, 825f0, 830f0
)

"""
    D65_ILLUMINANT_VALUES

D65 illuminant spectral power distribution values (normalized to 100 at 560nm).
"""
const D65_ILLUMINANT_VALUES = (
    0.0341f0, 1.6643f0, 3.2945f0, 11.7652f0, 20.236f0, 28.6447f0, 37.0535f0,
    38.5011f0, 39.9488f0, 42.4302f0, 44.9117f0, 45.775f0, 46.6383f0, 49.3637f0,
    52.0891f0, 51.0323f0, 49.9755f0, 52.3118f0, 54.6482f0, 68.7015f0, 82.7549f0,
    87.1204f0, 91.486f0, 92.4589f0, 93.4318f0, 90.057f0, 86.6823f0, 95.7736f0,
    104.865f0, 110.936f0, 117.008f0, 117.41f0, 117.812f0, 116.336f0, 114.861f0,
    115.392f0, 115.923f0, 112.367f0, 108.811f0, 109.082f0, 109.354f0, 108.578f0,
    107.802f0, 106.296f0, 104.79f0, 106.239f0, 107.689f0, 106.047f0, 104.405f0,
    104.225f0, 104.046f0, 102.023f0, 100.0f0, 98.1671f0, 96.3342f0, 96.0611f0,
    95.788f0, 92.2368f0, 88.6856f0, 89.3459f0, 90.0062f0, 89.8026f0, 89.5991f0,
    88.6489f0, 87.6987f0, 85.4936f0, 83.2886f0, 83.4939f0, 83.6992f0, 81.863f0,
    80.0268f0, 80.1207f0, 80.2146f0, 81.2462f0, 82.2778f0, 80.281f0, 78.2842f0,
    74.0027f0, 69.7213f0, 70.6652f0, 71.6091f0, 72.979f0, 74.349f0, 67.9765f0,
    61.604f0, 65.7448f0, 69.8856f0, 72.4863f0, 75.087f0, 69.3398f0, 63.5927f0,
    55.0054f0, 46.4182f0, 56.6118f0, 66.8054f0, 65.0941f0, 63.3828f0, 63.8434f0,
    64.304f0, 61.8779f0, 59.4519f0, 55.7054f0, 51.959f0, 54.6998f0, 57.4406f0,
    58.8765f0, 60.3125f0
)

"""
    sample_d65(lambda::Float32) -> Float32

Sample the D65 illuminant spectrum at wavelength lambda (nm).
Uses linear interpolation between tabulated values.
"""
@propagate_inbounds function sample_d65(lambda::Float32)::Float32
    # Clamp to valid range
    if lambda <= 300f0
        return D65_ILLUMINANT_VALUES[1]
    elseif lambda >= 830f0
        return D65_ILLUMINANT_VALUES[107]
    end

    # Find interval (5nm spacing starting at 300nm)
    t = (lambda - 300f0) / 5f0
    idx = floor(Int32, t) + Int32(1)
    idx = clamp(idx, Int32(1), Int32(106))

    # Linear interpolation
    frac = t - floor(t)
    @inbounds begin
        v0 = D65_ILLUMINANT_VALUES[idx]
        v1 = D65_ILLUMINANT_VALUES[idx + 1]
    end
    return v0 * (1f0 - frac) + v1 * frac
end

"""
    sample_d65_spectral(lambda::Wavelengths) -> SpectralRadiance

Sample D65 illuminant at multiple wavelengths.
Returns raw D65 values (around 80-120 across visible spectrum, normalized to 100 at 560nm).
Matches pbrt-v4's illuminant->Sample(lambda) behavior.
"""
@propagate_inbounds function sample_d65_spectral(lambda::Wavelengths)::SpectralRadiance
    @inbounds begin
        v1 = sample_d65(lambda.lambda[1])
        v2 = sample_d65(lambda.lambda[2])
        v3 = sample_d65(lambda.lambda[3])
        v4 = sample_d65(lambda.lambda[4])
    end
    # Return raw D65 values matching pbrt-v4's DenselySampledSpectrum::Sample()
    return SpectralRadiance((v1, v2, v3, v4))
end

# =============================================================================
# RGBIlluminantSpectrum Sampling (defined here after sample_d65 is available)
# =============================================================================

"""
    Sample(s::RGBIlluminantSpectrum, lambda::Wavelengths) -> SpectralRadiance

Sample the illuminant spectrum at multiple wavelengths.
Matches pbrt-v4's RGBIlluminantSpectrum::Sample(const SampledWavelengths &lambda).

Returns: scale * rsp(λ) * D65(λ) for each wavelength.
"""
@propagate_inbounds function Sample(s::RGBIlluminantSpectrum, lambda::Wavelengths)::SpectralRadiance
    @inbounds begin
        d65 = sample_d65_spectral(lambda)
        v1 = s.scale * s.poly(lambda.lambda[1]) * d65.data[1]
        v2 = s.scale * s.poly(lambda.lambda[2]) * d65.data[2]
        v3 = s.scale * s.poly(lambda.lambda[3]) * d65.data[3]
        v4 = s.scale * s.poly(lambda.lambda[4]) * d65.data[4]
    end
    return SpectralRadiance((v1, v2, v3, v4))
end

# =============================================================================
# RGB to Illuminant Spectrum (for light sources)
# =============================================================================

"""
    rgb_to_spectral_sigmoid_illuminant(table::RGBToSpectrumTable, r, g, b, lambda) -> SpectralRadiance

Convert RGB to spectral radiance for illuminants/light sources.
Following pbrt-v4's RGBIlluminantSpectrum: multiplies sigmoid polynomial by D65 illuminant.

This is the correct conversion for environment maps and other light sources that
are specified in sRGB. The D65 multiplication is necessary because sRGB's white
point is D65, so an RGB=(1,1,1) light source should emit a D65-like spectrum.
"""
@propagate_inbounds function rgb_to_spectral_sigmoid_illuminant(
    table::RGBToSpectrumTable, r::Float32, g::Float32, b::Float32, lambda::Wavelengths
)::SpectralRadiance
    # Find scale factor (like RGBUnboundedSpectrum)
    m = max(r, g, b)
    if m <= 0.0f0
        return SpectralRadiance(0.0f0)
    end

    # Get polynomial for normalized color
    # pbrt-v4 uses scale = 2*m and normalizes by scale
    scale = 2f0 * m
    poly = rgb_to_spectrum(table, r / scale, g / scale, b / scale)

    # Sample polynomial at wavelengths and multiply by D65 illuminant
    # Following pbrt-v4's RGBIlluminantSpectrum::Sample()
    @inbounds begin
        d65 = sample_d65_spectral(lambda)
        v1 = scale * poly(lambda.lambda[1]) * d65.data[1]
        v2 = scale * poly(lambda.lambda[2]) * d65.data[2]
        v3 = scale * poly(lambda.lambda[3]) * d65.data[3]
        v4 = scale * poly(lambda.lambda[4]) * d65.data[4]
    end
    return SpectralRadiance((v1, v2, v3, v4))
end

"""
    uplift_rgb_illuminant(table::RGBToSpectrumTable, rgb::RGBSpectrum, lambda::Wavelengths) -> SpectralRadiance

Convert RGB to spectral radiance for illuminants (light sources, environment maps).
Following pbrt-v4's RGBIlluminantSpectrum which multiplies by the D65 illuminant spectrum.

Use this for:
- Environment maps (ImageInfiniteLight)
- Any RGB-specified light source

Do NOT use for:
- Material reflectance/albedo (use uplift_rgb instead)
- Emission from non-illuminant sources
"""
@propagate_inbounds function uplift_rgb_illuminant(
    table::RGBToSpectrumTable, rgb::RGBSpectrum, lambda::Wavelengths
)::SpectralRadiance
    @inbounds return rgb_to_spectral_sigmoid_illuminant(table, rgb.c[1], rgb.c[2], rgb.c[3], lambda)
end

"""
    Sample(table::RGBToSpectrumTable, rgb::RGBSpectrum, lambda::Wavelengths) -> SpectralRadiance

Sample an RGBSpectrum as an illuminant at multiple wavelengths.
This provides a unified interface for light sampling - RGBSpectrum uses uplift_rgb_illuminant
while RGBIlluminantSpectrum uses its baked-in polynomial.
"""
@propagate_inbounds function Sample(
    table::RGBToSpectrumTable, rgb::RGBSpectrum, lambda::Wavelengths
)::SpectralRadiance
    return uplift_rgb_illuminant(table, rgb, lambda)
end

"""
    Sample(::RGBToSpectrumTable, s::RGBIlluminantSpectrum, lambda::Wavelengths) -> SpectralRadiance

Sample an RGBIlluminantSpectrum at multiple wavelengths.
The table argument is ignored since the polynomial is already baked in.
"""
@propagate_inbounds function Sample(
    ::RGBToSpectrumTable, s::RGBIlluminantSpectrum, lambda::Wavelengths
)::SpectralRadiance
    return Sample(s, lambda)
end
