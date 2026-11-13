# RGB to Spectral Uplift
# Convert RGB colors to sampled spectral values at specific wavelengths

# =============================================================================
# Sigmoid Polynomial Method (pbrt-v4 compatible, lowest variance)
# =============================================================================

# Include the lookup table and functions
include("rgb2spec.jl")

"""
    uplift_rgb(table, rgb, lambda) -> SpectralRadiance

The reflectance `rgb` (each channel in 0..1) at the wavelengths `lambda`: the
sigmoid polynomial `table` fits to it, pbrt-v4's RGBAlbedoSpectrum.
"""
@propagate_inbounds function uplift_rgb(table::RGBToSpectrumTable, rgb::RGBSpectrum, lambda::Wavelengths)
    @inbounds poly = rgb_to_spectrum(table, rgb.c[1], rgb.c[2], rgb.c[3])
    # Manually unrolled to avoid closure allocations
    @inbounds begin
        v1 = poly(lambda.lambda[1])
        v2 = poly(lambda.lambda[2])
        v3 = poly(lambda.lambda[3])
        v4 = poly(lambda.lambda[4])
    end
    return SpectralRadiance((v1, v2, v3, v4))
end

"""
    uplift_rgb_unbounded(table, rgb, lambda) -> SpectralRadiance

The unbounded `rgb` (a coefficient, an emission) at the wavelengths `lambda`,
pbrt-v4's RGBUnboundedSpectrum exactly: `scale = 2 max(r, g, b)`, the polynomial
fitted to `rgb / scale`, times `scale`. The 2 keeps the normalised colour in
[0, 0.5], the sigmoid's linear regime, which preserves the spectrum's shape.
"""
@propagate_inbounds function uplift_rgb_unbounded(table::RGBToSpectrumTable, rgb::RGBSpectrum, lambda::Wavelengths)
    @inbounds r, g, b = rgb.c[1], rgb.c[2], rgb.c[3]
    m = max(r, g, b)
    m <= 0.0f0 && return SpectralRadiance(0.0f0)
    scale = 2f0 * m
    poly = rgb_to_spectrum(table, r / scale, g / scale, b / scale)
    @inbounds begin
        v1 = scale * poly(lambda.lambda[1])
        v2 = scale * poly(lambda.lambda[2])
        v3 = scale * poly(lambda.lambda[3])
        v4 = scale * poly(lambda.lambda[4])
    end
    return SpectralRadiance((v1, v2, v3, v4))
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
    d65_cpu_values() -> Vector{Float32}

Return D65 illuminant values as a CPU Vector for use in RGBToSpectrumTable.
This avoids runtime NTuple indexing which fails on Metal GPU.
"""
function d65_cpu_values()::Vector{Float32}
    return collect(Float32, D65_ILLUMINANT_VALUES)
end

"""
    sample_d65(lambda::Float32, d65_values::AbstractVector{Float32}) -> Float32

The D65 illuminant at `lambda` nm, linearly interpolated in the 5 nm table
`d65_values` (300-830 nm), clamped at its ends.
"""
@propagate_inbounds function sample_d65(lambda::Float32, d65_values::AbstractVector{Float32})::Float32
    if lambda <= 300f0
        return @inbounds d65_values[1]
    elseif lambda >= 830f0
        return @inbounds d65_values[107]
    end

    t = (lambda - 300f0) / 5f0
    idx = floor_int32(t) + Int32(1)
    idx = clamp(idx, Int32(1), Int32(106))
    frac = t - Float32(floor_int32(t))
    @inbounds begin
        v0 = d65_values[idx]
        v1 = d65_values[idx + Int32(1)]
    end
    return v0 * (1f0 - frac) + v1 * frac
end

"""
    sample_d65_spectral(lambda, d65_values) -> SpectralRadiance

The D65 illuminant at the wavelengths `lambda`, raw (about 80-120 across the visible
range, 100 at 560 nm), as pbrt-v4's illuminant `Sample(lambda)`. From a table array:
runtime NTuple indexing reads 0 on Metal.
"""
@propagate_inbounds function sample_d65_spectral(lambda::Wavelengths, d65_values::AbstractVector{Float32})::SpectralRadiance
    @inbounds begin
        v1 = sample_d65(lambda.lambda[1], d65_values)
        v2 = sample_d65(lambda.lambda[2], d65_values)
        v3 = sample_d65(lambda.lambda[3], d65_values)
        v4 = sample_d65(lambda.lambda[4], d65_values)
    end
    return SpectralRadiance((v1, v2, v3, v4))
end

# =============================================================================
# Illuminant sampling
# =============================================================================

"""
    uplift_rgb_illuminant(table, s, lambda) -> SpectralRadiance

The illuminant `s` at the wavelengths `lambda`: pbrt-v4's
`RGBIlluminantSpectrum::Sample`, `scale * rsp(λ) * D65(λ)`. An `RGBSpectrum` is the
illuminant pbrt-v4 makes of it ([`rgb_illuminant_spectrum`](@ref)): sRGB's white
point is D65, so (1, 1, 1) emits a D65-like spectrum.

Use it for lights and environment maps; reflectances and albedos take
[`uplift_rgb`](@ref).
"""
@propagate_inbounds function uplift_rgb_illuminant(
    table::RGBToSpectrumTable, s::RGBIlluminantSpectrum, lambda::Wavelengths
)::SpectralRadiance
    @inbounds begin
        d65 = sample_d65_spectral(lambda, table.d65_values)
        v1 = s.scale * s.poly(lambda.lambda[1]) * d65.data[1]
        v2 = s.scale * s.poly(lambda.lambda[2]) * d65.data[2]
        v3 = s.scale * s.poly(lambda.lambda[3]) * d65.data[3]
        v4 = s.scale * s.poly(lambda.lambda[4]) * d65.data[4]
    end
    return SpectralRadiance((v1, v2, v3, v4))
end

@propagate_inbounds uplift_rgb_illuminant(table::RGBToSpectrumTable, rgb::RGBSpectrum, lambda::Wavelengths) =
    @inbounds uplift_rgb_illuminant(table, rgb_illuminant_spectrum(table, rgb.c[1], rgb.c[2], rgb.c[3]), lambda)

# ============================================================================
# Spectrum to Photometric (SpectrumToPhotometric from pbrt-v4)
# ============================================================================

"""
    spectrum_to_photometric(s::Spectrum) -> Float32

Compute photometric luminance of a spectrum, matching pbrt-v4's SpectrumToPhotometric.

For RGBIlluminantSpectrum, this extracts only the D65 illuminant component (ignoring
the RGB multiplier), matching pbrt-v4's behavior where SpectrumToPhotometric extracts
s.Illuminant() first.

This is used for light normalization: `scale = 1 / spectrum_to_photometric(spectrum)`
"""
spectrum_to_photometric(s::RGBSpectrum)::Float32 = luminance(s)
spectrum_to_photometric(s::RGBIlluminantSpectrum)::Float32 = D65_PHOTOMETRIC
