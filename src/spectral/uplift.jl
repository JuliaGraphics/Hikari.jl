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
@inline function rgb_to_spectral_simple(r::Float32, g::Float32, b::Float32, lambda::Wavelengths)
    data = ntuple(Val(4)) do i
        λ = lambda.lambda[i]
        rgb_to_spectral_at_wavelength(r, g, b, λ)
    end
    return SpectralRadiance(data)
end

"""
    rgb_to_spectral_at_wavelength(r, g, b, λ) -> Float32

Get spectral value at a single wavelength from RGB.
Uses smooth blending between spectral bands.
"""
@inline function rgb_to_spectral_at_wavelength(r::Float32, g::Float32, b::Float32, λ::Float32)::Float32
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
@inline function rgb_to_spectral(rgb::RGBSpectrum, lambda::Wavelengths)
    return rgb_to_spectral_simple(rgb.c[1], rgb.c[2], rgb.c[3], lambda)
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
@inline function lerp_smits_basis(basis::NTuple{10, Float32}, λ::Float32)::Float32
    wavelengths = SMITS_BASIS.wavelengths

    # Clamp to range
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

"""
    rgb_to_spectral_smits(r::Float32, g::Float32, b::Float32, lambda::Wavelengths) -> SpectralRadiance

Convert RGB to spectral radiance using Smits' method.
More accurate than the simple piecewise linear approach.
"""
@inline function rgb_to_spectral_smits(r::Float32, g::Float32, b::Float32, lambda::Wavelengths)
    data = ntuple(Val(4)) do i
        λ = lambda.lambda[i]
        rgb_to_spectral_smits_at_wavelength(r, g, b, λ)
    end
    return SpectralRadiance(data)
end

"""
    rgb_to_spectral_smits_at_wavelength(r, g, b, λ) -> Float32

Compute spectral value at wavelength λ using Smits' method.
"""
@inline function rgb_to_spectral_smits_at_wavelength(r::Float32, g::Float32, b::Float32, λ::Float32)::Float32
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
# Convenience functions for Hikari types
# =============================================================================

"""
    uplift_rgb(rgb::RGBSpectrum, lambda::Wavelengths; method=:simple) -> SpectralRadiance

Convert Hikari RGBSpectrum to spectral radiance at given wavelengths.

Methods:
- `:simple` - Fast piecewise linear (default)
- `:smits` - More accurate Smits' method
"""
@inline function uplift_rgb(rgb::RGBSpectrum, lambda::Wavelengths; method::Symbol=:simple)
    r, g, b = rgb.c[1], rgb.c[2], rgb.c[3]
    if method === :smits
        return rgb_to_spectral_smits(r, g, b, lambda)
    else
        return rgb_to_spectral_simple(r, g, b, lambda)
    end
end

"""
    uplift_scalar(value::Float32, lambda::Wavelengths) -> SpectralRadiance

Convert a scalar value to uniform spectral radiance.
"""
@inline function uplift_scalar(value::Float32, lambda::Wavelengths)
    return SpectralRadiance(value)
end
