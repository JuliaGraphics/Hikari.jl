"""
    blackbody(λ, T)

Planck's law: emitted spectral radiance of a blackbody at wavelength `λ` (in nm)
and temperature `T` (in Kelvin). Matches pbrt-v4 `util/spectrum.h Blackbody()`.

Computed in the promoted type of its arguments, so a `Float64` wavelength gives a
`Float64` result — the CIE integral in `blackbody_to_rgb` needs that precision.
"""
function blackbody(λ::Real, T::Real)
    c = 299792458.0
    ℎ = 6.62606957e-34
    kb = 1.3806488e-23
    l = λ * 1e-9  # Convert nanometers to meters.
    return (2 * ℎ * c * c) / (l^5 * (exp((ℎ * c) / (l * kb * T)) - 1))
end

"""
    blackbody_peak_wavelength(T)

Wavelength (nm) of peak emission for temperature `T`, from Wien's displacement
law. pbrt-v4 normalizes blackbody spectra by their value here.
"""
blackbody_peak_wavelength(T::Real) = 2.8977721e-3 / T * 1e9

"""
    blackbody_normalized(λ, T)

Blackbody SPD scaled so its maximum over all wavelengths is exactly 1, which is
the normalization pbrt-v4's `BlackbodySpectrum` applies. This is NOT the same as
normalizing luminance to 1: the peak sits at Wien's wavelength, which for most
temperatures is off the ȳ curve's peak, so the resulting spectrum's luminance is
well below 1 (0.98 at 5500 K, 0.28 at 2700 K). Conflating the two scales every
blackbody emitter by `1/Y`.
"""
blackbody_normalized(λ::Real, T::Real) = blackbody(λ, T) / blackbody(blackbody_peak_wavelength(T), T)

function blackbody!(Le::Vector{Float32}, λ::Vector{Float32}, T::Float32)
    for i in eachindex(λ, Le)
        Le[i] = blackbody(λ[i], T)
    end
    return Le
end

function blackbody(λ::Vector{Float32}, T::Float32)
    return blackbody!(Vector{Float32}(undef, length(λ)), λ, T)
end

function blackbody_normalized!(Le::Vector{Float32}, λ::Vector{Float32}, T::Float32)
    max_Le = blackbody(blackbody_peak_wavelength(T), T)
    for i in eachindex(λ, Le)
        Le[i] = blackbody(λ[i], T) / max_Le
    end
    return Le
end

function blackbody_normalized(λ::Vector{Float32}, T::Float32)
    return blackbody_normalized!(Vector{Float32}(undef, length(λ)), λ, T)
end
