"""
Compute emitted radiance by blackbody
at the given temperature for the wavelengths.

# Args
- `Le::Vector{Float32}`: Preallocated output vector for computed radiance.
- `λ::Vector{Float32}`:
    Wavelengths for which to compute radiance.
    Their values should be in `nm`.
- `T::Float32`: Temperature in Kelvin at which to compute radiance.
"""
function blackbody!(Le::Vector{Float32}, λ::Vector{Float32}, T::Float32)
    c = 299792458f0
    ℎ = 6.62606957f-34
    kb = 1.3806488f-23
    for i in 1:length(λ)
        # Compute emmited radiance for blackbody at wavelength λi.
        l = λ[i] * 1f-9 # Convert nanometers to meters.
        Le[i] = (2 * ℎ * c * c) / (l^5 * (exp((ℎ * c) / (l * kb * T)) - 1f0))
    end
end
"""
Allocating version of `blackbody!` function.
"""
function blackbody(λ::Vector{Float32}, T::Float32)
    Le = Vector{Float32}(undef, length(λ))
    blackbody!(Le, λ, T)
    Le
end

"""
Compute normalized SPD for a blackbody, with maximum value of the SPD
at any wavelength is 1.

# Args
- `Le::Vector{Float32}`: Preallocated output vector for computed radiance.
- `λ::Vector{Float32}`:
    Wavelengths for which to compute radiance.
    Their values should be in `nm`.
- `T::Float32`: Temperature in Kelvin at which to compute radiance.
"""
function blackbody_normalized!(
    Le::Vector{Float32}, λ::Vector{Float32}, T::Float32,
)
    blackbody!(Le, λ, T)
    # Normalize Le values based on maximum blackbody radiance.
    λ_max = [2.8977721f-3 / T * 1e9]
    max_Le = blackbody(λ_max, T)
    for i in 1:length(Le)
        Le[i] /= max_Le
    end
end

function blackbody_normalized(λ::Vector{Float32}, T::Float32)
    Le = Vector{Float32}(undef, length(λ))
    blackbody_normalized!(Le, λ, T)
    Le
end
