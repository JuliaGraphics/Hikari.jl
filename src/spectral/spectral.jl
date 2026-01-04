# Sampled spectrum representation for PhysicalWavefront
# Hero wavelength sampling with 4 wavelength samples for GPU efficiency

"""
    SampledSpectrum{N}

A spectrum sampled at N wavelengths. Default is 4 for GPU efficiency (matches Float4).
Used by PhysicalWavefront for spectral path tracing.
"""
struct SampledSpectrum{N}
    data::NTuple{N, Float32}
end

# Default to 4 samples - alias for convenience
const SpectralRadiance = SampledSpectrum{4}

# Constructors
@inline SampledSpectrum{N}(v::Float32) where {N} = SampledSpectrum{N}(ntuple(_ -> v, Val(N)))
@inline SampledSpectrum{N}(v::Real) where {N} = SampledSpectrum{N}(Float32(v))

@inline function SpectralRadiance(r::Real, g::Real, b::Real, a::Real)
    return SpectralRadiance((Float32(r), Float32(g), Float32(b), Float32(a)))
end

# Zero spectrum
@inline SampledSpectrum{N}() where {N} = SampledSpectrum{N}(0f0)
@inline SpectralRadiance() = SpectralRadiance(0f0)

# Array-like interface
@inline Base.getindex(s::SampledSpectrum, i::Int) = s.data[i]
@inline Base.length(::SampledSpectrum{N}) where {N} = N
@inline Base.eltype(::Type{SampledSpectrum{N}}) where {N} = Float32

# Arithmetic operations - all tuple-based for GPU efficiency (no allocations)
@inline Base.:+(a::SampledSpectrum{N}, b::SampledSpectrum{N}) where {N} =
    SampledSpectrum{N}(ntuple(i -> a.data[i] + b.data[i], Val(N)))
@inline Base.:-(a::SampledSpectrum{N}, b::SampledSpectrum{N}) where {N} =
    SampledSpectrum{N}(ntuple(i -> a.data[i] - b.data[i], Val(N)))
@inline Base.:*(a::SampledSpectrum{N}, b::SampledSpectrum{N}) where {N} =
    SampledSpectrum{N}(ntuple(i -> a.data[i] * b.data[i], Val(N)))
@inline Base.:/(a::SampledSpectrum{N}, b::SampledSpectrum{N}) where {N} =
    SampledSpectrum{N}(ntuple(i -> a.data[i] / b.data[i], Val(N)))

@inline Base.:*(a::SampledSpectrum{N}, s::Real) where {N} =
    SampledSpectrum{N}(ntuple(i -> a.data[i] * Float32(s), Val(N)))
@inline Base.:*(s::Real, a::SampledSpectrum{N}) where {N} = a * s
@inline Base.:/(a::SampledSpectrum{N}, s::Real) where {N} =
    SampledSpectrum{N}(ntuple(i -> a.data[i] / Float32(s), Val(N)))

# Unary minus
@inline Base.:-(a::SampledSpectrum{N}) where {N} =
    SampledSpectrum{N}(ntuple(i -> -a.data[i], Val(N)))

# sqrt for BSDF computations
@inline Base.sqrt(s::SampledSpectrum{N}) where {N} =
    SampledSpectrum{N}(ntuple(i -> sqrt(s.data[i]), Val(N)))

# exp for transmittance
@inline Base.exp(s::SampledSpectrum{N}) where {N} =
    SampledSpectrum{N}(ntuple(i -> exp(s.data[i]), Val(N)))

# Utility functions
@inline function average(s::SampledSpectrum{N}) where {N}
    sum = 0.0f0
    for i in 1:N
        sum += s.data[i]
    end
    return sum / N
end

@inline function max_component(s::SampledSpectrum{N}) where {N}
    m = s.data[1]
    for i in 2:N
        m = max(m, s.data[i])
    end
    return m
end

@inline function min_component(s::SampledSpectrum{N}) where {N}
    m = s.data[1]
    for i in 2:N
        m = min(m, s.data[i])
    end
    return m
end

@inline is_black(s::SampledSpectrum{N}) where {N} = all(x -> x == 0.0f0, s.data)
@inline is_positive(s::SampledSpectrum) = !is_black(s)

# Check for NaN/Inf
@inline function has_nan(s::SampledSpectrum{N}) where {N}
    for i in 1:N
        isnan(s.data[i]) && return true
    end
    return false
end

@inline function has_inf(s::SampledSpectrum{N}) where {N}
    for i in 1:N
        isinf(s.data[i]) && return true
    end
    return false
end

# Safe division (avoid NaN)
@inline function safe_div(a::SampledSpectrum{N}, b::SampledSpectrum{N}) where {N}
    SampledSpectrum{N}(ntuple(i -> b.data[i] != 0.0f0 ? a.data[i] / b.data[i] : 0.0f0, Val(N)))
end

# Clamp to zero (remove negative values)
@inline function clamp_zero(s::SampledSpectrum{N}) where {N}
    SampledSpectrum{N}(ntuple(i -> max(0.0f0, s.data[i]), Val(N)))
end

# Clamp to range
@inline function Base.clamp(s::SampledSpectrum{N}, lo::Real, hi::Real) where {N}
    SampledSpectrum{N}(ntuple(i -> clamp(s.data[i], Float32(lo), Float32(hi)), Val(N)))
end

# ============================================================================
# Sampled Wavelengths
# ============================================================================

"""
    SampledWavelengths{N}

Represents N sampled wavelengths and their PDFs for hero wavelength sampling.
"""
struct SampledWavelengths{N}
    lambda::NTuple{N, Float32}  # Wavelengths in nm
    pdf::NTuple{N, Float32}     # PDF for each wavelength
end

const Wavelengths = SampledWavelengths{4}

# Visible spectrum range
const LAMBDA_MIN = 380.0f0  # nm
const LAMBDA_MAX = 780.0f0  # nm
const LAMBDA_RANGE = LAMBDA_MAX - LAMBDA_MIN

"""
    sample_wavelengths_uniform(u::Float32) -> Wavelengths

Sample 4 wavelengths using hero wavelength sampling with stratified offsets.
This gives better spectral coverage than independent uniform samples.
"""
@inline function sample_wavelengths_uniform(u::Float32)
    # Hero wavelength sampling: one uniform sample determines all wavelengths
    # with stratified offsets for better coverage
    lambda1 = LAMBDA_MIN + u * LAMBDA_RANGE
    lambda2 = lambda1 + LAMBDA_RANGE / 4
    lambda3 = lambda1 + LAMBDA_RANGE / 2
    lambda4 = lambda1 + 3 * LAMBDA_RANGE / 4

    # Wrap around to stay in visible range
    wrap(l) = l > LAMBDA_MAX ? l - LAMBDA_RANGE : l

    lambdas = (lambda1, wrap(lambda2), wrap(lambda3), wrap(lambda4))
    # Uniform PDF: 1 / range for each wavelength
    pdf = ntuple(_ -> 1.0f0 / LAMBDA_RANGE, Val(4))

    return Wavelengths(lambdas, pdf)
end

"""
    sample_wavelengths_stratified(u::NTuple{4, Float32}) -> Wavelengths

Sample 4 wavelengths with stratified sampling (one per stratum).
"""
@inline function sample_wavelengths_stratified(u::NTuple{4, Float32})
    stratum_size = LAMBDA_RANGE / 4

    lambdas = ntuple(Val(4)) do i
        stratum_start = LAMBDA_MIN + (i - 1) * stratum_size
        stratum_start + u[i] * stratum_size
    end

    # Uniform PDF within each stratum
    pdf = ntuple(_ -> 1.0f0 / LAMBDA_RANGE, Val(4))

    return Wavelengths(lambdas, pdf)
end

"""
    terminate_secondary_wavelengths(lambda::Wavelengths) -> Wavelengths

Set PDF to zero for secondary wavelengths when a wavelength-dependent
event occurs (e.g., refraction with dispersion). This indicates that
only the hero wavelength (first) should contribute to the pixel.
"""
@inline function terminate_secondary_wavelengths(lambda::Wavelengths)
    # Keep first wavelength, zero out others
    new_pdf = (lambda.pdf[1], 0.0f0, 0.0f0, 0.0f0)
    return Wavelengths(lambda.lambda, new_pdf)
end

"""
    pdf_is_nonzero(lambda::Wavelengths, i::Int) -> Bool

Check if wavelength i has non-zero PDF (should contribute).
"""
@inline pdf_is_nonzero(lambda::Wavelengths, i::Int) = lambda.pdf[i] > 0.0f0
