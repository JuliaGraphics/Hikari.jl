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
@propagate_inbounds SampledSpectrum{N}(v::Float32) where {N} = SampledSpectrum{N}(ntuple(_ -> v, Val(N)))
@propagate_inbounds SampledSpectrum{N}(v::Real) where {N} = SampledSpectrum{N}(Float32(v))

@propagate_inbounds function SpectralRadiance(r::Real, g::Real, b::Real, a::Real)
    return SpectralRadiance((Float32(r), Float32(g), Float32(b), Float32(a)))
end

# Zero spectrum
@propagate_inbounds SampledSpectrum{N}() where {N} = SampledSpectrum{N}(0f0)
@propagate_inbounds SpectralRadiance() = SpectralRadiance(0f0)

# Array-like interface
@propagate_inbounds Base.getindex(s::SampledSpectrum, i::Int) = s.data[i]
@propagate_inbounds Base.length(::SampledSpectrum{N}) where {N} = N
@propagate_inbounds Base.eltype(::Type{SampledSpectrum{N}}) where {N} = Float32

# Arithmetic operations - all tuple-based for GPU efficiency (no allocations)
@propagate_inbounds Base.:+(a::SampledSpectrum{N}, b::SampledSpectrum{N}) where {N} =
    SampledSpectrum{N}(ntuple(i -> a.data[i] + b.data[i], Val(N)))
@propagate_inbounds Base.:-(a::SampledSpectrum{N}, b::SampledSpectrum{N}) where {N} =
    SampledSpectrum{N}(ntuple(i -> a.data[i] - b.data[i], Val(N)))
@propagate_inbounds Base.:*(a::SampledSpectrum{N}, b::SampledSpectrum{N}) where {N} =
    SampledSpectrum{N}(ntuple(i -> a.data[i] * b.data[i], Val(N)))
@propagate_inbounds Base.:/(a::SampledSpectrum{N}, b::SampledSpectrum{N}) where {N} =
    SampledSpectrum{N}(ntuple(i -> a.data[i] / b.data[i], Val(N)))

@propagate_inbounds Base.:*(a::SampledSpectrum{N}, s::Real) where {N} =
    SampledSpectrum{N}(ntuple(i -> a.data[i] * Float32(s), Val(N)))
@propagate_inbounds Base.:*(s::Real, a::SampledSpectrum{N}) where {N} = a * s
@propagate_inbounds Base.:/(a::SampledSpectrum{N}, s::Real) where {N} =
    SampledSpectrum{N}(ntuple(i -> a.data[i] / Float32(s), Val(N)))

# Unary minus
@propagate_inbounds Base.:-(a::SampledSpectrum{N}) where {N} =
    SampledSpectrum{N}(ntuple(i -> -a.data[i], Val(N)))

# sqrt for BSDF computations
@propagate_inbounds Base.sqrt(s::SampledSpectrum{N}) where {N} =
    SampledSpectrum{N}(ntuple(i -> sqrt(s.data[i]), Val(N)))

# exp for transmittance
@propagate_inbounds Base.exp(s::SampledSpectrum{N}) where {N} =
    SampledSpectrum{N}(ntuple(i -> exp(s.data[i]), Val(N)))

# Utility functions
@propagate_inbounds function average(s::SampledSpectrum{N}) where {N}
    return sum(s.data) / N
end

@propagate_inbounds function max_component(s::SampledSpectrum{N}) where {N}
    return maximum(s.data)
end

@propagate_inbounds function min_component(s::SampledSpectrum{N}) where {N}
    return minimum(s.data)
end

@propagate_inbounds is_black(s::SampledSpectrum{N}) where {N} = all(x -> x == 0.0f0, s.data)
@propagate_inbounds is_positive(s::SampledSpectrum) = !is_black(s)

# Check for NaN/Inf
@propagate_inbounds function has_nan(s::SampledSpectrum{N}) where {N}
    return any(isnan, s.data)
end

@propagate_inbounds function has_inf(s::SampledSpectrum{N}) where {N}
    return any(isinf, s.data)
end

# Safe division (avoid NaN)
@propagate_inbounds function safe_div(a::SampledSpectrum{N}, b::SampledSpectrum{N}) where {N}
    SampledSpectrum{N}(ntuple(i -> b.data[i] != 0.0f0 ? a.data[i] / b.data[i] : 0.0f0, Val(N)))
end

# Clamp to zero (remove negative values)
@propagate_inbounds function clamp_zero(s::SampledSpectrum{N}) where {N}
    SampledSpectrum{N}(ntuple(i -> max(0.0f0, s.data[i]), Val(N)))
end

# Clamp to range
@propagate_inbounds function Base.clamp(s::SampledSpectrum{N}, lo::Real, hi::Real) where {N}
    SampledSpectrum{N}(ntuple(i -> clamp(s.data[i], Float32(lo), Float32(hi)), Val(N)))
end

# Element-wise max with scalar
@propagate_inbounds Base.max(x::Real, s::SampledSpectrum{N}) where {N} =
    SampledSpectrum{N}(ntuple(i -> max(Float32(x), s.data[i]), Val(N)))
@propagate_inbounds Base.max(s::SampledSpectrum{N}, x::Real) where {N} = max(x, s)

# Element-wise min with scalar
@propagate_inbounds Base.min(x::Real, s::SampledSpectrum{N}) where {N} =
    SampledSpectrum{N}(ntuple(i -> min(Float32(x), s.data[i]), Val(N)))
@propagate_inbounds Base.min(s::SampledSpectrum{N}, x::Real) where {N} = min(x, s)

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
@propagate_inbounds function sample_wavelengths_uniform(u::Float32)
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
@propagate_inbounds function sample_wavelengths_stratified(u::NTuple{4, Float32})
    stratum_size = LAMBDA_RANGE / 4

    lambdas = ntuple(Val(4)) do i
        stratum_start = LAMBDA_MIN + (i - 1) * stratum_size
        stratum_start + u[i] * stratum_size
    end

    # Uniform PDF within each stratum
    pdf = ntuple(_ -> 1.0f0 / LAMBDA_RANGE, Val(4))

    return Wavelengths(lambdas, pdf)
end

# ============================================================================
# Importance-Sampled Wavelengths (pbrt-v4 style)
# ============================================================================

# Extended visible range for importance sampling (matches pbrt-v4)
const LAMBDA_MIN_VISIBLE = 360.0f0
const LAMBDA_MAX_VISIBLE = 830.0f0

"""
    visible_wavelengths_pdf(lambda::Float32) -> Float32

PDF for importance-sampled visible wavelengths, centered at 538nm.
This distribution reduces variance by sampling more where human vision is sensitive.

From pbrt-v4: PDF = 0.0039398042 / cosh²(0.0072 * (lambda - 538))
"""
@propagate_inbounds function visible_wavelengths_pdf(lambda::Float32)::Float32
    if lambda < LAMBDA_MIN_VISIBLE || lambda > LAMBDA_MAX_VISIBLE
        return 0.0f0
    end
    # Hyperbolic secant squared distribution centered at 538nm
    x = 0.0072f0 * (lambda - 538.0f0)
    cosh_x = cosh(x)
    return 0.0039398042f0 / (cosh_x * cosh_x)
end

"""
    sample_visible_wavelengths(u::Float32) -> Float32

Sample a single wavelength using importance sampling for visible light.
Inverse CDF of the hyperbolic secant squared distribution.

From pbrt-v4: lambda = 538 - 138.888889 * atanh(0.85691062 - 1.82750197 * u)
"""
@propagate_inbounds function sample_visible_wavelengths(u::Float32)::Float32
    # Inverse CDF for hyperbolic secant squared distribution
    return 538.0f0 - 138.888889f0 * atanh(0.85691062f0 - 1.82750197f0 * u)
end

"""
    sample_wavelengths_visible(u::Float32) -> Wavelengths

Sample 4 wavelengths using importance sampling with hero wavelength method.
Uses pbrt-v4's visible wavelength distribution for reduced variance.
"""
@propagate_inbounds function sample_wavelengths_visible(u::Float32)
    # Sample hero wavelength using importance sampling
    lambda1 = sample_visible_wavelengths(u)

    # Secondary wavelengths with stratified offsets in [0,1)
    # These map to different parts of the spectrum
    u2 = u + 0.25f0
    u2 = u2 >= 1.0f0 ? u2 - 1.0f0 : u2
    u3 = u + 0.5f0
    u3 = u3 >= 1.0f0 ? u3 - 1.0f0 : u3
    u4 = u + 0.75f0
    u4 = u4 >= 1.0f0 ? u4 - 1.0f0 : u4

    lambda2 = sample_visible_wavelengths(u2)
    lambda3 = sample_visible_wavelengths(u3)
    lambda4 = sample_visible_wavelengths(u4)

    lambdas = (lambda1, lambda2, lambda3, lambda4)

    # PDF for each wavelength
    pdf = (
        visible_wavelengths_pdf(lambda1),
        visible_wavelengths_pdf(lambda2),
        visible_wavelengths_pdf(lambda3),
        visible_wavelengths_pdf(lambda4)
    )

    return Wavelengths(lambdas, pdf)
end

"""
    terminate_secondary_wavelengths(lambda::Wavelengths) -> Wavelengths

Set PDF to zero for secondary wavelengths when a wavelength-dependent
event occurs (e.g., refraction with dispersion). This indicates that
only the hero wavelength (first) should contribute to the pixel.
"""
@propagate_inbounds function terminate_secondary_wavelengths(lambda::Wavelengths)
    # Keep first wavelength, zero out others
    new_pdf = (lambda.pdf[1], 0.0f0, 0.0f0, 0.0f0)
    return Wavelengths(lambda.lambda, new_pdf)
end

"""
    pdf_is_nonzero(lambda::Wavelengths, i::Int) -> Bool

Check if wavelength i has non-zero PDF (should contribute).
"""
@propagate_inbounds pdf_is_nonzero(lambda::Wavelengths, i::Int) = lambda.pdf[i] > 0.0f0


@propagate_inbounds function accumulate_spectrum!(pixel_L, base_idx::Int32, contrib::SpectralRadiance)
    Atomix.@atomic pixel_L[base_idx+Int32(1)] += contrib[1]
    Atomix.@atomic pixel_L[base_idx+Int32(2)] += contrib[2]
    Atomix.@atomic pixel_L[base_idx+Int32(3)] += contrib[3]
    Atomix.@atomic pixel_L[base_idx+Int32(4)] += contrib[4]
end

_pointer(x, idx) = pointer(x, idx)
_pointer(x::Base.Experimental.Const, idx) = pointer(x.a, idx)

Base.@propagate_inbounds function load(array::AbstractArray{Float32}, index::Integer, ::Type{T}) where T
    ptr = _pointer(array, index)
    ptr32 = as_pointer(T, ptr)
    return Base.unsafe_load(ptr32)
end
