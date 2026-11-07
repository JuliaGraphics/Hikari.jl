# PiecewiseLinearSpectrum — GPU-compatible piecewise-linear spectral data
# Used for measured metal IOR data (eta, k) from pbrt-v4

struct PiecewiseLinearSpectrum{N}
    lambdas::NTuple{N, Float32}
    values::NTuple{N, Float32}
end

# Sample at a single wavelength via binary search + linear interpolation
# Matches pbrt-v4's PiecewiseLinearSpectrum::operator()
@propagate_inbounds function sample(s::PiecewiseLinearSpectrum{N}, λ::Float32) where {N}
    # `@inbounds` LOCALLY, not merely `@propagate_inbounds`. These are runtime
    # indices into an `NTuple{N,Float32}`, so without it every access carries a
    # bounds check that can throw — and a throwing call cannot be a separate
    # OpFunction (GPUCompiler force-inlines anything containing `unreachable`),
    # so the error paths land inline in the shader. This one function contributed
    # 96 of the 218 error/boxing call sites in the bump-gold closest-hit shader.
    #
    # Safe by construction rather than by assumption: `lo` starts at 1, `hi` at
    # N, `mid = (lo + hi) >>> 1` is strictly between them, and the loop only
    # ever assigns `lo = mid` or `hi = mid`, so `1 <= lo < hi <= N` holds at
    # every access below.
    @inbounds begin
        # Outside range: return endpoint values
        λ <= s.lambdas[1] && return s.values[1]
        λ >= s.lambdas[N] && return s.values[N]

        # Binary search for the interval containing λ
        lo = 1
        hi = N
        while lo + 1 < hi
            mid = (lo + hi) >>> 1
            if s.lambdas[mid] <= λ
                lo = mid
            else
                hi = mid
            end
        end

        # Linear interpolation within [lo, hi]
        t = (λ - s.lambdas[lo]) / (s.lambdas[hi] - s.lambdas[lo])
        return s.values[lo] * (1f0 - t) + s.values[hi] * t
    end
end

# Sample at 4 wavelengths → SpectralRadiance (matches pbrt-v4's Sample(SampledWavelengths))
@propagate_inbounds function sample(s::PiecewiseLinearSpectrum, lambda::Wavelengths)
    SpectralRadiance((
        sample(s, lambda.lambda[1]),
        sample(s, lambda.lambda[2]),
        sample(s, lambda.lambda[3]),
        sample(s, lambda.lambda[4]),
    ))
end

# Factory from interleaved (λ₁, v₁, λ₂, v₂, ...) data
# Matches pbrt-v4's PiecewiseLinearSpectrum::FromInterleaved
function from_interleaved(::Type{PiecewiseLinearSpectrum{N}}, data::NTuple{M, Float32}) where {N, M}
    @assert M == 2N "Expected $(2N) interleaved values, got $M"
    lambdas = ntuple(i -> data[2i - 1], Val(N))
    values = ntuple(i -> data[2i], Val(N))
    return PiecewiseLinearSpectrum{N}(lambdas, values)
end

# Accept Float64 tuples (auto-convert to Float32)
function from_interleaved(::Type{PiecewiseLinearSpectrum{N}}, data::NTuple{M, Float64}) where {N, M}
    from_interleaved(PiecewiseLinearSpectrum{N}, Float32.(data))
end

# Convert to RGB by sampling at representative wavelengths
# Used as fallback for Whitted/FastWavefront path
@propagate_inbounds function to_rgb(s::PiecewiseLinearSpectrum)
    # Sample at approximate sRGB primary peaks
    r = sample(s, 630f0)
    g = sample(s, 532f0)
    b = sample(s, 467f0)
    return RGBSpectrum(r, g, b)
end
