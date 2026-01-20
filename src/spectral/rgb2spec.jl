# RGB to Spectrum Conversion using Sigmoid Polynomial
# Runtime lookup using precomputed tables (pbrt-v4 compatible)

# ============================================================================
# RGBSigmoidPolynomial - Represents a smooth spectrum from 3 coefficients
# ============================================================================

"""
    RGBSigmoidPolynomial

Represents a smooth spectrum using a sigmoid-wrapped polynomial.
The spectrum value at wavelength λ is: sigmoid(c0*λ² + c1*λ + c2)
where sigmoid(x) = 0.5 + x / (2*sqrt(1 + x²))

This provides a smooth, bounded [0,1] spectrum from just 3 coefficients.
"""
struct RGBSigmoidPolynomial
    c0::Float32
    c1::Float32
    c2::Float32
end

"""Sigmoid function for spectrum evaluation"""
@propagate_inbounds function sigmoid(x::Float32)::Float32
    if isinf(x)
        return x > 0 ? 1.0f0 : 0.0f0
    end
    return 0.5f0 + x / (2.0f0 * sqrt(1.0f0 + x * x))
end

"""Evaluate spectrum at wavelength λ (in nm)"""
@propagate_inbounds function (poly::RGBSigmoidPolynomial)(lambda::Float32)::Float32
    # Polynomial: c0*λ² + c1*λ + c2
    x = poly.c0 * lambda * lambda + poly.c1 * lambda + poly.c2
    return sigmoid(x)
end

"""Maximum value of the spectrum (for normalization)"""
function max_value(poly::RGBSigmoidPolynomial)::Float32
    # Check endpoints
    result = max(poly(360.0f0), poly(830.0f0))

    # Check critical point of polynomial (where derivative = 0)
    # d/dλ(c0*λ² + c1*λ + c2) = 2*c0*λ + c1 = 0  =>  λ = -c1/(2*c0)
    if poly.c0 != 0
        lambda_crit = -poly.c1 / (2.0f0 * poly.c0)
        if 360.0f0 <= lambda_crit <= 830.0f0
            result = max(result, poly(lambda_crit))
        end
    end

    return result
end

# ============================================================================
# RGBToSpectrumTable - Lookup table for RGB to polynomial coefficients
# ============================================================================

"""
    RGBToSpectrumTable{V1, V5}

Precomputed lookup table for converting RGB colors to RGBSigmoidPolynomial coefficients.
Uses trilinear interpolation for smooth results.

Parameterized by array types for GPU compatibility:
- V1: 1D array type for scale (AbstractVector{Float32})
- V5: 5D array type for coeffs (AbstractArray{Float32, 5})

Use `to_gpu(ArrayType, table)` to convert to GPU-compatible arrays.
"""
struct RGBToSpectrumTable{V1 <: AbstractVector{Float32}, V5 <: AbstractArray{Float32, 5}}
    res::Int32
    scale::V1           # [res] - z-axis (max component) scale values
    coeffs::V5          # [3, res, res, res, 3] - [maxc, z, y, x, coeff]
end

"""
    rgb_to_spectrum(table, rgb) -> RGBSigmoidPolynomial

Convert an RGB color to a sigmoid polynomial spectrum representation.
RGB values should be in [0, 1] range.
"""
@propagate_inbounds function rgb_to_spectrum(table::RGBToSpectrumTable, r::Float32, g::Float32, b::Float32)
    # Clamp to valid range
    r = clamp(r, 0.0f0, 1.0f0)
    g = clamp(g, 0.0f0, 1.0f0)
    b = clamp(b, 0.0f0, 1.0f0)

    # Handle uniform RGB (gray) - special case
    if r == g && g == b
        # For gray, polynomial is constant: sigmoid(c2) = r
        # Solving: 0.5 + c2/(2*sqrt(1+c2²)) = r
        # => c2 = (r - 0.5) / sqrt(r * (1 - r))
        c2 = if r > 0.0f0 && r < 1.0f0
            (r - 0.5f0) / sqrt(r * (1.0f0 - r))
        elseif r <= 0.0f0
            -1.0f10  # Very negative -> sigmoid ≈ 0
        else
            1.0f10   # Very positive -> sigmoid ≈ 1
        end
        return RGBSigmoidPolynomial(0.0f0, 0.0f0, c2)
    end

    # Find maximum component and remap - use Int32, no type asserts
    maxc = r > g ? (r > b ? Int32(1) : Int32(3)) : (g > b ? Int32(2) : Int32(3))
    z = maxc == Int32(1) ? r : (maxc == Int32(2) ? g : b)

    # Get components for x and y based on maxc
    # maxc=1(R): x=G, y=B; maxc=2(G): x=B, y=R; maxc=3(B): x=R, y=G
    x_comp = maxc == Int32(1) ? g : (maxc == Int32(2) ? b : r)
    y_comp = maxc == Int32(1) ? b : (maxc == Int32(2) ? r : g)

    # Remap other components relative to max
    res = table.res  # Already Int32
    x = x_comp * Float32(res - Int32(1)) / z
    y = y_comp * Float32(res - Int32(1)) / z

    # Find z index using linear search in scale table
    zi = Int32(1)
    @inbounds for i in Int32(1):(res - Int32(1))
        if table.scale[i] < z
            zi = i
        end
    end
    zi = min(zi, res - Int32(1))

    # Compute integer indices and fractional offsets
    xi = min(u_int32(x) + Int32(1), res - Int32(1))
    yi = min(u_int32(y) + Int32(1), res - Int32(1))

    dx = x - Float32(xi - Int32(1))
    dy = y - Float32(yi - Int32(1))
    @inbounds dz = (z - table.scale[zi]) / (table.scale[zi + Int32(1)] - table.scale[zi])

    # Trilinear interpolation of coefficients - fully unrolled, no closures
    coeffs = table.coeffs
    @inbounds begin
        # Coefficient 0 (c0)
        c0 = (1.0f0 - dz) * (
            (1.0f0 - dy) * ((1.0f0 - dx) * coeffs[maxc, zi, yi, xi, Int32(1)] + dx * coeffs[maxc, zi, yi, xi+Int32(1), Int32(1)]) +
                     dy * ((1.0f0 - dx) * coeffs[maxc, zi, yi+Int32(1), xi, Int32(1)] + dx * coeffs[maxc, zi, yi+Int32(1), xi+Int32(1), Int32(1)])
        ) + dz * (
            (1.0f0 - dy) * ((1.0f0 - dx) * coeffs[maxc, zi+Int32(1), yi, xi, Int32(1)] + dx * coeffs[maxc, zi+Int32(1), yi, xi+Int32(1), Int32(1)]) +
                     dy * ((1.0f0 - dx) * coeffs[maxc, zi+Int32(1), yi+Int32(1), xi, Int32(1)] + dx * coeffs[maxc, zi+Int32(1), yi+Int32(1), xi+Int32(1), Int32(1)])
        )

        # Coefficient 1 (c1)
        c1 = (1.0f0 - dz) * (
            (1.0f0 - dy) * ((1.0f0 - dx) * coeffs[maxc, zi, yi, xi, Int32(2)] + dx * coeffs[maxc, zi, yi, xi+Int32(1), Int32(2)]) +
                     dy * ((1.0f0 - dx) * coeffs[maxc, zi, yi+Int32(1), xi, Int32(2)] + dx * coeffs[maxc, zi, yi+Int32(1), xi+Int32(1), Int32(2)])
        ) + dz * (
            (1.0f0 - dy) * ((1.0f0 - dx) * coeffs[maxc, zi+Int32(1), yi, xi, Int32(2)] + dx * coeffs[maxc, zi+Int32(1), yi, xi+Int32(1), Int32(2)]) +
                     dy * ((1.0f0 - dx) * coeffs[maxc, zi+Int32(1), yi+Int32(1), xi, Int32(2)] + dx * coeffs[maxc, zi+Int32(1), yi+Int32(1), xi+Int32(1), Int32(2)])
        )

        # Coefficient 2 (c2)
        c2 = (1.0f0 - dz) * (
            (1.0f0 - dy) * ((1.0f0 - dx) * coeffs[maxc, zi, yi, xi, Int32(3)] + dx * coeffs[maxc, zi, yi, xi+Int32(1), Int32(3)]) +
                     dy * ((1.0f0 - dx) * coeffs[maxc, zi, yi+Int32(1), xi, Int32(3)] + dx * coeffs[maxc, zi, yi+Int32(1), xi+Int32(1), Int32(3)])
        ) + dz * (
            (1.0f0 - dy) * ((1.0f0 - dx) * coeffs[maxc, zi+Int32(1), yi, xi, Int32(3)] + dx * coeffs[maxc, zi+Int32(1), yi, xi+Int32(1), Int32(3)]) +
                     dy * ((1.0f0 - dx) * coeffs[maxc, zi+Int32(1), yi+Int32(1), xi, Int32(3)] + dx * coeffs[maxc, zi+Int32(1), yi+Int32(1), xi+Int32(1), Int32(3)])
        )
    end

    return RGBSigmoidPolynomial(c0, c1, c2)
end

# Convenience method
rgb_to_spectrum(table::RGBToSpectrumTable, rgb::NTuple{3, Float32}) =
    rgb_to_spectrum(table, rgb[1], rgb[2], rgb[3])

# ============================================================================
# GPU-compatible inline version (for kernels)
# ============================================================================

"""
    rgb_to_spectrum_coeffs(scale, coeffs, res, r, g, b) -> (c0, c1, c2)

GPU-compatible version that takes raw arrays and returns coefficient tuple.
"""
@propagate_inbounds function rgb_to_spectrum_coeffs(
    scale::AbstractVector{Float32},
    coeffs::AbstractArray{Float32, 5},
    res::Int32,
    r::Float32, g::Float32, b::Float32
)::NTuple{3, Float32}
    # Clamp
    r = clamp(r, 0.0f0, 1.0f0)
    g = clamp(g, 0.0f0, 1.0f0)
    b = clamp(b, 0.0f0, 1.0f0)

    # Gray case
    if r == g && g == b
        if r > 0.0f0 && r < 1.0f0
            c2 = (r - 0.5f0) / sqrt(r * (1.0f0 - r))
        elseif r <= 0.0f0
            c2 = -1.0f10
        else
            c2 = 1.0f10
        end
        return (0.0f0, 0.0f0, c2)
    end

    # Find max component
    maxc = r > g ? (r > b ? Int32(1) : Int32(3)) : (g > b ? Int32(2) : Int32(3))
    z = maxc == 1 ? r : (maxc == 2 ? g : b)

    # Remap
    next_c = maxc == 3 ? Int32(1) : maxc + Int32(1)
    next2_c = next_c == 3 ? Int32(1) : next_c + Int32(1)
    rgb_next = next_c == 1 ? r : (next_c == 2 ? g : b)
    rgb_next2 = next2_c == 1 ? r : (next2_c == 2 ? g : b)

    x = rgb_next * Float32(res - 1) / z
    y = rgb_next2 * Float32(res - 1) / z

    # Find z index
    zi = Int32(1)
    @inbounds for i in Int32(1):Int32(res-1)
        if scale[i] < z
            zi = i
        end
    end
    zi = min(zi, Int32(res - 1))

    # Integer indices
    xi = min(u_int32(x) + Int32(1), Int32(res - 1))
    yi = min(u_int32(y) + Int32(1), Int32(res - 1))

    dx = x - Float32(xi - 1)
    dy = y - Float32(yi - 1)
    @inbounds dz = (z - scale[zi]) / (scale[zi + 1] - scale[zi])

    # Trilinear interpolation
    c0 = 0.0f0
    c1 = 0.0f0
    c2 = 0.0f0

    @inbounds for i in Int32(1):Int32(3)
        val = (1.0f0 - dz) * (
            (1.0f0 - dy) * ((1.0f0 - dx) * coeffs[maxc, zi, yi, xi, i] +
                                     dx * coeffs[maxc, zi, yi, xi+1, i]) +
                     dy * ((1.0f0 - dx) * coeffs[maxc, zi, yi+1, xi, i] +
                                     dx * coeffs[maxc, zi, yi+1, xi+1, i])
        ) + dz * (
            (1.0f0 - dy) * ((1.0f0 - dx) * coeffs[maxc, zi+1, yi, xi, i] +
                                     dx * coeffs[maxc, zi+1, yi, xi+1, i]) +
                     dy * ((1.0f0 - dx) * coeffs[maxc, zi+1, yi+1, xi, i] +
                                     dx * coeffs[maxc, zi+1, yi+1, xi+1, i])
        )

        if i == Int32(1)
            c0 = val
        elseif i == Int32(2)
            c1 = val
        else
            c2 = val
        end
    end

    return (c0, c1, c2)
end

"""Evaluate sigmoid polynomial at wavelength (GPU-compatible)"""
@propagate_inbounds function eval_sigmoid_polynomial(c0::Float32, c1::Float32, c2::Float32, lambda::Float32)::Float32
    x = c0 * lambda * lambda + c1 * lambda + c2
    if isinf(x)
        return x > 0 ? 1.0f0 : 0.0f0
    end
    return 0.5f0 + x / (2.0f0 * sqrt(1.0f0 + x * x))
end

# ============================================================================
# Spectrum Types using RGBSigmoidPolynomial
# ============================================================================

"""
    RGBAlbedoSpectrum

Bounded [0,1] spectrum for surface reflectance/albedo.
"""
struct RGBAlbedoSpectrum
    poly::RGBSigmoidPolynomial
end

@propagate_inbounds (s::RGBAlbedoSpectrum)(lambda::Float32) = s.poly(lambda)

"""
    RGBUnboundedSpectrum

Unbounded spectrum for illumination/emission. Scales the sigmoid polynomial
to match the maximum RGB component.
"""
struct RGBUnboundedSpectrum
    poly::RGBSigmoidPolynomial
    scale::Float32
end

@propagate_inbounds function (s::RGBUnboundedSpectrum)(lambda::Float32)::Float32
    return s.scale * s.poly(lambda)
end

"""Create unbounded spectrum from RGB (for lights/emission)"""
function rgb_unbounded_spectrum(table::RGBToSpectrumTable, r::Float32, g::Float32, b::Float32)::RGBUnboundedSpectrum
    # Find scale factor
    m = max(r, g, b)
    if m <= 0
        return RGBUnboundedSpectrum(RGBSigmoidPolynomial(0.0f0, 0.0f0, -1.0f10), 0.0f0)
    end

    # Normalize and get polynomial
    scale = 2.0f0 * m  # Factor of 2 because sigmoid max is ~0.5 for large positive input
    poly = rgb_to_spectrum(table, r / m, g / m, b / m)

    return RGBUnboundedSpectrum(poly, scale / max_value(poly))
end

"""
    RGBIlluminantSpectrum

Illuminant spectrum for light sources, matching pbrt-v4's RGBIlluminantSpectrum.
Stores RGB coefficients and multiplies by D65 illuminant when sampled.

Fields:
- `poly::RGBSigmoidPolynomial` - Sigmoid polynomial for spectral shape
- `scale::Float32` - Scale factor (2 * max(r,g,b) from construction)

When sampled at wavelength λ: returns `scale * poly(λ) * D65(λ)`
"""
struct RGBIlluminantSpectrum <: Spectrum
    poly::RGBSigmoidPolynomial
    scale::Float32
end

# Single wavelength evaluation (matches pbrt-v4's operator())
@propagate_inbounds function (s::RGBIlluminantSpectrum)(lambda::Float32)::Float32
    # Note: sample_d65 is defined in uplift.jl, must be available at runtime
    return s.scale * s.poly(lambda) * sample_d65(lambda)
end

# Maximum value of D65 illuminant (peak at ~560nm where D65 ≈ 100)
# This is used for MaxValue computation matching pbrt-v4's illuminant->MaxValue()
const D65_MAX_VALUE = 100.0f0

"""
    max_value(s::RGBIlluminantSpectrum) -> Float32

Maximum value of the spectrum, matching pbrt-v4's RGBIlluminantSpectrum::MaxValue():
    scale * rsp.MaxValue() * illuminant->MaxValue()
"""
@propagate_inbounds function max_value(s::RGBIlluminantSpectrum)::Float32
    return s.scale * max_value(s.poly) * D65_MAX_VALUE
end

"""
    rgb_illuminant_spectrum(table::RGBToSpectrumTable, r, g, b) -> RGBIlluminantSpectrum

Create an illuminant spectrum from RGB values, matching pbrt-v4's RGBIlluminantSpectrum constructor.
"""
function rgb_illuminant_spectrum(table::RGBToSpectrumTable, r::Float32, g::Float32, b::Float32)::RGBIlluminantSpectrum
    # Find scale factor (matches pbrt-v4: scale = 2 * max(r,g,b))
    m = max(r, g, b)
    if m <= 0
        return RGBIlluminantSpectrum(RGBSigmoidPolynomial(0.0f0, 0.0f0, -1.0f10), 0.0f0)
    end

    # Normalize RGB by scale and get polynomial
    # pbrt-v4: rsp = cs.ToRGBCoeffs(scale ? rgb / scale : RGB(0, 0, 0))
    scale = 2.0f0 * m
    poly = rgb_to_spectrum(table, r / scale, g / scale, b / scale)

    return RGBIlluminantSpectrum(poly, scale)
end

# ============================================================================
# Table Loading
# ============================================================================

# Type alias for CPU table
const RGBToSpectrumTableCPU = RGBToSpectrumTable{Vector{Float32}, Array{Float32, 5}}

# Global table instance (loaded lazily) - always CPU version
const _srgb_table = Ref{Union{Nothing, RGBToSpectrumTableCPU}}(nothing)

"""Load the sRGB spectrum table from raw binary format"""
function load_srgb_table_binary(path::String)::RGBToSpectrumTableCPU
    open(path, "r") do io
        res = read(io, Int32)
        scale = Vector{Float32}(undef, res)
        read!(io, scale)
        coeffs = Array{Float32, 5}(undef, 3, res, res, res, 3)
        read!(io, coeffs)
        return RGBToSpectrumTable(res, scale, coeffs)
    end
end

"""Save the sRGB spectrum table to raw binary format"""
function save_srgb_table_binary(path::String, table::RGBToSpectrumTable)
    open(path, "w") do io
        write(io, table.res)
        write(io, Array(table.scale))   # Convert to CPU array if needed
        write(io, Array(table.coeffs))  # Convert to CPU array if needed
    end
end

"""Load the sRGB spectrum table (generates if not cached)"""
function get_srgb_table()::RGBToSpectrumTableCPU
    if _srgb_table[] === nothing
        bin_path = joinpath(@__DIR__, "srgb_spectrum_table.dat")
        if isfile(bin_path)
            # Load from binary cache
            _srgb_table[] = load_srgb_table_binary(bin_path)
        else
            # Generate and save
            println("Generating sRGB spectrum table (this may take a minute)...")
            include(joinpath(@__DIR__, "rgb2spec_gen.jl"))
            table_data = RGB2SpecGen.generate_rgb2spec_table(64; verbose=true)
            table = RGBToSpectrumTable(Int32(table_data.res), table_data.scale, table_data.coeffs)
            save_srgb_table_binary(bin_path, table)
            _srgb_table[] = table
        end
    end
    return _srgb_table[]
end

# ============================================================================
# GPU Conversion
# ============================================================================

"""
    to_gpu(backend, table::RGBToSpectrumTable) -> RGBToSpectrumTable

Convert RGBToSpectrumTable to use GPU-compatible arrays.
Uses KernelAbstractions backend for allocation.
"""
function to_gpu(backend, table::RGBToSpectrumTable)
    scale_gpu = transfer(backend, table.scale)
    coeffs_gpu = transfer(backend, table.coeffs)
    return RGBToSpectrumTable(table.res, scale_gpu, coeffs_gpu)
end
