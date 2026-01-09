# RGB to Spectrum Table Generator
# Port of pbrt-v4's rgb2spec_opt.cpp to Julia
# Generates lookup tables for RGBSigmoidPolynomial coefficients

module RGB2SpecGen

using LinearAlgebra
using Base.Threads

export generate_rgb2spec_table, RGBToSpectrumTableData

# ============================================================================
# CIE 1931 Color Matching Functions (5nm intervals, 360-830nm)
# ============================================================================

const CIE_LAMBDA_MIN = 360.0
const CIE_LAMBDA_MAX = 830.0
const CIE_SAMPLES = 95

const CIE_X = Float64[
    0.000129900000, 0.000232100000, 0.000414900000, 0.000741600000, 0.001368000000,
    0.002236000000, 0.004243000000, 0.007650000000, 0.014310000000, 0.023190000000,
    0.043510000000, 0.077630000000, 0.134380000000, 0.214770000000, 0.283900000000,
    0.328500000000, 0.348280000000, 0.348060000000, 0.336200000000, 0.318700000000,
    0.290800000000, 0.251100000000, 0.195360000000, 0.142100000000, 0.095640000000,
    0.057950010000, 0.032010000000, 0.014700000000, 0.004900000000, 0.002400000000,
    0.009300000000, 0.029100000000, 0.063270000000, 0.109600000000, 0.165500000000,
    0.225749900000, 0.290400000000, 0.359700000000, 0.433449900000, 0.512050100000,
    0.594500000000, 0.678400000000, 0.762100000000, 0.842500000000, 0.916300000000,
    0.978600000000, 1.026300000000, 1.056700000000, 1.062200000000, 1.045600000000,
    1.002600000000, 0.938400000000, 0.854449900000, 0.751400000000, 0.642400000000,
    0.541900000000, 0.447900000000, 0.360800000000, 0.283500000000, 0.218700000000,
    0.164900000000, 0.121200000000, 0.087400000000, 0.063600000000, 0.046770000000,
    0.032900000000, 0.022700000000, 0.015840000000, 0.011359160000, 0.008110916000,
    0.005790346000, 0.004109457000, 0.002899327000, 0.002049190000, 0.001439971000,
    0.000999949300, 0.000690078600, 0.000476021300, 0.000332301100, 0.000234826100,
    0.000166150500, 0.000117413000, 0.000083075270, 0.000058706520, 0.000041509940,
    0.000029353260, 0.000020673830, 0.000014559770, 0.000010253980, 0.000007221456,
    0.000005085868, 0.000003581652, 0.000002522525, 0.000001776509, 0.000001251141
]

const CIE_Y = Float64[
    0.000003917000, 0.000006965000, 0.000012390000, 0.000022020000, 0.000039000000,
    0.000064000000, 0.000120000000, 0.000217000000, 0.000396000000, 0.000640000000,
    0.001210000000, 0.002180000000, 0.004000000000, 0.007300000000, 0.011600000000,
    0.016840000000, 0.023000000000, 0.029800000000, 0.038000000000, 0.048000000000,
    0.060000000000, 0.073900000000, 0.090980000000, 0.112600000000, 0.139020000000,
    0.169300000000, 0.208020000000, 0.258600000000, 0.323000000000, 0.407300000000,
    0.503000000000, 0.608200000000, 0.710000000000, 0.793200000000, 0.862000000000,
    0.914850100000, 0.954000000000, 0.980300000000, 0.994950100000, 1.000000000000,
    0.995000000000, 0.978600000000, 0.952000000000, 0.915400000000, 0.870000000000,
    0.816300000000, 0.757000000000, 0.694900000000, 0.631000000000, 0.566800000000,
    0.503000000000, 0.441200000000, 0.381000000000, 0.321000000000, 0.265000000000,
    0.217000000000, 0.175000000000, 0.138200000000, 0.107000000000, 0.081600000000,
    0.061000000000, 0.044580000000, 0.032000000000, 0.023200000000, 0.017000000000,
    0.011920000000, 0.008210000000, 0.005723000000, 0.004102000000, 0.002929000000,
    0.002091000000, 0.001484000000, 0.001047000000, 0.000740000000, 0.000520000000,
    0.000361100000, 0.000249200000, 0.000171900000, 0.000120000000, 0.000084800000,
    0.000060000000, 0.000042400000, 0.000030000000, 0.000021200000, 0.000014990000,
    0.000010600000, 0.000007465700, 0.000005257800, 0.000003702900, 0.000002607800,
    0.000001836600, 0.000001293400, 0.000000910930, 0.000000641530, 0.000000451810
]

const CIE_Z = Float64[
    0.000606100000, 0.001086000000, 0.001946000000, 0.003486000000, 0.006450001000,
    0.010549990000, 0.020050010000, 0.036210000000, 0.067850010000, 0.110200000000,
    0.207400000000, 0.371300000000, 0.645600000000, 1.039050100000, 1.385600000000,
    1.622960000000, 1.747060000000, 1.782600000000, 1.772110000000, 1.744100000000,
    1.669200000000, 1.528100000000, 1.287640000000, 1.041900000000, 0.812950100000,
    0.616200000000, 0.465180000000, 0.353300000000, 0.272000000000, 0.212300000000,
    0.158200000000, 0.111700000000, 0.078249990000, 0.057250010000, 0.042160000000,
    0.029840000000, 0.020300000000, 0.013400000000, 0.008749999000, 0.005749999000,
    0.003900000000, 0.002749999000, 0.002100000000, 0.001800000000, 0.001650001000,
    0.001400000000, 0.001100000000, 0.001000000000, 0.000800000000, 0.000600000000,
    0.000340000000, 0.000240000000, 0.000190000000, 0.000100000000, 0.000049999990,
    0.000030000000, 0.000020000000, 0.000010000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
    0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000
]

# D65 illuminant (normalized to unit luminance)
const CIE_D65_NORM = 10566.864005283874576
const CIE_D65 = Float64[
    46.6383, 49.3637, 52.0891, 51.0323, 49.9755, 52.3118, 54.6482,
    68.7015, 82.7549, 87.1204, 91.486,  92.4589, 93.4318, 90.057,
    86.6823, 95.7736, 104.865, 110.936, 117.008, 117.41,  117.812,
    116.336, 114.861, 115.392, 115.923, 112.367, 108.811, 109.082,
    109.354, 108.578, 107.802, 106.296, 104.79,  106.239, 107.689,
    106.047, 104.405, 104.225, 104.046, 102.023, 100.0,   98.1671,
    96.3342, 96.0611, 95.788,  92.2368, 88.6856, 89.3459, 90.0062,
    89.8026, 89.5991, 88.6489, 87.6987, 85.4936, 83.2886, 83.4939,
    83.6992, 81.863,  80.0268, 80.1207, 80.2146, 81.2462, 82.2778,
    80.281,  78.2842, 74.0027, 69.7213, 70.6652, 71.6091, 72.979,
    74.349,  67.9765, 61.604,  65.7448, 69.8856, 72.4863, 75.087,
    69.3398, 63.5927, 55.0054, 46.4182, 56.6118, 66.8054, 65.0941,
    63.3828, 63.8434, 64.304,  61.8779, 59.4519, 55.7054, 51.959,
    54.6998, 57.4406, 58.8765, 60.3125
] ./ CIE_D65_NORM

# ============================================================================
# Color Space Matrices
# ============================================================================

const XYZ_TO_SRGB = [
    3.240479  -1.537150  -0.498535;
   -0.969256   1.875991   0.041556;
    0.055648  -0.204043   1.057311
]

const SRGB_TO_XYZ = [
    0.412453  0.357580  0.180423;
    0.212671  0.715160  0.072169;
    0.019334  0.119193  0.950227
]

# ============================================================================
# Helper Functions
# ============================================================================

"""Linearly interpolate CIE data at arbitrary wavelength"""
function cie_interp(data::Vector{Float64}, lambda::Float64)::Float64
    x = lambda - CIE_LAMBDA_MIN
    x *= (CIE_SAMPLES - 1) / (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN)
    offset = floor(Int, x)
    offset = clamp(offset, 0, CIE_SAMPLES - 2)
    weight = x - offset
    return (1.0 - weight) * data[offset + 1] + weight * data[offset + 2]
end

"""Sigmoid function for spectrum representation"""
@inline sigmoid(x::Float64)::Float64 = 0.5 * x / sqrt(1.0 + x * x) + 0.5

"""Smoothstep for scale table"""
@inline smoothstep(x::Float64)::Float64 = x * x * (3.0 - 2.0 * x)

"""Convert RGB to CIE Lab for perceptual error metric"""
function rgb_to_lab(rgb::Vector{Float64}, rgb_to_xyz::Matrix{Float64}, xyz_whitepoint::Vector{Float64})::Vector{Float64}
    # RGB to XYZ
    xyz = rgb_to_xyz * rgb

    # XYZ to Lab
    Xw, Yw, Zw = xyz_whitepoint

    f(t) = t > (6.0/29.0)^3 ? cbrt(t) : t / (3.0 * (6.0/29.0)^2) + 4.0/29.0

    L = 116.0 * f(xyz[2] / Yw) - 16.0
    a = 500.0 * (f(xyz[1] / Xw) - f(xyz[2] / Yw))
    b = 200.0 * (f(xyz[2] / Yw) - f(xyz[3] / Zw))

    return [L, a, b]
end

# ============================================================================
# Precomputed Tables for Integration
# ============================================================================

struct IntegrationTables
    lambda::Vector{Float64}
    rgb_weights::Matrix{Float64}  # 3 x n_samples
    xyz_whitepoint::Vector{Float64}
    rgb_to_xyz::Matrix{Float64}
    xyz_to_rgb::Matrix{Float64}
end

"""Initialize integration tables for a color space"""
function init_tables(; xyz_to_rgb::Matrix{Float64}=XYZ_TO_SRGB,
                      rgb_to_xyz::Matrix{Float64}=SRGB_TO_XYZ,
                      illuminant::Vector{Float64}=CIE_D65)::IntegrationTables
    # Fine sampling for Simpson's 3/8 rule
    n_fine = (CIE_SAMPLES - 1) * 3 + 1
    h = (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN) / (n_fine - 1)

    lambda = Vector{Float64}(undef, n_fine)
    rgb_weights = zeros(Float64, 3, n_fine)
    xyz_whitepoint = zeros(Float64, 3)

    for i in 1:n_fine
        λ = CIE_LAMBDA_MIN + (i - 1) * h
        lambda[i] = λ

        # Get CIE values at this wavelength
        xyz = [cie_interp(CIE_X, λ), cie_interp(CIE_Y, λ), cie_interp(CIE_Z, λ)]
        I = cie_interp(illuminant, λ)

        # Simpson's 3/8 rule weights
        weight = 3.0 / 8.0 * h
        if i == 1 || i == n_fine
            # Endpoints: weight = 1
        elseif (i - 2) % 3 == 2
            weight *= 2.0
        else
            weight *= 3.0
        end

        # Precompute RGB weights for integration
        for k in 1:3
            for j in 1:3
                rgb_weights[k, i] += xyz_to_rgb[k, j] * xyz[j] * I * weight
            end
        end

        # Accumulate whitepoint
        xyz_whitepoint .+= xyz .* I .* weight
    end

    return IntegrationTables(lambda, rgb_weights, xyz_whitepoint, rgb_to_xyz, xyz_to_rgb)
end

# ============================================================================
# Gauss-Newton Optimization
# ============================================================================

const RGB2SPEC_EPSILON = 1e-4

"""Evaluate residual between target RGB and spectrum-derived RGB"""
function eval_residual!(residual::Vector{Float64}, coeffs::Vector{Float64},
                        target_rgb::Vector{Float64}, tables::IntegrationTables)
    out_rgb = zeros(Float64, 3)
    n = length(tables.lambda)

    @_inbounds for i in 1:n
        # Normalized wavelength [0, 1]
        λ_norm = (tables.lambda[i] - CIE_LAMBDA_MIN) / (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN)

        # Evaluate polynomial: c0*λ² + c1*λ + c2
        x = coeffs[1] * λ_norm * λ_norm + coeffs[2] * λ_norm + coeffs[3]

        # Apply sigmoid
        s = sigmoid(x)

        # Integrate against precomputed RGB curves
        for j in 1:3
            out_rgb[j] += tables.rgb_weights[j, i] * s
        end
    end

    # Convert both to Lab for perceptual comparison
    out_lab = rgb_to_lab(out_rgb, tables.rgb_to_xyz, tables.xyz_whitepoint)
    target_lab = rgb_to_lab(copy(target_rgb), tables.rgb_to_xyz, tables.xyz_whitepoint)

    residual .= target_lab .- out_lab
    return nothing
end

"""Evaluate Jacobian using finite differences"""
function eval_jacobian!(jac::Matrix{Float64}, coeffs::Vector{Float64},
                        target_rgb::Vector{Float64}, tables::IntegrationTables)
    r0 = zeros(Float64, 3)
    r1 = zeros(Float64, 3)
    tmp = copy(coeffs)

    for i in 1:3
        tmp .= coeffs
        tmp[i] -= RGB2SPEC_EPSILON
        eval_residual!(r0, tmp, target_rgb, tables)

        tmp .= coeffs
        tmp[i] += RGB2SPEC_EPSILON
        eval_residual!(r1, tmp, target_rgb, tables)

        for j in 1:3
            jac[j, i] = (r1[j] - r0[j]) / (2 * RGB2SPEC_EPSILON)
        end
    end
    return nothing
end

"""Gauss-Newton optimization to find polynomial coefficients for target RGB"""
function gauss_newton!(coeffs::Vector{Float64}, target_rgb::Vector{Float64},
                       tables::IntegrationTables; max_iter::Int=15)
    residual = zeros(Float64, 3)
    jac = zeros(Float64, 3, 3)

    for _ in 1:max_iter
        eval_residual!(residual, coeffs, target_rgb, tables)
        eval_jacobian!(jac, coeffs, target_rgb, tables)

        # Solve J * x = residual using LU decomposition
        try
            x = jac \ residual
            coeffs .-= x

            # Clamp coefficients to prevent explosion
            max_coeff = maximum(abs, coeffs)
            if max_coeff > 200.0
                coeffs .*= 200.0 / max_coeff
            end

            # Check convergence
            r = sum(abs2, residual)
            if r < 1e-6
                break
            end
        catch
            # LU decomposition failed, keep current coefficients
            break
        end
    end
    return nothing
end

# ============================================================================
# Table Generation
# ============================================================================

"""
RGB to Spectrum lookup table data.

The table is indexed as: coeffs[maxc, z, y, x, coeff_idx]
where maxc ∈ {1,2,3} (R,G,B), z,y,x ∈ 1:res, coeff_idx ∈ {1,2,3}
"""
struct RGBToSpectrumTableData
    res::Int
    scale::Vector{Float32}
    coeffs::Array{Float32, 5}  # [3, res, res, res, 3]
end

"""
Generate RGB to spectrum lookup table.

# Arguments
- `res::Int`: Resolution of the table (default 64, like pbrt-v4)

# Returns
- `RGBToSpectrumTableData`: The lookup table
"""
function generate_rgb2spec_table(res::Int=64; verbose::Bool=true)::RGBToSpectrumTableData
    verbose && println("Initializing integration tables...")
    tables = init_tables()

    # Generate scale table (smoothstepped twice for better distribution)
    scale = Float32[smoothstep(smoothstep(k / (res - 1))) for k in 0:(res-1)]

    # Output array: [maxc, z, y, x, coeff]
    # maxc: which RGB component is largest (1=R, 2=G, 3=B)
    coeffs = zeros(Float32, 3, res, res, res, 3)

    verbose && println("Optimizing spectra for $(res)^3 × 3 = $(res^3 * 3) RGB values...")

    # Process each "max component" slice
    for l in 1:3
        verbose && println("  Processing max component $l/3...")

        # Parallel over y
        Threads.@threads for j in 1:res
            y = (j - 1) / (res - 1)

            # Working arrays for this thread
            opt_coeffs = zeros(Float64, 3)
            rgb = zeros(Float64, 3)

            for i in 1:res
                x = (i - 1) / (res - 1)

                # Start from middle and work outward
                start_k = res ÷ 5

                # Forward pass (start_k to res)
                fill!(opt_coeffs, 0.0)
                for k in (start_k + 1):res
                    b = Float64(scale[k])

                    rgb[l] = b
                    rgb[mod1(l + 1, 3)] = x * b
                    rgb[mod1(l + 2, 3)] = y * b

                    gauss_newton!(opt_coeffs, rgb, tables)

                    # Convert coefficients from normalized to actual wavelength scale
                    c0 = 360.0
                    c1 = 1.0 / (830.0 - 360.0)
                    A, B, C = opt_coeffs

                    coeffs[l, k, j, i, 1] = Float32(A * c1^2)
                    coeffs[l, k, j, i, 2] = Float32(B * c1 - 2 * A * c0 * c1^2)
                    coeffs[l, k, j, i, 3] = Float32(C - B * c0 * c1 + A * (c0 * c1)^2)
                end

                # Backward pass (start_k to 1)
                fill!(opt_coeffs, 0.0)
                for k in (start_k + 1):-1:1
                    b = Float64(scale[k])

                    rgb[l] = b
                    rgb[mod1(l + 1, 3)] = x * b
                    rgb[mod1(l + 2, 3)] = y * b

                    gauss_newton!(opt_coeffs, rgb, tables)

                    c0 = 360.0
                    c1 = 1.0 / (830.0 - 360.0)
                    A, B, C = opt_coeffs

                    coeffs[l, k, j, i, 1] = Float32(A * c1^2)
                    coeffs[l, k, j, i, 2] = Float32(B * c1 - 2 * A * c0 * c1^2)
                    coeffs[l, k, j, i, 3] = Float32(C - B * c0 * c1 + A * (c0 * c1)^2)
                end
            end
        end
    end

    verbose && println("Done!")
    return RGBToSpectrumTableData(res, scale, coeffs)
end

"""Save table to a Julia-loadable file"""
function save_table(filename::String, table::RGBToSpectrumTableData)
    open(filename, "w") do f
        # Write as Julia code that can be included
        println(f, "# Auto-generated RGB to Spectrum lookup table")
        println(f, "# Resolution: $(table.res)")
        println(f, "")
        println(f, "const RGB2SPEC_RES = $(table.res)")
        println(f, "")
        println(f, "const RGB2SPEC_SCALE = Float32[")
        for (i, s) in enumerate(table.scale)
            print(f, "    $s,")
            i % 8 == 0 && println(f)
        end
        println(f, "\n]")
        println(f, "")
        println(f, "# Coefficients array: [maxc, z, y, x, coeff]")
        println(f, "const RGB2SPEC_COEFFS = reshape(Float32[")

        # Flatten and write coefficients
        count = 0
        for val in table.coeffs
            print(f, "$val, ")
            count += 1
            count % 8 == 0 && println(f)
        end
        println(f, "\n], $(size(table.coeffs)))")
    end
end

"""Load table from a saved file or binary"""
function load_table(filename::String)::RGBToSpectrumTableData
    include(filename)
    return RGBToSpectrumTableData(RGB2SPEC_RES, RGB2SPEC_SCALE, RGB2SPEC_COEFFS)
end

end # module
