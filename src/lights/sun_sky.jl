# Hosek-Wilkie Spectral Sky Model
# Port of ArHosekSkyModel.c from pbrt-v4
# Copyright (c) 2012-2013, Lukas Hosek and Alexander Wilkie (BSD 3-clause)

# Bernstein polynomial evaluation for degree 5 (6 control points)
function _hosek_bernstein5(t::Float64, c0, c1, c2, c3, c4, c5)
    s = 1.0 - t
    return (s^5 * c0 +
            5.0 * s^4 * t * c1 +
            10.0 * s^3 * t^2 * c2 +
            10.0 * s^2 * t^3 * c3 +
            5.0 * s * t^4 * c4 +
            t^5 * c5)
end

# Port of ArHosekSkyModel_CookConfiguration
# dataset: flat array of 1080 Float64 (9 coeffs × 6 elev × 10 turb × 2 albedo)
# Returns 9 config coefficients
function hosek_cook_config(dataset::Vector{Float64}, turbidity::Float64, albedo::Float64, solar_elevation::Float64)
    int_turbidity = clamp(floor(Int, turbidity), 1, 10)
    turbidity_rem = turbidity - Float64(int_turbidity)

    # Remap elevation to cube-root space
    t = (solar_elevation / (π / 2.0))^(1.0 / 3.0)

    config = MVector{9, Float64}(undef)

    # albedo 0, low turbidity
    offset = 9 * 6 * (int_turbidity - 1) + 1  # +1 for Julia 1-indexing
    for i in 1:9
        config[i] = (1.0 - albedo) * (1.0 - turbidity_rem) *
            _hosek_bernstein5(t,
                dataset[offset + i - 1],
                dataset[offset + i - 1 + 9],
                dataset[offset + i - 1 + 18],
                dataset[offset + i - 1 + 27],
                dataset[offset + i - 1 + 36],
                dataset[offset + i - 1 + 45])
    end

    # albedo 1, low turbidity
    offset = 9 * 6 * 10 + 9 * 6 * (int_turbidity - 1) + 1
    for i in 1:9
        config[i] += albedo * (1.0 - turbidity_rem) *
            _hosek_bernstein5(t,
                dataset[offset + i - 1],
                dataset[offset + i - 1 + 9],
                dataset[offset + i - 1 + 18],
                dataset[offset + i - 1 + 27],
                dataset[offset + i - 1 + 36],
                dataset[offset + i - 1 + 45])
    end

    if int_turbidity < 10
        # albedo 0, high turbidity
        offset = 9 * 6 * int_turbidity + 1
        for i in 1:9
            config[i] += (1.0 - albedo) * turbidity_rem *
                _hosek_bernstein5(t,
                    dataset[offset + i - 1],
                    dataset[offset + i - 1 + 9],
                    dataset[offset + i - 1 + 18],
                    dataset[offset + i - 1 + 27],
                    dataset[offset + i - 1 + 36],
                    dataset[offset + i - 1 + 45])
        end

        # albedo 1, high turbidity
        offset = 9 * 6 * 10 + 9 * 6 * int_turbidity + 1
        for i in 1:9
            config[i] += albedo * turbidity_rem *
                _hosek_bernstein5(t,
                    dataset[offset + i - 1],
                    dataset[offset + i - 1 + 9],
                    dataset[offset + i - 1 + 18],
                    dataset[offset + i - 1 + 27],
                    dataset[offset + i - 1 + 36],
                    dataset[offset + i - 1 + 45])
        end
    end

    return NTuple{9, Float64}(config)
end

# Port of ArHosekSkyModel_CookRadianceConfiguration
# dataset: flat array of 120 Float64 (6 elev × 10 turb × 2 albedo)
function hosek_cook_radiance(dataset::Vector{Float64}, turbidity::Float64, albedo::Float64, solar_elevation::Float64)
    int_turbidity = clamp(floor(Int, turbidity), 1, 10)
    turbidity_rem = turbidity - Float64(int_turbidity)

    t = (solar_elevation / (π / 2.0))^(1.0 / 3.0)

    # albedo 0, low turbidity
    offset = 6 * (int_turbidity - 1) + 1
    res = (1.0 - albedo) * (1.0 - turbidity_rem) *
        _hosek_bernstein5(t,
            dataset[offset], dataset[offset+1], dataset[offset+2],
            dataset[offset+3], dataset[offset+4], dataset[offset+5])

    # albedo 1, low turbidity
    offset = 6 * 10 + 6 * (int_turbidity - 1) + 1
    res += albedo * (1.0 - turbidity_rem) *
        _hosek_bernstein5(t,
            dataset[offset], dataset[offset+1], dataset[offset+2],
            dataset[offset+3], dataset[offset+4], dataset[offset+5])

    if int_turbidity < 10
        # albedo 0, high turbidity
        offset = 6 * int_turbidity + 1
        res += (1.0 - albedo) * turbidity_rem *
            _hosek_bernstein5(t,
                dataset[offset], dataset[offset+1], dataset[offset+2],
                dataset[offset+3], dataset[offset+4], dataset[offset+5])

        # albedo 1, high turbidity
        offset = 6 * 10 + 6 * int_turbidity + 1
        res += albedo * turbidity_rem *
            _hosek_bernstein5(t,
                dataset[offset], dataset[offset+1], dataset[offset+2],
                dataset[offset+3], dataset[offset+4], dataset[offset+5])
    end

    return res
end

# Port of ArHosekSkyModel_GetRadianceInternal
function hosek_radiance(config::NTuple{9, Float64}, theta::Float64, gamma::Float64)
    cos_gamma = cos(gamma)
    cos_theta = max(cos(theta), 0.0)

    expM = exp(config[5] * gamma)
    rayM = cos_gamma * cos_gamma
    mieM = (1.0 + cos_gamma * cos_gamma) /
           ((1.0 + config[9] * config[9] - 2.0 * config[9] * cos_gamma)^1.5)
    zenith = sqrt(cos_theta)

    return (1.0 + config[1] * exp(config[2] / (cos_theta + 0.01))) *
           (config[3] + config[4] * expM + config[6] * rayM + config[7] * mieM + config[8] * zenith)
end

# ============================================================================
# Spectral Sky State (11 wavelength bands, 320-720nm)
# Port of ArHosekSkyModelState from pbrt-v4
# ============================================================================

struct HosekState
    configs::NTuple{11, NTuple{9, Float64}}   # 11 spectral band config coefficients
    radiances::NTuple{11, Float64}             # 11 spectral band radiance scalars
    turbidity::Float64
    solar_radius::Float64                      # angular radius in radians (0.51°/2)
    albedo::Float64
    elevation::Float64
end

# Port of arhosekskymodelstate_alloc_init
function HosekState(turbidity::Float64, albedo::Float64, solar_elevation::Float64)
    configs = ntuple(wl -> hosek_cook_config(HOSEK_SPECTRAL_CONFIGS[wl], turbidity, albedo, solar_elevation), 11)
    radiances = ntuple(wl -> hosek_cook_radiance(HOSEK_SPECTRAL_RADIANCES[wl], turbidity, albedo, solar_elevation), 11)
    solar_radius = deg2rad(0.51) / 2.0  # terrestrial sun angular radius
    HosekState(configs, radiances, turbidity, solar_radius, albedo, solar_elevation)
end

# Port of arhosekskymodel_radiance
# Evaluates spectral sky radiance at arbitrary wavelength by interpolating between bands
function hosek_spectral_radiance(state::HosekState, theta::Float64, gamma::Float64, wavelength::Float64)
    low_wl = floor(Int, (wavelength - 320.0) / 40.0)

    if low_wl < 0 || low_wl >= 11
        return 0.0
    end

    interp = mod((wavelength - 320.0) / 40.0, 1.0)

    # Julia: bands are 1-indexed, low_wl is 0-indexed from C
    band = low_wl + 1
    val_low = hosek_radiance(state.configs[band], theta, gamma) * state.radiances[band]

    if interp < 1e-6
        return val_low
    end

    result = (1.0 - interp) * val_low

    if band < 11
        result += interp * hosek_radiance(state.configs[band + 1], theta, gamma) * state.radiances[band + 1]
    end

    return result
end

# Port of arhosekskymodel_sr_internal — solar radiance piecewise polynomial
const _SOLAR_PIECES = 45
const _SOLAR_ORDER = 4

function _hosek_solar_sr_internal(state::HosekState, turbidity_idx::Int, wl_idx::Int, elevation::Float64)
    pos = floor(Int, (2.0 * elevation / π)^(1.0 / 3.0) * _SOLAR_PIECES)
    if pos > 44
        pos = 44
    end

    break_x = (Float64(pos) / Float64(_SOLAR_PIECES))^3.0 * (π * 0.5)

    # Coefficients pointer: C uses 0-indexed turbidity and 0-indexed wl
    # solarDatasets[wl] + (order * pieces * turbidity + order * (pos+1) - 1)
    # In Julia: 1-indexed array, coefs walks backwards
    dataset = HOSEK_SOLAR_DATA[wl_idx]
    base = _SOLAR_ORDER * _SOLAR_PIECES * turbidity_idx + _SOLAR_ORDER * (pos + 1)
    # C code reads coefs[0], coefs[-1], coefs[-2], coefs[-3] (walks backwards)

    x = elevation - break_x
    x_exp = 1.0
    res = 0.0
    for i in 0:(_SOLAR_ORDER - 1)
        res += x_exp * dataset[base - i]  # Julia 1-indexed: base corresponds to C's (base-1 + 1)
        x_exp *= x
    end

    return res  # emission_correction_factor_sun is 1.0
end

# Port of arhosekskymodel_solar_radiance_internal2 — direct solar radiance with limb darkening
function _hosek_solar_radiance_direct(state::HosekState, wavelength::Float64, elevation::Float64, gamma::Float64)
    sol_rad_sin = sin(state.solar_radius)
    ar2 = 1.0 / (sol_rad_sin * sol_rad_sin)
    singamma = sin(gamma)
    sc2 = 1.0 - ar2 * singamma * singamma
    if sc2 < 0.0
        sc2 = 0.0
    end
    sampleCosine = sqrt(sc2)
    if sampleCosine == 0.0
        return 0.0
    end

    turb_low = clamp(floor(Int, state.turbidity) - 1, 0, 8)
    turb_frac = state.turbidity - Float64(turb_low + 1)
    if turb_low == 8 && state.turbidity >= 10.0
        turb_frac = 1.0
    end

    wl_low = floor(Int, (wavelength - 320.0) / 40.0)
    wl_frac = mod(wavelength, 40.0) / 40.0
    if wl_low == 10
        wl_low = 9
        wl_frac = 1.0
    end

    # Julia: wl_idx is 1-indexed, turb_idx is 0-indexed (matching C's array layout)
    wl_idx_lo = wl_low + 1
    wl_idx_hi = min(wl_low + 2, 11)

    direct_radiance =
        (1.0 - turb_frac) * (
            (1.0 - wl_frac) * _hosek_solar_sr_internal(state, turb_low, wl_idx_lo, elevation) +
            wl_frac * _hosek_solar_sr_internal(state, turb_low, wl_idx_hi, elevation)
        ) +
        turb_frac * (
            (1.0 - wl_frac) * _hosek_solar_sr_internal(state, turb_low + 1, wl_idx_lo, elevation) +
            wl_frac * _hosek_solar_sr_internal(state, turb_low + 1, wl_idx_hi, elevation)
        )

    # Limb darkening: interpolate coefficients between wavelength bands
    ld_lo = HOSEK_LIMB_DARKENING[wl_idx_lo]
    ld_hi = HOSEK_LIMB_DARKENING[wl_idx_hi]
    ldCoeff = ntuple(i -> (1.0 - wl_frac) * ld_lo[i] + wl_frac * ld_hi[i], 6)

    # 5th order polynomial limb darkening
    darkeningFactor = ldCoeff[1] +
        ldCoeff[2] * sampleCosine +
        ldCoeff[3] * sampleCosine^2 +
        ldCoeff[4] * sampleCosine^3 +
        ldCoeff[5] * sampleCosine^4 +
        ldCoeff[6] * sampleCosine^5

    return direct_radiance * darkeningFactor
end

# Port of arhosekskymodel_solar_radiance — sky + sun disk combined
function hosek_solar_radiance(state::HosekState, theta::Float64, gamma::Float64, wavelength::Float64)
    # Direct solar radiance (limb-darkened sun disk)
    elevation = (π / 2.0) - theta
    direct = _hosek_solar_radiance_direct(state, wavelength, elevation, gamma)
    # Inscattered sky radiance
    inscattered = hosek_spectral_radiance(state, theta, gamma, wavelength)
    return direct + inscattered
end

# ============================================================================
# Spectral → XYZ conversion for sky baking
# Matches pbrt-v4's SpectrumToXYZ: XYZ = InnerProduct(CMF, spectrum) / CIE_Y_integral
# ============================================================================

# Evaluate piecewise linear spectrum at a single wavelength
function _piecewise_linear_eval(lambdas::Vector{Float64}, values::Vector{Float64}, lambda::Float64)
    n = length(lambdas)
    if lambda <= lambdas[1]
        return values[1]
    end
    if lambda >= lambdas[n]
        return values[n]
    end
    # Binary search for interval
    lo = 1
    hi = n
    while hi - lo > 1
        mid = (lo + hi) >> 1
        if lambdas[mid] <= lambda
            lo = mid
        else
            hi = mid
        end
    end
    t = (lambda - lambdas[lo]) / (lambdas[hi] - lambdas[lo])
    return (1.0 - t) * values[lo] + t * values[hi]
end

# Convert spectral sky samples to XYZ, matching pbrt-v4's SpectrumToXYZ
# Integrates piecewise linear spectrum against CIE CMFs from 360-830nm at 1nm steps
function _spectrum_to_xyz(lambdas::Vector{Float64}, values::Vector{Float64})
    x_sum = 0.0
    y_sum = 0.0
    z_sum = 0.0
    for i in 1:N_CIE_SAMPLES
        lambda = Float64(CIE_LAMBDA_MIN + i - 1)
        s = _piecewise_linear_eval(lambdas, values, lambda)
        x_sum += Float64(CIE_X[i]) * s
        y_sum += Float64(CIE_Y[i]) * s
        z_sum += Float64(CIE_Z[i]) * s
    end
    return (x_sum / Float64(CIE_Y_INTEGRAL),
            y_sum / Float64(CIE_Y_INTEGRAL),
            z_sum / Float64(CIE_Y_INTEGRAL))
end

# ============================================================================
# Pre-bake Hosek-Wilkie sky to EnvironmentLight (pbrt-v4 approach)
# ============================================================================

"""
    sunsky_to_envlight(; direction, intensity=1f0, turbidity=2.5f0, ...) -> (EnvironmentLight, SunLight)

Pre-bake the Hosek-Wilkie spectral sky model into an equal-area EnvironmentMap and create
a separate SunLight for the sun disk. Matches pbrt-v4's `makesky` approach: evaluate the
spectral model at 13 wavelengths, convert to XYZ via CIE color matching functions (dividing
by CIE_Y_integral), then to sRGB for storage.

# Arguments
- `direction::Vec3f`: Direction TO the sun (normalized internally)
- `intensity::Float32 = 1f0`: Overall brightness multiplier (scale parameter, default 1.0)
- `turbidity::Float32 = 2.5f0`: Atmospheric turbidity (1=clear, 10=hazy)
- `ground_albedo::RGBSpectrum = RGBSpectrum(0.3f0)`: Ground color below horizon
- `ground_enabled::Bool = true`: Whether to show ground below horizon
- `resolution::Int = 512`: Resolution of equal-area square map

# Returns
A tuple of `(EnvironmentLight, SunLight)` to be added to the scene.
"""
function sunsky_to_envlight(;
    direction::Vec3f,
    intensity::Float32 = 1f0,
    turbidity::Float32 = 2.5f0,
    ground_albedo::RGBSpectrum = RGBSpectrum(0.3f0),
    ground_enabled::Bool = true,
    resolution::Int = 512,
)
    dir = normalize(direction)

    # Sun elevation = angle above horizon (complement of zenith angle)
    # dir[3] = cos(zenith_angle) = sin(elevation)
    solar_elevation = Float64(asin(clamp(dir[3], 0f0, 1f0)))

    # Initialize Hosek-Wilkie spectral state (11 bands, 320-720nm)
    state = HosekState(Float64(turbidity), Float64(0.5), solar_elevation)

    # Spectral sampling wavelengths — matches pbrt-v4's makesky:
    # 13 points from 320nm to 720nm, uniformly spaced
    n_lambda = 1 + div(720 - 320, 32)  # = 13
    sample_lambdas = [320.0 + i * (720.0 - 320.0) / (n_lambda - 1) for i in 0:(n_lambda - 1)]

    # Bake sky to equal-area octahedral map
    sky_data = Matrix{RGBSpectrum}(undef, resolution, resolution)
    sky_values = Vector{Float64}(undef, n_lambda)

    for v_idx in 1:resolution
        for u_idx in 1:resolution
            uv = Point2f((u_idx - 0.5f0) / resolution, (v_idx - 0.5f0) / resolution)

            # Convert to world direction via equal-area mapping (Z-up)
            wi = equal_area_square_to_sphere(uv)

            if wi[3] <= 0f0 && ground_enabled
                sky_data[v_idx, u_idx] = ground_albedo * 0.3f0
                continue
            end

            # Zenith angle (clamp to horizon for below-horizon when ground disabled)
            theta = Float64(acos(clamp(wi[3], 0f0, 1f0)))
            # Angle between view direction and sun
            cos_gamma = clamp(dot(wi, dir), -1f0, 1f0)
            gamma = Float64(acos(cos_gamma))

            # Evaluate spectral sky at all sample wavelengths
            # Using sky-only radiance (no sun disk — sun is a separate SunLight)
            for i in 1:n_lambda
                sky_values[i] = hosek_spectral_radiance(state, theta, gamma, sample_lambdas[i])
            end

            # Convert spectrum to XYZ then sRGB
            # Matches pbrt-v4: SpectrumToXYZ divides by CIE_Y_integral
            x, y, z = _spectrum_to_xyz(sample_lambdas, sky_values)
            rgb = xyz_to_linear_srgb(Vec3f(Float32(x), Float32(y), Float32(z)))

            sky_data[v_idx, u_idx] = RGBSpectrum(
                max(0f0, rgb[1]),
                max(0f0, rgb[2]),
                max(0f0, rgb[3]))
        end
    end

    # Build EnvironmentMap with luminance-weighted importance sampling distribution
    env_map = EnvironmentMap(sky_data)
    # sample_light_spectral uplifts env map pixels via D65 illuminant (uplift_rgb_illuminant).
    # Standard photometric normalization cancels the D65 factor: scale = intensity / D65_PHOTOMETRIC.
    env_light = EnvironmentLight(env_map, RGBSpectrum(intensity / D65_PHOTOMETRIC))

    # Sun as separate delta directional light
    # SunLight direction = direction light TRAVELS (away from sun), so negate
    # Sun:sky illuminance ratio ≈ 5:1 for clear conditions
    sun_scale = 5f0 * intensity
    sun_rgb = RGB{Float32}(sun_scale, sun_scale * 0.95f0, sun_scale * 0.85f0)
    sun_light = SunLight(sun_rgb, -dir)

    return env_light, sun_light
end
