"""
SunSkyLight - Combined sun and procedural sky light using atmospheric scattering.

Provides both:
- Directional sun illumination via sample_li() for surface/volume lighting
- Procedural sky background via le() for rays that escape the scene

The sky color is computed from the sun position using a simplified atmospheric
scattering model (Rayleigh + Mie).

Uses importance sampling based on pre-computed sky radiance distribution for
efficient sampling with low variance.
"""
struct SunSkyLight{S<:Spectrum, D} <: Light
    # Sun parameters
    sun_direction::Vec3f      # Direction TO the sun (normalized)
    sun_intensity::S          # Sun radiance
    sun_angular_radius::Float32  # Angular radius in radians (~0.00465 for real sun)

    # Atmosphere parameters
    turbidity::Float32        # Atmospheric turbidity (2.0 = clear, 10.0 = hazy)
    ground_albedo::S          # Ground color for hemisphere below horizon
    ground_enabled::Bool      # Whether to show ground plane below horizon

    # Precomputed sky coefficients (Preetham model)
    # Perez function coefficients for luminance Y and chromaticity x, y
    perez_Y::NTuple{5, Float32}
    perez_x::NTuple{5, Float32}
    perez_y::NTuple{5, Float32}

    # Zenith values
    zenith_Y::Float32
    zenith_x::Float32
    zenith_y::Float32

    # Importance sampling distribution for the sky hemisphere
    distribution::D

    """Inner constructor for pre-computed lights (used by Adapt)."""
    function SunSkyLight(
        sun_direction::Vec3f,
        sun_intensity::S,
        sun_angular_radius::Float32,
        turbidity::Float32,
        ground_albedo::S,
        ground_enabled::Bool,
        perez_Y::NTuple{5, Float32},
        perez_x::NTuple{5, Float32},
        perez_y::NTuple{5, Float32},
        zenith_Y::Float32,
        zenith_x::Float32,
        zenith_y::Float32,
        distribution::D,
    ) where {S<:Spectrum, D}
        new{S, D}(
            sun_direction, sun_intensity, sun_angular_radius,
            turbidity, ground_albedo, ground_enabled,
            perez_Y, perez_x, perez_y, zenith_Y, zenith_x, zenith_y,
            distribution,
        )
    end

    function SunSkyLight(
        sun_direction::Vec3f,
        sun_intensity::S;
        turbidity::Float32 = 2.5f0,
        ground_albedo::S = RGBSpectrum(0.3f0),
        ground_enabled::Bool = true,
        sun_angular_radius::Float32 = 0.00465f0,  # Real sun ~0.53 degrees
        distribution_resolution::Int = 128,  # Resolution for importance sampling grid
    ) where S<:Spectrum
        dir = normalize(sun_direction)

        # Sun elevation angle (theta_s is angle from zenith)
        theta_s = acos(clamp(dir[3], -1f0, 1f0))

        # Compute Preetham sky model coefficients
        perez_Y, perez_x, perez_y = compute_perez_coefficients(turbidity)
        zenith_Y, zenith_x, zenith_y = compute_zenith_values(turbidity, theta_s)

        # Build importance sampling distribution for the sky hemisphere
        # We use equirectangular-like mapping for the upper hemisphere:
        # u ∈ [0,1] -> φ ∈ [0, 2π] (azimuth)
        # v ∈ [0,1] -> θ ∈ [0, π/2] (zenith angle, 0=up, π/2=horizon)
        n_phi = distribution_resolution
        n_theta = distribution_resolution ÷ 2  # Half resolution for hemisphere

        radiance_grid = Matrix{Float32}(undef, n_theta, n_phi)

        for j in 1:n_theta
            # θ goes from 0 (zenith) to π/2 (horizon)
            # Use center of each bin
            θ = Float32(π / 2) * (j - 0.5f0) / n_theta
            sin_θ = sin(θ)

            for i in 1:n_phi
                # φ goes from 0 to 2π
                φ = 2f0 * Float32(π) * (i - 0.5f0) / n_phi

                # Convert to direction (Z-up)
                direction = Vec3f(sin_θ * cos(φ), sin_θ * sin(φ), cos(θ))

                # Compute sky + sun radiance for this direction
                # We need to compute this without using the light struct yet
                sky_rad = _compute_sky_radiance(
                    direction, dir, perez_Y, perez_x, perez_y,
                    zenith_Y, zenith_x, zenith_y, ground_albedo, ground_enabled,
                )
                sun_rad = _compute_sun_disk_radiance(
                    direction, dir, sun_intensity, sun_angular_radius,
                )
                total_rad = sky_rad + sun_rad

                # Weight by sin(θ) for solid angle and luminance
                # This accounts for the fact that near-horizon directions
                # cover more solid angle in equirectangular mapping
                radiance_grid[j, i] = to_Y(total_rad) * sin_θ
            end
        end

        distribution = Distribution2D(radiance_grid)

        new{S, typeof(distribution)}(
            dir, sun_intensity, sun_angular_radius,
            turbidity, ground_albedo, ground_enabled,
            perez_Y, perez_x, perez_y,
            zenith_Y, zenith_x, zenith_y,
            distribution,
        )
    end
end

# SunSky lights are infinite (provide background sky)
is_infinite_light(::SunSkyLight) = true

"""
Internal function to compute sky radiance without light struct.
Used during construction before the struct exists.
"""
function _compute_sky_radiance(
    direction::Vec3f,
    sun_direction::Vec3f,
    perez_Y::NTuple{5, Float32},
    perez_x::NTuple{5, Float32},
    perez_y::NTuple{5, Float32},
    zenith_Y::Float32,
    zenith_x::Float32,
    zenith_y::Float32,
    ground_albedo::S,
    ground_enabled::Bool,
) where S<:Spectrum
    # Below horizon - return ground albedo if enabled
    if direction[3] <= 0f0 && ground_enabled
        return ground_albedo * 0.3f0
    end

    # Zenith angle of view direction
    theta = acos(clamp(direction[3], 0f0, 1f0))

    # Angle between view direction and sun
    cos_gamma = clamp(dot(direction, sun_direction), -1f0, 1f0)
    gamma = acos(cos_gamma)

    # Sun's zenith angle
    theta_s = acos(clamp(sun_direction[3], 0f0, 1f0))

    # Compute relative luminance using Perez function
    perez_ratio_Y = perez(theta, gamma, perez_Y) / perez(0f0, theta_s, perez_Y)
    Y = zenith_Y * perez_ratio_Y

    # Compute chromaticity
    perez_ratio_x = perez(theta, gamma, perez_x) / perez(0f0, theta_s, perez_x)
    perez_ratio_y = perez(theta, gamma, perez_y) / perez(0f0, theta_s, perez_y)
    x = zenith_x * perez_ratio_x
    y = zenith_y * perez_ratio_y

    # Convert xyY to XYZ
    Y_scaled = Y * 0.04f0
    X = (x / y) * Y_scaled
    Z = ((1f0 - x - y) / y) * Y_scaled

    # Convert XYZ to RGB
    r =  3.2406f0 * X - 1.5372f0 * Y_scaled - 0.4986f0 * Z
    g = -0.9689f0 * X + 1.8758f0 * Y_scaled + 0.0415f0 * Z
    b =  0.0557f0 * X - 0.2040f0 * Y_scaled + 1.0570f0 * Z

    RGBSpectrum(max(0f0, r), max(0f0, g), max(0f0, b))
end

"""
Internal function to compute sun disk radiance without light struct.
"""
function _compute_sun_disk_radiance(
    direction::Vec3f,
    sun_direction::Vec3f,
    sun_intensity::S,
    sun_angular_radius::Float32,
) where S<:Spectrum
    cos_angle = dot(direction, sun_direction)
    angle = acos(clamp(cos_angle, -1f0, 1f0))

    if angle < sun_angular_radius
        return sun_intensity
    elseif angle < sun_angular_radius * 2f0
        t = (angle - sun_angular_radius) / sun_angular_radius
        return sun_intensity * (1f0 - t) * 0.5f0
    else
        return RGBSpectrum(0f0)
    end
end

"""
Compute Perez function coefficients for given turbidity.
Based on Preetham et al. "A Practical Analytic Model for Daylight"
"""
function compute_perez_coefficients(T::Float32)
    # Coefficients for Y (luminance)
    perez_Y = (
        0.1787f0 * T - 1.4630f0,
        -0.3554f0 * T + 0.4275f0,
        -0.0227f0 * T + 5.3251f0,
        0.1206f0 * T - 2.5771f0,
        -0.0670f0 * T + 0.3703f0,
    )

    # Coefficients for x (chromaticity)
    perez_x = (
        -0.0193f0 * T - 0.2592f0,
        -0.0665f0 * T + 0.0008f0,
        -0.0004f0 * T + 0.2125f0,
        -0.0641f0 * T - 0.8989f0,
        -0.0033f0 * T + 0.0452f0,
    )

    # Coefficients for y (chromaticity)
    perez_y = (
        -0.0167f0 * T - 0.2608f0,
        -0.0950f0 * T + 0.0092f0,
        -0.0079f0 * T + 0.2102f0,
        -0.0441f0 * T - 1.6537f0,
        -0.0109f0 * T + 0.0529f0,
    )

    perez_Y, perez_x, perez_y
end

"""
Compute zenith luminance and chromaticity for given turbidity and sun angle.
theta_s is the sun's zenith angle (0 = sun at zenith, π/2 = sun at horizon).
"""
function compute_zenith_values(T::Float32, theta_s::Float32)
    # Zenith luminance (in kcd/m²)
    chi = (4f0/9f0 - T/120f0) * (Float32(π) - 2f0 * theta_s)
    zenith_Y = (4.0453f0 * T - 4.9710f0) * tan(chi) - 0.2155f0 * T + 2.4192f0
    zenith_Y = max(0f0, zenith_Y)

    # Zenith chromaticity
    T2 = T * T
    theta_s2 = theta_s * theta_s
    theta_s3 = theta_s2 * theta_s

    zenith_x = (0.00166f0 * theta_s3 - 0.00375f0 * theta_s2 + 0.00209f0 * theta_s) * T2 +
               (-0.02903f0 * theta_s3 + 0.06377f0 * theta_s2 - 0.03202f0 * theta_s + 0.00394f0) * T +
               (0.11693f0 * theta_s3 - 0.21196f0 * theta_s2 + 0.06052f0 * theta_s + 0.25886f0)

    zenith_y = (0.00275f0 * theta_s3 - 0.00610f0 * theta_s2 + 0.00317f0 * theta_s) * T2 +
               (-0.04214f0 * theta_s3 + 0.08970f0 * theta_s2 - 0.04153f0 * theta_s + 0.00516f0) * T +
               (0.15346f0 * theta_s3 - 0.26756f0 * theta_s2 + 0.06670f0 * theta_s + 0.26688f0)

    zenith_Y, zenith_x, zenith_y
end

"""
Perez sky luminance distribution function.
F(θ, γ) = (1 + A*exp(B/cos(θ))) * (1 + C*exp(D*γ) + E*cos²(γ))
Where θ is zenith angle of sky point, γ is angle between sky point and sun.
"""
@propagate_inbounds function perez(theta::Float32, gamma::Float32, coeffs::NTuple{5, Float32})
    A, B, C, D, E = coeffs
    cos_theta = max(0.001f0, cos(theta))
    (1f0 + A * exp(B / cos_theta)) * (1f0 + C * exp(D * gamma) + E * cos(gamma)^2)
end

"""
Compute sky radiance for a given view direction.
"""
@propagate_inbounds function sky_radiance(light::SunSkyLight, direction::Vec3f)
    # Below horizon - return ground albedo if enabled, otherwise continue sky
    if direction[3] <= 0f0 && light.ground_enabled
        return light.ground_albedo * 0.3f0
    end

    # Zenith angle of view direction
    theta = acos(clamp(direction[3], 0f0, 1f0))

    # Angle between view direction and sun
    cos_gamma = clamp(dot(direction, light.sun_direction), -1f0, 1f0)
    gamma = acos(cos_gamma)

    # Sun's zenith angle
    theta_s = acos(clamp(light.sun_direction[3], 0f0, 1f0))

    # Compute relative luminance using Perez function
    # Y(θ,γ) = Yz * F(θ,γ) / F(0, θs)
    perez_ratio_Y = perez(theta, gamma, light.perez_Y) /
                    perez(0f0, theta_s, light.perez_Y)
    Y = light.zenith_Y * perez_ratio_Y

    # Compute chromaticity
    perez_ratio_x = perez(theta, gamma, light.perez_x) /
                    perez(0f0, theta_s, light.perez_x)
    perez_ratio_y = perez(theta, gamma, light.perez_y) /
                    perez(0f0, theta_s, light.perez_y)
    x = light.zenith_x * perez_ratio_x
    y = light.zenith_y * perez_ratio_y

    # Convert xyY to XYZ
    Y_scaled = Y * 0.04f0  # Scale for reasonable brightness
    X = (x / y) * Y_scaled
    Z = ((1f0 - x - y) / y) * Y_scaled

    # Convert XYZ to RGB (sRGB primaries)
    r =  3.2406f0 * X - 1.5372f0 * Y_scaled - 0.4986f0 * Z
    g = -0.9689f0 * X + 1.8758f0 * Y_scaled + 0.0415f0 * Z
    b =  0.0557f0 * X - 0.2040f0 * Y_scaled + 1.0570f0 * Z

    RGBSpectrum(max(0f0, r), max(0f0, g), max(0f0, b))
end

"""
Compute sun disk radiance (with soft edge).
"""
@propagate_inbounds function sun_disk_radiance(light::SunSkyLight, direction::Vec3f)
    cos_angle = dot(direction, light.sun_direction)
    angle = acos(clamp(cos_angle, -1f0, 1f0))

    if angle < light.sun_angular_radius
        # Inside sun disk - full intensity
        return light.sun_intensity
    elseif angle < light.sun_angular_radius * 2f0
        # Corona/limb darkening region
        t = (angle - light.sun_angular_radius) / light.sun_angular_radius
        return light.sun_intensity * (1f0 - t) * 0.5f0
    else
        return RGBSpectrum(0f0)
    end
end

# ============================================================================
# UV <-> Direction conversion for hemisphere importance sampling
# ============================================================================

"""
Convert UV coordinates to direction on the upper hemisphere.
- u ∈ [0,1] -> φ ∈ [0, 2π] (azimuth angle)
- v ∈ [0,1] -> θ ∈ [0, π/2] (zenith angle, 0=up/+Z, π/2=horizon)
Returns direction with z >= 0.
"""
@propagate_inbounds function hemisphere_uv_to_direction(uv::Point2f)::Vec3f
    φ = uv[1] * 2f0 * Float32(π)
    θ = uv[2] * Float32(π) / 2f0
    sin_θ = sin(θ)
    Vec3f(sin_θ * cos(φ), sin_θ * sin(φ), cos(θ))
end

"""
Convert direction to UV coordinates for hemisphere.
Inverse of hemisphere_uv_to_direction.
"""
@propagate_inbounds function hemisphere_direction_to_uv(dir::Vec3f)::Point2f
    # θ is zenith angle (0 at +Z, π/2 at horizon)
    θ = acos(clamp(dir[3], 0f0, 1f0))
    # φ is azimuth angle
    φ = atan(dir[2], dir[1])
    if φ < 0f0
        φ += 2f0 * Float32(π)
    end
    # Convert to UV
    u = φ / (2f0 * Float32(π))
    v = θ / (Float32(π) / 2f0)
    Point2f(u, v)
end

# ============================================================================
# Light Interface Implementation
# ============================================================================

"""
Sample incident radiance at a point from the sky hemisphere.

Uses importance sampling based on pre-computed sky radiance distribution
for efficient sampling with low variance. The distribution is built at
construction time from the sky + sun radiance weighted by sin(θ) for
solid angle correction.

The visibility tester ensures proper shadow testing - if geometry blocks
the sampled sky direction, no contribution is added.
"""
@propagate_inbounds function sample_li(
    light::SunSkyLight{S, D}, ref::Interaction, u::Point2f, scene::AbstractScene,
)::Tuple{S,Vec3f,Float32,VisibilityTester} where {S<:Spectrum, D}
    # Importance sample direction from pre-computed distribution
    uv, map_pdf = sample_continuous(light.distribution, u)

    # Convert UV to direction on hemisphere
    wi = hemisphere_uv_to_direction(uv)

    # Convert PDF from image space to solid angle
    # For hemisphere equirectangular: pdf_solidangle = pdf_image / (π² sin(θ))
    # where π² = 2π (azimuth range) * π/2 (zenith range)
    θ = uv[2] * Float32(π) / 2f0
    sin_θ = sin(θ)
    pdf_val = sin_θ > 0f0 ? map_pdf / (Float32(π) * Float32(π) * sin_θ) : 0f0

    # Get actual sky radiance for this direction
    radiance = sky_radiance(light, wi) + sun_disk_radiance(light, wi)

    # Create visibility tester to sky (at infinity in sampled direction)
    outside_point = ref.p .+ wi .* (2f0 * world_radius(scene))
    tester = VisibilityTester(
        ref, Interaction(outside_point, ref.time, Vec3f(0f0), Normal3f(0f0)),
    )

    radiance, wi, pdf_val, tester
end

"""
PDF for sampling a particular direction from the SunSkyLight.
Returns the probability density for importance sampling this direction.
"""
@propagate_inbounds function pdf_li(light::SunSkyLight{S, D}, ::Interaction, wi::Vec3f)::Float32 where {S<:Spectrum, D}
    # Below horizon has zero probability
    if wi[3] <= 0f0
        return 0f0
    end

    # Convert direction to UV
    uv = hemisphere_direction_to_uv(wi)

    # Get PDF from 2D distribution
    map_pdf = pdf(light.distribution, uv)

    # Convert from image space to solid angle
    θ = uv[2] * Float32(π) / 2f0
    sin_θ = sin(θ)
    sin_θ > 0f0 ? map_pdf / (Float32(π) * Float32(π) * sin_θ) : 0f0
end

"""
Return sky radiance for rays that escape the scene.
This is what makes SunSkyLight provide the sky background.
"""
# Separate methods to avoid Union type allocation
@propagate_inbounds function le(light::SunSkyLight, ray::Ray)
    dir = normalize(Vec3f(ray.d))
    sky = sky_radiance(light, dir)
    sun = sun_disk_radiance(light, dir)
    sky + sun
end

@propagate_inbounds function le(light::SunSkyLight, ray::RayDifferentials)
    dir = normalize(Vec3f(ray.d))
    sky = sky_radiance(light, dir)
    sun = sun_disk_radiance(light, dir)
    sky + sun
end

"""
Approximate power - not accurate but needed for interface.
"""
@propagate_inbounds function power(light::SunSkyLight{S}, scene::AbstractScene)::S where S<:Spectrum
    light.sun_intensity * Float32(π) * world_radius(scene)^2
end

# ============================================================================
# Pre-bake sky to EnvironmentLight (pbrt-v4 approach)
# ============================================================================

"""
    sunsky_to_envlight(; direction, intensity=1f0, turbidity=2.5f0, ...) -> (EnvironmentLight, SunLight)

Pre-bake the Preetham sky model into an equal-area EnvironmentMap and create a separate
SunLight for the sun disk. This matches pbrt-v4's approach of using a pre-baked HDR sky
image as an environment light, combined with a delta directional light for the sun.

Separating sun and sky avoids aliasing issues (the sun disk is smaller than a pixel at
typical resolutions) and gives low-variance results:
- `EnvironmentLight` importance-samples the sky dome correctly
- `SunLight` samples the exact sun direction with PDF=1 (delta distribution)
- MIS naturally weights between the two

The EnvironmentLight uses D65 illuminant uplift with photometric normalization
(`scale = 1/D65_PHOTOMETRIC`), which is the standard spectral rendering pipeline.

# Arguments
- `direction::Vec3f`: Direction TO the sun (normalized internally)
- `intensity::Float32 = 1f0`: Sun brightness multiplier
- `turbidity::Float32 = 2.5f0`: Atmospheric turbidity (2=clear, 10=hazy)
- `ground_albedo::RGBSpectrum = RGBSpectrum(0.3f0)`: Ground color below horizon
- `ground_enabled::Bool = true`: Whether to show ground below horizon
- `resolution::Int = 512`: Resolution of equal-area square map (must be square for pbrt-v4 mapping)

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
    theta_s = acos(clamp(dir[3], -1f0, 1f0))

    # Compute Preetham sky model coefficients
    perez_Y, perez_x, perez_y = compute_perez_coefficients(turbidity)
    zenith_Y, zenith_x, zenith_y = compute_zenith_values(turbidity, theta_s)

    # Bake sky (WITHOUT sun disk) to equal-area octahedral map (square, matching pbrt-v4)
    sky_data = Matrix{RGBSpectrum}(undef, resolution, resolution)

    for v_idx in 1:resolution
        for u_idx in 1:resolution
            # UV at pixel center
            uv = Point2f((u_idx - 0.5f0) / resolution, (v_idx - 0.5f0) / resolution)

            # Convert to world direction via equal-area mapping (Z-up, matching scene)
            wi = equal_area_square_to_sphere(uv)

            # Sky radiance from Preetham model (uses Y * 0.04 scaling, no sun disk)
            sky_rgb = _compute_sky_radiance(
                wi, dir, perez_Y, perez_x, perez_y,
                zenith_Y, zenith_x, zenith_y, ground_albedo, ground_enabled,
            )

            sky_data[v_idx, u_idx] = sky_rgb
        end
    end

    # Build EnvironmentMap with luminance-weighted importance sampling distribution
    env_map = EnvironmentMap(sky_data)

    # Photometric normalization: scale = 1/D65_PHOTOMETRIC ensures that the D65
    # illuminant uplift in the spectral path roundtrips correctly:
    # pixel_rgb → (pixel_rgb / D65_PHOTOMETRIC) → uplift_rgb_illuminant → integrate → ≈ pixel_rgb
    env_light = EnvironmentLight(env_map, RGBSpectrum(1f0 / D65_PHOTOMETRIC))

    # Sun as separate delta directional light.
    # SunLight(RGB{Float32}, direction) creates an RGBIlluminantSpectrum with
    # automatic photometric normalization (scale = 1/D65_PHOTOMETRIC).
    # SunLight direction = direction light TRAVELS (away from sun), so negate.
    #
    # Scale sun by 10x to get realistic sun:sky illumination ratio (~10:1).
    # Sky hemisphere integral gives ~0.66 irradiance, sun at 10x gives ~8 irradiance
    # on a surface perpendicular to sun direction, yielding ~12:1 ratio.
    sun_scale = 10f0 * intensity
    sun_rgb = RGB{Float32}(sun_scale, sun_scale * 0.95f0, sun_scale * 0.85f0)
    sun_light = SunLight(sun_rgb, -dir)

    return env_light, sun_light
end

# ============================================================================
# Helper function to create separated sun + sky lights for low-variance rendering
# ============================================================================

"""
    create_sun_and_sky_lights(sun_direction; kwargs...) -> (EnvironmentLight, SunLight)

Create separated sun and sky lights for lower-variance volumetric rendering.

When rendering volumetric media like clouds, using a combined `SunSkyLight` can cause
high variance because the tiny bright sun disk (~0.5° radius) gets blended with the
dim sky in the importance sampling distribution. This leads to PDF/radiance mismatch:
- Sometimes we sample sky but hit sun disk → radiance >> PDF → bright fireflies
- Sometimes we sample sun but miss → PDF biased high → dark spots

By separating into two lights:
- `SunLight` samples exactly the sun direction with PDF=1 (delta distribution)
- `EnvironmentLight` importance-samples the sky (without sun disk) properly

# Arguments
- `sun_direction::Vec3f`: Direction TO the sun (will be normalized)

# Keyword Arguments
- `sun_intensity::RGBSpectrum = RGBSpectrum(100f0)`: Radiance of the sun
- `turbidity::Float32 = 2.5f0`: Atmospheric turbidity (2=clear, 10=hazy)
- `ground_albedo::RGBSpectrum = RGBSpectrum(0.3f0)`: Ground color below horizon
- `ground_enabled::Bool = true`: Whether to show ground below horizon
- `sky_resolution::Int = 256`: Resolution of sky environment map

# Returns
A tuple of (EnvironmentLight, SunLight) to be added to the scene.

# Example
```julia
env_light, sun_light = create_sun_and_sky_lights(
    Vec3f(0.5f0, 0.5f0, 0.8f0);
    sun_intensity = RGBSpectrum(80f0),
    turbidity = 3.0f0,
)
scene = Scene([...], [env_light, sun_light])
```
"""
function create_sun_and_sky_lights(
    sun_direction::Vec3f;
    sun_intensity::RGBSpectrum = RGBSpectrum(100f0),
    turbidity::Float32 = 2.5f0,
    ground_albedo::RGBSpectrum = RGBSpectrum(0.3f0),
    ground_enabled::Bool = true,
    sky_resolution::Int = 256,
)
    dir = normalize(sun_direction)

    # Sun elevation angle (theta_s is angle from zenith, z-up)
    theta_s = acos(clamp(dir[3], -1f0, 1f0))

    # Compute Preetham sky model coefficients
    perez_Y, perez_x, perez_y = compute_perez_coefficients(turbidity)
    zenith_Y, zenith_x, zenith_y = compute_zenith_values(turbidity, theta_s)

    # Render sky to equirectangular environment map (WITHOUT sun disk)
    # Full sphere: u ∈ [0,1] -> φ ∈ [-π, π], v ∈ [0,1] -> θ ∈ [0, π]
    h = sky_resolution
    w = sky_resolution * 2
    sky_data = Matrix{RGBSpectrum}(undef, h, w)

    for v_idx in 1:h
        # θ goes from 0 (top/+Y in env map convention) to π (bottom/-Y)
        θ = Float32(π) * (v_idx - 0.5f0) / h

        for u_idx in 1:w
            # φ goes from -π to π
            φ = 2f0 * Float32(π) * (u_idx - 0.5f0) / w - Float32(π)

            # Convert to direction (Y-up for environment map)
            sin_θ = sin(θ)
            direction_yup = Vec3f(sin_θ * cos(φ), cos(θ), sin_θ * sin(φ))

            # Convert to Z-up for sky_radiance computation
            # Y-up: (x, y, z) -> Z-up: (x, z, y)
            direction_zup = Vec3f(direction_yup[1], direction_yup[3], direction_yup[2])

            # Compute sky radiance (no sun disk!) using internal helper
            sky_rad = _compute_sky_radiance(
                direction_zup, dir, perez_Y, perez_x, perez_y,
                zenith_Y, zenith_x, zenith_y, ground_albedo, ground_enabled,
            )

            sky_data[v_idx, u_idx] = sky_rad
        end
    end

    # Create EnvironmentMap and EnvironmentLight
    env_map = EnvironmentMap(sky_data, 0f0)
    env_light = EnvironmentLight(env_map)

    # Create SunLight
    # SunLight expects light_to_world transformation and direction in light space
    # For simplicity, we use identity transform and pass world direction directly
    # Note: SunLight direction is the direction light TRAVELS (away from sun)
    # So we negate sun_direction (which points TO sun) to get travel direction
    sun_light = SunLight(sun_intensity, -dir)

    return env_light, sun_light
end
