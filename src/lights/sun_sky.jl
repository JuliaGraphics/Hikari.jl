"""
SunSkyLight - Combined sun and procedural sky light using atmospheric scattering.

Provides both:
- Directional sun illumination via sample_li() for surface/volume lighting
- Procedural sky background via le() for rays that escape the scene

The sky color is computed from the sun position using a simplified atmospheric
scattering model (Rayleigh + Mie).
"""
struct SunSkyLight{S<:Spectrum} <: Light
    flags::LightFlags

    # Sun parameters
    sun_direction::Vec3f      # Direction TO the sun (normalized)
    sun_intensity::S          # Sun radiance
    sun_angular_radius::Float32  # Angular radius in radians (~0.00465 for real sun)

    # Atmosphere parameters
    turbidity::Float32        # Atmospheric turbidity (2.0 = clear, 10.0 = hazy)
    ground_albedo::S          # Ground color for hemisphere below horizon

    # Precomputed sky coefficients (Preetham model)
    # Perez function coefficients for luminance Y and chromaticity x, y
    perez_Y::NTuple{5, Float32}
    perez_x::NTuple{5, Float32}
    perez_y::NTuple{5, Float32}

    # Zenith values
    zenith_Y::Float32
    zenith_x::Float32
    zenith_y::Float32

    function SunSkyLight(
        sun_direction::Vec3f,
        sun_intensity::S;
        turbidity::Float32 = 2.5f0,
        ground_albedo::S = RGBSpectrum(0.3f0),
        sun_angular_radius::Float32 = 0.00465f0,  # Real sun ~0.53 degrees
    ) where S<:Spectrum
        dir = normalize(sun_direction)

        # Sun elevation angle (theta_s is angle from zenith)
        theta_s = acos(clamp(dir[3], -1f0, 1f0))

        # Compute Preetham sky model coefficients
        perez_Y, perez_x, perez_y = compute_perez_coefficients(turbidity)
        zenith_Y, zenith_x, zenith_y = compute_zenith_values(turbidity, theta_s)

        new{S}(
            LightInfinite,  # Infinite light (provides le() for background)
            dir, sun_intensity, sun_angular_radius,
            turbidity, ground_albedo,
            perez_Y, perez_x, perez_y,
            zenith_Y, zenith_x, zenith_y,
        )
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
@inline function perez(theta::Float32, gamma::Float32, coeffs::NTuple{5, Float32})
    A, B, C, D, E = coeffs
    cos_theta = max(0.001f0, cos(theta))
    (1f0 + A * exp(B / cos_theta)) * (1f0 + C * exp(D * gamma) + E * cos(gamma)^2)
end

"""
Compute sky radiance for a given view direction.
"""
function sky_radiance(light::SunSkyLight, direction::Vec3f)
    # Below horizon - return ground albedo
    if direction[3] <= 0f0
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
function sun_disk_radiance(light::SunSkyLight, direction::Vec3f)
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
# Light Interface Implementation
# ============================================================================

"""
Sample incident radiance at a point - returns directional sun illumination.
"""
function sample_li(
    light::SunSkyLight{S}, ref::Interaction, u::Point2f, scene::AbstractScene,
)::Tuple{S,Vec3f,Float32,VisibilityTester} where S<:Spectrum
    # Direction TO the sun
    wi = light.sun_direction

    # Create visibility tester to sun (at infinity)
    outside_point = ref.p .+ wi .* (2f0 * scene.world_radius)
    tester = VisibilityTester(
        ref, Interaction(outside_point, ref.time, Vec3f(0f0), Normal3f(0f0)),
    )

    light.sun_intensity, wi, 1f0, tester
end

"""
Return sky radiance for rays that escape the scene.
This is what makes SunSkyLight provide the sky background.
"""
# Separate methods to avoid Union type allocation
function le(light::SunSkyLight, ray::Ray)
    dir = normalize(Vec3f(ray.d))
    sky = sky_radiance(light, dir)
    sun = sun_disk_radiance(light, dir)
    sky + sun
end

function le(light::SunSkyLight, ray::RayDifferentials)
    dir = normalize(Vec3f(ray.d))
    sky = sky_radiance(light, dir)
    sun = sun_disk_radiance(light, dir)
    sky + sun
end

"""
Approximate power - not accurate but needed for interface.
"""
@inline function power(light::SunSkyLight{S}, scene::AbstractScene)::S where S<:Spectrum
    light.sun_intensity * Float32(π) * scene.world_radius^2
end
