"""
Environment map texture for HDR image-based lighting.
Supports sampling by direction vector (for environment lights).
Includes importance sampling distribution based on luminance.
"""
struct EnvironmentMap{S<:Spectrum, T<:AbstractMatrix{S}, D}
    """HDR image data (lat-long / equirectangular format)."""
    data::T

    """Rotation matrix (render space to light/map space). Apply inverse to transform directions."""
    rotation::Mat3f

    """2D distribution for importance sampling based on luminance."""
    distribution::D

    """Inner constructor for pre-computed environment maps (used by to_gpu)."""
    function EnvironmentMap(data::T, rotation::Mat3f, distribution::D) where {S<:Spectrum, T<:AbstractMatrix{S}, D<:Union{Distribution2D, FlatDistribution2D}}
        new{S, T, D}(data, rotation, distribution)
    end

    function EnvironmentMap(data::AbstractMatrix{S}, rotation::Mat3f=Mat3f(I)) where {S<:Spectrum}
        # Build luminance-weighted distribution for importance sampling
        # Weight by sin(θ) to account for solid angle distortion in equirectangular maps
        h, w = size(data)
        luminance = Matrix{Float32}(undef, h, w)
        for v in 1:h
            # θ goes from 0 (top, +Y) to π (bottom, -Y)
            θ = π * (v - 0.5f0) / h
            sin_θ = sin(θ)
            for u in 1:w
                luminance[v, u] = to_Y(data[v, u]) * sin_θ
            end
        end
        distribution = Distribution2D(luminance)
        new{S, typeof(data), typeof(distribution)}(data, rotation, distribution)
    end
end

"""
Create rotation matrix from axis-angle representation (like pbrt's Rotate command).
angle: rotation angle in degrees
axis: rotation axis (will be normalized)
"""
function rotation_matrix(angle_degrees::Real, axis::Vec3f)::Mat3f
    θ = Float32(deg2rad(angle_degrees))
    a = normalize(axis)
    s, c = sincos(θ)
    t = 1f0 - c

    # Rodrigues' rotation formula as matrix
    Mat3f(
        t*a[1]*a[1] + c,      t*a[1]*a[2] - s*a[3], t*a[1]*a[3] + s*a[2],
        t*a[1]*a[2] + s*a[3], t*a[2]*a[2] + c,      t*a[2]*a[3] - s*a[1],
        t*a[1]*a[3] - s*a[2], t*a[2]*a[3] + s*a[1], t*a[3]*a[3] + c
    )
end

"""
Convert a direction vector to equirectangular UV coordinates.
Uses standard lat-long mapping where:
- U (horizontal) maps to longitude: 0 at +X, increases counter-clockwise
- V (vertical) maps to latitude: 0 at top (+Y), 1 at bottom (-Y)

The rotation matrix transforms from render space to light/map space.
We apply the INVERSE (transpose for orthogonal matrices) to get the direction in map space.
"""
@propagate_inbounds function direction_to_uv(dir::Vec3f, rotation::Mat3f)::Point2f
    # Transform direction from render space to light space (inverse = transpose)
    dir_light = transpose(rotation) * dir

    # Compute spherical coordinates
    # θ (theta) is the polar angle from +Y axis
    # φ (phi) is the azimuthal angle in XZ plane from +X axis
    θ = acos(clamp(dir_light[2], -1f0, 1f0))  # Y is up
    φ = atan(dir_light[3], dir_light[1])  # atan2(z, x)

    # Convert to UV coordinates [0,1]
    # U: longitude, φ ∈ [-π, π] -> [0, 1]
    u = (φ + Float32(π)) / (2f0 * Float32(π))
    # V: latitude, θ ∈ [0, π] -> [0, 1]
    v = θ / Float32(π)

    # Wrap U to [0,1]
    u = mod(u, 1f0)

    Point2f(u, v)
end

"""
Convert equirectangular UV coordinates to a direction vector.
Inverse of direction_to_uv.

The rotation matrix transforms from render space to light/map space.
We apply the rotation (not inverse) to transform from light space back to render space.
"""
@propagate_inbounds function uv_to_direction(uv::Point2f, rotation::Mat3f)::Vec3f
    # Convert UV to spherical coordinates
    # U: [0, 1] -> φ ∈ [-π, π]
    φ = uv[1] * 2f0 * Float32(π) - Float32(π)
    # V: [0, 1] -> θ ∈ [0, π]
    θ = uv[2] * Float32(π)

    # Convert to Cartesian (Y-up) in light space
    sin_θ = sin(θ)
    dir_light = Vec3f(sin_θ * cos(φ), cos(θ), sin_θ * sin(φ))

    # Transform from light space to render space
    rotation * dir_light
end

"""
Sample the environment map by direction vector.
"""
@propagate_inbounds function (env::EnvironmentMap{S, T, D})(dir::Vec3f)::S where {S<:Spectrum, T, D}
    uv = direction_to_uv(dir, env.rotation)

    # Bilinear interpolation
    h, w = size(env.data)

    # Convert to pixel coordinates
    x = uv[1] * (w - 1) + 1
    y = uv[2] * (h - 1) + 1

    # Get integer pixel coordinates
    x0 = floor_int32(x)
    y0 = floor_int32(y)
    x1 = x0 + Int32(1)
    y1 = y0 + Int32(1)

    # Clamp to valid range
    w32 = u_int32(w)
    h32 = u_int32(h)
    x0 = clamp(x0, Int32(1), w32)
    x1 = clamp(x1, Int32(1), w32)
    y0 = clamp(y0, Int32(1), h32)
    y1 = clamp(y1, Int32(1), h32)

    # Wrap x coordinates for seamless horizontal tiling
    x1 = x1 > w32 ? Int32(1) : x1

    # Interpolation weights
    fx = x - floor(x)
    fy = y - floor(y)

    # Bilinear interpolation
    c00 = env.data[y0, x0]
    c10 = env.data[y0, x1]
    c01 = env.data[y1, x0]
    c11 = env.data[y1, x1]

    c0 = c00 * (1f0 - fx) + c10 * fx
    c1 = c01 * (1f0 - fx) + c11 * fx

    c0 * (1f0 - fy) + c1 * fy
end

"""
Sample method alias for compatibility.
"""
@propagate_inbounds sample(env::EnvironmentMap, dir::Vec3f) = env(dir)

"""
    lookup_uv(env::EnvironmentMap, uv::Point2f) -> Spectrum

Look up environment map directly by UV coordinates.
This is the equivalent of pbrt-v4's ImageLe(uv, lambda) for ImageInfiniteLight.
Used when UV is already known (e.g., from importance sampling the distribution).

IMPORTANT: Uses nearest-neighbor lookup to match the discrete PDF from importance sampling.
Bilinear interpolation would cause bias because the PDF is computed for discrete pixels,
not interpolated values. This matches pbrt-v4's LookupNearestChannel in ImageLe.
"""
@propagate_inbounds function lookup_uv(env::EnvironmentMap{S, T, D}, uv::Point2f)::S where {S<:Spectrum, T, D}
    # Nearest-neighbor lookup to match discrete PDF (like pbrt-v4's LookupNearestChannel)
    h, w = size(env.data)

    # Convert UV to pixel indices (nearest neighbor)
    # UV [0,1] maps to pixel centers at (0.5/w, 1.5/w, ..., (w-0.5)/w)
    u_idx = clamp(floor_int32(uv[1] * w) + Int32(1), Int32(1), u_int32(w))
    v_idx = clamp(floor_int32(uv[2] * h) + Int32(1), Int32(1), u_int32(h))

    @inbounds env.data[v_idx, u_idx]
end

"""
Load an environment map from an HDR/EXR file.
Converts the image to RGBSpectrum format.

rotation: Mat3f rotation matrix, or nothing for identity
"""
function load_environment_map(path::String; rotation::Mat3f=Mat3f(I))
    img = FileIO.load(path)
    # Convert to RGBSpectrum matrix
    data = map(c -> RGBSpectrum(Float32(c.r), Float32(c.g), Float32(c.b)), img)
    EnvironmentMap(data, rotation)
end
