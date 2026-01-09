"""
Environment map texture for HDR image-based lighting.
Supports sampling by direction vector (for environment lights).
Includes importance sampling distribution based on luminance.
"""
struct EnvironmentMap{S<:Spectrum, T<:AbstractMatrix{S}, D}
    """HDR image data (lat-long / equirectangular format)."""
    data::T

    """Rotation around Y axis (radians)."""
    rotation::Float32

    """2D distribution for importance sampling based on luminance."""
    distribution::D

    """Inner constructor for pre-computed environment maps (used by to_gpu)."""
    function EnvironmentMap(data::T, rotation::Float32, distribution::D) where {S<:Spectrum, T<:AbstractMatrix{S}, D<:Union{Distribution2D, FlatDistribution2D}}
        new{S, T, D}(data, rotation, distribution)
    end

    function EnvironmentMap(data::AbstractMatrix{S}, rotation::Float32=0f0) where {S<:Spectrum}
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
Convert a direction vector to equirectangular UV coordinates.
Uses standard lat-long mapping where:
- U (horizontal) maps to longitude: 0 at +X, increases counter-clockwise
- V (vertical) maps to latitude: 0 at top (+Y), 1 at bottom (-Y)
"""
@inline function direction_to_uv(dir::Vec3f, rotation::Float32=0f0)::Point2f
    # Compute spherical coordinates
    # θ (theta) is the polar angle from +Y axis
    # φ (phi) is the azimuthal angle in XZ plane from +X axis
    θ = acos(clamp(dir[2], -1f0, 1f0))  # Y is up
    φ = atan(dir[3], dir[1])  # atan2(z, x)

    # Apply rotation
    φ = φ + rotation

    # Convert to UV coordinates [0,1]
    # U: longitude, φ ∈ [-π, π] -> [0, 1]
    u = (φ + π) / (2f0 * π)
    # V: latitude, θ ∈ [0, π] -> [0, 1]
    v = θ / π

    # Wrap U to [0,1]
    u = mod(u, 1f0)

    Point2f(u, v)
end

"""
Convert equirectangular UV coordinates to a direction vector.
Inverse of direction_to_uv.
"""
@inline function uv_to_direction(uv::Point2f, rotation::Float32=0f0)::Vec3f
    # Convert UV to spherical coordinates
    # U: [0, 1] -> φ ∈ [-π, π]
    φ = uv[1] * 2f0 * π - π
    # V: [0, 1] -> θ ∈ [0, π]
    θ = uv[2] * π

    # Remove rotation
    φ = φ - rotation

    # Convert to Cartesian (Y-up)
    sin_θ = sin(θ)
    Vec3f(sin_θ * cos(φ), cos(θ), sin_θ * sin(φ))
end

"""
Sample the environment map by direction vector.
"""
@inline function (env::EnvironmentMap{S, T, D})(dir::Vec3f)::S where {S<:Spectrum, T, D}
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
@inline sample(env::EnvironmentMap, dir::Vec3f) = env(dir)

"""
    lookup_uv(env::EnvironmentMap, uv::Point2f) -> Spectrum

Look up environment map directly by UV coordinates.
This is the equivalent of pbrt-v4's ImageLe(uv, lambda) for ImageInfiniteLight.
Used when UV is already known (e.g., from importance sampling the distribution).
"""
@inline function lookup_uv(env::EnvironmentMap{S, T, D}, uv::Point2f)::S where {S<:Spectrum, T, D}
    # Bilinear interpolation (same as direction-based lookup, but UV already computed)
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
Load an environment map from an HDR/EXR file.
Converts the image to RGBSpectrum format.
"""
function load_environment_map(path::String; rotation::Float32=0f0)
    img = FileIO.load(path)
    # Convert to RGBSpectrum matrix
    data = map(c -> RGBSpectrum(Float32(c.r), Float32(c.g), Float32(c.b)), img)
    EnvironmentMap(data, rotation)
end
