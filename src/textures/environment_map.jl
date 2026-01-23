"""
Environment map texture for HDR image-based lighting.
Supports sampling by direction vector (for environment lights).
Includes importance sampling distribution based on luminance.

Uses equal-area (octahedral) mapping like pbrt-v4's ImageInfiniteLight.
Expects SQUARE images in equal-area format.
"""
struct EnvironmentMap{S<:Spectrum, T<:AbstractMatrix{S}, D}
    """HDR image data in equal-area (octahedral) format. Must be square."""
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
        h, w = size(data)

        # pbrt-v4 expects square images for equal-area mapping
        if h != w
            @warn "Environment map is not square ($w x $h). pbrt-v4 uses equal-area mapping which expects square images."
        end

        # Build luminance-weighted distribution for importance sampling
        # For equal-area mapping, NO sin(θ) weighting is needed because
        # equal-area projection already preserves solid angle uniformity.
        # This matches pbrt-v4's Image::GetSamplingDistribution() for equal-area images.
        luminance = Matrix{Float32}(undef, h, w)
        for v in 1:h
            for u in 1:w
                luminance[v, u] = to_Y(data[v, u])
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

# =============================================================================
# Equal-Area Sphere Mapping (pbrt-v4 octahedral mapping)
# =============================================================================

"""
    equal_area_sphere_to_square(d::Vec3f) -> Point2f

Convert a direction vector to equal-area square UV coordinates.
This is pbrt-v4's octahedral/equal-area mapping used for environment maps.

Reference: Clarberg, "Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD"
"""
@propagate_inbounds function equal_area_sphere_to_square(d::Vec3f)::Point2f
    x, y, z = abs(d[1]), abs(d[2]), abs(d[3])

    # Compute the radius r = sqrt(1 - |z|)
    r = sqrt(1f0 - z)

    # Compute the argument to atan (detect a=0 to avoid div-by-zero)
    a = max(x, y)
    b = a == 0f0 ? 0f0 : min(x, y) / a

    # Polynomial approximation of atan(x)*2/pi, x=b
    # Coefficients for 6th degree minimax approximation of atan(x)*2/pi, x=[0,1]
    t1 = 0.406758566246788489601959989f-5
    t2 = 0.636226545274016134946890922156f0
    t3 = 0.61572017898280213493197203466f-2
    t4 = -0.247333733281268944196501420480f0
    t5 = 0.881770664775316294736387951347f-1
    t6 = 0.419038818029165735901852432784f-1
    t7 = -0.251390972343483509333252996350f-1

    # Evaluate polynomial: t1 + b*(t2 + b*(t3 + b*(t4 + b*(t5 + b*(t6 + b*t7)))))
    phi = t1 + b * (t2 + b * (t3 + b * (t4 + b * (t5 + b * (t6 + b * t7)))))

    # Extend phi if the input is in the range 45-90 degrees (x < y)
    if x < y
        phi = 1f0 - phi
    end

    # Find (u,v) based on (r,phi)
    v = phi * r
    u = r - v

    # Southern hemisphere -> mirror u,v
    if d[3] < 0f0
        u, v = v, u
        u = 1f0 - u
        v = 1f0 - v
    end

    # Move (u,v) to the correct quadrant based on the signs of (x,y)
    u = copysign(u, d[1])
    v = copysign(v, d[2])

    # Transform (u,v) from [-1,1] to [0,1]
    Point2f(0.5f0 * (u + 1f0), 0.5f0 * (v + 1f0))
end

"""
    equal_area_square_to_sphere(p::Point2f) -> Vec3f

Convert equal-area square UV coordinates to a direction vector.
Inverse of equal_area_sphere_to_square.

Reference: pbrt-v4's EqualAreaSquareToSphere
"""
@propagate_inbounds function equal_area_square_to_sphere(p::Point2f)::Vec3f
    # Transform p from [0,1]² to [-1,1]² and compute absolute values
    u = 2f0 * p[1] - 1f0
    v = 2f0 * p[2] - 1f0
    up = abs(u)
    vp = abs(v)

    # Compute radius r as signed distance from diagonal
    signed_distance = 1f0 - (up + vp)
    d = abs(signed_distance)
    r = 1f0 - d

    # Compute angle φ for square to sphere mapping
    phi = (r == 0f0 ? 1f0 : (vp - up) / r + 1f0) * Float32(π) / 4f0

    # Find z coordinate for spherical direction
    z = copysign(1f0 - r * r, signed_distance)

    # Compute cos(φ) and sin(φ) for original quadrant and return vector
    cos_phi = copysign(cos(phi), u)
    sin_phi = copysign(sin(phi), v)

    # Compute x, y from cylindrical coordinates
    # r_cyl = r * sqrt(2 - r²)
    r_cyl = r * sqrt(2f0 - r * r)

    Vec3f(cos_phi * r_cyl, sin_phi * r_cyl, z)
end

"""
    wrap_equal_area_square(uv::Point2f) -> Point2f

Wrap UV coordinates for equal-area sphere mapping (octahedral wrapping).
Handles coordinates outside [0,1]² by mirroring appropriately.
"""
@propagate_inbounds function wrap_equal_area_square(uv::Point2f)::Point2f
    u, v = uv[1], uv[2]

    if u < 0f0
        u = -u           # mirror across u = 0
        v = 1f0 - v      # mirror across v = 0.5
    elseif u > 1f0
        u = 2f0 - u      # mirror across u = 1
        v = 1f0 - v      # mirror across v = 0.5
    end

    if v < 0f0
        u = 1f0 - u      # mirror across u = 0.5
        v = -v           # mirror across v = 0
    elseif v > 1f0
        u = 1f0 - u      # mirror across u = 0.5
        v = 2f0 - v      # mirror across v = 1
    end

    Point2f(u, v)
end

# =============================================================================
# Direction <-> UV conversion (with rotation support)
# =============================================================================

"""
    direction_to_uv_equal_area(dir::Vec3f, rotation::Mat3f) -> Point2f

Convert direction to UV using equal-area (octahedral) mapping.
This is what pbrt-v4 uses for ImageInfiniteLight.

The rotation matrix transforms from render space to light/map space.
"""
@propagate_inbounds function direction_to_uv_equal_area(dir::Vec3f, rotation::Mat3f)::Point2f
    # Transform direction from render space to light space (inverse = transpose)
    dir_light = transpose(rotation) * dir
    equal_area_sphere_to_square(dir_light)
end

"""
    uv_to_direction_equal_area(uv::Point2f, rotation::Mat3f) -> Vec3f

Convert UV to direction using equal-area (octahedral) mapping.
Inverse of direction_to_uv_equal_area.
"""
@propagate_inbounds function uv_to_direction_equal_area(uv::Point2f, rotation::Mat3f)::Vec3f
    dir_light = equal_area_square_to_sphere(uv)
    # Transform from light space to render space
    rotation * dir_light
end

# Default direction_to_uv and uv_to_direction now use equal-area (pbrt-v4 compatible)
@propagate_inbounds function direction_to_uv(dir::Vec3f, rotation::Mat3f)::Point2f
    direction_to_uv_equal_area(dir, rotation)
end

@propagate_inbounds function uv_to_direction(uv::Point2f, rotation::Mat3f)::Vec3f
    uv_to_direction_equal_area(uv, rotation)
end

# =============================================================================
# Equirectangular Mapping (lat-long, legacy - for non-pbrt-v4 images)
# =============================================================================

"""
Convert a direction vector to equirectangular UV coordinates.
Uses standard lat-long mapping where:
- U (horizontal) maps to longitude: 0 at +X, increases counter-clockwise
- V (vertical) maps to latitude: 0 at top (+Y), 1 at bottom (-Y)

The rotation matrix transforms from render space to light/map space.
We apply the INVERSE (transpose for orthogonal matrices) to get the direction in map space.
"""
@propagate_inbounds function direction_to_uv_equirect(dir::Vec3f, rotation::Mat3f)::Point2f
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
Inverse of direction_to_uv_equirect.

The rotation matrix transforms from render space to light/map space.
We apply the rotation (not inverse) to transform from light space back to render space.
"""
@propagate_inbounds function uv_to_direction_equirect(uv::Point2f, rotation::Mat3f)::Vec3f
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

    # Interpolation weights (use floor_int32 for GPU compatibility)
    fx = x - Float32(floor_int32(x))
    fy = y - Float32(floor_int32(y))

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
