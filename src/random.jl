# Procedural noise functions for volumetric effects
# Perlin noise and fractional Brownian motion (fBm)

# Smoothstep interpolation
_lerp(t, a, b) = a + t * (b - a)
_fade(t) = t * t * t * (t * (t * 6 - 15) + 10)

function _grad3d(hash, x, y, z)
    h = hash & 15
    u = h < 8 ? x : y
    v = h < 4 ? y : (h == 12 || h == 14 ? x : z)
    ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v)
end

# Permutation table for Perlin noise
const _PERM = UInt8[
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
    8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
    35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
    134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
    55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,
    18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
    250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
    189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
    172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
    228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
    107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
]
_perm(i) = @inbounds _PERM[(i & 255) + 1]

"""
    perlin3d(x, y, z) -> Float64

Classic 3D Perlin noise. Returns values in approximately [-1, 1].
"""
function perlin3d(x, y, z)
    X, Y, Z = floor(Int, x) & 255, floor(Int, y) & 255, floor(Int, z) & 255
    x, y, z = x - floor(x), y - floor(y), z - floor(z)
    u, v, w = _fade(x), _fade(y), _fade(z)
    A, B = _perm(X) + Y, _perm(X + 1) + Y
    AA, AB, BA, BB = _perm(A) + Z, _perm(A + 1) + Z, _perm(B) + Z, _perm(B + 1) + Z
    _lerp(w,
        _lerp(v, _lerp(u, _grad3d(_perm(AA), x, y, z), _grad3d(_perm(BA), x-1, y, z)),
                 _lerp(u, _grad3d(_perm(AB), x, y-1, z), _grad3d(_perm(BB), x-1, y-1, z))),
        _lerp(v, _lerp(u, _grad3d(_perm(AA+1), x, y, z-1), _grad3d(_perm(BA+1), x-1, y, z-1)),
                 _lerp(u, _grad3d(_perm(AB+1), x, y-1, z-1), _grad3d(_perm(BB+1), x-1, y-1, z-1))))
end

"""
    fbm3d(x, y, z; octaves=4, persistence=0.5) -> Float64

Fractional Brownian motion (fBm) using Perlin noise.
Combines multiple octaves of noise at different frequencies for natural-looking detail.

# Arguments
- `x, y, z`: 3D coordinates
- `octaves`: Number of noise layers to combine (default: 4)
- `persistence`: Amplitude multiplier per octave (default: 0.5)

Returns values approximately in [-1, 1].
"""
function fbm3d(x, y, z; octaves=4, persistence=0.5)
    total, frequency, amplitude, max_value = 0.0, 1.0, 1.0, 0.0
    for _ in 1:octaves
        total += perlin3d(x * frequency, y * frequency, z * frequency) * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= 2.0
    end
    total / max_value
end

# ============================================================================
# Worley (Cellular) Noise - Essential for puffy cloud structures
# ============================================================================

"""
    worley3d(x, y, z; seed=0) -> Float64

3D Worley (cellular) noise. Returns distance to nearest feature point in [0, ~1.5].
The characteristic "cell" structure creates puffy, billowy patterns ideal for clouds.
"""
function worley3d(x, y, z; seed=0)
    # Cell coordinates
    xi, yi, zi = floor(Int, x), floor(Int, y), floor(Int, z)
    fx, fy, fz = x - xi, y - yi, z - zi

    min_dist = 10.0

    # Check 3x3x3 neighborhood
    for dz in -1:1, dy in -1:1, dx in -1:1
        # Hash to get feature point position within cell
        cx, cy, cz = xi + dx, yi + dy, zi + dz
        h = _perm(_perm(_perm((cx + seed) & 255) + (cy & 255)) + (cz & 255))

        # Feature point position (0-1 within cell)
        px = dx + (h & 63) / 64.0
        py = dy + ((h >> 2) & 63) / 64.0
        pz = dz + ((h >> 4) & 63) / 64.0

        # Distance to feature point
        ddx, ddy, ddz = fx - px, fy - py, fz - pz
        dist = sqrt(ddx*ddx + ddy*ddy + ddz*ddz)
        min_dist = min(min_dist, dist)
    end

    min_dist
end

"""
    worley_fbm3d(x, y, z; octaves=3, persistence=0.5, lacunarity=2.0) -> Float64

Multi-octave Worley noise for more detailed cellular patterns.
"""
function worley_fbm3d(x, y, z; octaves=3, persistence=0.5, lacunarity=2.0)
    total, frequency, amplitude, max_value = 0.0, 1.0, 1.0, 0.0
    for i in 1:octaves
        total += worley3d(x * frequency, y * frequency, z * frequency; seed=i*17) * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    end
    total / max_value
end

# ============================================================================
# Cloud Density Generation
# ============================================================================

"""
    generate_cloud_density(resolution; kwargs...) -> Array{Float32, 3}

Generate a 3D density grid for volumetric cloud rendering.

# Keyword Arguments
- `scale=4.0`: Base frequency scale for noise
- `sphere_falloff=true`: Apply spherical boundary mask
- `threshold=0.3`: Density threshold (negative values include more volume)
- `worley_weight=0.6`: Weight of Worley noise (0-1, higher = puffier clouds)
- `edge_sharpness=1.5`: Controls edge falloff (lower = softer, puffier edges)
- `density_scale=3.0`: Scale factor for final density (match real cloud data ~2-3 max)

# Cloud Appearance Tips
- For puffy cumulus-like clouds: scale=2.5, worley_weight=0.6, threshold=0.15, density_scale=3.5
- For wispy cirrus-like clouds: scale=5.0, worley_weight=0.3, threshold=0.3, density_scale=2.0
- For dense fog-like volumes: scale=3.0, worley_weight=0.2, threshold=0.0, density_scale=4.0
"""
function generate_cloud_density(resolution::Int;
    scale=4.0,
    sphere_falloff=true,
    threshold=0.3,
    worley_weight=0.6,
    edge_sharpness=1.5,
    density_scale=3.0
)
    density = Array{Float32, 3}(undef, resolution, resolution, resolution)
    center, radius = 0.5f0, 0.45f0

    for iz in 1:resolution, iy in 1:resolution, ix in 1:resolution
        x = (ix - 0.5f0) / resolution
        y = (iy - 0.5f0) / resolution
        z = (iz - 0.5f0) / resolution

        dx, dy, dz = x - center, y - center, z - center
        dist = sqrt(dx*dx + dy*dy + dz*dz)

        # === Cloud noise recipe ===
        # 1. Worley noise for puffy cell structure (inverted: cells are dense)
        worley = 1.0 - worley_fbm3d(x * scale * 0.8, y * scale * 0.8, z * scale * 0.8; octaves=3)

        # 2. Billowed Perlin for cloud-like ridges (1 - abs(noise))
        billow = 1.0 - abs(fbm3d(x * scale * 1.5, y * scale * 1.5, z * scale * 1.5; octaves=4, persistence=0.55))

        # 3. Combine Worley structure with billowed detail
        base = worley_weight * worley + (1.0 - worley_weight) * billow

        # 4. Add fine turbulence for realistic detail
        turb = fbm3d(x * scale * 4.0 + 13.7, y * scale * 4.0 - 5.3, z * scale * 4.0 + 9.1; octaves=3) * 0.12
        base += turb

        if sphere_falloff
            # Use noise to create irregular, puffy cloud boundaries
            boundary_noise = 0.15 * fbm3d(x * scale * 2.0 + 7.1, y * scale * 2.0, z * scale * 2.0 - 3.3; octaves=3)
            effective_radius = radius * (1.0 + boundary_noise)

            if dist < effective_radius
                # Density-modulated falloff: denser areas extend further (puffy protrusions)
                t = dist / effective_radius
                falloff_mod = 0.3 + 0.7 * base
                edge_fade = clamp(1.0 - (t / falloff_mod)^edge_sharpness, 0.0, 1.0)

                # Apply threshold and scale
                val = clamp((base - threshold) / (1.0 - threshold), 0.0, 1.0)
                density[ix, iy, iz] = Float32(val * edge_fade * density_scale)
            else
                density[ix, iy, iz] = 0f0
            end
        else
            # Pure noise without spherical mask
            val = clamp((base - threshold) / (1.0 - threshold), 0.0, 1.0)
            density[ix, iy, iz] = Float32(val * density_scale)
        end
    end
    density
end
