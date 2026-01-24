# ============================================================================
# Stratified Sampler - GPU-compatible deterministic sampling
# ============================================================================
# Implements pbrt-v4 style deterministic sampling that works on GPU.
# Each sample is computed purely from (pixel_x, pixel_y, sample_index, dimension).

# ============================================================================
# Hash functions for sampler
# ============================================================================

"""
    int32_to_bytes(v::Int32) -> NTuple{4,UInt8}

GPU-compatible conversion of Int32 to bytes.
"""
@inline function int32_to_bytes(v::Int32)::NTuple{4,UInt8}
    bits = Core.bitcast(UInt32, v)
    return (
        UInt8(bits & 0xff),
        UInt8((bits >> 8) & 0xff),
        UInt8((bits >> 16) & 0xff),
        UInt8((bits >> 24) & 0xff)
    )
end

"""
    uint32_to_bytes(v::UInt32) -> NTuple{4,UInt8}

GPU-compatible conversion of UInt32 to bytes.
"""
@inline function uint32_to_bytes(v::UInt32)::NTuple{4,UInt8}
    return (
        UInt8(v & 0xff),
        UInt8((v >> 8) & 0xff),
        UInt8((v >> 16) & 0xff),
        UInt8((v >> 24) & 0xff)
    )
end

"""
    sampler_hash(px::Int32, py::Int32, sample_idx::Int32, dim::Int32, seed::UInt32=0) -> UInt64

Hash function for stratified sampler, matching pbrt-v4's Hash() behavior.
Produces deterministic, well-distributed hash for (pixel, sample, dimension) tuples.
"""
@inline function sampler_hash(px::Int32, py::Int32, sample_idx::Int32, dim::Int32, seed::UInt32=UInt32(0))::UInt64
    # Pack all values into bytes
    px_bytes = int32_to_bytes(px)
    py_bytes = int32_to_bytes(py)
    sample_bytes = int32_to_bytes(sample_idx)
    dim_bytes = int32_to_bytes(dim)
    seed_bytes = uint32_to_bytes(seed)

    bytes = (px_bytes..., py_bytes..., sample_bytes..., dim_bytes..., seed_bytes...)
    murmur_hash_64a(bytes, UInt64(0))
end

"""
    sampler_hash_2d(px::Int32, py::Int32, sample_idx::Int32, dim::Int32) -> UInt64

Simplified hash for 2D samples where we need two correlated values.
"""
@inline function sampler_hash_2d(px::Int32, py::Int32, sample_idx::Int32, dim::Int32)::UInt64
    sampler_hash(px, py, sample_idx, dim, UInt32(0))
end

# ============================================================================
# Core sampling functions
# ============================================================================

# One minus epsilon for clamping to [0, 1)
const ONE_MINUS_EPSILON = Float32(1.0) - eps(Float32)

"""
    stratified_sample_1d(px::Int32, py::Int32, sample_idx::Int32, dim::Int32) -> Float32

Generate a deterministic 1D sample in [0, 1) for the given pixel, sample index, and dimension.
This matches pbrt-v4's IndependentSampler behavior.
"""
@inline function stratified_sample_1d(px::Int32, py::Int32, sample_idx::Int32, dim::Int32)::Float32
    hash = sampler_hash(px, py, sample_idx, dim, UInt32(0))
    # Convert to [0, 1) - use high bits for better distribution
    return min(Float32(hash >> 40) / Float32(1 << 24), ONE_MINUS_EPSILON)
end

"""
    stratified_sample_2d(px::Int32, py::Int32, sample_idx::Int32, dim::Int32) -> Tuple{Float32, Float32}

Generate a deterministic 2D sample in [0, 1)² for the given pixel, sample index, and dimension.
Uses two consecutive dimensions for the two components.
"""
@inline function stratified_sample_2d(px::Int32, py::Int32, sample_idx::Int32, dim::Int32)::Tuple{Float32, Float32}
    # Use two separate hashes for better 2D distribution
    hash1 = sampler_hash(px, py, sample_idx, dim, UInt32(0))
    hash2 = sampler_hash(px, py, sample_idx, dim + Int32(1), UInt32(0))

    u1 = min(Float32(hash1 >> 40) / Float32(1 << 24), ONE_MINUS_EPSILON)
    u2 = min(Float32(hash2 >> 40) / Float32(1 << 24), ONE_MINUS_EPSILON)

    return (u1, u2)
end

# ============================================================================
# R2 Quasi-Random Sequence (Cranley-Patterson rotation)
# ============================================================================
# R2 sequence provides better 2D stratification than pure random.
# We use pixel hash as the offset (Cranley-Patterson rotation).

# R2 sequence constants (Roberts, 2018)
# Plastic constant roots for 2D
const R2_PHI2 = Float32(1.32471795724474602596)  # Plastic constant
const R2_ALPHA1 = Float32(1.0 / R2_PHI2)
const R2_ALPHA2 = Float32(1.0 / (R2_PHI2 * R2_PHI2))

"""
    r2_sample(n::Int32) -> Tuple{Float32, Float32}

Generate the n-th point of the R2 quasi-random sequence.
The R2 sequence has excellent 2D discrepancy properties.
"""
@inline function r2_sample(n::Int32)::Tuple{Float32, Float32}
    # R2 sequence: x_n = fract(0.5 + n * alpha1), y_n = fract(0.5 + n * alpha2)
    u1 = mod(0.5f0 + Float32(n) * R2_ALPHA1, 1.0f0)
    u2 = mod(0.5f0 + Float32(n) * R2_ALPHA2, 1.0f0)
    return (u1, u2)
end

"""
    r2_sample_rotated(n::Int32, offset_x::Float32, offset_y::Float32) -> Tuple{Float32, Float32}

Generate the n-th point of the R2 sequence with Cranley-Patterson rotation.
The offset is typically derived from a hash of the pixel coordinates.
"""
@inline function r2_sample_rotated(n::Int32, offset_x::Float32, offset_y::Float32)::Tuple{Float32, Float32}
    u1, u2 = r2_sample(n)
    # Cranley-Patterson rotation (add offset and wrap)
    u1 = mod(u1 + offset_x, 1.0f0)
    u2 = mod(u2 + offset_y, 1.0f0)
    return (u1, u2)
end

"""
    pixel_offset_2d(px::Int32, py::Int32, dim::Int32) -> Tuple{Float32, Float32}

Compute a deterministic 2D offset for Cranley-Patterson rotation based on pixel coordinates.
"""
@inline function pixel_offset_2d(px::Int32, py::Int32, dim::Int32)::Tuple{Float32, Float32}
    hash = sampler_hash(px, py, Int32(0), dim, UInt32(0x12345678))
    offset_x = Float32((hash >> 32) & 0xFFFFFFFF) / Float32(0x100000000)
    offset_y = Float32(hash & 0xFFFFFFFF) / Float32(0x100000000)
    return (offset_x, offset_y)
end

# ============================================================================
# High-level sampler interface for VolPath
# ============================================================================

"""
    PixelSample

Holds all sample values needed for a single pixel sample in path tracing.
This is a stateless struct computed deterministically from (pixel, sample_index).
"""
struct PixelSample
    # Camera ray jitter
    jitter_x::Float32
    jitter_y::Float32

    # Wavelength selection
    wavelength_u::Float32

    # Lens sampling (for DoF)
    lens_u::Float32
    lens_v::Float32

    # Time (for motion blur)
    time::Float32
end

"""
    compute_pixel_sample(px::Int32, py::Int32, sample_idx::Int32) -> PixelSample

Compute all sample values for a pixel sample deterministically.
Uses R2 sequence for pixel jitter (better 2D stratification) and
hash-based sampling for other dimensions.

NOTE: This is the old hash-based version. Use compute_pixel_sample_sobol for better convergence.
"""
@inline function compute_pixel_sample(px::Int32, py::Int32, sample_idx::Int32)::PixelSample
    # Dimensions:
    # 0-1: pixel jitter (use R2 for better 2D stratification)
    # 2: wavelength
    # 3-4: lens
    # 5: time

    # R2 sequence with per-pixel rotation for pixel jitter
    offset_x, offset_y = pixel_offset_2d(px, py, Int32(0))
    jitter_x, jitter_y = r2_sample_rotated(sample_idx, offset_x, offset_y)

    # Hash-based sampling for other dimensions
    wavelength_u = stratified_sample_1d(px, py, sample_idx, Int32(2))
    lens_u, lens_v = stratified_sample_2d(px, py, sample_idx, Int32(3))
    time = stratified_sample_1d(px, py, sample_idx, Int32(5))

    return PixelSample(jitter_x, jitter_y, wavelength_u, lens_u, lens_v, time)
end

"""
    compute_path_sample_1d(px::Int32, py::Int32, sample_idx::Int32, depth::Int32, local_dim::Int32) -> Float32

Compute a 1D sample for path tracing at a given depth.
Each depth gets a separate set of dimensions to avoid correlation.

NOTE: This is the old hash-based version. Use compute_path_sample_1d_sobol for better convergence.
"""
@inline function compute_path_sample_1d(px::Int32, py::Int32, sample_idx::Int32, depth::Int32, local_dim::Int32)::Float32
    # Base dimension for camera samples is 6
    # Each depth uses 8 dimensions (for BSDF, light, RR, etc.)
    dim = Int32(6) + depth * Int32(8) + local_dim
    return stratified_sample_1d(px, py, sample_idx, dim)
end

"""
    compute_path_sample_2d(px::Int32, py::Int32, sample_idx::Int32, depth::Int32, local_dim::Int32) -> Tuple{Float32, Float32}

Compute a 2D sample for path tracing at a given depth.

NOTE: This is the old hash-based version. Use compute_path_sample_2d_sobol for better convergence.
"""
@inline function compute_path_sample_2d(px::Int32, py::Int32, sample_idx::Int32, depth::Int32, local_dim::Int32)::Tuple{Float32, Float32}
    # Each depth uses 8 dimensions
    dim = Int32(6) + depth * Int32(8) + local_dim
    return stratified_sample_2d(px, py, sample_idx, dim)
end

# ============================================================================
# ZSobol-based sampling (matching PBRT-v4)
# ============================================================================
# These functions use the Sobol low-discrepancy sequence with Owen scrambling
# for better convergence than the hash-based samplers above.

"""
    compute_pixel_sample_sobol(px, py, sample_idx, log2_spp, n_base4_digits, seed, sobol_matrices) -> PixelSample

Compute all sample values for a pixel sample using ZSobol sampler.
This matches PBRT-v4's ZSobolSampler for better convergence.
"""
@inline function compute_pixel_sample_sobol(
    px::Int32, py::Int32, sample_idx::Int32,
    log2_spp::Int32, n_base4_digits::Int32, seed::UInt32, sobol_matrices
)::PixelSample
    # Dimensions (matching PBRT-v4):
    # 0-1: pixel jitter
    # 2: wavelength
    # 3-4: lens
    # 5: time

    jitter_x, jitter_y = zsobol_sample_2d(px, py, sample_idx, Int32(0), log2_spp, n_base4_digits, seed, sobol_matrices)
    wavelength_u = zsobol_sample_1d(px, py, sample_idx, Int32(2), log2_spp, n_base4_digits, seed, sobol_matrices)
    lens_u, lens_v = zsobol_sample_2d(px, py, sample_idx, Int32(3), log2_spp, n_base4_digits, seed, sobol_matrices)
    time = zsobol_sample_1d(px, py, sample_idx, Int32(5), log2_spp, n_base4_digits, seed, sobol_matrices)

    return PixelSample(jitter_x, jitter_y, wavelength_u, lens_u, lens_v, time)
end

"""
    compute_path_sample_1d_sobol(px, py, sample_idx, depth, local_dim, log2_spp, n_base4_digits, seed, sobol_matrices) -> Float32

Compute a 1D sample for path tracing using ZSobol sampler.
"""
@inline function compute_path_sample_1d_sobol(
    px::Int32, py::Int32, sample_idx::Int32, depth::Int32, local_dim::Int32,
    log2_spp::Int32, n_base4_digits::Int32, seed::UInt32, sobol_matrices
)::Float32
    # Base dimension for camera samples is 6
    # Each depth uses 8 dimensions (for BSDF, light, RR, etc.)
    dim = Int32(6) + depth * Int32(8) + local_dim
    return zsobol_sample_1d(px, py, sample_idx, dim, log2_spp, n_base4_digits, seed, sobol_matrices)
end

"""
    compute_path_sample_2d_sobol(px, py, sample_idx, depth, local_dim, log2_spp, n_base4_digits, seed, sobol_matrices) -> Tuple{Float32, Float32}

Compute a 2D sample for path tracing using ZSobol sampler.
"""
@inline function compute_path_sample_2d_sobol(
    px::Int32, py::Int32, sample_idx::Int32, depth::Int32, local_dim::Int32,
    log2_spp::Int32, n_base4_digits::Int32, seed::UInt32, sobol_matrices
)::Tuple{Float32, Float32}
    # Each depth uses 8 dimensions
    dim = Int32(6) + depth * Int32(8) + local_dim
    return zsobol_sample_2d(px, py, sample_idx, dim, log2_spp, n_base4_digits, seed, sobol_matrices)
end
