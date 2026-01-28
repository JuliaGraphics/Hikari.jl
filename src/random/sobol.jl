# Sobol sampler implementation matching PBRT-v4's ZSobolSampler
# Reference: pbrt-v4/src/pbrt/samplers.h, pbrt-v4/src/pbrt/util/lowdiscrepancy.h

const ONE_MINUS_EPSILON = Float32(1.0) - eps(Float32)
const FLOAT32_SCALE = Float32(2.3283064365386963e-10)  # 1/2^32 = 0x1p-32

# =============================================================================
# Hash functions for ZSobol scrambling (matching pbrt-v4/src/pbrt/util/hash.h)
# =============================================================================

"""
    zsobol_hash(dimension::Int32, seed::UInt32) -> UInt64

Hash function for ZSobol scrambling, matching pbrt-v4's Hash(dimension, seed).
Uses MurmurHash64A on the byte representation of (dimension, seed).
"""
@inline function zsobol_hash(dimension::Int32, seed::UInt32)::UInt64
    # Convert to bytes (little-endian)
    dim_bits = Core.bitcast(UInt32, dimension)
    bytes = (
        UInt8(dim_bits & 0xff),
        UInt8((dim_bits >> 8) & 0xff),
        UInt8((dim_bits >> 16) & 0xff),
        UInt8((dim_bits >> 24) & 0xff),
        UInt8(seed & 0xff),
        UInt8((seed >> 8) & 0xff),
        UInt8((seed >> 16) & 0xff),
        UInt8((seed >> 24) & 0xff)
    )
    murmur_hash_64a(bytes, UInt64(0))
end

# =============================================================================
# Bit manipulation functions (matching pbrt-v4/src/pbrt/util/math.h)
# =============================================================================

"""
    reverse_bits32(n::UInt32) -> UInt32

Reverse the bits of a 32-bit unsigned integer.
Uses Julia's built-in bitreverse which works on both CPU and GPU.
Reference: pbrt-v4/src/pbrt/util/math.h ReverseBits32
"""
@inline reverse_bits32(n::UInt32)::UInt32 = bitreverse(n)

# =============================================================================
# Morton/Z-order encoding (matching pbrt-v4/src/pbrt/util/vecmath.h)
# =============================================================================

"""
    left_shift2(x::UInt64) -> UInt64

Spread bits of x for Morton encoding (interleave with zeros).
"""
@inline function left_shift2(x::UInt64)::UInt64
    x &= 0xffffffff
    x = (x ⊻ (x << 16)) & 0x0000ffff0000ffff
    x = (x ⊻ (x << 8))  & 0x00ff00ff00ff00ff
    x = (x ⊻ (x << 4))  & 0x0f0f0f0f0f0f0f0f
    x = (x ⊻ (x << 2))  & 0x3333333333333333
    x = (x ⊻ (x << 1))  & 0x5555555555555555
    return x
end

"""
    encode_morton2(x::UInt32, y::UInt32) -> UInt64

Encode two 32-bit coordinates into a single 64-bit Morton code.
Reference: pbrt-v4/src/pbrt/util/vecmath.h EncodeMorton2
"""
@inline function encode_morton2(x::UInt32, y::UInt32)::UInt64
    return (left_shift2(UInt64(y)) << 1) | left_shift2(UInt64(x))
end

# =============================================================================
# Scramblers (matching pbrt-v4/src/pbrt/util/lowdiscrepancy.h)
# =============================================================================

"""
    fast_owen_scramble(v::UInt32, seed::UInt32) -> UInt32

Fast approximation of Owen scrambling for Sobol sequences.
Reference: pbrt-v4/src/pbrt/util/lowdiscrepancy.h FastOwenScrambler (lines 220-237)
"""
@inline function fast_owen_scramble(v::UInt32, seed::UInt32)::UInt32
    v = reverse_bits32(v)
    v ⊻= v * 0x3d20adea
    v += seed
    v *= (seed >> 16) | UInt32(1)
    v ⊻= v * 0x05526c56
    v ⊻= v * 0x53a22864
    return reverse_bits32(v)
end

"""
    binary_permute_scramble(v::UInt32, perm::UInt32) -> UInt32

Simple XOR-based scrambling (less quality than Owen, but faster).
Reference: pbrt-v4/src/pbrt/util/lowdiscrepancy.h BinaryPermuteScrambler
"""
@inline function binary_permute_scramble(v::UInt32, perm::UInt32)::UInt32
    return perm ⊻ v
end

# =============================================================================
# Core Sobol sample function (matching pbrt-v4/src/pbrt/util/lowdiscrepancy.h)
# =============================================================================

"""
    sobol_sample(a::Int64, dimension::Int32, scramble_seed::UInt32, sobol_matrices) -> Float32

Generate a single Sobol sample for the given index and dimension.
Reference: pbrt-v4/src/pbrt/util/lowdiscrepancy.h SobolSample (lines 167-180)

Arguments:
- a: Sample index (0-based)
- dimension: Sobol dimension (0-based, max NSOBOL_DIMENSIONS-1)
- scramble_seed: Seed for FastOwen scrambling
- sobol_matrices: The Sobol generator matrices array (must be GPU-accessible)

Uses compile-time unrolled loop with branchless bitwise operations for SPIR-V compatibility.
"""
@inline function sobol_sample(a::Int64, dimension::Int32, scramble_seed::UInt32, sobol_matrices)::Float32
    # Compute Sobol sample via generator matrix multiplication (XOR)
    v = UInt32(0)
    base_i = dimension * SOBOL_MATRIX_SIZE + Int32(1)  # Julia is 1-indexed

    # Compile-time unrolled loop (52 iterations for SOBOL_MATRIX_SIZE)
    # Uses pure branchless bitwise operations for SPIR-V structured control flow
    Base.Cartesian.@nexprs 52 bit -> begin
        bit0 = Int32(bit) - Int32(1)
        # Extract bit from 'a': either 0 or 1
        bit_val = UInt32((a >> bit0) & Int64(1))
        # Create mask: 0xffffffff if bit set, 0x00000000 otherwise
        mask = bit_val * UInt32(0xffffffff)
        # XOR with masked matrix value (branchless conditional)
        @inbounds v ⊻= sobol_matrices[base_i + bit0] & mask
    end

    # Apply FastOwen scrambling for decorrelation
    v = fast_owen_scramble(v, scramble_seed)

    # Convert to [0, 1) float
    return min(Float32(v) * FLOAT32_SCALE, ONE_MINUS_EPSILON)
end

"""
    sobol_sample_unscrambled(a::Int64, dimension::Int32) -> Float32

Generate an unscrambled Sobol sample (for debugging/comparison).
"""
@inline function sobol_sample_unscrambled(a::Int64, dimension::Int32)::Float32
    v = UInt32(0)
    i = dimension * SOBOL_MATRIX_SIZE + Int32(1)
    a_bits = a
    while a_bits != 0
        if (a_bits & 1) != 0
            @inbounds v ⊻= SobolMatrices32[i]
        end
        a_bits >>= 1
        i += Int32(1)
    end
    return min(Float32(v) * FLOAT32_SCALE, ONE_MINUS_EPSILON)
end

# =============================================================================
# ZSobol sampler (matching pbrt-v4/src/pbrt/samplers.h ZSobolSampler)
# =============================================================================

# 24 permutations of base-4 digits (0,1,2,3)
# Using Tuple{Tuple{...}} so it works as a compile-time constant on GPU
# Reference: pbrt-v4/src/pbrt/samplers.h:303-330
const PERMUTATIONS_4WAY = (
    (UInt64(0), UInt64(1), UInt64(2), UInt64(3)),  # perm 0
    (UInt64(0), UInt64(1), UInt64(3), UInt64(2)),  # perm 1
    (UInt64(0), UInt64(2), UInt64(1), UInt64(3)),  # perm 2
    (UInt64(0), UInt64(2), UInt64(3), UInt64(1)),  # perm 3
    (UInt64(0), UInt64(3), UInt64(2), UInt64(1)),  # perm 4
    (UInt64(0), UInt64(3), UInt64(1), UInt64(2)),  # perm 5
    (UInt64(1), UInt64(0), UInt64(2), UInt64(3)),  # perm 6
    (UInt64(1), UInt64(0), UInt64(3), UInt64(2)),  # perm 7
    (UInt64(1), UInt64(2), UInt64(0), UInt64(3)),  # perm 8
    (UInt64(1), UInt64(2), UInt64(3), UInt64(0)),  # perm 9
    (UInt64(1), UInt64(3), UInt64(2), UInt64(0)),  # perm 10
    (UInt64(1), UInt64(3), UInt64(0), UInt64(2)),  # perm 11
    (UInt64(2), UInt64(1), UInt64(0), UInt64(3)),  # perm 12
    (UInt64(2), UInt64(1), UInt64(3), UInt64(0)),  # perm 13
    (UInt64(2), UInt64(0), UInt64(1), UInt64(3)),  # perm 14
    (UInt64(2), UInt64(0), UInt64(3), UInt64(1)),  # perm 15
    (UInt64(2), UInt64(3), UInt64(0), UInt64(1)),  # perm 16
    (UInt64(2), UInt64(3), UInt64(1), UInt64(0)),  # perm 17
    (UInt64(3), UInt64(1), UInt64(2), UInt64(0)),  # perm 18
    (UInt64(3), UInt64(1), UInt64(0), UInt64(2)),  # perm 19
    (UInt64(3), UInt64(2), UInt64(1), UInt64(0)),  # perm 20
    (UInt64(3), UInt64(2), UInt64(0), UInt64(1)),  # perm 21
    (UInt64(3), UInt64(0), UInt64(2), UInt64(1)),  # perm 22
    (UInt64(3), UInt64(0), UInt64(1), UInt64(2)),  # perm 23
)

# Branchless max for Int32 - avoids potential branching in max()
@inline function branchless_max_i32(a::Int32, b::Int32)::Int32
    diff = a - b
    # If a >= b, diff >= 0, so (diff >> 31) is 0, result is a
    # If a < b, diff < 0, so (diff >> 31) is -1 (all 1s), result is b
    mask = diff >> Int32(31)  # arithmetic shift: 0 if a >= b, -1 if a < b
    return b + (diff & ~mask)
end

# Direct permutation lookup from PERMUTATIONS_4WAY using tuple indexing
# Returns the permuted digit for a given permutation index p (1-24) and digit (0-3)
@inline function lookup_permutation(p::Int32, digit::Int32)::UInt64
    # p is 1-indexed (1-24), digit is 0-indexed (0-3)
    # Direct tuple indexing - GPU-safe with @inbounds
    @inbounds perm_tuple = PERMUTATIONS_4WAY[p]
    @inbounds return perm_tuple[digit + Int32(1)]
end

"""
    zsobol_get_sample_index(morton_index, dimension, log2_spp, n_base4_digits) -> UInt64

Compute the permuted sample index for ZSobol sampling.
Reference: pbrt-v4/src/pbrt/samplers.h ZSobolSampler::GetSampleIndex (lines 301-356)

This applies random base-4 digit permutations to the Morton-encoded index,
ensuring good sample distribution across pixels while maintaining low-discrepancy.

Uses compile-time unrolled loop with branchless operations for SPIR-V compatibility.
"""
@inline function zsobol_get_sample_index(
    morton_index::UInt64,
    dimension::Int32,
    log2_spp::Int32,
    n_base4_digits::Int32
)::UInt64
    sample_index = UInt64(0)

    # Branchless computation of pow2_samples flag and derived values
    pow2_flag = (log2_spp & Int32(1))  # 0 or 1
    last_digit = pow2_flag
    pow2_adjust = pow2_flag

    # Compile-time unrolled loop (32 iterations covers up to 64-bit Morton codes)
    Base.Cartesian.@nexprs 32 iter -> begin
        iter0 = Int32(iter) - Int32(1)
        i = n_base4_digits - Int32(1) - iter0

        # Branchless max to ensure digit_shift >= 0
        raw_shift = Int32(2) * i - pow2_adjust
        digit_shift = branchless_max_i32(Int32(0), raw_shift)

        # Extract base-4 digit
        digit = u_int32((morton_index >> digit_shift) & UInt64(3))

        # Compute permutation index from higher digits and dimension
        higher_digits = morton_index >> (digit_shift + Int32(2))
        hash_val = mix_bits(higher_digits ⊻ (UInt64(0x55555555) * u_uint64(dimension)))
        p = u_int32((hash_val >> 24) % UInt64(24)) + Int32(1)  # 1-indexed

        # Branchless permutation lookup
        permuted_digit = lookup_permutation(p, digit)

        # Branchless conditional: only apply if i >= last_digit
        # Create mask: all 1s if i >= last_digit, all 0s otherwise
        apply_mask = UInt64(u_int32(i >= last_digit)) * UInt64(0xffffffffffffffff)
        sample_index |= (permuted_digit << digit_shift) & apply_mask
    end

    # Handle power-of-2 (but not power-of-4) sample count
    digit = morton_index & UInt64(1)
    xor_bit = mix_bits((morton_index >> 1) ⊻ (UInt64(0x55555555) * u_uint64(dimension))) & UInt64(1)
    # Branchless: mask by pow2_flag
    pow2_mask = UInt64(pow2_flag) * UInt64(0xffffffffffffffff)
    sample_index |= (digit ⊻ xor_bit) & pow2_mask

    return sample_index
end

# =============================================================================
# High-level sampling functions for use in integrators
# =============================================================================

"""
    zsobol_sample_1d(px, py, sample_idx, dim, log2_spp, n_base4_digits, seed, sobol_matrices) -> Float32

Generate a 1D Sobol sample for the given pixel and sample index.
"""
@inline function zsobol_sample_1d(
    px::Int32, py::Int32, sample_idx::Int32, dim::Int32,
    log2_spp::Int32, n_base4_digits::Int32, seed::UInt32, sobol_matrices
)::Float32
    # Convert pixel coords to UInt32 (px, py are always non-negative)
    morton_index = (encode_morton2(u_uint32(px), u_uint32(py)) << log2_spp) | u_uint64(sample_idx)
    sobol_index = zsobol_get_sample_index(morton_index, dim, log2_spp, n_base4_digits)
    # pbrt-v4 compatibility: Hash uses dimension AFTER increment (dim + 1 for 1D samples)
    # See pbrt-v4/src/pbrt/samplers.h ZSobolSampler::Get1D() lines 258-262
    # Uses MurmurHash64A on (dimension, seed) bytes, then truncates to 32-bit
    hash_dim = dim + Int32(1)
    sample_hash = u_uint32(zsobol_hash(hash_dim, seed))
    return sobol_sample(Int64(sobol_index), Int32(0), sample_hash, sobol_matrices)
end

"""
    zsobol_sample_2d(px, py, sample_idx, dim, log2_spp, n_base4_digits, seed, sobol_matrices) -> (Float32, Float32)

Generate a 2D Sobol sample for the given pixel and sample index.
Uses two consecutive Sobol dimensions with independent scrambling seeds.
"""
@inline function zsobol_sample_2d(
    px::Int32, py::Int32, sample_idx::Int32, dim::Int32,
    log2_spp::Int32, n_base4_digits::Int32, seed::UInt32, sobol_matrices
)::Tuple{Float32, Float32}
    # Convert pixel coords to UInt32 (px, py are always non-negative)
    morton_index = (encode_morton2(u_uint32(px), u_uint32(py)) << log2_spp) | u_uint64(sample_idx)
    sobol_index = zsobol_get_sample_index(morton_index, dim, log2_spp, n_base4_digits)

    # pbrt-v4 compatibility: Hash uses dimension AFTER increment (dim + 2 for 2D samples)
    # See pbrt-v4/src/pbrt/samplers.h ZSobolSampler::Get2D() lines 274-279
    # Uses MurmurHash64A on (dimension, seed) bytes, then splits into two 32-bit hashes
    hash_dim = dim + Int32(2)
    bits = zsobol_hash(hash_dim, seed)
    hash1 = u_uint32(bits)
    hash2 = u_uint32(bits >> 32)

    u1 = sobol_sample(Int64(sobol_index), Int32(0), hash1, sobol_matrices)
    u2 = sobol_sample(Int64(sobol_index), Int32(1), hash2, sobol_matrices)
    return (u1, u2)
end

"""
    compute_zsobol_params(samples_per_pixel::Int, width::Int, height::Int) -> (Int32, Int32)

Compute the ZSobol sampler parameters from render settings.
Returns (log2_spp, n_base4_digits).
"""
function compute_zsobol_params(samples_per_pixel::Int, width::Int, height::Int)
    log2_spp = Int32(ceil(Int, log2(max(1, samples_per_pixel))))
    resolution_log2 = Int32(ceil(Int, log2(max(width, height))))
    log4_spp = (log2_spp + Int32(1)) ÷ Int32(2)
    n_base4_digits = resolution_log2 + log4_spp
    return (log2_spp, n_base4_digits)
end

# =============================================================================
# SobolRNG - GPU-compatible Sobol sampler struct
# =============================================================================

using Adapt
import KernelAbstractions as KA

"""
    SobolRNG{M}

GPU-compatible Sobol random number generator.
Holds the Sobol matrices and precomputed parameters for ZSobol sampling.

This struct can be passed directly to GPU kernels via Adapt.jl integration.
All sampling state is computed on-the-fly from pixel coordinates and sample index,
making it stateless and thread-safe.

# Fields
- `matrices::M`: GPU array of Sobol generator matrices (UInt32)
- `log2_spp::Int32`: log2 of samples per pixel
- `n_base4_digits::Int32`: number of base-4 digits for Morton encoding
- `seed::UInt32`: scrambling seed
- `width::Int32`: image width (for pixel coordinate recovery)
"""
struct SobolRNG{M <: AbstractVector{UInt32}}
    matrices::M
    log2_spp::Int32
    n_base4_digits::Int32
    seed::UInt32
    width::Int32
end

"""
    SobolRNG(backend, seed::UInt32, width::Integer, height::Integer, samples_per_pixel::Integer)

Create a SobolRNG for the given render settings.
Allocates Sobol matrices on the specified backend (CPU/GPU).

# Arguments
- `backend`: KernelAbstractions backend (e.g., `CPU()`, `CUDABackend()`, `OpenCLBackend()`)
- `seed`: Scrambling seed for decorrelation
- `width`: Image width in pixels
- `height`: Image height in pixels
- `samples_per_pixel`: Number of samples per pixel
"""
function SobolRNG(backend, seed::UInt32, width::Integer, height::Integer, samples_per_pixel::Integer)
    # Allocate and copy Sobol matrices to GPU
    matrices = KA.allocate(backend, UInt32, length(SobolMatrices32))
    KA.copyto!(backend, matrices, SobolMatrices32)

    # Compute ZSobol parameters
    log2_spp, n_base4_digits = compute_zsobol_params(Int(samples_per_pixel), Int(width), Int(height))

    return SobolRNG(matrices, log2_spp, n_base4_digits, seed, Int32(width))
end

# Adapt.jl integration for GPU kernels
function Adapt.adapt_structure(to, rng::SobolRNG)
    SobolRNG(
        Adapt.adapt(to, rng.matrices),
        rng.log2_spp,
        rng.n_base4_digits,
        rng.seed,
        rng.width
    )
end

"""
    cleanup!(rng::SobolRNG)

Release GPU memory held by the SobolRNG.
"""
function cleanup!(rng::SobolRNG)
    finalize(rng.matrices)
    return nothing
end

# =============================================================================
# SobolRNG Sampling Interface
# =============================================================================

"""
    sample_1d(rng::SobolRNG, px::Int32, py::Int32, sample_idx::Int32, dim::Int32) -> Float32

Generate a 1D Sobol sample for the given pixel and dimension.
"""
@inline function sample_1d(rng::SobolRNG, px::Int32, py::Int32, sample_idx::Int32, dim::Int32)::Float32
    zsobol_sample_1d(px, py, sample_idx, dim, rng.log2_spp, rng.n_base4_digits, rng.seed, rng.matrices)
end

"""
    sample_2d(rng::SobolRNG, px::Int32, py::Int32, sample_idx::Int32, dim::Int32) -> Tuple{Float32, Float32}

Generate a 2D Sobol sample for the given pixel and dimension.
"""
@inline function sample_2d(rng::SobolRNG, px::Int32, py::Int32, sample_idx::Int32, dim::Int32)::Tuple{Float32, Float32}
    zsobol_sample_2d(px, py, sample_idx, dim, rng.log2_spp, rng.n_base4_digits, rng.seed, rng.matrices)
end

"""
    compute_pixel_sample(rng::SobolRNG, px::Int32, py::Int32, sample_idx::Int32)

Compute all camera sample values for a pixel using SobolRNG.
Returns a PixelSample struct with jitter, wavelength, lens, and time samples.

# Dimension allocation (matching PBRT-v4 exactly):
# See pbrt-v4/src/pbrt/wavefront/camera.cpp:51-60 and samplers.h:797-814
# pbrt-v4's Get1D/Get2D increment dimension BEFORE computing hash:
# - StartPixelSample(pPixel, sampleIndex, 0): dimension = 0
# - Get1D() for wavelength: dimension++ → 1, hash uses 1
# - GetPixel2D() which is Get2D(): dimension += 2 → 3, hash uses 3 (for jitter)
# - Get1D() for time: dimension++ → 4, hash uses 4
# - Get2D() for lens: dimension += 2 → 6, hash uses 6
# So dimensions used are: 1 (wavelength), 3 (jitter), 4 (time), 6 (lens)
"""
@inline function compute_pixel_sample(rng::SobolRNG, px::Int32, py::Int32, sample_idx::Int32)
    # Match pbrt-v4 dimension usage after Get1D/Get2D increments
    wavelength_u = sample_1d(rng, px, py, sample_idx, Int32(1))       # dim 1
    jitter_x, jitter_y = sample_2d(rng, px, py, sample_idx, Int32(3)) # dim 3
    time = sample_1d(rng, px, py, sample_idx, Int32(4))               # dim 4
    lens_u, lens_v = sample_2d(rng, px, py, sample_idx, Int32(6))     # dim 6
    return PixelSample(jitter_x, jitter_y, wavelength_u, lens_u, lens_v, time)
end

"""
    compute_path_sample_1d(rng::SobolRNG, px::Int32, py::Int32, sample_idx::Int32, depth::Int32, local_dim::Int32) -> Float32

Compute a 1D sample for path tracing at a given depth using SobolRNG.

# Dimension allocation:
- Base dimension = 6 (camera uses 0-5)
- Each depth uses 8 dimensions for BSDF, light, RR, etc.
"""
@inline function compute_path_sample_1d(rng::SobolRNG, px::Int32, py::Int32, sample_idx::Int32, depth::Int32, local_dim::Int32)::Float32
    dim = Int32(6) + depth * Int32(8) + local_dim
    sample_1d(rng, px, py, sample_idx, dim)
end

"""
    compute_path_sample_2d(rng::SobolRNG, px::Int32, py::Int32, sample_idx::Int32, depth::Int32, local_dim::Int32) -> Tuple{Float32, Float32}

Compute a 2D sample for path tracing at a given depth using SobolRNG.
"""
@inline function compute_path_sample_2d(rng::SobolRNG, px::Int32, py::Int32, sample_idx::Int32, depth::Int32, local_dim::Int32)::Tuple{Float32, Float32}
    dim = Int32(6) + depth * Int32(8) + local_dim
    sample_2d(rng, px, py, sample_idx, dim)
end
