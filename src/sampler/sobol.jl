# Sobol sampler implementation matching PBRT-v4's ZSobolSampler
# Reference: pbrt-v4/src/pbrt/samplers.h, pbrt-v4/src/pbrt/util/lowdiscrepancy.h

const ONE_MINUS_EPSILON = Float32(1.0) - eps(Float32)
const FLOAT32_SCALE = Float32(2.3283064365386963e-10)  # 1/2^32 = 0x1p-32

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

# Helper for sobol_sample unrolled loop - must not capture variables
@inline function _sobol_xor_bit!(bit::Int32, v_ref::Base.RefValue{UInt32}, a::Int64, base_i::Int32, sobol_matrices)
    # bit is 1-indexed from for_unrolled, convert to 0-indexed
    bit0 = bit - Int32(1)
    if (a & (Int64(1) << bit0)) != Int64(0)
        @inbounds v_ref[] ⊻= sobol_matrices[base_i + bit0]
    end
    return nothing
end

"""
    sobol_sample(a::Int64, dimension::Int32, scramble_seed::UInt32, sobol_matrices) -> Float32

Generate a single Sobol sample for the given index and dimension.
Reference: pbrt-v4/src/pbrt/util/lowdiscrepancy.h SobolSample (lines 167-180)

Arguments:
- a: Sample index (0-based)
- dimension: Sobol dimension (0-based, max NSOBOL_DIMENSIONS-1)
- scramble_seed: Seed for FastOwen scrambling
- sobol_matrices: The Sobol generator matrices array (must be GPU-accessible)
"""
@inline function sobol_sample(a::Int64, dimension::Int32, scramble_seed::UInt32, sobol_matrices)::Float32
    # Compute Sobol sample via generator matrix multiplication (XOR)
    # Use fixed-count loop for GPU compatibility (max 52 bits = SOBOL_MATRIX_SIZE)
    v_ref = Ref(UInt32(0))
    base_i = dimension * SOBOL_MATRIX_SIZE + Int32(1)  # Julia is 1-indexed

    # Unroll 52 iterations (for_unrolled iterates 1 to N)
    for_unrolled(_sobol_xor_bit!, Val(52), v_ref, a, base_i, sobol_matrices)

    # Apply FastOwen scrambling for decorrelation
    v = fast_owen_scramble(v_ref[], scramble_seed)

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

# Helper for zsobol_get_sample_index unrolled loop - must not capture variables
@inline function _zsobol_permute_digit!(
    iter::Int32,
    sample_index_ref::Base.RefValue{UInt64},
    morton_index::UInt64,
    dimension::Int32,
    n_base4_digits::Int32,
    last_digit::Int32,
    pow2_adjust::Int32
)
    # iter is 1-indexed from for_unrolled, convert to 0-indexed
    iter0 = iter - Int32(1)
    i = n_base4_digits - Int32(1) - iter0
    if i >= last_digit
        digit_shift = 2 * i - pow2_adjust
        digit = u_int32((morton_index >> digit_shift) & UInt64(3))

        # Choose permutation based on higher digits and dimension
        higher_digits = morton_index >> (digit_shift + 2)
        hash_val = mix_bits(higher_digits ⊻ (0x55555555 * u_uint64(dimension)))
        # p in [0, 23] - select one of 24 permutations
        p = u_int32((hash_val >> 24) % UInt64(24)) + Int32(1)  # 1-indexed for tuple

        # Lookup permuted digit from nested tuple (both 1-indexed)
        @inbounds perm = PERMUTATIONS_4WAY[p]
        @inbounds permuted_digit = perm[digit + Int32(1)]
        sample_index_ref[] |= permuted_digit << digit_shift
    end
    return nothing
end

"""
    zsobol_get_sample_index(morton_index, dimension, log2_spp, n_base4_digits) -> UInt64

Compute the permuted sample index for ZSobol sampling.
Reference: pbrt-v4/src/pbrt/samplers.h ZSobolSampler::GetSampleIndex (lines 301-356)

This applies random base-4 digit permutations to the Morton-encoded index,
ensuring good sample distribution across pixels while maintaining low-discrepancy.
"""
@inline function zsobol_get_sample_index(
    morton_index::UInt64,
    dimension::Int32,
    log2_spp::Int32,
    n_base4_digits::Int32
)::UInt64
    sample_index_ref = Ref(UInt64(0))
    pow2_samples = (log2_spp & Int32(1)) != Int32(0)
    last_digit = pow2_samples ? Int32(1) : Int32(0)
    pow2_adjust = pow2_samples ? Int32(1) : Int32(0)

    # Unrolled loop with max 32 iterations (covers up to 64-bit Morton codes)
    # GPU-friendly: fixed iteration count with conditional execution
    for_unrolled(_zsobol_permute_digit!, Val(32),
        sample_index_ref, morton_index, dimension, n_base4_digits, last_digit, pow2_adjust)

    # Handle power-of-2 (but not power-of-4) sample count
    if pow2_samples
        digit = morton_index & UInt64(1)
        xor_bit = mix_bits((morton_index >> 1) ⊻ (0x55555555 * u_uint64(dimension))) & UInt64(1)
        sample_index_ref[] |= digit ⊻ xor_bit
    end

    return sample_index_ref[]
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
    # Truncate 64-bit hash to 32-bit for scrambling
    sample_hash = u_uint32(mix_bits(u_uint64(dim) ⊻ u_uint64(seed)))
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

    # Generate two independent scrambling seeds from dimension and seed
    bits = mix_bits(u_uint64(dim) ⊻ u_uint64(seed))
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
