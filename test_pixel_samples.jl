# Unit tests for pixel_samples refactor
# Run via MCP: include("test_pixel_samples.jl")

using Hikari
using Hikari: VPRaySamples, Point2f, Vec3f, Point3f, SpectralRadiance
using Hikari: zsobol_sample_1d, zsobol_sample_2d, compute_zsobol_params
using Hikari: mix_bits, u_uint32, u_uint64, u_int32, ONE_MINUS_EPSILON

println("=" ^ 60)
println("Testing pixel_samples refactor components")
println("=" ^ 60)

# ============================================================================
# Test 1: LCG RNG
# ============================================================================
println("\n[Test 1] LCG RNG functions...")

# Inline the LCG functions for testing (they're in delta-tracking.jl)
const LCG_MULTIPLIER = UInt64(0x5DEECE66D)
const LCG_INCREMENT = UInt64(11)
const FLOAT32_SCALE = 2.3283064365386963f-10

function lcg_init_test(ray_o::Point3f, ray_d::Vec3f, t_max::Float32)::UInt64
    ox = reinterpret(UInt32, ray_o[1])
    oy = reinterpret(UInt32, ray_o[2])
    oz = reinterpret(UInt32, ray_o[3])
    tm = reinterpret(UInt32, t_max)
    dx = reinterpret(UInt32, ray_d[1])
    dy = reinterpret(UInt32, ray_d[2])
    dz = reinterpret(UInt32, ray_d[3])
    seed1 = mix_bits(u_uint64(ox) ⊻ (u_uint64(oy) << 16) ⊻ (u_uint64(oz) << 32) ⊻ u_uint64(tm))
    seed2 = mix_bits(u_uint64(dx) ⊻ (u_uint64(dy) << 16) ⊻ (u_uint64(dz) << 32))
    return seed1 ⊻ seed2
end

function lcg_next_test(state::UInt64)::Tuple{UInt64, Float32}
    new_state = state * LCG_MULTIPLIER + LCG_INCREMENT
    r = Float32(u_uint32(new_state >> 32)) * FLOAT32_SCALE
    return (new_state, min(r, ONE_MINUS_EPSILON))
end

# Test LCG produces valid values
ray_o = Point3f(0.5f0, 0.5f0, 0f0)
ray_d = Vec3f(0f0, 0f0, 1f0)
t_max = 10f0

state = lcg_init_test(ray_o, ray_d, t_max)
println("  Initial LCG state: $state")

all_valid = true
for i in 1:100
    state, r = lcg_next_test(state)
    if r < 0f0 || r >= 1f0
        println("  ERROR: LCG produced invalid value $r at iteration $i")
        all_valid = false
    end
end
println("  LCG 100 samples all in [0,1): $all_valid")

# Test determinism
state1 = lcg_init_test(ray_o, ray_d, t_max)
state2 = lcg_init_test(ray_o, ray_d, t_max)
_, r1 = lcg_next_test(state1)
_, r2 = lcg_next_test(state2)
println("  LCG is deterministic: $(r1 == r2)")

# ============================================================================
# Test 2: Sobol sample generation
# ============================================================================
println("\n[Test 2] Sobol sample generation...")

# Get Sobol matrices
sobol_matrices = Hikari.get_sobol_matrices()
println("  Sobol matrices loaded: $(size(sobol_matrices))")

# Compute params for 64 spp
log2_spp, n_base4_digits = compute_zsobol_params(64, 100, 100)
println("  For 64 spp: log2_spp=$log2_spp, n_base4_digits=$n_base4_digits")

# Test sample generation
sampler_seed = UInt32(42)
px, py = Int32(50), Int32(50)
sample_idx = Int32(0)
depth = Int32(0)

# Base dimension for this depth (camera uses dims 0-5, then 7 per bounce)
base_dim = Int32(6) + Int32(7) * depth

# Generate samples like the kernel does
direct_uc = zsobol_sample_1d(px, py, sample_idx, base_dim, log2_spp, n_base4_digits, sampler_seed, sobol_matrices)
direct_u1, direct_u2 = zsobol_sample_2d(px, py, sample_idx, base_dim + Int32(1), log2_spp, n_base4_digits, sampler_seed, sobol_matrices)
indirect_uc = zsobol_sample_1d(px, py, sample_idx, base_dim + Int32(3), log2_spp, n_base4_digits, sampler_seed, sobol_matrices)
indirect_u1, indirect_u2 = zsobol_sample_2d(px, py, sample_idx, base_dim + Int32(4), log2_spp, n_base4_digits, sampler_seed, sobol_matrices)
indirect_rr = zsobol_sample_1d(px, py, sample_idx, base_dim + Int32(6), log2_spp, n_base4_digits, sampler_seed, sobol_matrices)

println("  direct_uc = $direct_uc")
println("  direct_u = ($direct_u1, $direct_u2)")
println("  indirect_uc = $indirect_uc")
println("  indirect_u = ($indirect_u1, $indirect_u2)")
println("  indirect_rr = $indirect_rr")

# Check all values are in valid range
samples = [direct_uc, direct_u1, direct_u2, indirect_uc, indirect_u1, indirect_u2, indirect_rr]
all_valid_sobol = all(s -> 0f0 <= s < 1f0, samples)
println("  All Sobol samples in [0,1): $all_valid_sobol")

# Check samples are not all zero
any_nonzero = any(s -> s > 0f0, samples)
println("  At least one sample non-zero: $any_nonzero")

# ============================================================================
# Test 3: Different pixels produce different samples
# ============================================================================
println("\n[Test 3] Sample variation across pixels...")

# Generate samples for different pixels
samples_per_pixel = []
for (px, py) in [(Int32(0), Int32(0)), (Int32(50), Int32(50)), (Int32(99), Int32(99))]
    s = zsobol_sample_1d(px, py, sample_idx, base_dim, log2_spp, n_base4_digits, sampler_seed, sobol_matrices)
    push!(samples_per_pixel, s)
end
println("  Samples at (0,0), (50,50), (99,99): $samples_per_pixel")
all_different = length(unique(samples_per_pixel)) == 3
println("  All pixels have different samples: $all_different")

# ============================================================================
# Test 4: Different depths produce different samples
# ============================================================================
println("\n[Test 4] Sample variation across depths...")

samples_per_depth = []
for d in Int32(0):Int32(3)
    dim = Int32(6) + Int32(7) * d
    s = zsobol_sample_1d(px, py, sample_idx, dim, log2_spp, n_base4_digits, sampler_seed, sobol_matrices)
    push!(samples_per_depth, s)
end
println("  Samples at depths 0-3: $samples_per_depth")
all_different_depth = length(unique(samples_per_depth)) == 4
println("  All depths have different samples: $all_different_depth")

# ============================================================================
# Test 5: VPRaySamples struct
# ============================================================================
println("\n[Test 5] VPRaySamples struct...")

rs = VPRaySamples(
    direct_uc,
    Point2f(direct_u1, direct_u2),
    indirect_uc,
    Point2f(indirect_u1, indirect_u2),
    indirect_rr
)
println("  Created VPRaySamples: $(typeof(rs))")
println("  direct_uc = $(rs.direct_uc)")
println("  indirect_u = $(rs.indirect_u)")

# Default constructor
rs_default = VPRaySamples()
println("  Default VPRaySamples all zero: $(rs_default.direct_uc == 0f0 && rs_default.indirect_rr == 0f0)")

# ============================================================================
# Summary
# ============================================================================
println("\n" * "=" ^ 60)
println("Test Summary")
println("=" ^ 60)
all_pass = all_valid && all_valid_sobol && any_nonzero && all_different && all_different_depth
if all_pass
    println("✓ All tests passed!")
else
    println("✗ Some tests failed - check output above")
end

nothing
