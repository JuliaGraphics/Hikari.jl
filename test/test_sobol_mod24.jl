# `mod24` — the 32-bit replacement for a 64-bit remainder in the ZSobol sampler.
#
# `zsobol_get_sample_index` picks one of pbrt-v4's 24 base-4 digit permutations
# with `(hash >> 24) % 24`, once per base-4 digit, ~16 digits per sample, for
# every sample of every pixel. Written as `% UInt64(24)` that is a 64-bit
# integer division: Apple GPUs have no 64-bit divider and SPIR-V drivers emulate
# one, and it dominated the whole sampler. Measured on an M5 over 350k threads,
# a 16-iteration loop of `mix_bits` alone took 0.69 ms and the same loop with
# one `% UInt64(24)` took 4.09 ms.
#
# The replacement is exact, not approximate, and the sample stream depends on it
# bit for bit — a single wrong permutation index silently changes the sequence
# and every render with it. So this pins the identity against the plain
# remainder rather than against recorded values.

using Test
using Random
using Hikari
using Hikari: mod24

@testset "mod24 is x % 24, exactly" begin
    # Small values, where an off-by-one in the fold-in would show immediately.
    @test all(mod24(UInt64(x)) == UInt32(x % 24) for x in 0:2000)

    # Every power of two and its predecessor: these straddle the 32-bit split
    # (`2^32 ≡ 16 mod 24` is the whole trick) and are where a wrong carry lives.
    for k in 0:63
        p = UInt64(1) << k
        @test mod24(p) == UInt32(p % UInt64(24))
        @test mod24(p - UInt64(1)) == UInt32((p - UInt64(1)) % UInt64(24))
    end

    # The extremes, and a broad random sweep over the full 64-bit range.
    @test mod24(typemax(UInt64)) == UInt32(typemax(UInt64) % UInt64(24))
    @test mod24(UInt64(0)) == UInt32(0)
    rng = Random.MersenneTwister(0x5150)
    @test all(mod24(x) == UInt32(x % UInt64(24)) for x in rand(rng, UInt64, 100_000))

    # The actual argument shape at the call site: `mod24(hash_val >> 24)`, so
    # the input is a 40-bit value rather than a full 64-bit one.
    @test all(mod24(x >> 24) == UInt32((x >> 24) % UInt64(24))
              for x in rand(rng, UInt64, 20_000))

    # Result is always a valid 0-based permutation index; the caller adds 1 and
    # indexes a 24-entry table with it, so anything else is an out-of-bounds read.
    @test all(mod24(x) < UInt32(24) for x in rand(rng, UInt64, 20_000))
end

@testset "the first two Sobol dimensions are the matrix products" begin
    M = Hikari.SobolMatrices32
    rng = Random.MersenneTwister(0x50b0)
    indices = [rand(rng, UInt64) >> rand(rng, 12:63) for _ in 1:50_000]
    seeds = rand(rng, UInt32, length(indices))
    @test all(Hikari._sobol_float(Hikari._sobol_dim0(a), h) === Hikari.sobol_sample(Int64(a), Int32(0), h, M)
              for (a, h) in zip(indices, seeds))
    @test all(Hikari._sobol_float(Hikari._sobol_dim1(a, M), h) === Hikari.sobol_sample(Int64(a), Int32(1), h, M)
              for (a, h) in zip(indices, seeds))
end
