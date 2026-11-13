include("lavapipe.jl")

using Test
using Hikari
using GeometryBasics
using LinearAlgebra
using StaticArrays
using Raycore
# Mantle for everything with a device behind it: the backend, arrays, the graph.
# Lava for the files that compile Hikari's kernels to SPIR-V themselves.
using Lava, Mantle

if get(ENV, "HIKARI_TEST_LAVAPIPE", "false") == "true"
    @testset "Lavapipe_jll test device" begin
        @test isfile(Lavapipe_jll.lvp_icd)
        @test Lava.HAS_OBJECT_CACHE
        if Lava.HAS_OBJECT_CACHE
            @test Lava.ObjCache.enabled()
        end
        @test occursin("llvmpipe", lowercase(Mantle.devicename(Mantle.todevice(Mantle.defaultbackend()))))
    end
end

# The Vulkan backend module, for the test files that name backend internals.
#
# Hikari's own source does not, and must not — it goes through Mantle's portable
# API. Its TESTS do: `Mantle.LavaBackend`, `Mantle.vk_flush!` and friends, which
# since the 2026-08-27 split live in `MantleVulkanExt` rather than in Mantle. A
# `Main`-level binding serves every file, since they are all `include`d here.
#
# Phase 1.5 deleted this binding and did NOT migrate the nineteen files that
# name it, so the whole GPU half of this suite died on `UndefVarError: MVE` —
# 23 errors that read as "Hikari is broken". It is back, and it is DEBT: the
# names still reached for are `vk_context`, `LavaBackend`, `vk_flush!`,
# `live_buffer_count`, `drain_deferred_frees!`, `device_lost`, `gpu_live_bytes`,
# and each needs either a portable spelling or a deliberate decision that a
# backend-internals test belongs in the backend's own suite.
# NOT `using JET` here. Only `gpu_compat.jl` needs it, and it is a test-target
# dependency, so an environment without it made this line throw before the first
# testset and took all 24 files down with it. That file imports it itself; if JET
# is missing it fails and the rest of the suite still runs, which is the point of
# the wrapper below.

# One outer testset around every file.
#
# These were bare top-level `include`s, and each file opens its own top-level
# `@testset`. A top-level testset THROWS when it finishes with a failure, so the
# first bad file aborted the run and every later include was silently skipped —
# nothing in the output says "17 files did not run", it just looks like one
# failing testset. That is how `denoise.jl` calling an unbound `KA` (from
# 6ea9555, 2026-08-19) hid files 15 through 23 of this suite until 2026-08-24.
#
# Nested, they report instead of throwing, so one failure costs one file.
include("test_files.jl")

# ── pbrt reference suite ────────────────────────────────────────────────────
# Runs the full pbrt-v4 reference image comparison against every scene in
# test/pbrt/scenes for which a committed reference EXR exists.  Driven by
# `run_pbrt_suite(; backend, samples, hw_accel)` (defined in
# test/pbrt/test_pbrt_all_materials.jl).
#
# SW BVH always runs. HW RT runs additionally when the device builds hardware
# acceleration structures (`Mantle.supports_hwtlas`). Both are opt-out via env vars so CI can
# disable paths that are known to be broken on a given runner without
# editing this file.
include(joinpath(@__DIR__, "pbrt", "test_pbrt_all_materials.jl"))

const PBRT_SPP = parse(Int, get(ENV, "HIKARI_PBRT_SPP", "256"))
# A comma-separated subset of materials, textures and other, so CI can split the suite.
const PBRT_CATEGORIES = Tuple(Symbol.(split(get(ENV, "HIKARI_PBRT_CATEGORIES", "materials,textures,other"), ",")))

# Pkg.test("Hikari"; test_args = ["gpu_compat.jl", ...]) supports focused
# compiler/device checks. With no arguments every file and reference runs.
const REQUESTED_TESTS = Set(ARGS)
const UNKNOWN_TESTS = setdiff(REQUESTED_TESTS, Set([TEST_FILES; "pbrt"]))
isempty(UNKNOWN_TESTS) || error("Unknown Hikari test files: $(join(sort!(collect(UNKNOWN_TESTS)), ", "))")
const RUN_PBRT = isempty(REQUESTED_TESTS) || "pbrt" in REQUESTED_TESTS

# One testset over the unit files AND the reference suite. The reference suite
# has to be inside it: a top-level `@testset` throws when it ends with a
# failure, so anything after it in this file is skipped. With the suite outside,
# two unrelated unit errors were enough to silently skip the entire pbrt gate —
# the run went green-ish in 95 seconds and never rendered a scene.
@testset "Hikari" begin
    for fname in TEST_FILES
        isempty(REQUESTED_TESTS) || fname in REQUESTED_TESTS || continue
        @info "Hikari tests" file = fname
        @testset "$fname" begin
            include(fname)
        end
    end

    if RUN_PBRT && get(ENV, "HIKARI_SKIP_PBRT_SW", "false") != "true"
        run_pbrt_suite(; samples = PBRT_SPP, hw_accel = false, categories = PBRT_CATEGORIES)
    end

    if RUN_PBRT && get(ENV, "HIKARI_SKIP_PBRT_HW", "false") != "true" && hw_rt_available()
        run_pbrt_suite(; samples = PBRT_SPP, hw_accel = true, categories = PBRT_CATEGORIES)
    end
end
