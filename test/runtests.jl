using Test
using Hikari
using GeometryBasics
using LinearAlgebra
using StaticArrays
using Raycore
using Lava
# NOT `using JET` here. Only `type_stability.jl` and `gpu_compat.jl` need it, and
# it is a test-target dependency, so an environment without it made this line
# throw before the first testset and took all 24 files down with it. The two
# files that use it import it themselves; if JET is missing they fail and the
# rest of the suite still runs, which is the point of the wrapper below.

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
const TEST_FILES = [
    "materials.jl",
    "type_stability.jl",
    "film.jl",
    "gpu_compat.jl",
    "volpath_integration.jl",
    "denoise.jl",
    "test_aux_alignment.jl",
    "test_scene_mesh_reuse.jl",
    "test_caching_gc_correctness.jl",
    "test_texture_wrap.jl",
    "test_blackbody_emitter_scale.jl",
    "test_ray_differentials.jl",
    "test_checkerboard_texture.jl",
    "test_const_texture_value.jl",
    "test_workqueue.jl",
    "test_hw_sw_parity.jl",
    "test_update_material_null.jl",
    "test_null_material_only_scene.jl",
    "test_multitypeset_updates.jl",
    "test_material_type_collapse.jl",
    "test_bxdf_dispatch.jl",
    "test_mantle_device.jl",
    "test_precompile_statements.jl",
    "test_pbrt_camera_lookat_distance.jl",
    "test_sphere_uv.jl",
    "test_volpath_graph.jl",
    "test_volpath_per_iter_lifecycle.jl",
]

# ── pbrt reference suite ────────────────────────────────────────────────────
# Runs the full pbrt-v4 reference image comparison against every scene in
# test/pbrt/scenes for which a committed reference EXR exists.  Driven by
# `run_pbrt_suite(; backend, samples, hw_accel)` (defined in
# test/pbrt/test_pbrt_all_materials.jl).
#
# SW BVH always runs. HW RT runs additionally when the bound Vulkan device
# reports `rt_pipeline_properties`. Both are opt-out via env vars so CI can
# disable paths that are known to be broken on a given runner without
# editing this file.
include(joinpath(@__DIR__, "pbrt", "test_pbrt_all_materials.jl"))

const PBRT_SPP = parse(Int, get(ENV, "HIKARI_PBRT_SPP", "256"))

# One testset over the unit files AND the reference suite. The reference suite
# has to be inside it: a top-level `@testset` throws when it ends with a
# failure, so anything after it in this file is skipped. With the suite outside,
# two unrelated unit errors were enough to silently skip the entire pbrt gate —
# the run went green-ish in 95 seconds and never rendered a scene.
@testset "Hikari" begin
    for fname in TEST_FILES
        @testset "$fname" begin
            include(fname)
        end
    end

    if get(ENV, "HIKARI_SKIP_PBRT_SW", "false") != "true"
        run_pbrt_suite(; samples=PBRT_SPP, hw_accel=false)
    end

    if get(ENV, "HIKARI_SKIP_PBRT_HW", "false") != "true" && hw_rt_available()
        run_pbrt_suite(; samples=PBRT_SPP, hw_accel=true)
    end
end
