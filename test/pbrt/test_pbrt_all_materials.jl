#
# pbrt reference suite — drives Hikari through every shipped .pbrt scene
# and compares against the committed pbrt-v4 EXR references.
#
# Defines `run_pbrt_suite(; backend, samples, hw_accel, ...)`; the bottom of
# this file auto-runs with sensible defaults when invoked as a standalone
# script (via `julia test_pbrt_all_materials.jl`) but NOT when included from
# runtests.jl (runtests picks the parameters and invokes the function
# explicitly).

using Test
using Hikari
using Lava, Mantle
include(joinpath(@__DIR__, "suite.jl"))

# ── Defaults ────────────────────────────────────────────────────────────────

const DEFAULT_SPP = 256

# `TILE_THRESHOLD`, `ENERGY_LOW` and `ENERGY_HIGH` come from `suite.jl`. They
# were declared here as well, which is how the gallery came to run a different
# band than this runner.

const BASE_MATERIALS = [
    "diffuse", "diffuse_colored",
    "conductor_gold", "conductor_mirror", "conductor_rough",
    "conductor_silver", "conductor_copper", "conductor_reflectance", "conductor_aniso",
    "dielectric", "dielectric_rough", "dielectric_rough_high", "dielectric_diamond", "dielectric_bk7",
    "thindielectric",
    "coateddiffuse", "coateddiffuse_rough", "coateddiffuse_aniso", "coateddiffuse_f11",
    "coatedconductor", "coatedconductor_rough", "coatedconductor_reflectance", "coatedconductor_bk7",
    "diffusetransmission",
]
const BASE_LIGHTS = ["point", "distant", "spot", "area", "ambient"]

# ── Test helpers (parametrized) ─────────────────────────────────────────────

function test_scene(scene_name; backend, samples, hw_accel)
    scene_file = joinpath(SCENES_DIR, "$(scene_name).pbrt")
    isfile(scene_file) || return nothing

    # The tolerance band is `suite.jl`'s, uniform for every scene class. There is
    # deliberately no way to widen it per scene: the threshold kwargs that used
    # to be here were an override channel no caller ever used, while the gallery
    # grew its own table that reached tile=0.55 for scenes checked at 0.07 here.
    ref = ensure_reference(scene_name)
    fb  = render_scene(scene_name; backend, samples, hw_accel)
    m   = compute_metrics(ref, fb)

    # Record before asserting, so a failing scene is still in the gallery — that
    # is the one you most want to look at.
    record_scene!(scene_name, fb, m; hw_accel, samples)

    println("  $(scene_name): tile=$(round(m.tile, digits=4)) energy=$(round(m.energy_ratio, digits=3))")
    @test m.tile         < TILE_THRESHOLD
    @test m.energy_ratio > ENERGY_LOW
    @test m.energy_ratio < ENERGY_HIGH
    return m
end

function test_scenes_matching(prefix; backend, samples, hw_accel)
    for name in list_scenes(prefix)
        @testset "$name" begin
            test_scene(name; backend, samples, hw_accel)
        end
    end
end

# ── Main entry point ────────────────────────────────────────────────────────

"""
    run_pbrt_suite(; backend=Mantle.defaultbackend(), samples=DEFAULT_SPP,
                     hw_accel=false, media_energy=(0.80, 1.20))

Run the full pbrt reference comparison under `@testset`s. All parameters are
explicit — callers (runtests.jl, standalone invocation) pick the backend and
hw_accel flag. Skips scenes whose `.pbrt` is missing.
"""
function run_pbrt_suite(; backend=Mantle.defaultbackend(),
                          samples::Int=DEFAULT_SPP,
                          hw_accel::Bool=false)
    label = hw_accel ? "HW RT" : "SW BVH"
    if samples != REFERENCE_SPP
        @warn """Rendering at $samples spp against references rendered at $REFERENCE_SPP spp.
                 The two sides now have different noise floors and the difference is
                 scored against Hikari. Results are indicative, not a pass/fail signal."""
    end
    begin_record!(hw_accel)
    @testset "pbrt reference ($label, $samples spp)" begin
        @testset "Materials × Lights" begin
            for mat in BASE_MATERIALS
                @testset "$mat" begin
                    for light in BASE_LIGHTS
                        scene = "mat_$(mat)_light_$(light)"
                        @testset "$light" begin
                            test_scene(scene; backend, samples, hw_accel)
                        end
                    end
                end
            end
            @testset "mix" begin
                test_scene("mat_mix_light_point"; backend, samples, hw_accel)
            end
        end

        # `cam_` and `integ_` were dead prefixes: scenes sat in scenes/ that no
        # testset matched, so cam_dof_light_point.pbrt had never once run.
        @testset "Camera"         test_scenes_matching("cam_";    backend, samples, hw_accel)
        @testset "Integrator"     test_scenes_matching("integ_";  backend, samples, hw_accel)
        @testset "Textures"       test_scenes_matching("tex_";    backend, samples, hw_accel)
        @testset "Cast shadows"   test_scenes_matching("shadow_"; backend, samples, hw_accel)
        @testset "Light variants" test_scenes_matching("light_";  backend, samples, hw_accel)
        @testset "Filters"        test_scenes_matching("filter_"; backend, samples, hw_accel)
        @testset "Sensors"        test_scenes_matching("sensor_"; backend, samples, hw_accel)
        @testset "Media"          test_scenes_matching("medium_"; backend, samples, hw_accel)

    end
end

# ── Standalone invocation ───────────────────────────────────────────────────
# When this file is the entry point (e.g. `julia test_pbrt_all_materials.jl`)
# run the suite in SW mode by default, and also in HW mode if the device
# supports it. When included from runtests.jl we do nothing here —
# runtests.jl calls `run_pbrt_suite` explicitly with its own parameters.

if abspath(PROGRAM_FILE) == @__FILE__
    spp = parse(Int, get(ENV, "HIKARI_PBRT_SPP", string(DEFAULT_SPP)))
    run_pbrt_suite(; samples=spp, hw_accel=false)
    if hw_rt_available()
        run_pbrt_suite(; samples=spp, hw_accel=true)
    end
end
