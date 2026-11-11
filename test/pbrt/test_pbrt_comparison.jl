#
# pbrt vs Hikari — hand-curated scene comparison.
#
# A smaller, tighter-tolerance subset of the scenes in `scenes/`. Compared to
# test_pbrt_all_materials.jl this uses stricter per-pixel thresholds (mean/
# median diff) in addition to the tile + energy checks.  Useful for catching
# algorithmic regressions that are diluted by the broader suite.

using Test
using Hikari
using Lava, Mantle
include(joinpath(@__DIR__, "suite.jl"))

# Force lavapipe for deterministic software rendering. Callers that want the
# real device should override `LAVA_ICD` before `using Lava`.
get!(ENV, "VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json")

# ── One-shot comparison helper ──────────────────────────────────────────────

function compare_scene(scene_name; backend, samples, hw_accel,
                       ref_spp=1024, max_depth=10,
                       max_mean_diff=0.06,
                       min_energy_ratio=0.95, max_energy_ratio=1.05)
    ref = ensure_reference(scene_name; spp=ref_spp)
    fb  = render_scene(scene_name; backend, samples, max_depth, hw_accel)
    m   = compute_metrics(ref, fb)

    println("  Scene: $scene_name")
    println("    Mean diff:    $(round(m.mean_diff, digits=4))")
    println("    Energy ratio: $(round(m.energy_ratio, digits=4))")
    @test m.energy_ratio > min_energy_ratio
    @test m.energy_ratio < max_energy_ratio
    @test m.mean_diff    < max_mean_diff
    @test m.median_diff  < max_mean_diff
    return m
end

# ── Parametrized suite ──────────────────────────────────────────────────────

function run_pbrt_comparison(; backend=Mantle.defaultbackend(),
                               samples::Int=256, hw_accel::Bool=false)
    label = hw_accel ? "HW RT" : "SW BVH"
    @testset "PBRT vs Hikari ($label, $samples spp)" begin
        for (name, depth) in (
                ("shared_geom_diffuse",   5),
                ("single_sphere_point",   5),
                ("single_sphere_distant", 5),
                ("single_sphere_area",    5),
                ("all_materials",        10),
                ("named_materials",       5),
                ("transforms",            5))
            @testset "$name" begin
                compare_scene(name; backend, samples, hw_accel,
                              max_depth=depth,
                              max_mean_diff=(name == "all_materials" ? 0.08 : 0.06))
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_pbrt_comparison(; hw_accel=false)
    if hw_rt_available()
        run_pbrt_comparison(; hw_accel=true)
    end
end
