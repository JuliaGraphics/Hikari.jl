# Profiling Infra + Orthogonal MWE Sweep — Plan

Status: planning. Author: 2026-06-03.

## Goals

1. Understand precisely where Lava trails OptiX on `killeroo_gold`
   (2.84×) — material divergence? Fresnel spectrum tables? RT
   traversal? atomic contention?
2. Sharpen the per-material-kernel decision (`per-material-kernel-plan.md`)
   with **data**, not theory. The "materials wins despite 20 material
   types" data point already half-refuted my own reasoning; need real
   measurements before committing to that or any other refactor.
3. Build durable profiling infra in Lava — this is useful far beyond
   the current investigation.

## What we have today

- `Lava.disassemble_spirv` / `Lava.disassemble` — text SPIR-V per kernel
- `Lava.dump_spirv_to_disk` + `Lava.SPIRV_DUMP_DIR` — auto-dump pipeline SPIR-V
- `Lava.DISPATCH_LOG` + `set_dispatch_logging!` — per-dispatch *names* (no timing)
- `Lava.MEMORY_STATS` — VRAM accounting
- `KA.synchronize(::LavaBackend)` — coarse host-side barrier we can wrap
- `pbrt --stats` "Total rendering time" — pbrt render-only baseline
  (already wired in RayDemo `5b6e464`)
- `RayDemo` `assert_pbrt_match` drift guard (already wired)

## What we're missing

| capability                          | who builds it          | difficulty |
| ----------------------------------- | ---------------------- | ---------- |
| Per-kernel GPU time                 | new in Lava            | medium     |
| Driver register count per kernel    | new in Lava            | medium     |
| SPIR-V instruction histogram        | Lava (parses bytes)    | easy       |
| Orthogonal MWE scene scaffolding    | RayDemo + Hikari       | medium     |
| pbrt-side scene generation from same canonical config | RayDemo | easy   |

## Phase 0 — what we can do right now without any code

Even before the new infra lands, run a Lava render with
`set_dispatch_logging!(true)`, grab the resulting dispatch log, and
attribute SPIR-V *size* per kernel name. That's not timing, but it
gives us a rough sense of where the SPIR-V volume is — useful as a
sanity check on the per-material-kernel theory ("the shading kernel
SPIR-V is enormous because every material is inlined").

Concrete output:
```
killeroo_gold render, per-kernel SPIR-V volume:
  vp_trace_and_shade_kernel:   N1 KB, 1 dispatch / sample
  vp_handle_escaped_rays:      N2 KB, 1 dispatch / sample / bounce
  vp_trace_shadow_rays_kernel: N3 KB, 1 dispatch / sample / bounce
  vp_shade_surface_hits_kernel: N4 KB (likely empty on killeroo)
  vp_accumulate_to_rgb_kernel: N5 KB, 1 dispatch / sample
```

This is 30 minutes of work and is a useful smoke test for Phase 1.

## Phase 1 — coarse per-phase host-side timing (no Lava changes)

Modify `Hikari/src/integrators/volpath/volpath.jl::render!` to
**optionally** call `KA.synchronize(backend)` + `@elapsed` between
the render-loop phases when `vp.profile === true`, recording the
timing into a `vp.profile_log::Vector{NamedTuple}`. The default flag
is `false` (no overhead in production).

Phases to bucket:
- camera-ray gen
- per-bounce: trace_and_shade
- per-bounce: medium_interaction (skip if no media)
- per-bounce: medium_direct_lighting + medium_scatter
- per-bounce: handle_escaped_rays
- per-bounce: shade_surface_hits (medium-exit path)
- per-bounce: trace_shadow_rays
- final accumulate_to_rgb + finalize_film

This is coarse — it forces `KA.synchronize` between phases so it
*does* perturb the measurement — but it's enough to tell us whether
50 % of killeroo's time is in `trace_and_shade` or in
`trace_shadow_rays` or somewhere else.

If the data clearly points at one phase, we may not need Phase 2.

## Phase 2 — Vulkan timestamp queries (per-dispatch GPU time)

Add `VkQueryPool` of `TIMESTAMP` type to Lava. Wrap each
`vk_dispatch!` with `vkCmdWriteTimestamp` before/after. After
`vk_flush!`, read back the query pool and report per-dispatch GPU ns.
Aggregate by kernel name for a final report.

```julia
Lava.with_timestamp_queries() do
    render!(vp, scene, film, camera)
end
report = Lava.get_timestamp_report()
# report :: Vector{(; kernel_name, dispatch_idx, gpu_ns)}
```

Implementation notes:
- Pool sized to MAX_DISPATCH_LOG (2000) timestamps
- One pool per `LavaBackend`; reset between captures
- Disabled by default (timestamp writes are cheap but non-zero)
- Compatible with `set_dispatch_logging!` for context: dispatch log
  + timestamps merged into one record

This actually attributes GPU time to specific kernels — no
host-side perturbation. Pairs with Phase 1 to disambiguate "this
kernel is slow on GPU" vs "this kernel issues many dispatches".

## Phase 3 — `VK_KHR_pipeline_executable_properties`

NVIDIA's driver implements this extension. After pipeline creation,
calling `vkGetPipelineExecutablePropertiesKHR` returns:

- `VkPipelineExecutableStatisticsKHR` (per-statistic):
  - "Number of registers used"
  - "Scratch space used"
  - "Spill stores / loads"
  - etc. — vendor-specific list
- `VkPipelineExecutableInternalRepresentationsKHR`:
  - The actual NVIDIA SASS disassembly (driver permitting)

Add to Lava:
- Enable the extension at device creation (gate behind a config flag
  to keep production runs lean)
- After pipeline link, query stats; attach to `LavaComputePipeline`
- `pipeline_stats(kernel_fn, args...) :: NamedTuple`

This gives us:
- **Register count per kernel**, the smoking gun for the
  "monolithic kernel = too many materials = low occupancy" theory
- Scratch / spill numbers, the second smoking gun for the same theory
- Optional NVIDIA SASS for surgical analysis

Output we want:
```
Pipeline stats for vp_trace_and_shade_kernel! on killeroo_gold:
  registers per thread: 96
  scratch bytes:        128
  spirv bytes:          18432

Pipeline stats for vp_handle_escaped_rays_kernel! on killeroo_gold:
  registers per thread: 32
  scratch bytes:        0
  spirv bytes:           2048
```

If the trace-and-shade kernel reports >80 registers/thread on
killeroo, the per-material hypothesis has direct evidence. If it
reports 40, the hypothesis is wrong and we should pivot.

## Phase 4 — orthogonal MWE scene scaffolding

A single canonical config drives both Hikari and pbrt-v4:

```julia
struct MWEConfig
    name              :: String
    resolution        :: Tuple{Int,Int}
    samples           :: Int
    max_depth         :: Int
    materials         :: Symbol   # :gold, :diffuse, :gold_8mat_grid, ...
    geometry          :: Symbol   # :killeroo, :single_sphere
    light_setup       :: Symbol   # :point_top, :env_only, :area_above, ...
end
```

For each MWEConfig, two outputs are generated:
1. Hikari `create_scene(; config)` that builds the matching scene
2. `<name>.pbrt` written to a tmp dir with matching settings

Both flow through the existing `RayDemo/benchmark/run_benchmarks.jl`
machinery — `assert_pbrt_match` is taught to recognise the generated
configs. Result JSONs are written to
`RayDemo/benchmark/results/orthogonal/`.

The point: **never again ship a "Lava vs pbrt" timing where the two
sides aren't running the same workload** (the lesson from the
mismatched-config benchmarks we already fixed).

## Phase 5 — the actual sweep

Baseline:
- `killeroo_gold` (existing)

Variants (each varies ONE axis from `killeroo_gold`):

| name                   | varies                                | tests                        |
| ---------------------- | ------------------------------------- | ---------------------------- |
| `killeroo_diffuse`     | BSDF: Au tables → const-RGB Diffuse    | Fresnel spectrum-table reads |
| `killeroo_8mats`       | 1 conductor → 8 conductor variants tiled across surfaces | material divergence / register pressure |
| `killeroo_depth1`      | maxdepth: 5 → 1                       | primary cost vs indirect cost |
| `killeroo_depth16`     | maxdepth: 5 → 16                      | indirect bounce scaling      |
| `killeroo_lowres`      | res: 1368² → 684²                     | atomics + per-pixel state cost |
| `killeroo_no_light`    | direct lighting off (env only)        | shadow + light sampling cost |
| `single_sphere_gold`   | geometry: 0.5M tris → 1 sphere        | BVH cost vs shading cost     |
| `single_sphere_diffuse`| above + diffuse                       | pure per-ray fixed cost      |

For each scene:
1. Run Lava HW with Phase 1+2+3 profiling enabled.
2. Run pbrt-v4 OptiX with `--stats`.
3. Record per-kernel GPU ns, register count, total render time on
   both sides.

## Phase 6 — analysis

Build one canonical table. Each row is a scene; columns are
Lava-ms, OptiX-ms, ratio, plus the dominant Lava per-kernel split
(top-3 kernels by GPU ns) and the trace-and-shade register count.

What the data tells us, mechanically:

- `killeroo_diffuse` ≪ `killeroo_gold` → spectrum-table reads are
  the bottleneck → next optimisation is **sampled-image textures**
- `killeroo_8mats` ≫ `killeroo_gold` → divergence/register pressure
  is the bottleneck → **per-material kernel split is the right call**
- `killeroo_depth1` ratio ≈ `killeroo_gold` ratio → indirect scaling
  is fine; the gap is per-ray fixed cost
- `killeroo_lowres` ratio ≈ `killeroo_gold` ratio → atomics aren't
  the issue
- `single_sphere_gold` ratio ≈ `killeroo_gold` ratio → BVH cost is
  not where we trail OptiX
- Register count > 80 on trace-and-shade → kernel size matters →
  per-material split worth doing

The data either points at one optimisation or rules it out. Either
way we make a real decision instead of guessing.

## Work breakdown (estimated)

- Phase 0: 30 min — runnable today, sanity check only
- Phase 1: half day — Hikari profile_log + a few `KA.synchronize`s
- Phase 2: 1–2 days — Vulkan timestamp queries plumbing in Lava
- Phase 3: 1–2 days — `VK_KHR_pipeline_executable_properties`
- Phase 4: half day — MWEConfig + Hikari/pbrt generator
- Phase 5: half day — runs (mostly waiting)
- Phase 6: half day — table + decision

Total: ~5–6 days. Suggested order: 0 → 1 → 4 → 5 (partial) → 6
(partial decision) → 2 → 3 → 5 (full re-run) → 6 (final). That lets
Phase 0/1 inform Phase 2/3 — if coarse timing already nails it down,
we may not need driver-side stats.

## Risks

- `VK_KHR_pipeline_executable_properties` may not return useful
  statistics on NVIDIA (mitigation: check at extension probe time;
  fall back to spirv-size + dispatch counts).
- Vulkan timestamp queries can perturb runtime on tight inner loops
  (mitigation: opt-in flag; document the perturbation).
- Synthesised .pbrt files may not produce sane pbrt output if the
  matrix syntax drifts from pbrt-v4 expectations (mitigation:
  diff-test the generator against existing `.pbrt` files in
  `RayDemo/`; fail loudly on parse error).
- Lava's pipeline cache caches by SPIR-V hash; a hot reload may not
  re-trigger pipeline stats query (mitigation: cache the stats next
  to the pipeline, not next to the kernel function).

## What this is NOT

- Not a refactor of the integrator.
- Not committing to per-material kernel split until the data says so.
- Not a competition with pbrt-v4 OptiX on synthesised scenes — those
  exist to attribute Lava's own time, not to declare a winner.

## Decision log

To be updated as we go.
