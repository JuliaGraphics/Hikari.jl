# Material implementation comparison

## Current Bosgame investigation, 2026-11-05

After releasing all Desktop Julia/GPU workers, measurements continue only on
Bosgame's Radeon 8060S. The canonical checkout now retains local-array storage
for large BSDF programs and packs constant Principled inputs into the existing
instruction payload. Computed inputs retain their gathered indices, including
independent emission colour and strength. These changes use the same production
renderer and material API; there is no optional renderer to select.

Paired 512²/64-sample measurements, alternating order over 31 repetitions:

| Principled graph | Previous input decoding | Packed inputs | Change |
| --- | ---: | ---: | ---: |
| Mix of matte and metallic Principled | 73.384 ms | 65.927 ms | -10.2% |
| Mix of layered and transmissive Principled | 145.927 ms | 134.751 ms | -7.7% |

Both comparisons retain local-array storage and produce identical pixels.
The final records are `out/material_bosgame_resume_20261105/final_packed_inputs.json`;
`production_packed_inputs.json` retains the preceding repeat.
These Radeon measurements cannot establish the RTX/Cycles target. First-frame
times in this paired experiment include unequal Julia warmup and must not be
reported as an isolated compilation improvement.

The storage change initially broke subsurface GPU shading: branches merged
separately allocated arrays, which Julia's GPU allocation pass could not promote
to local storage. Reusing the hit-owned array through walk outcomes fixes the
48-assertion GPU reduction and the subsurface scene with software and hardware
ray tracing (`r534`). The subsequent nine-file acceptance run passes all 2,481
assertions (`r535`), including all Cycles references, furnace conservation,
shader sharing and material updates. Direct GPU comparisons of every packed
Principled field and subsurface-selection checks pass another 1,218 assertions
(`r545`).

The same run compares production against typed Principled rendering, with the
same scene and alternating 31-frame measurements:

| Principled settings | Typed | Production | Change |
| --- | ---: | ---: | ---: |
| Matte | 48.395 ms | 47.741 ms | -1.4% |
| Rough transmission | 95.104 ms | 86.916 ms | -8.6% |
| Rough metal | 49.841 ms | 50.478 ms | +1.3% |
| Plastic | 50.679 ms | 49.674 ms | -2.0% |
| Metal, coat and sheen | 55.971 ms | 56.077 ms | +0.2% |
| Mix: matte and metal | 89.936 ms | 65.502 ms | -27.2% |
| Mix: layered and transmission | 180.034 ms | 135.520 ms | -24.7% |

Single-node rows use the point light; Mix rows also use the uniform environment
from `mixed_graph_cases.json`. Six rows have exact typed/production pixels. The
layered/transmission Mix has maximum absolute error 0.001801 and passes the
unchanged image comparison; the packed/unpacked comparison itself remains exact.
These are typed Hikari comparisons, not original pbrt or Cycles timings.

After a clean source rebuild and another Julia restart, all eight fixtures
(five single-node, the environment-lit matte control and two Mixes) have zero
SPIR-V misses and zero native pipeline-cache refusals (`r555`/`r556`). Package
build IDs, complete source hashes and image hashes agree across processes.
The first matte frame still takes 21.68 seconds, including 19.56 seconds of
Julia compilation; shader-cache reuse does not eliminate host startup work.

Two investigated Cycles techniques did not give a useful Radeon improvement.
Preparing F82 coefficients during setup changed time by less than 1%. Reusing
the sampled term's Fresnel/evaluation result instead of evaluating it again was
about three times slower with tuple storage: native VGPR spills rose from zero
to 266 and scratch grew from 55,296 to 144,640 bytes per wave. Keeping arrays
removes those spills, but the control still gives no useful speedup. Neither
technique is enabled in production. Full sampled-half-vector reuse also exposed
low-roughness cancellation in the existing GGX formula, so the comparison
retains the original evaluation normal rather than loosening its checks.

`ggx_cartesian_probe.jl` isolates the Cartesian GGX equations used in Cycles'
anisotropic evaluator. Against independent Float64 slope equations, 1,442 checks
pass: maximum relative D error is 4.29e-7, versus 3.0 in the old expression near
normal incidence at low roughness. Nevertheless, two normal-settings runs are
2–3% slower. A run with executable-statistics capture reverses that result;
its native code removes 407 instructions but keeps 240 VGPRs and 55,296 bytes
of scratch per wave. This is not a demonstrated production speedup and remains
a probe. The raw normal, captured and normal-repeat results are retained.

The investigation also fixed `pipeline_exec_stats` labelling spilled-register
counts as scratch bytes. Six unit checks and a native-pipeline check pass
(`r553`); older records' raw `Scratch size` statistic is authoritative.

The sections below preserve earlier RTX/Mac measurements and controls. Their
results predate these source changes unless explicitly identified otherwise.

## Four-implementation benchmark

Rows identify material settings. Columns identify four implementations:

* Blender 5.2.1 Cycles Principled, OptiX.
* Historical Hikari pbrt materials at `d24058b103fb2c47e7709e7fd91c75aea4851a0b`,
  before the Cycles merge, running on today's dependency stack and ray queries.
  This does **not** reproduce the former RT-pipeline renderer.
* Hikari Cycles Principled with the retained typed BSDF implementation.
* Hikari Cycles Principled with the current common BSDF storage and bounded dispatches.

`cases.json` supplies the geometry, camera, light, sampling, and material settings.
The Blender mesh is exported from the Hikari fixture. Timings cover a complete
render, including host submission, but exclude image readback and file writes.
Hikari explicitly waits for GPU completion. The original four-implementation
runs select the RTX 4000 Ada.
Five warmups precede 31 measurements; the two Hikari Cycles implementations
alternate order. Individual measurements and medians are saved.

The Cycles variants use the default multiple-scattering GGX. Original pbrt
uses single-scattering GGX; its alpha is set explicitly to Principled roughness
squared. The conductor's normal-incidence reflectance matches Principled's
base colour. Plastic compares pbrt's coated diffuse random walk with Principled's
dielectric-over-diffuse model. These are model comparisons, not identical BSDFs.
The mixed metal/coat/sheen Principled has no equivalent original model and gets
no original-pbrt timing. Images from the two Hikari Cycles implementations must
agree; images from different models are retained for inspection.

First-render and setup measurements use existing caches in the active process;
they must not be described as isolated cold compilation times. SPIR-V, LLVM,
and function-cache counts are recorded alongside them.

Only the canonical `dev/Hikari` checkout is used. A historical run temporarily
checks out the pinned commit there and restores the working branch afterwards.
The compiler and dependency checkout versions stay fixed, so the historical
column measures old Hikari on the current dependency stack.
Its compatibility change disables the former RT-pipeline/closest-hit path.
Consequently this column cannot establish the regression against the original
fast renderer, and the original runtime performance target remains unverified.

From a Julia session in the workspace environment, with a working display:

```julia
using Hikari, Mantle
include(joinpath(pkgdir(Hikari), "benchmarks", "material_comparison", "hikari.jl"))
MaterialComparison.run("out/material_comparison/main")
# In the pinned original checkout, with a fresh Julia session:
include(joinpath(pkgdir(Hikari), "benchmarks", "material_comparison", "original_compat.jl"))
MaterialComparison.run("out/material_comparison/main"; original = true)
```

After the Hikari timings finish, run Blender with `cycles.py`, followed by `--`,
the `cases.json` path, and the same output directory. GPU measurements from
different renderers must run sequentially.

The original checkout needs the supplied `original-host-compat.patch`: it drops
an obsolete SBT keyword from mesh registration. `original_compat.jl` selects
ray queries through the old host capability query. Neither changes material
or integrator kernels. Restore that patch before returning to the current branch.

## Latest completed runtime measurements

All five measured medians meet the within-ten-percent Cycles target on the
RTX 4000 Ada. These are complete 512²/64-sample renders of the matched benchmark
scene, depth four, five warmups and 61 measurements. First, warmed and
final images pass the same differential comparisons. Every row uses public
Principled settings and the one production renderer. The same source also
passes at 1024² with 31 measurements. These runs include the upstream dependency updates, early setup inlining,
stable memory-budget wave sizing, progressive accumulator ownership and 32-invocation shading described below.

| Settings | Cycles OptiX | Current Hikari | Maximum target |
| --- | ---: | ---: | ---: |
| Matte | 44.282 ms | 45.401 ms | 48.710 ms |
| Rough glass | 89.906 ms | 75.869 ms | 98.896 ms |
| Rough metal | 46.379 ms | 47.593 ms | 51.017 ms |
| Plastic | 48.157 ms | 47.151 ms | 52.973 ms |
| Mixed metal/coat/sheen | 53.664 ms | 52.919 ms | 59.030 ms |

| Settings at 1024² | Cycles OptiX | Current Hikari | Difference |
| --- | ---: | ---: | ---: |
| Matte | 176.888 ms | 184.077 ms | +4.1% |
| Rough glass | 350.929 ms | 307.208 ms | -12.5% |
| Rough metal | 185.766 ms | 193.469 ms | +4.1% |
| Plastic | 192.600 ms | 192.748 ms | +0.1% |
| Mixed metal/coat/sheen | 214.721 ms | 216.198 ms | +0.7% |

Raw results, images and source patches are in
`target_10_percent/byte_range_readonly_*`. All 20 image hashes match the preceding
inline-environment runs; the target checks are recorded in
`byte_range_readonly_acceptance.json`. The preceding raw images remain in
`production_inline_escape_*`. The earlier source patches
in `final_implicit_index/` predate these changes. This demonstrates the target
for these fixtures, not every possible Cycles graph or scene.
The preceding 101-frame repeat, before removing the primary index store, is
retained in `final_quiet_confirmation/`; metal there missed by 0.011 ms.
The isolated index control saves 0.68/0.47 ms on RTX matte/metal with exact
pixels and unchanged Radeon timings. The camera kernel's primary slot equals
its pixel/sample index, so shading reconstructs it without a store or load.

A post-upstream repeat (`post_upstream_512/`, 31 measurements) gives
45.362 / 76.509 / 47.716 / 47.466 / 53.099 ms for the same five settings,
still within all five Cycles limits. It includes CompilerCaching 0.5.3 and
the current KernelAbstractions/GPUCompiler sources, with the local compiled
method-version diagnostic fix retained. These first-render times use existing
cache contents; they are not isolated cold-start measurements.

The final inactive-specular guard saves work only at exactly zero remaining
weight and preserves the original eta/film predicate first. Both storage
classes pass 37,434 CPU differential assertions, including a zero-weight test
that supplies no albedo table. Camera-path reconstruction passes 249 checks
on RTX/Radeon/Lavapipe; Radeon also passes emitting/curved-medium transport.
Mixed/singleton in-place edits and shader reuse passed on all three devices.
The final setup guard passed another 27,417 Lavapipe assertions. The subsequent
index change passed all 249 camera checks on each device and 27 medium checks
on Radeon. No commits, pushes or PRs were made.

Earlier unchanged-source restart acceptance is in `stable_source_cache_restart.json`:
all five fixtures have zero SPIR-V misses during both construction and first
render, and exact images on repeat. The first matte construction takes 2.46 s
and its first render 38.37 s, of which Julia reports 31.46 s compiling host
code. Later first renders take 0.091–0.171 s. Package loading is excluded.
This confirms shader reuse, not fast host startup. The earlier failed restart
probe followed a package rebuild with a different Hikari build ID; that
invalidated its inferred-code keys. Both stable-source processes record the
same package identities. The failed logs remain alongside the passing ones.

Earlier compilation figures predate these final changes. They must not be
interpreted as current startup measurements.

## Extended acceptance, 2026-11-05

The initial five-setting run at 1024² exposed a remaining gap: current/Cycles times
are 195.29/176.89 ms matte, 325.52/350.93 ms glass, 207.42/185.77 ms metal,
206.79/192.60 ms plastic and 232.39/214.72 ms layered. Matte and metal miss
by 10.4% and 11.7%. Raw results are in `resolution_1024/`. Shading groups
32/64/128 do not materially improve this; `shade_groups_1024.json` retains
all measurements. The later memory-budget wave sizing closes this gap:
budgets of at least 12 GiB permit four million paths rather than two million;
smaller and unknown budgets retain the conservative bound. Surface shadow
rays are traced inline, so the unused medium-only shadow queue now follows
the medium capacity. At 1024², the larger-wave state uses 2.31 GB rather than
1.17 GB; matte/metal controls improve 196.66→185.36 and 207.42→196.98 ms
with exact images. No vendor selector is used. Outlined host argument packers also failed to improve
startup (41.34 s, 34.25 s Julia compilation); this was not adopted.

The extended software transport suite passes 125 assertions. The Radeon suite
initially passed 1,132 and failed 34, including subsurface furnace and Cycles
references. A 32² furnace reduction gives approximately 0.035 with counted
packed storage, 1.001 with typed storage and 0.999 on Lavapipe. Both fused
and queued camera paths fail, so fusion is excluded. Packing, selection,
walk parameters and isolated exit evaluation match CPU values on the GPU.
Removing function-object linking, outlining the walk, and using full padded
tuples do not fix the failure. Inlining `surface_setup_slots` at the Julia
level restores 1.00073 and exact fused/queued images. Its late LLVM inlining
had retained an aggregate return copy; the helper does not survive as a
separately reusable SPIR-V function on that path anyway. Current source
applies early inlining. The full Radeon furnace/Cycles run improves to 331
passing assertions and four failures, confined to a zero-radius subsurface
channel and a subsurface/diffuse mix. Lavapipe passes the full furnace and
28,463 focused storage/direct/render assertions; RTX passes the isolated
hardware-traced furnace with mean 1.00087 and exact fused/queued images.
At this point four Radeon assertions remained; the later workgroup control
below resolves them. The diffuse-only control agrees exactly with typed
storage when the random walk is removed.

The smaller GPU reduction shows that retaining and evaluating the returned
walk state changes survival from 255/256 to 156/256 before the change; early
inlining restores 255/256 with valid evaluation. The compiler reduction also
exposed a constant table-subobject/null comparison that Lava rejected.
The emitter now handles inbounds subobjects of defined private constants;
17 compiler checks include both operand orders and refusal of extern-weak
and unbounded addresses. No vendor-specific code or tests were introduced.
All artifacts and logs remain local under `target_10_percent/`.

That investigation also exposed an independent private-pointer bounds bug:
an LLVM one-element view over a four-element allocation was clamped to one
element, changing executable results from [1, 2, 3, 9] to [1, 1, 1, 9].
Bounds now use the complete byte offset, allocation size and access width;
speculative loads do not strengthen alignment. Ninety focused checks pass,
including executable vector/narrow-index/multi-index/wide-access reductions.
This fix alone does not resolve the remaining Radeon subsurface failures.

Allocation controls isolate those failures to flattening the mutable setup
buffer from 54 i64 words to 108 i32 lanes. Keeping just that allocation in its
original form fixes both cases; changing the immutable tuple copy does not.
Neither volatile accesses, extra SPIR-V SSA cleanup, nor selecting integer
rather than floating lane types fixes them. Paired vector lanes and refusing
heterogeneous retyping both fix the two cases but fail the full pure-subsurface
furnace, so neither change is retained. The original allocation policy remains.
Forcing 32-lane subgroups fixes both cases without changing the shader's
storage representation. The first complete control passes 335 assertions,
but clearing only pipelines left linked kernels and launch plans reusable.
The corrected matrix clears all kernel caches between controls. Default-width
64-invocation groups fail; default-width 32-invocation groups and both forced
32-lane variants pass, including the pure-subsurface furnace. The production
shade group is now 32 on every backend, with no subgroup requirement or vendor
branch. Production Radeon acceptance passes all 418 furnace, Cycles-reference,
in-place update and runtime-parity assertions. The failing shade module contains no subgroup collectives, and the
native IR/register-allocation validator reports no error. These observations
narrow the issue but do not yet establish a general compiler fix.

The complete current Lava suite passes 1,619 assertions with two existing
broken tests. Its source-boundary inventory now includes the previously
omitted inferred-cache and private-argument passes. Two pointer-clamp tests
check the new byte-offset comparisons rather than the previous index spelling.

The final upstream integration also passes 1,630 Lavapipe assertions, including
primary-camera equivalence, capacity routing, furnace, runtime BSDF parity,
material edits and stable wave sizing. Radeon passes all 321 Cycles-reference
assertions, 14 furnace assertions, runtime parity and material-type updates;
the updated wave tests pass separately after reloading their source. RayMakie
passes 131 nested material/image update checks. The latest compiler run passes
1,619 Lava assertions (two existing broken), 1,295 CompilerCaching body assertions
and seven method-table checks. KernelAbstractions passes CPU launch geometry
and constructor recovery on both CPU and the RTX.

`wave_budget_stability_probe.jl` supplies a small controlled path budget. With
the previous resampling policy, 4096 -> 2048 -> 4096 changes lanes 16 -> 8 -> 16
and replaces the plans twice. The production integrator retains its original limit,
lanes and plans across all three budgets. All six images match. The integrator's
stored bound still allows wave sizes to follow a caller's sample count.

`stable_wave_cold_seed.json` and `stable_wave_fresh_verify.json` record two fresh
processes using an isolated object cache. The verification has zero shader misses
and zero native pipeline compilations for construction and first draw of all five
materials, with exact repeat images. Its first matte draw takes 46.85 s, including
38.86 s of Julia compilation, while software validation was running concurrently.
This is cache correctness evidence, not a quiet startup benchmark. Host startup
remains substantially slower than warmed rendering. The earlier 42.15 s native
cache proof is retained in `isolated_native_cache_restart.json`.

The Makie upstream merge preserves deferred render-object rebuilding alongside
the new stroke toggle. Seventeen raster checks include simultaneous stroke and
colour-representation changes. Scene teardown now disconnects callbacks owned
by its whole subtree before deleting plots; a failing callback reproducer and
all 73 scene tests pass, while external plot-deletion notifications remain.
This avoids legend recomputation during teardown after font resources expire.

A further progressive-render check exposed loss of history when changing a
sample batch (1 + 7 versus eight single samples, maximum RGB error 0.0996).
The two per-pixel accumulators now have their own memory owner, transferred
when path queues resize for the same film. No device copy or duplicate
accumulator allocation is needed. The reproducer now agrees bit-for-bit;
RTX passes 206 progressive/lifetime/plan checks, Radeon and Lavapipe each
pass the corresponding 205 checks. A fresh-source RayMakie run also passes all
131 nested material/image-update checks after this change. Both accumulator identities and final
release are asserted. The steady-state sample test still requires zero host
allocations; keeping the selected path budget on the concrete integrator
avoids boxing it through the backend-parametric state.

The final unchanged-source cache verification (`progressive_owner_fresh_verify.json`)
passes all five fixtures with zero shader and native-pipeline misses, with exact
repeat images; its first draw takes 41.706 s, including 34.881 s Julia compilation.

A quiet cached-startup control measures 41.826 s first draw (35.018 s Julia
compilation), with zero shader/native-pipeline misses and exact output.
A small device-free access-analysis primer costs 2.745 s before setup and
reduces draw time to 39.900 s, so it does not improve total startup by itself.
That small workload is not adopted. Its own compiled code survives package
loading, but it does not cover the shader-specific inference that dominates startup. The earlier sampled
profile filled its buffer; it is retained as partial diagnostic data and is
excluded from timing comparisons. The corrected profiler is optional and
uses a larger buffer and lower sampling rate.

The adopted workload describes all four common opaque Principled shading
signatures before and after readonly adaptation for hardware and software triangle traversal without constructing a
device, arrays or a scene. Mantle supplies the device-free ABI and interpreter;
Hikari contains no compiler-backend names. It runs the actual GPU inference/access walk during
package precompilation. Regression tests compare those descriptions with real
recorded dispatches, and graph edits still reuse their shading kernels.

The final quiet fresh-process verification takes 23.922 s (19.424 s Julia
compilation), versus the preceding 41.706 s (34.881 s compilation), with all
five cases reporting zero shader misses and zero native pipeline compilations.
Scene setup and package loading are excluded from those clocks. Package
precompilation grows from about 35 s to about 67 s in the measured builds;
a later build overlapped other compilation and took 80 s. This pays inference
once during package precompilation rather than on every fresh first draw.
The earlier quiet repeat gave 24.530 s. A later 35.658 s repeat ran with only
5.4 GiB available memory and nearly full swap after cross-device compilation;
clearing the completed validation sessions restored more than 20 GiB of available
memory before the final measurement. Raw current-source results are in
`shader_access_workload_final_quiet_verify.json`; all five repeat images match.
The raw+readonly update additionally precompiles the access-analysis signatures,
which differ from the compiled signatures in their readonly wrappers. The tests
now compare both forms with real hardware/software dispatches (59 shading/view
assertions, plus seven device-free workload assertions on RTX). Its first quiet
fresh-process draw takes 15.161 s, including 13.248 s reported Julia compilation,
with all shader/native misses zero and all five image hashes unchanged. The
package-precompilation capture takes 90.8 s. A bounded residual profile is retained
in `raw_shader_startup_profile.json.flat.txt`; its timed draw is diagnostic, not a
runtime comparison. The profile identifies the missing host `rebuild_plans!`
signature. Its two camera variants now join the 34 resolved host signatures;
the separate restart captures are `rebuild_plans_precompile_*`.

The small-workload control remains in `generic_access_precompile_control.patch`;
it gave 41.914 s and is not part of the production source.

The shading/view/reuse suite passes 55 assertions on RTX, and its precompile
and shading checks pass 62 each on Radeon and Lavapipe. Both software and
hardware traversal requests are covered, with the backend's normal capability
fallback. These Linux checks predate the Mac validation below. No commits, pushes or PRs were made.

The following tables describe earlier source snapshots.

## Readonly ranges and cache persistence, 2026-11-05

The final readonly-range run still meets the Cycles target in all 20 matched
cases: five settings, two resolutions and black/uniform environments. Its
largest ratio is 1.08882, and every image SHA-256 matches the preceding source
exactly. Individual medians and comparisons are in
`byte_range_readonly_acceptance.json` and `byte_range_readonly_*`.

Mantle previously classified aliases by pooled storage owner. Disjoint arrays
in one slab consequently lost readonly wrappers, creating allocation-dependent
kernel signatures. Both eager and graph launches now compare native arrays'
byte ranges. Overlapping views, mixed element widths and aliases in aggregates
remain writable. The new regression fails eight assertions before the fix;
all 81 readonly assertions pass afterwards, including empty byte ranges.
Hikari's device-free workload now
describes the more precise readonly signatures, verified against actual
hardware/software dispatches.

The follow-up native-cache check exposed a separate persistence bug: an older
Julia process could replace a newer driver's cache file on exit. Seven native
pipeline compilations recurred despite zero SPIR-V misses. Saving now locks the
file and merges its latest contents with the process's cache through Vulkan,
using a separate merge target ([Vulkan cache merge API](https://docs.vulkan.org/refpages/latest/refpages/source/vkMergePipelineCaches.html)). A stale writer then preserves the complete
62,929,460-byte seed. The final unchanged-source restart then reports zero
shader and native pipeline misses for all ten point-light/environment fixtures,
with identical image hashes. With the byte-region host optimization, first
point-light draw is 15.287 s (previously 16.564 s); first draw with the environment
subsequently added is 18.274 s. Raw records are
`final_region_cache_verified_env_*`, compared with
`final_merged_pipeline_cache_verify_env_*`. The isolated 15.130 s control is
therefore confirmed by the integrated production source. Radeon and Lavapipe_jll
each pass all 114 focused alias, persistence and progressive/sample assertions.
The final 512² runtime repeat is 45.356 / 75.884 / 47.528 / 47.077 / 52.938 ms
for matte / rough glass / rough metal / plastic / mixed metal-coat-sheen, with
all five image hashes unchanged (`final_regions_runtime_512`).
The FileWatching dependency is Julia's standard-library
pidfile implementation. Cache tests compare each driver's own save/load
capability: persisting intermediate shader code need not persist executable
machine code. A separate system Mesa 26.2.5 driver emits a 20-byte blob whose
header claims 32 bytes. Mantle rejects that truncated header; the same test passes
the installed Lavapipe_jll 26.2.2 driver. The raw diagnostic is
`system_lavapipe_cache_diagnostic.json`; no driver-specific exception was added.

Current-source Metal validation uses the existing canonical Mac repositories
and GPUCompiler 2.17. It found 64 host bytes allocated per sample in Objective-C
release calls. A typed release method for Metal's unmanaged command references
removes that boxing while retaining managed wrappers' ownership checks. All
15 focused ownership/allocation assertions pass, including the Metal 4 encoder,
and the complete sample again allocates zero bytes, also after a fresh package
restart. Shader reuse, device-free precompilation and progressive-rendering checks pass on the Mac. The same
current-source material references and furnace pass 335 assertions each on
Radeon and Lavapipe_jll; RayMakie passes all 131 nested material/image-update
assertions after these changes. The Mac additionally passes all 321 Cycles-reference and 14 furnace
assertions. These checks do not establish a Cycles runtime bound for Metal.

A paired M5 measurement of the same five settings at 512², 64 samples and depth
four alternates typed/current draws for 31 repetitions after five warmups:

| Principled setting | Typed | Current | Current / typed |
| --- | ---: | ---: | ---: |
| Matte | 134.879 ms | 124.868 ms | 0.926 |
| Rough glass | 258.782 ms | 223.719 ms | 0.865 |
| Rough metal | 149.574 ms | 139.464 ms | 0.932 |
| Plastic | 153.656 ms | 143.224 ms | 0.932 |
| Mixed metal/coat/sheen | 203.840 ms | 199.431 ms | 0.978 |

Every paired image is exactly equal, and each image remains unchanged after
warmup and timing. This is a comparison against typed Principled on the same M5,
not against Cycles on another GPU. Raw samples are retained in
`current_metal/paired_typed_current.json`. The backend-independent protocol is
`PairedMaterialRuntime.run(output)` from `paired_runtime.jl`, after loading
`hikari.jl`; unlike the original Vulkan compiler-accounting benchmark, it needs
no Lava compiler counters. The saved protocol also passes all 20
image assertions on RTX: current/typed ratios are 0.946 / 0.872 / 0.930 / 0.922 /
1.012 for the same five settings (`final_paired_runtime_512.json`). The latest Vulkan-only readonly host optimization
was not copied to the Mac: automatic approval review rejected that source
transfer. No attempt was made to bypass the rejection.

## Recorded RTX 4000 Ada results, 2026-11-04

512×512, 64 samples, depth four, hardware traversal, five warmups and 31 frames.
All times below are full-frame medians in milliseconds. Hikari current commit
is `aa42817`; original commit is `d24058b` with the two host compatibility changes
described above. Blender is 5.2.1 with OptiX; Hikari uses Vulkan ray queries.

| Material settings | Blender Cycles | Historical pbrt on current runtime | Typed Cycles Principled | Current Cycles Principled |
| --- | ---: | ---: | ---: | ---: |
| Matte | 43.913 ms | 46.704 ms | 175.774 ms | 191.253 ms |
| Rough glass | 89.491 ms | 79.906 ms | 349.535 ms | 325.031 ms |
| Rough metal | 46.167 ms | 49.002 ms | 184.683 ms | 201.086 ms |
| Plastic | 47.922 ms | 48.959 ms | 192.059 ms | 200.021 ms |
| Mixed metal/coat/sheen | 53.461 ms | 54.726 ms | 214.266 ms | 224.646 ms |

Current versus historical pbrt on the current runtime is -9.56%, -29.37%,
-1.89%, and +0.81% for the first four rows. This does not establish a ≤5%
regression against the original renderer: its execution path differs.
In particular, the plastic models differ visibly: pbrt's
random-walk coat gives mean RGB 0.03798 versus 0.06494 for Hikari Principled and
0.06497 for Blender. Matte and metal means agree within 0.3%; rough glass has
sampling/model differences. The two Hikari Cycles images agree in every row,
with maximum component error 4.55e-7. The current implementation still has a
large cost against typed Cycles: +131.5% for the mixed metal/coat/sheen case.
The image matrix also shows a faceted dark-edge difference in both Hikari
Cycles variants relative to Blender and original pbrt in this low-tessellation
fixture; image-mean agreement is not a claim of pixel-identical cross-renderer
output. The typed/current differential checks do not detect that shared difference.

Fresh-session cached first-render checks of the plastic fixture measure
35.855 s original, 27.018 s typed Cycles, and 47.967 s current Cycles. All have
zero SPIR-V cache misses. These include first-frame host work and pipeline
creation, and exclude package loading and scene setup; they are not isolated
shader-compilation timings. Package loading rebuilt images after historical
source switches, so its measured times are recorded separately, not compared
as equivalent cache states. Startup performance remains unfinished.

Raw JSON, float images, EXRs, source identities, compatibility patch, CSV,
bootstrap intervals, and the image comparison are in
`out/material_comparison_20261104/` under the workspace. `report.py` regenerates
the summary and image figure after `read_cycles_images.py` exports the EXRs.

## Compilation measurements

`compilation.jl` records separate clocks for setup, first frame, Julia JIT,
compiler calls, LLVM/SPIR-V phases, and Vulkan pipeline creation. It temporarily
wraps the original compiler and pipeline methods in the active session and
restores their tracked definitions in `finally`. No compiler/runtime source
files are edited. Phase/JIT clocks overlap and must not be added together.

Each implementation gets a fresh object-cache directory, configured with
`JULIA_OBJCACHE_PATH` before loading GPU packages, and a separate application
Vulkan pipeline-cache blob. The second run restarts Julia and reuses those
directories. Package images and driver-private caches are not cleared; "cold"
here means empty application object/pipeline caches. Cache blobs were saved
explicitly and their nonzero sizes verified on restart. The initial
`compile/current/` captures had a world-age issue saving the pipeline cache;
the corrected current results are `compile/current_v2/`, indexed in `compile/index.json`.

Shading-kernel compiler-call wall times, in seconds, in case order below:

| Material settings | Historical pbrt on current runtime | Typed Cycles Principled | Current Cycles Principled |
| --- | ---: | ---: | ---: |
| Matte (first case, empty application caches) | 2.357 | 12.346 | 30.544 |
| Rough glass | 1.242 | reused | reused |
| Rough metal | 1.138 | reused | reused |
| Plastic | 5.908 | reused | reused |
| Mixed metal/coat/sheen | No equivalent | reused | reused |

The Principled compilation cost is charged to the first case: one typed
shading kernel, versus three current capacity kernels. These are compiler-call
boundaries, including inference/JIT and cache work within those calls, not
whole first-frame durations. Common tracing/camera/other kernels are excluded
from this table but included in the raw per-frame compiler totals. Other
small GPU kernels can still compile in the later cases even when shading reuses.

After restart, all three implementations had zero SPIR-V misses, including
scene setup. Their first shading-kernel lookups still cost 0.255 s original,
2.635 s typed, and 3.478 s current. First-frame wall times were 28.031 s,
23.025 s, and 32.442 s respectively; that remaining latency is not called
GPU compilation time. Per-case same-session repeats created no new pipelines
and had no SPIR-V misses. First-frame native pipeline creation was 0.017 /
0.026 / 0.109 s with empty application caches and 0.002 / 0.004 / 0.030 s
after restart, original / typed / current.

Blender's GPU statistics do not expose CPU shader/SVM compilation separately.
Nsight Systems CUDA/NVTX traces do capture native OptiX compilation: eight
`optixTaskExecute` calls occupy 0.481 s wall time (0.906 summed thread seconds,
with parallel tasks), followed by 0.00575 s pipeline creation. After restarting
Blender with the same configured cache paths, one compilation-task call takes
0.00131 s and pipeline creation takes 0.00697 s. This shared native-module
work occurs once for the suite, not once per material. It is a different stage
from Julia-to-SPIR-V compilation. Both `.nsys-rep` and SQLite traces are retained.
Blender's cache directory is redirected with `XDG_CACHE_HOME`, as in
[Cycles' cache-path implementation](https://raw.githubusercontent.com/blender/blender/main/intern/cycles/util/path.cpp).

Run `compile_report.py out/material_comparison_20261104/compile` to regenerate
the compile summary and CSV. Complete phase records, compiler/cache counts,
pipeline-call timings, package-load measurements and raw traces live there.
Compilation/startup optimization remains incomplete: the current implementation
has a larger initial shading compilation cost than typed Principled in this suite.

## Uniform environment-light extension

`environment_cases.json` in `target_10_percent/` adds a uniform world of strength
0.2 to the matched five-setting scene. The Blender and Hikari image means agree
within 0.3%, including the glass case. This exposed a gap that the original black
world did not exercise: 82.086 / 112.558 / 83.815 / 84.089 / 90.131 ms in Hikari,
versus 43.913 / 89.491 / 46.167 / 47.922 / 53.461 ms in Cycles at 512².

GPU timestamps place 9.05 ms per wave in primary tracing, with a further 1.90 ms
in escaped-ray evaluation. Computing primary misses inline reduces full frames
to 59.066 / 89.561 / 61.100 / 61.089 / 67.179 ms. Computing all surface misses
inline gives 46.801 / 79.670 / 49.116 / 49.106 / 54.962 ms in paired controls,
meeting every ten-percent limit with differential image checks. Raw controls are
`primary_environment_inline.json`, `environment_all_inline.json` and
`environment_control_summary.json`; timestamp captures are `environment_passes_512.json`
and `environment_primary_inline_passes.json`.

The production source now calls the unchanged escaped-ray evaluator directly
from surface tracing, including its original MIS/window handling. The deferred
queue and stage remain for media, where attenuation must be computed first.
Surface-only scenes allocate a one-entry placeholder instead of a full escaped
queue. This uses the same path on every backend and preserves material shading
signatures. The first acceptance run passes 314 camera/alpha/medium/sky checks
on each of RTX, Radeon and Lavapipe; the follow-up lifetime, progressive and new
vacuum-medium escape checks pass 208 assertions each on Radeon and Lavapipe.
All five black-world images also match the preceding production images exactly.
The direct-source repeat meets all ten environment-light limits:

| Settings | Cycles 512² | Hikari 512² | Cycles 1024² | Hikari 1024² |
| --- | ---: | ---: | ---: | ---: |
| Matte | 43.913 ms | 46.683 ms | 175.774 ms | 190.343 ms |
| Rough glass | 89.491 ms | 79.752 ms | 349.535 ms | 322.649 ms |
| Rough metal | 46.167 ms | 49.109 ms | 184.683 ms | 201.026 ms |
| Plastic | 47.922 ms | 48.818 ms | 192.059 ms | 198.709 ms |
| Mixed metal/coat/sheen | 53.461 ms | 55.048 ms | 214.266 ms | 224.294 ms |

All twenty point/world/resolution measurements meet the target. The largest gap is
8.9% for rough metal under the 1024² environment. The first fifteen images are
bit-identical to the preceding implementation. The additional 1024² environment
repeat retains timing JSON and image hashes, with in-memory first/warm/final
comparisons; it writes no additional raw images. Its cross-renderer image means
agree within 0.22%. Raw records are in `production_inline_escape_environment_*`.
Only one production rendering path is used.

## Optimization controls retained as artifacts

The numbered JSON captures and compiler modules in `target_10_percent/` retain
both successful and rejected controls. The source uses one renderer and the same
launch policy on all Vulkan drivers.

| Control | Result and disposition |
| --- | --- |
| Deferred wavelengths | Exact images, slower full frames; not adopted (`deferred_wavelengths.json`). |
| Outlined host packers | No startup improvement; not adopted (`host_pack_outline.json`). |
| Precompile host plan constructors | Already cached: 0.003 s to prime, 16.708 s first draw, no improvement; no production change (`host_plan_precompile_mwe.json`). |
| Erase recording dispatch types / generic packing | 16.204 / 16.273 s individually, 15.912 s combined. Trace shows cost moving into adaptation and buffer retention; not adopted. |
| Remove readonly closures | 16.173 s: the same 1.2 s moves into per-array-pair overlap methods; not adopted. |
| Precompute readonly byte regions | 15.130 s in the isolated control; removes the 1.2 s array-pair specialization group and preserves exact pixels. Integrated: 15.287 s after restart with zero shader/native misses, plus overlap and empty-range regressions. |
| Capacity partition | Routes once per hit using bounded common kernels; integrated. Mixed and singleton edits preserve shader reuse. |
| Singleton shortcut | Exact images, lower dispatch/routing cost on RTX and Radeon; integrated (`singleton_capacity_*.json`). |
| Primary payload reconstruction | Exact image checks, including transparent cameras and media arrivals; integrated with an explicit primary tag. |
| Zero remaining specular weight | Preserves the original eta/film condition and exact equations; integrated after CPU/GPU differential checks. |
| Retyping controls for subsurface | Some fixed one case but broke another. Original policy retained; early setup inlining and common 32-invocation shading pass the full acceptance set. |
| Larger waves | Exact images, closes the 1024² gap within the available-memory bound; integrated. |
| Resampling the memory budget each frame | Replaced plans on budget fluctuations. Integrator retains its initial bound until closed (`final_wave_budget_stability.json`). |
| Progressive batch resizing | Previously lost accumulated samples. Dedicated accumulator ownership now preserves both buffers without device copies. |

Earlier experiment notes, source patches and logs are preserved in
`pre_upstream_source_snapshot/` and `post_upstream_validated_snapshot/`.
The retained controls are diagnostics, not public alternative rendering APIs.


## Compositions and heterogeneous scenes, 2026-11-05

`multi_material_cases.json` places the five Principled settings in one scene,
with a point light and uniform world, at the same 64 samples and depth four.
Hikari takes 44.461 / 179.276 ms at 512² / 1024²; Cycles takes 48.433 / 184.648 ms.
Four shading signatures serve the five materials. The full-frame means differ
by +0.029% / -0.061%; inner-disk sphere means differ by at most 1.71% / 0.29%.
The fixed image orientation is inherited from the original comparison.
`compare_scenes.py` regenerates per-image and per-object statistics from the raw
captures in `multi_material_512/` and `multi_material_1024/`.

This additional scene passes, but explicit graph compositions expose a separate
performance gap. `mixed_graph_cases.json` measures `Mix(Principled, Principled)`:

| Composition | Current Hikari | Cycles |
| --- | ---: | ---: |
| Matte and rough metal | 139.055 ms | 48.177 ms |
| Layered Principled and rough glass | 256.393 ms | 113.509 ms |

Both graphs use general sixteen-record storage instead of the single-node
bounds. Their image means agree within 0.51%, and inner-disk means within 1.38%.
They do **not** yet meet the ten-percent target. The earlier twenty-case target
claim applies to the single-node settings, not to these compositions.

Controlled capacity/layout experiments retain the program and all BSDF formulas.
`composition_capacity_probe.jl` and `composition_array_probe.jl` restore every
session override after each comparison. All successful controls below retain
exact pixels; none has been adopted as the renderer's default:

| Control | Matte/metal | Layered/glass | Finding |
| --- | ---: | ---: | --- |
| Tuple storage, original sixteen records | 139 ms | 256 ms | Composition baseline. |
| Tuple storage, four/eight records | 89 ms | 2,505 ms | Smaller alone is insufficient. |
| Two records with general tagged interpretation | 64 ms | Not applicable | A controlled bound for the two active records. |
| Two records, opacity proof and constant-index traversal | 61 ms | Not applicable | Still above the Cycles target. |
| Local array, sixteen records | 81 ms | 163 ms | Removing tuple selection helps both. |
| Local array, four/eight records | 89 ms | 601 ms | Capacity sensitivity remains. |

The slow eight-record tuple case also remains slow with partitioning bypassed.
Its native binary grows from 815,872 to 5,864,064 bytes and register use from
168 to 255. Disabling loop unrolling on the traversals emits two DontUnroll
controls, but leaves that expansion and slowdown essentially unchanged; this
rules out queue routing and those two loop controls as sufficient explanations.
Raw resource records and shader captures are `composition_capacity_resources*`
and `composition_rolled_capture*`.

The flat-array control exposed two independent correctness gaps. Mantle's access
walk rejected Julia's `jl_value_ptr` address conversion. It now preserves the
object's provenance without marking a payload read or write; actual pointer
loads/stores retain their accesses, and unknown foreign calls remain errors.
An outlined static-array helper then retained a heap allocation until GPU GC
lowering. Lava now inlines helpers with managed/derived Julia object arguments
before allocation optimization, allowing stack promotion. The address tests
fail before the first fix, and the device test fails before the second. Current
checks pass 44 host assertions, 116 RTX access/readonly assertions, and 85 Radeon
compiler/access assertions. Lavapipe_jll additionally passes the same host,
compiler and device checks plus all 81 readonly assertions. The complete Lava suite additionally passes 1,625 checks, with two existing
broken cases. Composition performance investigation remains open.


The follow-up flat-float storage control initially hit GPU allocation errors.
Inlining derived as well as tracked managed arguments fixes that compiler case;
it then renders exactly, but takes 86 ms for matte/metal at capacity sixteen.
Capacity three takes 80 ms. Nine-record local arrays, structured or flat, take
about 670 ms for layered/glass, so a one-slot padding control does not solve the
capacity sensitivity. These negative controls are retained in
`composition_capacity_*_flat_*.json`.

`decoded_inputs_probe.jl` instead predecodes constant Principled parameters into
the existing program payload, leaving the renderer's sixteen-record BSDF storage
unchanged. The first image comparison exposed a second decoder: metadata still
read the old parameter layout. Updating both decoders restores exact pixels in
`r429`, taking 117.692 / 231.141 ms for matte/metal and layered/glass. Parameter
decoding alone does not close the composition gap; this control is not integrated.
The refreshed source changes in Mantle/Lava have not been copied to the Mac:
the source-transfer approval remains unanswered.

The paired typed/current composition run (`r431`) measures 58.776 / 138.395 ms
for matte/metal and 95.525 / 257.715 ms for layered/glass. This isolates a large
shared-storage cost even against the typed implementation on the same renderer.

The initial blanket managed-argument inlining failed six assertions in the full
Radeon run (`r424`). A cache-free A/B identifies early inlining of nonreturning
empty-set dispatch helpers as the trigger (`r459`); inlining the normally returning
dispatch helpers passes (`r457`). The failure also occurs with software traversal
on Radeon (`r460`), whereas the identical forced control passes on Lavapipe_jll
(`r461`). A four-assertion scalar loop MWE around a nonreturning helper passes on
Radeon (`r463`), so that reduction does not yet reproduce the material failure.
`managed_allocation_probe.jl` retains the cache-free reference controls, including
normal restoration of the compiler methods and cache settings.

The production rule now follows newly allocated objects, including stack-held
wrappers, and marks only helpers reached by those objects for early inlining.
It does not override unrelated immutable by-reference arguments. The expanded
19-assertion IR test covers direct pointers, wrappers copied with memcpy, and
returned allocations; its initial form fails five assertions under the blanket
rule. Direct and wrapped local-object GPU access tests both pass. The full compiler
suite passes 1,632 assertions with two existing broken cases (`r454`), followed by
six additional targeted carrier assertions (`r458`). Current-source RTX pointer,
access and readonly checks pass 181 assertions (`r462`); Radeon compiler/access
checks pass 100 (`r464`). After a package rebuild and fresh process, all 321 Cycles
material assertions and 14 furnace assertions pass on Radeon (`r453`). A fresh
Lavapipe_jll process also passes all 321 references and 14 furnace checks (`r465`).

A diagnostic worker crashed with Julia SIGSEGV (`r456`), confirmed in the system
journal for PID 3263253; no native stack was retained by the journal. The same
control completes in a fresh worker (`r457`). This is retained as an unresolved
host diagnostic issue, separate from the now passing production material gate.

Current-source local-array controls retain exact pixels (`r466`): 85.887 /
162.522 ms. Combining that layout with decoded parameters takes 60.413 /
120.915 ms (`r471`), still exact. Layered/glass now reaches the target in this
restricted control; matte/metal does not. No renderer format change is integrated.

Two negative controls clarify the remaining cost. Excluding thin-film code
reduces SPIR-V from 1,720,476 to 1,330,184 bytes and native code from 816,128 to
618,496 bytes, but leaves registers at 168, local memory at 16 bytes and shared
memory at 8,320 bytes. Frame time stays about 139 / 257 ms (`r468`), so the
unused film branch is not the runtime bottleneck here. Leaving unused array
records uninitialized and summing only the live prefix takes 84.088 / 163.864 ms
(`r469`), also exact, without a useful improvement over initialized arrays.
