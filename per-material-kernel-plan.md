# Per-Material Kernel Split — Plan

> **2026-08-19.** The split landed; what is dated here is how it is
> *dispatched*. `foreach_type` inside a `Lava.concurrent_dispatch_group` is
> gone — the per-type kernels are one `compute!` pass of the round graph
> (`src/integrators/volpath/graph.jl`), which is what says they are independent,
> and the backend fuses their prepare-indirect kernels. Read the rest as the
> record of a decision, not as the current API.

Status: **on hold pending data showing this is actually the bottleneck**.
Per-kernel GPU timing on killeroo (see `profiling-and-orthogonal-mwes.md`
Phase 5 results, 2026-06-03) shows shading kernels collectively run for
0.1 ms / sample (0.1 % of GPU time). 94 % of GPU is in
`vp_generate_ray_samples` + `vp_generate_camera_rays`. This plan can
only matter for scenes whose shading is actually expensive, which
remains to be demonstrated — current evidence says killeroo is NOT
such a scene.

Author: 2026-06-03.

## Goal

Close as much of the OptiX gap as possible on surface-heavy scenes
without regressing the volumetric scenes we already win.

Current honest baseline on RTX 4000 Ada (render-only, native settings,
min of 4 trials — from `RayDemo` commit `5b6e464`):

| scene         | res        | spp | maxdepth | Lava HW   | OptiX     | Lava/OptiX |
| ------------- | ---------- | --- | -------- | --------- | --------- | ---------- |
| killeroo_gold | 1368×1026  | 32  | 5        | 2023.4 ms | 712.7 ms  | 2.84×      |
| crown         | 1000×1400  | 16  | 100      | 3675.5 ms | 1676.1 ms | 2.19×      |
| bunny_cloud   | 1920×1080  | 8   | 50       | 2078.5 ms | 2808.8 ms | 0.74×      |
| materials     | 1200×900   | 10  | 50       | 893.8 ms  | 1004.0 ms | 0.89×      |

Acceptance bar: average ratio improves; killeroo OR crown improves
≥10 %; bunny does not regress >10 %; materials does not regress >10 %.

## Background: previous attempt (3a9307c, NOT committed to master)

3a9307c added per-material queues + `Lava.concurrent_dispatch_group`
overlap. Bunny regressed 5333 → 9033 ms (~70 %). Root causes, from
re-reading the diff:

1. **It removed the inline-shade fast path entirely.** Every
   non-medium hit re-materialised → re-read → shaded. Bunny's
   depth-0 primary rays (camera is outside the cloud volume) hit
   surface directly and used to take the inline path.
2. **The monolithic kernel was still launched.** The render loop
   added `vp_shade_typed!` but kept `vp_shade_surface_hits!`. Bunny's
   medium-exit surface hits flowed through the OLD kernel, which
   still inlined every material's BSDF code. So bunny paid for both
   shading systems.
3. **Pre-SOA.** AOS work-item materialisation was 170 MB/bounce on a
   1.4 Mpx scene because every field read pulled all fields through
   cache. SOA (committed b008bfa) drops that to ~50–100 MB.
4. **`push_typed_hit!` may have leaked `with_index` into the trace
   kernel SPIR-V.** Need to verify the @generated routing dispatch
   actually compiles to a direct branch and doesn't keep every
   material's code reachable in the trace kernel.

None of those four is "per-material kernel = bad". They're an
implementation that paid the costs without delivering the wins.

## What's different now

- SOA queues exist (b008bfa) → per-field-cost, not per-work-item-cost
- opt4 (361a8d1) inline shade in `vp_trace_and_shade_kernel!`
- Lava emit fixes:
  - eefc75c `concurrent_dispatch_group` + `skip_pre_barrier`
  - 139a0dd `emit_select!` PSB-pointer reconciliation
- Honest baseline + drift guard (RayDemo 5b6e464) so any per-material
  experiment lands against a clean, reproducible reference.

## Design

### Key insight: the texture-evaluator split is FREE in Julia

Material structs already carry texture types as type parameters:

```julia
struct Conductor{EtaTex, KTex, RoughTex, ReflTex} <: Material; ... end
struct Diffuse{KdTex, σTex} <: Material; ... end
struct CoatedConductor{...} <: Material; ... end
```

A `Conductor{TextureRef{ConstantTexture{...}}, ...}` IS a different
Julia concrete type from `Conductor{TextureRef{ImageTexture{...}}, ...}`.
`StaticMultiTypeSet` already buckets by concrete material type. So
per-material kernel split via `MultiTypeWorkQueue{Tuple{Queue{Mat1},
Queue{Mat2}, ...}}` automatically gives us pbrt-v4's basic-vs-universal
texture-evaluator separation without any explicit dispatch — the texture
flavour is already baked into the material's concrete type.

For killeroo (1 Conductor + 1 Diffuse, both with constant textures):
2 queues → 2 kernels per bounce. Each kernel SPIR-V contains exactly
that one material's BSDF + texture eval code; no with_index, no
unreachable code.

For crown (~6 material variants): ~6 queues → ~6 kernels per bounce.

For a scene with one material: 1 queue → 1 kernel. **Same kernel
launch count as today's monolithic shade**, but the kernel SPIR-V is
smaller (no with_index switch) so the registers / instruction cache
are healthier. We pay nothing extra for the degenerate case.

### Per-material work-item type

```julia
struct VPMaterialEvalWorkItem{M}
    # All the fields VPHitSurfaceWorkItem has today, minus material_idx —
    # the queue's element type already encodes the material.
    work::VPRayWorkItem        # carrier of beta/r_u/r_l/lambda/pixel_index
    pi::Point3f
    n::Vec3f
    dpdu::Vec3f
    dpdv::Vec3f
    ns::Vec3f
    dpdus::Vec3f
    dpdvs::Vec3f
    uv::Point2f
    interface::MediumInterface
    primitive_index::UInt32
    bary::SVector{3,Float32}
    arealight_flat_idx::UInt32
    triangle_area::Float32
    t_hit::Float32
    wo::Vec3f                  # -ray.d, hoisted so the trace kernel pays for it once
end

should_use_soa(::Type{VPMaterialEvalWorkItem{M}}) where M = true
```

One concrete `VPMaterialEvalWorkItem{Mat}` per material type lives in
its own SOA queue.

### `MultiTypeWorkQueue` per-material container

```julia
state.per_material_queue::MultiTypeWorkQueue{...}
```

Built lazily on first `render!`, keyed on the scene materials'
`StaticMultiTypeSet` type tuple. Cached across renders; rebuilt only
when the scene's material-type tuple changes (handled by signature
check à la `ensure_per_material_queue!`).

Capacity per typed queue: a single material can in principle eat the
whole image's worth of hits. Conservative: capacity = max queue size
needed by the scene. Start with `width * height` per queue (same as
today's hit_surface_queue capacity). If VRAM pressure shows up,
revisit (e.g. shared backing buffer with offsets).

### Trace kernel (was `vp_trace_and_shade_kernel!`)

Becomes `vp_trace_kernel!`. Responsibilities:

1. Trace + alpha-test loop (unchanged)
2. Medium-ray branch → push to `medium_sample_queue` (unchanged)
3. Non-medium hit:
   - Compute geometry + bump perturbation (unchanged)
   - Resolve `MixMaterial` up front (was inside shade)
   - Null-material boundary → push to `next_ray_queue` inline; no
     shading (was inside `evaluate_material_inner!`)
   - Otherwise: `push_typed_hit!(per_material_queue, materials,
     hit_work)` to route by concrete material type
4. **No inline shading. No `with_index` switch. No light/BSDF code.**

The inline-shade fast path from opt4 is **replaced** by per-material
shading, not deleted alongside it. Inline shade only made sense when
there was a single monolithic shade kernel — once per-material kernels
exist, each one IS the inline shade for its material.

### `push_typed_hit!` — must NOT leak `with_index` into trace SPIR-V

The routing dispatch needs to be:

```julia
@inline @generated function push_typed_hit!(
    mtwq::MultiTypeWorkQueue{Qs}, materials::StaticMultiTypeSet{Data}, hit
) where {Qs, Data}
    N = length(Data.parameters)
    branches = Expr[]
    for i in 1:N
        push!(branches, quote
            if hit.material_idx.type_idx === UInt32($i)
                push!(mtwq.queues[$i], _to_typed_item(hit, materials.data[$i][hit.material_idx.vec_idx]))
                return
            end
        end)
    end
    return Expr(:block, branches..., :(return))
end
```

This is structurally identical to `with_index` (same N-way switch),
but importantly:
- Each branch's body is a `push!` to a typed queue. No material
  evaluation. The body is the same size regardless of which material
  type the branch is for.
- The trace kernel SPIR-V will contain N small `push!` paths instead
  of N material BSDFs. Vastly smaller than the current shading kernel.

**Critical check** (Phase 1 below): dump the trace kernel SPIR-V
size before/after the per-material change and confirm it shrinks.

### Per-material shading kernel

```julia
@kernel function vp_shade_material_kernel!(
    typed_work,                    # VPMaterialEvalWorkItem{M}
    next_ray_queue, pixel_L,
    accel, media_interfaces, media,
    materials, lights, rgb2spec_table,
    bvh_nodes, ...,
    camera, samples_per_pixel,
    rr_depth,
)
    # The kernel is monomorphic on the queue's element type — only this
    # material's BSDF code links in. Calls evaluate_material<M> /
    # sample_material<M> directly, no with_index.
    mat = material_of_type(materials, typeof(typed_work)) # zero-overhead lookup
    # ... emission + direct lighting (inline shadow trace, opt4 style) + bsdf sample ...
end
```

`material_of_type` resolves the concrete material instance at compile
time from the queue's element type — Julia inlines it to a direct
struct load.

### `vp_shade_typed!` (drains all per-material queues)

```julia
function vp_shade_typed!(state, accel, media_interfaces, media, materials, lights,
                         camera, samples_per_pixel, regularize)
    Lava.concurrent_dispatch_group() do
        foreach_type(vp_shade_material_kernel!, state.per_material_queue,
                     state.next_ray_queue,
                     state.pixel_L,
                     accel, media_interfaces, media,
                     materials, lights, ...)
    end
end
```

Called once per bounce in the render loop. Replaces both:
- `vp_shade_surface_hits!` (was draining `hit_surface_queue`)
- The inline-shade body inside `vp_trace_and_shade_kernel!`

### Medium-exit surface hits

Today the medium kernels push to `hit_surface_queue` and the
monolithic `vp_shade_surface_hits!` drains it. New flow: medium
kernels push directly into `per_material_queue` via the same
`push_typed_hit!` (or a thin wrapper). Removes the second shading
system; one drain in the render loop, one set of per-material kernels.

`hit_surface_queue` is deleted from `VolPathState`.

### Render-loop shape after the change

```julia
for depth in 0:(vp.max_depth - 1)
    vp_generate_ray_samples!(...)
    reset_iteration_queues!(state)        # also clears per_material_queue

    vp_trace_kernel!(...)                  # NOT trace_and_shade — no shading inside

    if !isempty(media)
        vp_sample_medium_interaction!(...) # may push to per_material_queue
    end
    if !isempty(media)
        vp_sample_medium_direct_lighting!(...)
        vp_sample_medium_scatter!(...)
    end

    if length(lights) > 0
        vp_handle_escaped_rays!(...)
    end

    # ONE call drains every typed queue:
    vp_shade_typed!(state, accel, media_interfaces, media, materials, lights,
                    camera, vp.samples_per_pixel, vp.regularize)

    vp_trace_shadow_rays!(...)             # medium-originated shadow rays only
    swap_ray_queues!(state)
end
```

## Phase plan

### Phase 0 — paperwork (this doc + the existing state)
- [x] Commit the per-material plan (this file)
- [x] Confirm baseline assert passes and matches the JSONs

### Phase 1 — pre-flight
1. Add `VPMaterialEvalWorkItem{M}` + `should_use_soa` declaration.
2. Add `push_typed_hit!` with the `@generated` direct-branch routing.
3. **Sanity check #1:** in a microbench scene (1 conductor + 1 diffuse),
   dump the trace-kernel SPIR-V module size before/after. Expect a
   significant shrink (no material BSDF code in the trace kernel).
4. **Sanity check #2:** in that same scene, the per-material queue
   should land exactly the expected items (gold hits to queue 1,
   diffuse hits to queue 2). Tested via a host-side `foreach` that
   counts items per queue.

If either sanity check fails, stop and diagnose before doing the rest.

### Phase 2 — integration
5. Replace `vp_trace_and_shade_kernel!` with `vp_trace_kernel!`.
   Routing instead of inline shade.
6. Add `vp_shade_material_kernel!` + `vp_shade_typed!`.
7. Wire `vp_shade_typed!` into the render loop, removing the
   monolithic `vp_shade_surface_hits!` call.
8. Route medium-exit surface hits through `push_typed_hit!` too;
   delete `hit_surface_queue` from `VolPathState`.

### Phase 3 — measurement
9. Run the per-pbrt-scene regression suite (HW RT) at low spp to
   confirm correctness (energy ratio, tile score) on the same scenes
   we used to certify opt4.
10. Run the honest benchmark (`run_julia_benchmarks` against
    `lava_hw_match_v?`) at the full native settings, 1 warmup + 4
    trials. Compare against the committed
    `lava_hw_match_v1.json` baseline.
11. Per-scene win/loss table.

### Phase 4 — land / roll back
**Land** iff:
- Avg (Lava ms / OptiX ms) across the 4 scenes improves
- killeroo OR crown improves ≥ 10 %
- bunny does NOT regress > 10 %
- materials does NOT regress > 10 %
- Full Hikari pbrt regression suite (`test_pbrt_all_materials.jl`) at
  256 spp still passes both SW and HW

**Roll back** otherwise. Memory entry documents what happened so the
next attempt has the actual numbers.

## Risks and mitigations

| risk | mitigation |
|------|------------|
| `push_typed_hit!` accidentally inlines all material code into trace kernel | Phase 1 sanity check #1 (SPIR-V size dump) before anything else |
| Per-material queue capacity blows VRAM on big scenes | Per-queue capacity = `width * height` initially; if VRAM-bound, share a single backing buffer with per-type offsets |
| `concurrent_dispatch_group` overlap doesn't actually overlap on NVIDIA | Re-run with and without it; concurrent overlap is bonus, not load-bearing |
| Bunny regresses despite all care | Roll back. Document the bunny-specific pattern (which queue dominates, which kernel size) so a follow-up can target it |
| Mix-material handling becomes hairy with typed queues | MixMaterial resolves to a concrete type before push_typed_hit; the queue only sees concrete types. Same as today. |

## Things to try after this lands (or instead, if it doesn't)

- **Aggregated film writes.** pbrt-v4 doesn't do this and we don't
  either; both atomically increment per-pixel. A per-warp shared-mem
  staging buffer flushed once per warp could cut atomic contention.
- **Vulkan sampler-descriptor arrays for textures.** Today every
  texture access goes through PSB pointers + manual decode. Sampled
  images via Vulkan's sampler descriptors would use HW texture units
  including their dedicated caches.
- **Recursive per-MixMaterial split.** If a scene uses MixMaterial
  heavily, each branch could become its own typed queue.
- **Per-light-type queues for direct lighting.** Same pattern as
  per-material, applied to lights. Probably small win — most scenes
  have very few light types.
- **CB caching across samples.** Today every sample rebuilds the same
  command buffer. A recorded-once-replayed-N-times cache could save a
  few ms per sample at our dispatch overhead.

## Decision log

This section will be updated as we go.
