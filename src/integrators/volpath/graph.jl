# One volpath sample, as Mantle graphs.
#
# The wavefront loop is ~20 dispatches a round whose only ordering constraints
# are which queue each stage fills and which one it drains. Those constraints
# used to be expressed by the order of the calls plus two hand-placed
# `concurrent_dispatch_group` / `concurrent_indirect_group` scopes asserting
# which dispatches were independent — an assertion nothing checked, and the one
# that was wrong (a prepare-indirect read racing its own producer) cost ~15 % of
# the energy on a scene where the queue happened to empty mid-pipeline.
#
# Here each stage declares what it reads and what it writes, and the barriers
# between them are derived from that. A stage that shares nothing with its
# neighbour gets no barrier and the two overlap; a stage that drains what
# another filled waits for exactly those bytes.
#
# Three graphs, because two of the boundaries are the host's:
#
#   setup       clear the film, reset the ray queue, generate camera rays
#   round       one bounce, built twice for the two directions of the ray
#               queue ping-pong
#   accumulate  this sample's spectral radiance into the RGB accumulators
#
# and `finalize` for the divide, which is its own plan because a batched caller
# runs it once after every sample rather than per sample.
#
# The bounce loop stays a host loop: a `Plan` is compiled once and run many
# times, so a plan per round expresses it, and the early exit reads a device
# counter and breaks — which no graph-level loop node would make cheaper.

# What is NOT here: the queues are Hikari's own allocations, read by the passes
# as foreign buffers. A graph can order accesses to a buffer it does not own; it
# cannot adopt one into its arena, and aliasing is the arena's. Moving them was
# the obvious last step of the port and it is measured to be worth nothing on
# this loop — `benchmarks/queue_aliasing/aliasing.jl` puts every intermediate
# queue in the placer's hands at 1280x1080 and gets 174 MiB back for ONE round
# and exactly zero for two or more, because every queue is refilled every round
# and so is live from the first to the last. The shipped chunk is eight.

import Mantle
import StructArrays

# ─────────────────────────────────────────────────────────────────────────────
# What a stage touches
# ─────────────────────────────────────────────────────────────────────────────

# `devicebuffers!`, `use!` and `accumulates!` were here: the leaf walk that turns
# a work queue into the buffers a barrier is scoped to, and the two verbs that
# declared them. All three are gone.
#
# The walk is `Mantle.resourceleaves!` — it opens up any container it is handed,
# so a `WorkQueue`, a `StructArray` and a `MultiTypeWorkQueue` need no method of
# their own. The declarations are not written at all: `dispatch!` reads what the
# kernel does to each argument off the kernel, which is where the answer was.
#
# `accumulates!` is worth a sentence of its own, because it was the subtlest
# thing in this file. Six stages did one thing to the per-pixel radiance —
# `atomic +=` — and saying so let them overlap instead of taking a barrier
# between every pair. That is now inferred: a resource whose every write is an
# atomic read-modify-write comes back `Unordered`, and the stages that merely
# clear or read it still get their barriers, because those writes are not atomic.

# ─────────────────────────────────────────────────────────────────────────────
# The values a recorded plan reads per render
# ─────────────────────────────────────────────────────────────────────────────

"""
    render_refs(...) -> NamedTuple

What a sample is rendered with. Two kinds of value:

- **Per-run values are `Mantle.GPURef`s** — `camera`, `initial_medium`,
  `filter_params`, `sample_idx`. Device storage at an address that does not
  move: every dispatch that reads one is packed with that address, once, and
  a new value is stored through the ref (`ref[] = x`) and lands in the run's
  own submission. So one recording renders every sample, and a moved camera
  is one command, not a new plan. `sample_idx` is not even that: the sample
  plan's first pass increments it on the device, and the host stores to it
  only to start the count over (see [`resetsamples!`](@ref)).
- **Everything else is baked** — resolved when the plans record and fixed for
  as long as they live. A change to any of them is a scene or integrator edit,
  and those invalidate the plans where they happen (`sync!(scene)`, the scene
  mutators, [`invalidate!`](@ref)) — nothing is compared per sample.

They were ALL `Ref`s, and a `Ref` meant "re-read me every run": Mantle rewrote
the argument bytes of every dispatch holding one, per run, on the host. Measured
on the fused sample that is 602 writes to move 268 bytes — and because those
bytes could be in flight, the plan needed three copies of its arguments and a
wait to rotate between them.
"""
render_refs(accel, media_interfaces, media, materials, lights,
            camera::Mantle.GPURef, initial_medium::Mantle.GPURef{SetKey},
            filter_params::Mantle.GPURef{GPUFilterParams}, filter_sampler,
            sample_idx::Mantle.GPURef{Int32},
            regularize::Bool, samples_per_pixel::Int32,
            max_component_value::Float32) =
    (accel = accel, media_interfaces = media_interfaces,
     media = media, materials = materials, lights = lights,
     camera = camera, initial_medium = initial_medium,
     filter_params = filter_params, filter_sampler = filter_sampler,
     sample_idx = sample_idx, regularize = regularize,
     samples_per_pixel = samples_per_pixel,
     max_component_value = max_component_value)

# ─────────────────────────────────────────────────────────────────────────────
# Kernels the graph needs and the KA path did not
# ─────────────────────────────────────────────────────────────────────────────

# `KA.fill!` is a launch rather than a `@kernel`, and a pass records kernels.
# Writing the fill as one is also what lets it share a pass with the counter
# reset: the two touch disjoint memory, which is the fact the hand-placed
# `concurrent_dispatch_group` around them used to assert.
@kernel inbounds = true function vp_clear_kernel!(dst, value)
    i = @index(Global)
    dst[i] = value
end

# ─────────────────────────────────────────────────────────────────────────────
# The passes
# ─────────────────────────────────────────────────────────────────────────────

"""Zero every counter the coming round fills. One dispatch for all of them —
a `fill!` per queue was ~20 commands and their barriers per round."""
function reset_pass!(g, state::VolPathState, nxt::WorkQueue)
    queues = round_queues(state, nxt)
    counters = _collect_size_counters((), queues...)
    Mantle.dispatch!(g, zero_size_counters_kernel!, (counters,), 1;
                     group = 1, name = "reset")
end

"""Every queue a round fills, so the reset covers all of them and nothing
carries a count over from the previous bounce."""
function round_queues(state::VolPathState, nxt::WorkQueue)
    pmq = state.per_material_queue
    base = (nxt, state.medium_sample_queue, state.medium_scatter_queue,
            state.hit_surface_queue, state.shadow_queue, state.escaped_queue,
            state.hit_area_light_queue)
    return pmq === nothing ? base : (base..., pmq)
end

"""
Trace the live rays and shade whatever needs no queue of its own.

Two implementations on the accel axis, and the pass kind differs with them: the
software BVH and inline ray queries are one compute kernel over the ray queue,
while the hardware RT pipeline is a `vkCmdTraceRaysIndirect` with an SBT, which
is not a dispatch. `custom!` is the pass for that — the body records through the
backend and the graph still orders it, because what it touches is declared here
either way.
"""
function trace_pass!(g, accel, state::VolPathState, refs, cur::WorkQueue, nxt::WorkQueue)
    Mantle.dispatch!(g, workqueue_map_kernel!,
                         (vp_trace_and_shade_kernel!, cur, nxt,
                          state.escaped_queue, state.medium_sample_queue,
                          state.per_material_queue, state.hit_surface_queue,
                          state.hit_area_light_queue, state.pixel_L,
                          refs.accel, refs.media_interfaces, refs.media,
                          refs.materials, refs.lights,
                          state.rgb2spec_table,
                          state.bvh_nodes, state.infinite_light_indices,
                          state.light_to_bit_trail,
                          state.num_infinite_lights, state.num_bvh_lights,
                          state.num_lights,
                          state.max_depth, refs.regularize,
                          state.sobol_rng, refs.sample_idx,
                          refs.camera, refs.samples_per_pixel, state.rr_depth),
                     Mantle.DeviceRange(cur.size; max = Int(cur.capacity));
                     group = DEFAULT_WORKGROUPSIZE, name = "trace")
end

"""Delta tracking through the media the trace deferred. Fills the scatter queue
and, for rays that leave their medium and land on a surface, the same surface
queues the trace fills."""
function medium_sample_pass!(g, state::VolPathState, refs, nxt::WorkQueue)
    Mantle.dispatch!(g, workqueue_map_kernel!,
                         (vp_sample_medium_kernel!, state.medium_sample_queue,
                          state.medium_scatter_queue, state.per_material_queue,
                          state.hit_surface_queue, state.hit_area_light_queue,
                          nxt, state.escaped_queue, state.pixel_L,
                          refs.media, refs.materials, state.rgb2spec_table,
                          state.max_depth),
                     Mantle.DeviceRange(state.medium_sample_queue.size;
                                        max = Int(state.medium_sample_queue.capacity));
                     group = DEFAULT_WORKGROUPSIZE, name = "medium-sample")
end

"""Direct lighting at a real scatter event. The only producer of shadow rays."""
function medium_dl_pass!(g, state::VolPathState, refs)
    Mantle.dispatch!(g, workqueue_map_kernel!,
                         (vp_medium_direct_lighting_kernel!,
                          state.medium_scatter_queue, state.shadow_queue,
                          refs.lights, state.rgb2spec_table,
                          state.bvh_nodes, state.infinite_light_indices,
                          state.num_infinite_lights, state.num_bvh_lights,
                          state.num_lights,
                          state.sobol_rng, refs.sample_idx),
                     Mantle.DeviceRange(state.medium_scatter_queue.size;
                                        max = Int(state.medium_scatter_queue.capacity));
                     group = DEFAULT_WORKGROUPSIZE, name = "medium-dl")
end

"""The phase-function bounce out of a scatter event."""
function medium_scatter_pass!(g, state::VolPathState, refs, nxt::WorkQueue)
    Mantle.dispatch!(g, workqueue_map_kernel!,
                         (vp_medium_scatter_kernel!, state.medium_scatter_queue,
                          nxt, state.max_depth, state.sobol_rng, refs.sample_idx),
                     Mantle.DeviceRange(state.medium_scatter_queue.size;
                                        max = Int(state.medium_scatter_queue.capacity));
                     group = DEFAULT_WORKGROUPSIZE, name = "medium-scatter")
end

"""Rays that hit nothing: the environment lights, with MIS."""
function escaped_pass!(g, state::VolPathState, refs)
    Mantle.dispatch!(g, workqueue_map_kernel!,
                         (vp_handle_escaped_rays_kernel!, state.escaped_queue,
                          state.pixel_L, state.rgb2spec_table, refs.lights,
                          state.bvh_nodes, state.light_to_bit_trail,
                          state.infinite_light_indices,
                          state.num_infinite_lights, state.num_bvh_lights),
                     Mantle.DeviceRange(state.escaped_queue.size;
                                        max = Int(state.escaped_queue.capacity));
                     group = DEFAULT_WORKGROUPSIZE, name = "escaped")
end

"""Emission MIS for indirect rays that landed on an area light — pbrt-v4's
"Handle emitters hit by indirect rays"."""
function emitters_pass!(g, state::VolPathState, refs)
    Mantle.dispatch!(g, workqueue_map_kernel!,
                         (vp_handle_emitters_kernel!, state.hit_area_light_queue,
                          state.pixel_L, refs.lights, state.rgb2spec_table,
                          state.bvh_nodes, state.light_to_bit_trail,
                          state.num_infinite_lights, state.num_bvh_lights,
                          state.num_lights),
                     Mantle.DeviceRange(state.hit_area_light_queue.size;
                                        max = Int(state.hit_area_light_queue.capacity));
                     group = DEFAULT_WORKGROUPSIZE, name = "emitters")
end

"""
Surface shading, one dispatch per concrete material type.

They go in ONE pass because they are mutually independent — each drains its own
typed queue and writes atomically-claimed slots in the shared next-ray queue —
and a pass is the unit Mantle emits barriers between. That is what the
hand-placed `concurrent_indirect_group` used to say; here it follows from the
dispatches sharing a pass, and the backend fuses their prepares into one.
"""
function shade_pass!(g, state::VolPathState, refs, nxt::WorkQueue)
    for q in state.per_material_queue.queues
        Mantle.dispatch!(g, workqueue_map_kernel!,
                             (vp_shade_material_kernel!, q,
                              state.hit_surface_queue, nxt, state.pixel_L,
                              refs.accel, refs.media_interfaces, refs.media,
                              refs.materials, refs.lights, state.rgb2spec_table,
                              state.bvh_nodes, state.infinite_light_indices,
                              state.light_to_bit_trail,
                              state.num_infinite_lights, state.num_bvh_lights,
                              state.num_lights, state.max_depth, refs.regularize,
                              state.sobol_rng, refs.sample_idx,
                              refs.camera, refs.samples_per_pixel, state.rr_depth),
                         Mantle.DeviceRange(q.size; max = Int(q.capacity));
                         group = DEFAULT_WORKGROUPSIZE, name = "shade")
    end
end

"""Transmittance along the shadow rays the medium direct lighting queued."""
function shadow_pass!(g, state::VolPathState, refs)
    Mantle.dispatch!(g, workqueue_map_kernel!,
                         (vp_trace_shadow_rays_kernel!, state.shadow_queue,
                          state.pixel_L, state.rgb2spec_table, refs.accel,
                          refs.media_interfaces, refs.media, refs.materials),
                     Mantle.DeviceRange(state.shadow_queue.size;
                                        max = Int(state.shadow_queue.capacity));
                     group = DEFAULT_WORKGROUPSIZE, name = "shadow")
end

# ─────────────────────────────────────────────────────────────────────────────
# The graphs
# ─────────────────────────────────────────────────────────────────────────────

"""
One bounce, appended to `g`.

Which stages exist is a property of the scene, not a runtime branch: a scene
without media never fills the medium queues, and on the hardware per-material
chit path the closest-hit shaders have already shaded the surface by the time
the trace returns. A stage that could only ever dispatch over an empty queue is
left out rather than run for nothing — which is also why the shadow pass is
gated: medium direct lighting is its only producer.
"""
function round_passes!(g, state::VolPathState, refs, cur::WorkQueue, nxt::WorkQueue;
                       chit_owns_surface::Bool, has_media::Bool, has_lights::Bool)
    reset_pass!(g, state, nxt)
    trace_pass!(g, refs.accel, state, refs, cur, nxt)
    if has_media
        medium_sample_pass!(g, state, refs, nxt)
        has_lights && medium_dl_pass!(g, state, refs)
        medium_scatter_pass!(g, state, refs, nxt)
    end
    has_lights && escaped_pass!(g, state, refs)
    if !chit_owns_surface
        has_lights && emitters_pass!(g, state, refs)
        shade_pass!(g, state, refs, nxt)
    end
    has_media && has_lights && shadow_pass!(g, state, refs)
    return g
end

"""
    rounds_passes!(g, state, refs, n; …)

`n` consecutive bounces in ONE graph, starting from ray queue A.

A round is not independent of the round before it — it drains the queue that one
filled — so this is not about finding parallelism between them. It is about how
much the host has to say per sample. A plan is recorded when it runs, so a plan
per round means `max_depth` recordings, `max_depth` submissions and `max_depth`
walks over the pass list; `n` rounds in one plan means one of each per `n`.

What makes it expressible at all is that nothing in a round is decided on the
host. Every stage sizes itself off a device-resident count, so a recording is
valid whatever the ray population turns out to be — including zero, where every
dispatch is a no-op and the round costs its commands and nothing else.

The queues alternate inside the graph rather than between plans, which is why
there is one of these and not two: the ping-pong is now an implementation detail
of the recording. `n` rounds starting at A end at A when `n` is even, which is
what lets the same plan run back to back.
"""
function rounds_passes!(g, state::VolPathState, refs, n::Integer;
                        chit_owns_surface::Bool, has_media::Bool, has_lights::Bool)
    cur, nxt = state.ray_queue_a, state.ray_queue_b
    for _ in 1:n
        round_passes!(g, state, refs, cur, nxt;
                      chit_owns_surface, has_media, has_lights)
        cur, nxt = nxt, cur
    end
    return g
end


"""
The head of a sample: clear the film, empty the ray queue, generate camera rays.

The clear and the reset share a pass because they touch disjoint memory, so
nothing orders them against each other — the fact the `concurrent_dispatch_group`
around them was there to assert.
"""
function setup_passes!(g, state::VolPathState, refs, cur::WorkQueue)
    Mantle.dispatch!(g, vp_clear_kernel!, (state.pixel_L, 0f0),
                     length(state.pixel_L); name = "clear")
    Mantle.dispatch!(g, zero_size_counters_kernel!, ((cur.size,),), 1;
                     group = 1, name = "clear-queue")
    Mantle.dispatch!(g, vp_generate_camera_rays_kernel!,
                     (cur, state.wavelengths_per_pixel, state.pdf_per_pixel,
                      state.filter_weight_per_pixel, state.height,
                      refs.camera, refs.sample_idx, refs.initial_medium,
                      refs.filter_params, refs.filter_sampler, state.sobol_rng),
                     Int(state.width) * Int(state.height); name = "camera")
    return g
end


"""This sample's spectral radiance, weighted into the RGB accumulators."""
function accumulate_passes!(g, state::VolPathState, refs)
    n_pixels = Int(state.width) * Int(state.height)
    Mantle.dispatch!(g, vp_accumulate_to_rgb_kernel!,
                     (state.pixel_rgb, state.pixel_weight_sum, state.pixel_L,
                      state.wavelengths_per_pixel, state.pdf_per_pixel,
                      state.filter_weight_per_pixel,
                      state.cie_table.cie_x, state.cie_table.cie_y,
                      state.cie_table.cie_z, Int32(n_pixels),
                      refs.max_component_value, state.output_matrix,
                      state.imaging_ratio),
                     n_pixels; name = "accumulate")
    return g
end


"""The divide that turns the accumulators into the picture."""
function finalize_passes!(g, state::VolPathState, framebuffer)
    n_pixels = Int(state.width) * Int(state.height)
    Mantle.dispatch!(g, vp_finalize_film_kernel!,
                     (framebuffer, state.pixel_rgb, state.pixel_weight_sum,
                      state.width, state.height),
                     n_pixels; name = "finalize")
    return g
end

finalize_graph(dev, state::VolPathState, framebuffer) =
    finalize_passes!(Mantle.Graph(dev), state, framebuffer)

"""
    sample_graph(dev, state, refs, max_depth; …) -> Graph

A whole sample as one graph: setup, the bounce loop, accumulate.

The loop is a `Mantle.repeat!` gated on the ray queue's own live count. That is
the whole reason this can be one graph — the host neither reads that count nor
decides how many rounds run, so there is nothing to break the recording up for.

**The body is TWO rounds, and two is the minimum rather than a tuning knob.** A
round reads one ray queue and writes the other, so an odd body would leave the
queues swapped and the next iteration would read the wrong one; two returns them
to where it found them, which is what makes one recording valid for every
iteration. It also fixes the overshoot: the loop can only stop on an even
boundary, so at most one dead round runs after the rays are gone.

This replaces a host loop that drained the pipeline every eight rounds to read
the same counter. Eight was that drain's amortisation constant — `live_rounds/8`
stalls per sample, bought with up to seven dead rounds of overshoot — and with
the test on the device there is nothing left to amortise.

`finalize` is not here. It is one dispatch, and `render!` takes a kwarg to skip
it between samples: a per-call decision, which is the one thing a single
recording cannot express.
"""
function sample_graph(dev, state::VolPathState, refs, max_depth::Int32;
                      chit_owns_surface::Bool, has_media::Bool, has_lights::Bool)
    a = state.ray_queue_a
    g = Mantle.Graph(dev)
    # The sample counter lives on the device: one thread, one increment, ordered
    # by its declared read and write ahead of every pass that reads the index.
    # The per-run values a sample CAN change — the camera, the initial medium,
    # the filter — are `GPURef`s the plan registers as host-writable by their
    # declared reads; a store through one lands in the run's own submission,
    # ahead of the recorded commands that read it. Everything else was fixed at
    # `record!`.
    Mantle.dispatch!(g, vp_next_sample_kernel!, (refs.sample_idx,), 1;
                     group = 1, name = "next sample")
    setup_passes!(g, state, refs, a)
    d = Int(max_depth)
    if d >= 2
        Mantle.repeat!(g, d ÷ 2; while_nonzero = a.size) do _
            rounds_passes!(g, state, refs, 2;
                           chit_owns_surface, has_media, has_lights)
        end
    end
    # An odd depth has one round the pairs cannot cover. It runs unconditionally:
    # a round on an empty queue is every stage sizing itself to zero work, which
    # is what makes the loop legal in the first place.
    isodd(d) && rounds_passes!(g, state, refs, 1;
                               chit_owns_surface, has_media, has_lights)
    accumulate_passes!(g, state, refs)
    return g
end

@kernel function vp_next_sample_kernel!(sample_idx)
    i = @index(Global)
    if i == 1
        @inbounds sample_idx[1] += Int32(1)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Building and keeping them
# ─────────────────────────────────────────────────────────────────────────────

"""
The compiled plans for one scene shape, film and integrator configuration.

Two plans, because a sample is two things: everything that produces the picture,
and the divide that writes it out. `sample` holds setup, the whole bounce loop
and accumulate; it used to be four plans driven by a host `while` loop, which
existed only so the host could read the ray count between chunks.

There is no key and no per-sample comparison: `refs` holds the values the plans
were packed with, the per-run ones as `GPURef`s a store lands in the run's own
submission — and `stored` what was last stored to them, which is the one thing
a sample compares against (see [`storechanged!`](@ref)). The plans live until the scene or the integrator is edited —
which is where they are dropped (`sync!(scene)`, the scene mutators,
[`invalidate!`](@ref)) — or the film's framebuffer is replaced, which is the one
identity a `render!` still checks, because the film is a per-call argument and
no verb announces its swap.
"""
# `S` and `F`, not a bare `Mantle.Plan`. `Plan{D}` is parametric, so the
# unparameterised field type is a UnionAll and `plans.sample` comes back
# abstract — which makes `Mantle.run!(plans.sample)` a dynamic call that boxes
# its argument: 1015 bytes a sample, on a `run!` that costs 338 by itself.
#
# Mutable, because `VolPathState.plans` is `Any` (its type is scene-dependent)
# and a read of an IMMUTABLE struct through an `Any` field copies it onto the
# heap — every sample, once per `render!` and again in `finalize_film!`. A
# reference crosses the same boundary for free, and identity (`===`) is what
# the invalidation tests pin.
mutable struct VPPlans{P,S,F}
    framebuffer::Any
    # The per-run refs and what they were last stored with (see `PerRun`). The
    # plans own it: `free!` frees the refs with the plans.
    perrun::P
    sample::S
    finalize::F
end

"""
    PerRun{C <: Camera}

The values a recorded sample reads per run — the camera, the initial medium,
the filter parameters, the sample index — as the `GPURef`s the plan's commands
hold the address of, and what each was last stored with. A call argument
cannot announce itself, so identity against `stored_*` is how a sample knows
what to store, and a still scene stores nothing.

Parametric in the camera type, and reached through `VolPath.perrun`, a field
typed as the small union over the camera types — NOT through
`VolPathState.plans`: that field is `Any`, and a compare or a store made
through it was a dynamic call that boxed the camera it was handed, 684 bytes
every sample of a still scene. Through the union the compare is bitwise and
the store is a typed `ref[] = x`.
"""
mutable struct PerRun{C <: Camera}
    const camera::Mantle.GPURef{C}
    const initial_medium::Mantle.GPURef{SetKey}
    const filter_params::Mantle.GPURef{GPUFilterParams}
    const sample_idx::Mantle.GPURef{Int32}
    stored_camera::C
    stored_medium::SetKey
    stored_filter::GPUFilterParams
end

function PerRun(dev, camera::C, initial_medium::SetKey, filter_params::GPUFilterParams,
                sample_idx::Int32) where {C <: Camera}
    return PerRun{C}(Mantle.GPURef(dev, camera), Mantle.GPURef(dev, initial_medium),
                     Mantle.GPURef(dev, filter_params), Mantle.GPURef(dev, sample_idx),
                     camera, initial_medium, filter_params)
end

# Does `perrun` hold a ref of this camera's type? A camera of another type
# needs new plans: the ref's type is baked into every dispatch that reads it.
holdscamera(::PerRun{C}, ::C) where {C <: Camera} = true
holdscamera(::PerRun, ::Camera) = false
holdscamera(::Nothing, ::Camera) = false

"""
    runsample!(plans)

Run the sample plan.

A function barrier, and the whole reason it exists: `VolPathState.plans` is
`Any` — it holds a `VPPlans` whose type depends on the scene — so every field
read through it is dynamic. One dynamic call here makes `plans.sample` concrete
inside, where the `run!` is a static call on a `Plan{D}`.
"""
runsample!(plans::VPPlans) = Mantle.run!(plans.sample)

"""
    storechanged!(perrun, camera, initial_medium, filter_params)

Store the per-sample values that changed since the last store, by identity
against what was last stored — a call argument cannot announce itself, so this
is the one compare a sample makes per value. A still scene stores nothing, and
a moved camera stores the camera and nothing else.

Typed all the way: `perrun` arrives through `VolPath.perrun`, a union over the
camera type, so this is a static call whose compares are bitwise. It used to
take the plans, which are `Any` on the state, and the dynamic call boxed the
camera it was handed every sample.
"""
function storechanged!(pr::PerRun{C}, camera::C, initial_medium::SetKey,
                       filter_params::GPUFilterParams) where {C <: Camera}
    if pr.stored_camera !== camera
        pr.camera[] = camera
        pr.stored_camera = camera
    end
    if pr.stored_medium !== initial_medium
        pr.initial_medium[] = initial_medium
        pr.stored_medium = initial_medium
    end
    if pr.stored_filter !== filter_params
        pr.filter_params[] = filter_params
        pr.stored_filter = filter_params
    end
    return nothing
end

"""
    resetsamples!(perrun)

Start the device's sample count over. The sample plan's first pass increments
`sample_idx` on the device, so the host never feeds it a number per sample;
`film.iteration_index` stays the caller's count of samples taken. A film that
was cleared reads zero there, and that is the one event this hears: a store of
zero, so the next sample is sample one again.
"""
function resetsamples!(pr::PerRun)
    pr.sample_idx[] = Int32(0)
    return nothing
end



"""
    build_plans(backend, state, framebuffer, refs, perrun, max_depth; ...) -> VPPlans

Compile the sample. Called when the key changes, which for a still scene is
once. `perrun` holds the per-run refs `refs` names and what they were made
with, so the first sample compares against it and stores nothing.

`Mantle.Plan` turns the graph into passes, barriers and pipelines; recording
writes the command buffers, and `run!` does it on the first run. A caller who
wants that cost outside a timing loop calls `Mantle.record!` on each of
[`allplans`](@ref) first.
"""
function build_plans(backend, state::VolPathState, framebuffer, refs, perrun::PerRun,
                     max_depth::Int32;
                     chit_owns_surface::Bool, has_media::Bool, has_lights::Bool)
    dev = mantle_device(backend)
    g = sample_graph(dev, state, refs, max_depth;
                     chit_owns_surface, has_media, has_lights)
    # Recorded here, once: `run!` submits the recording and never records.
    return VPPlans(framebuffer, perrun, Mantle.record!(Mantle.Plan(g)),
                   Mantle.record!(Mantle.Plan(finalize_graph(dev, state, framebuffer))))
end

"""Every plan of a sample, in the order a sample runs them."""
allplans(p::VPPlans) = (p.sample, p.finalize)

# On recording, which `build_plans` above can do up front.
#
# Recording writes the command buffers once; a run submits them. The host saving
# is real — a round's recording is 0.101 ms and submitting it 0.0058 ms — and it
# used to be swallowed whole by how a replay reached the queue. `replay!`
# submitted on its own, behind a semaphore wait on the previous replay, so
# replays were serialised against each other: an ordering that one recording
# expresses with an intra-submission barrier became a GPU round-trip between
# submissions. Measured then, paired and interleaved in one session: at a chunk
# of 8 rounds baking COST 5.2 % on medium_null, and only at a chunk of 64 — the
# whole sample in one plan, hence one replay — did it win 14.5 %.
#
# That is gone. A recording is appended to the batch the host is already building
# and leaves in the same `vkQueueSubmit2`, so there is no wait and no extra
# submission. Re-measured 2026-08-31 on an RTX 4000 Ada, every plan below baked,
# paired in one session against the same integrator:
#
#     materials  1200x900   10 spp  depth 50    0.744 s -> 0.717 s   1.037x
#     crown      1000x1400  16 spp  depth 100   2.226 s -> 2.130 s   1.045x
#
# Both bit identical to the interpreted render, at every sample count from 1 to 8
# and at 10 and 16. So the chunk stays at 8 — the early exit is worth far more
# than either number.
#
# Four things had to be right before any of this could be compared, and each was
# wrong at some point, so each is worth knowing: `bake!` used to RUN the plan as
# it captured it (the accumulate pass would contribute a spurious sample); a
# baked plan replayed the arguments it captured unless the caller remembered
# `Mantle.rebind!`; `rebind!` could not reach a `custom!` pass's arguments, which
# is what the hardware RT trace used to be; and a recording belongs to the
# argument slot it was captured in, which `bake!` got wrong for any plan that had
# already run. All four are pinned by tests in Mantle and all four are
# unreachable now — `run!` records for itself, `custom!` is gone, and the only
# value a run feeds is a `GPURef` the commands hold the address of, so there is
# no rebinding and no ring to be out of phase with.


"""
    rebuild_plans!(state, film, backend, scene parts..., per-render values...)

Build this render's plans fresh, retiring the old ones. Reached from
`rebuild_render_state!`, which runs on the first sample and on invalidation
events — never on the reuse path, so there is nothing to check here. What this
decided per sample by comparing, events decide now: `sync!(scene)` and the
scene mutators for the scene, [`invalidate!`](@ref) for the integrator's own
fields; the film is a call argument, and the caller reads its identity once
per sample.
"""
function rebuild_plans!(state::VolPathState, film::Film, backend,
                        accel, media_interfaces, media, materials, lights,
                        camera, initial_medium::SetKey, filter_params, filter_sampler,
                        regularize::Bool, samples_per_pixel::Int32,
                        max_component_value::Float32, max_depth::Int32;
                        chit_owns_surface::Bool, has_media::Bool, has_lights::Bool)
    # The old plans go back without a wait. Their recordings may well still be in
    # flight; `Mantle.free!` retires the regions rather than releasing them, so
    # the pool hands those bytes on only once the device says so. This used to be
    # `KA.synchronize(backend)` first.
    state.plans === nothing || free!(state.plans)
    dev = mantle_device(backend)
    # One device slot per per-run value, addressed by the recorded commands and
    # stored through when a value changes (`storechanged!`) — the plan itself
    # is never touched. The sample counter starts where the film's count is,
    # so plans rebuilt mid-progression continue the sequence rather than
    # replaying sample one.
    perrun = PerRun(dev, camera, initial_medium, filter_params, film.iteration_index[])
    refs = render_refs(accel, media_interfaces, media, materials, lights,
                       perrun.camera, perrun.initial_medium, perrun.filter_params,
                       filter_sampler, perrun.sample_idx,
                       regularize, samples_per_pixel, max_component_value)
    state.plans = build_plans(backend, state, film.framebuffer, refs, perrun, max_depth;
                              chit_owns_surface, has_media, has_lights)
    return nothing
end

"""Give every plan's pool regions back, and with them the per-run refs only the
plans read. Explicit, like every other release here — see the `sync!`/`free!`
contract."""
function free!(plans::VPPlans)
    foreach(Mantle.free!, allplans(plans))
    # The per-run refs go with them: the plans are the only readers of them,
    # and their regions are retired rather than released, so a recording still
    # in flight keeps the bytes until the device says otherwise.
    pr = plans.perrun
    Mantle.free!(pr.sample_idx)
    Mantle.free!(pr.camera)
    Mantle.free!(pr.initial_medium)
    Mantle.free!(pr.filter_params)
    nothing
end
