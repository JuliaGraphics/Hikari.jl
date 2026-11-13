# Hikari Hardware RT Integration
#
# VolPath-specific dispatch overrides for hardware-accelerated ray tracing.
# Uses only Raycore types and stubs -- no backend (Lava/Metal) imports.
# Backend extensions in Raycore implement the actual dispatch.

# `Mantle`, not `Lava`. The Vulkan runtime — acceleration structures, the
# backend, the device — moved into Mantle on 2026-08-27; Lava is the SPIR-V
# compiler and has none of these names any more.
import Mantle
import Mantle: HWTLAS, AdaptedAccel

# Any backend with hw_accel=true creates an HWTLAS.  Parametrised on
# `Raycore.Triangle{TriangleMeta}` because Hikari's scene API pushes meshes
# with `TriangleMeta` per-face data — `Mantle.HWTLAS(backend)`'s default
# narrowing to `Triangle{UInt32}` would reject those pushes with a convert
# error.
default_accel(backend, ::Val{true}) = HWTLAS{Raycore.Triangle{TriangleMeta}}(backend)

# Scene sync! for HWTLAS scenes
function sync!(scene::Scene{<:HWTLAS})
    # Same contract as the TLAS method: invalidate plans only when the sync!
    # actually rebuilds (`dirty`); a transforms-only refit writes the same
    # buffers in place and a recorded plan survives it.
    changed = scene.accel.dirty
    Raycore.sync!(scene.accel)
    bound = Raycore.world_bound(scene.accel)
    scene.bounds[] = (bound, bounding_sphere(bound))
    changed && notify_scene_changed(scene)
    return scene
end

# ============================================================================
# fill_aux_buffers! for HWTLAS: trace primary rays via HW RT for depth
# ============================================================================

# The guides through the hardware structure: the same kernel as the software one,
# whose `Raycore.closest_hit` lowers to an inline ray query here.
function fill_aux_buffers!(film::Film, scene::Scene{<:AdaptedAccel}, camera;
                           has_infinite_lights::Bool=false,
                           cull_mask::UInt32=UInt32(0xFF), grid::Integer=GUIDE_GRID)
    hwtlas = scene.accel.hwtlas
    hwtlas === nothing && return nothing  # Adapt has not run; nothing to trace
    Raycore.n_instances(hwtlas) == 0 && return nothing

    backend = KA.get_backend(film.depth)
    miss_depth = has_infinite_lights ? MISS_DEPTH : Inf32

    accel = Adapt.adapt(backend, hwtlas)  # CPU-form AdaptedAccel; LavaAdaptor
                                           # will strip hwtlas at kernel-arg time
    aux_buffer_kernel!(backend)(
        film.albedo, film.normal, film.depth,
        accel, camera, film.filter_params, film.filter_sampler, miss_depth, Int32(grid),
        scene.materials, scene.media_interfaces; ndrange = length(film.albedo))
    Mantle.waitidle(mantle_device(backend))
    return film
end

# ============================================================================
# VolPath Dispatch Overrides
# ============================================================================

function detect_initial_medium(backend, accel::AdaptedAccel, mi, pos, vp::VolPath)
    return Raycore.SetKey()
end

# Tracing needs no hardware override of its own: `Raycore.closest_hit_record`
# on an `AdaptedAccel` is Mantle's ray-query loop on every backend, so
# `vp_trace_kernel!` covers every traversal.

# ============================================================================
# Mantle device
# ============================================================================

"""
    mantle_device(backend) -> Mantle.Device

The Mantle device for a KernelAbstractions backend, which is what a graph
allocates transients and records passes against.

`Mantle.Device(VulkanAPI())` is cached one per `VkContext` and wraps that context's
existing `default_bq`, so this reuses the runtime's device rather than creating a
second one — a second would mean a second `Pool` over one `VkDevice`, which is
two allocators over the same memory.
"""
# `Mantle.Device(backend)` dispatches on the KernelAbstractions backend itself,
# so this needs no per-backend method and no backend import — which is what the
# header of this file promises.
mantle_device(backend) = Mantle.Device(backend)

# Smallest kernel that reads one buffer and writes another, so a test can pin
# that a Mantle pass reaches Hikari's own allocations. Lives here rather than in
# the test file because a `@kernel` has to be compiled into the package that
# dispatches it.
@kernel function mantle_probe_scale!(dst, @Const(src), a::Float32)
    i = @index(Global)
    @inbounds dst[i] = src[i] * a
end
