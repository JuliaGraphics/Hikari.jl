# Hikari Hardware RT Integration
#
# VolPath-specific dispatch overrides for hardware-accelerated ray tracing.
# Uses only Raycore types and stubs -- no backend (Lava/Metal) imports.
# Backend extensions in Raycore implement the actual dispatch.

# The bare module too, not just the two names: `mantle_device` below needs
# `Lava.LavaBackend` and hands the module itself to `Mantle.Device`.
import Lava
import Lava: HWTLAS, HWAdaptedAccel

# Any backend with hw_accel=true creates an HWTLAS.  Parametrised on
# `Raycore.Triangle{TriangleMeta}` because Hikari's scene API pushes meshes
# with `TriangleMeta` per-face data — `Lava.HWTLAS(backend)`'s default
# narrowing to `Triangle{UInt32}` would reject those pushes with a convert
# error.
default_accel(backend, ::Val{true}) = HWTLAS{Raycore.Triangle{TriangleMeta}}(backend)

# Scene sync! for HWTLAS scenes
function sync!(scene::Scene{<:HWTLAS})
    Raycore.sync!(scene.accel)
    bound = Raycore.world_bound(scene.accel)
    scene.bounds[] = (bound, bounding_sphere(bound))
    return scene
end

# ============================================================================
# fill_aux_buffers! for HWTLAS: trace primary rays via HW RT for depth
# ============================================================================

# Fused primary-ray + closest-hit + depth/normal/albedo kernel.  One thread
# per pixel: generate the camera ray, run inline ray query via
# `Raycore.closest_hit(accel, ray)` (lowers to OpRayQueryInitializeKHR /
# Proceed / Get*KHR through the polymorphic dispatch we added in step 2),
# write the aux buffers.  Replaces the previous 3-step dance
# (generate_rays → trace_closest_hits! → extract_depth) plus the dependency
# on `aux_rays`/`aux_results` scratch buffers.
@kernel inbounds = true function hw_fill_aux_kernel!(
    depth, normal, albedo,
    @Const(camera), @Const(crop_p_min),
    @Const(miss_depth::Float32),
    @Const(width::Int32), @Const(height::Int32),
    accel,
)
    idx = @index(Global)
    if idx <= width * height
        h = height
        row = Int32(((idx - Int32(1)) % h) + Int32(1))
        col = Int32(((idx - Int32(1)) ÷ h) + Int32(1))
        px = Float32(col) + crop_p_min[1] - 1f0
        # Film rows run downward; camera raster y runs upward, as in
        # vp_generate_camera_rays_kernel!. Keep guide buffers aligned with
        # radiance or denoising leaves a vertically mirrored ghost silhouette.
        py = Float32(height - row) + crop_p_min[2]
        pixel = Point2f(px + 0.5f0, py + 0.5f0)
        cs = CameraSample(pixel, Point2f(0.5f0, 0.5f0), 0f0)
        ray, w = generate_ray(camera, cs)
        if w > 0f0
            hit, _prim, t, _bary, _icx = Raycore.closest_hit(accel, ray)
            if hit
                depth[idx] = t
                normal[idx] = Vec3f(0f0, 0f0, 1f0)
                albedo[idx] = RGB{Float32}(0.8f0, 0.8f0, 0.8f0)
            else
                depth[idx] = miss_depth
                normal[idx] = Vec3f(0f0, 0f0, 0f0)
                albedo[idx] = RGB{Float32}(0f0, 0f0, 0f0)
            end
        else
            depth[idx] = miss_depth
            normal[idx] = Vec3f(0f0, 0f0, 0f0)
            albedo[idx] = RGB{Float32}(0f0, 0f0, 0f0)
        end
    end
end

function fill_aux_buffers!(film::Film, scene::Scene{<:HWAdaptedAccel}, camera;
                           has_infinite_lights::Bool=false,
                           cull_mask::UInt32=UInt32(0xFF))
    hwtlas = scene.accel.hwtlas
    hwtlas === nothing && return nothing  # Adapt has not run; nothing to trace
    Raycore.n_instances(hwtlas) == 0 && return nothing

    backend = KA.get_backend(film.depth)
    h, w = size(film.depth)
    n = h * w
    miss_depth = has_infinite_lights ? Float32(1e30) : Inf32

    accel = Adapt.adapt(backend, hwtlas)  # CPU-form HWAdaptedAccel; LavaAdaptor
                                           # will strip hwtlas at kernel-arg time
    hw_fill_aux_kernel!(backend)(
        film.depth, film.normal, film.albedo,
        camera, film.crop_bounds.p_min,
        miss_depth, Int32(w), Int32(h),
        accel; ndrange=n)
    KA.synchronize(backend)
    return film
end

# ============================================================================
# VolPath Dispatch Overrides
# ============================================================================

function detect_initial_medium(backend, accel::HWAdaptedAccel, mi, pos, vp::VolPath)
    return Raycore.SetKey()
end

# The per-material closest-hit shaders shade a hit before the trace returns.
shades_surfaces_inline(::HWAdaptedAccel) = true

# Tracing needs no HW override. It used to: extract_rays → prepare_indirect →
# cmd_trace_rays_indirect_khr → process. Inline ray queries on
# `Raycore.closest_hit(::HWAdaptedAccel, ray)` collapse that to a single
# dispatch, so `vp_trace_and_shade!` covers both backends.

# ============================================================================
# Mantle device
# ============================================================================

"""
    mantle_device(backend) -> Mantle.Device

The Mantle device for a KernelAbstractions backend, which is what a graph
allocates transients and records passes against.

`Mantle.Device(Lava)` is cached one per `VkContext` and wraps that context's
existing `default_bq`, so this reuses Lava's device rather than creating a
second one — a second would mean a second `Pool` over one `VkDevice`, which is
two allocators over the same memory.
"""
mantle_device(::Lava.LavaBackend) = Mantle.Device(Lava)
mantle_device(::KA.CPU) = Mantle.Device(Mantle.Host())

# Smallest kernel that reads one buffer and writes another, so a test can pin
# that a Mantle pass reaches Hikari's own allocations. Lives here rather than in
# the test file because a `@kernel` has to be compiled into the package that
# dispatches it.
@kernel function mantle_probe_scale!(dst, @Const(src), a::Float32)
    i = @index(Global)
    @inbounds dst[i] = src[i] * a
end
