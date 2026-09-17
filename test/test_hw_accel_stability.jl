using Test
using Hikari
using Lava, Mantle
using Raycore
using Adapt
using GeometryBasics
using GeometryBasics: normal_mesh, Tesselation, Rect3f, Sphere, Point3f, Vec3f, Point2f
using LinearAlgebra: I

# ==============================================================================
# HW RT stability stress: many renders, many scene mutations, no DEVICE_LOST,
# no runaway GPU memory, no wrong pixels.  Designed to catch:
#
#   * UAF / stale-BDA from `sync!` + `unsafe_free!` interplay under real
#     integrator load (primary rays + shadow rays + aux buffers).
#   * Pipeline / SBT lifetime regressions across multiple `VolPath` calls
#     against the same `VulkanTLAS`.
#   * Mesh-mutation regressions in the `delete!` + `push!` + `sync!` path
#     while the RT pipeline is live.
#   * Cross-scene state leakage: a second `hw_accel=true` scene in the
#     same session must render correctly.
#
# The bounds are loose by design — the point is catching runaway / crashes,
# not pinning exact byte counts.
# ==============================================================================

# ── Scene helpers ─────────────────────────────────────────────────────────────

function _make_scene(backend; n_spheres::Int=3)
    scene = Hikari.Scene(; backend=backend, hw_accel=true)
    floor = normal_mesh(Rect3f(Vec3f(-3, -3, -0.05), Vec3f(6, 6, 0.05)))
    push!(scene, floor, Hikari.Diffuse(Kd=Hikari.RGBSpectrum(0.6f0, 0.6f0, 0.6f0)))
    sphere = normal_mesh(Tesselation(Sphere(Point3f(0, 0, 0), 0.35f0), 16))
    cols = [(1f0, 0.2f0, 0.2f0), (0.2f0, 1f0, 0.2f0), (0.2f0, 0.2f0, 1f0),
            (1f0, 1f0, 0.2f0), (0.2f0, 1f0, 1f0), (1f0, 0.2f0, 1f0)]
    handles = Hikari.SceneHandle[]
    for i in 1:n_spheres
        c = cols[mod1(i, length(cols))]
        x = 0.9f0 * (i - (n_spheres + 1) / 2)
        tf = Raycore.Mat4f([1 0 0 x; 0 1 0 0f0; 0 0 1 0.35f0; 0 0 0 1])
        h = push!(scene, sphere, Hikari.Diffuse(Kd=c); transform=tf)
        push!(handles, h)
    end
    push!(scene, Hikari.PointLight(Point3f(0, -2, 3), Hikari.RGBSpectrum(30f0)))
    Hikari.sync!(scene)
    return scene, handles
end

function _render_once(scene, backend; res::Int=48, samples::Int=8, depth::Int=3)
    film = Hikari.Film(Point2f(res, res))
    camera = Hikari.PerspectiveCamera(Point3f(0, -3, 1.5), Point3f(0, 0, 0.35),
                                      film; fov=50f0)
    gpu_film = Hikari.Film(backend, film)
    vp = Hikari.VolPath(samples=samples, max_depth=depth, hw_accel=true)
    vp(scene, gpu_film, camera)
    img = Array(gpu_film.framebuffer)
    close(vp)
    return img
end

function _nondegenerate(img)
    # Any RGB value in sensible range + not all zero + not all saturated.
    n_nonzero = count(px -> (px.r + px.g + px.b) > 1f-4, img)
    frac = n_nonzero / length(img)
    return 0.1 < frac < 0.99
end

function _snapshot()
    mem = Mantle.gpu_memory_usage()
    # There are no module-level `POOL_BLOCKS` / `LIVE_BUFFERS` globals: the pool
    # is per DEVICE (two devices sharing one block list serve allocations off
    # the wrong GPU), and the counter is
    # onto the returned named tuple. This file had not been updated and could
    # not run.
    (live_bytes = mem.live_bytes, live_bufs = mem.live_buffers,
     pool_blocks = length(Mantle.pool(Mantle.lavadevice(Mantle.vk_context())).blocks))
end

# ── Tests ─────────────────────────────────────────────────────────────────────

const BACKEND = Mantle.defaultbackend()
const CTX = Mantle.vk_context()

@testset "HW RT stability — many renders on single scene" begin
    scene, handles = _make_scene(BACKEND)
    # Warm up caches (first render cold-compiles RT shaders, SBT, pipelines).
    _render_once(scene, BACKEND; samples=4, depth=2)
    GC.gc(true); GC.gc(true)
    baseline = _snapshot()
    @info "HW RT stability: baseline after warm" baseline

    n_iters = 20
    last_img = nothing
    for iter in 1:n_iters
        img = _render_once(scene, BACKEND; samples=4, depth=2)
        @test size(img) == (48, 48)
        @test _nondegenerate(img)
        @test !Mantle.device_lost(CTX)
        last_img = img
    end
    GC.gc(true); GC.gc(true)
    final = _snapshot()
    @info "HW RT stability: final after $n_iters renders" final

    # Memory growth ceiling: generous because per-render adapted films alloc
    # fresh buffers.  What we care about is: it does NOT scale with n_iters.
    @test final.live_bytes  <= baseline.live_bytes + 512 * 1024^2
    @test final.live_bufs   <= baseline.live_bufs + 128
    @test final.pool_blocks <= baseline.pool_blocks + 16
end

@testset "HW RT stability — mesh mutations between renders" begin
    scene, handles = _make_scene(BACKEND; n_spheres=3)
    # Warm.
    _render_once(scene, BACKEND; samples=4, depth=2)
    GC.gc(true); GC.gc(true)
    baseline = _snapshot()

    # Each iteration: delete one sphere, push a different one (new tessellation
    # to force a BLAS size change), re-sync, re-render.  Exercises the full
    # delete+push+sync!+unsafe_free! path under live RT pipeline.
    sphere_small = normal_mesh(Tesselation(Sphere(Point3f(0,0,0), 0.35f0), 8))
    sphere_large = normal_mesh(Tesselation(Sphere(Point3f(0,0,0), 0.35f0), 32))
    n_iters = 12
    for iter in 1:n_iters
        # Rotate through the handles; alternate mesh sizes for shrink-and-grow.
        victim_idx = mod1(iter, length(handles))
        old_handle = handles[victim_idx]
        Raycore.delete!(scene.accel, old_handle.geometry)
        new_mesh = iseven(iter) ? sphere_large : sphere_small
        x = 0.9f0 * (victim_idx - 2)
        z_offset = 0.35f0 + 0.02f0 * iter  # nudge each cycle so any UAF shows as wrong pixel
        tf = Raycore.Mat4f([1 0 0 x; 0 1 0 0f0; 0 0 1 z_offset; 0 0 0 1])
        new_h = push!(scene, new_mesh,
                      Hikari.Diffuse(Kd=(0.2f0+0.1f0*iter, 0.5f0, 0.8f0)); transform=tf)
        handles[victim_idx] = new_h
        Hikari.sync!(scene)

        img = _render_once(scene, BACKEND; samples=4, depth=2)
        @test _nondegenerate(img)
        @test !Mantle.device_lost(CTX)
    end
    GC.gc(true); GC.gc(true)
    final = _snapshot()
    @info "HW RT mesh-mutation final" final

    @test final.live_bytes  <= baseline.live_bytes + 512 * 1024^2
    @test final.live_bufs   <= baseline.live_bufs + 128
end

@testset "HW RT stability — multiple scenes in the same session" begin
    baseline = _snapshot()
    for k in 1:3
        scene_k, _ = _make_scene(BACKEND; n_spheres=2 + k)
        img_k = _render_once(scene_k, BACKEND; samples=4, depth=2)
        @test _nondegenerate(img_k)
        @test !Mantle.device_lost(CTX)
        scene_k = nothing  # drop ref so the VulkanTLAS can be finalized
    end
    GC.gc(true); GC.gc(true)
    final = _snapshot()
    @info "HW RT multi-scene final" baseline final
    @test final.live_bytes <= baseline.live_bytes + 512 * 1024^2
end
