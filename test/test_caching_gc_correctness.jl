# Hikari Caching, GC, and Correctness Tests
#
# Tests VolPath integrator caching (adapted scene, filter sampler, initial medium),
# VolPathState allocation/free lifecycle, WorkQueue GPU operations on Lava,
# close() cleanup, and rendering correctness across multiple renders.

using Test
using Hikari
using Lava, Mantle
using Mantle
using KernelAbstractions
import KernelAbstractions as KA
using GeometryBasics
using GeometryBasics: normal_mesh, Tesselation
using LinearAlgebra

# Helper: create a minimal scene for testing
function _make_test_scene()
    scene = Hikari.Scene()

    # Floor
    floor_mat = Hikari.Diffuse(Kd=Hikari.RGBSpectrum(0.7f0, 0.7f0, 0.7f0))
    floor_mesh = normal_mesh(Rect3f(Vec3f(-2, -2, -0.01), Vec3f(4, 4, 0.01)))
    push!(scene, floor_mesh, floor_mat)

    # Sphere
    sphere_mat = Hikari.Diffuse(Kd=(0.8, 0.2, 0.2))
    sphere_mesh = normal_mesh(Tesselation(Sphere(Point3f(0, 0, 0.5), 0.5f0), 16))
    push!(scene, sphere_mesh, sphere_mat)

    # Light
    push!(scene, Hikari.PointLight(Point3f(0f0, 0f0, 3f0), Hikari.RGBSpectrum(20f0)))

    Hikari.sync!(scene)
    return scene
end

function _make_test_camera_film(; res=32)
    resolution = Point2f(res, res)
    film = Hikari.Film(resolution)
    camera = Hikari.PerspectiveCamera(
        Point3f(0f0, -3f0, 1f0), Point3f(0f0, 0f0, 0.5f0), film; fov=40f0,
    )
    return camera, film
end

@testset "Hikari Caching, GC & Correctness" begin

    # ── 1. VolPath close() frees all state ──
    @testset "VolPath close() lifecycle" begin
        @testset "close() clears all caches" begin
            vp = Hikari.VolPath(samples=1, max_depth=2)

            scene = _make_test_scene()
            camera, film = _make_test_camera_film()
            Hikari.clear!(film)

            # Render once to populate all caches
            vp(scene, film, camera)

            @test vp.state !== nothing
            # filter_sampler_gpu may be nothing for Gaussian filter with default sampler

            # Close should clear everything
            close(vp)

            @test vp.state === nothing
            @test vp.initial_medium_camera_pos === nothing
            @test vp.initial_medium_key === nothing
            @test vp.filter_sampler_gpu === nothing
        end

        @testset "close() is idempotent" begin
            vp = Hikari.VolPath(samples=1, max_depth=2)
            close(vp)  # No state yet
            close(vp)  # Should not error
            @test vp.state === nothing
        end
    end

    # ── 2. VolPathState allocation and free ──
    @testset "VolPathState allocation/free" begin
        backend = Mantle.defaultbackend()

        # NOT a count of `Mantle.live_buffer_count()` before and after with a
        # tolerance of ten either side, which is what `finalize` deferring to
        # the GC forces and
        # the GC runs when it likes. Nothing here finalizes any more: the state
        # holds one `DeviceMemory`, every allocation is a region of it, and
        # `free!` gives them all back at a point the caller chose. So the
        # assertion can be exact, and it is about the pool rather than about how
        # many `VkBuffer`s the pool happened to need.
        @testset "state takes its memory from the pool and gives all of it back" begin
            GC.gc(true)
            Mantle.flush!(Mantle.Device())
            Mantle.drain!(Mantle.batchqueue(Mantle.Device()))
            pool = Mantle.pool(Hikari.mantle_device(backend))

            scene = _make_test_scene()
            state = Hikari.VolPathState(
                backend, 16, 16, scene.lights;
                max_depth=4, samples_per_pixel=1
            )
            Mantle.flush!(Mantle.Device())
            @test Hikari.nallocations(state.memory) > 10   # queues, accumulators, BVH, Sobol
            reserved = Mantle.reserved(pool)

            Hikari.free!(state)
            Mantle.flush!(Mantle.Device())
            @test Hikari.nallocations(state.memory) == 0
            @test Hikari.nallocations(state.per_material_memory) == 0

            # And the regions really are back: more identical states cost the
            # device nothing, which a leak could not do.
            #
            # `live_buffer_count` is checked alongside the pool because the two
            # catch different mistakes. The pool number stays flat if an
            # allocation never reached the pool at all; Lava's count is what
            # notices a `KA.allocate` that crept back into the constructor and
            # was never given back. Everything the state owns — the queues, the
            # accumulators, the light BVH, the Sobol matrices and both spectral
            # tables — has to go through `DeviceMemory` for both to hold.
            buffers = Mantle.live_buffer_count()
            for _ in 1:3
                st = Hikari.VolPathState(
                    backend, 16, 16, scene.lights;
                    max_depth=4, samples_per_pixel=1
                )
                Mantle.flush!(Mantle.Device())
                Hikari.free!(st)
            end
            Mantle.flush!(Mantle.Device())
            GC.gc(true)
            Mantle.drain!(Mantle.batchqueue(Mantle.Device()))
            @test Mantle.reserved(pool) == reserved
            @test Mantle.live_buffer_count() == buffers
        end

        # The reason `DeviceMemory` carries a finalizer at all. `free!` is still
        # how memory goes back, and every path in this package calls it — this
        # is about the path that does NOT, which before had no way back at all:
        # an `Adapt`-allocated film was reclaimed by the GC, and a pooled one
        # would simply have been lost. The finalizer retires; `reclaim!`
        # releases, from the owning thread, a submission boundary later.
        @testset "a film nobody freed is reclaimed, not lost" begin
            backend = Mantle.defaultbackend()
            dev = Hikari.mantle_device(backend)
            pool = Mantle.pool(dev)

            "Build a device film, drop it on the floor, and let the pool catch up."
            function churn!()
                Hikari.Film(backend, Hikari.Film(Point2f(128, 128)))
                nothing                      # no free!, no reference kept
            end
            "Run the finalizers, then let the pool catch up with the device."
            function settle!()
                GC.gc(true)                  # finalizers run, regions retire
                KA.fill!(KA.allocate(backend, Float32, 4), 1f0)
                Mantle.flush!(Mantle.Device())
                KA.synchronize(backend)      # so the fences they were stamped with pass
                while Mantle.reclaim!(pool, dev; wait = true) > 0 end
                return nothing
            end

            churn!(); settle!()
            reserved = Mantle.reserved(pool)
            for _ in 1:5
                churn!()
                settle!()
            end
            # The pool never had to ask the device for more, across five films
            # built and dropped without a `free!` between them. That is the whole
            # claim: without the finalizer each one's regions stay allocated and
            # this grows by a film per iteration. Counting what `reclaim!`
            # returns would NOT show it — `acquire!` reclaims before it grows, so
            # the next film's allocation collects the previous one's bytes and
            # the explicit call finds nothing left to do.
            @test Mantle.reserved(pool) == reserved
        end

        @testset "double free! is safe" begin
            scene = _make_test_scene()
            state = Hikari.VolPathState(
                backend, 8, 8, scene.lights;
                max_depth=2, samples_per_pixel=1
            )
            Mantle.flush!(Mantle.Device())
            Hikari.free!(state)
            # Second free is a no-op: `free!` empties the owned list, so there is
            # nothing left to hand back twice.
            Hikari.free!(state)
            Mantle.flush!(Mantle.Device())
            Mantle.drain!(Mantle.batchqueue(Mantle.Device()))
        end
    end

    # ── 3. WorkQueue on Lava backend ──
    @testset "WorkQueue on LavaBackend" begin
        backend = Mantle.defaultbackend()

        @testset "push and read on GPU" begin
            mem = Hikari.DeviceMemory(backend)
            queue = Hikari.WorkQueue{Int32}(mem, 256)

            @kernel function push_items!(queue)
                i = @index(Global)
                push!(queue, Int32(i * 10))
            end

            push_items!(backend)(queue; ndrange=8)
            Mantle.flush!(Mantle.Device())

            @test length(queue) == 8
            items = sort(Array(queue.items)[1:8])
            @test items == Int32[10, 20, 30, 40, 50, 60, 70, 80]

            Mantle.flush!(Mantle.Device())
            Hikari.free!(mem)
        end

        @testset "empty and reuse" begin
            mem = Hikari.DeviceMemory(backend)
            queue = Hikari.WorkQueue{Int32}(mem, 64)

            @kernel function push_val!(queue, val)
                i = @index(Global)
                push!(queue, val)
            end

            push_val!(backend)(queue, Int32(42); ndrange=10)
            Mantle.flush!(Mantle.Device())
            @test length(queue) == 10

            empty!(queue)
            @test length(queue) == 0

            push_val!(backend)(queue, Int32(99); ndrange=5)
            Mantle.flush!(Mantle.Device())
            @test length(queue) == 5

            Mantle.flush!(Mantle.Device())
            Hikari.free!(mem)
        end

        # A queue's arrays are regions of the state's `DeviceMemory`, so
        # "freed" means back in the pool, not back to the driver. That is the
        # whole point of the pool and it is what makes the assertion below
        # stronger than `live_buffer_count() == baseline`: the second round of
        # queues has to cost the device NOTHING,
        # which a driver-level count could satisfy while the pool quietly grew
        # a second block.
        @testset "freed queues come back from the pool, not the device" begin
            GC.gc(true)
            Mantle.flush!(Mantle.Device())
            Mantle.drain!(Mantle.batchqueue(Mantle.Device()))
            pool = Mantle.pool(Hikari.mantle_device(backend))

            mem = Hikari.DeviceMemory(backend)
            qs = [Hikari.WorkQueue{Int32}(mem, 1 << 16) for _ in 1:4]
            @test Hikari.nallocations(mem) == 8          # items + counter each
            Mantle.flush!(Mantle.Device())
            after_first = Mantle.reserved(pool)

            Hikari.free!(mem)
            @test Hikari.nallocations(mem) == 0
            @test Mantle.reserved(pool) == after_first   # nothing given back yet

            mem2 = Hikari.DeviceMemory(backend)
            qs2 = [Hikari.WorkQueue{Int32}(mem2, 1 << 16) for _ in 1:4]
            Mantle.flush!(Mantle.Device())
            @test Mantle.reserved(pool) == after_first   # reused, not reallocated
            Hikari.free!(mem2)
        end
    end

    # ── 4. Filter sampler caching ──
    @testset "filter sampler cache" begin
        @testset "filter sampler cached on struct" begin
            # LanczosSinc filter produces GPUFilterSamplerData that gets cached
            vp = Hikari.VolPath(samples=1, max_depth=2,
                                filter=Hikari.LanczosSincFilter(Point2f(1.0f0), 3.0f0))

            @test vp.filter_sampler_gpu === nothing

            scene = _make_test_scene()
            camera, film = _make_test_camera_film()
            Hikari.clear!(film)
            vp(scene, film, camera)

            # After render, should be cached
            # Filter sampler data should be cached from film after render
            @test vp.filter_sampler_gpu !== nothing

            close(vp)
        end
    end

    # ── 6. State reuse across samples ──
    @testset "state reuse across samples" begin
        @testset "state persists between render! calls" begin
            vp = Hikari.VolPath(samples=2, max_depth=2)
            scene = _make_test_scene()
            camera, film = _make_test_camera_film()
            Hikari.clear!(film)

            # First render populates state
            vp(scene, film, camera)
            state_after_first = vp.state
            @test state_after_first !== nothing

            # Clear and render again — state should be reused (same dimensions)
            Hikari.clear!(film)
            vp(scene, film, camera)
            @test vp.state === state_after_first

            close(vp)
        end
    end

    # ── 7. Rendering correctness ──
    @testset "rendering correctness" begin
        @testset "output has valid pixel values" begin
            vp = Hikari.VolPath(samples=4, max_depth=4)
            scene = _make_test_scene()
            camera, film = _make_test_camera_film(; res=32)
            Hikari.clear!(film)

            vp(scene, film, camera)

            img = Hikari.postprocess!(film; exposure=1.0f0, tonemap=:aces, gamma=2.2f0)
            img_arr = Array(img)

            @test size(img_arr) == (32, 32)

            # No NaN or Inf
            @test !any(px -> isnan(px.r) || isnan(px.g) || isnan(px.b), img_arr)
            @test !any(px -> isinf(px.r) || isinf(px.g) || isinf(px.b), img_arr)

            # Something was rendered (not all black)
            mean_val = sum(px -> (px.r + px.g + px.b) / 3f0, img_arr) / length(img_arr)
            @test mean_val > 0.001f0

            close(vp)
        end

        @testset "deterministic across runs with same seed" begin
            scene = _make_test_scene()

            vp1 = Hikari.VolPath(samples=2, max_depth=4)
            camera1, film1 = _make_test_camera_film(; res=16)
            Hikari.clear!(film1)
            vp1(scene, film1, camera1)
            img1 = Array(Hikari.postprocess!(film1; exposure=1.0f0, tonemap=nothing, gamma=1.0f0))

            close(vp1)

            vp2 = Hikari.VolPath(samples=2, max_depth=4)
            camera2, film2 = _make_test_camera_film(; res=16)
            Hikari.clear!(film2)
            vp2(scene, film2, camera2)
            img2 = Array(Hikari.postprocess!(film2; exposure=1.0f0, tonemap=nothing, gamma=1.0f0))

            close(vp2)

            # Same scene + camera + spp should produce identical results
            # (SobolRNG uses deterministic seed 0 by default)
            @test img1 == img2
        end

        @testset "more samples produces non-zero output" begin
            scene = _make_test_scene()

            vp = Hikari.VolPath(samples=4, max_depth=4)
            camera, film = _make_test_camera_film(; res=16)
            Hikari.clear!(film)
            vp(scene, film, camera)
            img = Array(Hikari.postprocess!(film; exposure=1.0f0, tonemap=nothing, gamma=1.0f0))
            close(vp)

            # 4 samples should produce a reasonable image
            mean_val = sum(px -> Float64(px.r + px.g + px.b) / 3.0, img) / length(img)
            @test mean_val > 0.001
        end
    end

    # ── 8. Memory stability across repeated renders ──
    @testset "memory stability" begin
        @testset "buffer count stable across renders" begin
            GC.gc(true)
            Mantle.flush!(Mantle.Device())
            Mantle.drain!(Mantle.batchqueue(Mantle.Device()))

            scene = _make_test_scene()
            vp = Hikari.VolPath(samples=1, max_depth=2)
            camera, film = _make_test_camera_film()

            # Warm-up render
            Hikari.clear!(film)
            vp(scene, film, camera)
            Mantle.flush!(Mantle.Device())
            GC.gc(true)
            Mantle.drain!(Mantle.batchqueue(Mantle.Device()))
            baseline = Mantle.live_buffer_count()
            pool = Mantle.pool(Hikari.mantle_device(Mantle.defaultbackend()))
            reserved = Mantle.reserved(pool)
            owned = Hikari.nallocations(vp.state.memory)

            # Multiple renders — buffer count should not grow
            for _ in 1:5
                Hikari.clear!(film)
                vp(scene, film, camera)
                Mantle.flush!(Mantle.Device())
            end
            GC.gc(true)
            Mantle.drain!(Mantle.batchqueue(Mantle.Device()))
            after = Mantle.live_buffer_count()

            @test after == baseline
            # The same claim one level up, where the state's memory actually
            # lives now: a render neither takes a new region nor makes the pool
            # ask the device for more.
            @test Hikari.nallocations(vp.state.memory) == owned
            @test Mantle.reserved(pool) == reserved

            close(vp)
            Mantle.flush!(Mantle.Device())
            Mantle.drain!(Mantle.batchqueue(Mantle.Device()))
        end
    end

    # ── 9. LavaArray resize! does not leak ──
    @testset "resize! does not leak buffers" begin
        GC.gc(true)
        Mantle.flush!(Mantle.Device())
        Mantle.drain!(Mantle.batchqueue(Mantle.Device()))
        baseline = Mantle.live_buffer_count()

        # Pre-pool-block test bookkeeping: a tiny LavaArray comes out of an
        # existing 64-MiB pool block (no new `VkManagedBuffer`) — only allocs
        # larger than `Mantle.POOL_LARGE_THRESHOLD` (= POOL_BLOCK_SIZE = 64 MiB)
        # bypass the pool. So `live_buffer_count` may stay flat or grow by at most
        # one (if the pool runs out and a new block is allocated). The test
        # invariant is "no leak": the count must NEVER grow past `baseline +
        # 1` no matter how many resize!s we do.

        a = Mantle.LavaArray{Int32}(undef, 10)
        Mantle.flush!(Mantle.Device())
        after_alloc = Mantle.live_buffer_count()
        @test after_alloc <= baseline + 1

        resize!(a, 100)
        Mantle.flush!(Mantle.Device())
        Mantle.drain!(Mantle.batchqueue(Mantle.Device()))
        after_resize = Mantle.live_buffer_count()
        @test after_resize <= baseline + 1

        # Multiple resizes should not accumulate
        for sz in [200, 50, 500, 10]
            resize!(a, sz)
        end
        Mantle.flush!(Mantle.Device())
        Mantle.drain!(Mantle.batchqueue(Mantle.Device()))
        after_multi = Mantle.live_buffer_count()
        @test after_multi <= baseline + 1

        # Free the array itself
        finalize(a)
        Mantle.flush!(Mantle.Device())
        Mantle.drain!(Mantle.batchqueue(Mantle.Device()))
        after_free = Mantle.live_buffer_count()
        @test after_free <= baseline + 1
    end

    # ── 10. Film lifecycle ──
    @testset "Film lifecycle" begin
        @testset "film clear resets iteration index" begin
            film = Hikari.Film(Point2f(16, 16))
            film.iteration_index[] = Int32(42)
            Hikari.clear!(film)
            @test film.iteration_index[] == Int32(0)
        end

        @testset "film free releases memory" begin
            film = Hikari.Film(Point2f(16, 16))
            Hikari.free!(film)
            # Should not crash on double free
            Hikari.free!(film)
        end
    end
end
