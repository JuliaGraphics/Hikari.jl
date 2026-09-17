# TLAS Per-Instance Interface Override — Stress Tests
#
# Exercises the Phase-C refactor across both SW (StaticTLAS) and HW RT
# (VulkanTLAS → PrecomputedHitsAccel) paths:
#
#   - `push!(scene, mesh, materials::Vector, transforms::Vector)` builds
#     ONE BLAS + N instances; each instance's material routes through
#     `InstanceDescriptor.instance_id` and resolves at hit time via
#     `resolve_mi_idx`.
#   - `Raycore.InstanceDescriptor.instance_id` semantics: 0 = inherit
#     from triangle metadata; nonzero = override.
#   - Rapid push/delete cycles must not leak BLASes (dolphin meshscatter
#     was ~1 GB/frame before the fix).
#   - Transform updates must preserve `instance_id`.

using Test
using Hikari
using Lava, Mantle
using Raycore
using KernelAbstractions
using GeometryBasics
using GeometryBasics: normal_mesh, Tesselation, Rect3f, Sphere
using Adapt
using LinearAlgebra

# ── Helpers ─────────────────────────────────────────────────────────────────

_cube_mesh(cx=0, cy=0, cz=0) = normal_mesh(Rect3f(GeometryBasics.Vec3f(cx-0.3, cy-0.3, cz-0.3),
                                                    GeometryBasics.Vec3f(0.6, 0.6, 0.6)))

_floor_mesh() = normal_mesh(Rect3f(GeometryBasics.Vec3f(-5, -5, -0.01),
                                     GeometryBasics.Vec3f(10, 10, 0.01)))

_shift_mat(x, y, z=0) = Raycore.Mat4f([1 0 0 Float32(x); 0 1 0 Float32(y); 0 0 1 Float32(z); 0 0 0 1])

function _make_film_camera(res=48)
    film = Hikari.Film(GeometryBasics.Point2f(res, res))
    camera = Hikari.PerspectiveCamera(
        GeometryBasics.Point3f(0, -4, 2),
        GeometryBasics.Point3f(0, 0, 0),
        film; fov=45f0,
    )
    return film, camera
end

_colors_of(materials, idx) = Raycore.is_valid(idx) ? materials[idx].Kd : nothing

function _render(scene, film, camera; backend, hw::Bool, samples::Int=2, depth::Int=3)
    gpu_film = Hikari.Film(backend, film)
    vp = Hikari.VolPath(samples=samples, max_depth=depth, hw_accel=hw)
    vp(scene, gpu_film, camera)
    close(vp)
    return Array(gpu_film.framebuffer)
end

# Max channel value per pixel → tells us which colored sphere dominated
_channel_maxes(img) = (
    r = maximum(p.r for p in img),
    g = maximum(p.g for p in img),
    b = maximum(p.b for p in img),
)

# ── Test Suite ──────────────────────────────────────────────────────────────

@testset "TLAS instance override — both SW and HW paths" begin

    # ── 1. InstanceDescriptor.instance_id semantics ─────────────────────────
    @testset "InstanceDescriptor.instance_id round-trip" begin
        cube = _cube_mesh()
        tlas = Raycore.TLAS(CPU())

        # Default: no override (0 = inherit)
        h1 = push!(tlas, cube)
        @test Array(tlas.instances)[end].instance_id == UInt32(0)

        # Single-arg form with explicit override
        h2 = push!(tlas, cube, Raycore.Mat4f(I); instance_id=UInt32(42))
        @test Array(tlas.instances)[end].instance_id == UInt32(42)

        # Multi-instance form with per-instance overrides
        h3 = push!(tlas, cube, [Raycore.Mat4f(I), _shift_mat(2,0)];
                   instance_ids=UInt32[7, 9])
        descs = Array(tlas.instances)
        @test descs[end-1].instance_id == UInt32(7)
        @test descs[end].instance_id == UInt32(9)

        # Multi-instance without override → all 0 (inherit)
        h4 = push!(tlas, cube, [Raycore.Mat4f(I), _shift_mat(3,0)])
        descs = Array(tlas.instances)
        @test descs[end-1].instance_id == UInt32(0)
        @test descs[end].instance_id == UInt32(0)

        # instance_ids length mismatch → throws
        @test_throws ArgumentError push!(tlas, cube, [Raycore.Mat4f(I), _shift_mat(4,0)];
                                          instance_ids=UInt32[1])
    end

    # ── 2. closest_hit forwards the override ────────────────────────────────
    @testset "closest_hit returns 1-based instance array index" begin
        cube = _cube_mesh()
        tlas = Raycore.TLAS(CPU())
        h1 = push!(tlas, cube)                             # idx 1, override 0
        h2 = push!(tlas, cube, _shift_mat(2,0); instance_id=UInt32(99))  # idx 2, override 99
        Raycore.sync!(tlas)
        static = Adapt.adapt(CPU(), tlas)

        ray1 = Raycore.Ray(Raycore.Point3f(0, 0, 2), Raycore.Vec3f(0, 0, -1), 0f0, 100f0, 0f0)
        hit, _, _, _, idx1 = Raycore.closest_hit(static, ray1)
        @test hit
        @test idx1 == UInt32(1)
        @test static.instances[idx1].instance_id == UInt32(0)  # inherit

        ray2 = Raycore.Ray(Raycore.Point3f(2, 0, 2), Raycore.Vec3f(0, 0, -1), 0f0, 100f0, 0f0)
        hit, _, _, _, idx2 = Raycore.closest_hit(static, ray2)
        @test hit
        @test idx2 == UInt32(2)
        @test static.instances[idx2].instance_id == UInt32(99)
    end

    # ── 3. Meshscatter-style push! builds 1 BLAS + N instances ──────────────
    @testset "push!(scene, mesh, materials, transforms) — 1 BLAS + N instances" begin
        scene = Hikari.Scene()
        push!(scene, _floor_mesh(),
              Hikari.Diffuse(Kd=Hikari.RGBSpectrum(0.5f0, 0.5f0, 0.5f0)))
        sphere = normal_mesh(Tesselation(Sphere(GeometryBasics.Point3f(0,0,0), 0.25f0), 12))
        colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0)]
        materials = Hikari.Material[Hikari.Diffuse(Kd=c) for c in colors]
        transforms = Raycore.Mat4f[_shift_mat(-1.5, 0), _shift_mat(-0.5, 0),
                                    _shift_mat(0.5, 0),  _shift_mat(1.5, 0)]

        handles = push!(scene, sphere, materials, transforms)

        # Exactly 2 BLASes (floor + sphere) regardless of instance count
        @test length(scene.accel.blas_storage) == 2
        # 5 instances: 1 floor + 4 scatter
        @test Raycore.n_total_instances(scene.accel) == 5
        # All 4 sphere instances share BLAS 2
        sphere_descs = Array(scene.accel.instances)[2:end]
        @test all(d.blas_index == UInt32(2) for d in sphere_descs)
        # Each has a distinct nonzero instance_id (= media_interfaces idx)
        @test length(Set(d.instance_id for d in sphere_descs)) == 4
        @test all(d.instance_id != UInt32(0) for d in sphere_descs)
        # One SceneHandle per instance (4 returned)
        @test length(handles) == 4

        # Emissive materials rejected (clear boundary — emission needs per-
        # instance area lights which isn't implemented yet)
        em_mat = Hikari.Emissive(Le=Hikari.RGBSpectrum(1f0, 1f0, 1f0), scale=1f0, two_sided=false)
        @test_throws ArgumentError push!(scene, sphere,
                                          Hikari.Material[em_mat, materials[1]],
                                          transforms[1:2])
    end

    # ── 4. SW-path render: each instance shows its own color ────────────────
    @testset "SW render: per-instance materials produce distinct colors" begin
        backend = Mantle.defaultbackend()
        # Scene's TLAS lives on the same backend the render dispatches on —
        # cross-backend `Adapt.adapt(::LavaBackend, tlas::TLAS{CPU})` is not
        # supported (static_tlas is owned per-TLAS-backend; the `to` argument
        # to adapt is intentionally ignored). See instanced-bvh.jl docstring.
        scene = Hikari.Scene(; backend=backend)
        push!(scene, _floor_mesh(),
              Hikari.Diffuse(Kd=Hikari.RGBSpectrum(0.3f0, 0.3f0, 0.3f0)))

        sphere = normal_mesh(Tesselation(Sphere(GeometryBasics.Point3f(0,0,0), 0.35f0), 16))
        # Pure R, G, B, Yellow — guarantees channel separation
        colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0)]
        materials = Hikari.Material[Hikari.Diffuse(Kd=c) for c in colors]
        transforms = Raycore.Mat4f[_shift_mat(-1.5, 0, 0.5), _shift_mat(-0.5, 0, 0.5),
                                    _shift_mat(0.5, 0, 0.5),  _shift_mat(1.5, 0, 0.5)]
        push!(scene, sphere, materials, transforms)
        push!(scene, Hikari.PointLight(GeometryBasics.Point3f(0, -2, 3),
                                        Hikari.RGBSpectrum(30f0)))
        Hikari.sync!(scene)

        film, camera = _make_film_camera(64)
        img = _render(scene, film, camera; backend=backend, hw=false, samples=4, depth=3)

        # All three channels must be illuminated (we have pure R, G, B spheres).
        # Yellow (R+G) alone can't produce B, so B > 0 proves the blue sphere
        # rendered — i.e. instance_id 3's material was selected, not just the
        # first triangle's default.
        m = _channel_maxes(img)
        @test m.r > 0.1f0
        @test m.g > 0.1f0
        @test m.b > 0.1f0
        @test !Mantle.device_lost(Mantle.ctxof(backend.dispatch_bq))
    end

    # ── 5. HW-path render: each instance shows its own color ────────────────
    # Mirrors the SW test above on the HW backend. A second HW render in the
    # same session DEVICE_LOSTs around dispatch ~104 without the fix;
    # root cause was `combined_instance_buf` being shared across rebuilds (each
    # build_tlas appended it to its TLAS's preserves, so freeing the older TLAS
    # freed the buffer out from under the newer one). Fixed by allocating a
    # fresh combined buf per rebuild — see rebuild_hw_tlas_from_batch!.
    @testset "HW render: per-instance materials produce distinct colors" begin
        backend = Mantle.defaultbackend()
        scene = Hikari.Scene(; backend=backend, hw_accel=true)
        push!(scene, _floor_mesh(),
              Hikari.Diffuse(Kd=Hikari.RGBSpectrum(0.3f0, 0.3f0, 0.3f0)))

        sphere = normal_mesh(Tesselation(Sphere(GeometryBasics.Point3f(0,0,0), 0.35f0), 16))
        colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0)]
        materials = Hikari.Material[Hikari.Diffuse(Kd=c) for c in colors]
        transforms = Raycore.Mat4f[_shift_mat(-1.5, 0, 0.5), _shift_mat(-0.5, 0, 0.5),
                                    _shift_mat(0.5, 0, 0.5),  _shift_mat(1.5, 0, 0.5)]
        push!(scene, sphere, materials, transforms)
        push!(scene, Hikari.PointLight(GeometryBasics.Point3f(0, -2, 3),
                                        Hikari.RGBSpectrum(30f0)))
        Hikari.sync!(scene)

        # VulkanTLAS structure: 1 floor BLAS + 1 sphere BLAS = 2 total, 5 instances,
        # 2 batches (floor=1, sphere=4). All sphere instances share the second BLAS;
        # their per-instance custom_indices are distinct nonzero overrides.
        @test length(scene.accel.blas_list) == 2
        @test Raycore.n_instances(scene.accel) == 5
        @test length(scene.accel.instance_batches) == 2
        @test scene.accel.instance_batches[1].n == 1
        @test scene.accel.instance_batches[2].n == 4
        @test scene.accel.instance_batches[2].blas === scene.accel.blas_list[2]
        sphere_records = Array(scene.accel.instance_batches[2].instance_buf)
        overrides = UInt32[(r.custom_index_and_mask & 0x00FFFFFF) for r in sphere_records[1:4]]
        @test all(o != UInt32(0) for o in overrides)
        @test length(Set(overrides)) == 4

        # Actual render: every instance must show its own color, proving the
        # custom_index → media_interfaces lookup works on the HW path too.
        film, camera = _make_film_camera(64)
        img = _render(scene, film, camera; backend=backend, hw=true, samples=4, depth=3)
        m = _channel_maxes(img)
        @test m.r > 0.1f0
        @test m.g > 0.1f0
        @test m.b > 0.1f0
        @test !Mantle.device_lost(Mantle.ctxof(backend.dispatch_bq))
    end

    # ── 6. Rapid push/delete cycles — no BLAS growth ────────────────────────
    @testset "rapid push/delete cycles: BLAS count stays bounded" begin
        scene = Hikari.Scene()
        push!(scene, _floor_mesh(),
              Hikari.Diffuse(Kd=Hikari.RGBSpectrum(0.5f0, 0.5f0, 0.5f0)))
        sphere = normal_mesh(Tesselation(Sphere(GeometryBasics.Point3f(0,0,0), 0.25f0), 12))

        # 10 push/delete cycles, each with 8 scatter instances. If the old
        # N-BLASes-per-scatter bug were back, we'd see 10*8 = 80 BLASes grow
        # without bound. Under the fix: 1 floor + 1 scatter BLAS = 2, stable.
        for cycle in 1:10
            mats = Hikari.Material[Hikari.Diffuse(Kd=(i/8.0, 0.5, 0.5)) for i in 1:8]
            xfs = Raycore.Mat4f[_shift_mat(i*0.3, cycle*0.1) for i in 1:8]
            handles = push!(scene, sphere, mats, xfs)
            Hikari.sync!(scene)
            @test length(scene.accel.blas_storage) <= 2 + 1   # allow at most +1 for transient
            for h in handles
                delete!(scene.accel, h.geometry)
            end
            Raycore.sync!(scene.accel)
        end

        # After all cycles: floor BLAS + the most recent scatter BLAS at most
        @test length(scene.accel.blas_storage) <= 2
        @test Raycore.n_total_instances(scene.accel) == 1   # just the floor
    end

    # ── 7. Transform updates preserve instance_id ───────────────────────────
    @testset "update_instance_transform! preserves override" begin
        scene = Hikari.Scene()
        push!(scene, _floor_mesh(),
              Hikari.Diffuse(Kd=Hikari.RGBSpectrum(0.5f0, 0.5f0, 0.5f0)))
        sphere = normal_mesh(Tesselation(Sphere(GeometryBasics.Point3f(0,0,0), 0.25f0), 12))

        mats = Hikari.Material[Hikari.Diffuse(Kd=c) for c in ((1.0,0,0), (0,1.0,0))]
        xfs = Raycore.Mat4f[_shift_mat(-1,0), _shift_mat(1,0)]
        push!(scene, sphere, mats, xfs)
        Hikari.sync!(scene)

        # Remember the per-instance override IDs
        before = [d.instance_id for d in Array(scene.accel.instances)]

        # Update one transform
        Raycore.update_instance_transform!(scene.accel, 2, _shift_mat(-2, 0))
        Hikari.sync!(scene)

        after = [d.instance_id for d in Array(scene.accel.instances)]
        @test before == after  # instance_id must survive transform update
    end

    # ── 8. instance_id 0 ↔ per-triangle fallback ────────────────────────────
    @testset "override=0 → per-triangle medium_interface_idx fallback (CPU-side)" begin
        # Build a scene via the legacy (non-scattering) push! — every instance
        # gets instance_id=0.
        scene = Hikari.Scene()
        red_mat = Hikari.Diffuse(Kd=(1.0, 0.1, 0.1))
        push!(scene, _floor_mesh(),
              Hikari.Diffuse(Kd=Hikari.RGBSpectrum(0.5f0, 0.5f0, 0.5f0)))
        sphere = normal_mesh(Tesselation(Sphere(GeometryBasics.Point3f(0,0,0.5), 0.35f0), 16))
        push!(scene, sphere, red_mat)
        Hikari.sync!(scene)

        # All instances have instance_id == 0 → fallback path taken
        @test all(d.instance_id == UInt32(0) for d in Array(scene.accel.instances))

        # Trace a ray directly against the CPU TLAS — returns idx.
        # accel.instances[idx].instance_id is 0, so resolve_mi_idx falls
        # through to `tri.metadata.medium_interface_idx`.
        static = Adapt.adapt(CPU(), scene.accel)
        ray = Raycore.Ray(Raycore.Point3f(0, 0, 3), Raycore.Vec3f(0, 0, -1), 0f0, 100f0, 0f0)
        hit, tri, _, _, idx = Raycore.closest_hit(static, ray)
        @test hit
        @test static.instances[idx].instance_id == UInt32(0)
        @test tri.metadata.medium_interface_idx != UInt32(0)   # triangle carries material
    end
end
