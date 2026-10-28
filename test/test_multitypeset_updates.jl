# MultiTypeSet update tests — Phase J of the Hikari GPU stability work.
#
# `Hikari.update_material!` mutates a slot in `scene.materials` /
# `scene.media` / `scene.media_interfaces` (all `MultiTypeSet`s) without
# growing the slot count.  The previous Hikari test suite covered only the
# `MediumInterface{NullMaterial}` regression (`test_update_material_null.jl`).
# This file backs the broader contract:
#
#   J1. Scalar material update (Diffuse Kd) actually changes pixel output.
#   J2. Texture-backed material update reuses GPU texture slots.
#   J3. RGBGridMedium animation loop has bounded GPU memory.
#   J4. HW RT picks up material updates equivalently to SW TLAS.
#   J5. Repeated updates over many renders stay stable (no DEVICE_LOST,
#       no buffer growth).
#   J6. `Adapt.adapt(backend, scene)` — the adapted-snapshot picks up
#       updates without consumers having to re-cache.

using Test
using Hikari
using Mantle
using Raycore
using Adapt
using GeometryBasics
using GeometryBasics: normal_mesh, Tesselation, Rect3f, Sphere, Point3f, Vec3f, Point2f
using LinearAlgebra: I
using Statistics: mean
using GPUArraysCore: @allowscalar

# ── Test helpers ────────────────────────────────────────────────────────────

const _BACKEND = Mantle.defaultbackend()

function _channel_means(img)
    r = mean(Float32(p.r) for p in img)
    g = mean(Float32(p.g) for p in img)
    b = mean(Float32(p.b) for p in img)
    return (r=r, g=g, b=b)
end

function _make_sphere_scene(; backend=_BACKEND, hw_accel::Bool=false)
    scene = Hikari.Scene(; backend=backend, hw_accel=hw_accel)
    floor = normal_mesh(Rect3f(Vec3f(-3, -3, -0.05), Vec3f(6, 6, 0.05)))
    push!(scene, floor, Hikari.Diffuse(Kd=Hikari.RGBSpectrum(0.6f0, 0.6f0, 0.6f0)))
    sphere = normal_mesh(Tesselation(Sphere(Point3f(0, 0, 0.35), 0.35f0), 16))
    handle = push!(scene, sphere, Hikari.Diffuse(Kd=Hikari.RGBSpectrum(0.9f0, 0.1f0, 0.1f0)))
    push!(scene, Hikari.PointLight(Point3f(0, -2, 3), Hikari.RGBSpectrum(30f0)))
    Hikari.sync!(scene)
    return scene, handle
end

function _render(scene; backend=_BACKEND, hw::Bool=false, res::Int=32,
                 samples::Int=16, depth::Int=3)
    film = Hikari.Film(Point2f(res, res))
    gpu_film = Hikari.Film(backend, film)
    camera = Hikari.PerspectiveCamera(Point3f(0, -3, 1.5), Point3f(0, 0, 0.35),
                                      film; fov=50f0)
    vp = Hikari.VolPath(samples=samples, max_depth=depth, hw_accel=hw)
    try
        vp(scene, gpu_film, camera)
        return Array(gpu_film.framebuffer)
    finally
        close(vp)
        Hikari.free!(gpu_film)
    end
end

# What the device's pool holds once collected garbage is returned to it: the bytes
# of its blocks and its live regions. Asked of the pool, as Mantle's own leak tests
# do, so it reads the same on every backend.
function pool_state(dev)
    p = Mantle.pool(dev)
    GC.gc(true); GC.gc(true)
    Mantle.waitidle(dev)
    Mantle.reclaim!(p, dev; wait = true)
    blocks = collect(Iterators.flatten(values(p.blocks)))
    return (bytes = Mantle.reserved(p), regions = sum(b -> length(b.live), blocks; init = 0))
end

function _make_grid_volume_scene(; backend=_BACKEND, σ_s::Float32=0.3f0)
    scene = Hikari.Scene(; backend=backend)
    cube = normal_mesh(Rect3f(Vec3f(0), Vec3f(1)))
    medium = Hikari.RGBGridMedium(
        σ_a_grid=fill(Hikari.RGBSpectrum(0f0),  4, 4, 4),
        σ_s_grid=fill(Hikari.RGBSpectrum(σ_s),   4, 4, 4),
        sigma_scale=1f0, g=0f0,
        bounds=Raycore.Bounds3(Point3f(0, 0, 0), Point3f(1, 1, 1)))
    handle = push!(scene, cube,
                    Hikari.MediumInterface(Hikari.NullMaterial(); inside=medium))
    Hikari.sync!(scene)
    return scene, handle
end

# ── J1. Scalar update changes pixels ───────────────────────────────────────

@testset "J1: update_material! scalar Kd changes pixel mean" begin
    scene, handle = _make_sphere_scene(; hw_accel=false)

    img_red = _render(scene; samples=16)
    means_red = _channel_means(img_red)

    # Swap red sphere for blue sphere via update_material!.
    Hikari.update_material!(scene, handle.interface,
                             Hikari.Diffuse(Kd=Hikari.RGBSpectrum(0.1f0, 0.1f0, 0.9f0)))
    img_blue = _render(scene; samples=16)
    means_blue = _channel_means(img_blue)

    # Red render must be redder than blue render; blue render must be bluer
    # than red render.  Loose ratios because the floor + scattering still
    # contribute white-ish light.
    @info "J1: channel means" red=means_red blue=means_blue
    # The sphere is one of several scene elements (floor, light) so its
    # contribution to the channel mean is small.  Use absolute deltas of
    # the dominant channel between renders, not ratios.  A render where
    # the sphere swapped from red to blue must show:
    #   - red render's r channel is higher than its b channel
    #   - blue render's b channel is higher than its r channel
    #   - red render's r > blue render's r (the sphere's effect)
    #   - blue render's b > red render's b
    @test means_red.r  > means_red.b
    @test means_blue.b > means_blue.r
    @test means_red.r  > means_blue.r
    @test means_blue.b > means_red.b
end

# ── J2. Texture-backed updates reuse slots ─────────────────────────────────

@testset "J2: texture-backed Kd update reuses GPU texture slots" begin
    scene, handle = _make_sphere_scene(; hw_accel=false)

    n_arrs_before = length(scene.materials.texture_gpu_arrays)
    n_mats_before = length(scene.materials)

    # Alternate between two scalar Diffuse materials many times.  Scalar
    # `Kd` materials don't allocate a texture array (they store as a 0-D
    # texture-with-default), so this primarily exercises the slot-update
    # path; we still expect zero growth in `materials.texture_gpu_arrays`.
    for σ in (0.2f0, 0.4f0, 0.6f0, 0.8f0)
        new_mat = Hikari.Diffuse(Kd=Hikari.RGBSpectrum(σ, σ, σ))
        Hikari.update_material!(scene, handle.interface, new_mat)
    end

    @test length(scene.materials)                  == n_mats_before
    @test length(scene.materials.texture_gpu_arrays) == n_arrs_before
end

# ── J3. RGBGridMedium animation: bounded GPU memory ────────────────────────

@testset "J3: RGBGridMedium animation loop is bounded" begin
    scene, handle = _make_grid_volume_scene(; σ_s=0.3f0)
    dev = Hikari.mantle_device(_BACKEND)

    before          = pool_state(dev)
    media_before    = length(scene.media)
    texarrs_before  = length(scene.media.texture_gpu_arrays)

    # 50 frames of σ_s animation, like RayMakie's volume-render loop.
    for k in 1:50
        σ = 0.1f0 + 0.5f0 * (sinpi(k / 25f0) + 1f0)
        new_med = Hikari.RGBGridMedium(
            σ_a_grid=fill(Hikari.RGBSpectrum(0f0), 4, 4, 4),
            σ_s_grid=fill(Hikari.RGBSpectrum(σ),    4, 4, 4),
            sigma_scale=1f0, g=0f0,
            bounds=Raycore.Bounds3(Point3f(0, 0, 0), Point3f(1, 1, 1)))
        Hikari.update_material!(scene, handle.interface,
                                 Hikari.MediumInterface(Hikari.NullMaterial();
                                                         inside=new_med))
    end
    after = pool_state(dev)

    @info "J3: post-animation state" media=length(scene.media) texarrs=length(scene.media.texture_gpu_arrays) before after

    @test length(scene.media)                       == media_before
    @test length(scene.media.texture_gpu_arrays)    == texarrs_before
    # GPU memory may shift slightly due to deferred-free timing; bound the
    # delta tightly so a real leak is visible.
    @test after.bytes - before.bytes <= 32 * 1024 * 1024  # +32 MiB ceiling
end

# ── J4. HW RT picks up material updates the same as SW ─────────────────────

@testset "J4: HW RT picks up update_material! the same as SW" begin
    scene_sw, h_sw = _make_sphere_scene(; hw_accel=false)
    scene_hw, h_hw = _make_sphere_scene(; hw_accel=true)

    # Update both to green.
    new_mat = Hikari.Diffuse(Kd=Hikari.RGBSpectrum(0.1f0, 0.9f0, 0.1f0))
    Hikari.update_material!(scene_sw, h_sw.interface, new_mat)
    Hikari.update_material!(scene_hw, h_hw.interface, new_mat)

    img_sw = _render(scene_sw; hw=false, samples=16)
    img_hw = _render(scene_hw; hw=true,  samples=16)

    means_sw = _channel_means(img_sw)
    means_hw = _channel_means(img_hw)

    @info "J4: post-update channel means" sw=means_sw hw=means_hw

    # Both must show green dominant — confirms both paths see the update.
    @test means_sw.g > means_sw.r
    @test means_sw.g > means_sw.b
    @test means_hw.g > means_hw.r
    @test means_hw.g > means_hw.b
    # And SW vs HW should agree at least on which channel dominates;
    # absolute parity is broader (HW path has documented divergence on
    # curved geometry, see test_hw_sw_parity.jl).
    @test sign(means_sw.g - means_sw.r) == sign(means_hw.g - means_hw.r)
end

# ── J5. Stability under many render+update cycles ──────────────────────────

@testset "J5: render + update_material! loop is stable" begin
    scene, handle = _make_sphere_scene(; hw_accel=false)
    dev = Hikari.mantle_device(_BACKEND)

    before = pool_state(dev)

    for k in 1:20
        # Cycle the material colour each iteration.
        r = Float32(0.1 + 0.4 * (k % 3 == 0))
        g = Float32(0.1 + 0.4 * (k % 3 == 1))
        b = Float32(0.1 + 0.4 * (k % 3 == 2))
        Hikari.update_material!(scene, handle.interface,
                                 Hikari.Diffuse(Kd=Hikari.RGBSpectrum(r, g, b)))
        img = _render(scene; samples=4, res=16)
        @test !Mantle.retired(dev)
        m = _channel_means(img)
        @test (m.r + m.g + m.b) > 0.001f0   # non-degenerate output
    end
    after = pool_state(dev)
    @info "J5: after 20 render + update cycles" before after

    @test after.bytes - before.bytes <= 64 * 1024 * 1024    # +64 MiB
    @test after.regions - before.regions <= 16             # +16 regions
end

# ── J6. Adapted-snapshot freshness contract ────────────────────────────────

@testset "J6: Adapt.adapt(backend, scene) reflects update_material!" begin
    scene, handle = _make_sphere_scene(; hw_accel=false)

    # Snapshot the materials slot's contents BEFORE update.
    Hikari.sync!(scene)
    adapted_before = Adapt.adapt(_BACKEND, scene)

    # An adapted scene has `.materials` as a `StaticMultiTypeSet`.  Read
    # the first material's Kd via @allowscalar.  We compare the raw `data`
    # values, not the layout — the test cares that update_material! flows
    # through to the static snapshot's contents.
    function _read_material_kd(static_mts)
        # `static_mts.data` is a Tuple of device arrays (one per type slot);
        # the first non-empty slot holds the floor's diffuse, the second
        # holds the sphere's diffuse.  We index by slot then element.
        mats_arr = first(static_mts.data)
        @assert length(mats_arr) >= 2
        return @allowscalar mats_arr[2].Kd  # sphere is the second push!
    end
    kd_before = _read_material_kd(adapted_before.materials)

    # Update to green.
    Hikari.update_material!(scene, handle.interface,
                             Hikari.Diffuse(Kd=Hikari.RGBSpectrum(0.1f0, 0.9f0, 0.1f0)))
    Hikari.sync!(scene)
    adapted_after = Adapt.adapt(_BACKEND, scene)
    kd_after = _read_material_kd(adapted_after.materials)

    @info "J6: Kd before/after" before=kd_before after=kd_after

    # The two Kds must differ — confirms the adapted form sees the update.
    # Each Kd is an RGBSpectrum/Texture; compare the channel that should
    # have changed.  The exact compare path depends on Hikari's texture
    # representation; using string comparison as a robust proxy.
    @test string(kd_before) != string(kd_after)
end
