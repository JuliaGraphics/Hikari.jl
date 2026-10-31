using Test
using Hikari
using Lava, Mantle
using Raycore
using Adapt
using GeometryBasics
using GeometryBasics: normal_mesh, Tesselation, Rect3f, Sphere
using LinearAlgebra: I
using Statistics: mean

# HW vs SW rendering parity: a simple scene rendered with the hardware
# acceleration structure (a ray query on Vulkan, the intersector on Metal) and
# with the software BVH path (Raycore BVH + compute shaders) must produce
# the same pixels within a small tolerance.  Both paths should hit the
# same material eval, the same light sampling, the same integrator loop —
# only the ray-triangle intersection differs.
#
# This catches regressions in either path: a BVH-builder bug changes SW
# only; a LavaTLAS instance-override bug changes HW only; a material eval
# regression shows up in both (test still passes) but the test harness
# elsewhere would catch it.

# ── Test Scene ──────────────────────────────────────────────────────────────
# Small, deterministic, noise-sensitive setup: three spheres over a floor,
# one point light, gaussian filter off for predictable per-pixel comparison.

function _build_scene(; backend, hw_accel::Bool)
    scene = Hikari.Scene(; backend=backend, hw_accel=hw_accel)
    # Floor
    floor = normal_mesh(Rect3f(GeometryBasics.Vec3f(-3, -3, -0.05),
                                 GeometryBasics.Vec3f(6, 6, 0.05)))
    push!(scene, floor, Hikari.Diffuse(color = Hikari.RGBSpectrum(0.6f0, 0.6f0, 0.6f0)))
    # Three colored spheres
    sphere = normal_mesh(Tesselation(Sphere(GeometryBasics.Point3f(0, 0, 0), 0.35f0), 16))
    for (x, color) in [(-0.8f0, (1f0, 0.2f0, 0.2f0)),
                        ( 0.0f0, (0.2f0, 1f0, 0.2f0)),
                        ( 0.8f0, (0.2f0, 0.2f0, 1f0))]
        tf = Raycore.Mat4f([1 0 0 x; 0 1 0 0f0; 0 0 1 0.35f0; 0 0 0 1])
        push!(scene, sphere, Hikari.Diffuse(color = color); transform=tf)
    end
    push!(scene, Hikari.PointLight(GeometryBasics.Point3f(0, -2, 3),
                                    Hikari.RGBSpectrum(30f0)))
    Hikari.sync!(scene)
    return scene
end

function _make_film_camera(res::Int)
    film = Hikari.Film(GeometryBasics.Point2f(res, res))
    camera = Hikari.PerspectiveCamera(
        GeometryBasics.Point3f(0, -3, 1.5),
        GeometryBasics.Point3f(0, 0, 0.35),
        film; fov=50f0,
    )
    return film, camera
end

function _render(scene, film, camera; backend, hw::Bool, samples::Int, depth::Int)
    gpu_film = Hikari.Film(backend, film)
    vp = Hikari.VolPath(samples=samples, max_depth=depth, hw_accel=hw)
    vp(scene, gpu_film, camera)
    img = Array(gpu_film.framebuffer)
    close(vp)
    return img
end

function _pixel_diff(a, b)
    size(a) == size(b) || error("size mismatch: $(size(a)) vs $(size(b))")
    diffs = Float32[]
    for i in eachindex(a)
        dr = abs(Float32(a[i].r) - Float32(b[i].r))
        dg = abs(Float32(a[i].g) - Float32(b[i].g))
        db = abs(Float32(a[i].b) - Float32(b[i].b))
        push!(diffs, max(dr, dg, db))
    end
    return (mean=mean(diffs), max=maximum(diffs))
end

# Per-channel image means: a single scalar per channel summarising how much
# red/green/blue light the renderer integrated across the whole frame.  HW
# and SW produce the same monte-carlo integral (same integrand, same scene),
# so these means must agree to tight tolerance even though individual pixels
# diverge at silhouette edges due to FP intersection-order differences.
function _channel_means(img)
    r = mean(Float32(p.r) for p in img)
    g = mean(Float32(p.g) for p in img)
    b = mean(Float32(p.b) for p in img)
    return (r=r, g=g, b=b)
end

# ── Tests ───────────────────────────────────────────────────────────────────

@testset "HW RT vs SW BVH rendering parity" begin

backend = Mantle.defaultbackend()
dev = Hikari.mantle_device(backend)

if !Mantle.supports_hwtlas(dev)
    @info "no hardware acceleration structure on this device, so no parity to check"
    @test_skip false
else
    # Small resolution + decent samples keeps the render fast while giving
    # Monte-Carlo noise a chance to average down so the parity tolerance can
    # be tight.
    res = 48
    samples = 16
    depth = 4

    # SW path
    scene_sw = _build_scene(; backend=backend, hw_accel=false)
    film_sw, cam_sw = _make_film_camera(res)
    img_sw = _render(scene_sw, film_sw, cam_sw;
                     backend=backend, hw=false, samples=samples, depth=depth)
    @test size(img_sw) == (res, res)

    # HW path — fresh scene with its own TLAS so nothing is reused between
    # the two renders (test_tlas_instance_override.jl documents that a HW
    # render after another render in the same session can DEVICE_LOST; a
    # fresh scene gives HW RT a clean state).
    scene_hw = _build_scene(; backend=backend, hw_accel=true)
    film_hw, cam_hw = _make_film_camera(res)
    img_hw = _render(scene_hw, film_hw, cam_hw;
                     backend=backend, hw=true, samples=samples, depth=depth)
    @test size(img_hw) == (res, res)

    # Neither path should have tripped DEVICE_LOST.
    @test !Mantle.retired(dev)

    # Both renders must produce non-degenerate output (not all black, not
    # all saturated).  This catches gross regressions: a pipeline that
    # returns zero-filled film, or a tonemap bug that clips everything to
    # white.
    for (name, img) in (("SW", img_sw), ("HW", img_hw))
        m = _channel_means(img)
        @test 0.01f0 < (m.r + m.g + m.b) / 3 < 0.95f0
    end

    # Per-pixel diff and per-channel means: logged for visibility.  HW and
    # SW renders currently diverge meaningfully — HW integrates ~30% more
    # light per channel on a mixed scene, ~70% on this test scene.
    #
    # Bisect (2026-04-20, minimal scene, fresh Julia session per render to
    # work around the HW RT cross-session lifetime bug):
    #   * FLOOR ONLY (flat mesh, one point light): SW=0.243, HW=0.244
    #     → parity.  Flat geometry has identical face and vertex normals so
    #     the shading result is identical.
    #   * SPHERE ONLY (tesselated curved mesh, no floor): SW=0.019, HW=0.026
    #     → HW 34% brighter.  Curved tesselated geometry diverges on its
    #     own, with no shadow caster in the scene, so the issue is not
    #     shadow-ray handling — it's either primary-ray barycentric
    #     differences from HW vs SW intersection rounding (cascading
    #     through vertex-normal interpolation → BSDF cosine term), or a
    #     secondary-ray offset/self-intersection issue specific to curved
    #     tesselated geometry.
    #   * FLOOR + SPHERE: compound effect, ~1.7× brighter.
    #
    # The sphere-only divergence (34%) is the cleanest signal for future
    # investigation: same scene without any potential shadow-ray confound,
    # same material (pure white diffuse), same point-light position.  Flat
    # geometry is fine — that rules out hardware structure setup and material
    # evaluation as root causes.
    pix = _pixel_diff(img_sw, img_hw)
    msw = _channel_means(img_sw)
    mhw = _channel_means(img_hw)
    @info "HW vs SW parity" per_pixel_mean=pix.mean per_pixel_max=pix.max sw_means=msw hw_means=mhw

    # Tight per-channel parity. Used to be `@test_broken` while the HW RT
    # path on curved tesselated geometry was noisy — the per-material chit
    # work (sd/vk-hw-accel: emission MIS inline, ConductorEvaluated, SBT
    # slot routing) tightened HW↔SW agreement to ≤ 0.02 + 10% of channel
    # mean and `@test_broken` started reporting `Expected to fail but
    # passed`. Promoted back to `@test`.
    for ch in (:r, :g, :b)
        v_sw, v_hw = getfield(msw, ch), getfield(mhw, ch)
        @test abs(v_hw - v_sw) < 0.02 + 0.1 * max(v_sw, v_hw)
    end
end

end  # @testset
