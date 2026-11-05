using Test, Hikari, Mantle, GeometryBasics, Raycore

@testset "primary payload reconstructs only the camera path defaults" begin
    ray = Raycore.Ray(o=Point3f(1,2,3),d=Vec3f(0,0,1),t_max=10f0)
    lambda = Hikari.sample_wavelengths_visible(0.17f0)
    for pixel in (Int32(1),Int32(37)), beta in (0f0,0.3f0,1f0)
        expected = Hikari.with_beta(Hikari.VPRayWorkItem(ray,lambda,pixel),Hikari.SpectralRadiance(beta))
        # No other fields exist here: reconstruction must not read unwritten
        # scattering/MIS state left in the primary ray's backing storage.
        payload = (;ray=fill(ray,pixel),lambda=fill(lambda,pixel),beta=fill(expected.beta,pixel),
                   medium_idx=fill(expected.medium_idx,pixel))
        @test isequal(Hikari.primary_work(payload,pixel),expected)
    end
end

function unfused_sample_plan(vp)
    state, adapted, pr = vp.state, vp.adapted, vp.perrun
    reachable = Hikari.emission_reachable(adapted.lights, adapted.materials)
    refs = Hikari.render_refs(adapted.accel, adapted.media_interfaces, adapted.media,
        adapted.materials, adapted.lights, pr.camera, pr.initial_medium,
        pr.filter_params, vp.filter_sampler_gpu, pr.sample_idx, vp.regularize,
        reachable, Int32(vp.samples_per_pixel), vp.max_component_value)
    d = Hikari.sample_rounds(vp.max_depth, reachable, Hikari.has_crossings(adapted.media_interfaces))
    g = Mantle.Graph(state.memory.dev)
    Mantle.dispatch!(g, Hikari.vp_next_sample_kernel!, (pr.sample_idx, state.sobol_rng.lanes), 1;
                     group = 1, name = "next sample")
    cur, nxt = state.ray_queue_a, state.ray_queue_b
    Hikari.setup_passes!(g, state, refs, cur)
    for _ in 1:d
        Hikari.round_passes!(g, state, refs, cur, nxt;
            has_media = !isempty(adapted.media), has_lights = length(adapted.lights) > 0)
        cur, nxt = nxt, cur
    end
    Hikari.accumulate_passes!(g, state, refs)
    Mantle.record!(Mantle.Plan(g))
end

@testset "primary tracing matches queued camera rays (hardware=$hw, alpha=$alpha, camera=$style)" for hw in (false, true), alpha in (1f0, 0.5f0), style in (:perspective, :matrix, :lens)
    backend = Mantle.defaultbackend()
    scene = Hikari.Scene(; backend, hw_accel = hw)
    mesh = GeometryBasics.normal_mesh(GeometryBasics.Tesselation(Sphere(Point3f(0), 0.7f0), 12))
    push!(scene, mesh, Hikari.Diffuse(color = Hikari.RGBSpectrum(0.6f0, 0.3f0, 0.2f0, alpha)))
    push!(scene, Hikari.AmbientLight(Hikari.RGBSpectrum(0.5f0)))
    Hikari.sync!(scene)
    film = Hikari.Film(backend, Hikari.Film(Point2f(12, 12)))
    perspective = style === :lens ? Hikari.PerspectiveCamera(
        Hikari.look_at(Point3f(0, -3, 1), Point3f(0), Vec3f(0, 1, 0)),
        Hikari.Bounds2(Point2f(-1), Point2f(1)), 0f0, 0f0, 0.1f0, sqrt(10f0), 50f0, film) :
        Hikari.PerspectiveCamera(Point3f(0, -3, 1), Point3f(0), film; fov = 50f0)
    camera = style === :matrix ? Hikari.MatrixCamera(
        Hikari.get_camera_to_world(perspective).inv_m, perspective.core.camera_to_screen.m, Point2f(12, 12)) : perspective
    for depth in 0:4
        vp = Hikari.VolPath(samples = 2, max_depth = depth, hw_accel = hw)
        reference = nothing
        try
            vp(scene, film, camera)
            actual = Array(film.framebuffer)
            @test any(c -> c.r > 0f0 || c.g > 0f0 || c.b > 0f0, actual)
            @test any(pp -> pp.pass.name == "primary", vp.state.plans.sample.passes)
            reference = unfused_sample_plan(vp)
            Hikari.clear!(vp)
            Hikari.resetsamples!(vp.perrun)
            for _ in 1:(vp.samples_per_pixel ÷ vp.state.sobol_rng.lanes)
                Mantle.run!(reference)
            end
            Hikari.finalize_film!(vp, film)
            Mantle.waitidle(vp.state.memory.dev)
            expected = Array(film.framebuffer)
            @test all(c -> isfinite(c.r + c.g + c.b), actual)
            @test all(isapprox(a.r, b.r; rtol = 5f-5, atol = 2f-5) &&
                      isapprox(a.g, b.g; rtol = 5f-5, atol = 2f-5) &&
                      isapprox(a.b, b.b; rtol = 5f-5, atol = 2f-5)
                      for (a, b) in zip(actual, expected))
        finally
            reference === nothing || Mantle.free!(reference)
            close(vp)
        end
    end
    Hikari.free!(film)
end

@testset "zero-depth render without emission needs no primary trace" begin
    backend = Mantle.defaultbackend()
    scene = Hikari.Scene(; backend)
    push!(scene, Hikari.PointLight(Point3f(0, 2, 3), Hikari.RGBSpectrum(1f0)))
    Hikari.sync!(scene)
    film = Hikari.Film(backend, Hikari.Film(Point2f(8, 8)))
    camera = Hikari.PerspectiveCamera(Point3f(0, -3, 1), Point3f(0), film; fov = 50f0)
    vp = Hikari.VolPath(samples = 1, max_depth = 0)
    try
        vp(scene, film, camera)
        @test !any(pp -> pp.pass.name == "primary", vp.state.plans.sample.passes)
        @test all(c -> c.r == c.g == c.b == 0f0, Array(film.framebuffer))
        @test film.iteration_index[] == 1
    finally
        close(vp)
        Hikari.free!(film)
    end
end
