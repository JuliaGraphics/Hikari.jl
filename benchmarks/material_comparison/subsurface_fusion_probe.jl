module SubsurfaceFusionProbe
using Hikari, Mantle, GeometryBasics, Statistics, JSON3

function queued_plan(vp)
    state, adapted, pr = vp.state, vp.adapted, vp.perrun
    reachable = Hikari.emission_reachable(adapted.lights, adapted.materials)
    refs = Hikari.render_refs(adapted.accel, adapted.media_interfaces, adapted.media,
        adapted.materials, adapted.lights, pr.camera, pr.initial_medium, pr.filter_params,
        vp.filter_sampler_gpu, pr.sample_idx, vp.regularize, reachable,
        Int32(vp.samples_per_pixel), vp.max_component_value)
    g = Mantle.Graph(state.memory.dev)
    Mantle.dispatch!(g, Hikari.vp_next_sample_kernel!, (pr.sample_idx, state.sobol_rng.lanes), 1;
                     group = 1, name = "next sample")
    cur, nxt = state.ray_queue_a, state.ray_queue_b
    Hikari.setup_passes!(g, state, refs, cur)
    for _ in 1:Hikari.sample_rounds(vp.max_depth, reachable, Hikari.has_crossings(adapted.media_interfaces))
        Hikari.round_passes!(g, state, refs, cur, nxt; has_media = !isempty(adapted.media),
                            has_lights = !isempty(adapted.lights))
        cur, nxt = nxt, cur
    end
    Hikari.accumulate_passes!(g, state, refs)
    Mantle.record!(Mantle.Plan(g))
end

function run(output; device, res = 32, samples = 64, hw_accel = true, typed = false, observe = nothing,
             material = Hikari.Principled(base_color = (1, 1, 1), roughness = 0.4,
                 subsurface_weight = 1, subsurface_radius = (1, 1, 1), subsurface_scale = 0.1, ior = 1.45))
    Mantle.defaultdevice!(device)
    backend = Mantle.defaultbackend()
    scene = Hikari.Scene(; backend, hw_accel)
    push!(scene, normal_mesh(Tesselation(Sphere(Point3f(0), 1f0), 48)), material)
    push!(scene, Hikari.AmbientLight(Hikari.RGBSpectrum(1f0)))
    Hikari.sync!(scene)
    film = Hikari.Film(backend, Hikari.Film(Point2f(res, res)))
    camera = Hikari.PerspectiveCamera(Point3f(0, -4, 0), Point3f(0), film;
                                    up = Vec3f(0, 0, 1), fov = 30f0)
    vp = Hikari.VolPath(; samples, max_depth = 64, regularize = false, hw_accel)
    reference = nothing
    centermean(img) = mean((p.r + p.g + p.b) / 3 for i in 1:res, j in 1:res
        if (i - res ÷ 2)^2 + (j - res ÷ 2)^2 <= (res ÷ 5)^2 for p in (img[i, j],))
    try
        vp(scene, film, camera)
        actual = Array(film.framebuffer)
        observe === nothing || observe(vp)
        println("SS_FUSED ", centermean(actual)); flush(stdout)
        reference = queued_plan(vp)
        Hikari.clear!(vp)
        Hikari.resetsamples!(vp.perrun)
        for _ in 1:(vp.samples_per_pixel ÷ vp.state.sobol_rng.lanes)
            Mantle.run!(reference)
        end
        Hikari.finalize_film!(vp, film)
        Mantle.waitidle(device)
        expected = Array(film.framebuffer)
        typed_mean = nothing
        if typed
            reference_vp = Hikari.VolPath(; samples, max_depth = 64, regularize = false, hw_accel)
            try
                Main.MaterialComparison.first_draw((; vp = reference_vp, scene, film, camera, dev = device), :typed_cycles)
                typed_mean = centermean(Array(film.framebuffer))
            finally
                close(reference_vp)
            end
        end
        record = (; device = Mantle.devicename(device), res, samples, hw_accel,
                  fused = centermean(actual), queued = centermean(expected), typed_mean)
        mkpath(dirname(output))
        open(io -> JSON3.write(io, record), output, "w")
        println("SS_CONTROL ", record); flush(stdout)
    finally
        reference === nothing || Mantle.free!(reference)
        close(vp)
        Hikari.free!(film)
    end
    nothing
end
end
