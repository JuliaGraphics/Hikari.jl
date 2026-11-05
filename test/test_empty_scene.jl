# An EMPTY scene renders, on both traversal paths.
#
# A software `TLAS` with no BLAS used to type itself over `Triangle{UInt32}`,
# since there was no first BLAS to read the primitive type from. Hikari's
# `resolve_mi_idx` reads `primitive.metadata.medium_interface_idx`, and a bare
# `UInt32` has no such field, so the empty scene's kernel held a field access
# that could only throw. Metal refuses to compile a throw it cannot report
# (`unsupported call to jl_f_getfield`), so every render before the first plot
# landed, and every one after the last was deleted, failed to compile. RayMakie
# met it twice: a scene emptied and refilled, and the `hw_accel` switch.
#
# The hardware default never had this: `HWTLAS{Triangle{TriangleMeta}}` is told
# its primitive type by its type parameter. The software default is now told the
# same thing, so an empty and a populated scene compile to one kernel.

using Test, Hikari, Mantle, Raycore, GeometryBasics, Statistics

const EMPTY_META = Raycore.Triangle{Hikari.TriangleMeta}

@testset "an empty software TLAS is typed over what Hikari pushes" begin
    backend = Mantle.defaultbackend()
    acc = Hikari.default_accel(backend, Val(false))
    @test eltype(acc) == EMPTY_META
    Raycore.sync!(acc)
    @test eltype(acc.static_tlas) == EMPTY_META
    # And a populated one agrees, which is the point: one kernel, not two.
    scene = Hikari.Scene(; backend)
    push!(scene, GeometryBasics.normal_mesh(Sphere(Point3f(0), 0.5f0)),
          Hikari.Diffuse(color = Hikari.RGBSpectrum(0.7f0)))
    Hikari.sync!(scene)
    @test eltype(scene.accel) == EMPTY_META
end

@testset "surface misses evaluate environment lighting inline (hardware=$hw, sky=$sky)" for hw in (false, true), sky in (false, true)
    backend = Mantle.defaultbackend()
    dev = Hikari.mantle_device(backend)
    scene = Hikari.Scene(; backend, hw_accel = hw)
    push!(scene, Hikari.PointLight(Point3f(0, 2, 3), Hikari.RGBSpectrum(1f0)))
    sky && push!(scene, Hikari.AmbientLight(Hikari.RGBSpectrum(0.5f0)))
    Hikari.sync!(scene)
    res = 8
    film = Hikari.Film(backend, Hikari.Film(Point2f(res, res)))
    cam = Hikari.PerspectiveCamera(Point3f(0, -3, 1), Point3f(0), film; fov = 50f0)
    vp = Hikari.VolPath(samples = 1, max_depth = 2, hw_accel = hw)
    trace_plan = nothing
    try
        Hikari.render!(vp, scene, film, cam)
        px = Array(film.framebuffer)
        @test !any(pp -> pp.pass.name == "escaped", vp.state.plans.sample.passes)
        @test vp.state.escaped_queue.capacity == 1
        @test all(c -> isfinite(c.r + c.g + c.b), px)
        @test sky ? any(c -> c.r > 0, px) : all(c -> c.r == c.g == c.b == 0f0, px)

        # Stop after the first trace, before a later bounce resets its counters.
        # Every primary ray misses; its radiance is written without a queue.
        g = Mantle.Graph(dev)
        wanted = Set(("clear", "clear-queue", "reset", "primary"))
        for pp in vp.state.plans.sample.passes
            name = pp.pass.name
            name in wanted || continue
            delete!(wanted, name)
            for d in pp.pass.dispatches
                Mantle.dispatch!(g, d.kernel, d.args, d.ndrange; group = d.group, name)
            end
            isempty(wanted) && break
        end
        @test isempty(wanted)
        trace_plan = Mantle.record!(Mantle.Plan(g))
        Mantle.run!(trace_plan)
        Mantle.waitidle(dev)
        @test only(Array(vp.state.escaped_queue.size)) == 0
        L = Array(vp.state.pixel_L)
        @test sky ? any(>(0f0), L) : all(iszero, L)
    finally
        trace_plan === nothing || Mantle.free!(trace_plan)
        close(vp)
        Hikari.free!(film)
    end
end

@testset "medium misses retain deferred environment evaluation" begin
    backend = Mantle.defaultbackend()
    dev = Hikari.mantle_device(backend)
    function environment_frame(in_medium)
        scene = Hikari.Scene(; backend)
        push!(scene, Hikari.AmbientLight(Hikari.RGBSpectrum(0.5f0)))
        key = in_medium ? push!(scene.media,
            Hikari.HomogeneousMedium(σ_a = Hikari.RGBSpectrum(0f0), σ_s = Hikari.RGBSpectrum(0f0))) :
            Raycore.SetKey()
        Hikari.sync!(scene)
        film = Hikari.Film(backend, Hikari.Film(Point2f(8, 8)))
        camera = Hikari.PerspectiveCamera(Point3f(0, -3, 1), Point3f(0), film; fov = 50f0)
        vp = Hikari.VolPath(samples = 1, max_depth = 2)
        # An unbounded vacuum has no interface for the camera probe to find.
        # Supply its cached answer to exercise a medium ray whose trace misses.
        vp.initial_medium_key = key
        vp.initial_medium_camera_pos = Hikari.get_camera_position(camera)
        return (; scene, film, camera, vp)
    end
    vacuum = environment_frame(false)
    medium = environment_frame(true)
    prefix = nothing
    try
        for f in (vacuum, medium)
            Hikari.render!(f.vp, f.scene, f.film, f.camera)
        end
        @test Array(medium.film.framebuffer) == Array(vacuum.film.framebuffer)
        @test any(c -> c.r > 0f0, Array(medium.film.framebuffer))
        state = medium.vp.state
        @test state.escaped_queue.capacity == state.medium_sample_queue.capacity
        @test any(pp -> pp.pass.name == "escaped", state.plans.sample.passes)
        g = Mantle.Graph(dev)
        found = false
        for pp in state.plans.sample.passes
            for d in pp.pass.dispatches
                Mantle.dispatch!(g, d.kernel, d.args, d.ndrange; group = d.group, name = pp.pass.name)
            end
            if pp.pass.name == "medium-sample"
                found = true
                break
            end
        end
        @test found
        prefix = Mantle.record!(Mantle.Plan(g))
        Mantle.run!(prefix)
        Mantle.waitidle(dev)
        @test only(Array(state.escaped_queue.size)) == 64
    finally
        prefix === nothing || Mantle.free!(prefix)
        for f in (vacuum, medium)
            close(f.vp)
            Hikari.free!(f.film)
        end
    end
end

@testset "an empty scene renders ($(hw ? "hardware" : "software") traversal)" for hw in (false, true)
    backend = Mantle.defaultbackend()
    # Ungated, as RayMakie's own `hw_accel = true` default is: every backend
    # Hikari runs on has hardware traversal.
    scene = Hikari.Scene(; backend, hw_accel = hw)
    # A light and no geometry: every ray escapes, and the kernel that decides
    # that has to compile.
    push!(scene, Hikari.AmbientLight(Hikari.RGBSpectrum(0.5f0)))
    Hikari.sync!(scene)
    res = 16
    film = Hikari.Film(backend, Hikari.Film(Point2f(res, res)))
    cam = Hikari.PerspectiveCamera(Point3f(0, -3, 1), Point3f(0, 0, 0), film; fov = 50f0)
    vp = Hikari.VolPath(samples = 1, max_depth = 2, hw_accel = hw)
    Hikari.render!(vp, scene, film, cam)
    px = Array(film.framebuffer)
    @test size(px) == (res, res)
    @test all(c -> isfinite(c.r) && isfinite(c.g) && isfinite(c.b), px)
end
