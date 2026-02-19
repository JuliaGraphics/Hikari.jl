# VolPath Integration Test
#
# Renders a Cornell-box-like scene with the VolPath integrator on CPU
# and validates the output.

using GeometryBasics: normal_mesh, Tesselation

@testset "VolPath Integration: Cornell Box Render" begin
    # =========================================================================
    # Scene Setup
    # =========================================================================

    # Materials
    white = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.73f0, 0.73f0, 0.73f0))
    red = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.65f0, 0.05f0, 0.05f0))
    green = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.12f0, 0.45f0, 0.15f0))
    glass = Hikari.GlassMaterial(
        Kr=Hikari.RGBSpectrum(1f0),
        Kt=Hikari.RGBSpectrum(1f0),
        index=1.5f0,
    )

    # Fog medium inside glass sphere
    fog = Hikari.HomogeneousMedium(
        σ_a=Hikari.RGBSpectrum(0.01f0),
        σ_s=Hikari.RGBSpectrum(0.3f0),
        Le=Hikari.RGBSpectrum(0f0),
        g=0.3f0,
    )
    glass_with_fog = Hikari.MediumInterface(glass; inside=fog, outside=nothing)

    # Conductor sphere (gold)
    gold = Hikari.ConductorMaterial(
        eta=Hikari.RGBSpectrum(0.15557f0, 0.42415f0, 1.3831f0),
        k=Hikari.RGBSpectrum(3.6024f0, 2.4721f0, 1.9155f0),
    )

    # Box dimensions
    box_size = 2f0
    half = box_size / 2

    # Build scene
    scene = Hikari.Scene()

    # Helper: tessellate and push
    function add_mesh!(scene, prim, material)
        prim_tess = prim isa Sphere ? Tesselation(prim, 32) : prim
        mesh = normal_mesh(prim_tess)
        push!(scene, mesh, material)
    end

    # Walls
    add_mesh!(scene, Rect3f(Vec3f(-half, 0, -half), Vec3f(box_size, 0.01f0, box_size)), white)  # floor
    add_mesh!(scene, Rect3f(Vec3f(-half, 0, half - 0.01f0), Vec3f(box_size, box_size, 0.01f0)), white)  # back
    add_mesh!(scene, Rect3f(Vec3f(-half, 0, -half), Vec3f(0.01f0, box_size, box_size)), red)   # left
    add_mesh!(scene, Rect3f(Vec3f(half - 0.01f0, 0, -half), Vec3f(0.01f0, box_size, box_size)), green)  # right

    # Glass sphere with fog (left side)
    add_mesh!(scene, Sphere(Point3f(-0.4f0, 0.4f0, 0f0), 0.35f0), glass_with_fog)

    # Gold sphere (right side)
    add_mesh!(scene, Sphere(Point3f(0.4f0, 0.35f0, 0f0), 0.3f0), gold)

    # Light
    push!(scene, Hikari.PointLight(Point3f(0f0, 1.8f0, 0f0), Hikari.RGBSpectrum(15f0)))

    Hikari.sync!(scene)

    @test scene isa Hikari.Scene

    # =========================================================================
    # Camera & Film
    # =========================================================================

    resolution = Point2f(64, 64)  # Small for fast testing
    film = Hikari.Film(resolution)
    camera = Hikari.PerspectiveCamera(
        Point3f(0f0, 1f0, -3.5f0), Point3f(0f0, 1f0, 0f0), film; fov=40f0,
    )
    Hikari.clear!(film)

    @test film.resolution == resolution

    # =========================================================================
    # Render
    # =========================================================================

    integrator = Hikari.VolPath(samples=4, max_depth=4)
    integrator(scene, film, camera)
    @test true  # render completed without error

    # =========================================================================
    # Validate output
    # =========================================================================

    img = Hikari.postprocess!(film; exposure=1.0f0, tonemap=:aces, gamma=2.2f0)
    img_array = Array(img)

    @testset "Output sanity checks" begin
        # Image has correct size
        @test size(img_array) == (64, 64)

        # Image has non-zero pixels (something was rendered)
        @test any(px -> (px.r + px.g + px.b) > 0f0, img_array)

        # No NaN or Inf values
        @test !any(px -> isnan(px.r) || isnan(px.g) || isnan(px.b), img_array)
        @test !any(px -> isinf(px.r) || isinf(px.g) || isinf(px.b), img_array)

        # Mean pixel value is reasonable (not all black, not blown out)
        mean_val = sum(px -> (px.r + px.g + px.b) / 3f0, img_array) / length(img_array)
        @test mean_val > 0.001f0   # Not all black
        @test mean_val < 10f0      # Not blown out
    end
end
