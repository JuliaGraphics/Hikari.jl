
@testset "LanczosSincFilter" begin
    l = Hikari.LanczosSincFilter(Point2f(4f0), 3f0)
    @test l(Point2f(0f0)) ≈ 1f0
    @test l(Point2f(4f0)) < 1f-6
    @test l(Point2f(5f0)) ≈ 0f0
end

@testset "Film" begin
    filter = Hikari.LanczosSincFilter(Point2f(4f0), 3f0)
    film = Hikari.Film(
        Point2f(1920f0, 1080f0);
        filter=filter, crop_bounds=Hikari.Bounds2(Point2f(0f0), Point2f(1f0)),
        diagonal=35f0, scale=1f0,
    )
    @test size(film.framebuffer) == (1080, 1920)
    @test size(film.albedo) == (1080, 1920)
    @test size(film.normal) == (1080, 1920)
    @test size(film.depth) == (1080, 1920)
    @test size(film.postprocess) == (1080, 1920)
    @test film.iteration_index[] == Int32(0)

    # clear! resets iteration index
    film.iteration_index[] = Int32(42)
    Hikari.clear!(film)
    @test film.iteration_index[] == Int32(0)

    # free! doesn't crash
    small_film = Hikari.Film(Point2f(16, 16))
    Hikari.free!(small_film)
    Hikari.free!(small_film)  # double free is safe
end

@testset "tonemap names" begin
    # One mapping for the film and for a rasteriser drawing into the same picture.
    @test Hikari.tonemap_code(nothing) == Hikari.TONEMAP_NONE
    @test Hikari.tonemap_code(:aces) == Hikari.TONEMAP_ACES
    @test Hikari.tonemap_code(:reinhard) == Hikari.TONEMAP_REINHARD
    @test Hikari.tonemap_code(:reinhard_extended) == Hikari.TONEMAP_REINHARD_EXT
    @test Hikari.tonemap_code(:uncharted2) == Hikari.TONEMAP_UNCHARTED2
    @test Hikari.tonemap_code(:filmic) == Hikari.TONEMAP_FILMIC
    # A typo used to mean "no tonemapping": a washed-out picture and no error.
    @test_throws ArgumentError Hikari.tonemap_code(:acse)
end

@testset "Perspective Camera" begin
    filter = Hikari.LanczosSincFilter(Point2f(4f0), 3f0)
    film = Hikari.Film(
        Point2f(1920f0, 1080f0);
        filter=filter, crop_bounds=Hikari.Bounds2(Point2f(0f0), Point2f(1f0)),
        diagonal=35f0, scale=1f0,
    )
    camera = Hikari.PerspectiveCamera(
        Hikari.translate(Vec3f(0)), Hikari.Bounds2(Point2f(0), Point2f(10)),
        0f0, 1f0, 0f0, 700f0, 45f0, film,
    )

    sample1 = Hikari.CameraSample(Point2f(1f0), Point2f(1f0), 0f0)
    ray1, contribution = Hikari.generate_ray(camera, sample1)
    sample2 = Hikari.CameraSample(
        Point2f(film.resolution[1]), Point2f(film.resolution[2]), 0f0,
    )
    ray2, contribution = Hikari.generate_ray(camera, sample2)

    @test contribution == 1f0
    @test ray1.o == ray2.o == Point3f(0f0)
    @test ray1.time == ray2.time == camera.core.core.shutter_open
    @test ray1.d[1] < ray2.d[1] && ray1.d[2] < ray2.d[2]
    @test argmax(abs.(ray1.d)) == 3
    @test argmax(abs.(ray2.d)) in (2, 3)

    if isdefined(Hikari, :generate_ray_differential)
        ray_differential, contribution = Hikari.generate_ray_differential(camera, sample1)
        @test ray_differential.has_differentials
        @test ray_differential.o == Point3f(0f0)
        @test ray_differential.d ≈ Point3f(ray1.d)
    end
end

# The look-at constructor used a [-1, 1]² screen window for every film, so a wide
# image was stretched sideways by its aspect ratio (spheres rendered as ellipses).
@testset "Perspective Camera: square pixels on a non-square film" begin
    for (w, h) in ((200, 100), (100, 200))
        film = Hikari.Film(Point2f(w, h))
        camera = Hikari.PerspectiveCamera(Point3f(0, 0, 5), Point3f(0), film; fov = 40)
        direction(x, y) = Hikari.generate_ray(camera, Hikari.CameraSample(Point2f(x, y), Point2f(0.5), 0f0))[1].d
        centre = direction(w / 2, h / 2)
        step_x = acos(clamp(dot(centre, direction(w / 2 + 1, h / 2)), -1, 1))
        step_y = acos(clamp(dot(centre, direction(w / 2, h / 2 + 1)), -1, 1))
        @test step_x ≈ step_y rtol = 1e-3
        # `fov` spans the shorter side: its edges are 20° off the axis.
        short_edge = w >= h ? direction(w / 2, 0) : direction(0, h / 2)
        @test rad2deg(acos(dot(centre, short_edge))) ≈ 20 rtol = 1e-2
    end
end
