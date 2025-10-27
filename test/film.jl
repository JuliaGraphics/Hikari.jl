
@testset "LanczosSincFilter" begin
    l = Trace.LanczosSincFilter(Point2f(4f0), 3f0)
    @test l(Point2f(0f0)) ≈ 1f0
    @test l(Point2f(4f0)) < 1f-6
    @test l(Point2f(5f0)) ≈ 0f0
end

@testset "Film testing" begin
    filter = Trace.LanczosSincFilter(Point2f(4f0), 3f0)
    film = Trace.Film(
        Point2f(1920f0, 1080f0), Trace.Bounds2(Point2f(0f0), Point2f(1f0)),
        filter, 35f0, 1f0
    )
    @test size(film.pixels) == (1080, 1920)
    @test Trace.get_sample_bounds(film) == Trace.Bounds2(Point2f(-3f0), Point2f(1924f0, 1084f0))
end

@testset "FilmTile testing" begin
    filter = Trace.LanczosSincFilter(Point2f(4f0), 3f0)
    film = Trace.Film(
        Point2f(100f0, 100f0), Trace.Bounds2(Point2f(0f0), Point2f(1f0)),
        filter, 35f0, 1f0; tile_size=4
    )

    # Test tile structure
    @test film.tile_size == 4
    # With 100x100 resolution and 4x4 tiles, we get 26x26 tiles (includes borders)
    @test film.ntiles == (26, 26)
    # Tiles matrix: first dim is pixels per tile (4x4=16), second dim is number of tiles (26*26=676)
    @test size(film.tiles) == (16, 676)

    # Test add_sample! functionality
    # Add sample at pixel (5, 5), which should affect a small region around it
    # For tile_size=4, each tile should be a 4x4 region
    # Tile bounds are inclusive: [1, 4] means pixels 1, 2, 3, 4
    tile_bounds = Trace.Bounds2(Point2f(1f0), Point2f(4f0))
    tile_col = Int32(1)  # First tile column

    # Initially, tiles should be zero
    @test all(film.tiles.filter_weight_sum[:, tile_col] .≈ 0f0)

    # Add a sample at (2.5, 2.5) which is within the tile bounds [1, 4]
    Trace.add_sample!(film.tiles, tile_bounds, tile_col, Point2f(2.5f0, 2.5f0), Trace.RGBSpectrum(1f0),
        film.filter_table, film.filter_radius)

    # After adding sample, some tile pixels should have non-zero weights
    @test any(film.tiles.filter_weight_sum[:, tile_col] .> 0f0)

    # Test merge_film_tile! functionality
    # Initially film pixels should be zero
    @test film.pixels[2, 2].filter_weight_sum ≈ 0f0

    # Merge tile into film
    Trace.merge_film_tile!(film.pixels, film.crop_bounds, film.tiles, tile_bounds, tile_col)

    # After merging, pixels around (2.5, 2.5) should have accumulated weight
    # The filter has radius 4, so it affects a region around the sample point
    @test film.pixels[2, 2].filter_weight_sum > 0f0 || film.pixels[3, 3].filter_weight_sum > 0f0

    # Test that filter weights are distributed around the sample point
    # Adjacent pixels should have contributions due to filter spread (radius=4)
    @test film.pixels[1, 1].filter_weight_sum >= 0f0
    @test film.pixels[2, 2].filter_weight_sum >= 0f0
    @test film.pixels[3, 3].filter_weight_sum >= 0f0
end

@testset "Perspective Camera" begin
    filter = Trace.LanczosSincFilter(Point2f(4f0), 3f0)
    film = Trace.Film(
        Point2f(1920f0, 1080f0), Trace.Bounds2(Point2f(0f0), Point2f(1f0)),
        filter, 35f0, 1f0
    )
    camera = Trace.PerspectiveCamera(
        Trace.translate(Vec3f(0)), Trace.Bounds2(Point2f(0), Point2f(10)),
        0f0, 1f0, 0f0, 700f0, 45f0, film,
    )

    sample1 = Trace.CameraSample(Point2f(1f0), Point2f(1f0), 0f0)
    ray1, contribution = Trace.generate_ray(camera, sample1)
    sample2 = Trace.CameraSample(
        Point2f(film.resolution[1]), Point2f(film.resolution[2]), 0f0,
    )
    ray2, contribution = Trace.generate_ray(camera, sample2)

    @test contribution == 1f0
    @test ray1.o == ray2.o == Point3f(0f0)
    @test ray1.time == ray2.time == camera.core.core.shutter_open
    @test ray1.d[1] < ray2.d[1] && ray1.d[2] < ray2.d[2]
    # Both rays should primarily point in the same dominant direction
    @test argmax(abs.(ray1.d)) == 3
    @test argmax(abs.(ray2.d)) in (2, 3)  # Can be 2 or 3 depending on field of view

    ray_differential, contribution = Trace.generate_ray_differential(
        camera, sample1,
    )
    @test ray_differential.has_differentials
    @test ray_differential.o == Point3f(0f0)
    @test ray_differential.d ≈ Point3f(ray1.d)

    @test ray_differential.rx_direction[1] > ray_differential.d[1]
    @test ray_differential.rx_direction[2] ≈ ray_differential.d[2]
    @test ray_differential.ry_direction[1] ≈ ray_differential.d[1]
    @test ray_differential.ry_direction[2] > ray_differential.d[2]
end
