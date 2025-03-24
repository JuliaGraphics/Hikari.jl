using Trace: Pixel, get_sample_bounds, diagonal, FilmTilePixel, RGBSpectrum, to_XYZ
using StructArrays
using Trace, GeometryBasics
using Trace: u_int32, Bounds2, inclusive_sides
using Trace
using Trace: XYZ_to_RGB, to_XYZ
using Trace: li_iterative, get_camera_sample, generates_ray_differential, scale_differentials
import KernelAbstractions as KA
using KernelAbstractions
using CUDA


function generate_filter_table(filter)
    filter_table_width = 16
    filter_table = Matrix{Float32}(undef, filter_table_width, filter_table_width)
    r = filter.radius ./ filter_table_width
    for y in 0:filter_table_width-1, x in 0:filter_table_width-1
        p = Point2f((x + 0.5f0) * r[1], (y + 0.5f0) * r[2])
        filter_table[y+1, x+1] = filter(p)
    end

    point = Point2f(filter_table_width)
    # Compute sample's raster bounds.
    discrete_point = point .- 0.5f0
    # Compute sample radius around point
    p0 = ceil.(Int, discrete_point .- filter.radius)
    p1 = floor.(Int, discrete_point .+ filter.radius) .+ 1
    # Make sure we're inbounds
    inv_radius = 1.0f0 ./ filter.radius
    # Precompute x & y filter offsets.
    offsets_x = Trace.filter_offsets(p0[1], p1[1], discrete_point[1], inv_radius[1], filter_table_width)
    offsets_y = Trace.filter_offsets(p0[2], p1[2], discrete_point[2], inv_radius[2], filter_table_width)
    # Loop over filter support & add sample to pixel array.
    xrange = p0[1]:p1[1]
    yrange = p0[2]:p1[2]
    weights = zeros(Float32, length(xrange), length(yrange))
    for i in 1:length(xrange), j in 1:length(yrange)
        w = filter_table[offsets_y[j], offsets_x[i]]
        weights[i, j] = w
    end
    return weights
end

"""
Point in (x, y) format.
"""
@inline function get_pixel_index(crop_bounds, p::Point2)
    i1, i2 = u_int32.((p .- crop_bounds.p_min .+ 1.0f0))
    return CartesianIndex(i1, i2)
end

@inline function merge_film_tile!(f::AbstractMatrix{Pixel}, crop_bounds::Bounds2, ft::AbstractMatrix{FilmTilePixel}, tile::Bounds2, tile_col::Int32)
    ft_contrib_sum = ft.contrib_sum
    ft_filter_weight_sum = ft.filter_weight_sum
    f_xyz = f.xyz
    f_filter_weight_sum = f.filter_weight_sum
    linear = Int32(1)
    @inbounds for pixel in tile
        f_idx = get_pixel_index(crop_bounds, pixel)
        f_xyz[f_idx] += to_XYZ(ft_contrib_sum[linear, tile_col])
        f_filter_weight_sum[f_idx] += ft_filter_weight_sum[linear, tile_col]
        linear += Int32(1)
    end
    return
end


@inline function get_tile_index(bounds::Bounds2, p::Point2)
    j, i = u_int32.((p .- bounds.p_min .+ 1.0f0))
    ncols = u_int32(inclusive_sides(bounds)[1])
    return (i - Int32(1)) * ncols + j
end

@inline function add_sample!(
        tiles::AbstractMatrix{FilmTilePixel}, tile::Bounds2, tile_column::Int32, point::Point2f, spectrum::RGBSpectrum,
        filter_table, filter_radius::Point2f, sample_weight::Float32=1.0f0,
    )
    # Compute sample's raster bounds.
    discrete_point = point .- 0.5f0
    # Compute sample radius around point
    p0 = u_int32.(ceil.(discrete_point .- filter_radius))
    p1 = u_int32.(floor.(discrete_point .+ filter_radius)) .+ Int32(1)
    # Make sure we're inbounds
    pmin = u_int32.(tile.p_min)
    pmax = u_int32.(tile.p_max)
    p0 = max.(p0, max.(pmin, Point2{Int32}(1)))::Point2{Int32}
    p1 = min.(p1, pmax)::Point2{Int32}
    # Loop over filter support & add sample to pixel array.
    contrib_sum = tiles.contrib_sum
    filter_weight_sum = tiles.filter_weight_sum
    xrange = p0[1]:p1[1]
    yrange = p0[2]:p1[2]
    xn = length(xrange) % Int32
    yn = length(yrange) % Int32
    @inbounds for i in Int32(1):xn, j in Int32(1):yn
        x = xrange[i]
        y = yrange[j]
        w = filter_table[i, j]
        idx = get_tile_index(tile, Point2(x, y))
        contrib_sum[idx, tile_column] += spectrum * sample_weight * w
        filter_weight_sum[idx, tile_column] += w
    end
end

@inline function filter_offset(x, discrete_point, inv_filter_radius, filter_table_width)
    fx = abs((x - discrete_point) * inv_filter_radius * filter_table_width)
    return clamp(u_int32(ceil(fx)), Int32(1), Int32(filter_table_width))  # TODO is clipping ok?
end


function sample_kernel_inner!(
        tiles, tile, tile_column::Int32, max_depth::Int32,
        scene, t_sampler, camera, pixel, spp_sqr, filter_table, filter_radius::Point2f, resolution::Point2f
    )
    px, py = pixel
    cpix = Point2f(py, resolution[1] - px)
    for _ in t_sampler.samples_per_pixel
        camera_sample = @inline get_camera_sample(t_sampler, cpix)
        ray, ω = generate_ray_differential(camera, camera_sample)
        ray = scale_differentials(ray, spp_sqr)
        l = RGBSpectrum(0f0)
        if ω > 0.0f0
            l = li_iterative(t_sampler, max_depth, ray, scene)
        end
        # TODO check l for invalid values
        if isnan(l)
            l = RGBSpectrum(0f0)
        end
        add_sample!(
            tiles, tile, tile_column, pixel, l,
            filter_table, filter_radius, ω,
        )
    end
end

struct Whitten5{TMat<:AbstractMatrix{FilmTilePixel},PMat<:AbstractMatrix{Pixel}}
    tiles::TMat
    pixel::PMat
    sample_bounds::Bounds2
    crop_bounds::Bounds2
    resolution::Point2f
    ntiles::NTuple{2, Int32}
    tile_size::Int32
    fiter_table::Matrix{Float32}
    filter_radius::Point2f
    sampler::Trace.UniformSampler
    max_depth::Int32
end

function Trace.to_gpu(ArrayType, w::Whitten5; preserve=[])
    Whitten5(
        KA.adapt(ArrayType, w.tiles),
        KA.adapt(ArrayType, w.pixel),
        w.sample_bounds,
        w.crop_bounds,
        w.resolution,
        w.ntiles,
        w.tile_size,
        w.fiter_table,
        w.filter_radius,
        w.sampler,
        w.max_depth,
    )
end

function Whitten5(film; samples_per_pixel=8, tile_size=16, max_depth=5)
    sample_bounds = get_sample_bounds(film)
    sample_extent = diagonal(sample_bounds)
    resolution = film.resolution
    n_tiles = Int64.(floor.((sample_extent .+ tile_size) ./ tile_size))
    wtiles, htiles = n_tiles .- 1
    filter_radius = film.filter.radius
    filter_table = generate_filter_table(film.filter)
    ntiles = (wtiles) * (htiles)
    tile_size_l = tile_size * tile_size
    contrib_sum = RGBSpectrum.(zeros(Vec3f, tile_size_l, ntiles))
    filter_weight_sum = zeros(Float32, tile_size_l, ntiles)
    tiles = StructArray{FilmTilePixel}(; contrib_sum, filter_weight_sum)
    sampler = Trace.UniformSampler(samples_per_pixel)
    li = LinearIndices((wtiles, htiles))
    @assert length(li) == ntiles
    Whitten5(tiles, film.pixels, sample_bounds, film.crop_bounds, resolution, Int32.((wtiles, htiles)), Int32(tile_size), filter_table, filter_radius, sampler, Int32(max_depth))
end

"""
Render scene.
"""
function (w::Whitten5)(scene::Trace.Scene, camera)
    tiles = w.tiles
    resolution = w.resolution
    filter_table = w.fiter_table
    filter_radius = w.filter_radius
    sampler = w.sampler
    max_depth = w.max_depth
    sample_bounds = w.sample_bounds
    tile_linear = LinearIndices(w.ntiles)
    spp_sqr = 1.0f0 / √Float32(sampler.samples_per_pixel)
    Threads.@threads for tile_xy in CartesianIndices(w.ntiles)
        i, j = Int32.(Tuple(tile_xy)) .- Int32(1)
        tile_column = tile_linear[tile_xy]
        tile_start = Point2f(i, j)
        tb_min = (sample_bounds.p_min .+ tile_start .* w.tile_size) .+ 1
        tb_max = min.(tb_min .+ (w.tile_size - 1), sample_bounds.p_max)
        tile_bounds = Bounds2(tb_min, tb_max)
        spp_sqr = 1.0f0 / √Float32(sampler.samples_per_pixel)
        for pixel in tile_bounds
            sample_kernel_inner!(tiles, tile_bounds, tile_column, max_depth, scene, sampler, camera, pixel, spp_sqr, filter_table, filter_radius, resolution)
        end
        merge_film_tile!(film.pixels, film.crop_bounds, tiles, tile_bounds, Int32(tile_column))
    end
end

@kernel function sample_kernel2!(
        pixels, tiles, tile_size::Int32,
        sample_bounds,
        max_depth::Int32, scene, camera,
        sampler, spp_sqr, filter_table,
        filter_radius::Point2f, resolution, crop_bounds
    )
    tile_xy = @index(Global, Cartesian)
    i, j = u_int32.(Tuple(tile_xy)) .- Int32(1)
    tc = @index(Global)
    tile_column = tc % Int32
    tile_start = Point2f(i, j)
    tb_min = u_int32.(sample_bounds.p_min .+ tile_start .* tile_size) .+ Int32(1)
    tb_max = u_int32.(min.(tb_min .+ (tile_size - Int32(1)), sample_bounds.p_max))
    tile_bounds = Bounds2(tb_min, tb_max)
    for x in tb_min[1]:tb_max[1]
        for y in tb_min[2]:tb_max[2]
            pixel = Point2f(x, y)
            @inline sample_kernel_inner!(tiles, tile_bounds, tile_column, max_depth, scene, sampler, camera, pixel, spp_sqr, filter_table, filter_radius, resolution)
        end
    end
    merge_film_tile!(pixels, crop_bounds, tiles, tile_bounds, tile_column)
end

using AMDGPU

function launch_trace_image!(w::Whitten5, camera, scene)
    backend = KA.get_backend(w.tiles.contrib_sum)
    kernel! = sample_kernel2!(backend, (16, 16))
    spp_sqr = 1.0f0 / √Float32(w.sampler.samples_per_pixel)
    static_filter_table = Mat{size(w.fiter_table)...}(w.fiter_table)
    # open("../trace-tiles.ir", "w") do io
        # @device_code_llvm io kernel!(
        kernel!(
            w.pixel, w.tiles, w.tile_size, w.sample_bounds, w.max_depth,
            scene, camera, w.sampler, spp_sqr,
            static_filter_table, w.filter_radius,
            w.resolution, w.crop_bounds, ndrange=w.ntiles
        )
    # end
    KA.synchronize(backend)
    return w
end

include("./../docs/code/basic-scene.jl")

begin
    # Trace.clear!(film)
    w = Whitten5(film; samples_per_pixel=1, max_depth=1)
    @time launch_trace_image!(w, cam, scene)
    Trace.to_framebuffer!(film.framebuffer, w.pixel)
end

begin
    Trace.clear!(film)
    p = []
    w_gpu = Trace.to_gpu(ROCArray, Whitten5(film; samples_per_pixel=1, max_depth=1); preserve=p)
    gpu_scene = Trace.to_gpu(ROCArray, scene; preserve=p)
    GC.@preserve p begin
        @time launch_trace_image!(w_gpu, cam, gpu_scene)
        Trace.to_framebuffer!(film.framebuffer, KA.adapt(Array, w_gpu.pixel))
    end
end




#=
@code_warntype merge_film_tile!(film.pixels, film.crop_bounds, tiles, Bounds2(Point2f(1), Point2f(16)), Int32(1))

@code_warntype add_sample!(
    tiles, Bounds2(Point2f(1), Point2f(16)), Int32(1), Point2f(1), RGBSpectrum(1f0), filter_table, Point2f(1), 1.0f0
)
@code_warntype sample_kernel_inner!(tiles, Bounds2(Point2f(1), Point2f(16)), Int32(1), Int32(5), scene, w.sampler, cam, Point2f(1), 1f0, filter_table, filter_radius, resolution)
@code_warntype sample_kernel2!(
    film.pixels, tiles, Int32(16),
    sample_bounds,
    Int32(5), scene, cam,
    w.sampler, 1.0f0, filter_table,
    filter_radius, resolution, w.crop_bounds
)
=#
