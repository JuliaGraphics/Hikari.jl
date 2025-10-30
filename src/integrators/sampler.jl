using KernelAbstractions.Extras.LoopInfo: @unroll


abstract type SamplerIntegrator <: Integrator end

struct WhittedIntegrator{C<: Camera, S <: AbstractSampler} <: SamplerIntegrator
    camera::C
    sampler::S
    max_depth::Int64
end

@inline function sample_kernel_inner!(
        tiles, tile, tile_column::Int32, resolution::Point2f, max_depth::Int32,
        scene, sampler, camera, pixel, spp_sqr, filter_table,
        filter_radius::Point2f
    )
    campix = Point2f(pixel[2], resolution[1] - pixel[1])
    for _ in 1:sampler.samples_per_pixel
        camera_sample = @inline get_camera_sample(sampler, campix)
        ray, ω = @inline generate_ray_differential(camera, camera_sample)
        ray = @inline scale_differentials(ray, spp_sqr)
        l = RGBSpectrum(0.0f0)
        if ω > 0.0f0
            l = @inline li_iterative(sampler, max_depth, ray, scene)
        end
        # TODO check l for invalid values
        if isnan(l)
            l = RGBSpectrum(0.0f0)
        end
        @inline add_sample!(
            tiles, tile, tile_column, pixel, l,
            filter_table, filter_radius, ω,
        )
    end
end

@kernel function whitten_kernel!(pixels, crop_bounds, sample_bounds, tiles, tile_size, max_depth, scene, sampler, camera, filter_table, filter_radius)
    _tile_xy = @index(Global, Cartesian)
    linear_idx = @index(Global)

    tile_xy = map(u_int32, Tuple(_tile_xy))
    tile_column = u_int32(linear_idx)
    i, j = tile_xy .- Int32(1)
    tile_start = Point2f(i, j)
    tb_min = (sample_bounds.p_min .+ tile_start .* tile_size) .+ Int32(1)
    tb_max = min.(tb_min .+ (tile_size .- Int32(1)), sample_bounds.p_max)
    tile_bounds = Bounds2(tb_min, tb_max)
    spp_sqr = 1.0f0 / √Float32(sampler.samples_per_pixel)

    # Explicit loop instead of iterating over Bounds2 to avoid GPU allocation issues
    px_size = Point2f(size(pixels))
    for py in u_int32(tb_min[2]):u_int32(tb_max[2])
        for px in u_int32(tb_min[1]):u_int32(tb_max[1])
            pixel = Point2f(px, py)
            @inline sample_kernel_inner!(
                tiles, tile_bounds, tile_column, px_size,
                max_depth, scene, sampler, camera,
                pixel, spp_sqr, filter_table, filter_radius
            )
        end
    end
    @inline merge_film_tile!(pixels, crop_bounds, tiles, tile_bounds, Int32(tile_column))
end

"""
Render scene using KernelAbstractions.
"""
function (i::SamplerIntegrator)(scene::Scene, film, camera)
    # TODO visualize tile bounds to see if they overlap
    sample_bounds = get_sample_bounds(film)
    tile_size = film.tile_size
    filter_radius = film.filter_radius
    filter_table = film.filter_table
    tiles = film.tiles
    sampler = i.sampler
    max_depth = Int32(i.max_depth)
    backend = KA.get_backend(film.tiles.contrib_sum)
    kernel! = whitten_kernel!(backend, (Int(tile_size), Int(tile_size)))
    s_filter_table = Mat{size(filter_table)...}(filter_table)
    kernel!(
        film.pixels, film.crop_bounds, sample_bounds,
        tiles, tile_size,
        max_depth, scene, sampler,
        camera, s_filter_table, filter_radius;
        ndrange=film.ntiles
    )
    KA.synchronize(backend)
    to_framebuffer!(film, 1f0)
end

"""
Render scene using Julia threads (one thread per tile).
"""
function integrator_threaded(i::SamplerIntegrator, scene::Scene, film, camera)
    sample_bounds = get_sample_bounds(film)
    tile_size = film.tile_size
    filter_radius = film.filter_radius
    filter_table = film.filter_table
    tiles = film.tiles
    sampler = i.sampler
    max_depth = Int32(i.max_depth)
    pixels = film.pixels
    crop_bounds = film.crop_bounds
    spp_sqr = 1.0f0 / √Float32(sampler.samples_per_pixel)

    # Calculate tile grid dimensions
    ntiles_x, ntiles_y = film.ntiles

    # Process all tiles in parallel using threads
    @sync for tile_j in 1:ntiles_y
        Threads.@spawn for tile_i in 1:ntiles_x
            # Calculate tile index and bounds
            linear_idx = (tile_j - 1) * ntiles_x + tile_i
            tile_column = Int32(linear_idx)

            i_idx = Int32(tile_i - 1)
            j_idx = Int32(tile_j - 1)
            tile_start = Point2f(i_idx, j_idx)

            tb_min = (sample_bounds.p_min .+ tile_start .* tile_size) .+ Int32(1)
            tb_max = min.(tb_min .+ (tile_size .- Int32(1)), sample_bounds.p_max)
            tile_bounds = Bounds2(tb_min, tb_max)

            # Process all pixels in this tile
            for py in Int32(tb_min[2]):Int32(tb_max[2])
                for px in Int32(tb_min[1]):Int32(tb_max[1])
                    pixel = Point2f(px, py)
                    sample_kernel_inner!(
                        tiles, tile_bounds, tile_column, Point2f(size(pixels)),
                        max_depth, scene, sampler, camera,
                        pixel, spp_sqr, filter_table, filter_radius
                    )
                end
            end
            # Merge this tile into the film
            merge_film_tile!(pixels, crop_bounds, tiles, tile_bounds, tile_column)
        end
    end

    to_framebuffer!(film, 1f0)
end

# MaterialScene now returns material directly from intersect!
# These functions are kept for backward compatibility but not used anymore
function get_material(bvh::BVHAccel, shape::Triangle)
    materials = bvh.materials
    @_inbounds if shape.material_idx == 0
        return materials[1]
    else
        return materials[shape.material_idx]
    end
end
function get_material(scene::Scene, shape::Triangle)
    get_material(scene.aggregate, shape)
end
function get_material(ms::MaterialScene, shape::Triangle)
    materials = ms.materials
    @_inbounds if shape.material_idx == 0
        return materials[1]
    else
        return materials[shape.material_idx]
    end
end

@inline function only_light(lights, ray)
    l = RGBSpectrum(0.0f0)
    Base.Cartesian.@nexprs 8 i -> begin
        if i <= length(lights)
            light = lights[i]
            l += @inline le(light, ray)
        end
    end
    return l
end

@inline function light_contribution(l, lights, wo, scene, bsdf, sampler, si)
    core = si.core
    n = si.shading.n
    # Why can't I use KernelAbstraction.@unroll here, when in Trace.jl?
    # Worked just fined when the function was defined outside
    Base.Cartesian.@nexprs 8 i -> begin
        if i <= length(lights)
            @_inbounds light = lights[i]
            sampled_li, wi, pdf, tester = @inline sample_li(light, core, get_2d(sampler))
            if !(is_black(sampled_li) || pdf ≈ 0.0f0)
                f = @inline bsdf(wo, wi)
                if !is_black(f) && @inline unoccluded(tester, scene)
                    l += f * sampled_li * abs(wi ⋅ n) / pdf
                end
            end
        end
    end
    return l
end

function li(
    sampler, max_depth, ray::RayDifferentials, scene::Scene, depth::Int32,
)::RGBSpectrum

    l = RGBSpectrum(0.0f0)
    # Find closest ray intersection or return background radiance.
    hit, material, si = intersect!(scene, ray)
    lights = scene.lights
    if !hit
        return only_light(lights, ray)
    end
    # Compute emmited & reflected light at ray intersection point.
    # Initialize common variables for Whitted integrator.
    core = si.core
    n = si.shading.n
    wo = core.wo
    # Compute scattering functions for surface interaction.
    si = compute_differentials(si, ray)
    m = material
    if m.type === NO_MATERIAL
        return li(
            sampler, max_depth, RayDifferentials(spawn_ray(si, ray.d)),
            scene, depth,
        )
    end
    bsdf = m(si, false, Radiance)
    # Compute emitted light if ray hit an area light source.
    l += le(si, wo)
    # Add contribution of each light source.
    l = light_contribution(l, lights, wo, scene, bsdf, sampler, si)
    if depth + 1 ≤ max_depth
        # Trace rays for specular reflection & refraction.
        l += specular_reflect(bsdf, sampler, max_depth, ray, si, scene, depth)
        l += specular_transmit(bsdf, sampler, max_depth, ray, si, scene, depth)
    end
    l
end

@inline function specular_reflect(
        bsdf, sampler, max_depth, ray::RayDifferentials,
        si::SurfaceInteraction, scene::Scene, depth::Int32,
    )

    # Compute specular reflection direction `wi` and BSDF value.
    wo = si.core.wo
    type = BSDF_REFLECTION | BSDF_SPECULAR
    wi, f, pdf, sampled_type = sample_f(
        bsdf, wo, get_2d(sampler), type,
    )
    # Return contribution of specular reflection.
    ns = si.shading.n
    if !(pdf > 0.0f0 && !is_black(f) && abs(wi ⋅ ns) != 0.0f0)
        return RGBSpectrum(0.0f0)
    end
    # # Compute ray differential for specular reflection.
    rd = RayDifferentials(spawn_ray(si, wi))
    if ray.has_differentials
        rx_origin = si.core.p + si.∂p∂x
        ry_origin = si.core.p + si.∂p∂y
        # Compute differential reflected directions.
        ∂n∂x = (
            si.shading.∂n∂u * si.∂u∂x
            +
            si.shading.∂n∂v * si.∂v∂x
        )
        ∂n∂y = (
            si.shading.∂n∂u * si.∂u∂y
            +
            si.shading.∂n∂v * si.∂v∂y
        )
        ∂wo∂x = -ray.rx_direction - wo
        ∂wo∂y = -ray.ry_direction - wo
        ∂dn∂x = ∂wo∂x ⋅ ns + wo ⋅ ∂n∂x
        ∂dn∂y = ∂wo∂y ⋅ ns + wo ⋅ ∂n∂y
        rx_direction = wi - ∂wo∂x + 2.0f0 * (wo ⋅ ns) * ∂n∂x + ∂dn∂x * ns
        ry_direction = wi - ∂wo∂y + 2.0f0 * (wo ⋅ ns) * ∂n∂y + ∂dn∂y * ns
        rd = RayDifferentials(rd, rx_origin=rx_origin, ry_origin=ry_origin, rx_direction=rx_direction, ry_direction=ry_direction)
    end
    return f * li(sampler, max_depth, rd, scene, depth + Int32(1)) * abs(wi ⋅ ns) / pdf
end

@inline function specular_transmit(
    bsdf, sampler, max_depth, ray::RayDifferentials,
    surface_intersect::SurfaceInteraction, scene::Scene, depth::Int32,
)

    # Compute specular reflection direction `wi` and BSDF value.
    wo = surface_intersect.core.wo
    type = BSDF_TRANSMISSION | BSDF_SPECULAR
    wi, f, pdf, sampled_type = sample_f(
        bsdf, wo, get_2d(sampler), type,
    )

    ns = surface_intersect.shading.n
    if !(pdf > 0.0f0 && !is_black(f) && abs(wi ⋅ ns) != 0.0f0)
        return RGBSpectrum(0.0f0)
    end
    # TODO shift in ray direction instead of normal?
    rd = RayDifferentials(spawn_ray(surface_intersect, wi))
    if ray.has_differentials
        rx_origin = surface_intersect.core.p + surface_intersect.∂p∂x
        ry_origin = surface_intersect.core.p + surface_intersect.∂p∂y
        # Compute differential transmitted directions.
        ∂n∂x = (
            surface_intersect.shading.∂n∂u * surface_intersect.∂u∂x
            +
            surface_intersect.shading.∂n∂v * surface_intersect.∂v∂x
        )
        ∂n∂y = (
            surface_intersect.shading.∂n∂u * surface_intersect.∂u∂y
            +
            surface_intersect.shading.∂n∂v * surface_intersect.∂v∂y
        )
        # The BSDF stores the IOR of the interior of the object being
        # intersected. Compute the relative IOR by first out by assuming
        # that the ray is entering the object.
        η = 1.0f0 / bsdf.η
        if (ns ⋅ ns) < 0.0f0
            # If the ray isn't entering the object, then we need to invert
            # the relative IOR and negate the normal and its derivatives.
            η = 1.0f0 / η
            ∂n∂x, ∂n∂y, ns = -∂n∂x, -∂n∂y, -ns
        end
        ∂wo∂x = -ray.rx_direction - wo
        ∂wo∂y = -ray.ry_direction - wo
        ∂dn∂x = ∂wo∂x ⋅ ns + wo ⋅ ∂n∂x
        ∂dn∂y = ∂wo∂y ⋅ ns + wo ⋅ ∂n∂y
        μ = η * (wo ⋅ ns) - abs(wi ⋅ ns)
        ν = η - (η * η * (wo ⋅ ns)) / abs(wi ⋅ ns)
        ∂μ∂x = ν * ∂dn∂x
        ∂μ∂y = ν * ∂dn∂y
        rx_direction = wi - η * ∂wo∂x + μ * ∂n∂x + ∂μ∂x * ns
        ry_direction = wi - η * ∂wo∂y + μ * ∂n∂y + ∂μ∂y * ns
        rd = RayDifferentials(rd, rx_origin=rx_origin, ry_origin=ry_origin, rx_direction=rx_direction, ry_direction=ry_direction)
    end
    f * li(sampler, max_depth, rd, scene, depth + Int32(1)) * abs(wi ⋅ ns) / pdf
end


struct Reflect end
struct Transmit end

macro ntuple(N, value)
    expr = :(())
    for i in 1:N
        push!(expr.args, :($(esc(value))))
    end
    return expr
end

macro setindex(N, setindex_expr)
    @assert Meta.isexpr(setindex_expr, :(=))
    index_expr = setindex_expr.args[1]
    @assert Meta.isexpr(index_expr, :ref)
    tuple = index_expr.args[1]
    idx = index_expr.args[2]
    value = setindex_expr.args[2]
    expr = :(())
    for i in 1:N
        push!(expr.args, :(ifelse($i != $(esc(idx)), $(esc(tuple))[$i], $(esc(value)))))
    end
    return :($(esc(tuple)) = $expr)
end

@inline function li_iterative(
        sampler, max_depth::Int32, initial_ray::RayDifferentials, scene::S
    )::RGBSpectrum where {S<:Scene}
    accumulated_l = RGBSpectrum(0.0f0)
    # stack = MVector{8,Tuple{Trace.RayDifferentials,Int32,Trace.RGBSpectrum}}(undef)
    stack = @ntuple(8, (initial_ray, Int32(0), accumulated_l))
    pos = Int32(1)
    # stack[pos] = (initial_ray, Int32(0), accumulated_l)
    @_inbounds while pos > Int32(0)
        (ray, depth, accumulated_l) = stack[pos]
        pos -= Int32(1)
        if depth == max_depth
            continue
        end
        hit, material, si = intersect!(scene, ray)
        lights = scene.lights

        if !hit
            accumulated_l += only_light(lights, ray)
            continue
        end

        core = si.core
        wo = core.wo
        si = compute_differentials(si, ray)
        m = material
        if m.type === NO_MATERIAL
            new_ray = RayDifferentials(spawn_ray(si, ray.d))
            pos += Int32(1)
            @setindex 8 stack[pos] = (new_ray, depth, accumulated_l)
            # stack[pos] = (new_ray, depth, accumulated_l)
            continue
        end

        bsdf = @inline m(si, false, Radiance)
        accumulated_l += @inline le(si, wo)
        accumulated_l = @inline light_contribution(accumulated_l, lights, wo, scene, bsdf, sampler, si)

        if depth + 1 <= max_depth
            rd_reflect, reflect_l = @inline specular(Reflect, bsdf, sampler, ray, si)
            if rd_reflect !== ray && pos < 8
                pos += Int32(1)
                @setindex 8 stack[pos] = (rd_reflect, depth + Int32(1), reflect_l * accumulated_l)
                # stack[pos] = (rd_reflect, depth + Int32(1), reflect_l * accumulated_l)
            end
            rd_transmit, transmit_l = @inline specular(Transmit, bsdf, sampler, ray, si)
            if rd_transmit !== ray && pos < 8
                pos += Int32(1)
                @setindex 8 stack[pos] = (rd_transmit, depth + Int32(1), transmit_l * accumulated_l)
                # stack[pos] = (rd_transmit, depth + Int32(1), transmit_l * accumulated_l)
            end
        end
    end
    return accumulated_l
end


@inline get_type(::Type{Transmit}) = BSDF_TRANSMISSION | BSDF_SPECULAR
@inline get_type(::Type{Reflect}) = BSDF_REFLECTION | BSDF_SPECULAR

@inline function specular(
        type, bsdf, sampler, ray::RayDifferentials,
        si::SurfaceInteraction,
    )::Tuple{RayDifferentials, RGBSpectrum}

    wo = si.core.wo
    wi, f, pdf, sampled_type = @inline sample_f(bsdf, wo, get_2d(sampler), get_type(type))

    ns = si.shading.n
    if !(pdf > 0.0f0 && !is_black(f) && abs(wi ⋅ ns) != 0.0f0)
        return (ray, RGBSpectrum(0.0f0))
    end

    rd = RayDifferentials(spawn_ray(si, wi))
    if ray.has_differentials
        rd = @inline specular_differentials(type, rd, bsdf, si, ray, wo, wi)
    end
    return rd, f * abs(wi ⋅ ns) / pdf
end

@inline function specular_differentials(::Type{Reflect}, rd, bsdf, si, ray, wo, wi)
    ns = si.shading.n
    rx_origin = si.core.p + si.∂p∂x
    ry_origin = si.core.p + si.∂p∂y
    # Compute differential reflected directions.
    ∂n∂x = si.shading.∂n∂u * si.∂u∂x + si.shading.∂n∂v * si.∂v∂x
    ∂n∂y = si.shading.∂n∂u * si.∂u∂y + si.shading.∂n∂v * si.∂v∂y
    ∂wo∂x = -ray.rx_direction - wo
    ∂wo∂y = -ray.ry_direction - wo
    ∂dn∂x = ∂wo∂x ⋅ ns + wo ⋅ ∂n∂x
    ∂dn∂y = ∂wo∂y ⋅ ns + wo ⋅ ∂n∂y
    rx_direction = wi - ∂wo∂x + 2.0f0 * (wo ⋅ ns) * ∂n∂x + ∂dn∂x * ns
    ry_direction = wi - ∂wo∂y + 2.0f0 * (wo ⋅ ns) * ∂n∂y + ∂dn∂y * ns
    return RayDifferentials(rd, rx_origin=rx_origin, ry_origin=ry_origin, rx_direction=rx_direction, ry_direction=ry_direction)
end

@inline function specular_differentials(::Type{Transmit}, rd, bsdf, si, ray, wo, wi)

    ns = si.shading.n
    rx_origin = si.core.p + si.∂p∂x
    ry_origin = si.core.p + si.∂p∂y
    # Compute differential transmitted directions.
    ∂n∂x = si.shading.∂n∂u * si.∂u∂x + si.shading.∂n∂v * si.∂v∂x
    ∂n∂y = si.shading.∂n∂u * si.∂u∂y + si.shading.∂n∂v * si.∂v∂y
    # The BSDF stores the IOR of the interior of the object being
    # intersected. Compute the relative IOR by first out by assuming
    # that the ray is entering the object.
    η = 1.0f0 / bsdf.η
    if (ns ⋅ ns) < 0.0f0
        # If the ray isn't entering the object, then we need to invert
        # the relative IOR and negate the normal and its derivatives.
        η = 1.0f0 / η
        ∂n∂x, ∂n∂y, ns = -∂n∂x, -∂n∂y, -ns
    end
    ∂wo∂x = -ray.rx_direction - wo
    ∂wo∂y = -ray.ry_direction - wo
    ∂dn∂x = ∂wo∂x ⋅ ns + wo ⋅ ∂n∂x
    ∂dn∂y = ∂wo∂y ⋅ ns + wo ⋅ ∂n∂y
    μ = η * (wo ⋅ ns) - abs(wi ⋅ ns)
    ν = η - (η * η * (wo ⋅ ns)) / abs(wi ⋅ ns)
    ∂μ∂x = ν * ∂dn∂x
    ∂μ∂y = ν * ∂dn∂y
    rx_direction = wi - η * ∂wo∂x + μ * ∂n∂x + ∂μ∂x * ns
    ry_direction = wi - η * ∂wo∂y + μ * ∂n∂y + ∂μ∂y * ns
    return RayDifferentials(rd, rx_origin=rx_origin, ry_origin=ry_origin, rx_direction=rx_direction, ry_direction=ry_direction)
end


@inline function trace_pixel(camera, scene, pixel, sampler, max_depth)
    camera_sample = get_camera_sample(sampler, pixel)
    ray, ω = generate_ray_differential(camera, camera_sample)
    if ω > 0.0f0
        return li_iterative(sampler, max_depth, ray, scene)
    end
    return RGBSpectrum(0.0f0)
end

@noinline function sample_tile(sampler, camera, scene, film, film_tile, tile_bounds, max_depth)
    spp_sqr = 1.0f0 / √Float32(sampler.samples_per_pixel)
    for pixel in tile_bounds
        for _ in 1:sampler.samples_per_pixel
            camera_sample = get_camera_sample(sampler, pixel)
            ray, ω = generate_ray_differential(camera, camera_sample)
            ray = scale_differentials(ray, spp_sqr)
            l = RGBSpectrum(0.0f0)
            if ω > 0.0f0
                l = li_iterative(sampler, Int32(max_depth), ray, scene)
            end
            # TODO check l for invalid values
            l = ifelse(isnan(l), RGBSpectrum(0.0f0), l)
            add_sample!(film, film_tile, camera_sample.film, l, ω)
        end
    end
    merge_film_tile!(film, film_tile)
end

function sample_tiled(scene::Scene, film)
    sample_bounds = get_sample_bounds(film)
    sample_extent = diagonal(sample_bounds)
    tile_size = 16
    n_tiles = floor.(Int64, (sample_extent .+ tile_size) ./ tile_size)
    # TODO visualize tile bounds to see if they overlap
    width, height = n_tiles
    filter_radius = film.filter.radius
    filmtiles = similar(film.pixels, tile_size * tile_size, n_tiles)
    for tile_idx in CartesianIndices((width, height))
        tile_column, tile_row = Tuple(tile_idx)
        tile_bounds = Bounds2(tb_min, tb_max)
        film_tile = update_bounds!(film, film_tile, tile_bounds)
        sample_kernel(i, camera, scene, film, film_tile, tile_bounds)
    end
    return film
end

struct Whitten5{TMat<:AbstractMatrix{FilmTilePixel},PMat<:AbstractMatrix{Pixel}}
    tiles::TMat
    pixel::PMat
    sample_bounds::Bounds2
    crop_bounds::Bounds2
    resolution::Point2f
    ntiles::NTuple{2,Int32}
    tile_size::Int32
    fiter_table::Matrix{Float32}
    filter_radius::Point2f
    sampler::Trace.UniformSampler
    max_depth::Int32
end

function Whitten5(film; samples_per_pixel=8, tile_size=4, max_depth=5)
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
