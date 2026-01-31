using KernelAbstractions.Extras.LoopInfo: @unroll


abstract type SamplerIntegrator <: Integrator end

struct Whitted{S <: AbstractSampler} <: SamplerIntegrator
    sampler::S
    max_depth::Int64
end

function Whitted(; samples::Integer = 4, max_depth::Integer = 5)
    sampler = UniformSampler(Int64(samples))
    Whitted(sampler, Int64(max_depth))
end

@propagate_inbounds function sample_kernel_inner!(
        tiles, tile, tile_column::Int32, resolution::Point2f, max_depth::Int32,
        scene, sampler, camera, pixel, spp_sqr, filter_params::GPUFilterParams
    )
    # resolution is (height, width) from size(pixels) in Julia convention
    # pixel is (px, py) where px=x (column), py=y (row)
    # Camera expects raster coords with y increasing downward, so flip y
    # Use height (resolution[1]) for y-flip
    campix = Point2f(pixel[1], resolution[1] - pixel[2])

    # Use while loop to avoid iterate() protocol (causes PHI node errors in SPIR-V)
    sample_idx = Int32(1)
    while sample_idx <= sampler.samples_per_pixel
        # Get camera sample with filter importance sampling (pbrt-v4 compatible)
        camera_sample = get_camera_sample(sampler, campix, filter_params)
        ray, ω = generate_ray_differential(camera, camera_sample)
        ray = scale_differentials(ray, spp_sqr)
        l = RGBSpectrum(0.0f0)
        if ω > 0.0f0
            l = li_iterative(sampler, max_depth, ray, scene)
        end
        # TODO check l for invalid values
        if isnan(l)
            l = RGBSpectrum(0.0f0)
        end
        # Use camera_sample.film (the actual jittered position) and filter_weight
        add_sample!(
            tiles, tile, tile_column, camera_sample.film, l,
            camera_sample.filter_weight, ω,
        )
        sample_idx += Int32(1)
    end
end

@kernel inbounds=true function whitten_kernel!(pixels, crop_bounds, sample_bounds, tiles, tile_size, max_depth, scene, sampler, camera, filter_params)
    _tile_xy = @index(Global, Cartesian)
    linear_idx = @index(Global)

    # Explicit unpacking to avoid tuple iteration (causes PHI node errors in SPIR-V)
    _txy_tuple = Tuple(_tile_xy)
    tile_xy = (u_int32(_txy_tuple[1]), u_int32(_txy_tuple[2]))
    tile_column = u_int32(linear_idx)
    # Explicit unpacking instead of broadcasting to avoid iteration
    i = tile_xy[1] - Int32(1)
    j = tile_xy[2] - Int32(1)
    tile_start = Point2f(i, j)
    tb_min = (sample_bounds.p_min .+ tile_start .* tile_size) .+ Int32(1)
    tb_max = min.(tb_min .+ (tile_size .- Int32(1)), sample_bounds.p_max)
    tile_bounds = Bounds2(tb_min, tb_max)
    spp_sqr = 1.0f0 / √Float32(sampler.samples_per_pixel)

    # Explicit loop instead of iterating over Bounds2 to avoid GPU allocation issues
    # size(pixels) returns (rows, cols) = (height, width) in Julia convention
    resolution = Point2f(size(pixels)...)

    # Use while loops instead of for loops to avoid iterate() protocol
    # which returns Union{Nothing, Tuple{Int32, Int32}} causing PHI node errors in SPIR-V
    py = u_int32(tb_min[2])
    py_max = u_int32(tb_max[2])
    while py <= py_max
        px = u_int32(tb_min[1])
        px_max = u_int32(tb_max[1])
        while px <= px_max
            pixel = Point2f(px, py)
            sample_kernel_inner!(
                tiles, tile_bounds, tile_column, resolution,
                max_depth, scene, sampler, camera,
                pixel, spp_sqr, filter_params
            )
            px += Int32(1)
        end
        py += Int32(1)
    end
    merge_film_tile!(pixels, crop_bounds, tiles, tile_bounds, Int32(tile_column))
end

"""
Render scene using KernelAbstractions.
"""
function (i::SamplerIntegrator)(scene::AbstractScene, film, camera)
    sample_bounds = get_sample_bounds(film)
    tile_size = film.tile_size
    tiles = film.tiles
    sampler = i.sampler
    max_depth = Int32(i.max_depth)
    backend = KA.get_backend(film.tiles.contrib_sum)
    kernel! = whitten_kernel!(backend, (8, 8))
    kernel!(
        film.pixels, film.crop_bounds, sample_bounds,
        tiles, tile_size,
        max_depth, scene, sampler,
        camera, film.filter_params;
        ndrange=film.ntiles
    )
    KA.synchronize(backend)
    to_framebuffer!(film, 1f0)
end

# Get material from MaterialScene using triangle's metadata
function get_material(ms::MaterialScene, shape::Triangle)
    return get_material(ms.materials, shape.metadata)
end

# Non-allocating sum over lights for tuples (recursive for type stability)
@propagate_inbounds only_light(lights::Tuple{}, ray) = RGBSpectrum(0f0)
@propagate_inbounds function only_light(lights::Tuple, ray)
    return le(first(lights), ray) + only_light(Base.tail(lights), ray)
end

# Generated function for type-stable light iteration
# Unrolls only for the actual number of lights in the tuple
# Uses unique variable names per iteration to avoid union types
# NOTE: Must be @propagate_inbounds to avoid boxing arguments which causes allocations
@propagate_inbounds @generated function light_contribution(
    l::RGBSpectrum, lights::NTuple{N,Light}, wo, scene::S, bsdf, sampler, si
) where {N, S<:AbstractScene}
    # Generate unique symbols for each light iteration to avoid union types
    light_exprs = [quote
        let light =  lights[$i]
            sampled_li, wi, pdf, tester = sample_li(light, si.core, get_2d(sampler), scene)
            if !(is_black(sampled_li) || pdf ≈ 0.0f0)
                f = bsdf(wo, wi)
                if !is_black(f) && unoccluded(tester, scene)
                    l += f * sampled_li * abs(wi ⋅ si.shading.n) / pdf
                end
            end
        end
    end for i in 1:N]
    quote
        $(light_exprs...)
        return l
    end
end

# StaticMultiTypeSet versions using mapreduce for GPU compatibility
@propagate_inbounds function only_light(lights::Raycore.StaticMultiTypeSet, ray)
    return mapreduce(le, +, lights, ray; init=RGBSpectrum(0f0))
end

# Helper for light_contribution with StaticMultiTypeSet
@propagate_inbounds function _sample_light_contribution(light, wo, scene, bsdf, u, si_core, si_shading_n)
    sampled_li, wi, pdf, tester = sample_li(light, si_core, u, scene)
    if is_black(sampled_li) || pdf ≈ 0.0f0
        return RGBSpectrum(0f0)
    end
    f = bsdf(wo, wi)
    if is_black(f) || !unoccluded(tester, scene)
        return RGBSpectrum(0f0)
    end
    return f * sampled_li * abs(wi ⋅ si_shading_n) / pdf
end

@propagate_inbounds function light_contribution(
    l::RGBSpectrum, lights::Raycore.StaticMultiTypeSet, wo, scene::S, bsdf, sampler, si
) where {S<:AbstractScene}
    u = get_2d(sampler)
    contrib = mapreduce(_sample_light_contribution, +, lights, wo, scene, bsdf, u, si.core, si.shading.n; init=RGBSpectrum(0f0))
    return l + contrib
end

function li(
    sampler, max_depth, ray::RayDifferentials, scene::AbstractScene, depth::Int32,
)::RGBSpectrum

    l = RGBSpectrum(0.0f0)
    # Find closest ray intersection or return background radiance.
    hit, primitive, si = intersect!(scene, ray)
    lights = scene.lights
    if !hit
        return only_light(lights, ray)
    end
    # Compute emmited & reflected light at ray intersection point.
    # Initialize common variables for Whitted integrator.
    core = si.core
    wo = core.wo
    # Compute scattering functions for surface interaction.
    si = compute_differentials(si, ray)
    # Compute emitted light if ray hit an area light source.
    l += le(si, wo)
    # Use type-stable dispatch for material-dependent computation
    l += li_material(scene.materials, primitive.metadata,
                     sampler, max_depth, ray, si, scene, lights, wo, depth)
    l
end

# Helper for li_material - operates on concrete material type
@propagate_inbounds function _li_material_impl(
    mat, ctx, sampler, max_depth, ray::RayDifferentials, si::SurfaceInteraction,
    scene, lights, wo, depth::Int32
)
    bsdf = compute_bsdf(mat, ctx, si, false, Radiance)
    l = light_contribution(RGBSpectrum(0f0), lights, wo, scene, bsdf, sampler, si)
    if depth + Int32(1) ≤ max_depth
        l += specular_reflect(bsdf, sampler, max_depth, ray, si, scene, depth)
        l += specular_transmit(bsdf, sampler, max_depth, ray, si, scene, depth)
    end
    return l
end

# Type-stable material dispatch for li computation
# Uses with_index for type-stable dispatch over StaticMultiTypeSet
@propagate_inbounds function li_material(
    materials::StaticMultiTypeSet, idx::SetKey,
    sampler, max_depth, ray::RayDifferentials, si::SurfaceInteraction,
    scene::S, lights, wo, depth::Int32
) where {S<:AbstractScene}
    return with_index(_li_material_impl, materials, idx, materials, sampler, max_depth, ray, si, scene, lights, wo, depth)
end

@propagate_inbounds function specular_reflect(
        bsdf, sampler, max_depth, ray::RayDifferentials,
        si::SurfaceInteraction, scene::AbstractScene, depth::Int32,
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
    # Recursive call - works on CPU but can cause issues on GPU with complex scenes
    reflected_li = li(sampler, max_depth, rd, scene, depth + Int32(1))
    return f * reflected_li * abs(wi ⋅ ns) / pdf
end

"""
Glossy reflection for microfacet materials (metals, glossy plastics).
Uses sample_f for proper importance sampling of microfacet distribution.
The microfacet distribution already handles roughness via its α parameters,
so no additional perturbation is needed.
"""
@propagate_inbounds function glossy_reflect(
        bsdf, sampler, max_depth, ray::RayDifferentials,
        si::SurfaceInteraction, scene::AbstractScene, depth::Int32,
    )
    # Check if BSDF has glossy reflection component
    type = BSDF_REFLECTION | BSDF_GLOSSY
    num_components(bsdf, type) > 0 || return RGBSpectrum(0.0f0)

    wo = si.core.wo
    ns = si.shading.n

    # Use sample_f for proper microfacet importance sampling
    # The microfacet distribution handles roughness internally
    wi, f, pdf, sampled_type = sample_f(bsdf, wo, get_2d(sampler), type)

    if !(pdf > 0.0f0 && !is_black(f) && abs(wi ⋅ ns) != 0.0f0)
        return RGBSpectrum(0.0f0)
    end

    # Trace reflection ray
    rd = RayDifferentials(spawn_ray(si, wi))
    return f * li(sampler, max_depth, rd, scene, depth + Int32(1)) * abs(wi ⋅ ns) / pdf
end

@propagate_inbounds function specular_transmit(
    bsdf, sampler, max_depth, ray::RayDifferentials,
    surface_intersect::SurfaceInteraction, scene::AbstractScene, depth::Int32,
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
        # intersected. Compute the relative IOR by first assuming
        # that the ray is entering the object.
        η = 1.0f0 / bsdf.η
        # Check if ray is exiting the object (wo on opposite side of normal)
        if (wo ⋅ ns) < 0.0f0
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
    # Recursive call - works on CPU but can cause issues on GPU with complex scenes
    transmitted_li = li(sampler, max_depth, rd, scene, depth + Int32(1))
    f * transmitted_li * abs(wi ⋅ ns) / pdf
end




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

# Iterative path tracing for Whitted-style integrator
# Avoids recursion which is problematic on GPU
@propagate_inbounds function li_iterative(
        sampler, max_depth::Int32, initial_ray::RayDifferentials, scene::S
    )::RGBSpectrum where {S<:AbstractScene}

    L = RGBSpectrum(0.0f0)           # Accumulated radiance
    throughput = RGBSpectrum(1.0f0)  # Path throughput
    ray = initial_ray
    lights = scene.lights
    materials = scene.materials

    depth = Int32(0)
    while depth <= max_depth
        # Find intersection
        hit, primitive, si = intersect!(scene, ray)

        if !hit
            # Add background/environment light contribution
            L += throughput * only_light(lights, ray)
            break
        end

        # Compute surface interaction
        si = compute_differentials(si, ray)
        wo = si.core.wo

        # Add emitted light
        L += throughput * le(si, wo)

        # Get material and compute BSDF (or detect volume)
        idx = primitive.metadata
        valid, is_volume, bsdf = compute_bsdf_for_material(materials, idx, si)

        if !valid
            break
        end

        # Handle volume materials specially - they use shade() directly
        if is_volume
            # Volume materials handle their own ray marching and continuation rays
            vol_radiance = Raycore.with_index(shade, materials, idx, ray, si, scene, throughput, depth, max_depth)
            if !isnan(vol_radiance)
                L += vol_radiance
            end
            break  # Volume's shade() handles everything including continuation
        end

        # Add direct lighting contribution (surface materials only)
        direct = light_contribution(RGBSpectrum(0f0), lights, wo, scene, bsdf, sampler, si)
        # Check for NaN before adding
        if !isnan(direct)
            L += throughput * direct
        end

        # Check if we should continue with specular bounces
        if depth >= max_depth
            break
        end

        # Try specular reflection
        type = BSDF_REFLECTION | BSDF_SPECULAR
        wi, f, pdf, sampled_type = sample_f(bsdf, wo, get_2d(sampler), type)
        ns = si.shading.n

        if pdf > 0.0f0 && !is_black(f) && abs(wi ⋅ ns) != 0.0f0
            # Update throughput and continue with reflected ray
            new_throughput = f * abs(wi ⋅ ns) / pdf
            if !isnan(new_throughput)
                throughput *= new_throughput
                ray = RayDifferentials(spawn_ray(si, wi))
                depth += Int32(1)
                continue
            end
        end

        # Try specular transmission if reflection didn't work
        type = BSDF_TRANSMISSION | BSDF_SPECULAR
        wi, f, pdf, sampled_type = sample_f(bsdf, wo, get_2d(sampler), type)

        if pdf > 0.0f0 && !is_black(f) && abs(wi ⋅ ns) != 0.0f0
            # Update throughput and continue with transmitted ray
            new_throughput = f * abs(wi ⋅ ns) / pdf
            if !isnan(new_throughput)
                throughput *= new_throughput
                ray = RayDifferentials(spawn_ray(si, wi))
                depth += Int32(1)
                continue
            end
        end

        # No specular bounce possible, terminate
        break
    end

    # Final NaN check
    if isnan(L)
        return RGBSpectrum(0.0f0)
    end
    return L
end

# Helper to check if a material type is a volume (CloudVolume)
# Returns true for volume materials that need special handling via with_index(shade, ...)
@propagate_inbounds is_volume_material(::Material) = false
@propagate_inbounds is_volume_material(::CloudVolume) = true

# Helper for compute_bsdf_for_material - operates on concrete material type
@propagate_inbounds function _compute_bsdf_impl(mat, ctx, si::SurfaceInteraction)
    is_vol = is_volume_material(mat)
    bsdf = compute_bsdf(mat, ctx, si, false, Radiance)
    return (true, is_vol, bsdf)
end

# Helper to compute BSDF for a material index - returns (valid::Bool, is_volume::Bool, bsdf::BSDF)
# Returns a tuple to avoid Union{Nothing, BSDF} type instability
# For volume materials, is_volume=true indicates caller should use with_index(shade, ...) instead of BSDF
@propagate_inbounds function compute_bsdf_for_material(
    materials::StaticMultiTypeSet, idx::SetKey, si::SurfaceInteraction
)
    return with_index(_compute_bsdf_impl, materials, idx, materials, si)
end


@propagate_inbounds function trace_pixel(camera, scene, pixel, sampler, max_depth)
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
