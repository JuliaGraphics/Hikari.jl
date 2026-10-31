mutable struct AtomicVec3f
    x::Threads.Atomic{Float32}
    y::Threads.Atomic{Float32}
    z::Threads.Atomic{Float32}
end

function AtomicVec3f(x::Float32 = 0f0)
    AtomicVec3f(
        Threads.Atomic{Float32}(x),
        Threads.Atomic{Float32}(x),
        Threads.Atomic{Float32}(x),
    )
end

function AtomicVec3f(p::Point3f)
    AtomicVec3f(
        Threads.Atomic{Float32}(p[1]),
        Threads.Atomic{Float32}(p[2]),
        Threads.Atomic{Float32}(p[3]),
    )
end

function set!(a::AtomicVec3f, v::Float32)
    a.x[] = v
    a.y[] = v
    a.z[] = v
end

function set!(a::AtomicVec3f, p::Point3f)
    a.x[] = p[1]
    a.y[] = p[2]
    a.z[] = p[3]
end

function Threads.atomic_add!(a::AtomicVec3f, p::Point3f)
    Threads.atomic_add!(a.x, p[1])
    Threads.atomic_add!(a.y, p[2])
    Threads.atomic_add!(a.z, p[3])
end

function Threads.atomic_add!(a::AtomicVec3f, s::RGBSpectrum)
    Threads.atomic_add!(a.x, s.c[1])
    Threads.atomic_add!(a.y, s.c[2])
    Threads.atomic_add!(a.z, s.c[3])
end

function Base.convert(::Type{Point3f}, a::AtomicVec3f)
    Point3f(a.x[], a.y[], a.z[])
end

mutable struct VisiblePoint
    p::Point3f
    wo::Vec3f
    bsdf::Maybe{BSDF}
    β::RGBSpectrum

    function VisiblePoint(;
        p::Point3f = Point3f(0f0), wo::Vec3f = Vec3f(0f0),
        bsdf::Maybe{BSDF} = nothing, β::RGBSpectrum = RGBSpectrum(0f0),
    )
        new(p, wo, bsdf, β)
    end
end

mutable struct SPPMPixel
    Ld::RGBSpectrum
    ϕ::AtomicVec3f
    """
    Maintains the sum of products of photons with BSDF values.
    Aka. sum of ϕ from all of the iterations, weighted by radius ratio.
    """
    τ::RGBSpectrum
    """
    Photon search radius.
    By reducing the radius, we ensure that future photons that are used
    will be closer to the point and thus contribute to a more accurate
    estimate of the incident radiance distribution.
    """
    radius::Float32
    """Number of photons that contributed during the ith iteration."""
    M::Threads.Atomic{Int64}
    """Total number of photons that contributed up to the ith iteration."""
    N::Float64
    vp::VisiblePoint

    function SPPMPixel(;
        Ld::RGBSpectrum = RGBSpectrum(0f0),
        ϕ::AtomicVec3f = AtomicVec3f(0f0),
        τ::RGBSpectrum = RGBSpectrum(0f0),
        radius::Float32 = 0f0, M::Int64 = 0, N::Int64 = 0,
        vp::VisiblePoint = VisiblePoint(),
    )
        new(Ld, ϕ, τ, radius, Threads.Atomic{Int64}(M), N, vp)
    end
end

mutable struct SPPMPixelListNode
    pixel::SPPMPixel
    next::Maybe{SPPMPixelListNode}

    function SPPMPixelListNode(
        pixel::SPPMPixel, next::Maybe{SPPMPixelListNode} = nothing,
    )
        new(pixel, next)
    end
end

struct SPPMIntegrator <: Integrator
    camera::C where C<:Camera
    initial_search_radius::Float32
    max_depth::Int64
    n_iterations::Int64
    photons_per_iteration::Int64
    write_frequency::Int64

    function SPPMIntegrator(
        camera::C, initial_search_radius::Float32, max_depth::Int64,
        n_iterations::Int64, photons_per_iteration::Int64 = -1,
        write_frequency::Int64 = 1,
    ) where C<:Camera
        photons_per_iteration = (
            photons_per_iteration > 0
            ? photons_per_iteration : area(get_film(camera).crop_bounds)
        )
        new(
            camera, initial_search_radius, max_depth,
            n_iterations, photons_per_iteration, write_frequency,
        )
    end
end

function (i::SPPMIntegrator)(scene::Scene)
    pixel_bounds = get_film(i.camera).crop_bounds

    b_sides = inclusive_sides(pixel_bounds)
    n_pixels = UInt64(b_sides[1] * b_sides[2])
    pixels = [
        SPPMPixel(radius = i.initial_search_radius)
        for y in 1:b_sides[2], x in 1:b_sides[1]
    ]
    grid = Maybe{SPPMPixelListNode}[nothing for _ in 1:n_pixels]

    γ = 2f0 / 3f0
    inv_sqrt_spp = Float32(1f0 / sqrt(i.n_iterations))
    light_distribution = compute_light_power_distribution(scene)

    tile_size = 16
    pixel_extent = diagonal(pixel_bounds)
    n_tiles::Point2 = Int64.(floor.((pixel_extent .+ tile_size) ./ tile_size))

    sampler = UniformSampler(1)
    for iteration in 1:i.n_iterations
        _generate_visible_sppm_points!(
            i, pixels, scene,
            n_tiles, tile_size, sampler,
            pixel_bounds, inv_sqrt_spp,
        )
        _clean_grid!(grid)
        grid_bounds, grid_resolution = _populate_grid!(grid, pixels)
        _trace_photons!(
            i, scene, iteration, light_distribution,
            grid, grid_bounds, grid_resolution,
            n_pixels,
        )
        _update_pixels!(pixels, γ)
        # Periodically store SPPM image in film and save it.
        if iteration % i.write_frequency == 0 || iteration == i.n_iterations
            image = _sppm_to_image(i, pixels, iteration)
            set_image!(get_film(i.camera), image)
            save(get_film(i.camera))
        end
    end
end

function _generate_visible_sppm_points!(
    i::SPPMIntegrator, pixels::Matrix{SPPMPixel}, scene::Scene,
    n_tiles::Point2, tile_size::Int64, sampler::S,
    pixel_bounds::Bounds2, inv_sqrt_spp::Float32,
) where S<:AbstractSampler
    width, height = n_tiles
    total_tiles = width * height - 1

    bar = get_progress_bar(total_tiles, "Camera pass: ")
    Threads.@threads for k in 0:total_tiles
        x, y = k % width, k ÷ width
        tile = Point2f(x, y)
        tile_sampler = deepcopy(sampler)

        tb_min = pixel_bounds.p_min .+ tile .* tile_size
        tb_max = min.(tb_min .+ (tile_size - 1), pixel_bounds.p_max)
        tile_bounds = Bounds2(tb_min, tb_max)
        for pixel_point in tile_bounds
            start_pixel!(tile_sampler, pixel_point)
            # set_sample_number!(tile_sampler, iteration)

            camera_sample = get_camera_sample(tile_sampler, pixel_point)
            ray, β = generate_ray_differential(i.camera, camera_sample)
            β ≈ 0f0 && continue
            β = RGBSpectrum(β)
            @real_assert !isnan(β)
            scale_differentials!(ray, inv_sqrt_spp)
            # Follow camera ray path until a visible point is created.
            # Get SPPMPixel for current `pixel`.
            pixel_point = Int64.(pixel_point)
            pixel = pixels[pixel_point[2], pixel_point[1]]
            specular_bounce = false
            depth = 1
            while depth ≤ i.max_depth
                hit, surface_interaction = intersect!(scene, ray)
                if !hit # Accumulate light contributions to the background.
                    for light in scene.lights
                        pixel.Ld += β * le(light, ray)
                    end
                    break
                end
                # Process SPPM camera ray intersection.
                # Compute BSDF at SPPM camera ray intersection.
                compute_scattering!(surface_interaction, ray, true)
                if surface_interaction.bsdf ≡ nothing
                    ray = spawn_ray(surface_interaction, ray.d)
                    continue
                end
                # Accumulate direct illumination at
                # SPPM camera-ray intersection.
                wo = -ray.d
                if depth == 1 || specular_bounce
                    pixel.Ld += β * le(surface_interaction, wo)
                end
                pixel.Ld += uniform_sample_one_light(
                    surface_interaction, scene, tile_sampler,
                )
                # Possibly create visible point and end camera path.
                is_diffuse = num_components(surface_interaction.bsdf,
                    BSDF_DIFFUSE | BSDF_REFLECTION | BSDF_TRANSMISSION,
                ) > 0
                is_glossy = num_components(surface_interaction.bsdf,
                    BSDF_GLOSSY | BSDF_REFLECTION | BSDF_TRANSMISSION,
                ) > 0
                if is_diffuse || (is_glossy && depth == i.max_depth)
                    pixel.vp = VisiblePoint(
                        p = surface_interaction.core.p, wo = wo,
                        bsdf = surface_interaction.bsdf, β = β,
                    )
                    break
                end
                # If at max depth, no need to spawn ray again.
                depth == i.max_depth && (depth += 1; continue)
                # Spawn ray from SPPM camera path vertex.
                wi, f, pdf, sampled_type = sample_f(
                    surface_interaction.bsdf, wo,
                    get_2d(tile_sampler), BSDF_ALL,
                )
                (pdf ≈ 0f0 || is_black(f)) && break
                specular_bounce = (sampled_type & BSDF_SPECULAR) != 0
                β *= f * abs(wi ⋅ surface_interaction.shading.n) / pdf
                @real_assert !isnan(β)
                βy = to_Y(β)
                if βy < 0.25f0
                    continue_probability = min(1f0, βy)
                    get_1d(tile_sampler) > continue_probability && break
                    β /= continue_probability
                    @real_assert !isnan(β) && !isinf(β)
                end
                ray = RayDifferentials(spawn_ray(surface_interaction, wi))
                depth += 1
            end
        end
        next!(bar)
    end
end

@inline function _clean_grid!(grid)
    for i in 1:length(grid)
        grid[i] = nothing
    end
end

function _populate_grid!(
    grid::Vector{Maybe{SPPMPixelListNode}}, pixels::Matrix{SPPMPixel},
)
    n_pixels = UInt64(length(pixels))
    # Create grid of all SPPM visible points.
    grid_bounds = Bounds3()
    # Compute grid bounds for SPPM visible points.
    max_radius, min_radius = 0f0, Inf32
    for pixel in pixels
        is_black(pixel.vp.β) && continue
        grid_bounds = grid_bounds ∪ expand(Bounds3(pixel.vp.p), pixel.radius)
        max_radius = max(max_radius, pixel.radius)
        min_radius = min(min_radius, pixel.radius)
    end
    # Compute resolution of SPPM grid in each dimension.
    diag = diagonal(grid_bounds)
    max_diag = maximum(diag)
    # TODO can be inf if no visible points
    @real_assert max_diag > 0
    @real_assert !isinf(max_radius)
    base_grid_resolution = Int64(floor(max_diag / max_radius))
    grid_resolution = max.(
        1, Int64.(floor.(base_grid_resolution .* diag ./ max_diag)),
    )
    # Add visible points to SPPM grid.
    @inbounds for pixel in pixels
        is_black(pixel.vp.β) && continue
        # Add pixel's visible point to applicable grid cells.
        shift = pixel.radius
        _, p_min = to_grid(pixel.vp.p .- shift, grid_bounds, grid_resolution)
        _, p_max = to_grid(pixel.vp.p .+ shift, grid_bounds, grid_resolution)
        for z in p_min[3]:p_max[3], y in p_min[2]:p_max[2], x in p_min[1]:p_max[1]
            # Add visible point to grid cell (x, y, z).
            h = hash(x, y, z, n_pixels)
            # Add `node` to the start of `grid[h]` linked list.
            node = SPPMPixelListNode(pixel, grid[h])
            grid[h] = node
        end
    end
    grid_bounds, grid_resolution
end

function _trace_photons!(
    i::SPPMIntegrator, scene::Scene, iteration::Int64,
    light_distribution::Distribution1D,
    grid::Vector{Maybe{SPPMPixelListNode}},
    grid_bounds::Bounds3, grid_resolution::Point3,
    n_pixels::UInt64,
)
    # Trace photons and accumulate contributions.
    halton_base = UInt64(iteration - 1) * UInt64(i.photons_per_iteration)
    bar = get_progress_bar(
        i.photons_per_iteration, "[$iteration] Photon pass: ",
    )
    shutter_open = i.camera.core.core.shutter_open
    shutter_close = i.camera.core.core.shutter_close
    Threads.@threads for photon_index in 0:i.photons_per_iteration-1
        # Follow photon path for `photon_index`.
        halton_index = halton_base + photon_index
        halton_dim = 0
        # Choose light to shoot photon from.
        light_sample = radical_inverse(halton_dim, halton_index)
        halton_dim += 1
        light_num, light_pdf, _ = sample_discrete(
            light_distribution, light_sample,
        )
        light = scene.lights[light_num]
        # Compute sample values for photon ray leaving light source.
        u_light_0 = Point2f(
            radical_inverse(halton_dim, halton_index),
            radical_inverse(halton_dim + 1, halton_index),
        )
        u_light_1 = Point2f(
            radical_inverse(halton_dim + 2, halton_index),
            radical_inverse(halton_dim + 3, halton_index),
        )
        u_light_time = lerp(
            shutter_open,
            shutter_close,
            radical_inverse(halton_dim + 4, halton_index),
        )
        halton_dim += 5
        # Generate `photon_ray` from light source and initialize β.
        le, ray, light_normal, pdf_pos, pdf_dir = sample_le(
            light, u_light_0, u_light_1, u_light_time,
        )
        (pdf_pos ≈ 0f0 || pdf_dir ≈ 0f0 || is_black(le)) && continue
        photon_ray = RayDifferentials(ray)
        β = abs(light_normal ⋅ photon_ray.d) * le / (
            light_pdf * pdf_pos * pdf_dir
        )
        is_black(β) && continue
        βy = to_Y(β)

        # Follow photon path through scene and record intersections.
        depth = 1
        while depth ≤ i.max_depth
            hit, interaction = intersect!(scene, photon_ray)
            !hit && break
            if depth > 1
                # Add photon contribution to nearby visible points.
                in_bounds, gi = to_grid(
                    interaction.core.p, grid_bounds, grid_resolution,
                )
                if in_bounds
                    h = hash(gi[1], gi[2], gi[3], n_pixels)
                    # Add photon contribution to visible points in `grid[h]`.
                    node::Maybe{SPPMPixelListNode} = grid[h]
                    while node ≢ nothing
                        if distance_squared(
                            node.pixel.vp.p, interaction.core.p,
                        ) > (node.pixel.radius^2)
                            node = node.next
                            continue
                        end
                        # Update `pixel`'s ϕ & M for nearby photon.
                        ϕ = β * node.pixel.vp.bsdf(
                            node.pixel.vp.wo, -photon_ray.d,
                        )
                        @real_assert !isnan(ϕ)
                        Threads.atomic_add!(node.pixel.ϕ, ϕ)
                        Threads.atomic_add!(node.pixel.M, 1)
                        node = node.next
                    end
                end
            end
            # Sample new photon direction.
            # Compute BSDF at photon intersection point.
            compute_scattering!(interaction, photon_ray, true, Importance)
            if interaction.bsdf ≡ nothing
                photon_ray = spawn_ray(interaction, photon_ray.d)
                continue
            end
            # Sample BSDF spectrum and direction `wi` for reflected photon.
            bsdf_sample = Point2f(
                radical_inverse(halton_dim, halton_index),
                radical_inverse(halton_dim + 1, halton_index),
            )
            halton_dim += 2
            wi, fr, pdf, sampled_type = sample_f(
                interaction.bsdf, -photon_ray.d, bsdf_sample, BSDF_ALL,
            )
            (is_black(fr) || pdf ≈ 0f0) && break

            # Possibly terminate photon path with Russian roulette.
            β_new = β * fr * abs(wi ⋅ interaction.shading.n) / pdf
            q = max(0f0, 1f0 - to_Y(β_new) / βy)
            if radical_inverse(halton_dim, halton_index) < q
                halton_dim += 1
                break
            end
            halton_dim += 1
            # β = β_new / (1f0 - q)
            photon_ray = RayDifferentials(spawn_ray(interaction, wi))
            depth += 1
        end
        next!(bar)
    end
end

function _update_pixels!(pixels::Matrix{SPPMPixel}, γ::Float32)
    # Update pixel values from this pass's photons.
    for pixel in pixels
        M = pixel.M[]
        if M > 0
            ϕ = convert(Point3f, pixel.ϕ)
            # Update pixel photon count, search radius and τ from photons.
            N_new = pixel.N + γ * M
            radius_new = pixel.radius * √(N_new / (pixel.N + M))
            pixel.τ = (pixel.τ + ϕ) * (radius_new / pixel.radius)^2
            # TODO do not multiply by beta?
            # (pixel.τ + pixel.vp.β * pixel.ϕ)

            pixel.radius = radius_new
            pixel.N = N_new
            set!(pixel.ϕ, 0f0)
            pixel.M[] = 0
        end
        pixel.vp.β = RGBSpectrum(0f0)
        pixel.vp.bsdf = nothing
    end
end

function _sppm_to_image(
    i::SPPMIntegrator, pixels::Matrix{SPPMPixel}, iteration::Int64,
)
    @real_assert iteration > 0
    Np = iteration * i.photons_per_iteration * π
    image = fill(RGBSpectrum(0f0), size(pixels))
    @inbounds for (i, p) in enumerate(pixels)
        # Combine direct and indirect radiance estimates.
        image[i] = (p.Ld / iteration) + (p.τ / (Np * (p.radius^2)))
    end
    image
end

"""
Calculate indices of a point `p` in grid constrained by `bounds`.

Computed indices are in [0, resolution), which is the correct input for `hash`.
"""
@inline function to_grid(
    p::Point3f, bounds::Bounds3, grid_resolution::Point3,
)::Tuple{Bool,Point3{UInt64}}
    p_offset = offset(bounds, p)
    grid_point = Point3{Int64}(
        floor(grid_resolution[1] * p_offset[1]),
        floor(grid_resolution[2] * p_offset[2]),
        floor(grid_resolution[3] * p_offset[3]),
    )
    in_bounds = all(0 .≤ grid_point .< grid_resolution)
    grid_point = Point3{UInt64}(
        clamp(grid_point[1], 0, grid_resolution[1] - 1),
        clamp(grid_point[2], 0, grid_resolution[2] - 1),
        clamp(grid_point[3], 0, grid_resolution[3] - 1),
    )
    in_bounds, grid_point
end

@inline function hash(
    p1::UInt64, p2::UInt64, p3::UInt64, hash_size::UInt64,
)::UInt64
    (((p1 * 73856093) ⊻ (p2 * 19349663) ⊻ (p3 * 83492791)) % hash_size) + 1
end

function uniform_sample_one_light(
    i::SurfaceInteraction, scene::Scene, sampler::S,
)::RGBSpectrum where S<:AbstractSampler
    n_lights = length(scene.lights)
    n_lights == 0 && return RGBSpectrum(0f0)

    light_num = max(1, min(Int32(ceil(get_1d(sampler) * n_lights)), n_lights))
    light_pdf = 1f0 / n_lights

    light = scene.lights[light_num]
    u_light = get_2d(sampler)
    u_scatter = get_2d(sampler)

    estimate_direct(i, u_scatter, light, u_light, scene, sampler) / light_pdf
end

function estimate_direct(
    interaction::SurfaceInteraction, u_scatter::Point2f, light::L,
    u_light::Point2f, scene::Scene, sampler::S, specular::Bool = false,
)::RGBSpectrum where {L<:Light,S<:AbstractSampler}
    bsdf_flags = specular ? BSDF_ALL : (BSDF_ALL & ~BSDF_SPECULAR)
    Ld = RGBSpectrum(0f0)
    # Sample light source with multiple importance sampling.
    Li, wi, light_pdf, visibility = sample_li(light, interaction.core, u_light)
    if light_pdf > 0 && !is_black(Li)
        # Evaluate BSDF for light sampling strategy.
        f = (
            interaction.bsdf(interaction.core.wo, wi, bsdf_flags)
            *
            abs(wi ⋅ interaction.shading.n)
        )
        if !is_black(f)
            # Compute effect of visibility for light source sample.
            !unoccluded(visibility, scene) && (Li = RGBSpectrum(0f0))
            if !is_black(Li)
                if is_δ_light(light.flags)
                    Ld += f * Li / light_pdf
                else
                    @real_assert false # TODO no non delta lights right now
                    scattering_pdf = compute_pdf(
                        interaction.bsdf, interaction.core.wo, wi, bsdf_flags,
                    )
                    weight = power_heuristic(1, light_pdf, 1, scattering_pdf)
                    Ld += f * Li * weight / light_pdf
                end
            end
        end
    end
    # TODO Sample BSDF with multiple importance sampling.
    # This requires non-δ light sources.
    Ld
end

@inline function power_heuristic(
    nf::Int64, f_pdf::Float32, ng::Int64, g_pdf::Float32,
)
    f = (nf * f_pdf)^2
    g = (ng * g_pdf)^2
    f / (f + g)
end

@inline function compute_light_power_distribution(
    scene::Scene,
)::Maybe{Distribution1D}
    length(scene.lights) == 0 && return nothing
    Distribution1D([(to_Y(power(l))) for l in scene.lights])
end
