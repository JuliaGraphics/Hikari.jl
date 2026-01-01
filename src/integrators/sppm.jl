struct VisiblePoint
    p::Point3f
    wo::Vec3f
    bsdf::BSDF{RGBSpectrum}
    β::RGBSpectrum

    function VisiblePoint(;
            p::Point3f = Point3f(0f0), wo::Vec3f = Vec3f(0f0),
            bsdf::BSDF=BSDF(), β::RGBSpectrum=RGBSpectrum(0.0f0),
        )
        new(p, wo, bsdf, β)
    end
end

struct SPPMPixel
    Ld::RGBSpectrum
    ϕ_x::Float32 # Atomic!!
    ϕ_y::Float32 # Atomic!!
    ϕ_z::Float32 # Atomic!!
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
    M::Int64 # Atomic!!
    """Total number of photons that contributed up to the ith iteration."""
    N::Float64

    function SPPMPixel(;
            Ld::RGBSpectrum = RGBSpectrum(0f0),
            ϕ::Vec3f = Vec3f(0f0),
            τ::RGBSpectrum = RGBSpectrum(0f0),
            radius::Float32 = 0f0, M::Int64 = 0, N::Int64 = 0,
        )
        new(Ld, ϕ..., τ, radius, M, N)
    end
end

struct SPPMIntegrator <: Integrator
    initial_search_radius::Float32
    max_depth::Int64
    n_iterations::Int64
    photons_per_iteration::Int64
    write_frequency::Int64

    function SPPMIntegrator(
        initial_search_radius::Float32, max_depth::Int64,
        n_iterations::Int64, film, photons_per_iteration::Int64=-1,
        write_frequency::Int64 = 1,
    )
        photons_per_iteration = (
            photons_per_iteration > 0
            ? photons_per_iteration : area(film.crop_bounds)
        )
        new(
            initial_search_radius, max_depth,
            n_iterations, photons_per_iteration, write_frequency,
        )
    end
end

function (i::SPPMIntegrator)(scene::Scene, film::Film, camera::Camera)

    pixel_bounds = film.crop_bounds

    b_sides = inclusive_sides(pixel_bounds)
    n_pixels = UInt64(b_sides[1] * b_sides[2])
    pixels = StructArray{SPPMPixel}(undef, Int(b_sides[2]), Int(b_sides[1]))
    pixels.radius .= i.initial_search_radius
    pixels.Ld .= (RGBSpectrum(0.0f0),)
    pixels.ϕ_x .= 0f0
    pixels.ϕ_y .= 0f0
    pixels.ϕ_z .= 0f0
    pixels.τ .= (RGBSpectrum(0.0f0),)
    pixels.M .= 0
    pixels.N .= 0
    visible_points = StructArray{VisiblePoint}(undef, Int(b_sides[2]), Int(b_sides[1]))
    visible_points.p .= (Point3f(0f0),)
    visible_points.wo .= (Vec3f(0f0),)
    visible_points.β .= (RGBSpectrum(0f0),)


    grid = [Int32[] for _ in 1:n_pixels]

    γ = 2f0 / 3f0
    inv_sqrt_spp = Float32(1f0 / sqrt(i.n_iterations))
    light_distribution = compute_light_power_distribution(scene)

    tile_size = 16
    pixel_extent = diagonal(pixel_bounds)
    n_tiles = Int64.(floor.((pixel_extent .+ tile_size) ./ tile_size))

    sampler = UniformSampler(1)
    for iteration in 1:i.n_iterations
        _generate_visible_sppm_points!(
            i, pixels, visible_points, scene, camera,
            n_tiles, tile_size, sampler,
            pixel_bounds, inv_sqrt_spp,
        )
        _clean_grid!(grid)
        grid_bounds, grid_resolution = _populate_grid!(grid, pixels, visible_points)
        _trace_photons!(
            i, pixels, visible_points, scene, camera, iteration, light_distribution,
            grid, grid_bounds, grid_resolution,
            n_pixels,
        )
        _update_pixels!(pixels, visible_points, γ)
    end
    # Convert SPPM pixels to image and write to framebuffer
    image = _sppm_to_image(i, pixels, i.n_iterations)
    set_image!(film, image)
    to_framebuffer!(film, 1f0)
end


@inline function inner_kernel(
        scene::SC, tile_sampler::S,
        inv_sqrt_spp::Float32,
        vps::AbstractMatrix{VisiblePoint},
        pixel_point::Point2f, camera::C, max_depth::Int,
        Ld::AbstractMatrix{RGBSpectrum},
        height::Float32,
    ) where {SC<:Scene, S<:AbstractSampler, C<:Camera}

    # Flip y-coordinate for camera sample to match Julia array convention
    # pixel_point is in (x, y) format where y=1 is top row in Julia's array convention
    # Camera expects y increasing downward (y=0 at top, y=height at bottom)
    # Map: Julia row 1 → camera y = height-1, Julia row height → camera y = 0
    camera_pixel = Point2f(pixel_point[1], height - pixel_point[2])
    camera_sample = get_camera_sample(tile_sampler, camera_pixel)
    rayd, _β = generate_ray_differential(camera, camera_sample)
    _β ≈ 0.0f0 && return

    β = RGBSpectrum(_β)
    @real_assert !isnan(β)
    rayd = scale_differentials(rayd, inv_sqrt_spp)
    # Follow camera ray path until a visible point is created.
    # Get SPPMPixel for current `pixel`.
    pixel_idx = reverse(Int64.(pixel_point))
    specular_bounce = false
    depth = 1
    while depth ≤ max_depth
        hit, primitive, si = intersect!(scene, rayd)
        if !hit # Accumulate light contributions to the background.
            for light in scene.lights
                light_emission = β * le(light, rayd)
                if !isnan(light_emission) && !isinf(light_emission)
                    Ld[pixel_idx...] += light_emission
                end
            end
            return
        end
        # Process SPPM camera ray intersection.
        # Compute BSDF at SPPM camera ray intersection.
        si = compute_differentials(si, rayd)
        material = get_material(scene.aggregate.materials, primitive.metadata)
        bsdf = compute_bsdf(material, si, true, Radiance)
        if bsdf.bxdfs.last == 0
            rayd = RayDifferentials(spawn_ray(si, rayd.d))
            continue
        end
        # Accumulate direct illumination at
        # SPPM camera-ray intersection.
        wo = -rayd.d
        if depth == 1 || specular_bounce
            emission = β * le(si, wo)
            if !isnan(emission) && !isinf(emission)
                Ld[pixel_idx...] += emission
            end
        end
        light_contrib = β * uniform_sample_one_light(
            bsdf, si, scene, tile_sampler,
        )
        if !isnan(light_contrib) && !isinf(light_contrib)
            Ld[pixel_idx...] += light_contrib
        end
        # Possibly create visible point and end camera path.
        is_diffuse = num_components(bsdf,
            BSDF_DIFFUSE | BSDF_REFLECTION | BSDF_TRANSMISSION,
        ) > 0
        is_glossy = num_components(bsdf,
            BSDF_GLOSSY | BSDF_REFLECTION | BSDF_TRANSMISSION,
        ) > 0
        if is_diffuse || (is_glossy && depth == max_depth)
            vps[pixel_idx...] = VisiblePoint(
                p = si.core.p, wo = wo,
                bsdf = bsdf, β = β,
            )
            return
        end
        # If at max depth, no need to spawn ray again.
        if depth == max_depth
            depth += 1
            continue
        end
        # Spawn ray from SPPM camera path vertex.
        # Try multiple samples if needed to handle failed microfacet samples
        wi, f, pdf, sampled_type = sample_f(
            bsdf, wo,
            get_2d(tile_sampler), BSDF_ALL,
        )
        (pdf ≈ 0f0 || is_black(f)) && return
        specular_bounce = (sampled_type & BSDF_SPECULAR) != 0
        β *= f * abs(wi ⋅ si.shading.n) / pdf
        @real_assert !isnan(β)
        # Guard against NaN/Inf in β
        (isnan(β) || isinf(β)) && return
        βy = to_Y(β)
        if βy < 0.25f0
            continue_probability = min(1f0, βy)
            get_1d(tile_sampler) > continue_probability && return
            β /= continue_probability
            @real_assert !isnan(β) && !isinf(β)
        end
        rayd = RayDifferentials(spawn_ray(si, wi))
        depth += 1
    end
end

function _generate_visible_sppm_points!(
        i::SPPMIntegrator,
        pixels::AbstractMatrix{SPPMPixel}, vps::AbstractMatrix{VisiblePoint},
        scene::Scene, camera::Camera,
        n_tiles::Point2, tile_size::Int64, sampler::S,
        pixel_bounds::Bounds2, inv_sqrt_spp::Float32,
    ) where S<:AbstractSampler

    width, height = n_tiles
    total_tiles = width * height - 1
    # Get image height for y-flip (pixel_bounds.p_max[2] is the max y coordinate)
    image_height = Float32(pixel_bounds.p_max[2])

    bar = get_progress_bar(total_tiles, "Camera pass: ")
    Ld = pixels.Ld
    max_depth = i.max_depth
    Threads.@threads for k in 0:total_tiles
        x, y = k % width, k ÷ width
        tile = Point2f(x, y)
        tile_sampler = deepcopy(sampler)

        tb_min = pixel_bounds.p_min .+ tile .* tile_size
        tb_max = min.(tb_min .+ (tile_size - 1), pixel_bounds.p_max)
        tile_bounds = Bounds2(tb_min, tb_max)
        for pixel_point in tile_bounds
            inner_kernel(
                scene, tile_sampler,
                inv_sqrt_spp,
                vps,
                pixel_point, camera, max_depth,
                Ld,
                image_height,
            )
        end
        next!(bar)
    end
end

@inline function _clean_grid!(grid)
    for i in 1:length(grid)
        empty!(grid[i])
    end
end

function _populate_grid!(
        grid::Vector{Vector{Int32}}, pixels::AbstractMatrix{SPPMPixel}, vps::AbstractMatrix{VisiblePoint}
    )

    n_pixels = UInt64(length(pixels))
    # Create grid of all SPPM visible points.
    grid_bounds = Bounds3()
    # Compute grid bounds for SPPM visible points.
    max_radius, min_radius = 0f0, Inf32
    β = vps.β
    p = vps.p
    pradius = pixels.radius
    for i in eachindex(pixels)
        is_black(β[i]) && continue
        radius = pradius[i]
        grid_bounds = grid_bounds ∪ expand(Bounds3(p[i]), radius)
        max_radius = max(max_radius, radius)
        min_radius = min(min_radius, radius)
    end
    # Compute resolution of SPPM grid in each dimension.
    diag = diagonal(grid_bounds)
    max_diag = maximum(diag)
    # TODO can be inf if no visible points
    @real_assert max_diag > 0
    @real_assert !isinf(max_radius)
    base_grid_resolution = floor(Int64, max_diag / max_radius)
    grid_resolution = max.(
        1, Int64.(floor.(base_grid_resolution .* diag ./ max_diag)),
    )
    # Add visible points to SPPM grid.
    @_inbounds for (i, pixel) in enumerate(pixels)
        is_black(β[i]) && continue
        # Add pixel's visible point to applicable grid cells.
        shift = pradius[i]
        vpp = p[i]
        _, p_min = to_grid(vpp .- shift, grid_bounds, grid_resolution)
        _, p_max = to_grid(vpp .+ shift, grid_bounds, grid_resolution)
        for z in p_min[3]:p_max[3], y in p_min[2]:p_max[2], x in p_min[1]:p_max[1]
            # Add visible point to grid cell (x, y, z).
            h = hash(x, y, z, n_pixels)
            # Add `node` to the start of `grid[h]` linked list.
            pushfirst!(grid[h], i)
        end
    end
    grid_bounds, grid_resolution
end


function _trace_photons!(
        i::SPPMIntegrator, pixels::AbstractMatrix{SPPMPixel}, vps::AbstractMatrix{VisiblePoint},
        scene::Scene, camera::Camera, iteration::Int64,
        light_distribution::Distribution1D,
        grid::Vector{Vector{Int32}},
        grid_bounds::Bounds3, grid_resolution::Point3,
        n_pixels::UInt64,
    )
    # Trace photons and accumulate contributions.
    halton_base = UInt64(iteration - 1) * UInt64(i.photons_per_iteration)
    bar = get_progress_bar(
        i.photons_per_iteration, "[$iteration] Photon pass: ",
    )
    shutter_open = camera.core.core.shutter_open
    shutter_close = camera.core.core.shutter_close
    pϕ_x = pixels.ϕ_x
    pϕ_y = pixels.ϕ_y
    pϕ_z = pixels.ϕ_z
    pM = pixels.M
    radius = pixels.radius
    p = vps.p
    wo = vps.wo
    bsdfs = vps.bsdf
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
            # put it back after free
            hit, primitive, si = intersect!(scene, photon_ray)
            !hit && break
            if depth > 1
                # Add photon contribution to nearby visible points.
                in_bounds, gi = to_grid(
                    si.core.p, grid_bounds, grid_resolution,
                )
                if in_bounds
                    h = hash(gi[1], gi[2], gi[3], n_pixels)
                    # Add photon contribution to visible points in `grid[h]`.
                    pixel_list::Vector{Int} = grid[h]
                    for pixel_idx in pixel_list
                        if distance_squared(
                            p[pixel_idx], si.core.p,
                        ) > (radius[pixel_idx]^2)
                            continue
                        end
                        # Update `pixel`'s ϕ & M for nearby photon.
                        _ϕ = β * bsdfs[pixel_idx](
                            wo[pixel_idx], -photon_ray.d,
                        )
                        @real_assert !isnan(_ϕ)
                        Atomix.@atomic pϕ_x[pixel_idx] += _ϕ[1]
                        Atomix.@atomic pϕ_y[pixel_idx] += _ϕ[2]
                        Atomix.@atomic pϕ_z[pixel_idx] += _ϕ[3]
                        Atomix.@atomic pM[pixel_idx] += 1
                    end
                end
            end
            # Sample new photon direction.
            # Compute BSDF at photon intersection point.
            si = compute_differentials(si, photon_ray)
            material = get_material(scene.aggregate.materials, primitive.metadata)
            bsdf = compute_bsdf(material, si, true, Importance)
            if bsdf.bxdfs.last == 0
                photon_ray = RayDifferentials(spawn_ray(si, photon_ray.d))
                continue
            end
            # Sample BSDF spectrum and direction `wi` for reflected photon.
            bsdf_sample = Point2f(
                radical_inverse(halton_dim, halton_index),
                radical_inverse(halton_dim + 1, halton_index),
            )
            halton_dim += 2
            wi, fr, pdf, sampled_type = sample_f(
                bsdf, -photon_ray.d, bsdf_sample, BSDF_ALL,
            )
            (is_black(fr) || pdf ≈ 0f0) && break

            # Possibly terminate photon path with Russian roulette.
            β_new = β * fr * abs(wi ⋅ si.shading.n) / pdf
            q = max(0f0, 1f0 - to_Y(β_new) / βy)
            if radical_inverse(halton_dim, halton_index) < q
                halton_dim += 1
                break
            end
            halton_dim += 1
            # Update β for next bounce, accounting for Russian roulette survival
            β = β_new / (1f0 - q)
            βy = to_Y(β)
            photon_ray = RayDifferentials(spawn_ray(si, wi))
            depth += 1
        end
        next!(bar)
    end
end

function _update_pixels!(pixels::AbstractMatrix{SPPMPixel}, vps::AbstractMatrix{VisiblePoint}, γ::Float32)
    # Update pixel values from this pass's photons.

    pM = pixels.M
    pϕ_x = pixels.ϕ_x
    pϕ_y = pixels.ϕ_y
    pϕ_z = pixels.ϕ_z
    pN = pixels.N
    pradius = pixels.radius
    pτ = pixels.τ
    β = vps.β
    for pixel_idx in eachindex(pixels)
        M = pM[pixel_idx]
        if M > 0
            ϕ = RGBSpectrum(pϕ_x[pixel_idx], pϕ_y[pixel_idx], pϕ_z[pixel_idx])
            # Update pixel photon count, search radius and τ from photons.
            n = pN[pixel_idx]
            N_new = n + γ * M
            radius = pradius[pixel_idx]
            # Guard against division by zero
            denom = n + M
            if denom > 0f0 && radius > 0f0
                radius_new = radius * √(N_new / denom)
                pτ[pixel_idx] = (pτ[pixel_idx] + ϕ) * (radius_new / radius)^2
                # TODO do not multiply by beta?
                # (pixel.τ + pixel.vp.β * pixel.ϕ)

                pradius[pixel_idx] = radius_new
                pN[pixel_idx] = N_new
            end
            pϕ_x[pixel_idx] = 0.0f0
            pϕ_y[pixel_idx] = 0.0f0
            pϕ_z[pixel_idx] = 0.0f0
            pM[pixel_idx] = 0
        end
        β[pixel_idx] = RGBSpectrum(0f0)
    end
end

function _sppm_to_image(
        i::SPPMIntegrator, pixels::AbstractMatrix{SPPMPixel}, iteration::Int64,
    )

    @real_assert iteration > 0
    Np = iteration * i.photons_per_iteration * π
    image = fill(RGBSpectrum(0f0), size(pixels))
    @_inbounds for (i, p) in enumerate(pixels)
        # Combine direct and indirect radiance estimates.
        # Guard against division by zero when radius is very small
        indirect = if p.radius > 1f-6
            p.τ / (Np * (p.radius^2))
        else
            RGBSpectrum(0f0)
        end
        image[i] = (p.Ld / iteration) + indirect
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
    grid_point = u_int32.(floor.(grid_resolution .* p_offset))
    in_bounds = all(0 .≤ grid_point .< grid_resolution)
    grid_point = clamp.(grid_point, Int32(0), grid_resolution .- Int32(1))
    return in_bounds, grid_point
end

@inline function hash(
    p1::UInt64, p2::UInt64, p3::UInt64, hash_size::UInt64,
)::UInt64
    (((p1 * 73856093) ⊻ (p2 * 19349663) ⊻ (p3 * 83492791)) % hash_size) + 1
end

function uniform_sample_one_light(
        bsdf, i::SurfaceInteraction, scene::Scene, sampler::S,
    )::RGBSpectrum where S<:AbstractSampler


    n_lights = length(scene.lights)
    n_lights == 0 && return RGBSpectrum(0f0)

    light_num = max(1, min(Int32(ceil(get_1d(sampler) * n_lights)), n_lights))
    light_pdf = 1f0 / n_lights

    light = scene.lights[light_num]
    u_light = get_2d(sampler)
    u_scatter = get_2d(sampler)

    estimate_direct(bsdf, i, u_scatter, light, u_light, scene, sampler) / light_pdf
end

function estimate_direct(
        bsdf, interaction::SurfaceInteraction, u_scatter::Point2f, light::L,
        u_light::Point2f, scene::Scene, sampler::S, specular::Bool = false,
    )::RGBSpectrum where {L<:Light,S<:AbstractSampler}

    bsdf_flags = specular ? BSDF_ALL : (BSDF_ALL & ~BSDF_SPECULAR)
    Ld = RGBSpectrum(0f0)
    # Sample light source with multiple importance sampling.
    Li, wi, light_pdf, visibility = sample_li(light, interaction.core, u_light, scene)
    if light_pdf > 0 && !is_black(Li)
        # Evaluate BSDF for light sampling strategy.
        f = (
            bsdf(interaction.core.wo, wi, bsdf_flags)
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
                        bsdf, interaction.core.wo, wi, bsdf_flags,
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
