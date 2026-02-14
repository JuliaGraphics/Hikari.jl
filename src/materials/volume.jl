#=
Volumetric Materials for Hikari

Based on Monte Carlo radiative transfer methods for cloudy atmospheres.
References:
- Burchart et al. (2024) "A Stereo Camera Simulator for Large-Eddy Simulations
  of Continental Shallow Cumulus Clouds Based on Three-Dimensional Path-Tracing"
  DOI: 10.1029/2023MS003797
- Villefranque et al. (2019) "A Path-Tracing Monte Carlo Library for 3-D
  Radiative Transfer in Highly Resolved Cloudy Atmospheres"

Key physics:
- Beer-Lambert law for transmittance
- Henyey-Greenstein phase function for Mie scattering
- Ray marching with single scattering approximation
=#

# ============================================================================
# Cloud Volume Material
# ============================================================================

"""
    CloudVolume

A 3D grid representing cloud density (e.g., liquid water content in g/m³).
The volume is axis-aligned and defined by its bounding box.

This is both a data structure and a material type - when a ray hits geometry
with a CloudVolume material, volumetric rendering is performed.
"""
struct CloudVolume{T<:AbstractArray{Float32,3}} <: Material
    # 3D density field (e.g., LWC in g/m³)
    density::T
    # Bounding box (world space)
    origin::Point3f      # Lower corner
    extent::Vec3f        # Size in each dimension
    # Physical parameters
    extinction_scale::Float32  # Converts density to extinction coefficient
    # Scattering parameters
    asymmetry_g::Float32       # HG asymmetry parameter (0.85 for clouds)
    single_scatter_albedo::Float32  # ~1.0 for clouds (almost pure scattering)
end

function CloudVolume(density::AbstractArray{Float32,3};
                     origin=Point3f(-1, -1, -1),
                     extent=Vec3f(2, 2, 2),
                     extinction_scale=100.0f0,
                     asymmetry_g=0.85f0,
                     single_scatter_albedo=0.999f0)
    CloudVolume(density, origin, extent, extinction_scale, asymmetry_g, single_scatter_albedo)
end

"""Get the maximum extinction coefficient in the volume"""
@propagate_inbounds function max_extinction(cloud::CloudVolume)
    maximum(cloud.density) * cloud.extinction_scale
end

"""Sample density at a world position using trilinear interpolation"""
@propagate_inbounds function sample_density(cloud::CloudVolume, pos::Point3f)
    ox, oy, oz = cloud.origin[1], cloud.origin[2], cloud.origin[3]
    ex, ey, ez = cloud.extent[1], cloud.extent[2], cloud.extent[3]

    # Normalized coordinates (0-1)
    ux = (pos[1] - ox) / ex
    uy = (pos[2] - oy) / ey
    uz = (pos[3] - oz) / ez

    # Check bounds
    if ux < 0 || ux > 1 || uy < 0 || uy > 1 || uz < 0 || uz > 1
        return 0.0f0
    end

    # Grid dimensions
    nx, ny, nz = size(cloud.density)

    # Convert to grid coordinates
    gx = ux * (nx - 1) + 1
    gy = uy * (ny - 1) + 1
    gz = uz * (nz - 1) + 1

    # Integer indices
    ix = clamp(floor_int(gx), 1, nx - 1)
    iy = clamp(floor_int(gy), 1, ny - 1)
    iz = clamp(floor_int(gz), 1, nz - 1)

    # Fractional parts
    fx = Float32(gx - ix)
    fy = Float32(gy - iy)
    fz = Float32(gz - iz)

    # Trilinear interpolation
     begin
        d000 = cloud.density[ix, iy, iz]
        d100 = cloud.density[ix+1, iy, iz]
        d010 = cloud.density[ix, iy+1, iz]
        d110 = cloud.density[ix+1, iy+1, iz]
        d001 = cloud.density[ix, iy, iz+1]
        d101 = cloud.density[ix+1, iy, iz+1]
        d011 = cloud.density[ix, iy+1, iz+1]
        d111 = cloud.density[ix+1, iy+1, iz+1]
    end

    # Interpolate along x
    fx1 = 1.0f0 - fx
    d00 = d000 * fx1 + d100 * fx
    d10 = d010 * fx1 + d110 * fx
    d01 = d001 * fx1 + d101 * fx
    d11 = d011 * fx1 + d111 * fx

    # Interpolate along y
    fy1 = 1.0f0 - fy
    d0 = d00 * fy1 + d10 * fy
    d1 = d01 * fy1 + d11 * fy

    # Interpolate along z
    return d0 * (1.0f0 - fz) + d1 * fz
end

"""Get extinction coefficient at a world position"""
@propagate_inbounds function extinction_at(cloud::CloudVolume, pos::Point3f)
    return sample_density(cloud, pos) * cloud.extinction_scale
end

# ============================================================================
# Ray-Box Intersection
# ============================================================================

"""
    intersect_box(ray_o, ray_d, box_min, box_max) -> (hit, t_near, t_far)

Compute ray intersection with an axis-aligned bounding box.
"""
@propagate_inbounds function intersect_box(ray_o::Point3f, ray_d::Vec3f, t_max::Float32,
                               box_min::Point3f, box_max::Point3f)
    inv_dx = 1.0f0 / ray_d[1]
    inv_dy = 1.0f0 / ray_d[2]
    inv_dz = 1.0f0 / ray_d[3]

    t1x = (box_min[1] - ray_o[1]) * inv_dx
    t2x = (box_max[1] - ray_o[1]) * inv_dx
    t1y = (box_min[2] - ray_o[2]) * inv_dy
    t2y = (box_max[2] - ray_o[2]) * inv_dy
    t1z = (box_min[3] - ray_o[3]) * inv_dz
    t2z = (box_max[3] - ray_o[3]) * inv_dz

    tmin_x = min(t1x, t2x)
    tmax_x = max(t1x, t2x)
    tmin_y = min(t1y, t2y)
    tmax_y = max(t1y, t2y)
    tmin_z = min(t1z, t2z)
    tmax_z = max(t1z, t2z)

    t_near = max(tmin_x, max(tmin_y, tmin_z))
    t_far = min(tmax_x, min(tmax_y, tmax_z))

    # Clamp to valid range
    t_near = max(t_near, 0.0f0)
    t_far = min(t_far, t_max)

    return t_far > t_near, t_near, t_far
end

# ============================================================================
# Phase Functions
# ============================================================================

"""
    henyey_greenstein(cos_theta, g) -> Float32

Henyey-Greenstein phase function.
- g > 0: Forward scattering (clouds typically g ≈ 0.85)
- g = 0: Isotropic scattering
- g < 0: Backward scattering
"""
@propagate_inbounds function henyey_greenstein(cos_theta::Float32, g::Float32)
    g2 = g * g
    denom = 1.0f0 + g2 - 2.0f0 * g * cos_theta
    return (1.0f0 - g2) / (4.0f0 * Float32(π) * denom * sqrt(denom))
end

# ============================================================================
# Transmittance Computation
# ============================================================================

"""Compute transmittance from a point towards a direction through the volume"""
@propagate_inbounds function compute_transmittance(cloud::CloudVolume, pos::Point3f, dir::Vec3f,
                                       steps::Int=16)
    box_min = cloud.origin
    box_max = cloud.origin + cloud.extent

    hit, t_near, t_far = intersect_box(pos, dir, Inf32, box_min, box_max)

    if !hit || t_far <= 0
        return 1.0f0
    end

    t_near = max(t_near, 0.001f0)

    # March and accumulate optical depth
    optical_depth = 0.0f0
    dt = (t_far - t_near) / Float32(steps)

    t = t_near + 0.5f0 * dt
     for _ in 1:steps
        sample_pos = Point3f(
            pos[1] + dir[1] * t,
            pos[2] + dir[2] * t,
            pos[3] + dir[3] * t
        )
        σ_t = extinction_at(cloud, sample_pos)
        optical_depth += σ_t * dt
        t += dt
    end

    return exp(-optical_depth)
end

# ============================================================================
# Helper functions for GPU-safe light iteration
# ============================================================================

"""
    _accumulate_volume_scatter(acc, light, cloud, pos, ray_d, time, shadow_steps, scene)

Accumulator function for reduce_unrolled over lights in volume scattering.
Computes single light contribution and adds to accumulator.
"""
@propagate_inbounds function _accumulate_volume_scatter(
    acc::RGBSpectrum, light, cloud::CloudVolume, pos, ray_d, time, shadow_steps, scene
)
    # Create interaction at current position for light sampling
    interaction = Interaction(Point3f(pos), time, -ray_d, Normal3f(0f0, 0f0, 1f0))

    # Sample the light
    u_light = rand(Point2f)
    Li, wi, pdf, visibility = sample_li(light, interaction, u_light, scene)

    # Skip if no contribution
    (is_black(Li) || pdf ≈ 0f0) && return acc

    # Compute transmittance from sample point toward light
    light_transmittance = compute_transmittance(cloud, pos, wi, shadow_steps)

    # Phase function for this light direction
    cos_theta = dot(-ray_d, wi)
    phase = henyey_greenstein(cos_theta, cloud.asymmetry_g)

    # Add light contribution
    return acc + Li * light_transmittance * phase / pdf
end

"""
    _sum_light_le(light, ray)

Helper for sum_unrolled to accumulate le() contributions from lights.
"""
@propagate_inbounds _sum_light_le(light, ray) = le(light, ray)

# ============================================================================
# Material Interface: shade() and sample_bounce()
# ============================================================================

"""
    shade(cloud::CloudVolume, ray, si, scene, beta) -> RGBSpectrum

Shade a ray that hit a volume material using ray marching with single scattering.
The SurfaceInteraction gives us the entry point into the volume.

This version checks for scene intersections DURING ray marching, so objects
inside or in front of the volume are properly rendered with correct transmittance.
"""
@noinline function shade(cloud::CloudVolume, ray::RayDifferentials, si::SurfaceInteraction,
               scene::S, beta::RGBSpectrum, depth::Int32, max_depth::Int32;
               step_size::Float32=0.02f0, shadow_steps::Int=16) where {S<:AbstractScene}

    # Volume bounds
    box_min = cloud.origin
    box_max = cloud.origin + cloud.extent

    # Get ray parameters
    ray_o = Point3f(ray.o)
    ray_d = Vec3f(ray.d)

    # Find volume intersection
    hit, t_near, t_far = intersect_box(ray_o, ray_d, ray.t_max, box_min, box_max)

    if !hit
        # Shouldn't happen if we hit the volume mesh, but handle gracefully
        return RGBSpectrum(0f0)
    end

    # Check for scene intersections WITHIN the volume bounds
    # Start ray just past the volume entry to avoid hitting the cloud box front face
    entry_offset = 0.001f0
    entry_point = Point3f(
        ray_o[1] + ray_d[1] * (t_near + entry_offset),
        ray_o[2] + ray_d[2] * (t_near + entry_offset),
        ray_o[3] + ray_d[3] * (t_near + entry_offset)
    )
    # Ray should only travel up to the volume exit (minus a small offset to avoid back face)
    max_dist = (t_far - t_near) - 2*entry_offset
    volume_ray = RayDifferentials(Ray(entry_point, ray_d, max_dist, ray.time))
    scene_hit, scene_primitive, scene_si = intersect!(scene, volume_ray)

    # Determine effective march distance - stop at scene intersection if closer
    t_stop = t_far
    has_scene_hit_in_volume = false
    if scene_hit
        # Skip hits on CloudVolume materials (self-intersection with volume box)
        hit_mat = get_material(scene.materials, scene_primitive.metadata[1])
        if !(hit_mat isa CloudVolume)
            # Compute hit distance from the hit point (intersect! doesn't update t_max)
            hit_p = scene_si.core.p
            hit_dist_from_entry = norm(Vec3f(hit_p) - Vec3f(entry_point))
            # Check if hit is inside the volume (not at the back face)
            if hit_dist_from_entry < max_dist - 0.01f0
                # Convert to original ray space
                hit_t = hit_dist_from_entry + t_near + entry_offset
                t_stop = hit_t
                has_scene_hit_in_volume = true
            end
        end
    end

    # Ray march through the volume, computing both in-scatter and transmittance
    transmittance = 1.0f0
    in_scatter = RGBSpectrum(0f0)

    t = t_near
    while t < t_stop && transmittance > 0.001f0
        pos = ray_o .+ ray_d * t
        σ_t = extinction_at(cloud, pos)

        if σ_t > 0.001f0
            dt = min(step_size, t_stop - t)
            step_transmittance = exp(-σ_t * dt)

            # Accumulate scattering from ALL scene lights (using reduce_unrolled for GPU compatibility)
            scatter_light = reduce_unrolled(
                _accumulate_volume_scatter, scene.lights, RGBSpectrum(0f0),
                cloud, pos, ray_d, ray.time, shadow_steps, scene
            )

            # Accumulate in-scattering
            scatter_amount = transmittance * (1.0f0 - step_transmittance) * cloud.single_scatter_albedo
            in_scatter += scatter_amount * scatter_light

            transmittance *= step_transmittance
        end

        t += step_size
    end

    # Determine background color
    background = RGBSpectrum(0f0)
    if transmittance > 0.001f0 && depth < max_depth
        if has_scene_hit_in_volume
            # Hit an object inside/before the volume exit - shade it
            background = Raycore.with_index(
                shade, scene.materials, scene_primitive.metadata[1],
                volume_ray, scene_si, scene, RGBSpectrum(1f0), depth + Int32(1), max_depth
            )
        else
            # No hit inside volume - trace continuation ray past volume exit
            exit_point = Point3f(
                ray_o[1] + ray_d[1] * (t_far + 0.001f0),
                ray_o[2] + ray_d[2] * (t_far + 0.001f0),
                ray_o[3] + ray_d[3] * (t_far + 0.001f0)
            )
            continuation_ray = RayDifferentials(Ray(exit_point, ray_d, ray.t_max - t_far, ray.time))

            # Trace the continuation ray through the scene
            hit_found, primitive, cont_si = intersect!(scene, continuation_ray)

            if hit_found
                # Hit something behind the volume - shade it
                background = Raycore.with_index(
                    shade, scene.materials, primitive.metadata[1],
                    continuation_ray, cont_si, scene, RGBSpectrum(1f0), depth + Int32(1), max_depth
                )
            else
                # Miss - get background from scene lights via le() (e.g. EnvironmentLight)
                exit_ray = RayDifferentials(Ray(ray_o .+ ray_d * t_far, ray_d, Inf32, ray.time))
                # Use sum_unrolled for GPU-safe iteration over lights
                bg_light = sum_unrolled(_sum_light_le, scene.lights, exit_ray)
                if bg_light !== nothing
                    background = bg_light
                end
                # Fallback to default sky if no environment light
                if is_black(background)
                    sky_up = max(0f0, ray_d[3])
                    background = RGBSpectrum(0.5f0, 0.7f0, 1.0f0) * (0.5f0 + 0.5f0 * sky_up)
                end
            end
        end
    end

    # Final color: in-scattered light + attenuated background
    return beta * (in_scatter + transmittance * background)
end

"""
    sample_bounce(cloud::CloudVolume, ray, si, scene, beta, depth) -> (should_bounce, new_ray, new_beta, new_depth)

For volumes, we handle everything in shade() including the continuation ray,
so no bounce is needed here.
"""
function sample_bounce(cloud::CloudVolume, ray::RayDifferentials, si::SurfaceInteraction,
                       scene::AbstractScene, beta::RGBSpectrum, depth::Int32)
    # Everything is handled in shade() to avoid double ray-marching
    return (false, ray, RGBSpectrum(0f0), Int32(0))
end

"""
    compute_bsdf(cloud::CloudVolume, si, allow_multiple_lobes, transport) -> BSDF

Returns a dummy BSDF for volume materials. Volumes don't use BSDF-based shading;
they use ray marching in shade() instead. This method exists so that
compute_bsdf_for_material can handle CloudVolume without special-casing.
"""
@propagate_inbounds function compute_bsdf(::CloudVolume, textures, si::SurfaceInteraction, ::Bool, transport)
    return BSDF(si)
end

# ============================================================================
# Oceananigans/LES Integration
# ============================================================================

# Only define this method if Oceananigans is loaded
# This uses a conditional definition pattern

"""
    CloudVolume(qˡ_data::AbstractArray{<:Real,3}, grid_extent::Tuple; ...)

Create a CloudVolume from raw LWC data with physical grid info.

# Arguments
- `qˡ_data`: 3D array of liquid water specific humidity (kg/kg)
- `grid_extent`: Tuple (Lx, Ly, Lz) of domain size in meters
- `r_eff`: Effective droplet radius in meters (default: 10μm)
- `ρ_air`: Air density in kg/m³ (default: 1.0)
- `scale`: Additional scaling factor for visualization (default: 1.0)
"""
function CloudVolume(qˡ_data::AbstractArray{<:Real,3}, grid_extent::Tuple{<:Real,<:Real,<:Real};
                     r_eff=10e-6,
                     ρ_air=1.0,
                     ρ_water=1000.0,
                     scale=1.0,
                     asymmetry_g=0.85f0,
                     single_scatter_albedo=0.999f0)
    Lx, Ly, Lz = grid_extent

    # Convert qˡ (kg/kg) to extinction coefficient (1/m)
    # β_ext = (3/2) * ρ_air * qˡ / (ρ_water * r_eff)
    extinction_factor = Float32((3/2) * ρ_air / (ρ_water * r_eff) * scale)

    density = Float32.(Array(qˡ_data))

    # Normalize to a unit-ish box centered at origin for easier camera setup
    max_extent = max(Lx, Ly, Lz)
    origin = Point3f(
        -Lx / max_extent,
        -Ly / max_extent,
        0  # Keep z starting at 0 (ground level)
    )
    extent = Vec3f(
        2 * Lx / max_extent,
        2 * Ly / max_extent,
        2 * Lz / max_extent
    )

    CloudVolume(density, origin, extent, extinction_factor, asymmetry_g, single_scatter_albedo)
end
