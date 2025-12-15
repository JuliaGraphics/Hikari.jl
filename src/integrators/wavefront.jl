# ============================================================================
# Wavefront Path Tracer Integrator for Hikari
# ============================================================================
# A wavefront-style renderer that processes rays in coherent batches for
# better GPU utilization. Supports all Hikari materials and lights.

using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
import KernelAbstractions as KA

# ============================================================================
# SoA Access Macros
# ============================================================================

"""
    @get field1, field2, ... = soa[idx]

Macro to extract multiple fields from a Structure of Arrays (SoA) at index `idx`.
"""
macro get(expr)
    if expr.head != :(=)
        error("@get expects assignment syntax: @get field1, field2 = soa[idx]")
    end

    lhs = expr.args[1]
    rhs = expr.args[2]

    # Parse left side (field names)
    if lhs isa Symbol
        fields = [lhs]
    elseif lhs.head == :tuple
        fields = lhs.args
    else
        error("@get left side must be field names or tuple of field names")
    end

    # Parse right side (soa[idx])
    if rhs.head != :ref
        error("@get right side must be array indexing: soa[idx]")
    end
    soa = rhs.args[1]
    idx = rhs.args[2]

    # Generate field extraction code
    assignments = [:($(esc(field)) = $(esc(soa)).$(field)[$(esc(idx))]) for field in fields]

    return Expr(:block, assignments...)
end

"""
    @set soa[idx] = (field1=val1, field2=val2, ...)

Macro to set multiple fields in a Structure of Arrays (SoA) at index `idx`.
Expects named tuple syntax on the right side.
"""
macro set(expr)
    if expr.head != :(=)
        error("@set expects assignment syntax: @set soa[idx] = (field1=val1, ...)")
    end

    lhs = expr.args[1]
    rhs = expr.args[2]

    # Parse left side (soa[idx])
    if lhs.head != :ref
        error("@set left side must be array indexing: soa[idx]")
    end
    soa = lhs.args[1]
    idx = lhs.args[2]

    # Parse right side (named tuple or parameters)
    assignments = []
    if rhs.head == :tuple || rhs.head == :parameters
        for arg in rhs.args
            if arg isa Expr && arg.head == :(=)
                field = arg.args[1]
                val = arg.args[2]
                push!(assignments, :($(esc(soa)).$(field)[$(esc(idx))] = $(esc(val))))
            else
                error("@set expects named parameters: @set soa[idx] = (field=value, ...)")
            end
        end
    else
        error("@set expects a tuple with named fields: @set soa[idx] = (field=value, ...)")
    end

    return Expr(:block, assignments...)
end

"""
Create SoA storage for a given struct type.
"""
function similar_soa(template, ::Type{T}, num_elements) where T
    fields = [f => similar(template, fieldtype(T, f), num_elements) for f in fieldnames(T)]
    return (; fields...)
end

# ============================================================================
# Work Queue Structures
# ============================================================================

struct WavefrontRayWork
    ray::RayDifferentials
    pixel::Point2f
    beta::RGBSpectrum
    sample_idx::Int32
    depth::Int32
end

struct WavefrontHitWork
    hit_found::Bool
    material_idx::MaterialIndex  # Material lookup info
    # SurfaceInteraction fields stored flat for SoA compatibility
    hit_p::Point3f
    hit_n::Normal3f
    hit_wo::Vec3f
    hit_uv::Point2f
    shading_n::Normal3f
    shading_dpdu::Vec3f
    shading_dpdv::Vec3f
    # Original ray and pixel info
    ray::RayDifferentials
    pixel::Point2f
    beta::RGBSpectrum
    sample_idx::Int32
    depth::Int32
end

struct WavefrontShadowWork
    # Shadow ray info
    shadow_origin::Point3f
    shadow_dir::Vec3f
    shadow_t_max::Float32
    # Reference back
    hit_idx::Int32
    light_idx::Int32
    # Pre-computed contribution (will be added if unoccluded)
    contribution::RGBSpectrum
end

struct WavefrontShadowResult
    visible::Bool
    hit_idx::Int32
    contribution::RGBSpectrum
end

struct WavefrontShadedResult
    color::RGBSpectrum
    pixel::Point2f
    sample_idx::Int32
end

# ============================================================================
# WavefrontIntegrator Struct
# ============================================================================

"""
    WavefrontIntegrator

A wavefront-style path tracer that processes rays in coherent batches.
Supports all Hikari materials (Matte, Mirror, Glass, Plastic) and lights
(Point, Spot, Directional, Ambient, Environment).

# Fields
- `camera`: The camera to use for rendering
- `max_depth`: Maximum path depth (bounces)
- `samples_per_pixel`: Number of samples per pixel

# Example
```julia
integrator = WavefrontIntegrator(camera, max_depth=5, samples_per_pixel=4)
integrator(scene, film)
```
"""
struct WavefrontIntegrator{C<:Camera} <: Integrator
    camera::C
    max_depth::Int32
    samples_per_pixel::Int32

    function WavefrontIntegrator(
        camera::C;
        max_depth::Integer=5,
        samples_per_pixel::Integer=4
    ) where {C<:Camera}
        new{C}(camera, Int32(max_depth), Int32(samples_per_pixel))
    end
end

# ============================================================================
# Kernel: Generate Primary Camera Rays
# ============================================================================

# Inner function for generating a single camera ray sample
# Can be tested with @code_warntype
@inline function generate_primary_ray_inner(
    x::Int32, y::Int32, height::Int32, camera
)::Tuple{RayDifferentials, RGBSpectrum, Point2f}
    # Jittered sample position
    jitter = rand(Point2f)
    film_x = Float32(x) - 0.5f0 + jitter[1]
    film_y = Float32(height - y) + 0.5f0 + jitter[2]  # Flip y for camera convention
    film_point = Point2f(film_x, film_y)

    # Create camera sample
    camera_sample = CameraSample(film_point, Point2f(0.5f0), rand(Float32))

    # Generate ray
    ray, weight = generate_ray_differential(camera, camera_sample)

    # Initial throughput
    beta = RGBSpectrum(weight)

    # Pixel in raster coordinates for accumulation
    pixel = Point2f(x, y)

    return (ray, beta, pixel)
end

# Inner function for generating all samples for a pixel - uses generated function for unrolling
@generated function generate_pixel_samples_inner!(
    ray_queue, pixel_idx::Int32, x::Int32, y::Int32, height::Int32, camera,
    ::Val{NSamples}
) where {NSamples}
    quote
        Base.Cartesian.@nexprs $NSamples s -> begin
            s_idx = Int32(s)
            ray_idx = (pixel_idx - Int32(1)) * Int32($NSamples) + s_idx

            ray, beta, pixel = generate_primary_ray_inner(x, y, height, camera)

            @set ray_queue[ray_idx] = (
                ray=ray, pixel=pixel, beta=beta,
                sample_idx=s_idx, depth=Int32(0)
            )
        end
        return nothing
    end
end

@kernel function generate_primary_rays_kernel!(
    @Const(width), @Const(height),
    @Const(camera),
    @Const(sampler),
    ray_queue,
    samples::Val{NSamples}
) where {NSamples}
    i = @index(Global, Cartesian)
    y = u_int32(i[1])
    x = u_int32(i[2])

    if y <= height && x <= width
        pixel_idx = (y - Int32(1)) * width + x
        generate_pixel_samples_inner!(ray_queue, pixel_idx, x, y, height, camera, samples)
    end
end

# ============================================================================
# Kernel: Intersect Primary Rays
# ============================================================================

# Inner function for intersection - returns hit data tuple
# Can be tested with @code_warntype
@inline function intersect_ray_inner(
    bvh, ray::RayDifferentials
)::Tuple{Bool, MaterialIndex, Point3f, Normal3f, Vec3f, Point2f, Normal3f, Vec3f, Vec3f}
    hit_found, triangle, distance, bary_coords = closest_hit(bvh, ray)

    if hit_found
        # Convert to SurfaceInteraction
        si = triangle_to_surface_interaction(triangle, ray, bary_coords)
        mat_idx = triangle.metadata::MaterialIndex

        return (
            true, mat_idx,
            si.core.p, si.core.n, si.core.wo, si.uv,
            si.shading.n, si.shading.∂p∂u, si.shading.∂p∂v
        )
    else
        return (
            false, MaterialIndex(UInt8(0), UInt32(0)),
            Point3f(0), Normal3f(0), Vec3f(0), Point2f(0),
            Normal3f(0), Vec3f(0), Vec3f(0)
        )
    end
end

# Process a single ray for intersection - returns full hit queue entry
@inline function process_intersect_ray_inner(
    ray::RayDifferentials, pixel::Point2f, beta::RGBSpectrum,
    sample_idx::Int32, depth::Int32, bvh
)::Tuple{Bool, MaterialIndex, Point3f, Normal3f, Vec3f, Point2f, Normal3f, Vec3f, Vec3f,
         RayDifferentials, Point2f, RGBSpectrum, Int32, Int32}
    hit_found, mat_idx, hit_p, hit_n, hit_wo, hit_uv, shading_n, shading_dpdu, shading_dpdv = intersect_ray_inner(bvh, ray)

    return (hit_found, mat_idx, hit_p, hit_n, hit_wo, hit_uv, shading_n, shading_dpdu, shading_dpdv,
            ray, pixel, beta, sample_idx, depth)
end

# Full kernel body for intersection
@inline function intersect_rays_kernel_body!(
    scene_aggregate, ray_queue, hit_queue, idx::Int32
)
    @get ray, pixel, beta, sample_idx, depth = ray_queue[idx]

    bvh = scene_aggregate.bvh
    hit_found, mat_idx, hit_p, hit_n, hit_wo, hit_uv, shading_n, shading_dpdu, shading_dpdv,
        out_ray, out_pixel, out_beta, out_sample_idx, out_depth = process_intersect_ray_inner(
            ray, pixel, beta, sample_idx, depth, bvh
        )

    @set hit_queue[idx] = (
        hit_found=hit_found,
        material_idx=mat_idx,
        hit_p=hit_p,
        hit_n=hit_n,
        hit_wo=hit_wo,
        hit_uv=hit_uv,
        shading_n=shading_n,
        shading_dpdu=shading_dpdu,
        shading_dpdv=shading_dpdv,
        ray=out_ray,
        pixel=out_pixel,
        beta=out_beta,
        sample_idx=out_sample_idx,
        depth=out_depth
    )
    return nothing
end

@kernel function intersect_rays_kernel!(
    @Const(scene_aggregate),
    @Const(ray_queue),
    hit_queue
)
    i = @index(Global, Linear)
    idx = u_int32(i)

    if idx <= length(ray_queue.ray)
        intersect_rays_kernel_body!(scene_aggregate, ray_queue, hit_queue, idx)
    end
end

# ============================================================================
# Kernel: Shade Hits (Direct Lighting from All Lights)
# ============================================================================

# Inner function for reconstructing SurfaceInteraction from hit queue data
# Can be tested with @code_warntype
@inline function reconstruct_surface_interaction(
    hit_p::Point3f, hit_n::Normal3f, hit_wo::Vec3f, hit_uv::Point2f,
    shading_n::Normal3f, shading_dpdu::Vec3f, shading_dpdv::Vec3f,
    ray_time::Float32
)::SurfaceInteraction
    core = Interaction(hit_p, ray_time, hit_wo, hit_n)
    shading = ShadingInteraction(shading_n, shading_dpdu, shading_dpdv, Normal3f(0), Normal3f(0))
    return SurfaceInteraction(
        core, shading, hit_uv,
        shading_dpdu, shading_dpdv, Normal3f(0), Normal3f(0),
        0f0, 0f0, 0f0, 0f0, Vec3f(0), Vec3f(0)
    )
end

# Inner function for shade_primary_hits - handles a single hit
# Can be tested with @code_warntype
@inline function shade_hit_inner(
    hit_p::Point3f, hit_n::Normal3f, hit_wo::Vec3f, hit_uv::Point2f,
    shading_n::Normal3f, shading_dpdu::Vec3f, shading_dpdv::Vec3f,
    ray_time::Float32, material, scene, beta::RGBSpectrum
)
    # Reconstruct SurfaceInteraction
    si = reconstruct_surface_interaction(
        hit_p, hit_n, hit_wo, hit_uv,
        shading_n, shading_dpdu, shading_dpdv, ray_time
    )
    # Compute BSDF
    bsdf = @inline material(si, true, Radiance)
    # Compute direct lighting from all lights
    @inbounds return @inline shade_with_lights(
        scene.lights, hit_p, hit_wo, hit_n, shading_n, bsdf, scene, beta
    )
end

# Inner function for miss shading - samples environment light
# Can be tested with @code_warntype
@inline function shade_miss_inner(
    ray::RayDifferentials, scene, beta::RGBSpectrum, sky_color::RGBSpectrum
)
    miss_color = RGBSpectrum(0f0)
    @inbounds for light in scene.lights
        miss_color += le(light, ray)
    end
    if is_black(miss_color)
        return beta * sky_color
    else
        return beta * miss_color
    end
end

# Helper to sample a single light - handles all light types
@inline function sample_light_contribution(
    light::Light, hit_p::Point3f, hit_wo::Vec3f, hit_n::Normal3f,
    shading_n::Normal3f, bsdf::BSDF, scene, beta::RGBSpectrum
)
    # Create interaction for light sampling
    interaction = Interaction(hit_p, 0f0, hit_wo, hit_n)

    # Sample the light
    u_light = rand(Point2f)
    Li, wi, pdf, visibility = @inline sample_li(light, interaction, u_light, scene)

    # Skip if no contribution
    (is_black(Li) || pdf ≈ 0f0) && return RGBSpectrum(0f0), false

    # Evaluate BSDF
    f = bsdf(hit_wo, wi)
    is_black(f) && return RGBSpectrum(0f0), false

    # Check visibility
    !unoccluded(visibility, scene) && return RGBSpectrum(0f0), false

    # Compute contribution
    cos_theta = abs(wi ⋅ shading_n)
    contribution = beta * f * Li * cos_theta / pdf

    return contribution, true
end

# Simple light iteration - avoid @nexprs to prevent allocations
@inline function shade_with_lights(
    lights, hit_p, hit_wo, hit_n, shading_n, bsdf, scene, beta
)
    total = RGBSpectrum(0f0)
    # Manually unroll for single light (most common case)
    @inbounds if length(lights) >= 1
        light = lights[1]
        contrib, valid = sample_light_contribution(
            light, hit_p, hit_wo, hit_n, shading_n, bsdf, scene, beta
        )
        if valid
            total += contrib
        end
    end
    # Fall back to loop for additional lights
    @inbounds for i in 2:length(lights)
        light = lights[i]
        contrib, valid = sample_light_contribution(
            light, hit_p, hit_wo, hit_n, shading_n, bsdf, scene, beta
        )
        if valid
            total += contrib
        end
    end
    return total
end

# Fallback for variable light count (runtime loop)
function shade_with_lights_dynamic(
    lights, hit_p, hit_wo, hit_n, shading_n, bsdf, scene, beta
)
    total = RGBSpectrum(0f0)
    for light in lights
        contrib, valid = sample_light_contribution(
            light, hit_p, hit_wo, hit_n, shading_n, bsdf, scene, beta
        )
        if valid
            total += contrib
        end
    end
    return total
end

# Process a single hit for shading - returns (color, pixel, sample_idx)
# Uses generated dispatch to handle multiple material types
@inline function process_shade_hit_inner(
    hit_found::Bool, mat_idx::MaterialIndex,
    hit_p::Point3f, hit_n::Normal3f, hit_wo::Vec3f, hit_uv::Point2f,
    shading_n::Normal3f, shading_dpdu::Vec3f, shading_dpdv::Vec3f,
    ray::RayDifferentials, pixel::Point2f, beta::RGBSpectrum,
    sample_idx::Int32, depth::Int32, max_depth::Int32,
    materials::Tuple, scene, sky_color::RGBSpectrum
)::Tuple{RGBSpectrum, Point2f, Int32}
    if hit_found
        # Reconstruct SurfaceInteraction for material dispatch
        si = reconstruct_surface_interaction(
            hit_p, hit_n, hit_wo, hit_uv,
            shading_n, shading_dpdu, shading_dpdv, ray.time
        )
        # Use generated dispatch based on material index
        color = @inline shade_material(materials, mat_idx, ray, si, scene, beta, depth, max_depth)
        return (color, pixel, sample_idx)
    else
        miss_color = shade_miss_inner(ray, scene, beta, sky_color)
        return (miss_color, pixel, sample_idx)
    end
end

# Full kernel body for shading
@inline function shade_primary_hits_kernel_body!(
    hit_queue, scene, materials, sky_color::RGBSpectrum, max_depth::Int32,
    shading_queue, idx::Int32
)
    @get hit_found, material_idx, hit_p, hit_n, hit_wo, hit_uv, shading_n, shading_dpdu, shading_dpdv,
         ray, pixel, beta, sample_idx, depth = hit_queue[idx]

    color, out_pixel, out_sample_idx = process_shade_hit_inner(
        hit_found, material_idx,
        hit_p, hit_n, hit_wo, hit_uv,
        shading_n, shading_dpdu, shading_dpdv,
        ray, pixel, beta, sample_idx, depth, max_depth,
        materials, scene, sky_color
    )

    @set shading_queue[idx] = (color=color, pixel=out_pixel, sample_idx=out_sample_idx)
    return nothing
end

@kernel function shade_primary_hits_kernel!(
    @Const(hit_queue),
    @Const(scene),
    @Const(materials),
    @Const(sky_color),
    @Const(max_depth),
    shading_queue
)
    i = @index(Global, Linear)
    idx = u_int32(i)

    if idx <= length(hit_queue.hit_found)
        shade_primary_hits_kernel_body!(hit_queue, scene, materials, sky_color, max_depth, shading_queue, idx)
    end
end

# ============================================================================
# Kernel: Generate Bounce Rays (BSDF Sampling)
# ============================================================================

# Process a single hit for bounce ray generation
# Returns (should_bounce, bounce_ray, new_beta, pixel, sample_idx, new_depth)
# Uses generated dispatch to handle multiple material types
@inline function process_bounce_hit_inner(
    hit_found::Bool, mat_idx::MaterialIndex,
    hit_p::Point3f, hit_n::Normal3f, hit_wo::Vec3f, hit_uv::Point2f,
    shading_n::Normal3f, shading_dpdu::Vec3f, shading_dpdv::Vec3f,
    ray::RayDifferentials, pixel::Point2f, beta::RGBSpectrum,
    sample_idx::Int32, depth::Int32,
    materials::Tuple, scene, max_depth::Int32
)::Tuple{Bool, RayDifferentials, RGBSpectrum, Point2f, Int32, Int32}
    if !hit_found || depth >= max_depth
        dummy_ray = RayDifferentials(Ray(Point3f(0), Vec3f(0, 0, 1), Inf32, 0f0))
        return (false, dummy_ray, RGBSpectrum(0f0), pixel, sample_idx, Int32(0))
    end

    # Reconstruct SurfaceInteraction
    si = reconstruct_surface_interaction(
        hit_p, hit_n, hit_wo, hit_uv,
        shading_n, shading_dpdu, shading_dpdv, ray.time
    )

    # Use generated dispatch based on material index
    should_bounce, bounce_ray, new_beta, new_depth = sample_material_bounce(
        materials, mat_idx, ray, si, scene, beta, Int32(depth)
    )

    return (should_bounce, bounce_ray, new_beta, pixel, sample_idx, Int32(new_depth))
end

# Full kernel body for bounce ray generation - reads from hit_queue, writes to bounce_ray_queue and bounce_active
@inline function generate_bounce_rays_kernel_body!(
    hit_queue, materials, scene, max_depth::Int32,
    bounce_ray_queue, bounce_active,
    idx::Int32
)
    @get hit_found, material_idx, hit_p, hit_n, hit_wo, hit_uv, shading_n, shading_dpdu, shading_dpdv,
         ray, pixel, beta, sample_idx, depth = hit_queue[idx]

    should_bounce, bounce_ray, new_beta, out_pixel, out_sample_idx, new_depth = process_bounce_hit_inner(
        hit_found, material_idx,
        hit_p, hit_n, hit_wo, hit_uv,
        shading_n, shading_dpdu, shading_dpdv,
        ray, pixel, beta, sample_idx, depth,
        materials, scene, max_depth
    )

    bounce_active[idx] = should_bounce
    if should_bounce
        @set bounce_ray_queue[idx] = (
            ray=bounce_ray, pixel=out_pixel, beta=new_beta,
            sample_idx=out_sample_idx, depth=new_depth
        )
    end
    return nothing
end

@kernel function generate_bounce_rays_kernel!(
    @Const(hit_queue),
    @Const(materials),
    @Const(scene),
    @Const(max_depth),
    bounce_ray_queue,
    bounce_active
)
    i = @index(Global, Linear)
    idx = u_int32(i)

    if idx <= length(hit_queue.hit_found)
        generate_bounce_rays_kernel_body!(hit_queue, materials, scene, max_depth, bounce_ray_queue, bounce_active, idx)
    end
end

# ============================================================================
# Kernel: Accumulate Final Image
# ============================================================================

# Inner function for accumulate_shading - can be tested with @code_warntype
@inline function accumulate_shading_inner!(
    color::RGBSpectrum,
    sample_accumulator::AbstractVector{RGBSpectrum},
    idx::Int32
)
    # Only accumulate valid colors
    if !any(isnan, color.c) && !is_black(color)
        sample_accumulator[idx] += color
    end
    return nothing
end

# Kernel to accumulate shading results into sample accumulator
@kernel function accumulate_shading_kernel!(
    @Const(shading_queue),
    sample_accumulator
)
    i = @index(Global, Linear)
    idx = u_int32(i)

    if idx <= length(shading_queue.color)
        color = shading_queue.color[idx]
        accumulate_shading_inner!(color, sample_accumulator, idx)
    end
end

# Inner function for copy_bounce_rays - can be tested with @code_warntype
@inline function copy_bounce_ray_inner!(
    bounce_ray_queue,
    bounce_active::AbstractVector{Bool},
    primary_ray_queue,
    idx::Int32
)
    if bounce_active[idx]
        primary_ray_queue.ray[idx] = bounce_ray_queue.ray[idx]
        primary_ray_queue.pixel[idx] = bounce_ray_queue.pixel[idx]
        primary_ray_queue.beta[idx] = bounce_ray_queue.beta[idx]
        primary_ray_queue.sample_idx[idx] = bounce_ray_queue.sample_idx[idx]
        primary_ray_queue.depth[idx] = bounce_ray_queue.depth[idx]
    else
        # Mark as inactive by setting beta to 0
        primary_ray_queue.beta[idx] = RGBSpectrum(0f0)
    end
    return nothing
end

# Kernel to copy active bounce rays to primary queue for next iteration
@kernel function copy_bounce_rays_kernel!(
    @Const(bounce_ray_queue),
    @Const(bounce_active),
    primary_ray_queue
)
    i = @index(Global, Linear)
    idx = u_int32(i)

    if idx <= length(bounce_active)
        copy_bounce_ray_inner!(bounce_ray_queue, bounce_active, primary_ray_queue, idx)
    end
end

# Inner function for finalize_image - can be tested with @code_warntype
# Uses generated function to unroll the sample accumulation loop
@generated function finalize_pixel_inner(
    sample_accumulator::AbstractVector{RGBSpectrum},
    pixel_idx::Int32,
    ::Val{NSamples}
)::RGB{Float32} where {NSamples}
    quote
        # Average all samples for this pixel using unrolled loop
        total = RGBSpectrum(0f0)
        Base.Cartesian.@nexprs $NSamples s -> begin
            s_idx = Int32(s)
            sample_idx = (pixel_idx - Int32(1)) * Int32($NSamples) + s_idx
            total += sample_accumulator[sample_idx]
        end

        avg_color = total / Float32($NSamples)

        # Clamp and convert to RGB
        r = clamp(avg_color[1], 0f0, 1f0)
        g = clamp(avg_color[2], 0f0, 1f0)
        b = clamp(avg_color[3], 0f0, 1f0)
        return RGB{Float32}(r, g, b)
    end
end

@kernel function finalize_image_kernel!(
    @Const(sample_accumulator),
    image,
    samples::Val{NSamples}
) where {NSamples}
    i = @index(Global, Cartesian)
    y = u_int32(i[1])
    x = u_int32(i[2])
    height, width = Int32.(size(image))

    if y <= height && x <= width
        pixel_idx = (y - Int32(1)) * width + x
        image[y, x] = finalize_pixel_inner(sample_accumulator, pixel_idx, samples)
    end
end

# ============================================================================
# Main Render Function
# ============================================================================

"""
    (integrator::WavefrontIntegrator)(scene::Scene, film::Film)

Render the scene using the wavefront path tracing algorithm.
"""
function (integrator::WavefrontIntegrator)(scene::Scene, film::Film)
    camera = integrator.camera
    max_depth = integrator.max_depth
    samples_per_pixel = Int(integrator.samples_per_pixel)

    # Get dimensions from film
    height, width = size(film.pixels)
    height = Int32(height)
    width = Int32(width)

    num_pixels = width * height
    num_rays = num_pixels * samples_per_pixel

    # Get backend from film
    backend = KA.get_backend(film.pixels.xyz)

    # Create a simple framebuffer for direct rendering
    framebuffer = film.framebuffer

    # Allocate work queues as SoA
    primary_ray_queue = similar_soa(framebuffer, WavefrontRayWork, num_rays)
    hit_queue = similar_soa(framebuffer, WavefrontHitWork, num_rays)
    shading_queue = similar_soa(framebuffer, WavefrontShadedResult, num_rays)
    bounce_ray_queue = similar_soa(framebuffer, WavefrontRayWork, num_rays)
    bounce_active = similar(framebuffer, Bool, num_rays)
    sample_accumulator = similar(framebuffer, RGBSpectrum, num_rays)

    # Initialize accumulator
    fill!(sample_accumulator, RGBSpectrum(0f0))

    # Create sampler
    sampler = UniformSampler(samples_per_pixel)

    # Sky color for misses
    sky_color = RGBSpectrum(0.5f0, 0.7f0, 1.0f0)

    # ========================================
    # Stage 1: Generate Primary Camera Rays
    # ========================================
    gen_kernel! = generate_primary_rays_kernel!(backend)
    gen_kernel!(
        width, height, camera, sampler, primary_ray_queue,
        Val(samples_per_pixel),
        ndrange=(height, width)
    )
    KA.synchronize(backend)

    # ========================================
    # Main Path Tracing Loop
    # ========================================
    current_ray_queue = primary_ray_queue
    materials = scene.aggregate.materials
    intersect_kernel! = intersect_rays_kernel!(backend)
    shade_kernel! = shade_primary_hits_kernel!(backend)
    accum_kernel! = accumulate_shading_kernel!(backend)
    bounce_kernel! = generate_bounce_rays_kernel!(backend)
    copy_kernel! = copy_bounce_rays_kernel!(backend)

    for bounce in 0:max_depth
        # Stage 2: Intersect Rays
        intersect_kernel!(
            scene.aggregate, current_ray_queue, hit_queue,
            ndrange=num_rays
        )
        KA.synchronize(backend)

        # Stage 3: Shade Hits (Direct Lighting)
        shade_kernel!(
            hit_queue, scene, materials, sky_color, max_depth, shading_queue,
            ndrange=num_rays
        )
        KA.synchronize(backend)

        # Accumulate shading results (GPU kernel)
        accum_kernel!(shading_queue, sample_accumulator, ndrange=num_rays)
        KA.synchronize(backend)

        # Stage 4: Generate Bounce Rays (if not at max depth)
        if bounce < max_depth
            bounce_kernel!(
                hit_queue, materials, scene, max_depth, bounce_ray_queue, bounce_active,
                ndrange=num_rays
            )
            KA.synchronize(backend)

            # Copy bounce rays to primary queue (GPU kernel)
            copy_kernel!(bounce_ray_queue, bounce_active, primary_ray_queue, ndrange=num_rays)
            KA.synchronize(backend)
        end
    end

    # ========================================
    # Stage 5: Finalize Image
    # ========================================
    final_kernel! = finalize_image_kernel!(backend)
    final_kernel!(
        sample_accumulator, framebuffer, Val(samples_per_pixel),
        ndrange=(height, width)
    )
    KA.synchronize(backend)

    return framebuffer
end

# ============================================================================
# Convenience Methods
# ============================================================================

"""
    render!(integrator::WavefrontIntegrator, scene::Scene, film::Film)

Alias for calling the integrator as a function.
"""
render!(integrator::WavefrontIntegrator, scene::Scene, film::Film) = integrator(scene, film)
