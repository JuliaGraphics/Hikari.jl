# ============================================================================
# Fast Wavefront Path Tracer Integrator
# ============================================================================
# A simplified wavefront renderer with basic diffuse + shadow + reflection
# shading. Uses the same interface as Whitted: (scene, film, camera)

using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const
import KernelAbstractions as KA

# ============================================================================
# SoA Access Macros
# ============================================================================

macro fast_get(expr)
    if expr.head != :(=)
        error("@fast_get expects assignment syntax")
    end
    lhs = expr.args[1]
    rhs = expr.args[2]
    fields = lhs isa Symbol ? [lhs] : lhs.args
    if rhs.head != :ref
        error("@fast_get right side must be array indexing")
    end
    soa = rhs.args[1]
    idx = rhs.args[2]
    assignments = [:($(esc(field)) = $(esc(soa)).$(field)[$(esc(idx))]) for field in fields]
    return Expr(:block, assignments...)
end

macro fast_set(expr)
    if expr.head != :(=)
        error("@fast_set expects assignment syntax")
    end
    lhs = expr.args[1]
    rhs = expr.args[2]
    if lhs.head != :ref
        error("@fast_set left side must be array indexing")
    end
    soa = lhs.args[1]
    idx = lhs.args[2]
    assignments = []
    if rhs.head == :tuple || rhs.head == :parameters
        for arg in rhs.args
            if arg isa Expr && arg.head == :(=)
                field = arg.args[1]
                val = arg.args[2]
                push!(assignments, :($(esc(soa)).$(field)[$(esc(idx))] = $(esc(val))))
            end
        end
    end
    return Expr(:block, assignments...)
end

# ============================================================================
# Material Properties for Fast Shading
# ============================================================================

"""
Simple material properties extracted from Hikari materials for fast shading.
"""
struct FastMaterialProps
    base_color::Vec3f
    metallic::Float32
    roughness::Float32
end

# Material property extractors using proper texture evaluation
@propagate_inbounds function extract_fast_props(mat::MatteMaterial, uv::Point2f)
    kd = eval_tex((), mat.Kd, uv)
    σ = eval_tex((), mat.σ, uv)
    FastMaterialProps(Vec3f(kd.c[1], kd.c[2], kd.c[3]), 0f0, clamp(σ / 90f0, 0f0, 1f0))
end

@propagate_inbounds function extract_fast_props(mat::MirrorMaterial, uv::Point2f)
    kr = eval_tex((), mat.Kr, uv)
    FastMaterialProps(Vec3f(kr.c[1], kr.c[2], kr.c[3]), 1f0, 0f0)
end

@propagate_inbounds function extract_fast_props(mat::PlasticMaterial, uv::Point2f)
    # Plastic is a dielectric - no mirror reflections in FastWavefront
    # (true plastic specular requires BRDF evaluation which FastWavefront doesn't do)
    kd = eval_tex((), mat.Kd, uv)
    roughness = eval_tex((), mat.roughness, uv)
    FastMaterialProps(Vec3f(kd.c[1], kd.c[2], kd.c[3]), 0f0, roughness)
end

@propagate_inbounds function extract_fast_props(mat::GlassMaterial, uv::Point2f)
    kr = eval_tex((), mat.Kr, uv)
    roughness = eval_tex((), mat.u_roughness, uv)
    FastMaterialProps(Vec3f(kr.c[1], kr.c[2], kr.c[3]), 0.8f0, roughness)
end

@propagate_inbounds function extract_fast_props(mat::MetalMaterial, uv::Point2f)
    refl = eval_tex((), mat.reflectance, uv)
    roughness = eval_tex((), mat.roughness, uv)
    FastMaterialProps(Vec3f(refl.c[1], refl.c[2], refl.c[3]), 1f0, roughness)
end

# Fallback
@propagate_inbounds function extract_fast_props(mat, ::Point2f)
    FastMaterialProps(Vec3f(0.5f0), 0f0, 0.5f0)
end

# Generated dispatch for material property extraction (type-stable)
# Uses T<:Tuple to preserve concrete element types for proper dispatch
@propagate_inbounds @generated function extract_material_props(
    materials::T, idx::MaterialIndex, uv::Point2f
) where {T<:Tuple}
    N = length(T.parameters)
    branches = [quote
         if idx.material_type === UInt8($i)
            return extract_fast_props(materials[$i][idx.material_idx], uv)
        end
    end for i in 1:N]
    quote
        $(branches...)
        return FastMaterialProps(Vec3f(0.5f0), 0f0, 0.5f0)
    end
end

# ============================================================================
# Light helpers
# ============================================================================

@propagate_inbounds function get_light_contribution(light::PointLight, hit_point::Point3f, normal::Vec3f)
    light_vec = light.position - hit_point
    light_dist = norm(light_vec)
    light_dir = light_vec / light_dist
    diffuse = max(0.0f0, dot(normal, light_dir))
    # Match Whitted's sample_li: radiance = light.i / distance²
    # This uses the full RGB spectrum as intensity (physically correct)
    light_rgb = rgb(light.i)
    dist_sq = light_dist * light_dist
    radiance = Vec3f(light_rgb[1] / dist_sq, light_rgb[2] / dist_sq, light_rgb[3] / dist_sq)
    return radiance * diffuse, light_dir, light_dist, true  # true = needs shadow test
end

# AmbientLight: uniform contribution from all directions, no shadow test needed
# Approximate Whitted's direction-dependent ambient by using the hit point position
@propagate_inbounds function get_light_contribution(light::AmbientLight, hit_point::Point3f, normal::Vec3f)
    light_rgb = rgb(light.i)
    # Whitted uses wi = normalize(hit_point) and computes (albedo/π) * |wi·n|
    # We approximate this by computing the same directional factor
    wi = normalize(Vec3f(hit_point))
    # Lambertian BRDF factor: 1/π ≈ 0.318, times |cos(θ)|
    cos_factor = abs(dot(wi, normal))
    scale = 0.318f0 * cos_factor
    radiance = Vec3f(light_rgb[1] * scale, light_rgb[2] * scale, light_rgb[3] * scale)
    # Ambient doesn't need shadow test
    return radiance, Vec3f(0, 1, 0), 1f6, false  # false = no shadow test needed
end

# Fallback for other light types
@propagate_inbounds function get_light_contribution(light::Light, hit_point::Point3f, normal::Vec3f)
    return Vec3f(0), Vec3f(0, 1, 0), 1f6, true
end

# ============================================================================
# Shadow Ray Generation (per light type)
# ============================================================================

# PointLight: shadow ray from hit point to light position
@propagate_inbounds function make_shadow_ray(light::PointLight, hit_point::Point3f, normal::Vec3f)
    shadow_origin = hit_point + normal * 0.001f0
    light_vec = light.position - shadow_origin
    shadow_dir = normalize(light_vec)
    light_dist = norm(light_vec)
    return Ray(o=shadow_origin, d=shadow_dir, t_max=light_dist - 0.001f0)
end

# AmbientLight: dummy shadow ray with t_max=0 (will fail shadow test automatically)
@propagate_inbounds function make_shadow_ray(light::AmbientLight, hit_point::Point3f, normal::Vec3f)
    return Ray(o=Point3f(0,0,0), d=Vec3f(0,0,1), t_max=0.0f0)
end

# Fallback: dummy shadow ray
@propagate_inbounds function make_shadow_ray(light::Light, hit_point::Point3f, normal::Vec3f)
    return Ray(o=Point3f(0,0,0), d=Vec3f(0,0,1), t_max=0.0f0)
end

# ============================================================================
# Work Queue Structures (with UV)
# ============================================================================

struct FastPrimaryRayWork
    ray::Ray
    pixel_x::Int32
    pixel_y::Int32
    sample_idx::Int32
end

struct FastPrimaryHitWork{Tri}
    hit_found::Bool
    tri::Tri
    dist::Float32
    bary::Vec3f
    uv::Point2f
    ray::Ray
    pixel_x::Int32
    pixel_y::Int32
    sample_idx::Int32
end

struct FastShadowRayWork
    ray::Ray
    hit_idx::Int32
    light_idx::Int32
end

struct FastShadowResult
    visible::Bool
    hit_idx::Int32
    light_idx::Int32
end

struct FastReflectionRayWork
    ray::Ray
    hit_idx::Int32
end

struct FastReflectionHitWork
    hit_found::Bool
    material_idx::MaterialIndex
    dist::Float32
    bary::Vec3f
    uv::Point2f
    normal::Vec3f
    ray::Ray
    primary_hit_idx::Int32
end

struct FastShadedResult
    color::Vec3f
    pixel_x::Int32
    pixel_y::Int32
    sample_idx::Int32
end

# ============================================================================
# Buffer Cache
# ============================================================================

mutable struct FastWavefrontBuffers{ImgArr, Accel, Materials}
    width::Int32
    height::Int32
    samples_per_pixel::Int32
    num_lights::Int32

    framebuffer::ImgArr
    accel::Accel
    materials::Materials

    # Work queues
    primary_ray_queue::NamedTuple
    primary_hit_queue::NamedTuple
    shadow_ray_queue::NamedTuple
    shadow_result_queue::NamedTuple
    reflection_ray_soa::NamedTuple
    reflection_hit_soa::NamedTuple
    shading_queue::NamedTuple
    sample_accumulator::AbstractVector
    active_count::AbstractVector
end

function similar_soa(img, T, num_elements)
    fields = [f => similar(img, fieldtype(T, f), num_elements) for f in fieldnames(T)]
    return (; fields...)
end

function create_buffers(
    img::AbstractMatrix, accel, materials, samples_per_pixel::Int, num_lights::Int
)
    height, width = size(img)
    num_pixels = width * height
    num_rays = num_pixels * samples_per_pixel
    num_shadow_rays = num_rays * num_lights

    tri_type = eltype(accel)

    primary_ray_queue = similar_soa(img, FastPrimaryRayWork, num_rays)
    primary_hit_queue = similar_soa(img, FastPrimaryHitWork{tri_type}, num_rays)
    shadow_ray_queue = similar_soa(img, FastShadowRayWork, num_shadow_rays)
    shadow_result_queue = similar_soa(img, FastShadowResult, num_shadow_rays)
    reflection_ray_soa = similar_soa(img, FastReflectionRayWork, num_rays)
    reflection_hit_soa = similar_soa(img, FastReflectionHitWork, num_rays)
    shading_queue = similar_soa(img, FastShadedResult, num_rays)
    sample_accumulator = similar(img, Vec3f, num_rays)
    active_count = similar(img, Int32, 1)

    return FastWavefrontBuffers(
        Int32(width), Int32(height), Int32(samples_per_pixel), Int32(num_lights),
        img, accel, materials,
        primary_ray_queue, primary_hit_queue,
        shadow_ray_queue, shadow_result_queue,
        reflection_ray_soa, reflection_hit_soa,
        shading_queue, sample_accumulator, active_count
    )
end

# ============================================================================
# UV Computation
# ============================================================================

@propagate_inbounds function compute_uv(tri, bary::Vec3f)
    uv0, uv1, uv2 = Raycore.uvs(tri)
    u, v, w = bary[1], bary[2], bary[3]
    return Point2f(uv0 * u + uv1 * v + uv2 * w)
end

# ============================================================================
# Kernels
# ============================================================================

@generated function for_unrolled(f::F, ::Val{N}) where {F, N}
    return Expr(:block, [:(f($(Int32(i)))) for i in 1:N]...)
end

# Stage 1: Generate Primary Rays using standard camera interface
@kernel inbounds=true function fast_generate_rays!(
    @Const(width), @Const(height),
    @Const(camera),
    ray_queue,
    ::Val{NSamples}
) where {NSamples}
    i = @index(Global, NTuple)
    y = u_int32(i[1])
    x = u_int32(i[2])

     if y <= height && x <= width
        pixel_idx = (y - Int32(1)) * width + x
        for s in 1:NSamples
            s_idx = Int32(s)
            ray_idx = (pixel_idx - Int32(1)) * Int32(NSamples) + s_idx
            jitter = rand(Vec2f)

            # Use standard camera interface - flip Y to match Whitted convention
            p_film = Point2f(Float32(x) + jitter[1], Float32(height) - Float32(y) + jitter[2])
            camera_sample = CameraSample(p_film, Point2f(0), 0f0)
            ray, _ = generate_ray(camera, camera_sample)
            @fast_set ray_queue[ray_idx] = (ray=ray, pixel_x=x, pixel_y=y, sample_idx=s_idx)
        end
    end
end

# Stage 2: Intersect Primary Rays
@kernel inbounds=true function fast_intersect_primary!(
    @Const(accel),
    @Const(ray_queue),
    hit_queue
)
    i = @index(Global, Linear)
    idx = u_int32(i)

     if idx <= length(ray_queue.ray)
        @fast_get ray, pixel_x, pixel_y, sample_idx = ray_queue[idx]
        hit_found, tri, dist, bary = closest_hit(accel, ray)
        uv = hit_found ? compute_uv(tri, Vec3f(bary)) : Point2f(0)
        @fast_set hit_queue[idx] = (hit_found=hit_found, tri=tri, dist=dist, bary=Vec3f(bary),
                                    uv=uv, ray=ray, pixel_x=pixel_x, pixel_y=pixel_y, sample_idx=sample_idx)
    end
end

# GPU-safe helper: Generate shadow ray for one light at given index
@propagate_inbounds function _generate_shadow_ray_for_light!(
    light_idx::Int32, shadow_ray_queue, lights, hit_point, normal, idx::Int32, ::Val{NLights}
) where NLights
    shadow_ray_idx = (idx - Int32(1)) * Int32(NLights) + light_idx
    shadow_ray = make_shadow_ray(lights[light_idx], hit_point, normal)
    @fast_set shadow_ray_queue[shadow_ray_idx] = (ray=shadow_ray, hit_idx=idx, light_idx=light_idx)
    return nothing
end

# GPU-safe helper: Set dummy ray for one light index
@propagate_inbounds function _set_dummy_shadow_ray!(
    light_idx::Int32, shadow_ray_queue, dummy_ray, idx::Int32, ::Val{NLights}
) where NLights
    shadow_ray_idx = (idx - Int32(1)) * Int32(NLights) + light_idx
    shadow_ray_queue.ray[shadow_ray_idx] = dummy_ray
    return nothing
end

# Helper: Process one hit and generate all shadow rays for it
@propagate_inbounds function process_hit_and_generate_shadow_rays!(
    hit_queue,
    shadow_ray_queue,
    lights::T,
    idx::Int32,
    nlights::Val{NLights}
) where {NLights, T}
    @fast_get hit_found, tri, dist, bary, ray = hit_queue[idx]
    if hit_found
        hit_point = ray.o + ray.d * dist
        v0, v1, v2 = Raycore.normals(tri)
        u, v, w = bary[1], bary[2], bary[3]
        normal = Vec3f(normalize(v0 * u + v1 * v + v2 * w))

        # Generate shadow rays for ALL lights using GPU-safe unrolled iteration
        for_unrolled(_generate_shadow_ray_for_light!, Val(NLights),
                     shadow_ray_queue, lights, hit_point, normal, idx, nlights)
    else
        dummy_ray = Ray(o=Point3f(0,0,0), d=Vec3f(0,0,1), t_max=0.0f0)
        # Set dummy rays for all lights using GPU-safe unrolled iteration
        for_unrolled(_set_dummy_shadow_ray!, Val(NLights),
                     shadow_ray_queue, dummy_ray, idx, nlights)
    end
    return nothing
end

# Stage 3: Generate Shadow Rays
@kernel inbounds=true function fast_generate_shadow_rays!(
    @Const(hit_queue),
    @Const(lights),
    shadow_ray_queue,
    nlights::Val{NLights}
) where {NLights}
    i = @index(Global, Linear)
    idx = u_int32(i)

     if idx <= length(hit_queue.hit_found)
        process_hit_and_generate_shadow_rays!(
            hit_queue, shadow_ray_queue, lights, idx, nlights
        )
    end
end

# Stage 4: Test Shadow Rays
@kernel inbounds=true function fast_test_shadow_rays!(
    @Const(accel),
    @Const(shadow_ray_queue),
    shadow_result_queue
)
    i = @index(Global, Linear)
    idx = u_int32(i)
     if idx <= length(shadow_ray_queue.ray)
        @fast_get ray, hit_idx, light_idx = shadow_ray_queue[idx]

        visible = if ray.t_max > 0.0f0
            hit_found, _, _, _ = any_hit(accel, ray)
            !hit_found
        else
            false
        end

        @fast_set shadow_result_queue[idx] = (visible=visible, hit_idx=hit_idx, light_idx=light_idx)
    end
end

@generated function accumulate_lights(
    idx::Int32, hit_point::Point3f, normal::Vec3f, base_color::Vec3f, lights::NTuple{N,Light}, shadow_results
) where {N}
    quote
        acc = zero(Vec3f)
        $(Expr(:block, [
            quote
                light = lights[$i]
                shadow_idx = (idx - Int32(1)) * Int32($N) + Int32($i)
                visible = shadow_results.visible[shadow_idx]
                light_contrib, _, _, needs_shadow =
                    get_light_contribution(light, hit_point, normal)
                if !needs_shadow || visible
                    acc += Vec3f(base_color .* light_contrib)
                end
            end
            for i in 1:N
        ]...))
        acc
    end
end

# Helper: Accumulate light contributions using ntuple (sum is unrolled and fast on tuples)
@noinline function accumulate_lights(idx::Int32, hit_point::Point3f, normal::Vec3f, base_color::Vec3f, lights::Tuple, shadow_results)
    N = length(lights)
    contributions = ntuple(Val(N)) do li
        shadow_idx = (idx - Int32(1)) * Int32(N) + Int32(li)
        visible = shadow_results.visible[shadow_idx]
        light = lights[li]
        light_contrib, _, _, needs_shadow = get_light_contribution(light, hit_point, normal)

        if (!needs_shadow) || visible
            Vec3f(base_color .* light_contrib)
        else
            Vec3f(0)
        end
    end
    return sum(contributions)
end

# Stage 5: Shade Primary Hits
@kernel inbounds=true function fast_shade_primary!(
    @Const(hit_queue),
    @Const(materials),
    @Const(lights),
    @Const(shadow_results),
    @Const(sky_color),
    @Const(ambient),
    shading_queue,
    nlights::Val{NLights}
) where {NLights}
    i = @index(Global, Linear)
    idx = u_int32(i)

     if idx <= length(hit_queue.hit_found)
        @fast_get hit_found, tri, dist, bary, uv, ray, pixel_x, pixel_y, sample_idx = hit_queue[idx]

        if hit_found
            hit_point = ray.o + ray.d * dist
            v0, v1, v2 = Raycore.normals(tri)
            u, v, w = bary[1], bary[2], bary[3]
            normal = Vec3f(normalize(v0 * u + v1 * v + v2 * w))

            mat_idx = tri.metadata::MaterialIndex
            mat_props = extract_material_props(materials, mat_idx, uv)
            base_color = mat_props.base_color

            # Accumulate lighting from all lights
            final_color = accumulate_lights(idx, hit_point, normal, base_color, lights, shadow_results)
            @fast_set shading_queue[idx] = (color=final_color, pixel_x=pixel_x, pixel_y=pixel_y, sample_idx=sample_idx)
        else
            sky_vec = Vec3f(sky_color.r, sky_color.g, sky_color.b)
            @fast_set shading_queue[idx] = (color=sky_vec, pixel_x=pixel_x, pixel_y=pixel_y, sample_idx=sample_idx)
        end
    end
end

# Stage 6: Generate Reflection Rays
@kernel inbounds=true function fast_generate_reflections!(
    @Const(hit_queue),
    @Const(materials),
    reflection_ray_soa,
    active_count
)
    i = @index(Global, Linear)
    idx = u_int32(i)

     if idx <= length(hit_queue.hit_found)
        @fast_get hit_found, tri, dist, bary, uv, ray = hit_queue[idx]
        dummy_ray = Ray(o=Point3f(0), d=Vec3f(0, 0, 1), t_max=0.0f0)

        if hit_found
            mat_idx = tri.metadata::MaterialIndex
            mat_props = extract_material_props(materials, mat_idx, uv)

            if mat_props.metallic > 0.0f0
                hit_point = ray.o + ray.d * dist
                v0, v1, v2 = Raycore.normals(tri)
                u, v, w = bary[1], bary[2], bary[3]
                normal = Vec3f(normalize(v0 * u + v1 * v + v2 * w))

                wo = -ray.d
                reflect_dir = Raycore.reflect(wo, normal)

                if mat_props.roughness > 0.0f0
                    offset = (rand(Vec3f) .* 2.0f0 .- 1.0f0) * mat_props.roughness
                    reflect_dir = normalize(reflect_dir + offset)
                end

                reflect_ray = Ray(o=hit_point + normal * 0.001f0, d=reflect_dir)
                @fast_set reflection_ray_soa[idx] = (ray=reflect_ray, hit_idx=idx)
            else
                reflection_ray_soa.ray[idx] = dummy_ray
            end
        else
            reflection_ray_soa.ray[idx] = dummy_ray
        end
    end
end

# Stage 7: Intersect Reflection Rays
@kernel inbounds=true function fast_intersect_reflections!(
    @Const(accel),
    @Const(reflection_ray_soa),
    reflection_hit_soa
)
    i = @index(Global, Linear)
    idx = u_int32(i)

     if idx <= length(reflection_ray_soa.ray)
        @fast_get ray, hit_idx = reflection_ray_soa[idx]

        if ray.t_max > 0.0f0
            hit_found, tri, dist, bary = closest_hit(accel, ray)
            if hit_found
                v0, v1, v2 = Raycore.normals(tri)
                u, v, w = bary[1], bary[2], bary[3]
                normal = Vec3f(normalize(v0 * u + v1 * v + v2 * w))
                uv = compute_uv(tri, Vec3f(bary))

                mat_idx = tri.metadata::MaterialIndex
                @fast_set reflection_hit_soa[idx] = (hit_found=true, material_idx=mat_idx,
                    dist=dist, bary=Vec3f(bary), uv=uv, normal=normal,
                    ray=ray, primary_hit_idx=hit_idx)
            else
                reflection_hit_soa.hit_found[idx] = false
            end
        else
            reflection_hit_soa.hit_found[idx] = false
        end
    end
end

# Stage 8: Shade Reflections and Blend
@kernel inbounds=true function fast_shade_reflections!(
    @Const(hit_queue),
    @Const(reflection_hit_soa),
    @Const(materials),
    @Const(lights),
    @Const(sky_color),
    @Const(ambient),
    shading_queue
)
    i = @index(Global, Linear)
    idx = u_int32(i)

     if idx <= length(hit_queue.hit_found)
        @fast_get hit_found, tri, uv, pixel_x, pixel_y, sample_idx = hit_queue[idx]

        if hit_found
            mat_idx = tri.metadata::MaterialIndex
            mat_props = extract_material_props(materials, mat_idx, uv)

            if mat_props.metallic > 0.0f0
                @fast_get hit_found, material_idx, dist, bary, uv, normal, ray = reflection_hit_soa[idx]

                reflection_color = if hit_found
                    refl_point = ray.o + ray.d * dist
                    refl_normal = normal

                    refl_mat_props = extract_material_props(materials, material_idx, uv)
                    refl_base_color = refl_mat_props.base_color

                    refl_color = Vec3f(0)
                    if length(lights) > 0
                        light = lights[Int32(1)]
                        light_contrib, _, _, _ = get_light_contribution(light, refl_point, refl_normal)
                        refl_color = refl_base_color .* light_contrib
                    end
                    refl_color
                else
                    Vec3f(sky_color.r, sky_color.g, sky_color.b)
                end

                # For metals, tint the reflection by the metal's base color (reflectance)
                # This gives metals their characteristic colored reflections
                metal_tint = mat_props.base_color
                tinted_reflection = reflection_color .* metal_tint

                primary_color = shading_queue.color[idx]
                blended_color = primary_color * (1.0f0 - mat_props.metallic) + tinted_reflection * mat_props.metallic

                @fast_set shading_queue[idx] = (color=blended_color, pixel_x=pixel_x, pixel_y=pixel_y, sample_idx=sample_idx)
            end
        end
    end
end

# Stage 9: Accumulate
@kernel inbounds=true function fast_accumulate!(
    @Const(shading_queue),
    img,
    sample_accumulator
)
    i = @index(Global, Linear)
    idx = u_int32(i)

     if idx <= length(shading_queue.color)
        sample_accumulator[idx] = shading_queue.color[idx]
    end
end

@kernel inbounds=true function fast_finalize!(
    @Const(sample_accumulator),
    img,
    nsamples::Val{NSamples}
) where {NSamples}
    i = @index(Global, Cartesian)
    y = u_int32(i[1])
    x = u_int32(i[2])
    height, width = size(img)

     if y <= height && x <= width
        pixel_idx = (y - Int32(1)) * width + x
        avg_color = Vec3f(0.0f0)
        for s_idx in Int32(1):Int32(NSamples)
            sample_idx = (pixel_idx - Int32(1)) * Int32(NSamples) + s_idx
            avg_color += sample_accumulator[sample_idx]
        end
        avg_color = avg_color / Float32(NSamples)
        img[y, x] = RGB{Float32}(avg_color...)
    end
end

# ============================================================================
# FastWavefront Integrator
# ============================================================================

mutable struct FastWavefront <: Integrator
    samples::Int32
    ambient::Float32
    buffers::Union{Nothing, FastWavefrontBuffers}
end

function FastWavefront(;
        samples::Integer = 4,
        ambient::Real = 0.1f0,
    )
    return FastWavefront(Int32(samples), Float32(ambient), nothing)
end

# GPU-safe helper for extracting sky color from a light
@propagate_inbounds function _extract_sky_from_light(acc::RGB{Float32}, light)
    # If we already found a sky color, keep it
    acc != RGB{Float32}(0f0, 0f0, 0f0) && return acc
    # Check if this light is a SunSkyLight
    if light isa SunSkyLight
        zenith_dir = Vec3f(0f0, 0f0, 1f0)
        sky = sky_radiance(light, zenith_dir)
        return RGB{Float32}(sky.c[1], sky.c[2], sky.c[3])
    end
    return acc
end

"""
Extract sky color from scene lights. Returns RGB for background rays.
If SunSkyLight is present, samples sky at zenith direction.
"""
function extract_sky_color(lights)::RGB{Float32}
    # Use reduce_unrolled for GPU-safe iteration
    reduce_unrolled(_extract_sky_from_light, lights, RGB{Float32}(0f0, 0f0, 0f0))
end

"""
Convert SunSkyLight to DirectionalLight for shadow ray calculations.
Returns a new lights tuple with SunSkyLight replaced by DirectionalLight.
"""
function convert_lights_for_fast_wavefront(lights, world_radius::Float32)
    converted = map(lights) do light
        if light isa SunSkyLight
            # Convert to DirectionalLight using sun direction and intensity
            # DirectionalLight expects direction light TRAVELS (opposite of sun_direction)
            # sun_direction points TO the sun, so we negate it
            DirectionalLight(
                Transformation(),  # Identity transform
                light.sun_intensity,
                -light.sun_direction,  # Direction light travels (away from sun)
            )
        else
            light
        end
    end
    return converted
end

# ============================================================================
# Functor Interface - same as Whitted
# ============================================================================

function (integrator::FastWavefront)(scene::AbstractScene, film::Film, camera::Camera)
    img = film.framebuffer
    accel = scene.aggregate.accel
    materials = scene.aggregate.materials
    original_lights = scene.lights

    # Extract sky color from SunSkyLight if present
    sky_color = extract_sky_color(original_lights)

    # Convert SunSkyLight to DirectionalLight for shadow ray calculations
    lights = convert_lights_for_fast_wavefront(original_lights, scene.world_radius)

    num_lights = length(lights)
    samples_per_pixel = Int(integrator.samples)

    height, width = size(img)

    # Initialize or rebuild buffers if needed
    if integrator.buffers === nothing ||
       integrator.buffers.width != width ||
       integrator.buffers.height != height ||
       integrator.buffers.samples_per_pixel != samples_per_pixel ||
       integrator.buffers.num_lights != num_lights
        integrator.buffers = create_buffers(img, accel, materials, samples_per_pixel, num_lights)
    end

    buffers = integrator.buffers
    backend = KA.get_backend(img)

    num_pixels = width * height
    num_rays = num_pixels * samples_per_pixel
    num_shadow_rays = num_rays * num_lights

    # Stage 1: Generate primary rays
    gen_kernel! = fast_generate_rays!(backend)
    gen_kernel!(
        Int32(width), Int32(height),
        camera,
        buffers.primary_ray_queue,
        Val(samples_per_pixel),
        ndrange=(height, width)
    )

    # Stage 2: Intersect primary rays
    intersect_kernel! = fast_intersect_primary!(backend)
    intersect_kernel!(
        accel,
        buffers.primary_ray_queue,
        buffers.primary_hit_queue,
        ndrange=num_rays
    )

    # Stage 3: Generate shadow rays
    shadow_gen_kernel! = fast_generate_shadow_rays!(backend)
    shadow_gen_kernel!(
        buffers.primary_hit_queue,
        lights,
        buffers.shadow_ray_queue,
        Val(num_lights),
        ndrange=num_rays
    )

    # Stage 4: Test shadow rays
    shadow_test_kernel! = fast_test_shadow_rays!(backend)
    shadow_test_kernel!(
        accel,
        buffers.shadow_ray_queue,
        buffers.shadow_result_queue,
        ndrange=num_shadow_rays
    )

    # Stage 5: Shade primary hits
    shade_kernel! = fast_shade_primary!(backend)
    shade_kernel!(
        buffers.primary_hit_queue,
        materials,
        lights,
        buffers.shadow_result_queue,
        sky_color,
        integrator.ambient,
        buffers.shading_queue,
        Val(num_lights),
        ndrange=num_rays
    )

    # Stage 6: Generate reflection rays
    refl_gen_kernel! = fast_generate_reflections!(backend)
    refl_gen_kernel!(
        buffers.primary_hit_queue,
        materials,
        buffers.reflection_ray_soa,
        buffers.active_count,
        ndrange=num_rays
    )

    # Stage 7: Intersect reflection rays
    refl_intersect_kernel! = fast_intersect_reflections!(backend)
    refl_intersect_kernel!(
        accel,
        buffers.reflection_ray_soa,
        buffers.reflection_hit_soa,
        ndrange=num_rays
    )

    # Stage 8: Shade reflections
    refl_shade_kernel! = fast_shade_reflections!(backend)
    refl_shade_kernel!(
        buffers.primary_hit_queue,
        buffers.reflection_hit_soa,
        materials,
        lights,
        sky_color,
        integrator.ambient,
        buffers.shading_queue,
        ndrange=num_rays
    )

    # Stage 9: Accumulate final image
    accum_kernel! = fast_accumulate!(backend)
    accum_kernel!(
        buffers.shading_queue,
        img,
        buffers.sample_accumulator,
        ndrange=num_rays
    )

    final_kernel! = fast_finalize!(backend)
    final_kernel!(
        buffers.sample_accumulator,
        img,
        Val(samples_per_pixel),
        ndrange=(height, width)
    )
    KA.synchronize(backend)

    return img
end
