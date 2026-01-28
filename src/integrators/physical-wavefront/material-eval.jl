# Material evaluation and direct lighting for PhysicalWavefront
# Handles BSDF sampling, direct lighting with MIS, and path continuation
#
# NOTE: `materials` is a StaticMultiTypeVec containing both materials and textures.
# Use eval_tex(materials, field, uv) to sample textures via TextureRef.

# ============================================================================
# Direct Lighting Kernel
# ============================================================================

"""
    pw_sample_direct_lighting_kernel!(shadow_queue, material_queue, ...)

Sample direct lighting for all material evaluation work items.
For each item, selects a light, samples a direction, evaluates BSDF,
and creates a shadow ray work item.
"""
@kernel inbounds=true function pw_sample_direct_lighting_kernel!(
    shadow_queue,
    @Const(material_queue),
    @Const(materials),
    @Const(lights),        # Tuple of lights
    @Const(rgb2spec_table),
    @Const(num_lights::Int32),
    @Const(max_queued::Int32)
)
    idx = @index(Global)

     if idx <= max_queued
        current_size = material_queue.size[1]
        if idx <= current_size
            work = material_queue.items[idx]

            # Generate random numbers for light sampling using Julia's RNG
            u_light = rand(Point2f)
            light_select = rand(Float32)

            # Select a light uniformly
            light_idx = floor_int32(light_select * Float32(num_lights)) + Int32(1)
            light_idx = min(light_idx, num_lights)

            # Sample the selected light (works with both Tuple and StaticMultiTypeVec)
            p = work.pi
            light_sample = sample_light_spectral(rgb2spec_table, lights, light_idx, p, work.lambda, u_light)

            if light_sample.pdf > 0f0 && !is_black(light_sample.Li)
                # Evaluate BSDF for light direction
                bsdf_f, bsdf_pdf = evaluate_spectral_material(
                    rgb2spec_table, materials, work.material_idx,
                    work.wo, light_sample.wi, work.ns, work.uv, work.lambda
                )

                if !is_black(bsdf_f)
                    # Compute direct lighting contribution with MIS
                    result = compute_direct_lighting_spectral(
                        work.pi, work.ns, work.wo, work.beta, work.r_u, work.lambda,
                        light_sample, bsdf_f, bsdf_pdf
                    )

                    if result.valid
                        # Scale by number of lights (uniform selection PDF = 1/num_lights)
                        scaled_Ld = result.Ld * Float32(num_lights)

                        # Create shadow ray
                        shadow_ray = Raycore.Ray(o=result.ray_origin, d=result.ray_direction, t_max=result.t_max)
                        shadow_item = PWShadowRayWorkItem(
                            shadow_ray,
                            result.t_max,
                            work.lambda,
                            scaled_Ld,
                            result.r_u,
                            result.r_l,
                            work.pixel_index
                        )

                        push!(shadow_queue, shadow_item)
                    end
                end
            end
        end
    end
end

# ============================================================================
# Material Evaluation Kernel (BSDF Sampling + Path Continuation)
# ============================================================================

"""
    pw_evaluate_materials_kernel!(next_ray_queue, pixel_L, material_queue, ...)

Evaluate materials for all work items:
1. Sample BSDF for indirect lighting direction
2. Apply Russian roulette for path termination
3. Create continuation ray if path should continue
"""
@kernel inbounds=true function pw_evaluate_materials_kernel!(
    next_ray_queue,
    pixel_L,
    @Const(material_queue),
    @Const(materials),
    @Const(rgb2spec_table),
    @Const(max_depth::Int32),
    @Const(max_queued::Int32),
    @Const(do_regularize::Bool)  # Whether to apply BSDF regularization
)
    idx = @index(Global)

     if idx <= max_queued
        current_size = material_queue.size[1]
        if idx <= current_size
            work = material_queue.items[idx]

            # Generate random numbers for BSDF sampling using Julia's RNG
            u = rand(Point2f)
            rng = rand(Float32)
            rr_sample = rand(Float32)

            # Apply regularization if enabled and we've had a non-specular bounce
            # (pbrt-v4: regularize && anyNonSpecularBounces)
            regularize = do_regularize && work.any_non_specular_bounces

            # Sample BSDF
            sample = sample_spectral_material(
                rgb2spec_table, materials, work.material_idx,
                work.wo, work.ns, work.uv, work.lambda, u, rng, regularize
            )

            # Check if valid sample
            if sample.pdf > 0f0 && !is_black(sample.f)
                # Compute new throughput
                cos_theta = abs(dot(sample.wi, work.ns))
                new_beta = if sample.is_specular
                    # For specular, f already accounts for geometry
                    work.beta * sample.f
                else
                    work.beta * sample.f * cos_theta / sample.pdf
                end

                # Apply eta scale for refraction
                new_eta_scale = work.eta_scale * sample.eta_scale

                # Check depth limit
                new_depth = work.depth + Int32(1)
                if new_depth < max_depth
                    # Apply Russian roulette
                    should_continue, final_beta = russian_roulette_spectral(new_beta, new_depth, rr_sample)

                    if should_continue
                        # Spawn continuation ray
                        new_ray = spawn_spectral_ray(
                            work, sample.wi, final_beta,
                            sample.is_specular, sample.pdf, new_eta_scale
                        )

                        push!(next_ray_queue, new_ray)
                    end
                end
            end
        end
    end
end

# ============================================================================
# High-Level Functions
# ============================================================================

"""
    pw_sample_direct_lighting!(backend, shadow_queue, material_queue, materials, lights, rgb2spec_table)

Sample direct lighting for all material work items.
"""
function pw_sample_direct_lighting!(
    backend,
    shadow_queue::WorkQueue{PWShadowRayWorkItem},
    material_queue::WorkQueue{PWMaterialEvalWorkItem},
    materials,
    lights,
    rgb2spec_table::RGBToSpectrumTable
)
    n = length(material_queue)
    n == 0 && return nothing

    # Reset shadow queue
    empty!(shadow_queue)

    # Count lights in tuple
    num_lights = length(lights)

    kernel! = pw_sample_direct_lighting_kernel!(backend)
    kernel!(
        shadow_queue,
        material_queue,
        materials,
        lights,
        rgb2spec_table,
        num_lights, material_queue.capacity;
        ndrange=Int(material_queue.capacity)
    )

    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    pw_evaluate_materials!(backend, next_ray_queue, pixel_L, material_queue, materials, rgb2spec_table, max_depth, regularize=true)

Evaluate materials and spawn continuation rays.

When `regularize=true`, near-specular BSDFs are roughened after the first non-specular
bounce to reduce fireflies (matches pbrt-v4's BSDF::Regularize).
"""
function pw_evaluate_materials!(
    backend,
    next_ray_queue::WorkQueue{PWRayWorkItem},
    pixel_L::AbstractVector{Float32},
    material_queue::WorkQueue{PWMaterialEvalWorkItem},
    materials,
    rgb2spec_table::RGBToSpectrumTable,
    max_depth::Int32,
    regularize::Bool = true
)
    n = length(material_queue)
    n == 0 && return nothing

    kernel! = pw_evaluate_materials_kernel!(backend)
    kernel!(
        next_ray_queue,
        pixel_L,
        material_queue,
        materials,
        rgb2spec_table,
        max_depth, material_queue.capacity, regularize;
        ndrange=Int(material_queue.capacity)
    )

    KernelAbstractions.synchronize(backend)
    return nothing
end

# ============================================================================
# Auxiliary Buffer Population (for denoising)
# ============================================================================

"""
    pw_populate_aux_buffers_kernel!(aux_albedo, aux_normal, aux_depth, material_queue, ...)

Populate auxiliary buffers for denoising on first bounce.
Only processes depth=0 items (primary ray hits).
"""
@kernel inbounds=true function pw_populate_aux_buffers_kernel!(
    aux_albedo,   # 3 floats per pixel (RGB)
    aux_normal,   # 3 floats per pixel
    aux_depth,    # 1 float per pixel
    @Const(material_queue),
    @Const(materials),
    @Const(rgb2spec_table),
    @Const(max_queued::Int32)
)
    idx = @index(Global)

     if idx <= max_queued
        current_size = material_queue.size[1]
        if idx <= current_size
            work = material_queue.items[idx]

            # Only populate on first bounce
            if work.depth == Int32(0)
                pixel_idx = work.pixel_index

                # Get material albedo (spectral, then convert to RGB average)
                albedo_spec = get_albedo_spectral_dispatch(
                    rgb2spec_table, materials, work.material_idx, work.uv, work.lambda
                )
                # Use average of spectral values as RGB approximation
                albedo_avg = average(albedo_spec)

                # Store albedo (grey for now, could be RGB if we track hero wavelength)
                albedo_base = (pixel_idx - Int32(1)) * Int32(3)
                aux_albedo[albedo_base + Int32(1)] = albedo_avg
                aux_albedo[albedo_base + Int32(2)] = albedo_avg
                aux_albedo[albedo_base + Int32(3)] = albedo_avg

                # Store normal (world space, remapped to [0,1])
                normal_base = (pixel_idx - Int32(1)) * Int32(3)
                aux_normal[normal_base + Int32(1)] = work.ns[1] * 0.5f0 + 0.5f0
                aux_normal[normal_base + Int32(2)] = work.ns[2] * 0.5f0 + 0.5f0
                aux_normal[normal_base + Int32(3)] = work.ns[3] * 0.5f0 + 0.5f0

                # Store depth (distance from camera)
                # Compute from intersection point (would need camera position for proper depth)
                # For now, use z-coordinate as proxy
                aux_depth[pixel_idx] = work.pi[3]
            end
        end
    end
end

"""
    pw_populate_aux_buffers!(backend, aux_albedo, aux_normal, aux_depth,
                              material_queue, materials, rgb2spec_table)

Populate auxiliary buffers for denoising.
"""
function pw_populate_aux_buffers!(
    backend,
    aux_albedo::AbstractVector{Float32},
    aux_normal::AbstractVector{Float32},
    aux_depth::AbstractVector{Float32},
    material_queue::WorkQueue{PWMaterialEvalWorkItem},
    materials,
    rgb2spec_table::RGBToSpectrumTable
)
    n = length(material_queue)
    n == 0 && return nothing

    kernel! = pw_populate_aux_buffers_kernel!(backend)
    kernel!(
        aux_albedo, aux_normal, aux_depth,
        material_queue,
        materials,
        rgb2spec_table,
        material_queue.capacity;
        ndrange=Int(material_queue.capacity)
    )

    KernelAbstractions.synchronize(backend)
    return nothing
end
