# Surface material evaluation for VolPath
# Handles BSDF sampling, direct lighting, and path continuation at surface hits
#
# NOTE: All kernels take a `textures` parameter for GPU compatibility.
# On CPU, textures is ignored (Texture structs contain their data).
# On GPU, textures is a tuple of CLDeviceArrays, and materials contain TextureRef.

# ============================================================================
# Process Surface Hits - Emission and Material Queue Setup
# ============================================================================

"""
    vp_process_surface_hits_kernel!(...)

Process surface hits:
1. Add emission from emissive surfaces (with MIS weight)
2. Create material evaluation work items for BSDF sampling
"""
@kernel inbounds=true function vp_process_surface_hits_kernel!(
    # Output
    material_items, material_size,
    pixel_L,
    # Input
    @Const(hit_items), @Const(hit_size),
    @Const(materials),
    @Const(textures),
    @Const(rgb2spec_scale), @Const(rgb2spec_coeffs), @Const(rgb2spec_res::Int32),
    @Const(max_queued::Int32)
)
    idx = @index(Global)

    rgb2spec_table = RGBToSpectrumTable(rgb2spec_res, rgb2spec_scale, rgb2spec_coeffs)

    @inbounds if idx <= max_queued
        current_size = hit_size[1]
        if idx <= current_size
            work = hit_items[idx]

            wo = -work.ray.d

            # Resolve MixMaterial to get the actual material index
            # Following pbrt-v4: MixMaterial is resolved at intersection time
            # using stochastic selection based on the amount texture and a hash
            material_idx = resolve_mix_material(
                materials, textures, work.material_idx,
                work.pi, wo, work.uv
            )

            # Check if surface is emissive
            if is_emissive_dispatch(materials, material_idx)
                # Get emission
                Le = get_emission_spectral_dispatch(
                    rgb2spec_table, materials, textures, material_idx,
                    wo, work.n, work.uv, work.lambda
                )

                if !is_black(Le)
                    # Apply path throughput
                    contribution = work.beta * Le

                    # MIS weight: on first bounce or specular, no MIS
                    final_contrib = if work.depth == Int32(0) || work.specular_bounce
                        contribution / average(work.r_u)
                    else
                        # Full MIS weight
                        contribution / average(work.r_u + work.r_l)
                    end

                    # Add to pixel
                    pixel_idx = work.pixel_index
                    base_idx = (pixel_idx - Int32(1)) * Int32(4)

                    Atomix.@atomic pixel_L[base_idx + Int32(1)] += final_contrib[1]
                    Atomix.@atomic pixel_L[base_idx + Int32(2)] += final_contrib[2]
                    Atomix.@atomic pixel_L[base_idx + Int32(3)] += final_contrib[3]
                    Atomix.@atomic pixel_L[base_idx + Int32(4)] += final_contrib[4]
                end
            end

            # Create material evaluation work item (for non-emissive or mixed materials)
            # Skip pure emissive materials (using resolved material_idx)
            if !is_pure_emissive_dispatch(materials, material_idx)
                mat_work = VPMaterialEvalWorkItem(
                    work.pi,
                    work.n,
                    work.ns,
                    wo,
                    work.uv,
                    material_idx,  # Use resolved material index (MixMaterial already resolved)
                    work.lambda,
                    work.pixel_index,
                    work.beta,
                    work.r_u,
                    work.r_l,
                    work.depth,
                    work.eta_scale,
                    work.specular_bounce,
                    work.any_non_specular_bounces,
                    work.prev_intr_p,
                    work.prev_intr_n,
                    work.current_medium
                )

                new_idx = @atomic material_size[1] += Int32(1)
                if new_idx <= length(material_items)
                    material_items[new_idx] = mat_work
                end
            end
        end
    end
end

"""
    vp_process_surface_hits!(backend, state, materials, textures)

Process all surface hits.
"""
function vp_process_surface_hits!(
    backend,
    state::VolPathState,
    materials,
    textures
)
    n = queue_size(state.hit_surface_queue)
    n == 0 && return nothing

    kernel! = vp_process_surface_hits_kernel!(backend)
    kernel!(
        state.material_queue.items, state.material_queue.size,
        state.pixel_L,
        state.hit_surface_queue.items, state.hit_surface_queue.size,
        materials,
        textures,
        state.rgb2spec_table.scale, state.rgb2spec_table.coeffs, state.rgb2spec_table.res,
        state.material_queue.capacity;
        ndrange=Int(state.hit_surface_queue.capacity)  # Fixed ndrange to avoid OpenCL recompilation
    )

    KernelAbstractions.synchronize(backend)
    return nothing
end

# ============================================================================
# Direct Lighting Inner Function
# ============================================================================

"""Inner function for surface direct lighting - can use return statements.

Uses power-weighted light sampling via alias table for better importance sampling
in scenes with lights of varying intensities (pbrt-v4's PowerLightSampler approach).
"""
@propagate_inbounds function surface_direct_lighting_inner!(
    shadow_items, shadow_size,
    work::VPMaterialEvalWorkItem,
    materials,
    textures,
    lights,
    rgb2spec_table,
    light_sampler_p,
    light_sampler_q,
    light_sampler_alias,
    num_lights::Int32
)
    # Skip if no lights
    num_lights < Int32(1) && return

    # Random numbers for light sampling
    u_light = rand(Point2f)
    light_select = rand(Float32)

    # Select light using power-weighted alias table sampling
    # Returns (1-based index, PMF for that light)
    light_idx, light_pmf = sample_light_sampler(
        light_sampler_p, light_sampler_q, light_sampler_alias, light_select
    )

    # Validate index (should always be valid if sampler was built correctly)
    if light_idx < Int32(1) || light_idx > num_lights || light_pmf <= 0f0
        return
    end

    # Sample the light
    light_sample = sample_light_from_tuple(
        rgb2spec_table, lights, light_idx, work.pi, work.lambda, u_light
    )

    if light_sample.pdf > 0f0 && !is_black(light_sample.Li)
        # Evaluate BSDF for light direction
        bsdf_f, bsdf_pdf = evaluate_spectral_material(
            rgb2spec_table, materials, textures, work.material_idx,
            work.wo, light_sample.wi, work.ns, work.uv, work.lambda
        )

        if !is_black(bsdf_f)
            # Compute direct lighting contribution with MIS
            result = compute_direct_lighting_spectral(
                work.pi, work.ns, work.wo, work.beta, work.r_u, work.lambda,
                light_sample, bsdf_f, bsdf_pdf
            )

            if result.valid
                # Following pbrt-v4 (surfscatter.cpp lines 299-315):
                # - Ld = beta * f * Li (no PDF division - that happens at shadow ray resolution)
                # - r_l = r_u * lightPDF where lightPDF = ls.pdf * light_pmf
                # compute_direct_lighting_spectral already sets r_l = r_u * ls.pdf
                # So we multiply by light_pmf to get the full light PDF in r_l
                scaled_r_l = result.r_l * light_pmf

                # Create shadow ray
                shadow_ray = Raycore.Ray(
                    o = result.ray_origin,
                    d = result.ray_direction,
                    t_max = result.t_max
                )

                shadow_item = VPShadowRayWorkItem(
                    shadow_ray,
                    result.t_max,
                    work.lambda,
                    result.Ld,  # NOT divided by light_pmf - MIS handles this
                    result.r_u,
                    scaled_r_l,
                    work.pixel_index,
                    work.current_medium  # Shadow ray starts in same medium
                )

                @inbounds begin
                    new_idx = @atomic shadow_size[1] += Int32(1)
                    if new_idx <= length(shadow_items)
                        shadow_items[new_idx] = shadow_item
                    end
                end
            end
        end
    end
    return
end

# ============================================================================
# Direct Lighting at Surface Hits
# ============================================================================

"""
    vp_sample_surface_direct_lighting_kernel!(...)

Sample direct lighting at surface hits using power-weighted light sampling.
Creates shadow rays for unoccluded light contributions.
"""
@kernel inbounds=true function vp_sample_surface_direct_lighting_kernel!(
    # Output
    shadow_items, shadow_size,
    # Input
    @Const(material_items), @Const(material_size),
    @Const(materials),
    @Const(textures),
    @Const(lights),
    @Const(rgb2spec_scale), @Const(rgb2spec_coeffs), @Const(rgb2spec_res::Int32),
    @Const(light_sampler_p), @Const(light_sampler_q), @Const(light_sampler_alias),
    @Const(num_lights::Int32), @Const(max_queued::Int32)
)
    idx = @index(Global)

    rgb2spec_table = RGBToSpectrumTable(rgb2spec_res, rgb2spec_scale, rgb2spec_coeffs)

    @inbounds if idx <= max_queued
        current_size = material_size[1]
        if idx <= current_size
            work = material_items[idx]
            surface_direct_lighting_inner!(
                shadow_items, shadow_size,
                work, materials, textures, lights, rgb2spec_table,
                light_sampler_p, light_sampler_q, light_sampler_alias,
                num_lights
            )
        end
    end
end

"""
    vp_sample_surface_direct_lighting!(backend, state, materials, textures, lights)

Sample direct lighting at all surface material evaluation points.
Uses power-weighted light sampling from state.light_sampler_* arrays.
"""
function vp_sample_surface_direct_lighting!(
    backend,
    state::VolPathState,
    materials,
    textures,
    lights
)
    n = queue_size(state.material_queue)
    n == 0 && return nothing

    kernel! = vp_sample_surface_direct_lighting_kernel!(backend)
    kernel!(
        state.shadow_queue.items, state.shadow_queue.size,
        state.material_queue.items, state.material_queue.size,
        materials,
        textures,
        lights,
        state.rgb2spec_table.scale, state.rgb2spec_table.coeffs, state.rgb2spec_table.res,
        state.light_sampler_p, state.light_sampler_q, state.light_sampler_alias,
        state.num_lights, state.shadow_queue.capacity;
        ndrange=Int(state.material_queue.capacity)  # Fixed ndrange to avoid OpenCL recompilation
    )

    KernelAbstractions.synchronize(backend)
    return nothing
end

# ============================================================================
# BSDF Sampling Inner Function
# ============================================================================

"""Inner function for material evaluation - can use return statements."""
@propagate_inbounds function evaluate_material_inner!(
    next_ray_items, next_ray_size,
    work::VPMaterialEvalWorkItem,
    materials,
    textures,
    rgb2spec_table,
    max_depth::Int32,
    max_queued::Int32,
    do_regularize::Bool
)
    # Check depth limit
    new_depth = work.depth + Int32(1)
    if new_depth >= max_depth
        return
    end

    # Random numbers for BSDF sampling
    u = rand(Point2f)
    rng = rand(Float32)
    rr_sample = rand(Float32)

    # Apply regularization if enabled and we've had a non-specular bounce
    # (pbrt-v4: regularize && anyNonSpecularBounces)
    regularize = do_regularize && work.any_non_specular_bounces

    # Sample BSDF
    sample = sample_spectral_material(
        rgb2spec_table, materials, textures, work.material_idx,
        work.wo, work.ns, work.uv, work.lambda, u, rng, regularize
    )

    # Check if valid sample
    if sample.pdf > 0f0 && !is_black(sample.f)
        # Compute new throughput
        cos_theta = abs(dot(sample.wi, work.ns))
        new_beta = if sample.is_specular
            work.beta * sample.f
        else
            work.beta * sample.f * cos_theta / sample.pdf
        end

        # Update eta scale for refraction
        new_eta_scale = work.eta_scale * sample.eta_scale

        # Update MIS weights
        new_r_l = if sample.is_specular
            work.r_u  # No MIS for specular
        else
            work.r_u / sample.pdf
        end

        # Russian roulette
        should_continue, final_beta = russian_roulette_spectral(
            new_beta, new_depth, rr_sample
        )

        if should_continue
            # Determine medium for continuation ray
            # Following pbrt-v4: use ray direction relative to surface normal
            # to determine which medium the ray enters
            new_medium = if has_medium_interface_dispatch(materials, work.material_idx)
                # Material has MediumInterface - get medium based on ray direction
                # If wi · n > 0, ray goes "outside" the surface
                # If wi · n < 0, ray goes "inside" the surface
                get_medium_index_for_direction_dispatch(materials, work.material_idx, sample.wi, work.n)
            else
                # No MediumInterface on this material
                # Stay in current medium (reflection or transmission through regular material)
                work.current_medium
            end

            # Create continuation ray
            # Offset origin slightly to avoid self-intersection
            offset_dir = if dot(sample.wi, work.n) > 0f0
                work.n
            else
                -work.n
            end
            ray_origin = Point3f(work.pi + offset_dir * 0.0001f0)

            new_ray = Raycore.Ray(
                o = ray_origin,
                d = sample.wi,
                t_max = Inf32,
                time = 0f0
            )

            ray_item = VPRayWorkItem(
                new_ray,
                new_depth,
                work.lambda,
                work.pixel_index,
                final_beta,
                work.r_u,  # r_u unchanged
                new_r_l,
                work.pi,   # prev_intr_p
                work.ns,   # prev_intr_n
                new_eta_scale,
                sample.is_specular,
                work.any_non_specular_bounces || !sample.is_specular,
                new_medium
            )

            @inbounds begin
                new_idx = @atomic next_ray_size[1] += Int32(1)
                if new_idx <= max_queued
                    next_ray_items[new_idx] = ray_item
                end
            end
        end
    end
    return
end

# ============================================================================
# BSDF Sampling and Path Continuation
# ============================================================================

"""
    vp_evaluate_materials_kernel!(...)

Sample BSDF at surface hits to generate continuation rays.
Includes Russian roulette for path termination.
"""
@kernel inbounds=true function vp_evaluate_materials_kernel!(
    # Output
    next_ray_items, next_ray_size,
    # Input
    @Const(material_items), @Const(material_size),
    @Const(materials),
    @Const(textures),
    @Const(media),  # For determining medium after refraction
    @Const(rgb2spec_scale), @Const(rgb2spec_coeffs), @Const(rgb2spec_res::Int32),
    @Const(max_depth::Int32), @Const(max_queued::Int32),
    @Const(do_regularize::Bool)  # Whether to apply BSDF regularization
)
    idx = @index(Global)

    rgb2spec_table = RGBToSpectrumTable(rgb2spec_res, rgb2spec_scale, rgb2spec_coeffs)

    @inbounds if idx <= max_queued
        current_size = material_size[1]
        if idx <= current_size
            work = material_items[idx]
            evaluate_material_inner!(
                next_ray_items, next_ray_size,
                work, materials, textures, rgb2spec_table, max_depth, max_queued,
                do_regularize
            )
        end
    end
end

"""
    vp_evaluate_materials!(backend, state, materials, textures, media, regularize=true)

Evaluate materials and spawn continuation rays.

When `regularize=true`, near-specular BSDFs are roughened after the first non-specular
bounce to reduce fireflies (matches pbrt-v4's BSDF::Regularize).
"""
function vp_evaluate_materials!(
    backend,
    state::VolPathState,
    materials,
    textures,
    media,
    regularize::Bool = true
)
    n = queue_size(state.material_queue)
    n == 0 && return nothing

    output_queue = next_ray_queue(state)

    kernel! = vp_evaluate_materials_kernel!(backend)
    kernel!(
        output_queue.items, output_queue.size,
        state.material_queue.items, state.material_queue.size,
        materials,
        textures,
        media,
        state.rgb2spec_table.scale, state.rgb2spec_table.coeffs, state.rgb2spec_table.res,
        state.max_depth, output_queue.capacity, regularize;
        ndrange=Int(state.material_queue.capacity)  # Fixed ndrange to avoid OpenCL recompilation
    )

    KernelAbstractions.synchronize(backend)
    return nothing
end

# ============================================================================
# Helper: Check for pure emissive material
# ============================================================================

"""
    is_pure_emissive_dispatch(materials, mat_idx)

Check if material is purely emissive (no BSDF).
"""
@propagate_inbounds function is_pure_emissive_dispatch(materials, mat_idx::MaterialIndex)
    # Default: assume materials with emission also have BSDF
    # Override for EmissiveMaterial which has no BSDF
    return is_emissive_dispatch(materials, mat_idx) &&
           !has_bsdf_dispatch(materials, mat_idx)
end

"""
    has_bsdf_dispatch(materials, mat_idx)

Check if material has a BSDF component.
"""
@propagate_inbounds function has_bsdf_dispatch(materials, mat_idx::MaterialIndex)
    # Most materials have BSDF
    # EmissiveMaterial does not
    type_idx = mat_idx.material_type
    # This would need proper dispatch based on material type
    # For now, return true (most materials have BSDF)
    return true
end
