# Multi-Material Evaluation - Material Type Coherent Processing
# Following pbrt-v4's MultiWorkQueue pattern for reduced GPU warp divergence
#
# This module provides material type coherent evaluation by:
# 1. Routing work items to per-type queues at surface hit processing time
# 2. Processing each type's queue with a specialized kernel
# 3. Each kernel only processes items of its type (no branching/skipping)

# ============================================================================
# Multi-Material Queue Container
# ============================================================================

"""
    MultiMaterialQueue{N}

Container holding N separate work queues, one per material type.
Items are routed to the correct queue based on MaterialIndex.material_type.

Following pbrt-v4's MultiWorkQueue pattern where:
- Push routes to correct sub-queue based on type
- Each sub-queue is processed with type-specialized kernel
"""
struct MultiMaterialQueue{N, Q}
    queues::NTuple{N, Q}  # N queues, one per material type
end

function MultiMaterialQueue{N}(backend, capacity::Integer) where {N}
    # Each per-type queue needs full capacity because we don't know the distribution
    # of material types in advance. A scene might have 90% of rays hitting one material.
    # This uses more memory but ensures correctness.
    queues = ntuple(N) do _
        WorkQueue{VPMaterialEvalWorkItem}(backend, capacity)
    end
    MultiMaterialQueue{N, typeof(queues[1])}(queues)
end

"""Get queue for specific material type (1-indexed)"""
@propagate_inbounds function get_queue(mmq::MultiMaterialQueue{N}, type_idx::Integer) where {N}
    mmq.queues[type_idx]
end

"""Reset all queues"""
function reset_queues!(backend, mmq::MultiMaterialQueue{N}) where {N}
    for q in mmq.queues
        empty!(q)
    end
end

"""Get total size across all queues"""
function total_size(mmq::MultiMaterialQueue{N}) where {N}
    sum(length(q) for q in mmq.queues)
end

# ============================================================================
# Surface Hit Processing with Per-Type Routing
# ============================================================================

"""
    vp_process_surface_hits_multi_kernel!(...)

Process surface hits and route to per-type material queues.
This is the key difference from standard processing - items go directly
to the queue matching their material type.
"""
@kernel inbounds=true function vp_process_surface_hits_multi_kernel!(
    # Output - per-type material queues (up to 16 types supported)
    mat_queue_1,
    mat_queue_2,
    mat_queue_3,
    mat_queue_4,
    mat_queue_5,
    mat_queue_6,
    mat_queue_7,
    mat_queue_8,
    mat_queue_9,
    mat_queue_10,
    mat_queue_11,
    mat_queue_12,
    mat_queue_13,
    mat_queue_14,
    mat_queue_15,
    mat_queue_16,
    pixel_L,
    # Input
    @Const(hit_queue),
    @Const(materials),
    @Const(textures),
    @Const(rgb2spec_table),
    @Const(num_types::Int32),
    @Const(max_queued::Int32)
)
    idx = @index(Global)

    @inbounds if idx <= max_queued
        current_size = hit_queue.size[1]
        if idx <= current_size
            work = hit_queue.items[idx]

            wo = -work.ray.d

            # Resolve MixMaterial
            material_idx = resolve_mix_material(
                materials, textures, work.material_idx,
                work.pi, wo, work.uv
            )

            # Check emission
            if is_emissive_dispatch(materials, material_idx)
                # Get emission - pass wo and n like the standard version
                Le = get_emission_spectral_dispatch(
                    rgb2spec_table, materials, textures, material_idx,
                    wo, work.n, work.uv, work.lambda
                )

                if !is_black(Le)
                    # Apply path throughput
                    contribution = work.beta * Le

                    # MIS weight: on first bounce or specular, no MIS
                    # Match the standard vp_process_surface_hits_kernel! logic exactly
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

            # Only create material work item if not pure emissive (has BSDF)
            if !is_pure_emissive_dispatch(materials, material_idx)
                # Create material eval work item
                mat_work = VPMaterialEvalWorkItem(
                work.pi,
                work.n,
                work.ns,
                wo,
                work.uv,
                material_idx,
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

            # Route to correct per-type queue based on material_type
            mat_type = material_idx.material_type
            if mat_type == UInt8(1) && num_types >= Int32(1)
                push!(mat_queue_1, mat_work)
            elseif mat_type == UInt8(2) && num_types >= Int32(2)
                push!(mat_queue_2, mat_work)
            elseif mat_type == UInt8(3) && num_types >= Int32(3)
                push!(mat_queue_3, mat_work)
            elseif mat_type == UInt8(4) && num_types >= Int32(4)
                push!(mat_queue_4, mat_work)
            elseif mat_type == UInt8(5) && num_types >= Int32(5)
                push!(mat_queue_5, mat_work)
            elseif mat_type == UInt8(6) && num_types >= Int32(6)
                push!(mat_queue_6, mat_work)
            elseif mat_type == UInt8(7) && num_types >= Int32(7)
                push!(mat_queue_7, mat_work)
            elseif mat_type == UInt8(8) && num_types >= Int32(8)
                push!(mat_queue_8, mat_work)
            elseif mat_type == UInt8(9) && num_types >= Int32(9)
                push!(mat_queue_9, mat_work)
            elseif mat_type == UInt8(10) && num_types >= Int32(10)
                push!(mat_queue_10, mat_work)
            elseif mat_type == UInt8(11) && num_types >= Int32(11)
                push!(mat_queue_11, mat_work)
            elseif mat_type == UInt8(12) && num_types >= Int32(12)
                push!(mat_queue_12, mat_work)
            elseif mat_type == UInt8(13) && num_types >= Int32(13)
                push!(mat_queue_13, mat_work)
            elseif mat_type == UInt8(14) && num_types >= Int32(14)
                push!(mat_queue_14, mat_work)
            elseif mat_type == UInt8(15) && num_types >= Int32(15)
                push!(mat_queue_15, mat_work)
            elseif mat_type == UInt8(16) && num_types >= Int32(16)
                push!(mat_queue_16, mat_work)
            end
            end  # end if !is_pure_emissive_dispatch
        end
    end
end

# ============================================================================
# Per-Type Material Evaluation Kernel
# ============================================================================

"""
Kernel that evaluates materials from a single-type queue.
All items in the queue have the same material type - no branching needed.
The mat_array parameter is the specific Vector for this material type.

Uses pre-computed Sobol samples from pixel_samples (pbrt-v4 RaySamples style).
"""
@kernel inbounds=true function vp_evaluate_single_type_kernel!(
    # Output
    next_ray_queue,
    # Input - queue for THIS material type only
    @Const(material_queue),
    @Const(mat_array),  # The specific material array for this type
    @Const(textures),
    @Const(rgb2spec_table),
    @Const(max_depth::Int32), @Const(max_queued::Int32),
    @Const(do_regularize::Bool),
    # Pre-computed Sobol samples (SOA layout)
    @Const(pixel_samples_indirect_uc), @Const(pixel_samples_indirect_u), @Const(pixel_samples_indirect_rr)
)
    idx = @index(Global)

    @inbounds if idx <= material_queue.size[1]
        work = material_queue.items[idx]

        # Direct lookup - all items have same type, so mat_array is correct
        mat = mat_array[work.material_idx.material_idx]

        _evaluate_typed_material!(
            next_ray_queue,
            work, mat, textures, rgb2spec_table,
            max_depth, do_regularize,
            pixel_samples_indirect_uc, pixel_samples_indirect_u, pixel_samples_indirect_rr
        )
    end
end

"""
Inner evaluation for a typed material (no dispatch needed).

Uses pre-computed Sobol samples from pixel_samples (pbrt-v4 RaySamples style).
"""
@propagate_inbounds function _evaluate_typed_material!(
    next_ray_queue,
    work::VPMaterialEvalWorkItem,
    mat,  # Concrete material type (known at compile time)
    textures,
    rgb2spec_table,
    max_depth::Int32,
    do_regularize::Bool,
    # Pre-computed Sobol samples (SOA layout)
    pixel_samples_indirect_uc,
    pixel_samples_indirect_u,
    pixel_samples_indirect_rr
)
    new_depth = work.depth + Int32(1)
    new_depth >= max_depth && return

    # Use pre-computed Sobol samples for this pixel (pbrt-v4 RaySamples style)
    pixel_idx = work.pixel_index
    u = pixel_samples_indirect_u[pixel_idx]
    rng = pixel_samples_indirect_uc[pixel_idx]
    rr_sample = pixel_samples_indirect_rr[pixel_idx]

    regularize = do_regularize && work.any_non_specular_bounces

    # Direct BSDF sample - no dispatch
    sample = sample_bsdf_spectral(
        rgb2spec_table, mat, textures,
        work.wo, work.ns, work.uv, work.lambda, u, rng, regularize
    )

    if sample.pdf > 0f0 && !is_black(sample.f)
        cos_theta = abs(dot(sample.wi, work.ns))
        new_beta = if sample.is_specular
            work.beta * sample.f
        else
            work.beta * sample.f * cos_theta / sample.pdf
        end

        new_eta_scale = work.eta_scale * sample.eta_scale

        new_r_l = if sample.is_specular
            work.r_u
        else
            work.r_u / sample.pdf
        end

        should_continue, final_beta = russian_roulette_spectral(
            new_beta, new_depth, rr_sample
        )

        if should_continue
            new_medium = _get_medium_for_type(mat, sample.wi, work.n, work.current_medium)

            offset_dir = dot(sample.wi, work.n) > 0f0 ? work.n : -work.n
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
                work.r_u,
                new_r_l,
                work.pi,
                work.ns,
                new_eta_scale,
                sample.is_specular,
                work.any_non_specular_bounces || !sample.is_specular,
                new_medium
            )

            push!(next_ray_queue, ray_item)
        end
    end
    return
end

# Medium helpers
_get_medium_for_type(mat, wi, n, current) = current
_get_medium_for_type(mat::MediumInterfaceIdx, wi, n, current) =
    dot(wi, n) > 0f0 ? mat.outside : mat.inside

"""
DEBUG helper: Copy items from multi_queue to standard material_queue for testing.
"""
function _copy_multi_to_standard!(backend, state::VolPathState, multi_queue::MultiMaterialQueue{N}) where {N}
    # Reset the standard material queue
    empty!(state.material_queue)

    # Copy items from all per-type queues to standard queue
    total = 0
    for type_idx in 1:N
        type_queue = multi_queue.queues[type_idx]
        n_type = length(type_queue)
        if n_type > 0
            # Copy items
            src_items = Array(type_queue.items)
            dst_items = Array(state.material_queue.items)
            for i in 1:n_type
                dst_items[total + i] = src_items[i]
            end
            copyto!(state.material_queue.items, dst_items)
            total += n_type
        end
    end

    # Set the size
    size_arr = Array(state.material_queue.size)
    size_arr[1] = Int32(total)
    copyto!(state.material_queue.size, size_arr)

    return nothing
end

# ============================================================================
# Coherent Processing Entry Points
# ============================================================================

"""
    vp_process_surface_hits_coherent!(backend, state, multi_queue, materials, textures)

Process surface hits with routing to per-type queues.
"""
function vp_process_surface_hits_coherent!(
    backend,
    state::VolPathState,
    multi_queue::MultiMaterialQueue{N},
    materials::NTuple{N, Any},
    textures
) where {N}
    n = length(state.hit_surface_queue)
    n == 0 && return nothing

    # Reset per-type queues
    reset_queues!(backend, multi_queue)

    # Pad queues to 16 for kernel (supports up to 16 material types)
    # Create a dummy queue for unused slots
    dummy_queue = WorkQueue{VPMaterialEvalWorkItem}(backend, 1)

    queues = ntuple(16) do i
        i <= N ? multi_queue.queues[i] : dummy_queue
    end

    kernel! = vp_process_surface_hits_multi_kernel!(backend)
    kernel!(
        queues[1], queues[2], queues[3], queues[4],
        queues[5], queues[6], queues[7], queues[8],
        queues[9], queues[10], queues[11], queues[12],
        queues[13], queues[14], queues[15], queues[16],
        state.pixel_L,
        state.hit_surface_queue,
        materials,
        textures,
        state.rgb2spec_table,
        Int32(N),
        state.hit_surface_queue.capacity;
        ndrange=Int(state.hit_surface_queue.capacity)
    )
    return nothing
end

"""
    vp_evaluate_materials_coherent!(backend, state, multi_queue, materials, textures, media, regularize)

Evaluate materials using per-type queues with type-specialized kernels.

Following pbrt-v4's approach: each material type gets its own kernel launch with
the concrete material array passed directly. Julia specializes the kernel on the
material array element type, eliminating dispatch overhead.

This is the 1:1 port of pbrt-v4's ForEachType(EvaluateMaterialCallback{...}, Material::Types())
pattern from surfscatter.cpp.

Uses pre-computed Sobol samples from pixel_samples (pbrt-v4 RaySamples style).
"""
function vp_evaluate_materials_coherent!(
    backend,
    state::VolPathState,
    multi_queue::MultiMaterialQueue{N},
    materials::NTuple{N, Any},
    textures,
    _media,  # Unused - media handling is in _evaluate_typed_material! via work.current_medium
    regularize::Bool = true
) where {N}
    output_queue = next_ray_queue(state)

    # Extract pixel_samples SOA components for kernel
    pixel_samples = state.pixel_samples

    # pbrt-v4 pattern: for each material type, launch specialized kernel
    # Julia specializes vp_evaluate_single_type_kernel! on mat_array's element type
    #
    # NOTE: We launch with capacity (not actual size) to avoid GPU→CPU sync.
    # pbrt-v4 does the same - launches maxQueueSize threads and checks size inside kernel.
    # This means N × capacity threads total, but idle threads early-exit.
    for type_idx in 1:N
        type_queue = multi_queue.queues[type_idx]
        mat_array = materials[type_idx]  # Concrete Vector{MatType}

        # Launch type-specialized kernel (Julia specializes on mat_array type)
        kernel! = vp_evaluate_single_type_kernel!(backend)
        kernel!(
            output_queue,
            type_queue,
            mat_array,
            textures,
            state.rgb2spec_table,
            state.max_depth, output_queue.capacity, regularize,
            pixel_samples.indirect_uc, pixel_samples.indirect_u, pixel_samples.indirect_rr;
            ndrange=Int(type_queue.capacity)  # pbrt-v4 ForAllQueued pattern
        )
    end
end

"""Copy items from multi_queue to standard material_queue."""
function _copy_multi_to_material_queue!(backend, state::VolPathState, multi_queue::MultiMaterialQueue{N}) where {N}
    empty!(state.material_queue)

    # Copy each type queue sequentially (simpler than fused kernel)
    for type_idx in 1:N
        type_queue = multi_queue.queues[type_idx]
        kernel! = _copy_queue_kernel!(backend)
        kernel!(
            state.material_queue,
            type_queue,
            state.material_queue.capacity;
            ndrange=Int(type_queue.capacity)
        )
    end
end

@kernel inbounds=true function _copy_queue_kernel!(
    dst_queue,
    @Const(src_queue),
    @Const(max_dst::Int32)
)
    idx = @index(Global)
    @inbounds if idx <= src_queue.size[1]
        work = src_queue.items[idx]
        push!(dst_queue, work)
    end
end

"""
    vp_sample_direct_lighting_coherent!(backend, state, multi_queue, materials, textures, lights)

Sample direct lighting using per-type material queues.
Iterates over items in multi_queue directly without copying.
"""
function vp_sample_direct_lighting_coherent!(
    backend,
    state::VolPathState,
    multi_queue::MultiMaterialQueue{N},
    materials::NTuple{N, Any},
    textures,
    lights
) where {N}
    # Process each per-type queue for direct lighting
    # This avoids the need to copy items before direct lighting
    for type_idx in 1:N
        type_queue = multi_queue.queues[type_idx]
        n_type = length(type_queue)
        if n_type > 0
            # Use the per-type queue's items directly for direct lighting sampling
            vp_sample_direct_lighting_from_queue!(backend, state, type_queue, materials, textures, lights)
        end
    end
end

"""Sample direct lighting from a specific material queue (helper for coherent mode).

Uses pre-computed Sobol samples from pixel_samples (pbrt-v4 RaySamples style).
"""
function vp_sample_direct_lighting_from_queue!(
    backend,
    state::VolPathState,
    mat_queue::WorkQueue{VPMaterialEvalWorkItem},
    materials,
    textures,
    lights
)
    n = length(mat_queue)
    n == 0 && return nothing

    # Extract pixel_samples SOA components for kernel
    pixel_samples = state.pixel_samples

    kernel! = vp_sample_surface_direct_lighting_kernel!(backend)
    kernel!(
        state.shadow_queue,
        mat_queue,
        materials,
        textures,
        lights,
        state.rgb2spec_table,
        state.light_sampler_p, state.light_sampler_q, state.light_sampler_alias,
        state.num_lights, state.shadow_queue.capacity,
        pixel_samples.direct_uc, pixel_samples.direct_u;
        ndrange=Int(mat_queue.capacity)
    )
    return nothing
end

# ============================================================================
# Legacy functions (kept for :sorted mode which doesn't need full rewrite)
# ============================================================================

"""
    vp_evaluate_materials_sorted!(backend, state, materials, textures, media, regularize)

Sort-based coherence - sorts items by type before processing.
Less efficient than per-type queues but doesn't require architecture changes.
"""
function vp_evaluate_materials_sorted!(
    backend,
    state::VolPathState,
    materials::NTuple{N, Any},
    textures,
    media,
    regularize::Bool = true
) where {N}
    n_total = length(state.material_queue)
    n_total == 0 && return nothing

    # For small queues, sorting overhead isn't worth it
    if n_total < 512 || N == 1
        vp_evaluate_materials!(backend, state, materials, textures, media, regularize)
        return nothing
    end

    # Count items per type
    type_counts = _count_material_types(backend, state.material_queue, N)

    # Check if sorting is beneficial
    max_count = maximum(type_counts)
    if max_count >= n_total * 0.9
        vp_evaluate_materials!(backend, state, materials, textures, media, regularize)
        return nothing
    end

    # Compute prefix sums
    type_offsets = cumsum([0; type_counts[1:end-1]])

    # Scatter to sorted order
    sorted_items = KernelAbstractions.allocate(backend, VPMaterialEvalWorkItem, n_total)
    _scatter_by_material_type!(backend, state.material_queue, sorted_items, type_offsets, N)

    # Evaluate sorted buffer
    output_queue = next_ray_queue(state)

    # Extract pixel_samples SOA components for kernel
    pixel_samples = state.pixel_samples

    kernel! = vp_evaluate_materials_sorted_kernel!(backend)
    kernel!(
        output_queue,
        sorted_items, Int32(n_total),
        materials,
        textures,
        media,
        state.rgb2spec_table,
        state.max_depth, regularize,
        pixel_samples.indirect_uc, pixel_samples.indirect_u, pixel_samples.indirect_rr;
        ndrange=Int(n_total)
    )
    return nothing
end

# Helper functions for sorted mode
function _count_material_types(backend, queue::WorkQueue{VPMaterialEvalWorkItem}, N::Int)
    counts_gpu = KernelAbstractions.allocate(backend, Int32, N)
    KernelAbstractions.fill!(counts_gpu, Int32(0))

    n = length(queue)
    if n > 0
        kernel! = _count_types_kernel!(backend)
        kernel!(counts_gpu, queue, Int32(N); ndrange=Int(queue.capacity))
    end

    return Int.(Array(counts_gpu))
end

@kernel inbounds=true function _count_types_kernel!(
    counts,
    @Const(queue),
    @Const(num_types::Int32)
)
    idx = @index(Global)

    @inbounds if idx <= queue.size[1]
        work = queue.items[idx]
        mat_type = Int32(work.material_idx.material_type)
        if mat_type >= Int32(1) && mat_type <= num_types
            Atomix.@atomic counts[mat_type] += Int32(1)
        end
    end
end

function _scatter_by_material_type!(
    backend,
    src_queue::WorkQueue{VPMaterialEvalWorkItem},
    dst_items::AbstractVector{VPMaterialEvalWorkItem},
    type_offsets::Vector{Int},
    N::Int
)
    offsets_gpu = KernelAbstractions.allocate(backend, Int32, N)
    copyto!(offsets_gpu, Int32.(type_offsets))

    counters_gpu = KernelAbstractions.allocate(backend, Int32, N)
    KernelAbstractions.fill!(counters_gpu, Int32(0))

    n = length(src_queue)
    if n > 0
        kernel! = _scatter_kernel!(backend)
        kernel!(
            dst_items, offsets_gpu, counters_gpu,
            src_queue,
            Int32(N);
            ndrange=Int(src_queue.capacity)
        )
    end
end

@kernel inbounds=true function _scatter_kernel!(
    dst_items,
    @Const(offsets), counters,
    @Const(src_queue),
    @Const(num_types::Int32)
)
    idx = @index(Global)

    @inbounds if idx <= src_queue.size[1]
        work = src_queue.items[idx]
        mat_type = Int32(work.material_idx.material_type)

        if mat_type >= Int32(1) && mat_type <= num_types
            local_idx = Atomix.@atomic counters[mat_type] += Int32(1)
            dst_idx = offsets[mat_type] + local_idx
            if dst_idx >= Int32(1) && dst_idx <= length(dst_items)
                dst_items[dst_idx] = work
            end
        end
    end
end

@kernel inbounds=true function vp_evaluate_materials_sorted_kernel!(
    next_ray_queue,
    @Const(material_items), @Const(material_count::Int32),
    @Const(materials),
    @Const(textures),
    @Const(media),
    @Const(rgb2spec_table),
    @Const(max_depth::Int32),
    @Const(do_regularize::Bool),
    # Pre-computed Sobol samples (SOA layout)
    @Const(pixel_samples_indirect_uc), @Const(pixel_samples_indirect_u), @Const(pixel_samples_indirect_rr)
)
    idx = @index(Global)

    @inbounds if idx <= material_count
        work = material_items[idx]
        evaluate_material_inner!(
            next_ray_queue,
            work, materials, textures, rgb2spec_table, max_depth,
            do_regularize,
            pixel_samples_indirect_uc, pixel_samples_indirect_u, pixel_samples_indirect_rr
        )
    end
end

# ============================================================================
# Removed: Old per_type implementation that scanned entire queue N times
# The new vp_evaluate_materials_coherent! with MultiMaterialQueue is correct
# ============================================================================
