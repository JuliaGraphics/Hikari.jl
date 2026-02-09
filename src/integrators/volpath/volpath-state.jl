# VolPath state container
# Uses the unified WorkQueue from integrators/workqueue.jl

import KernelAbstractions as KA

# ============================================================================
# SOA Override for VolPath Work Items
# ============================================================================

# Override should_use_soa for VolPath work items that benefit from SOA layout
should_use_soa(::Type{VPRayWorkItem}) = true
should_use_soa(::Type{VPMaterialEvalWorkItem}) = true
should_use_soa(::Type{VPShadowRayWorkItem}) = true
should_use_soa(::Type{VPHitSurfaceWorkItem}) = true
should_use_soa(::Type{VPMediumSampleWorkItem}) = true
should_use_soa(::Type{VPMediumScatterWorkItem}) = true
should_use_soa(::Type{VPEscapedRayWorkItem}) = true
should_use_soa(::Type{VPRaySamples}) = true

# ============================================================================
# VolPath State Container
# ============================================================================

"""
    VolPathState

Contains all work queues and buffers for VolPath wavefront rendering.
"""
mutable struct VolPathState{Backend}
    backend::Backend

    # Ray queues (double-buffered for iteration)
    ray_queue_a::WorkQueue{VPRayWorkItem}
    ray_queue_b::WorkQueue{VPRayWorkItem}
    current_ray_queue::Symbol  # :a or :b

    # Medium sample queue (rays in medium with bounded t_max from intersection)
    medium_sample_queue::WorkQueue{VPMediumSampleWorkItem}

    # Medium scatter queue (real scattering events from delta tracking)
    medium_scatter_queue::WorkQueue{VPMediumScatterWorkItem}

    # Surface hit queue (rays that hit surfaces and are NOT in medium)
    hit_surface_queue::WorkQueue{VPHitSurfaceWorkItem}

    # Material evaluation queue
    material_queue::WorkQueue{VPMaterialEvalWorkItem}

    # Shadow ray queue
    shadow_queue::WorkQueue{VPShadowRayWorkItem}

    # Escaped ray queue
    escaped_queue::WorkQueue{VPEscapedRayWorkItem}

    # Film buffer (spectral radiance per pixel, 4 wavelengths)
    pixel_L::AbstractVector{Float32}

    # Accumulators for progressive rendering (moved from main loop)
    # NOTE: Default Float64 matches pbrt-v4's double precision accumulators (film.h:304-305)
    # Use Float32 for OpenCL backends that don't support double precision atomics
    pixel_rgb::AbstractVector  # n_pixels * 3 (RGB accumulator)
    pixel_weight_sum::AbstractVector  # n_pixels (filter weight accumulator)
    wavelengths_per_pixel::AbstractVector{Float32}  # n_pixels * 4 (wavelength samples)
    pdf_per_pixel::AbstractVector{Float32}  # n_pixels * 4 (wavelength PDFs)
    filter_weight_per_pixel::AbstractVector{Float32}  # n_pixels (filter weight per sample)

    # Pre-computed Sobol samples per pixel (pbrt-v4 RaySamples / PixelSampleState)
    # Updated each bounce via vp_generate_ray_samples_kernel!
    pixel_samples::Any  # StructArray{VPRaySamples} - SOA layout for GPU coalescing

    # RGB to spectrum table
    rgb2spec_table::RGBToSpectrumTable

    # CIE XYZ color matching table
    cie_table::CIEXYZTable

    # Light sampler data for power-weighted light selection
    # Stored as flat arrays for GPU compatibility
    light_sampler_p::AbstractVector{Float32}    # PMF values
    light_sampler_q::AbstractVector{Float32}    # Alias thresholds
    light_sampler_alias::AbstractVector{Int32}  # Alias indices
    num_lights::Int32

    # Render parameters
    max_depth::Int32
    width::Int32
    height::Int32

    # Sobol RNG for low-discrepancy sampling (allocated once, reused across frames)
    sobol_rng::Any  # SobolRNG or nothing

    # Multi-material queue for :per_type coherence mode
    # Stored as Any to allow different N values (number of material types)
    multi_material_queue::Any  # MultiMaterialQueue{N} or nothing
end

function VolPathState(
    backend,
    width::Integer,
    height::Integer,
    lights::Raycore.MultiTypeSet;  # MultiTypeSet of lights (has backend for GPU allocation)
    max_depth::Integer = 8,
    queue_capacity::Integer = width * height,
    scene_radius::Float32 = 10f0,  # Scene bounding sphere radius for light power estimation
    samples_per_pixel::Integer = 1,  # For SobolRNG parameter computation
    sampler_seed::UInt32 = UInt32(0),  # Scrambling seed for Sobol
    accumulation_eltype::DataType = Float32  # Element type for accumulators (Float32 for OpenCL)
)
    n_pixels = width * height

    # Create work queues
    ray_queue_a = WorkQueue{VPRayWorkItem}(backend, queue_capacity)
    ray_queue_b = WorkQueue{VPRayWorkItem}(backend, queue_capacity)
    medium_sample_queue = WorkQueue{VPMediumSampleWorkItem}(backend, queue_capacity)
    medium_scatter_queue = WorkQueue{VPMediumScatterWorkItem}(backend, queue_capacity)
    hit_surface_queue = WorkQueue{VPHitSurfaceWorkItem}(backend, queue_capacity)
    material_queue = WorkQueue{VPMaterialEvalWorkItem}(backend, queue_capacity)
    shadow_queue = WorkQueue{VPShadowRayWorkItem}(backend, queue_capacity)
    escaped_queue = WorkQueue{VPEscapedRayWorkItem}(backend, queue_capacity)

    # Film buffer (4 wavelengths per pixel)
    pixel_L = KA.allocate(backend, Float32, n_pixels * 4)
    KA.fill!(pixel_L, 0f0)

    # Accumulators for progressive rendering (configurable eltype for OpenCL compatibility)
    pixel_rgb = KA.allocate(backend, accumulation_eltype, n_pixels * 3)
    KA.fill!(pixel_rgb, zero(accumulation_eltype))
    pixel_weight_sum = KA.allocate(backend, accumulation_eltype, n_pixels)
    KA.fill!(pixel_weight_sum, zero(accumulation_eltype))
    wavelengths_per_pixel = KA.allocate(backend, Float32, n_pixels * 4)
    KA.fill!(wavelengths_per_pixel, 0f0)
    pdf_per_pixel = KA.allocate(backend, Float32, n_pixels * 4)
    KA.fill!(pdf_per_pixel, 0f0)
    filter_weight_per_pixel = KA.allocate(backend, Float32, n_pixels)
    KA.fill!(filter_weight_per_pixel, 0f0)

    # Pre-computed samples per pixel (SOA layout)
    pixel_samples = allocate_array(backend, VPRaySamples, n_pixels; soa=true)

    # Load lookup tables to GPU
    rgb2spec_table = to_gpu(backend, get_srgb_table())
    cie_table = to_gpu(backend, CIEXYZTable())

    # Build light sampler (alias table for power-weighted sampling)
    # PowerLightSampler computes powers on GPU and returns GPU arrays
    n_lights = length(lights)
    if n_lights > 0
        sampler = PowerLightSampler(lights; scene_radius=scene_radius)
        # Alias table already has GPU arrays from PowerLightSampler construction
        light_sampler_p = sampler.alias_table.p
        light_sampler_q = sampler.alias_table.q
        light_sampler_alias = sampler.alias_table.alias
    else
        light_sampler_p = KA.allocate(backend, Float32, 1)
        light_sampler_q = KA.allocate(backend, Float32, 1)
        light_sampler_alias = KA.allocate(backend, Int32, 1)
    end

    # Create SobolRNG (allocated once, reused across frames)
    sobol_rng = SobolRNG(backend, sampler_seed, width, height, samples_per_pixel)

    return VolPathState(
        backend,
        ray_queue_a, ray_queue_b, :a,
        medium_sample_queue, medium_scatter_queue,
        hit_surface_queue, material_queue, shadow_queue, escaped_queue,
        pixel_L, pixel_rgb, pixel_weight_sum,
        wavelengths_per_pixel, pdf_per_pixel, filter_weight_per_pixel,
        pixel_samples,
        rgb2spec_table, cie_table,
        light_sampler_p, light_sampler_q, light_sampler_alias, Int32(n_lights),
        Int32(max_depth), Int32(width), Int32(height),
        sobol_rng,
        nothing  # multi_material_queue
    )
end

# ============================================================================
# State Helpers
# ============================================================================

"""Get the current ray queue based on the queue selector."""
function current_ray_queue(state::VolPathState)
    state.current_ray_queue == :a ? state.ray_queue_a : state.ray_queue_b
end

"""Get the next ray queue (the one not currently active)."""
function next_ray_queue(state::VolPathState)
    state.current_ray_queue == :a ? state.ray_queue_b : state.ray_queue_a
end

"""Swap the ray queue selector."""
function swap_ray_queues!(state::VolPathState)
    state.current_ray_queue = state.current_ray_queue == :a ? :b : :a
end

"""Reset all processing queues for a new bounce."""
function reset_processing_queues!(state::VolPathState)
    empty!(state.medium_sample_queue)
    empty!(state.medium_scatter_queue)
    empty!(state.hit_surface_queue)
    empty!(state.material_queue)
    empty!(state.shadow_queue)
    empty!(state.escaped_queue)
    empty!(next_ray_queue(state))
end

"""Reset iteration queues before ray tracing."""
function reset_iteration_queues!(state::VolPathState)
    empty!(next_ray_queue(state))
    empty!(state.medium_sample_queue)
    empty!(state.medium_scatter_queue)
    empty!(state.hit_surface_queue)
    empty!(state.material_queue)
    empty!(state.shadow_queue)
    empty!(state.escaped_queue)
end

"""Reset film buffer for a new sample."""
function reset_film!(state::VolPathState)
    KA.fill!(state.pixel_L, 0f0)
end

# ============================================================================
# Resource Cleanup
# ============================================================================

"""
    cleanup!(state::VolPathState)

Release GPU memory held by the VolPath state.
"""
function cleanup!(state::VolPathState)
    # Cleanup work queues
    cleanup!(state.ray_queue_a)
    cleanup!(state.ray_queue_b)
    cleanup!(state.medium_sample_queue)
    cleanup!(state.medium_scatter_queue)
    cleanup!(state.hit_surface_queue)
    cleanup!(state.material_queue)
    cleanup!(state.shadow_queue)
    cleanup!(state.escaped_queue)

    # Cleanup film buffers
    finalize(state.pixel_L)
    finalize(state.pixel_rgb)
    finalize(state.pixel_weight_sum)
    finalize(state.wavelengths_per_pixel)
    finalize(state.pdf_per_pixel)
    finalize(state.filter_weight_per_pixel)
    finalize(state.pixel_samples)

    # Cleanup lookup tables
    finalize(state.rgb2spec_table)
    finalize(state.cie_table)

    # Cleanup light sampler
    finalize(state.light_sampler_p)
    finalize(state.light_sampler_q)
    finalize(state.light_sampler_alias)

    # Cleanup SobolRNG
    if state.sobol_rng !== nothing
        cleanup!(state.sobol_rng)
    end

    return nothing
end
