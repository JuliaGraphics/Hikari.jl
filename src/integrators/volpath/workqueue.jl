# Work queues for VolPath wavefront integrator
# Reuses PWWorkQueue pattern from PhysicalWavefront

using StructArrays

# ============================================================================
# SOA/AOS Array Allocation (following pbrt-v4's SOA pattern)
# ============================================================================

# Trait: should this type be decomposed into SOA?
# Only work item structs get SOA - their fields become separate contiguous arrays.
# Nested types (Ray, Vec3f, Spectrum, etc.) stay flat since they're small and
# accessed as units. The SOA benefit comes from coalesced access when threads
# read the same field across different work items.
function _should_soa(::Type{T}) where T
    T <: VPRayWorkItem && return true
    T <: VPMaterialEvalWorkItem && return true
    T <: VPShadowRayWorkItem && return true
    T <: VPHitSurfaceWorkItem && return true
    T <: VPMediumSampleWorkItem && return true
    T <: VPMediumScatterWorkItem && return true
    T <: VPEscapedRayWorkItem && return true
    return false
end

# Allocate array with AOS (soa=false) or SOA (soa=true) layout.
# Both support identical indexing: arr[i] returns T, arr[i] = val stores T.
function allocate_array(backend, ::Type{T}, n::Integer; soa::Bool=false) where T
    soa ? _allocate_soa(backend, T, n) : KernelAbstractions.allocate(backend, T, n)
end

function _allocate_soa(backend, ::Type{T}, n::Integer) where T
    if !_should_soa(T)
        return KernelAbstractions.allocate(backend, T, n)
    end
    if fieldcount(T) > 0
        fnames = fieldnames(T)
        ftypes = fieldtypes(T)
        components = NamedTuple{fnames}(
            ntuple(i -> _allocate_soa(backend, ftypes[i], n), length(fnames))
        )
        return StructArray{T}(components)
    end
    return KernelAbstractions.allocate(backend, T, n)
end

# ============================================================================
# Generic Work Queue (same as PhysicalWavefront)
# ============================================================================

"""
    VPWorkQueue{T}

Thread-safe work queue for wavefront rendering.
Uses atomic counter for GPU-safe push operations.
"""
struct VPWorkQueue{T, V<:AbstractVector{T}, S<:AbstractVector{Int32}}
    items::V
    size::S      # Single-element array for atomic operations
    capacity::Int32
end

function VPWorkQueue{T}(backend, capacity::Integer; soa::Bool=false) where T
    items = allocate_array(backend, T, capacity; soa=soa)
    size = KernelAbstractions.allocate(backend, Int32, 1)
    KernelAbstractions.fill!(size, Int32(0))
    VPWorkQueue{T, typeof(items), typeof(size)}(items, size, Int32(capacity))
end

"""Get current queue size (copies from GPU if needed)"""
@propagate_inbounds function queue_size(q::VPWorkQueue)
    s = Array(q.size)
    return Int(s[1])
end

"""Reset queue to empty"""
function reset_queue!(backend, q::VPWorkQueue)
    KernelAbstractions.fill!(q.size, Int32(0))
    KernelAbstractions.synchronize(backend)
end

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
    ray_queue_a::VPWorkQueue{VPRayWorkItem}
    ray_queue_b::VPWorkQueue{VPRayWorkItem}
    current_ray_queue::Symbol  # :a or :b

    # Medium sample queue (rays in medium with bounded t_max from intersection)
    medium_sample_queue::VPWorkQueue{VPMediumSampleWorkItem}

    # Medium scatter queue (real scattering events from delta tracking)
    medium_scatter_queue::VPWorkQueue{VPMediumScatterWorkItem}

    # Surface hit queue (rays that hit surfaces and are NOT in medium)
    hit_surface_queue::VPWorkQueue{VPHitSurfaceWorkItem}

    # Material evaluation queue
    material_queue::VPWorkQueue{VPMaterialEvalWorkItem}

    # Shadow ray queue
    shadow_queue::VPWorkQueue{VPShadowRayWorkItem}

    # Escaped ray queue
    escaped_queue::VPWorkQueue{VPEscapedRayWorkItem}

    # Film buffer (spectral radiance per pixel, 4 wavelengths)
    pixel_L::AbstractVector{Float32}

    # Accumulators for progressive rendering (moved from main loop)
    pixel_rgb::AbstractVector{Float32}  # n_pixels * 3 (RGB accumulator)
    wavelengths_per_pixel::AbstractVector{Float32}  # n_pixels * 4 (wavelength samples)
    pdf_per_pixel::AbstractVector{Float32}  # n_pixels * 4 (wavelength PDFs)

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

    # Multi-material queue for :per_type coherence mode
    # Stored as Any to allow different N values (number of material types)
    multi_material_queue::Any  # MultiMaterialQueue{N} or nothing
end

function VolPathState(
    backend,
    width::Integer,
    height::Integer,
    lights::Tuple;
    max_depth::Integer = 8,
    queue_capacity::Integer = width * height
)
    n_pixels = width * height

    # Create work queues
    ray_queue_a = VPWorkQueue{VPRayWorkItem}(backend, queue_capacity)
    ray_queue_b = VPWorkQueue{VPRayWorkItem}(backend, queue_capacity)
    medium_sample_queue = VPWorkQueue{VPMediumSampleWorkItem}(backend, queue_capacity)
    medium_scatter_queue = VPWorkQueue{VPMediumScatterWorkItem}(backend, queue_capacity)
    hit_surface_queue = VPWorkQueue{VPHitSurfaceWorkItem}(backend, queue_capacity)
    material_queue = VPWorkQueue{VPMaterialEvalWorkItem}(backend, queue_capacity)
    shadow_queue = VPWorkQueue{VPShadowRayWorkItem}(backend, queue_capacity)
    escaped_queue = VPWorkQueue{VPEscapedRayWorkItem}(backend, queue_capacity)

    # Film buffer (4 wavelengths per pixel)
    pixel_L = KernelAbstractions.allocate(backend, Float32, n_pixels * 4)
    KernelAbstractions.fill!(pixel_L, 0f0)

    # Accumulators for progressive rendering
    pixel_rgb = KernelAbstractions.allocate(backend, Float32, n_pixels * 3)
    KernelAbstractions.fill!(pixel_rgb, 0f0)
    wavelengths_per_pixel = KernelAbstractions.allocate(backend, Float32, n_pixels * 4)
    pdf_per_pixel = KernelAbstractions.allocate(backend, Float32, n_pixels * 4)

    # Load RGB to spectrum table and convert to appropriate array type
    rgb2spec_table_cpu = get_srgb_table()
    # Determine array type from an allocated array
    ArrayType = backend isa KernelAbstractions.CPU ? Array : typeof(pixel_L).name.wrapper
    rgb2spec_table = to_gpu(ArrayType, rgb2spec_table_cpu)

    # Load CIE XYZ table and convert to GPU
    cie_table_cpu = CIEXYZTable()
    cie_table = to_gpu(ArrayType, cie_table_cpu)

    # Build power-weighted light sampler
    n_lights = length(lights)
    if n_lights > 0
        sampler = PowerLightSampler(lights)
        sampler_data = LightSamplerData(sampler)
        light_sampler_p = ArrayType(sampler_data.p)
        light_sampler_q = ArrayType(sampler_data.q)
        light_sampler_alias = ArrayType(sampler_data.alias)
    else
        light_sampler_p = ArrayType{Float32}(undef, 0)
        light_sampler_q = ArrayType{Float32}(undef, 0)
        light_sampler_alias = ArrayType{Int32}(undef, 0)
    end

    VolPathState(
        backend,
        ray_queue_a,
        ray_queue_b,
        :a,
        medium_sample_queue,
        medium_scatter_queue,
        hit_surface_queue,
        material_queue,
        shadow_queue,
        escaped_queue,
        pixel_L,
        pixel_rgb,
        wavelengths_per_pixel,
        pdf_per_pixel,
        rgb2spec_table,
        cie_table,
        light_sampler_p,
        light_sampler_q,
        light_sampler_alias,
        Int32(n_lights),
        Int32(max_depth),
        Int32(width),
        Int32(height),
        nothing  # multi_material_queue - lazily initialized
    )
end

"""Get current input ray queue"""
@propagate_inbounds function current_ray_queue(state::VolPathState)
    state.current_ray_queue == :a ? state.ray_queue_a : state.ray_queue_b
end

"""Get next output ray queue"""
@propagate_inbounds function next_ray_queue(state::VolPathState)
    state.current_ray_queue == :a ? state.ray_queue_b : state.ray_queue_a
end

"""Swap ray queue buffers"""
function swap_ray_queues!(state::VolPathState)
    state.current_ray_queue = state.current_ray_queue == :a ? :b : :a
end

"""Reset all queues for new iteration"""
function reset_iteration_queues!(state::VolPathState)
    backend = state.backend
    reset_queue!(backend, next_ray_queue(state))
    reset_queue!(backend, state.medium_sample_queue)
    reset_queue!(backend, state.medium_scatter_queue)
    reset_queue!(backend, state.hit_surface_queue)
    reset_queue!(backend, state.material_queue)
    reset_queue!(backend, state.shadow_queue)
    reset_queue!(backend, state.escaped_queue)
end

"""Reset film for new sample"""
function reset_film!(state::VolPathState)
    KernelAbstractions.fill!(state.pixel_L, 0f0)
    KernelAbstractions.synchronize(state.backend)
end
