abstract type AbstractSampler end

mutable struct Sampler <: AbstractSampler
    samples_per_pixel::Int64
    current_pixel::Point2f
    current_pixel_sample_id::Int64

    samples_1d_array_sizes::Vector{Int32}
    samples_2d_array_sizes::Vector{Int32}

    sample_array_1d::Vector{Vector{Float32}}
    sample_array_2d::Vector{Vector{Point2f}}

    array_1d_offset::UInt64
    array_2d_offset::UInt64
end

function Sampler(samples_per_pixel::Integer)
    Sampler(
        samples_per_pixel, Point2f(-1), 1,
        Int32[], Int32[],
        Vector{Vector{Float32}}(undef, 0),
        Vector{Vector{Point2f}}(undef, 0),
        1, 1,
    )
end

@propagate_inbounds function get_camera_sample(sampler::AbstractSampler, p_raster::Point2f)
    p_film = p_raster .+ get_2d(sampler)
    time = get_1d(sampler)
    p_lens = get_2d(sampler)
    CameraSample(p_film, p_lens, time, 1.0f0)
end

# Filter-aware camera sample generation (pbrt-v4 compatible)
# Uses filter importance sampling to compute both position and weight
@propagate_inbounds function get_camera_sample(sampler::AbstractSampler, p_raster::Point2f, filter_params::GPUFilterParams)
    u = get_2d(sampler)
    fs = filter_sample(filter_params, u)
    # fs.p is offset from pixel center, fs.weight is the filter weight
    p_film = p_raster .+ Point2f(0.5f0) .+ fs.p
    time = get_1d(sampler)
    p_lens = get_2d(sampler)
    CameraSample(p_film, p_lens, time, fs.weight)
end

@propagate_inbounds round_count(sampler::AbstractSampler, n::Integer) = n

"""
Other samplers are required to explicitly call this,
in their respective implementations.
"""
function start_pixel(sampler::Sampler, p::Point2f)
    sampler.array_1d_offset = sampler.array_2d_offset = 1
    sampler.current_pixel_sample_id = 1
    sampler.current_pixel = p
end

function start_next_sample(sampler::Sampler)
    sampler.array_1d_offset = sampler.array_2d_offset = 1
    sampler.current_pixel_sample_id += 1
    sampler.current_pixel_sample_id < sampler.samples_per_pixel
end

function set_sample_number(sampler::Sampler, sample_num::Integer)
    sampler.array_1d_offset = sampler.array_2d_offset = 1
    sampler.current_pixel_sample_id = sample_num
    sampler.current_pixel_sample_id < sampler.samples_per_pixel
end

function request_1d_array(sampler::Sampler, n::Integer)
    push!(sampler.samples_1d_array_sizes, n)
    push!(sampler.sample_array_1d, Vector{Float32}(undef, n * sampler.samples_per_pixel))
end

function request_2d_array(sampler::Sampler, n::Integer)
    push!(sampler.samples_2d_array_sizes, n)
    push!(sampler.sample_array_2d, Vector{Point2f}(undef, n * sampler.samples_per_pixel))
end

function get_1d_array(sampler::Sampler, n::Integer)
    sampler.array_1d_offset == length(sampler.sample_array_1d) + 1 && return nothing
    arr = @view sampler.sample_array_1d[sampler.array_1d_offset][sampler.current_pixel_sample_id*n:end]
    sampler.array_1d_offset += 1
    arr
end

function get_2d_array(sampler::Sampler, n::Integer)
    sampler.array_2d_offset == length(sampler.sample_array_2d) + 1 && return nothing
    arr = @view sampler.sample_array_2d[sampler.array_2d_offset][sampler.current_pixel_sample_id*n:end]
    sampler.array_2d_offset += 1
    arr
end


mutable struct PixelSampler <: AbstractSampler
    sampler::Sampler
    samples_1d::Vector{Vector{Float32}}
    samples_2d::Vector{Vector{Point2f}}
    current_1d_dimension::Int64
    current_2d_dimension::Int64
end

function PixelSampler(samples_per_pixel::Integer, n_sampled_dimensions::Integer)
    samples_1d = Vector{Vector{Float32}}(undef, n_sampled_dimensions)
    samples_2d = Vector{Vector{Point2f}}(undef, n_sampled_dimensions)
    for i in 1:n_sampled_dimensions
        samples_1d[i] = Vector{Float32}(undef, samples_per_pixel)
        samples_2d[i] = Vector{Point2f}(undef, samples_per_pixel)
    end
    PixelSampler(Sampler(samples_per_pixel), samples_1d, samples_2d, 1, 1)
end

start_pixel(p::PixelSampler, point::Point2f) = start_pixel(p.sampler, point)

function start_next_sample(ps::PixelSampler)
    ps.current_1d_dimension = ps.current_2d_dimension = 1
    start_next_sample(ps.sampler)
end

function set_sample_number(ps::PixelSampler, sample_num::Integer)::Bool
    ps.current_1d_dimension = ps.current_2d_dimension = 1
    set_sample_number(ps.sampler, sample_num)
end

function get_1d(ps::PixelSampler)
    ps.current_1d_dimension > length(ps.samples_1d) && return rand()
    v = ps.samples_1d[ps.current_1d_dimension][ps.sampler.current_pixel_sample_id]
    ps.current_1d_dimension += 1
    v
end

function get_2d(ps::PixelSampler)::Point2f
    ps.current_2d_dimension > length(ps.samples_2d) && return rand(Point2f)
    v = ps.samples_2d[ps.current_2d_dimension][ps.sampler.current_pixel_sample_id]
    ps.current_2d_dimension += 1
    v
end


struct UniformSampler <: AbstractSampler
    current_sample::Int64
    samples_per_pixel::Int64
    UniformSampler(samples_per_pixel::Integer) = new(1, samples_per_pixel)
end

# Simple GPU-friendly hash-based RNG for camera ray dithering
# Uses Wang hash for good distribution
@propagate_inbounds function wang_hash(seed::UInt32)::UInt32
    seed = (seed ⊻ UInt32(61)) ⊻ (seed >> UInt32(16))
    seed = seed * UInt32(9)
    seed = seed ⊻ (seed >> UInt32(4))
    seed = seed * UInt32(0x27d4eb2d)
    seed = seed ⊻ (seed >> UInt32(15))
    seed
end

@propagate_inbounds function gpu_rand_float(x::UInt32, y::UInt32, offset::UInt32)::Float32
    hash_val = wang_hash(x ⊻ wang_hash(y ⊻ wang_hash(offset)))
    Float32(hash_val) / Float32(typemax(UInt32))
end

@propagate_inbounds function get_camera_sample(::UniformSampler, p_raster::Point2f)
    # Use rand() for jittering - simpler and works correctly with any resolution
    p_film = p_raster .+ rand(Point2f)
    p_lens = rand(Point2f)
    time = rand(Float32)
    CameraSample(p_film, p_lens, time, 1.0f0)
end

# Filter-aware camera sample generation for UniformSampler
@propagate_inbounds function get_camera_sample(::UniformSampler, p_raster::Point2f, filter_params::GPUFilterParams)
    u = rand(Point2f)
    fs = filter_sample(filter_params, u)
    p_film = p_raster .+ Point2f(0.5f0) .+ fs.p
    p_lens = rand(Point2f)
    time = rand(Float32)
    CameraSample(p_film, p_lens, time, fs.weight)
end

@propagate_inbounds function has_next_sample(u::UniformSampler)::Bool
    u.current_sample ≤ u.samples_per_pixel
end
@propagate_inbounds function start_next_sample!(u::UniformSampler)
    u.current_sample += 1
end
@propagate_inbounds function start_pixel!(u::UniformSampler, ::Point2f)
    u.current_sample = 1
end
# Use rand() for proper Monte Carlo sampling
# The deterministic hash-based approach was causing all calls to return
# the same values, breaking light sampling in scenes with multiple lights
@propagate_inbounds function get_1d(::UniformSampler)::Float32
    rand(Float32)
end

@propagate_inbounds function get_2d(::UniformSampler)::Point2f
    rand(Point2f)
end

# include("stratified.jl")
