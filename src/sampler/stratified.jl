struct StratifiedSampler <: AbstractSampler
    pixel_sampler::PixelSampler
    pixel_samlpes::Tuple{UInt32,UInt32}
    jitter_samples::Bool
end

function StratifiedSampler(
    pixel_samples::Tuple{Integer,Integer}, jitter_samples::Bool,
    n_sampled_dimensions::Int64,
)
    StratifiedSampler(
        PixelSampler(pixel_samples[1] * pixel_samples[2], n_sampled_dimensions),
        pixel_samples, jitter_samples,
    )
end

function start_pixel(ss::StratifiedSampler, p::Point2f) end
