abstract type Filter end

struct LanczosSincFilter <: Filter
    radius::Point2f
    τ::Float32
end

function (f::LanczosSincFilter)(p::Point2f)::Float32
    windowed_sinc(p[1], f.radius[1], f.τ) * windowed_sinc(p[2], f.radius[2], f.τ)
end

@propagate_inbounds function sinc(x::Float32)::Float32
    x = abs(x)
    x < 1f-5 && return 1f0
    x *= Float32(π)
    sin(x) / x
end

function windowed_sinc(x::Float32, r::Float32, τ::Float32)::Float32
    x = abs(x)
    x > r && return 0f0
    sinc(x) * sinc(x / τ)
end
