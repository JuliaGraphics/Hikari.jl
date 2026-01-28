const TextureType = Union{Float32,S} where S<:Spectrum

# Texture wraps actual texture data arrays. For constant values, use raw values directly
# in material fields (materials should have loose type parameters).
struct Texture{ElType, N, T<:AbstractArray{ElType, N}}
    data::T
    constval::ElType
    isconst::Bool
end
Texture(data::AbstractArray{T,N}) where {T,N} = Texture{T,N,typeof(data)}(data, zero(T), false)
function ConstTexture(val::T) where {T}
    arr = Array{T,0}(undef)
    Texture{T,0,typeof(arr)}(arr, val, true)
end

Base.zero(::Type{RGBSpectrum}) = RGBSpectrum(0.0f0, 0.0f0, 0.0f0, 1.0f0)

# Sample texture data array with UV flip (standard texture coordinate convention)
@propagate_inbounds function _sample_texture_data(data::AbstractArray{T,N}, uv::Point2f)::T where {T,N}
    uv_adj = Vec2f(1f0 - uv[2], uv[1])
    s = unsafe_trunc.(Int32, size(data))
    idx = map(x -> unsafe_trunc(Int32, x), Int32(1) .+ ((s .- Int32(1)) .* uv_adj))
    idx = clamp.(idx, Int32(1), s)
    return data[idx...]
end

# 0-dim arrays are scalar constants - just return the value, no UV sampling
@propagate_inbounds function _sample_texture_data(data::AbstractArray{T,0}, ::Point2f)::T where T
    return data[]
end

function (c::Texture{T})(si::SurfaceInteraction)::T where {T<:TextureType}
    return _sample_texture_data(c.data, si.uv)
end

# UV-only texture evaluation
@propagate_inbounds function evaluate_texture(tex::Texture{T}, uv::Point2f)::T where T
    tex.isconst && return tex.constval
    return _sample_texture_data(tex.data, uv)
end
