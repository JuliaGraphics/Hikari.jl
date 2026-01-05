const TextureType = Union{Float32,S} where S<:Spectrum

struct Texture{ElType, N, T<:AbstractArray{ElType, N}}
    data::T
    const_value::ElType
    isconst::Bool
    function Texture{ElType, N, T}() where {ElType, N, T}
        new{ElType, N, T}()
    end
    function Texture(data::AbstractArray{T, N}, const_value::T, isconst::Bool) where {T, N}
        new{T, N, typeof(data)}(data, const_value, isconst)
    end
end
Base.zero(::Type{RGBSpectrum}) = RGBSpectrum(0.0f0, 0.0f0, 0.0f0, 1.0f0)

Texture(data::AbstractArray{ElType, N}) where {ElType, N} = Texture(data, zero(ElType), false)
Texture(data::Eltype) where Eltype = Texture(Matrix{Eltype}(undef, 0, 0), data, true)
ConstantTexture(data::Eltype) where Eltype = Texture(data)
Texture() = Texture(0.0f0)
no_texture(t::Texture) = !isdefined(t, :data)

struct NoTexture end

function Base.convert(::Type{Texture{ElType,N,T}}, ::NoTexture) where {ElType,N,T}
    return Texture{ElType,N,T}()
end

# GPU-compatible texture evaluation using dispatch on array type
# For constant textures (with 0x0 SMatrix), directly return const_value
# For data textures, sample the array

# Dispatch for actual data textures (non-empty arrays)
@inline function _sample_texture_data(data::AbstractArray, const_value, isconst::Bool, uv::Vec2f)
    # On CPU, constant textures still have empty Matrix data
    # On GPU, we use SMatrix{0,0} which dispatches to the specialized method below
    if isconst
        return const_value
    end
    s = unsafe_trunc.(Int32, size(data))
    idx = map(x -> unsafe_trunc(Int32, x), Int32(1) .+ ((s .- Int32(1)) .* uv))
    idx = clamp.(idx, Int32(1), s)
    @_inbounds return data[idx...]
end

# Dispatch for empty static arrays (constant textures on GPU)
# SMatrix{0,0,T,0} is used as placeholder for constant textures
# This specialized method avoids GPU IR issues by not touching the empty array
@inline function _sample_texture_data(::SMatrix{0,0,T,0}, const_value::T, ::Bool, ::Vec2f) where T
    return const_value
end

function (c::Texture{T})(si::SurfaceInteraction)::T where {T<:TextureType}
    uv = Vec2f(1f0 - si.uv[2], si.uv[1])
    return _sample_texture_data(c.data, c.const_value, c.isconst, uv)
end

# UV-only texture evaluation (for FastWavefront and other simplified integrators)
@inline function evaluate_texture(tex::Texture{T}, uv::Point2f)::T where T
    uv_adj = Vec2f(1f0 - uv[2], uv[1])
    return _sample_texture_data(tex.data, tex.const_value, tex.isconst, uv_adj)
end
