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

function (c::Texture{T})(si::SurfaceInteraction)::T where {T<:TextureType}
    c.isconst && return c.const_value
    uv = Vec2f(1f0 - si.uv[2], si.uv[1])
    s = unsafe_trunc.(Int32, size(c.data))
    idx = map(x -> unsafe_trunc(Int32, x), Int32(1) .+ ((s .- Int32(1)) .* uv))
    idx = clamp.(idx, Int32(1), s)
    @_inbounds return c.data[idx...]
end
