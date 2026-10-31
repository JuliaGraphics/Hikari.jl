const TextureType = Union{Float32,S} where S<:Spectrum
abstract type Texture end

struct ConstantTexture{T<:TextureType} <: Texture
    value::T
end

function (c::ConstantTexture{T})(si::SurfaceInteraction)::T where T<:TextureType
    c.value
end

struct ScaleTexture{T<:Texture} <: Texture
    texture_1::T
    texture_2::T
end

function (s::ScaleTexture)(si::SurfaceInteraction)
    s.texture_1(si) * s.texture_2(si)
end

"""
`texture_1` & `texture_2` may be of any single texture type,
but `mix` should be a texture that returns floating-point value
that is used to interpolate between the first two.
"""
struct MixTexture{T<:Texture,S<:Texture} <: Texture
    texture_1::T
    texture_2::T
    mix::S
end

function (m::MixTexture)(si::SurfaceInteraction)
    t::Float32 = m.mix(si)
    (1 - t) * m.texture_1(si) + t * m.texture_2(si)
end

struct BilerpTexture{T<:Texture,M<:Mapping2D} <: Texture
    mapping::M
    v00::T
    v01::T
    v10::T
    v11::T
end

function (b::BilerpTexture)(si::SurfaceInteraction)
    st, _, _ = b.mapping(si)
    (1f0 - st[1]) * (1f0 - st[2]) * b.v00 +
    (1f0 - st[1]) * st[2] * b.v01 +
    st[1] * (1f0 - st[2]) * b.v10 +
    st[1] * st[2] * b.v11
end
