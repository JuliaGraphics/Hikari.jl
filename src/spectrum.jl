# Keep component comparisons limited to component values. Accepting every RHS
# also intersects unrelated comparisons inferred as `==(Any, T)`, invalidating
# their callers whenever Hikari loads.
Base.:(==)(c::C, i::Union{Number, AbstractArray, Tuple}) where C<:Spectrum = all(c.c .== i)
Base.:(==)(c1::C, c2::C) where C<:Spectrum = all(c1.c .== c2.c)
Base.:+(c1::C, c2::C) where C<:Spectrum = C(c1.c .+ c2.c)
Base.:+(c1::C, c) where C<:Spectrum = C(c1.c .+ c)
Base.:-(c::C) where C<:Spectrum = C(-c.c)(-c.c)
Base.:-(c1::C, c::Number) where C<:Spectrum = C(c1.c .- c)
Base.:-(c1::C, c2::C) where C<:Spectrum = C(c1.c .- c2.c)
Base.:*(c1::C, c2::C) where C<:Spectrum = C(c1.c .* c2.c)
Base.:*(c1::C, f::Number) where C<:Spectrum = C(c1.c .* f)
Base.:*(f::Number, c1::C) where C<:Spectrum = C(c1.c .* f)
Base.:/(c1::C, c2::C) where C<:Spectrum = C(c1.c ./ c2.c)
Base.:/(c1::C, f::Number) where C<:Spectrum = C(c1.c ./ f)
Base.sqrt(c::C) where C<:Spectrum = C(sqrt.(c.c))
Base.:^(c::C, e::Number) where C<:Spectrum = C(c.c .^ e)
Base.exp(c::C) where C<:Spectrum = C(exp.(c.c))
lerp(c1::C, c2::C, t::Float32) where C<:Spectrum = (1f0 - t) * c1 + t * c2

Base.getindex(c::C, i) where C<:Spectrum = c.c[i]

function Base.clamp(
        c::C, low::Float32 = 0f0, high::Float32 = Inf32,
    ) where C<:Spectrum
    C(clamp.(c.c, low, high))
end

# Clamp RGB to [0,1] for reflectance values
function Base.clamp(c::RGB{Float32})
    RGB{Float32}(clamp(c.r, 0f0, 1f0), clamp(c.g, 0f0, 1f0), clamp(c.b, 0f0, 1f0))
end

# Clamp scalar to [0,1] (used when textures evaluate to plain Float32)
Base.clamp(x::AbstractFloat) = clamp(x, zero(x), one(x))

function Base.isnan(c::C) where C<:Spectrum
    # Explicit checks to avoid tuple iteration (causes PHI node errors in SPIR-V)
    isnan(c.c[1]) || isnan(c.c[2]) || isnan(c.c[3])
end

function Base.isinf(c::C) where C<:Spectrum
    # Explicit checks to avoid tuple iteration (causes PHI node errors in SPIR-V)
    isinf(c.c[1]) || isinf(c.c[2]) || isinf(c.c[3])
end

is_black(c::C) where C<:Spectrum = iszero(c.c)

struct RGBSpectrum <: Spectrum
    c::Point4f  # r, g, b, alpha
end
RGBSpectrum(v::Float32 = 0f0) = RGBSpectrum(Point4f(v, v, v, 1f0))
RGBSpectrum(r, g, b) = RGBSpectrum(Point4f(r, g, b, 1f0))
RGBSpectrum(r, g, b, a) = RGBSpectrum(Point4f(r, g, b, a))
RGBSpectrum(c::Point3f) = RGBSpectrum(Point4f(c[1], c[2], c[3], 1f0))
RGBSpectrum(c::Vec3f) = RGBSpectrum(Point4f(c[1], c[2], c[3], 1f0))
RGBSpectrum(c::StaticVector{3, Float32}) = RGBSpectrum(Point4f(c[1], c[2], c[3], 1f0))
get_alpha(s::RGBSpectrum) = s.c[4]
get_alpha(::Real) = 1f0
is_black(c::RGBSpectrum) = c.c[1] == 0f0 && c.c[2] == 0f0 && c.c[3] == 0f0  # ignore alpha

rgb(s::RGBSpectrum) = Point3f(s.c[1], s.c[2], s.c[3])
