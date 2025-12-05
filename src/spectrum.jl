@inline function XYZ_to_RGB(xyz::Point3f)
    Point3f(
        3.240479f0 * xyz[1] - 1.537150f0 * xyz[2] - 0.498535f0 * xyz[3],
        -0.969256f0 * xyz[1] + 1.875991f0 * xyz[2] + 0.041556f0 * xyz[3],
        0.055648f0 * xyz[1] - 0.204043f0 * xyz[2] + 1.057311f0 * xyz[3],
    )
end
@inline function RGB_to_XYZ(rgb::Point3f)
    Point3f(
        0.412453f0 * rgb[1] + 0.357580f0 * rgb[2] + 0.180423f0 * rgb[3],
        0.212671f0 * rgb[1] + 0.715160f0 * rgb[2] + 0.072169f0 * rgb[3],
        0.019334f0 * rgb[1] + 0.119193f0 * rgb[2] + 0.950227f0 * rgb[3],
    )
end

Base.:(==)(c::C, i) where C<:Spectrum = all(c.c .== i)
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
# Also define lerp for Float32 and Point3f (originally in bounds.jl, now needed here)
lerp(v1::Float32, v2::Float32, t::Float32) = (1 - t) * v1 + t * v2
lerp(p0::Point3f, p1::Point3f, t::Float32) = (1 - t) .* p0 .+ t .* p1

Base.getindex(c::C, i) where C<:Spectrum = c.c[i]

function Base.clamp(
        c::C, low::Float32 = 0f0, high::Float32 = Inf32,
    ) where C<:Spectrum
    C(clamp.(c.c, low, high))
end

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
    c::Point3f
end
RGBSpectrum(v::Float32 = 0f0) = RGBSpectrum(Point3f(v))
RGBSpectrum(r, g, b) = RGBSpectrum(Point3f(r, g, b))

to_XYZ(s::RGBSpectrum) = RGB_to_XYZ(s.c)
@_inbounds function to_Y(s::RGBSpectrum)::Float32
    0.212671f0 * s.c[1] + 0.715160f0 * s.c[2] + 0.072169f0 * s.c[3]
end
function from_XYZ(::Type{RGBSpectrum}, xyz::Point3f)
    RGBSpectrum(XYZ_to_RGB(xyz))
end
