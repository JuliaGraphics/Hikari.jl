@propagate_inbounds function XYZ_to_RGB(xyz::Point3f)
    Point3f(
        3.240479f0 * xyz[1] - 1.537150f0 * xyz[2] - 0.498535f0 * xyz[3],
        -0.969256f0 * xyz[1] + 1.875991f0 * xyz[2] + 0.041556f0 * xyz[3],
        0.055648f0 * xyz[1] - 0.204043f0 * xyz[2] + 1.057311f0 * xyz[3],
    )
end
@propagate_inbounds function RGB_to_XYZ(rgb::Point3f)
    Point3f(
        0.412453f0 * rgb[1] + 0.357580f0 * rgb[2] + 0.180423f0 * rgb[3],
        0.212671f0 * rgb[1] + 0.715160f0 * rgb[2] + 0.072169f0 * rgb[3],
        0.019334f0 * rgb[1] + 0.119193f0 * rgb[2] + 0.950227f0 * rgb[3],
    )
end

lerp(c1::C, c2::C, t::Float32) where C<:Spectrum = (1f0 - t) * c1 + t * c2
# Also define lerp for Float32 and Point3f (originally in bounds.jl, now needed here)
lerp(v1::Float32, v2::Float32, t::Float32) = (1 - t) * v1 + t * v2
lerp(p0::Point3f, p1::Point3f, t::Float32) = (1 - t) .* p0 .+ t .* p1

# Clamp RGB to [0,1] for reflectance values
function Base.clamp(c::RGB{Float32})
    RGB{Float32}(clamp(c.r, 0f0, 1f0), clamp(c.g, 0f0, 1f0), clamp(c.b, 0f0, 1f0))
end

get_alpha(s::RGBSpectrum) = s.c[4]

to_XYZ(s::RGBSpectrum) = RGB_to_XYZ(Point3f(s.c[1], s.c[2], s.c[3]))
function from_XYZ(::Type{RGBSpectrum}, xyz::Point3f)
    RGBSpectrum(XYZ_to_RGB(xyz))
end
