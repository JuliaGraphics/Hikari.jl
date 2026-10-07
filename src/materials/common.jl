# Common shared math functions for spectral BSDF evaluation
#
# Extracted from spectral-eval.jl and reflection/ to centralize:
# - SpectralBSDFSample struct
# - IOR evaluation helpers
# - Fresnel functions (dielectric, conductor, complex)
# - TrowbridgeReitz distribution (GGX) functions
# - Local-space trigonometric helpers (cos_theta, sin_theta, etc.)
# - Coordinate system / frame transforms
# - Refraction helpers (refract_pbrt, refract_microfacet)
# - LayeredBxDF interface sampling (dielectric, diffuse)
# - pbrt-v4 compatible RNG / hash functions (MurmurHash, PCG32)
# - Phase function sampling (Henyey-Greenstein)
# - MIS power heuristic
# - Surface alpha evaluation
#
# These are used by all material-specific spectral evaluation code
# in spectral-eval.jl and the individual material files.

# ============================================================================
# Spectral BSDF Sample Result
# ============================================================================

"""
    SpectralBSDFSample

Result of sampling a BSDF with spectral wavelengths.
Used by PhysicalWavefront for spectral path tracing.
"""
# BxDFFlags — matches pbrt-v4 base/bxdf.h lines 48-62
const BXDF_REFLECTION        = UInt8(1 << 0)
const BXDF_TRANSMISSION      = UInt8(1 << 1)
const BXDF_DIFFUSE           = UInt8(1 << 2)
const BXDF_GLOSSY            = UInt8(1 << 3)
const BXDF_SPECULAR          = UInt8(1 << 4)
const BXDF_DIFFUSE_REFLECTION    = BXDF_DIFFUSE | BXDF_REFLECTION
const BXDF_DIFFUSE_TRANSMISSION  = BXDF_DIFFUSE | BXDF_TRANSMISSION
const BXDF_GLOSSY_REFLECTION     = BXDF_GLOSSY | BXDF_REFLECTION
const BXDF_GLOSSY_TRANSMISSION   = BXDF_GLOSSY | BXDF_TRANSMISSION
const BXDF_SPECULAR_REFLECTION   = BXDF_SPECULAR | BXDF_REFLECTION
const BXDF_SPECULAR_TRANSMISSION = BXDF_SPECULAR | BXDF_TRANSMISSION
const BXDF_ALL = BXDF_DIFFUSE | BXDF_GLOSSY | BXDF_SPECULAR | BXDF_REFLECTION | BXDF_TRANSMISSION

is_reflective(flags::UInt8)   = (flags & BXDF_REFLECTION) != 0
is_transmissive(flags::UInt8) = (flags & BXDF_TRANSMISSION) != 0
is_diffuse(flags::UInt8)      = (flags & BXDF_DIFFUSE) != 0
is_glossy(flags::UInt8)       = (flags & BXDF_GLOSSY) != 0
is_specular(flags::UInt8)     = (flags & BXDF_SPECULAR) != 0
is_non_specular(flags::UInt8) = (flags & (BXDF_DIFFUSE | BXDF_GLOSSY)) != 0

"""
    SpectralBSDFSample — matches pbrt-v4's BSDFSample (base/bxdf.h:121-152)

Fields match pbrt field order: f, wi, pdf, flags, eta, pdfIsProportional.
Additional: secondary_terminated (Hikari-specific for dispersive wavelength termination).
"""
struct SpectralBSDFSample
    f::SpectralRadiance          # Spectral BSDF value
    wi::Vec3f                    # Sampled incident direction
    pdf::Float32                 # Probability density
    flags::UInt8                 # BxDFFlags (reflection/transmission + specular/glossy/diffuse)
    eta::Float32                 # Index of refraction ratio (1.0 for reflection)
    pdf_is_proportional::Bool    # True if pdf is only proportional (LayeredBxDF)
    secondary_terminated::Bool   # Hikari-specific: terminate secondary wavelengths (dispersive IOR)
end

# Convenience constructors
@propagate_inbounds SpectralBSDFSample(f, wi, pdf, flags::UInt8, eta) =
    SpectralBSDFSample(f, wi, pdf, flags, eta, false, false)

@propagate_inbounds SpectralBSDFSample(f, wi, pdf, flags::UInt8) =
    SpectralBSDFSample(f, wi, pdf, flags, 1f0, false, false)

# Default invalid sample
@propagate_inbounds SpectralBSDFSample() =
    SpectralBSDFSample(SpectralRadiance(), Vec3f(0, 0, 1), 0f0, UInt8(0), 1f0, false, false)


# ============================================================================
# IOR Evaluation Helpers
# ============================================================================

# Helper for evaluating IOR values that may be PiecewiseLinearSpectrum or RGB textures.
# When eta/k is a PiecewiseLinearSpectrum, sample it directly at the wavelengths.
# When it's an RGB value (from texture), uplift via the sigmoid polynomial table.
@inline eval_ior_spectral(table, textures, spec::PiecewiseLinearSpectrum, tfc, lambda) = sample(spec, lambda)
@propagate_inbounds function eval_ior_spectral(table, textures, h::TexHandle, tfc, lambda)
    return uplift_rgb_unbounded(table, eval_handle_spectrum(textures, h, tfc), lambda)
end
@inline function eval_ior_spectral(table, textures, tex, tfc, lambda)
    rgb = eval_tex(textures, tex, tfc)
    return uplift_rgb_unbounded(table, rgb, lambda)
end

# ============================================================================
# Fresnel Functions
# ============================================================================

"""
    fresnel_dielectric(cos_θi::Float32, eta::Float32) -> Float32

Compute Fresnel reflection for dielectric materials. Matches pbrt-v4's FrDielectric().
"""
function fresnel_dielectric(cos_θi::Float32, eta::Float32)::Float32
    cos_θi = clamp(cos_θi, -1f0, 1f0)
    if cos_θi < 0f0
        eta = 1f0 / eta
        cos_θi = -cos_θi
    end
    sin2_θi = 1f0 - cos_θi^2
    sin2_θt = sin2_θi / (eta^2)
    sin2_θt >= 1f0 && return 1f0
    cos_θt = sqrt(1f0 - sin2_θt)
    r_parl = (eta * cos_θi - cos_θt) / (eta * cos_θi + cos_θt)
    r_perp = (cos_θi - eta * cos_θt) / (cos_θi + eta * cos_θt)
    return 0.5f0 * (r_parl^2 + r_perp^2)
end

fresnel_dielectric(cos_θi::Float32, ηi::Float32, ηt::Float32)::Float32 =
    fresnel_dielectric(cos_θi, ηt / ηi)

# pbrt-v4 compatible conductor Fresnel with complex IOR:

"""
    complex_sqrt(z::Complex{Float32}) -> Complex{Float32}

Square root of a complex number, following pbrt-v4's `pstd::sqrt` in
`util/complex.h` rather than Julia's `Base.sqrt(::Complex)`.

This is both the faithful port and the fast one. Base's version is
overflow-safe: it rescales via `exponent`/`ldexp` before taking the root. On
the GPU that scaling compiles to two `DomainError` throws ("Cannot be NaN or
Inf", "Cannot be ±0.0") plus an `InexactError` from the `Int` conversion, and
each one boxes a `Float32` through `ijl_box_float32` — machinery that can never
run on a device. Measured in the bump-gold closest-hit shader: 108 of its 218
error/boxing call sites came from this single `sqrt`, in the Fresnel term that
evaluates on every conductor hit at every bounce.

pbrt does not rescale, and it does not need to: the arguments here are bounded
(`cosθ ∈ [0,1]`, measured η/k are order 0.1–10), so there is nothing for the
overflow protection to protect against.
"""
@inline function complex_sqrt(z::Complex{Float32})
    re, im = reim(z)
    n = sqrt(re * re + im * im)
    n == 0f0 && return Complex{Float32}(0f0, 0f0)
    t1 = sqrt(0.5f0 * (n + abs(re)))
    t2 = 0.5f0 * im / t1
    return re >= 0f0 ? Complex{Float32}(t1, t2) :
                       Complex{Float32}(abs(t2), copysign(t1, im))
end

"""
    complex_div(a::Complex{Float32}, b::Complex{Float32}) -> Complex{Float32}

Complex division in Float32, following pbrt-v4's `operator/` in
`util/complex.h` rather than `Base.:/(::Complex, ::Complex)`.

Same reason as [`complex_sqrt`](@ref) above, one step further. Base's version is
overflow-safe by WIDENING: `inv(::Complex{Float32})` calls `widen`, which
promotes to `Complex{Float64}`. Vulkan tolerates that — the device has
`Float64`, so it is merely slow. **Apple GPUs do not have `Float64` at all**, and
the compile fails outright:

    InvalidIRError: unsupported use of double value
      inv @ complex.jl:474  /  @ complex.jl:354  fr_complex @ materials/common.jl

pbrt does not widen, and does not need to for the same reason given above: with
`cosθ ∈ [0,1]` and measured η/k of order 0.1–10, `b.re² + b.im²` cannot overflow
a `Float32`. The reciprocal is taken once and multiplied through, as pbrt does.
"""
@inline function complex_div(a::Complex{Float32}, b::Complex{Float32})
    are, aim = reim(a)
    bre, bim = reim(b)
    scale = 1f0 / (bre * bre + bim * bim)
    return Complex{Float32}(scale * (are * bre + aim * bim),
                            scale * (aim * bre - are * bim))
end

@inline complex_div(a::Float32, b::Complex{Float32}) = complex_div(Complex{Float32}(a, 0f0), b)

"""
    fr_complex(cos_theta_i, eta, k) -> Float32

Compute Fresnel reflectance for a conductor using complex IOR (matches pbrt-v4's FrComplex).

Arguments:
- `cos_theta_i`: Cosine of incident angle (clamped to [0, 1])
- `eta`: Real part of complex IOR (n)
- `k`: Imaginary part of complex IOR (extinction coefficient)

This uses the exact same formula as pbrt-v4 with complex arithmetic.
"""
@propagate_inbounds function fr_complex(cos_theta_i::Float32, eta::Float32, k::Float32)::Float32
    cos_theta_i = clamp(cos_theta_i, 0f0, 1f0)
    sin2_theta_i = 1f0 - cos_theta_i * cos_theta_i

    # Complex IOR and Snell's law (pbrt-v4 FrComplex)
    eta_c = Complex{Float32}(eta, k)
    # `complex_div`, not `/`: Base's complex division widens to `Float64`, which
    # an Apple GPU cannot compile at all. See its docstring.
    sin2_theta_t = complex_div(sin2_theta_i, eta_c * eta_c)
    cos_theta_t = complex_sqrt(1f0 - sin2_theta_t)

    r_parl = complex_div(eta_c * cos_theta_i - cos_theta_t, eta_c * cos_theta_i + cos_theta_t)
    r_perp = complex_div(cos_theta_i - eta_c * cos_theta_t, cos_theta_i + eta_c * cos_theta_t)

    return (abs2(r_parl) + abs2(r_perp)) * 0.5f0
end

"""
    fr_complex_spectral(cos_theta_i, eta, k) -> SpectralRadiance

Compute spectral Fresnel reflectance for a conductor (matches pbrt-v4's FrComplex for SampledSpectrum).
Evaluates fr_complex for each wavelength channel.
"""
@propagate_inbounds function fr_complex_spectral(cos_theta_i::Float32, eta::SpectralRadiance, k::SpectralRadiance)::SpectralRadiance
    return SpectralRadiance(
        fr_complex(cos_theta_i, eta[1], k[1]),
        fr_complex(cos_theta_i, eta[2], k[2]),
        fr_complex(cos_theta_i, eta[3], k[3]),
        fr_complex(cos_theta_i, eta[4], k[4])
    )
end

# ============================================================================
# Local-Space Trigonometric Helpers (pbrt-v4 compatible)
# ============================================================================
# In local shading space, z is the surface normal.

"""
    cos_theta(w) -> Float32

Get cos(θ) of a direction in local coordinates (matches pbrt-v4's CosTheta).
In local space, z is the surface normal.
"""
@inline cos_theta(w::Vec3f)::Float32 = w[3]

"""
    cos2_theta(w) -> Float32

Get cos²(θ) of a direction in local coordinates.
"""
@inline cos2_theta(w::Vec3f)::Float32 = w[3] * w[3]

"""
    abs_cos_theta(w) -> Float32

Get |cos(θ)| of a direction in local coordinates (matches pbrt-v4's AbsCosTheta).
"""
@inline abs_cos_theta(w::Vec3f)::Float32 = abs(w[3])

"""
    sin2_theta(w) -> Float32

Get sin²(θ) of a direction in local coordinates.
"""
@inline sin2_theta(w::Vec3f)::Float32 = max(0f0, 1f0 - cos2_theta(w))

"""
    sin_theta(w) -> Float32

Get sin(θ) of a direction in local coordinates.
"""
@inline sin_theta(w::Vec3f)::Float32 = sqrt(sin2_theta(w))

"""
    tan2_theta(w) -> Float32

Get tan²(θ) of a direction in local coordinates.
"""
@inline function tan2_theta(w::Vec3f)::Float32
    c2 = cos2_theta(w)
    return sin2_theta(w) / c2
end

"""
    cos_phi(w) -> Float32

Get cos(φ) of a direction in local coordinates (matches pbrt-v4's CosPhi).
"""
@inline function cos_phi(w::Vec3f)::Float32
    sin_θ = sin_theta(w)
    return sin_θ == 0f0 ? 1f0 : clamp(w[1] / sin_θ, -1f0, 1f0)
end

"""
    sin_phi(w) -> Float32

Get sin(φ) of a direction in local coordinates (matches pbrt-v4's SinPhi).
"""
@inline function sin_phi(w::Vec3f)::Float32
    sin_θ = sin_theta(w)
    return sin_θ == 0f0 ? 0f0 : clamp(w[2] / sin_θ, -1f0, 1f0)
end

# ============================================================================
# Coordinate System / Frame Transforms
# ============================================================================

"""
    coordinate_system(n::Vec3f) -> (tangent, bitangent)

Build orthonormal basis from a normal vector.
"""
@propagate_inbounds function coordinate_system(n::Vec3f)
    if abs(n[1]) > abs(n[2])
        inv_len = 1f0 / sqrt(n[1] * n[1] + n[3] * n[3])
        tangent = Vec3f(n[3] * inv_len, 0f0, -n[1] * inv_len)
    else
        inv_len = 1f0 / sqrt(n[2] * n[2] + n[3] * n[3])
        tangent = Vec3f(0f0, n[3] * inv_len, -n[2] * inv_len)
    end
    bitangent = cross(n, tangent)
    return (tangent, bitangent)
end

"""
    shading_frame(ns::Vec3f, dpdus::Vec3f) -> (tangent, bitangent)

Build shading frame matching pbrt-v4's Frame::FromXZ(Normalize(dpdus), ns).
X = Normalize(dpdus), Y = Cross(ns, X), Z = ns.
Falls back to coordinate_system(ns) if dpdus is degenerate.
"""
@propagate_inbounds function shading_frame(ns::Vec3f, dpdus::Vec3f)
    len_sq = dot(dpdus, dpdus)
    if len_sq < 1f-10
        return coordinate_system(ns)
    end
    tangent = dpdus / sqrt(len_sq)
    bitangent = cross(ns, tangent)
    return (tangent, bitangent)
end

"""
    local_to_world(local_dir, n, tangent, bitangent) -> Vec3f

Transform direction from local (shading) space to world space.
"""
@propagate_inbounds function local_to_world(local_dir::Vec3f, n::Vec3f, tangent::Vec3f, bitangent::Vec3f)::Vec3f
    return tangent * local_dir[1] + bitangent * local_dir[2] + n * local_dir[3]
end

"""
    world_to_local(v, n, tangent, bitangent) -> Vec3f

Transform direction from world space to local (shading) space.
In local space, the normal is (0, 0, 1).
"""
@propagate_inbounds function world_to_local(v::Vec3f, n::Vec3f, tangent::Vec3f, bitangent::Vec3f)::Vec3f
    return Vec3f(dot(v, tangent), dot(v, bitangent), dot(v, n))
end

"""
    face_forward(v, n) -> Vec3f

Flip v to be in the same hemisphere as n (matches pbrt-v4's FaceForward).
"""
@inline face_forward(v::Vec3f, n::Vec3f)::Vec3f = dot(v, n) < 0f0 ? -v : v

"""
    same_hemisphere(w1, w2) -> Bool

Check if two directions are in the same hemisphere (both have same sign of z).
"""
@inline same_hemisphere(w1::Vec3f, w2::Vec3f)::Bool = w1[3] * w2[3] > 0f0

"""
    reflect(wo, n) -> wi

Compute reflected direction: wi = -wo + 2*dot(wo,n)*n
"""
@inline function reflect(wo::Vec3f, n::Vec3f)::Vec3f
    return -wo + 2f0 * dot(wo, n) * n
end

# ============================================================================
# Refraction Helpers
# ============================================================================

"""
    refract_pbrt(wo, eta) -> (valid, wi, etap)

Compute refracted direction using pbrt-v4 convention.
eta = n_transmitted / n_incident
Returns (valid, wi, effective_eta).
"""
@inline function refract_pbrt(wo::Vec3f, eta::Float32)
    cos_θi = wo[3]
    # Flip eta if entering from below (wo.z < 0)
    etap = cos_θi > 0f0 ? eta : (1f0 / eta)

    sin2_θi = max(0f0, 1f0 - cos_θi^2)
    sin2_θt = sin2_θi / (etap^2)

    # Total internal reflection check
    if sin2_θt >= 1f0
        return (false, Vec3f(0), 1f0)
    end

    cos_θt = sqrt(1f0 - sin2_θt)
    # Flip cos_θt sign to match wo hemisphere convention
    cos_θt_signed = cos_θi > 0f0 ? -cos_θt : cos_θt

    wi = Vec3f(-wo[1] / etap, -wo[2] / etap, cos_θt_signed)
    wi = normalize(wi)

    return (true, wi, etap)
end

"""
    refract_microfacet(wo, wm, eta) -> (valid, wi, etap)

Compute refracted direction through a microfacet with normal wm.
"""
@inline function refract_microfacet(wo::Vec3f, wm::Vec3f, eta::Float32)
    cos_θi = dot(wo, wm)
    etap = cos_θi > 0f0 ? eta : (1f0 / eta)

    sin2_θi = max(0f0, 1f0 - cos_θi^2)
    sin2_θt = sin2_θi / (etap^2)

    if sin2_θt >= 1f0
        return (false, Vec3f(0), 1f0)
    end

    cos_θt = sqrt(1f0 - sin2_θt)
    # Sign convention: transmitted ray goes to opposite side of microfacet
    cos_θt_signed = cos_θi > 0f0 ? -cos_θt : cos_θt

    # Refracted direction formula: wi = -wo/etap + (cos_θi/etap + cos_θt) * wm
    wi = -wo / etap + (cos_θi / etap + cos_θt_signed) * wm
    wi = normalize(wi)

    return (true, wi, etap)
end

# ============================================================================
# GGX / TrowbridgeReitz Microfacet Distribution Functions (pbrt-v4)
# ============================================================================

"""
    roughness_to_α(roughness::Float32) -> Float32

Map [0, 1] scalar roughness to microfacet distribution alpha parameter.

Matches pbrt-v4's TrowbridgeReitzDistribution::RoughnessToAlpha which uses
sqrt(roughness). This provides a more intuitive perceptual mapping where
roughness values close to zero give near-perfect specular reflection.

Note: pbrt-v4 comments suggest Sqr(roughness) might be more perceptually
uniform, but sqrt is retained for compatibility with existing scenes.
"""
@propagate_inbounds function roughness_to_α(roughness::Float32)::Float32
    sqrt(roughness)
end

"""
    regularize_alpha(α::Float32) -> Float32

Regularize a microfacet distribution alpha value to reduce fireflies from
near-specular paths. Matches pbrt-v4's TrowbridgeReitzDistribution::Regularize().

If α < 0.3, doubles it and clamps to [0.1, 0.3]. This increases the roughness
of near-specular surfaces after the first non-specular bounce, reducing variance
from paths that hit nearly-specular surfaces.
"""
@propagate_inbounds function regularize_alpha(α::Float32)::Float32
    α < 0.3f0 ? clamp(2f0 * α, 0.1f0, 0.3f0) : α
end

"""
    trowbridge_reitz_effectively_smooth(alpha_x, alpha_y) -> Bool

Check if the distribution is effectively smooth (matches pbrt-v4's EffectivelySmooth).
"""
@inline trowbridge_reitz_effectively_smooth(alpha_x::Float32, alpha_y::Float32)::Bool =
    max(alpha_x, alpha_y) < 1f-3

"""
    trowbridge_reitz_d(wm, alpha_x, alpha_y) -> Float32

Evaluate the TrowbridgeReitz D (normal distribution function) at microfacet normal wm.
Matches pbrt-v4's TrowbridgeReitzDistribution::D(wm).
"""
@propagate_inbounds function trowbridge_reitz_d(wm::Vec3f, alpha_x::Float32, alpha_y::Float32)::Float32
    tan2_θ = tan2_theta(wm)
    isinf(tan2_θ) && return 0f0

    cos4_θ = cos2_theta(wm) * cos2_theta(wm)
    cos4_θ < 1f-16 && return 0f0

    e = tan2_θ * ((cos_phi(wm) / alpha_x)^2 + (sin_phi(wm) / alpha_y)^2)
    return 1f0 / (Float32(π) * alpha_x * alpha_y * cos4_θ * (1f0 + e)^2)
end

"""
    trowbridge_reitz_lambda(w, alpha_x, alpha_y) -> Float32

Compute Lambda(w) for Smith masking-shadowing (matches pbrt-v4's Lambda).
"""
@propagate_inbounds function trowbridge_reitz_lambda(w::Vec3f, alpha_x::Float32, alpha_y::Float32)::Float32
    tan2_θ = tan2_theta(w)
    isinf(tan2_θ) && return 0f0

    alpha2 = (cos_phi(w) * alpha_x)^2 + (sin_phi(w) * alpha_y)^2
    return (sqrt(1f0 + alpha2 * tan2_θ) - 1f0) * 0.5f0
end

"""
    trowbridge_reitz_g1(w, alpha_x, alpha_y) -> Float32

Compute G1(w) Smith masking function (matches pbrt-v4's G1).
"""
@inline trowbridge_reitz_g1(w::Vec3f, alpha_x::Float32, alpha_y::Float32)::Float32 =
    1f0 / (1f0 + trowbridge_reitz_lambda(w, alpha_x, alpha_y))

"""
    trowbridge_reitz_g(wo, wi, alpha_x, alpha_y) -> Float32

Compute G(wo, wi) Smith masking-shadowing function (matches pbrt-v4's G).
"""
@inline trowbridge_reitz_g(wo::Vec3f, wi::Vec3f, alpha_x::Float32, alpha_y::Float32)::Float32 =
    1f0 / (1f0 + trowbridge_reitz_lambda(wo, alpha_x, alpha_y) + trowbridge_reitz_lambda(wi, alpha_x, alpha_y))

"""
    trowbridge_reitz_d_pdf(w, wm, alpha_x, alpha_y) -> Float32

Evaluate the visible normal distribution D(w, wm) for PDF computation.
Matches pbrt-v4's D(w, wm) = G1(w) / AbsCosTheta(w) * D(wm) * AbsDot(w, wm).
"""
@propagate_inbounds function trowbridge_reitz_d_pdf(w::Vec3f, wm::Vec3f, alpha_x::Float32, alpha_y::Float32)::Float32
    return trowbridge_reitz_g1(w, alpha_x, alpha_y) / abs_cos_theta(w) *
           trowbridge_reitz_d(wm, alpha_x, alpha_y) * abs(dot(w, wm))
end

"""
    trowbridge_reitz_pdf(w, wm, alpha_x, alpha_y) -> Float32

Compute PDF for visible normal sampling (matches pbrt-v4's PDF).
"""
@inline trowbridge_reitz_pdf(w::Vec3f, wm::Vec3f, alpha_x::Float32, alpha_y::Float32)::Float32 =
    trowbridge_reitz_d_pdf(w, wm, alpha_x, alpha_y)

"""
    trowbridge_reitz_sample_wm(w, u, alpha_x, alpha_y) -> Vec3f

Sample visible normal from TrowbridgeReitz distribution (matches pbrt-v4's Sample_wm).
"""
@propagate_inbounds function trowbridge_reitz_sample_wm(w::Vec3f, u::Point2f, alpha_x::Float32, alpha_y::Float32)::Vec3f
    # Transform w to hemispherical configuration
    wh = normalize(Vec3f(alpha_x * w[1], alpha_y * w[2], w[3]))
    if wh[3] < 0f0
        wh = -wh
    end

    # Find orthonormal basis for visible normal sampling
    t1 = wh[3] < 0.99999f0 ? normalize(cross(Vec3f(0f0, 0f0, 1f0), wh)) : Vec3f(1f0, 0f0, 0f0)
    t2 = cross(wh, t1)

    # Generate uniformly distributed points on the unit disk (polar sampling)
    r = sqrt(u[1])
    phi = 2f0 * Float32(π) * u[2]
    p_x = r * cos(phi)
    p_y = r * sin(phi)

    # Warp hemispherical projection for visible normal sampling
    h = sqrt(1f0 - p_x * p_x)
    p_y = lerp(h, p_y, 0.5f0 * (1f0 + wh[3]))

    # Reproject to hemisphere and transform normal to ellipsoid configuration
    pz = sqrt(max(0f0, 1f0 - p_x * p_x - p_y * p_y))
    nh = p_x * t1 + p_y * t2 + pz * wh

    return normalize(Vec3f(alpha_x * nh[1], alpha_y * nh[2], max(1f-6, nh[3])))
end

"""
    sample_ggx_vndf(wo, alpha_x, alpha_y, u) -> Vec3f

Sample visible normal from GGX distribution.
"""
@propagate_inbounds function sample_ggx_vndf(wo::Vec3f, alpha_x::Float32, alpha_y::Float32, u::Point2f)::Vec3f
    # Transform to hemisphere configuration
    wh = normalize(Vec3f(alpha_x * wo[1], alpha_y * wo[2], wo[3]))

    if wh[3] < 0f0
        wh = -wh
    end

    # Sample projected area
    t1 = wh[3] < 0.9999f0 ? normalize(cross(Vec3f(0, 0, 1), wh)) : Vec3f(1, 0, 0)
    t2 = cross(wh, t1)

    r = sqrt(u[1])
    phi = 2f0 * Float32(π) * u[2]
    p1 = r * cos(phi)
    p2 = r * sin(phi)
    s = 0.5f0 * (1f0 + wh[3])
    p2 = (1f0 - s) * sqrt(max(0f0, 1f0 - p1 * p1)) + s * p2

    # Compute normal
    n = p1 * t1 + p2 * t2 + sqrt(max(0f0, 1f0 - p1 * p1 - p2 * p2)) * wh

    # Transform back
    return normalize(Vec3f(alpha_x * n[1], alpha_y * n[2], max(1f-6, n[3])))
end

# ============================================================================
# pbrt-v4 Compatible RNG and Hash Functions for LayeredBxDF
# ============================================================================
# These functions exactly match pbrt-v4's implementation for deterministic
# random number generation within LayeredBxDF. This is critical for
# reproducibility and correct firefly-free rendering.
#
# Reference: pbrt-v4 src/pbrt/util/hash.h, src/pbrt/util/rng.h

"""
    murmur_hash_64a(data::NTuple{N,UInt8}, seed::UInt64) -> UInt64

MurmurHash2 64-bit hash function, exactly matching pbrt-v4's MurmurHash64A.
Reference: https://github.com/explosion/murmurhash/blob/master/murmurhash/MurmurHash2.cpp
"""
@inline function murmur_hash_64a(data::NTuple{N,UInt8}, seed::UInt64=UInt64(0))::UInt64 where N
    m = 0xc6a4a7935bd1e995
    r = 47

    h = seed ⊻ (UInt64(N) * m)

    # Process 8-byte chunks
    n_chunks = N ÷ 8
    @inbounds for i in 0:(n_chunks-1)
        # Read 8 bytes as UInt64 (little-endian)
        k = UInt64(data[8*i + 1]) |
            (UInt64(data[8*i + 2]) << 8) |
            (UInt64(data[8*i + 3]) << 16) |
            (UInt64(data[8*i + 4]) << 24) |
            (UInt64(data[8*i + 5]) << 32) |
            (UInt64(data[8*i + 6]) << 40) |
            (UInt64(data[8*i + 7]) << 48) |
            (UInt64(data[8*i + 8]) << 56)

        k *= m
        k ⊻= k >> r
        k *= m

        h ⊻= k
        h *= m
    end

    # Handle remaining bytes (switch fallthrough in C++)
    remaining = N & 7
    offset = 8 * n_chunks
    @inbounds if remaining >= 7
        h ⊻= UInt64(data[offset + 7]) << 48
    end
    @inbounds if remaining >= 6
        h ⊻= UInt64(data[offset + 6]) << 40
    end
    @inbounds if remaining >= 5
        h ⊻= UInt64(data[offset + 5]) << 32
    end
    @inbounds if remaining >= 4
        h ⊻= UInt64(data[offset + 4]) << 24
    end
    @inbounds if remaining >= 3
        h ⊻= UInt64(data[offset + 3]) << 16
    end
    @inbounds if remaining >= 2
        h ⊻= UInt64(data[offset + 2]) << 8
    end
    @inbounds if remaining >= 1
        h ⊻= UInt64(data[offset + 1])
        h *= m
    end

    h ⊻= h >> r
    h *= m
    h ⊻= h >> r

    return h
end

"""
    mix_bits(v::UInt64) -> UInt64

Bit mixing function from pbrt-v4's hash.h.
Reference: http://zimbry.blogspot.ch/2011/09/better-bit-mixing-improving-on.html
"""
@inline function mix_bits(v::UInt64)::UInt64
    v ⊻= v >> 31
    v *= 0x7fb5d329728ea185
    v ⊻= v >> 27
    v *= 0x81dadef4bc2dd44d
    v ⊻= v >> 33
    return v
end

"""
    float32_to_bytes(v::Float32) -> NTuple{4,UInt8}

GPU-compatible conversion of Float32 to bytes using Core.bitcast.
"""
@inline function float32_to_bytes(v::Float32)::NTuple{4,UInt8}
    bits = Core.bitcast(UInt32, v)
    return (
        UInt8(bits & 0xff),
        UInt8((bits >> 8) & 0xff),
        UInt8((bits >> 16) & 0xff),
        UInt8((bits >> 24) & 0xff)
    )
end

"""
    uint64_to_bytes(v::UInt64) -> NTuple{8,UInt8}

GPU-compatible conversion of UInt64 to bytes.
"""
@inline function uint64_to_bytes(v::UInt64)::NTuple{8,UInt8}
    return (
        UInt8(v & 0xff),
        UInt8((v >> 8) & 0xff),
        UInt8((v >> 16) & 0xff),
        UInt8((v >> 24) & 0xff),
        UInt8((v >> 32) & 0xff),
        UInt8((v >> 40) & 0xff),
        UInt8((v >> 48) & 0xff),
        UInt8((v >> 56) & 0xff)
    )
end

"""
    pbrt_hash(args...) -> UInt64

Hash function matching pbrt-v4's variadic Hash() template.
Packs arguments into a byte buffer and applies MurmurHash64A.
Uses GPU-compatible bit manipulation instead of reinterpret.
"""
@inline function pbrt_hash(v::Float32)::UInt64
    bytes = float32_to_bytes(v)
    murmur_hash_64a(bytes, UInt64(0))
end

@inline function pbrt_hash(v::UInt64)::UInt64
    bytes = uint64_to_bytes(v)
    murmur_hash_64a(bytes, UInt64(0))
end

@inline function pbrt_hash(v::Vec3f)::UInt64
    # Vec3f is 12 bytes (3 x Float32)
    x_bytes = float32_to_bytes(v[1])
    y_bytes = float32_to_bytes(v[2])
    z_bytes = float32_to_bytes(v[3])
    bytes = (x_bytes..., y_bytes..., z_bytes...)
    murmur_hash_64a(bytes, UInt64(0))
end

@inline function pbrt_hash(p::Point3f)::UInt64
    # Point3f is 12 bytes (3 x Float32)
    x_bytes = float32_to_bytes(p[1])
    y_bytes = float32_to_bytes(p[2])
    z_bytes = float32_to_bytes(p[3])
    bytes = (x_bytes..., y_bytes..., z_bytes...)
    murmur_hash_64a(bytes, UInt64(0))
end

@inline function pbrt_hash(seed::UInt64, v::Vec3f)::UInt64
    # Hash(seed, wo) - combine seed with vector
    seed_bytes = uint64_to_bytes(seed)
    x_bytes = float32_to_bytes(v[1])
    y_bytes = float32_to_bytes(v[2])
    z_bytes = float32_to_bytes(v[3])
    bytes = (seed_bytes..., x_bytes..., y_bytes..., z_bytes...)
    murmur_hash_64a(bytes, UInt64(0))
end

@inline function pbrt_hash(a::UInt64, b::Float32)::UInt64
    a_bytes = uint64_to_bytes(a)
    b_bytes = float32_to_bytes(b)
    bytes = (a_bytes..., b_bytes...)
    murmur_hash_64a(bytes, UInt64(0))
end

@inline function pbrt_hash(a::Float32, b::Point2f)::UInt64
    a_bytes = float32_to_bytes(a)
    bx_bytes = float32_to_bytes(b[1])
    by_bytes = float32_to_bytes(b[2])
    bytes = (a_bytes..., bx_bytes..., by_bytes...)
    murmur_hash_64a(bytes, UInt64(0))
end

# ============================================================================
# PCG32 Random Number Generator
# ============================================================================
# GPU-compatible functional implementation
# Exact port of pbrt-v4's RNG class but using immutable state

const PCG32_DEFAULT_STATE = 0x853c49e6748fea9b
const PCG32_DEFAULT_STREAM = 0xda3e39cb94b95bdb
const PCG32_MULT = 0x5851f42d4c957f2d

"""
    PCG32State

GPU-compatible immutable PCG32 random number generator state.
Uses tuple-like struct for stack allocation on GPU.
"""
struct PCG32State
    state::UInt64
    inc::UInt64
end

"""
    pcg32_init(seq_index::UInt64, seed::UInt64) -> PCG32State

Initialize PCG32 with sequence index and seed, matching pbrt-v4's SetSequence.
Returns initialized state.
"""
@inline function pcg32_init(seq_index::UInt64, seed::UInt64)::PCG32State
    inc = (seq_index << 1) | UInt64(1)
    state = UInt64(0)

    # First advance
    state = state * PCG32_MULT + inc

    # Add seed
    state += seed

    # Second advance
    state = state * PCG32_MULT + inc

    return PCG32State(state, inc)
end

@inline function pcg32_init(seq_index::UInt64)::PCG32State
    pcg32_init(seq_index, mix_bits(seq_index))
end

"""
    pcg32_uniform_u32(rng::PCG32State) -> (UInt32, PCG32State)

Generate uniform random UInt32 and return new state.
Matching pbrt-v4's Uniform<uint32_t>().
"""
@inline function pcg32_uniform_u32(rng::PCG32State)::Tuple{UInt32, PCG32State}
    oldstate = rng.state
    newstate = oldstate * PCG32_MULT + rng.inc
    # Keep intermediate values as UInt64, then mask to UInt32
    xorshifted = ((oldstate >> 18) ⊻ oldstate) >> 27
    rot = oldstate >> 59
    xorshifted32 = UInt32(xorshifted & 0xFFFFFFFF)
    rot32 = UInt32(rot & 0x1F)
    result = (xorshifted32 >> rot32) | (xorshifted32 << ((32 - rot32) & 31))
    return (result, PCG32State(newstate, rng.inc))
end

"""
    pcg32_uniform_f32(rng::PCG32State) -> (Float32, PCG32State)

Generate uniform random Float32 in [0, 1) and return new state.
Matching pbrt-v4's Uniform<float>().
"""
@inline function pcg32_uniform_f32(rng::PCG32State)::Tuple{Float32, PCG32State}
    u32, new_rng = pcg32_uniform_u32(rng)
    # 0x1p-32f = 2^-32 as Float32 ≈ 2.3283064e-10
    f = min(Float32(1) - eps(Float32), Float32(u32) * 2.3283064f-10)
    return (f, new_rng)
end

# ============================================================================
# Exponential Sampling and Phase Functions
# ============================================================================

"""
    sample_exponential(u::Float32, a::Float32) -> Float32

Sample from exponential distribution with rate parameter a.
Returns -log(1-u)/a, matching pbrt-v4's SampleExponential.
"""
@inline function sample_exponential(u::Float32, a::Float32)::Float32
    return -log(1f0 - u) / a
end

"""
    Tr(thickness, w) -> Float32

Transmittance through a layer of given thickness along direction w.
Used in LayeredBxDF random walk.
"""
@propagate_inbounds function layer_transmittance(thickness::Float32, w::Vec3f)::Float32
    abs(thickness) <= eps(Float32) && return 1f0
    exp(-abs(thickness / w[3]))
end

"""
    sample_hg_phase_spectral(g, wo, u) -> (wi, phase_pdf)

Sample direction from Henyey-Greenstein phase function.
"""
@propagate_inbounds function sample_hg_phase_spectral(g::Float32, wo::Vec3f, u::Point2f)
    # Sample cos_θ from HG distribution
    cos_θ = if abs(g) < 1f-3
        # Isotropic case
        1f0 - 2f0 * u[1]
    else
        g2 = g * g
        sqr_term = (1f0 - g2) / (1f0 - g + 2f0 * g * u[1])
        clamp((1f0 + g2 - sqr_term * sqr_term) / (2f0 * g), -1f0, 1f0)
    end

    sin_θ = sqrt(max(0f0, 1f0 - cos_θ * cos_θ))
    ϕ = 2f0 * Float32(π) * u[2]

    # Build local frame around -wo
    t1, t2 = coordinate_system(-wo)
    wi = sin_θ * cos(ϕ) * t1 + sin_θ * sin(ϕ) * t2 + cos_θ * (-wo)
    wi = normalize(wi)

    # HG phase function value (equals PDF)
    g2 = g * g
    denom = 1f0 + g2 - 2f0 * g * cos_θ
    p = (1f0 - g2) / (4f0 * Float32(π) * denom * sqrt(max(1f-10, denom)))

    return (wi, p)
end

"""
    hg_phase_pdf(g, cos_θ) -> Float32

Evaluate Henyey-Greenstein phase function PDF.
"""
@propagate_inbounds function hg_phase_pdf(g::Float32, cos_θ::Float32)::Float32
    g2 = g * g
    denom = 1f0 + g2 - 2f0 * g * cos_θ
    (1f0 - g2) / (4f0 * Float32(π) * denom * sqrt(max(1f-10, denom)))
end

"""
    power_heuristic(nf, fPdf, ng, gPdf) -> Float32

Balance heuristic for MIS with power=2.
"""
@inline function power_heuristic(nf::Int, fPdf::Float32, ng::Int, gPdf::Float32)::Float32
    f = nf * fPdf
    g = ng * gPdf
    f_sq = f * f
    g_sq = g * g
    if f_sq + g_sq == 0f0
        return 0f0
    end
    return f_sq / (f_sq + g_sq)
end

# ============================================================================
# LayeredBxDF Helper Types and Interface Sampling
# ============================================================================

# BxDFReflTransFlags constants are defined at the top of this file (lines 31-42)

"""
    LayeredBSDFSample - Internal sample result for LayeredBxDF interfaces

Contains the sampled direction, BSDF value, pdf, and flags indicating
whether the sample is reflection/transmission and specular/glossy.
"""
struct LayeredBSDFSample
    f::SpectralRadiance      # BSDF value (spectral)
    wi::Vec3f                # Sampled direction
    pdf::Float32             # Probability density
    is_reflection::Bool      # True if reflection, false if transmission
    is_specular::Bool        # True if delta distribution
    eta::Float32             # Relative IOR (for transmission)
    valid::Bool              # Whether sample is valid
end

LayeredBSDFSample() = LayeredBSDFSample(SpectralRadiance(), Vec3f(0), 0f0, false, false, 1f0, false)

"""
    sample_dielectric_transmission_spectral(eta, wo, uc) -> (wi, T, valid)

Sample transmission through a dielectric interface.
Returns transmitted direction, transmittance, and validity flag.

Uses pbrt-v4 convention: eta = n_t / n_i (transmitted IOR / incident IOR).
The wo direction is in local shading space where z is the surface normal.
"""
@propagate_inbounds function sample_dielectric_transmission_spectral(
    eta::Float32, wo::Vec3f, uc::Float32
)
    cos_θo = abs(wo[3])
    F = fresnel_dielectric(cos_θo, eta)

    # Use uc to decide reflection vs transmission
    if uc < F
        # Reflection - not transmission
        return (Vec3f(0), 0f0, false)
    end

    # Compute transmitted direction using pbrt-v4 Snell's law formula
    # For pbrt-v4 convention: sin²θt = sin²θi / eta²
    sin2_θi = max(0f0, 1f0 - cos_θo^2)
    sin2_θt = sin2_θi / (eta^2)

    # Check for total internal reflection
    if sin2_θt >= 1f0
        return (Vec3f(0), 0f0, false)
    end

    cos_θt = sqrt(1f0 - sin2_θt)

    # pbrt-v4 refracted direction: -wo/eta + (cos_i/eta - cos_t) * n
    # In local coords, n = (0, 0, sign(wo.z))
    entering = wo[3] > 0f0
    n_sign = entering ? 1f0 : -1f0

    wi = Vec3f(
        -wo[1] / eta,
        -wo[2] / eta,
        (cos_θo / eta - cos_θt) * n_sign
    )
    wi = normalize(wi)

    T = 1f0 - F
    return (wi, T, true)
end

"""
    sample_dielectric_interface(wo, uc, u, alpha_x, alpha_y, eta, refl_trans_flags) -> LayeredBSDFSample

Sample the dielectric coating interface (top layer in CoatedDiffuse).
Handles both smooth (specular) and rough (microfacet) dielectric surfaces.

This matches pbrt-v4's DielectricBxDF::Sample_f exactly.
"""
@propagate_inbounds function sample_dielectric_interface(
    wo::Vec3f, uc::Float32, u::Point2f,
    alpha_x::Float32, alpha_y::Float32, eta::Float32,
    refl_trans_flags::UInt8,
    radiance_mode::Bool = true  # pbrt-v4 TransportMode: true=Radiance, false=Importance
)
    is_smooth = trowbridge_reitz_effectively_smooth(alpha_x, alpha_y)

    if is_smooth || eta == 1f0
        # Sample perfect specular dielectric BSDF
        cos_θo = wo[3]
        R = fresnel_dielectric(cos_θo, eta)
        T = 1f0 - R

        # Compute probabilities for sampling reflection vs transmission
        pr = (refl_trans_flags & BXDF_REFLECTION) != 0 ? R : 0f0
        pt = (refl_trans_flags & BXDF_TRANSMISSION) != 0 ? T : 0f0

        if pr == 0f0 && pt == 0f0
            return LayeredBSDFSample()
        end

        if uc < pr / (pr + pt)
            # Sample perfect specular reflection
            wi = Vec3f(-wo[1], -wo[2], wo[3])
            f_val = R / abs(wi[3])
            pdf = pr / (pr + pt)
            return LayeredBSDFSample(SpectralRadiance(f_val), wi, pdf, true, true, 1f0, true)
        else
            # Sample perfect specular transmission
            # Compute refracted direction
            valid, wi, etap = refract_pbrt(wo, eta)
            if !valid
                return LayeredBSDFSample()
            end

            f_val = T / abs(wi[3])
            # pbrt-v4: if (mode == TransportMode::Radiance) ft /= Sqr(etap);
            if radiance_mode
                f_val /= etap * etap
            end
            pdf = pt / (pr + pt)
            return LayeredBSDFSample(SpectralRadiance(f_val), wi, pdf, false, true, etap, true)
        end
    else
        # Sample rough dielectric BSDF using microfacet distribution
        wm = trowbridge_reitz_sample_wm(wo, u, alpha_x, alpha_y)
        cos_θo_m = dot(wo, wm)

        R = fresnel_dielectric(cos_θo_m, eta)
        T = 1f0 - R

        pr = (refl_trans_flags & BXDF_REFLECTION) != 0 ? R : 0f0
        pt = (refl_trans_flags & BXDF_TRANSMISSION) != 0 ? T : 0f0

        if pr == 0f0 && pt == 0f0
            return LayeredBSDFSample()
        end

        if uc < pr / (pr + pt)
            # Sample reflection at rough dielectric interface
            wi = reflect(wo, wm)
            if !same_hemisphere(wo, wi)
                return LayeredBSDFSample()
            end

            # Compute PDF of rough dielectric reflection
            pdf_m = trowbridge_reitz_pdf(wo, wm, alpha_x, alpha_y)
            pdf = pdf_m / (4f0 * abs(cos_θo_m)) * pr / (pr + pt)

            D = trowbridge_reitz_d(wm, alpha_x, alpha_y)
            G = trowbridge_reitz_g(wo, wi, alpha_x, alpha_y)
            f_val = D * G * R / (4f0 * wo[3] * wi[3])

            return LayeredBSDFSample(SpectralRadiance(f_val), wi, pdf, true, false, 1f0, true)
        else
            # Sample transmission at rough dielectric interface
            valid, wi, etap = refract_microfacet(wo, wm, eta)
            if !valid || same_hemisphere(wo, wi) || wi[3] == 0f0
                return LayeredBSDFSample()
            end

            # Compute PDF of rough dielectric transmission
            denom = (dot(wi, wm) + dot(wo, wm) / etap)^2
            dwm_dwi = abs(dot(wi, wm)) / denom
            pdf_m = trowbridge_reitz_pdf(wo, wm, alpha_x, alpha_y)
            pdf = pdf_m * dwm_dwi * pt / (pr + pt)

            D = trowbridge_reitz_d(wm, alpha_x, alpha_y)
            G = trowbridge_reitz_g(wo, wi, alpha_x, alpha_y)
            f_val = T * D * G * abs(dot(wi, wm) * dot(wo, wm) / (wi[3] * wo[3] * denom))
            # pbrt-v4: if (mode == TransportMode::Radiance) f /= Sqr(etap);
            if radiance_mode
                f_val /= etap * etap
            end

            return LayeredBSDFSample(SpectralRadiance(f_val), wi, pdf, false, false, etap, true)
        end
    end
end

"""
    eval_dielectric_interface(wo, wi, alpha_x, alpha_y, eta) -> (f, pdf)

Evaluate the dielectric interface BSDF for given directions.
Returns (f_value, pdf) for the given wo/wi pair.
"""
@propagate_inbounds function eval_dielectric_interface(
    wo::Vec3f, wi::Vec3f, alpha_x::Float32, alpha_y::Float32, eta::Float32,
    radiance_mode::Bool = true
)
    is_smooth = trowbridge_reitz_effectively_smooth(alpha_x, alpha_y)

    if is_smooth || eta == 1f0
        # Specular dielectric: f = 0 for non-delta directions
        return (SpectralRadiance(), 0f0)
    end

    # Rough dielectric evaluation
    if same_hemisphere(wo, wi)
        # Reflection
        wh = normalize(wo + wi)
        if wh[3] < 0f0
            wh = -wh
        end

        cos_θo_h = dot(wo, wh)
        R = fresnel_dielectric(cos_θo_h, eta)

        D = trowbridge_reitz_d(wh, alpha_x, alpha_y)
        G = trowbridge_reitz_g(wo, wi, alpha_x, alpha_y)

        f_val = D * G * R / (4f0 * wo[3] * wi[3])
        # PDF includes reflection mixing weight R/(R+T) = R (since R+T=1, BXDF_ALL assumed)
        # Matches pbrt-v4 DielectricBxDF::PDF: mfDistrib.PDF(wo,wm)/(4*AbsDot(wo,wm)) * pr/(pr+pt)
        pdf = trowbridge_reitz_pdf(wo, wh, alpha_x, alpha_y) / (4f0 * abs(cos_θo_h)) * R

        return (SpectralRadiance(f_val), pdf)
    else
        # Transmission
        etap = wo[3] > 0f0 ? eta : (1f0 / eta)

        # Compute half vector for transmission
        wh = normalize(wo + wi * etap)
        if wh[3] < 0f0
            wh = -wh
        end

        cos_θo_h = dot(wo, wh)
        cos_θi_h = dot(wi, wh)

        # Check for same side condition
        if cos_θo_h * cos_θi_h > 0f0
            return (SpectralRadiance(), 0f0)
        end

        R = fresnel_dielectric(cos_θo_h, eta)
        T = 1f0 - R

        denom = (cos_θi_h + cos_θo_h / etap)^2
        D = trowbridge_reitz_d(wh, alpha_x, alpha_y)
        G = trowbridge_reitz_g(wo, wi, alpha_x, alpha_y)

        f_val = T * D * G * abs(cos_θi_h * cos_θo_h / (wo[3] * wi[3] * denom))
        # pbrt-v4: if (mode == TransportMode::Radiance) f /= Sqr(etap);
        if radiance_mode
            f_val /= etap * etap
        end

        # PDF includes transmission mixing weight T/(R+T) = T (since R+T=1, BXDF_ALL assumed)
        # Matches pbrt-v4 DielectricBxDF::PDF: mfDistrib.PDF(wo,wm)*dwm_dwi * pt/(pr+pt)
        dwm_dwi = abs(cos_θi_h) / denom
        pdf = trowbridge_reitz_pdf(wo, wh, alpha_x, alpha_y) * dwm_dwi * T

        return (SpectralRadiance(f_val), pdf)
    end
end

"""
    pdf_dielectric_interface(wo, wi, alpha_x, alpha_y, eta, refl_trans_flags) -> Float32

Compute PDF of dielectric interface sampling.
"""
@propagate_inbounds function pdf_dielectric_interface(
    wo::Vec3f, wi::Vec3f, alpha_x::Float32, alpha_y::Float32, eta::Float32,
    refl_trans_flags::UInt8 = BXDF_ALL
)
    is_smooth = trowbridge_reitz_effectively_smooth(alpha_x, alpha_y)

    if is_smooth || eta == 1f0
        return 0f0  # Specular has delta PDF
    end

    if same_hemisphere(wo, wi)
        # Reflection
        if (refl_trans_flags & BXDF_REFLECTION) == 0
            return 0f0
        end

        wh = normalize(wo + wi)
        if wh[3] < 0f0
            wh = -wh
        end

        cos_θo_h = abs(dot(wo, wh))
        R = fresnel_dielectric(cos_θo_h, eta)
        T = 1f0 - R

        pr = (refl_trans_flags & BXDF_REFLECTION) != 0 ? R : 0f0
        pt = (refl_trans_flags & BXDF_TRANSMISSION) != 0 ? T : 0f0

        pdf = trowbridge_reitz_pdf(wo, wh, alpha_x, alpha_y) / (4f0 * cos_θo_h)
        return pdf * pr / (pr + pt)
    else
        # Transmission
        if (refl_trans_flags & BXDF_TRANSMISSION) == 0
            return 0f0
        end

        etap = wo[3] > 0f0 ? eta : (1f0 / eta)
        wh = normalize(wo + wi * etap)
        if wh[3] < 0f0
            wh = -wh
        end

        cos_θo_h = dot(wo, wh)
        cos_θi_h = dot(wi, wh)

        if cos_θo_h * cos_θi_h > 0f0
            return 0f0
        end

        R = fresnel_dielectric(abs(cos_θo_h), eta)
        T = 1f0 - R

        pr = (refl_trans_flags & BXDF_REFLECTION) != 0 ? R : 0f0
        pt = (refl_trans_flags & BXDF_TRANSMISSION) != 0 ? T : 0f0

        denom = (cos_θi_h + cos_θo_h / etap)^2
        dwm_dwi = abs(cos_θi_h) / denom
        pdf = trowbridge_reitz_pdf(wo, wh, alpha_x, alpha_y) * dwm_dwi

        return pdf * pt / (pr + pt)
    end
end

"""
    sample_diffuse_interface(wo, u, reflectance) -> LayeredBSDFSample

Sample the diffuse base layer (bottom in CoatedDiffuse).
This is a simple cosine-weighted hemisphere sampler.
"""
@propagate_inbounds function sample_diffuse_interface(
    wo::Vec3f, u::Point2f, reflectance::SpectralRadiance,
    refl_trans_flags::UInt8
)
    # Diffuse only reflects, never transmits
    if (refl_trans_flags & BXDF_REFLECTION) == 0
        return LayeredBSDFSample()
    end

    # Cosine-weighted hemisphere sampling
    wi = cosine_sample_hemisphere(u)

    # Ensure wi is in same hemisphere as wo
    if wo[3] < 0f0
        wi = Vec3f(wi[1], wi[2], -wi[3])
    end

    cos_θi = abs(wi[3])
    if cos_θi < 1f-6
        return LayeredBSDFSample()
    end

    # Lambertian: f = R/π, pdf = cos_θ/π
    f = reflectance * (1f0 / Float32(π))
    pdf = cos_θi / Float32(π)

    return LayeredBSDFSample(f, wi, pdf, true, false, 1f0, true)
end

"""
    eval_diffuse_interface(wo, wi, reflectance) -> (f, pdf)

Evaluate diffuse BSDF for given directions.
"""
@propagate_inbounds function eval_diffuse_interface(
    wo::Vec3f, wi::Vec3f, reflectance::SpectralRadiance
)
    if !same_hemisphere(wo, wi)
        return (SpectralRadiance(), 0f0)
    end
    f = reflectance * (1f0 / Float32(π))
    pdf = abs(wi[3]) / Float32(π)
    return (f, pdf)
end

"""
    pdf_diffuse_interface(wo, wi) -> Float32

Compute PDF of diffuse sampling.
"""
@inline function pdf_diffuse_interface(wo::Vec3f, wi::Vec3f)::Float32
    if !same_hemisphere(wo, wi)
        return 0f0
    end
    return abs(wi[3]) / Float32(π)
end

# ============================================================================
# Conductor Interface (for LayeredBxDF bottom — pbrt-v4 ConductorBxDF)
# ============================================================================

"""
Sample conductor interface (reflection only). Matches pbrt-v4 ConductorBxDF::Sample_f.
Returns LayeredBSDFSample. Used as bottom interface in LayeredBxDF for CoatedConductor.
"""
@propagate_inbounds function sample_conductor_interface(
    wo::Vec3f, u::Point2f,
    alpha_x::Float32, alpha_y::Float32,
    eta::SpectralRadiance, k::SpectralRadiance,
    refl_trans_flags::UInt8
)
    # Conductor only reflects
    if (refl_trans_flags & BXDF_REFLECTION) == 0
        return LayeredBSDFSample()
    end

    is_smooth = trowbridge_reitz_effectively_smooth(alpha_x, alpha_y)

    if is_smooth
        # Specular reflection: wi = (-wo.x, -wo.y, wo.z)
        wi = Vec3f(-wo[1], -wo[2], wo[3])
        F = fr_complex_spectral(abs(wi[3]), eta, k)
        f_val = F / abs(wi[3])
        return LayeredBSDFSample(f_val, wi, 1f0, true, true, 1f0, true)
    else
        # Rough microfacet reflection
        wm = trowbridge_reitz_sample_wm(wo, u, alpha_x, alpha_y)
        wi = -wo + 2f0 * dot(wo, wm) * wm
        if !same_hemisphere(wo, wi) || wi[3] == 0f0
            return LayeredBSDFSample()
        end

        cos_θo = abs(wo[3])
        cos_θi = abs(wi[3])
        if cos_θo == 0f0 || cos_θi == 0f0
            return LayeredBSDFSample()
        end

        F = fr_complex_spectral(abs(dot(wo, wm)), eta, k)
        D = trowbridge_reitz_d(wm, alpha_x, alpha_y)
        G = trowbridge_reitz_g(wo, wi, alpha_x, alpha_y)
        f_val = D * F * G / (4f0 * cos_θi * cos_θo)

        pdf_m = trowbridge_reitz_pdf(wo, wm, alpha_x, alpha_y)
        pdf = pdf_m / (4f0 * abs(dot(wo, wm)))

        return LayeredBSDFSample(f_val, wi, pdf, true, false, 1f0, true)
    end
end

"""
Evaluate conductor interface BSDF. Matches pbrt-v4 ConductorBxDF::f + PDF.
Returns (f::SpectralRadiance, pdf::Float32).
"""
@propagate_inbounds function eval_conductor_interface(
    wo::Vec3f, wi::Vec3f,
    alpha_x::Float32, alpha_y::Float32,
    eta::SpectralRadiance, k::SpectralRadiance
)
    is_smooth = trowbridge_reitz_effectively_smooth(alpha_x, alpha_y)

    if is_smooth
        return (SpectralRadiance(), 0f0)  # Delta function — zero for non-delta eval
    end

    if !same_hemisphere(wo, wi)
        return (SpectralRadiance(), 0f0)
    end

    cos_θo = abs(wo[3])
    cos_θi = abs(wi[3])
    if cos_θo == 0f0 || cos_θi == 0f0
        return (SpectralRadiance(), 0f0)
    end

    wh = normalize(wo + wi)
    if wh[3] < 0f0
        wh = -wh
    end

    F = fr_complex_spectral(abs(dot(wo, wh)), eta, k)
    D = trowbridge_reitz_d(wh, alpha_x, alpha_y)
    G = trowbridge_reitz_g(wo, wi, alpha_x, alpha_y)
    f_val = D * F * G / (4f0 * cos_θo * cos_θi)

    pdf = trowbridge_reitz_pdf(wo, wh, alpha_x, alpha_y) / (4f0 * abs(dot(wo, wh)))

    return (f_val, pdf)
end

"""PDF for conductor interface. Matches pbrt-v4 ConductorBxDF::PDF."""
@propagate_inbounds function pdf_conductor_interface(
    wo::Vec3f, wi::Vec3f,
    alpha_x::Float32, alpha_y::Float32
)::Float32
    if !same_hemisphere(wo, wi)
        return 0f0
    end
    if trowbridge_reitz_effectively_smooth(alpha_x, alpha_y)
        return 0f0
    end
    wh = normalize(wo + wi)
    if wh[3] < 0f0
        wh = -wh
    end
    return trowbridge_reitz_pdf(wo, wh, alpha_x, alpha_y) / (4f0 * abs(dot(wo, wh)))
end

# ============================================================================
# MediumInterface Forwarding
# ============================================================================

# MediumInterface is defined in integrators/volpath/media.jl
# These forwarding functions delegate BSDF operations to the wrapped material

"""
    get_bxdf for MediumInterface - forwards to the wrapped material.

Forwarding here rather than forwarding `sample_bsdf_spectral` /
`evaluate_bsdf_spectral` separately is what lets each leaf material keep one
BSDF implementation (on its evaluated form) instead of also retaining a raw
version for wrappers to land on.
"""
@propagate_inbounds get_bxdf(
    mi::MediumInterface, table::RGBToSpectrumTable, textures,
    tfc::TextureFilterContext, lambda::Wavelengths, regularize::Bool,
) = get_bxdf(mi.material, table, textures, tfc, lambda, regularize)

"""
    is_emissive for MediumInterface - forwards to wrapped material.
"""
@propagate_inbounds function is_emissive(mi::MediumInterface)
    return is_emissive(mi.material)
end

# ============================================================================
# Surface Alpha Evaluation (for shadow ray pass-through)
# ============================================================================
# Following pbrt-v4: alpha is evaluated at intersection time to determine
# if a surface should be treated as transparent. Used by trace_shadow_transmittance
# to allow light through alpha-masked geometry (e.g. GLTF BLEND mode foliage).

# All other material types are fully opaque (Diffuse override is in diffuse.jl)
@propagate_inbounds get_surface_alpha(::Material, ::Any, ::TextureFilterContext) = 1f0
@propagate_inbounds get_surface_alpha(mat::Material, textures, uv::Point2f) =
    get_surface_alpha(mat, textures, TextureFilterContext(uv))
