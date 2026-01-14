const MICROFACET_REFLECTION = UInt8(7)

function MicrofacetReflection(
    active::Bool, r::S, distribution::Union{Nothing, MicrofacetDistribution}, fresnel::Fresnel, transport,
    ) where {S<:Spectrum}

    UberBxDF{S}(
        active, MICROFACET_REFLECTION; r=r, distribution=distribution,
        fresnel=fresnel, type=BSDF_REFLECTION | BSDF_GLOSSY, transport=transport
    )
end

const MICROFACET_TRANSMISSION = UInt8(8)

function MicrofacetTransmission(
        active::Bool, t::S, distribution::Union{Nothing, MicrofacetDistribution}, η_a::Float32, η_b::Float32, transport,
    ) where {S<:Spectrum}

    UberBxDF{S}(
        active, MICROFACET_TRANSMISSION;
        t=t, distribution=distribution, η_a=η_a, η_b=η_b, fresnel=FresnelDielectric(η_a, η_b),
        type=BSDF_TRANSMISSION | BSDF_GLOSSY, transport=transport
    )
end

const OREN_NAYAR = UInt8(6)

function OrenNayar(active::Bool, r::S, σ::Float32) where {S<:Spectrum}

    σ = deg2rad(σ)
    σ2 = σ * σ
    a = 1.0f0 - (σ2 / (2.0f0 * (σ2 + 0.33f0)))
    b = 0.45f0 * σ2 / (σ2 + 0.09f0)

    return UberBxDF{S}(active, OREN_NAYAR; r=r, a=a, b=b, type=BSDF_DIFFUSE | BSDF_REFLECTION)
end

function distribution_orennayar(o::UberBxDF{S}, wo::Vec3f, wi::Vec3f)::S where {S}
    sin_θi = sin_θ(wi)
    sin_θo = sin_θ(wo)
    # Compute cosine term of Oren-Nayar model.
    max_cos = 0f0
    if sin_θi > 1f-4 && sin_θo > 1f-4
        sin_ϕi = sin_ϕ(wi)
        cos_ϕi = cos_ϕ(wi)
        sin_ϕo = sin_ϕ(wo)
        cos_ϕo = cos_ϕ(wo)
        max_cos = max(0f0, cos_ϕi * cos_ϕo + sin_ϕi * sin_ϕo)
    end
    # Compute sine & tangent terms of Oren-Nayar model.
    if abs(cos_θ(wi) > abs(cos_θ(wo)))
        sin_α = sin_θo
        tan_β = sin_θi / abs(cos_θ(wi))
    else
        sin_α = sin_θi
        tan_β = sin_θo / abs(cos_θ(wo))
    end
    return o.r * (1f0 / π) * (o.a + o.b * max_cos * sin_α * tan_β)
end


function λ(trd::TrowbridgeReitzDistribution, w::Vec3f)::Float32
    θ = abs(tan_θ(w))
    isinf(θ) && return 0f0

    α = √(cos_ϕ(w)^2 * trd.α_x^2 + sin_ϕ(w)^2 * trd.α_y^2)
    α²tanθ² = (α * θ)^2
    (-1f0 + √(1f0 + α²tanθ²)) / 2f0
end

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

@propagate_inbounds function G1(m::MicrofacetDistribution, w::Vec3f)::Float32
    1f0 / (1f0 + λ(m, w))
end

@propagate_inbounds function G(m::MicrofacetDistribution, wo::Vec3f, wi::Vec3f)::Float32
    1f0 / (1f0 + λ(m, wo) + λ(m, wi))
end

"""
Distribution function, which gives the differential area of microfacets
with the surface normal `w`.
"""
function D(trd::TrowbridgeReitzDistribution, w::Vec3f)::Float32
    tan_θ² = tan_θ(w)^2
    isinf(tan_θ²) && return 0f0

    # Calculate cos_θ⁴ without using ^4
    cos_θ² = cos_θ(w) * cos_θ(w)
    cos_θ⁴ = cos_θ² * cos_θ²

    e = (cos_ϕ(w)^2 / (trd.α_x^2) + sin_ϕ(w)^2 / (trd.α_y^2)) * tan_θ²
    1f0 / (π * trd.α_x * trd.α_y * cos_θ⁴ * (1f0 + e)^2)
end

function compute_pdf(m::MicrofacetDistribution, wo::Vec3f, wh::Vec3f)::Float32
    !m.sample_visible_area && return D(m, wh) * abs(cos_θ(wh))
    D(m, wh) * G1(m, wo) * abs(wo ⋅ wh) / abs(cos_θ(wo))
end

function _trowbridge_reitz_sample(
        cosθ::Float32, u1::Float32, u2::Float32,
    )::Tuple{Float32,Float32}

    # Special case -- normal incidence.
    if cosθ > 0.9999f0
        r = √(u1 / (1f0 - u1))
        ϕ = 6.28318530718 * u2
        return r * cos(ϕ), r * sin(ϕ)
    end

    sinθ = √(max(0f0, 1f0 - cosθ^2))
    tanθ = sinθ / cosθ
    a = 1f0 / tanθ
    g1 = 2f0 / (1f0 + √(1f0 + 1f0 / (a^2)))

    # Sample slope x.
    a = 2f0 * u1 / g1 - 1f0
    tmp = 1f0 / (a^2 - 1f0)
    tmp > 1f10 && (tmp = 1f10)
    b = tanθ
    b² = b^2
    d = √(max(0f0, b² * tmp^2 - (a^2 - b²) * tmp))
    slope_x1, slope_x2 = b * tmp - d, b * tmp + d
    slope_x = (a < 0 || slope_x2 > 1f0 / tanθ) ? slope_x1 : slope_x2

    # Sample slope y.
    if u2 > 0.5f0
        s = 1f0
        u2 = 2f0 * (u2 - 0.5f0)
    else
        s = -1f0
        u2 = 2f0 * (0.5f0 - u2)
    end
    z = (
        (u2 * (u2 * (u2 * 0.27385f0 - 0.73369f0) + 0.46341f0))
        /
        (u2 * (u2 * (u2 * 0.093073f0 + 0.309420f0) - 1f0) + 0.597999f0)
    )
    slope_y = s * z * √(1f0 + slope_x^2)

    @real_assert !isinf(slope_y) && !isnan(slope_y)
    slope_x, slope_y
end

function trowbridge_reitz_sample(
        wi::Vec3f, α_x::Float32, α_y::Float32, u1::Float32, u2::Float32,
    )::Vec3f

    # Stretch wi.

    wi_stretch = normalize(Vec3f(wi[1] * α_x, wi[2] * α_y, wi[3]))
    slope_x, slope_y = _trowbridge_reitz_sample(cos_θ(wi_stretch), u1, u2)
    # Rotate.
    c, s = cos_ϕ(wi_stretch), sin_ϕ(wi_stretch)
    tmp = c * slope_x - s * slope_y
    slope_y = s * slope_x + c * slope_y
    slope_x = tmp
    # Unstretch.
    slope_x *= α_x
    slope_y *= α_y
    # Compute normal.
    normalize(Vec3f(-slope_x, -slope_y, 1f0))
end

function sample_wh(
        trd::TrowbridgeReitzDistribution, wo::Vec3f, u::Point2f,
    )::Vec3f

    if trd.sample_visible_area
        flip = wo[3] < 0f0
        wh = trowbridge_reitz_sample(
            flip ? -wo : wo, trd.α_x, trd.α_y, u[1], u[2],
        )
        return flip ? -wh : wh
    end
    cosθ = 0f0
    ϕ = 2f0 * π * u[2]
    if trd.α_x ≈ trd.α_y
        tanθ² = trd.α_x^2 * u[1] / (1f0 - u[1])
        cosθ = 1f0 / √(1f0 + tanθ²)
    else
        ϕ = atan(trd.α_y / trd.α_x * tan(2f0 * π * u[2] + 0.5f0 * π))
        u[2] > 0.5f0 && (ϕ += π)
        sinϕ, cosϕ = sin(ϕ), cos(ϕ)
        α_x², α_y² = trd.α_x^2, trd.α_y^2
        α² = 1f0 / (cosϕ^2 / α_x² + sinϕ^2 / α_y²)
        tanθ² = α² * u[1] / (1f0 - u[1])
        cosθ = 1f0 / √(1f0 + tanθ²)
    end

    sinθ = √(max(0f0, 1f0 - cosθ^2))
    wh = spherical_direction(sinθ, cosθ, ϕ)
    same_hemisphere(wo, wh) ? wh : -wh
end


function distribution_microfacet_reflection(m::UberBxDF{S}, wo::Vec3f, wi::Vec3f)::S where {S<:Spectrum}

    cosθo = abs(cos_θ(wo))
    cosθi = abs(cos_θ(wi))
    wh = wi + wo
    # Degenerate cases for microfacet reflection.

    (cosθi ≈ 0f0 || cosθo ≈ 0f0) && return S(0f0)
    wh ≈ Vec3f(0) && return S(0f0)
    wh = normalize(wh)
    f = m.fresnel(wi ⋅ face_forward(wh, Vec3f(0, 0, 1)))
    return m.r * D(m.distribution, wh) * G(m.distribution, wo, wi) *
        f / (4f0 * cosθi * cosθo)
end

@propagate_inbounds function sample_microfacet_reflection(
        m::UberBxDF{S}, wo::Vec3f, u::Point2f,
    )::Tuple{Vec3f,Float32,RGBSpectrum,UInt8} where {S<:Spectrum}

    wo[3] ≈ 0f0 && return Vec3f(0.0f0), 0.0f0, S(0.0f0), UInt8(0)

    # Sample microfacet orientation `wh` and reflected direction `wi`.

    wh = sample_wh(m.distribution, wo, u)
    (wo ⋅ wh) < 0f0 && return Vec3f(0.0f0), 0.0f0, S(0.0f0), UInt8(0)

    wi = reflect(wo, wh)
    !same_hemisphere(wo, wi) && return Vec3f(0.0f0), 0.0f0, S(0.0f0), UInt8(0)
    # Copmute PDF of `wi` for microfacet reflection.

    pdf = pdf_microfacet_reflection(m, wo, wh)
    wi, pdf, m(wo, wi), UInt8(0)
end

@propagate_inbounds function pdf_microfacet_reflection(
        m::UberBxDF, wo::Vec3f, wi::Vec3f,
    )::Float32

    !same_hemisphere(wo, wi) && return 0f0
    wh = normalize((wo + wi))
    compute_pdf(m.distribution, wo, wh) / (4f0 * wo ⋅ wh)
end

function distribution_microfacet_transmission(m::UberBxDF{S}, wo::Vec3f, wi::Vec3f)::S where {S<:Spectrum}
    # Only transmission.

    same_hemisphere(wo, wi) && return S(0f0)

    cosθo, cosθi = cos_θ(wo), cos_θ(wi)
    (cosθo ≈ 0f0 || cosθi ≈ 0f0) && return S(0f0)
    # Compute `wh` from `wo` & `wi` for microfacet transmission.
    # η = η_t/η_i (transmitted/incident) for half-vector and BTDF formulas
    η = cos_θ(wo) > 0f0 ? (m.η_b / m.η_a) : (m.η_a / m.η_b)
    wh = normalize((wo + wi * η))
    wh[3] < 0 && (wh = -wh)
    # Only transmission if `wh` is on the same side.
    d_o, d_i = wo ⋅ wh, wi ⋅ wh
    (d_o * d_i) > 0 && return S(0f0)

    f = m.fresnel(d_o)
    denom = d_o + η * d_i
    factor = m.transport === Radiance ? (1.0f0 / η) : 1.0f0

    dd, dg = D(m.distribution, wh), G(m.distribution, wo, wi)
    return (S(1f0) - f) * m.t * abs(
        dd * dg * d_o * d_i * η^2 * factor^2
        /
        (cosθi * cosθo * denom^2),
    )
end

@propagate_inbounds function sample_microfacet_transmission(m::UberBxDF{S}, wo::Vec3f, u::Point2f) where {S<:Spectrum}
    wo[3] ≈ 0f0 && return Vec3f(0f0), 0f0, S(0f0), UInt8(0)
    wh = sample_wh(m.distribution, wo, u)
    (wo ⋅ wh) < 0 && return Vec3f(0f0), 0f0, S(0f0), UInt8(0)

    # refract() uses pbrt-v4 convention: eta = η_t / η_i
    # When wo.z > 0 (entering): eta = η_b / η_a
    # When wo.z < 0 (exiting): eta = η_a / η_b
    eta = cos_θ(wo) > 0f0 ? (m.η_b / m.η_a) : (m.η_a / m.η_b)
    refracted, wi = refract(wo, Normal3f(wh), eta)
    !refracted && return Vec3f(0f0), 0f0, S(0f0), UInt8(0)

    pdf = pdf_microfacet_transmission(m, wo, wi)
    wi, pdf, m(wo, wi), UInt8(0)
end

function pdf_microfacet_transmission(
        m::UberBxDF, wo::Vec3f, wi::Vec3f,
    )::Float32

    same_hemisphere(wo, wi) && return 0f0

    # η = η_t/η_i (transmitted/incident) for half-vector formula
    η = cos_θ(wo) > 0f0 ? (m.η_b / m.η_a) : (m.η_a / m.η_b)
    wh = normalize((wo + wi * η))
    @real_assert !isnan(wh)
    d_o, d_i = wo ⋅ wh, wi ⋅ wh
    (d_o * d_i) > 0 && return 0f0
    # Compute change of variables `∂wh∂wi` for microfacet transmission.
    denom = d_o + η * d_i
    ∂wh∂wi = abs(d_i * η^2 / (denom^2))
    return compute_pdf(m.distribution, wo, wh) * ∂wh∂wi
end

# =============================================================================
# FresnelMicrofacet - Combined reflection and transmission for rough dielectrics
# =============================================================================
# This BxDF uses Fresnel-weighted sampling to choose between reflection and
# transmission, avoiding the uniform 50/50 split that causes energy loss.

const FRESNEL_MICROFACET = UInt8(9)

function FresnelMicrofacet(
    active::Bool, r::S, t::S, distribution::MicrofacetDistribution,
    η_a::Float32, η_b::Float32, transport,
) where {S<:Spectrum}
    UberBxDF{S}(
        active, FRESNEL_MICROFACET;
        r=r, t=t, distribution=distribution, η_a=η_a, η_b=η_b,
        fresnel=FresnelDielectric(η_a, η_b),
        type=BSDF_GLOSSY | BSDF_REFLECTION | BSDF_TRANSMISSION,
        transport=transport
    )
end

"""
Distribution function for FresnelMicrofacet - sum of reflection and transmission.
"""
function distribution_fresnel_microfacet(m::UberBxDF{S}, wo::Vec3f, wi::Vec3f)::S where {S<:Spectrum}
    if same_hemisphere(wo, wi)
        # Reflection component
        cosθo = abs(cos_θ(wo))
        cosθi = abs(cos_θ(wi))
        wh = wi + wo
        (cosθi ≈ 0f0 || cosθo ≈ 0f0) && return S(0f0)
        wh ≈ Vec3f(0) && return S(0f0)
        wh = normalize(wh)
        f = m.fresnel(wi ⋅ face_forward(wh, Vec3f(0, 0, 1)))
        return m.r * D(m.distribution, wh) * G(m.distribution, wo, wi) *
            f / (4f0 * cosθi * cosθo)
    else
        # Transmission component
        cosθo, cosθi = cos_θ(wo), cos_θ(wi)
        (cosθo ≈ 0f0 || cosθi ≈ 0f0) && return S(0f0)
        η = cos_θ(wo) > 0f0 ? (m.η_b / m.η_a) : (m.η_a / m.η_b)
        wh = normalize((wo + wi * η))
        wh[3] < 0 && (wh = -wh)
        d_o, d_i = wo ⋅ wh, wi ⋅ wh
        (d_o * d_i) > 0 && return S(0f0)
        f = m.fresnel(d_o)
        denom = d_o + η * d_i
        factor = m.transport === Radiance ? (1.0f0 / η) : 1.0f0
        dd, dg = D(m.distribution, wh), G(m.distribution, wo, wi)
        return (S(1f0) - f) * m.t * abs(
            dd * dg * d_o * d_i * η^2 * factor^2
            /
            (cosθi * cosθo * denom^2),
        )
    end
end

"""
Sample FresnelMicrofacet using Fresnel-weighted probability to choose between
reflection and transmission.
"""
@propagate_inbounds function sample_fresnel_microfacet(
    m::UberBxDF{S}, wo::Vec3f, u::Point2f,
)::Tuple{Vec3f,Float32,S,UInt8} where {S<:Spectrum}

    wo[3] ≈ 0f0 && return Vec3f(0f0), 0f0, S(0f0), UInt8(0)

    # Sample microfacet half-vector first
    wh = sample_wh(m.distribution, wo, u)
    (wo ⋅ wh) < 0f0 && return Vec3f(0f0), 0f0, S(0f0), UInt8(0)

    # Evaluate Fresnel at the half-vector angle
    F = fresnel_dielectric(wo ⋅ wh, m.η_a, m.η_b)

    # Use u[1] directly for the Fresnel choice - it's used by sample_wh but
    # the mapping from u to wh is complex enough that using u[1] < F still
    # gives reasonable stratification of reflection vs transmission choices
    if u[1] < F
        # Reflection
        wi = reflect(wo, wh)
        !same_hemisphere(wo, wi) && return Vec3f(0f0), 0f0, S(0f0), UInt8(0)

        # PDF: product of half-vector pdf, jacobian, and selection probability F
        pdf_h = compute_pdf(m.distribution, wo, wh)
        pdf = pdf_h / (4f0 * abs(wo ⋅ wh)) * F

        cosθo = abs(cos_θ(wo))
        cosθi = abs(cos_θ(wi))
        f_val = m.r * D(m.distribution, wh) * G(m.distribution, wo, wi) *
            S(F) / (4f0 * cosθi * cosθo)

        return wi, pdf, f_val, BSDF_GLOSSY | BSDF_REFLECTION
    else
        # Transmission
        # refract() uses pbrt-v4 convention: eta = η_t / η_i
        eta = cos_θ(wo) > 0f0 ? (m.η_b / m.η_a) : (m.η_a / m.η_b)
        refracted, wi = refract(wo, Normal3f(wh), eta)
        !refracted && return Vec3f(0f0), 0f0, S(0f0), UInt8(0)

        # PDF: product of half-vector pdf, jacobian, and selection probability (1-F)
        d_o, d_i = wo ⋅ wh, wi ⋅ wh
        denom = d_o + eta * d_i
        ∂wh∂wi = abs(d_i * eta^2 / (denom^2))
        pdf_h = compute_pdf(m.distribution, wo, wh)
        pdf = pdf_h * ∂wh∂wi * (1f0 - F)

        cosθo, cosθi = cos_θ(wo), cos_θ(wi)
        # Radiance transport: scale by 1/eta² (matches pbrt-v4)
        factor = m.transport === Radiance ? (1.0f0 / eta) : 1.0f0
        dd, dg = D(m.distribution, wh), G(m.distribution, wo, wi)
        f_val = (S(1f0) - S(F)) * m.t * abs(
            dd * dg * d_o * d_i * eta^2 * factor^2
            /
            (cosθi * cosθo * denom^2),
        )

        return wi, pdf, f_val, BSDF_GLOSSY | BSDF_TRANSMISSION
    end
end

"""
Compute PDF for FresnelMicrofacet given a direction pair.
"""
function pdf_fresnel_microfacet(m::UberBxDF, wo::Vec3f, wi::Vec3f)::Float32
    if same_hemisphere(wo, wi)
        # Reflection PDF
        wh = normalize(wo + wi)
        F = fresnel_dielectric(wo ⋅ wh, m.η_a, m.η_b)
        pdf_h = compute_pdf(m.distribution, wo, wh)
        return pdf_h / (4f0 * abs(wo ⋅ wh)) * F
    else
        # Transmission PDF
        η = cos_θ(wo) > 0f0 ? (m.η_b / m.η_a) : (m.η_a / m.η_b)
        wh = normalize((wo + wi * η))
        wh[3] < 0 && (wh = -wh)
        d_o, d_i = wo ⋅ wh, wi ⋅ wh
        (d_o * d_i) > 0 && return 0f0
        F = fresnel_dielectric(d_o, m.η_a, m.η_b)
        denom = d_o + η * d_i
        ∂wh∂wi = abs(d_i * η^2 / (denom^2))
        pdf_h = compute_pdf(m.distribution, wo, wh)
        return pdf_h * ∂wh∂wi * (1f0 - F)
    end
end
