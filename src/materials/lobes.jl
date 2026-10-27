# Ported from Blender Cycles v5.2.1: intern/cycles/kernel/closure/bsdf_diffuse.h,
# bsdf_oren_nayar.h, bsdf_sheen.h, bssrdf.h and svm/closure.h, copyright 2011-2022 Blender
# Foundation, SPDX-License-Identifier: Apache-2.0; and kernel/closure/bsdf_microfacet.h,
# bsdf_util.h, adapted by Cycles from Open Shading Language, copyright 2009-2010 Sony Pictures
# Imageworks Inc., et al. All Rights Reserved, copyright 2011-2022 Blender Foundation,
# SPDX-License-Identifier: BSD-3-Clause. Modified: rewritten in Julia as typed lobes, spectral.
# See LICENSES/Apache-2.0.txt and LICENSES/BSD-3-Clause.txt.

# ============================================================================
# BSDF lobes: the building blocks every material's BSDF is made of
# ============================================================================
#
# A lobe is a small struct with its parameters resolved at the hit, evaluated and sampled in
# its own frame, where `z` is its normal. A material's BSDF is a tuple of weighted lobes in
# their frames (`LobeBSDF`), and the tuple's types are the material's: evaluating and sampling
# unroll over it when the kernel is compiled, so a kernel holds exactly the lobes its material
# can produce and no switch over kinds of lobe. Cycles evaluates the same lobes as a list of
# closures that one evaluator switches over, because its node graphs are interpreted; here the
# node programs compute the lobes' inputs, and the lobes themselves are code.
#
# A lobe's colours are `Vec3f` while the node program sets it up, as Cycles does in RGB, and
# `SpectralRadiance` once `lift` has taken the colours it reads to the path's wavelengths.
#
# The interface of a lobe `l`, in its frame:
#
#   lobe_f(l, wo, wi, lambda)          -> (f, pdf)   f without any cosine
#   lobe_sample(l, wo, u, uc, lambda)  -> (wi, f, pdf, flags, eta), f and pdf for a singular
#                                         sample only; `uc` picks between reflection and refraction
#   lift(l, table, lambda, regularize) -> the lobe with spectral colours
#   is_diffuse_lobe(l)                 -> whether a bump's shadowing applies to it as to a diffuse lobe

abstract type BSDFLobe end

@inline is_diffuse_lobe(::BSDFLobe) = false
"""The colour a lobe of RGB `weight` reflects, for the denoiser's albedo guide."""
@inline guide_albedo(::BSDFLobe, weight::Vec3f) = weight
@inline lobe_frame(::BSDFLobe, n::Vec3f, t::Vec3f, wo::Vec3f) =
    t == Vec3f(0) ? (n, coordinate_system(n)...) : (n, t, cross(n, t))

const NO_LOBE_SAMPLE = (Vec3f(0, 0, 1), SpectralRadiance(0f0), 0f0, 0x00, 1f0)

"""
What a lobe's sample may take, as pbrt's `BxDFReflTransFlags`: anything (`AnyScattering`), or a
reflection (`OnlyReflection`) or a transmission (`OnlyTransmission`) alone, which the interfaces of a
layered BSDF are sampled for. `lobe_f`, `lobe_sample`, `mixture` and `pick_sample` take one; the
pdf they give is that of sampling under it.
"""
abstract type Scattering end
struct AnyScattering <: Scattering end
struct OnlyReflection <: Scattering end
struct OnlyTransmission <: Scattering end

@inline allows_reflection(::Scattering) = true
@inline allows_reflection(::OnlyTransmission) = false
@inline allows_transmission(::Scattering) = true
@inline allows_transmission(::OnlyReflection) = false
@inline allows(s::Scattering, transmission::Bool) = transmission ? allows_transmission(s) : allows_reflection(s)

"""Whether `lobe` can scatter at all under `s`: a lobe that only reflects cannot transmit."""
@inline can_scatter(::BSDFLobe, s::Scattering) = allows_reflection(s)

"""
`lobe_f` and `lobe_sample` under a restriction, for a lobe that scatters to one side only: its pdf
and samples where `s` allows them, none where it does not.
"""
@inline function lobe_f(l::BSDFLobe, wo::Vec3f, wi::Vec3f, lambda::Wavelengths, s::Scattering)
    f, pdf = lobe_f(l, wo, wi, lambda)
    return (f, allows(s, wi[3] < 0f0) ? pdf : 0f0)
end
@inline function lobe_sample(l::BSDFLobe, wo::Vec3f, u::Point2f, uc::Float32, lambda::Wavelengths, s::Scattering)
    sample = lobe_sample(l, wo, u, uc, lambda)
    return allows(s, is_transmissive(sample[4])) ? sample : NO_LOBE_SAMPLE
end

"""The lobe seen from its other side, as a two-sided BxDF of pbrt's answers for `wo` below it."""
@inline other_side(l::BSDFLobe) = l
"""The lobe transporting importance where it transported radiance: pbrt's `!mode`."""
@inline adjoint(l::BSDFLobe) = l

# ── Diffuse ─────────────────────────────────────────────────────────────────

"""
Cycles' energy-preserving Oren-Nayar (OpenPBR's): `a + b t` for single scattering and `ms` for
the light that scatters more than once, scaled by what single scattering leaves at the light's
angle. Lambert is `a = 1/π`, `b = 0`, `ms = 0`.
"""
struct DiffuseLobe{C} <: BSDFLobe
    a::Float32
    b::Float32
    ms::C
end

"""Lambert's lobe, which the diffuse lobe is at no roughness, with colours of type `C`."""
LambertLobe(::Type{C} = Vec3f) where {C} = DiffuseLobe{C}(1f0 / Float32(π), 0f0, C(0f0))

"""
Cycles' `bsdf_oren_nayar_setup` for `color` and `roughness` seen from `wo`, in the frame of `n`;
Lambert where the roughness is almost zero.
"""
@inline function DiffuseLobe(roughness::Float32, color::Vec3f, n::Vec3f, wo::Vec3f)
    roughness < 1f-5 && return LambertLobe()
    sigma = clamp(roughness, 0f0, 1f0)
    a = 1f0 / (Float32(π) + sigma * (Float32(π) / 2f0 - 2f0 / 3f0))
    b = sigma * a
    albedo = _saturate3(color)
    e_avg = a * Float32(π) + ((2f0 * Float32(π) - 5.6f0) / 3f0) * b
    e_ms = (1f0 / Float32(π)) .* albedo .* albedo .* (e_avg / (1f0 - e_avg)) ./ (1f0 .- albedo .* (1f0 - e_avg))
    e_v = a * Float32(π) + b * _oren_nayar_g(max(dot(n, wo), 0f0))
    return DiffuseLobe{Vec3f}(a, b, e_ms .* (1f0 - e_v))
end

@inline lift(l::DiffuseLobe{Vec3f}, table, lambda, ::Bool) =
    DiffuseLobe{SpectralRadiance}(l.a, l.b, l.b == 0f0 ? SpectralRadiance(0f0) : _uplift(table, l.ms, lambda))
@inline is_diffuse_lobe(::DiffuseLobe) = true

"""
`f` and pdf of a diffuse lobe from the cosines of `wi` and `wo` with its normal and between
them: all an isotropic lobe of no tangent depends on, so no frame is built for it.
"""
@inline function diffuse_f(l::DiffuseLobe{SpectralRadiance}, cos_i::Float32, cos_o::Float32, cos_io::Float32)
    cos_i > 0f0 || return (SpectralRadiance(0f0), 0f0)
    l.b == 0f0 && return (SpectralRadiance(l.a), cos_i / Float32(π))
    nv = max(cos_o, 0f0)
    t = cos_io - cos_i * nv
    t > 0f0 && (t /= max(cos_i, nv) + floatmin(Float32))
    e_l = l.a * Float32(π) + l.b * _oren_nayar_g(cos_i)
    return (SpectralRadiance(l.a + l.b * t) + l.ms * (1f0 - e_l), cos_i / Float32(π))
end

@inline lobe_f(l::DiffuseLobe{SpectralRadiance}, wo::Vec3f, wi::Vec3f, ::Wavelengths) =
    diffuse_f(l, wi[3], wo[3], dot(wi, wo))
@inline cosine_f(l::DiffuseLobe{SpectralRadiance}, cos_i::Float32, cos_o::Float32, cos_io::Float32) =
    diffuse_f(l, cos_i, cos_o, cos_io)

@inline lobe_sample(::DiffuseLobe, wo::Vec3f, u::Point2f, ::Float32, ::Wavelengths) =
    (cosine_sample_hemisphere(u), SpectralRadiance(0f0), 0f0, BXDF_DIFFUSE_REFLECTION, 1f0)

"""Lambert's lobe through the surface: Blender's Translucent BSDF."""
struct TranslucentLobe <: BSDFLobe end

@inline lift(l::TranslucentLobe, table, lambda, ::Bool) = l
@inline can_scatter(::TranslucentLobe, s::Scattering) = allows_transmission(s)
@inline is_diffuse_lobe(::TranslucentLobe) = true

@inline function cosine_f(::TranslucentLobe, cos_i::Float32, cos_o::Float32, cos_io::Float32)
    cos_i < 0f0 || return (SpectralRadiance(0f0), 0f0)
    return (SpectralRadiance(1f0 / Float32(π)), -cos_i / Float32(π))
end
@inline lobe_f(l::TranslucentLobe, wo::Vec3f, wi::Vec3f, ::Wavelengths) = cosine_f(l, wi[3], wo[3], dot(wi, wo))

@inline function lobe_sample(::TranslucentLobe, wo::Vec3f, u::Point2f, ::Float32, ::Wavelengths)
    wi = cosine_sample_hemisphere(u)
    return (Vec3f(wi[1], wi[2], -wi[3]), SpectralRadiance(0f0), 0f0, BXDF_DIFFUSE_TRANSMISSION, 1f0)
end

# ── Weighted lobes and the BSDF they make ───────────────────────────────────

"""
Cycles' `bump_shadowing_term` with Bump Map Correction, for a lobe whose normal `n` a bump or
normal map tilted off the smooth shading normal `ns`: no light where `wi` is on the other side of
the smooth surface than `n` says, and for diffuse lobes the GGX shadowing of a microsurface as
rough as the tilt, which keeps the bumps from lighting up past the terminator.
"""
@inline function bump_shadowing(diffuse::Bool, n::Vec3f, ns::Vec3f, wi::Vec3f)
    n == ns && return 1f0
    cos_si, cos_sn = dot(ns, wi), dot(ns, n)
    cos_si * cos_sn * dot(n, wi) < 0f0 && return 0f0
    diffuse || return 1f0
    cos_i, cos_d = abs(cos_si), abs(cos_sn)
    (cos_d >= 1f0 || cos_i >= 1f0) && return 1f0
    cos_i < 1f-6 && return 0f0
    alpha2 = clamp(0.125f0 * (1f0 / cos_d^2 - 1f0), 0f0, 1f0)
    return 1f0 / (1f0 + 0.5f0 * (sqrt(1f0 + alpha2 * (1f0 / cos_i^2 - 1f0)) - 1f0))
end

"""
A lobe as a BSDF holds it: what it is weighted by, its share of the BSDF's samples, and its
frame, `n` its normal and `t` its tangent, zero for an isotropic lobe.
"""
struct Weighted{L <: BSDFLobe, C}
    lobe::L
    weight::C
    sample_weight::Float32
    n::Vec3f
    t::Vec3f
end

Weighted(lobe::BSDFLobe, weight::Vec3f, n::Vec3f; t::Vec3f = Vec3f(0), sample_weight::Float32 = _avg(weight)) =
    Weighted(lobe, weight, sample_weight, n, t)

"""`w` with its colours at the path's wavelengths, its weight scaled by `scale`."""
@inline lift(w::Weighted{<:BSDFLobe, Vec3f}, scale::Float32, table, lambda::Wavelengths, regularize::Bool) =
    Weighted(lift(w.lobe, table, lambda, regularize), _uplift(table, _saturate3(w.weight * scale), lambda),
             w.sample_weight, w.n, w.t)

"""
The BSDF a material makes at a hit: its weighted lobes, of which one is sampled in proportion
to its share of `total`, and what the hit emits.
"""
struct LobeBSDF{L <: Tuple}
    lobes::L
    total::Float32
    lambda::Wavelengths
    emission::SpectralRadiance
    dispersive::Bool
end

"""
    pbrt_frame(wo, ns, dpdus) -> (n, t)

The frame of a pbrt-v4 BSDF at a hit seen from `wo` with shading normal `ns`: pbrt's shading frame,
with the normal on the side `wo` is on, as pbrt's BxDFs are two-sided.
"""
@inline function pbrt_frame(wo::Vec3f, ns::Vec3f, dpdus::Vec3f)
    n = dot(wo, ns) < 0f0 ? -ns : ns
    return n, first(shading_frame(n, dpdus))
end

"""
pbrt-v4's `TrowbridgeReitzDistribution` roughness: each axis of a rough distribution at least 1e-4,
which its `D` needs.
"""
@inline function pbrt_alphas(alpha_x::Float32, alpha_y::Float32)
    trowbridge_reitz_effectively_smooth(alpha_x, alpha_y) && return (alpha_x, alpha_y)
    return (max(alpha_x, 1f-4), max(alpha_y, 1f-4))
end

"""pbrt-v4's `DielectricBxDF` of relative IOR `eta`, tinted by `kr` and `kt`, as a lobe."""
@inline function pbrt_dielectric(eta::Float32, alpha_x::Float32, alpha_y::Float32;
                                 kr::SpectralRadiance = SpectralRadiance(1f0), kt::SpectralRadiance = SpectralRadiance(1f0))
    ax, ay = pbrt_alphas(alpha_x, alpha_y)
    return MicrofacetLobe{:glass}(DielectricFresnel(kr, kt); alpha_x = ax, alpha_y = ay, ior = eta, convention = PbrtRadiance())
end

"""pbrt-v4's `ConductorBxDF` of complex IOR `eta` + i`k` as a lobe."""
@inline function pbrt_conductor(eta::SpectralRadiance, k::SpectralRadiance, alpha_x::Float32, alpha_y::Float32)
    ax, ay = pbrt_alphas(alpha_x, alpha_y)
    return MicrofacetLobe{:reflection}(ConductorFresnel(eta, k); alpha_x = ax, alpha_y = ay, convention = PbrtRadiance())
end

"""
The BSDF of the weighted lobes of a pbrt-v4 material, each sampled in proportion to its sample
weight; `dispersive` when an IOR in it is a spectrum.
"""
@inline pbrt_bsdf(lobes::Tuple, lambda::Wavelengths; dispersive::Bool = false) =
    LobeBSDF(lobes, reduce(+, map(w -> w.sample_weight, lobes); init = 0f0), lambda, SpectralRadiance(0f0), dispersive)

"""
`f` of the weighted lobe `w` in Hikari's convention, which leaves the cosine to the integrator and
takes it with the shading normal `ns`: the lobe's own cosine, with its normal, over that one, and
the shadowing of a normal a bump tilted off `ns`.
"""
@propagate_inbounds function weighted_f(w::Weighted, wo::Vec3f, wi::Vec3f, ns::Vec3f, lambda::Wavelengths,
                                        s::Scattering = AnyScattering())
    n, t, b = lobe_frame(w.lobe, w.n, w.t, wo)
    shadowing = bump_shadowing(is_diffuse_lobe(w.lobe), n, ns, wi)
    cos_s = abs(dot(wi, ns))
    (shadowing > 0f0 && cos_s > 1f-7) || return (SpectralRadiance(0f0), 0f0)
    f, pdf = lobe_f(w.lobe, world_to_local(wo, n, t, b), world_to_local(wi, n, t, b), lambda, s)
    return (w.weight * f * (shadowing * abs(dot(wi, n)) / cos_s), pdf)
end

"""The lobes evaluated from cosines alone (`cosine_f`), isotropic and of no tangent."""
const CosineLobe = Union{DiffuseLobe, TranslucentLobe}

@propagate_inbounds function weighted_f(w::Weighted{<:CosineLobe}, wo::Vec3f, wi::Vec3f, ns::Vec3f, ::Wavelengths,
                                        s::Scattering = AnyScattering())
    n = w.n
    shadowing = bump_shadowing(true, n, ns, wi)
    cos_s = abs(dot(wi, ns))
    (shadowing > 0f0 && cos_s > 1f-7) || return (SpectralRadiance(0f0), 0f0)
    cos_i = dot(wi, n)
    f, pdf = cosine_f(w.lobe, cos_i, dot(wo, n), dot(wi, wo))
    return (w.weight * f * (shadowing * abs(cos_i) / cos_s), allows(s, cos_i < 0f0) ? pdf : 0f0)
end

"""
The lobes' `f` summed, and their pdfs under `s` weighted by their shares of the samples: unrolled
over the tuple. The pdf is of sampling the lobes in proportion to their shares, to divide by the
shares of those that can scatter under `s` (`share`).
"""
@inline mixture(::Tuple{}, wo, wi, ns, lambda, s::Scattering = AnyScattering()) = (SpectralRadiance(0f0), 0f0)
@propagate_inbounds function mixture(lobes::Tuple{Any, Vararg{Any}}, wo::Vec3f, wi::Vec3f, ns::Vec3f, lambda::Wavelengths,
                                     s::Scattering = AnyScattering())
    f, pdf = mixture(Base.tail(lobes), wo, wi, ns, lambda, s)
    w = first(lobes)
    w.sample_weight > 0f0 || return (f, pdf)
    fi, pi = weighted_f(w, wo, wi, ns, lambda, s)
    return (f + fi, pdf + pi * w.sample_weight)
end

"""A lobe's share of the samples under `s`: none where it cannot scatter so."""
@inline share(w::Weighted, s::Scattering) = can_scatter(w.lobe, s) ? w.sample_weight : 0f0
"""The shares of the lobes under `s`, which a pick among them is in proportion to."""
@inline share(lobes::Tuple, s::Scattering) = reduce(+, map(w -> share(w, s), lobes); init = 0f0)

"""
The lobe whose share of `total` the sample `target` falls in, sampled: `(wi, f, pdf, flags, eta, sample_weight, n,
terminates)` in world space, `f` and pdf for a singular sample only, `n` the lobe's normal, `terminates` whether
the sample ends the path's secondary wavelengths at a dispersive BSDF (`terminates_secondary`).
"""
@inline pick_sample(::Tuple{}, target, before, wo, u, lambda, s::Scattering = AnyScattering()) =
    (Vec3f(0, 0, 1), SpectralRadiance(0f0), 0f0, 0x00, 1f0, 0f0, Vec3f(0, 0, 1), false)
@propagate_inbounds function pick_sample(lobes::Tuple{Any, Vararg{Any}}, target::Float32, before::Float32, wo::Vec3f,
                                         u::Point2f, lambda::Wavelengths, s::Scattering = AnyScattering())
    w = first(lobes)
    sw = share(w, s)
    after = before + sw
    (sw > 0f0 && (target < after || length(lobes) == 1)) ||
        return pick_sample(Base.tail(lobes), target, after, wo, u, lambda, s)
    uc = clamp((target - before) / max(sw, 1f-12), 0f0, 0.99999994f0)
    n, t, b = lobe_frame(w.lobe, w.n, w.t, wo)
    wi, f, pdf, flags, eta = lobe_sample(w.lobe, world_to_local(wo, n, t, b), u, uc, lambda, s)
    return (normalize(local_to_world(wi, n, t, b)), w.weight * f, pdf, flags, eta, sw, n,
            terminates_secondary(w.lobe, flags))
end

"""
Whether a sample of `lobe` with `flags` at a dispersive BSDF ends the path's secondary wavelengths:
a singular or refracted one, as Cycles' dispersion; every one of a lobe in pbrt's convention, as pbrt
terminates them at a BSDF whose IOR is a spectrum.
"""
@inline terminates_secondary(::BSDFLobe, flags::UInt8) = is_specular(flags) || is_transmissive(flags)

"""Whether a lobe's BSDF, where it disperses, ends a path's secondary wavelengths where the path meets it: pbrt's lobes."""
@inline ends_dispersion_at_vertex(::BSDFLobe) = false

"""
Whether a path's secondary wavelengths end where it meets `bsdf`, before it samples the lights
there, as pbrt-v4's materials end them in `GetBxDF` where an IOR is a spectrum. A shader's
dispersion ends them only with the refraction that disperses (`terminates_secondary`).
"""
@inline terminates_at_vertex(b::LobeBSDF) =
    b.dispersive && reduce(&, map(w -> ends_dispersion_at_vertex(w.lobe), b.lobes); init = true)
@inline terminates_at_vertex(bsdf) = false

@propagate_inbounds function evaluate_bsdf_spectral(
    bsdf::LobeBSDF, ::RGBToSpectrumTable, textures,
    wo::Vec3f, wi::Vec3f, n::Vec3f, dpdus::Vec3f, ::TextureFilterContext, ::Wavelengths,
    ::Bool = false,
)
    bsdf.total > 0f0 || return (SpectralRadiance(0f0), 0f0)
    f, pdf = mixture(bsdf.lobes, wo, wi, n, bsdf.lambda)
    return (f, pdf / bsdf.total)
end

@propagate_inbounds function sample_bsdf_spectral(
    bsdf::LobeBSDF, ::RGBToSpectrumTable, textures,
    wo::Vec3f, n::Vec3f, dpdus::Vec3f, ::TextureFilterContext,
    ::Wavelengths, sample_u::Point2f, rng::Float32,
    ::Bool = false,
)
    total = bsdf.total
    total > 0f0 || return SpectralBSDFSample()
    wi, f_singular, pdf_singular, flags, eta, sample_weight, cn, terminates =
        pick_sample(bsdf.lobes, rng * total, 0f0, wo, sample_u, bsdf.lambda)
    flags == 0x00 && return SpectralBSDFSample()
    side = dot(wo, n) * dot(wi, n) < 0f0 ? BXDF_TRANSMISSION : BXDF_REFLECTION
    flags = (flags & ~(BXDF_TRANSMISSION | BXDF_REFLECTION)) | side
    if is_specular(flags)
        cos_s = abs(dot(wi, n))
        cos_s > 1f-7 || return SpectralBSDFSample()
        return SpectralBSDFSample(f_singular * (abs(dot(wi, cn)) / cos_s), wi, pdf_singular * sample_weight / total,
                                  flags, eta, false, bsdf.dispersive && terminates)
    end
    f, pdf = mixture(bsdf.lobes, wo, wi, n, bsdf.lambda)
    pdf > 0f0 || return SpectralBSDFSample()
    return SpectralBSDFSample(f, wi, pdf / total, flags, eta, false, bsdf.dispersive && terminates)
end

# ── Microfacet ──────────────────────────────────────────────────────────────

"""How a microfacet lobe's reflectance and transmittance depend on the angle."""
abstract type FresnelModel end

"""No Fresnel: the lobe reflects and refracts all, its colour in its weight. `color` is what its energy compensation darkens towards."""
struct NoFresnel{C} <: FresnelModel
    color::C
end

"""Cycles' F82 tint, Gulbrandsen's artist-friendly conductor: `f0` at normal incidence, `tint` at about 82°, optionally under a thin film."""
struct F82Tint{C} <: FresnelModel
    f0::C
    tint::C
    film_thickness::Float32
    film_ior::Float32
end

"""The real dielectric Fresnel at the lobe's IOR, tinting what it reflects and what it lets through."""
struct DielectricFresnel{C} <: FresnelModel
    reflection_tint::C
    transmission_tint::C
end

"""Cycles' generalized Schlick: the dielectric Fresnel's shape from `f0` up, optionally under a thin film."""
struct SchlickFresnel{C} <: FresnelModel
    f0::C
    reflection_tint::C
    transmission_tint::C
    film_thickness::Float32
    film_ior::Float32
end

@inline lift(f::NoFresnel{Vec3f}, table, lambda) = NoFresnel(_uplift(table, _saturate3(f.color), lambda))
@inline lift(f::F82Tint{Vec3f}, table, lambda) =
    F82Tint(_uplift(table, _saturate3(f.f0), lambda), _uplift(table, _saturate3(f.tint), lambda), f.film_thickness,
            f.film_ior)
@inline lift(f::DielectricFresnel{Vec3f}, table, lambda) =
    DielectricFresnel(_uplift(table, _saturate3(f.reflection_tint), lambda), _uplift(table, _saturate3(f.transmission_tint), lambda))
@inline lift(f::SchlickFresnel{Vec3f}, table, lambda) =
    SchlickFresnel(_uplift(table, _saturate3(f.f0), lambda), _uplift(table, _saturate3(f.reflection_tint), lambda),
                   _uplift(table, _saturate3(f.transmission_tint), lambda), f.film_thickness, f.film_ior)

"""
A microfacet lobe's reflectance for one channel or wavelength with its thin film: the film's own
interference over the substrate, a conductor for the F82 tint, and for a Principled specular layer
the scaling Cycles applies towards `f0`.
"""
@inline function film_reflectance(::Type{<:F82Tint}, ior::Float32, thickness::Float32, film_ior::Float32, cos_i::Float32,
                                  f0::Float32, tint::Float32, lambda::Float32)
    n, k = _conductor_nk(f0, _f82_b_channel(f0, tint))
    return _thin_film_reflectance(cos_i, thickness, film_ior, n, k, lambda)
end

@inline function film_reflectance(::Type{<:SchlickFresnel}, ior::Float32, thickness::Float32, film_ior::Float32,
                                  cos_i::Float32, f0::Float32, tint::Float32, lambda::Float32)
    F = _thin_film_reflectance(cos_i, thickness, film_ior, ior, 0f0, lambda)
    f0_real = _f0_from_ior(ior)
    (f0_real > 1f-5 && F != 1f0) || return F
    s = clamp((F - 1f0) / (f0_real - 1f0), 0f0, 1f0)
    return F * (1f0 + (f0 / f0_real - 1f0) * s)
end

"""
A conductor's Fresnel from its complex IOR, `eta` and `k` at each wavelength: pbrt-v4's
`FrComplex`. Built at the path's wavelengths (measured spectra have no RGB).
"""
struct ConductorFresnel <: FresnelModel
    eta::SpectralRadiance
    k::SpectralRadiance
end

@inline has_film(::FresnelModel) = false
@inline has_film(f::Union{F82Tint, SchlickFresnel}) = f.film_thickness > THIN_FILM_THICKNESS_CUTOFF

"""The reflectance and transmittance at the cosine `cos_i` between the microfacet normal and `wo`, at the path's wavelengths."""
@inline fresnel_rt(::NoFresnel{SpectralRadiance}, cos_i::Float32, ior::Float32, lambda::Wavelengths) =
    (SpectralRadiance(1f0), SpectralRadiance(1f0))

@inline fresnel_rt(f::ConductorFresnel, cos_i::Float32, ior::Float32, lambda::Wavelengths) =
    (fr_complex_spectral(abs(cos_i), f.eta, f.k), SpectralRadiance(0f0))

@inline function fresnel_rt(f::F82Tint{SpectralRadiance}, cos_i::Float32, ior::Float32, lambda::Wavelengths)
    if has_film(f)
        F = SpectralRadiance(ntuple(k -> film_reflectance(F82Tint, ior, f.film_thickness, f.film_ior, cos_i,
                                                          f.f0.data[k], f.tint.data[k], lambda.lambda[k]), Val(4)))
        return (F, SpectralRadiance(0f0))
    end
    one = SpectralRadiance(1f0)
    s = 6f0 / 7f0
    s5 = s^5
    b = (f.f0 + (one - f.f0) * s5) * (one - f.tint) * (7f0 / (s5 * s))
    m = clamp(1f0 - cos_i, 0f0, 1f0)
    m5 = m^5
    return (clamp(f.f0 + (one - f.f0) * m5 - b * (cos_i * m5 * m), 0f0, 1f0), SpectralRadiance(0f0))
end

@inline function fresnel_rt(f::DielectricFresnel{SpectralRadiance}, cos_i::Float32, ior::Float32, lambda::Wavelengths)
    F = fresnel_dielectric(cos_i, ior)
    return (f.reflection_tint * F, f.transmission_tint * (1f0 - F))
end

@inline function fresnel_rt(f::SchlickFresnel{SpectralRadiance}, cos_i::Float32, ior::Float32, lambda::Wavelengths)
    one = SpectralRadiance(1f0)
    if has_film(f)
        F = SpectralRadiance(ntuple(k -> film_reflectance(SchlickFresnel, ior, f.film_thickness, f.film_ior, cos_i,
                                                          f.f0.data[k], f.reflection_tint.data[k], lambda.lambda[k]), Val(4)))
        return (f.reflection_tint * F, f.transmission_tint * (one - F))
    end
    F_real = fresnel_dielectric(cos_i, ior)
    f0_real = _f0_from_ior(ior)
    s = clamp((F_real - f0_real) / max(1f0 - f0_real, 1f-6), 0f0, 1f0)
    F = f.f0 + (one - f.f0) * s
    return (f.reflection_tint * F, f.transmission_tint * (one - F))
end

"""
Where Cycles' GGX and pbrt-v4's differ, which a microfacet lobe follows: below which roughness it is
a mirror, and whether refraction scales the radiance it carries by 1/η². Cycles' (`CyclesGGX`) is
singular at `αx·αy ≤ 2e-10` and does not scale. pbrt's is singular below `max(αx, αy) = 1e-3`
(`EffectivelySmooth`) and scales the radiance a path carries (`PbrtRadiance`) but not importance
(`PbrtImportance`), which pbrt's layered BSDF transports on its way back.
"""
abstract type GGXConvention end
struct CyclesGGX <: GGXConvention end
struct PbrtRadiance <: GGXConvention end
struct PbrtImportance <: GGXConvention end

"""
A GGX microfacet lobe, as Cycles' `bsdf_microfacet`: reflection only (`Mode` `:reflection`),
reflection and refraction (`:glass`), or the transmission of a thin glass, a reflection mirrored to
the far side (`:thin`). `energy` scales single scattering up by what it misses, Cycles'
multiscatter GGX, and is one for pbrt's single-scattering one; `inv_abbe` disperses the IOR over the
wavelengths. `C` is the `GGXConvention` it follows.
"""
struct MicrofacetLobe{Mode, F <: FresnelModel, C <: GGXConvention} <: BSDFLobe
    alpha_x::Float32
    alpha_y::Float32
    ior::Float32
    energy::Float32
    inv_abbe::Float32
    fresnel::F
end

MicrofacetLobe{Mode}(fresnel::FresnelModel; alpha_x::Float32, alpha_y::Float32 = alpha_x, ior::Float32 = 1f0,
                     inv_abbe::Float32 = 0f0, convention::GGXConvention = CyclesGGX()) where {Mode} =
    MicrofacetLobe{Mode, typeof(fresnel), typeof(convention)}(alpha_x, alpha_y, ior, 1f0, inv_abbe, fresnel)

with_energy(l::MicrofacetLobe{Mode, F, C}, energy::Float32) where {Mode, F, C} =
    MicrofacetLobe{Mode, F, C}(l.alpha_x, l.alpha_y, l.ior, energy, l.inv_abbe, l.fresnel)

@inline function lift(l::MicrofacetLobe{Mode, F, C}, table, lambda::Wavelengths, regularize::Bool) where {Mode, F, C}
    ax = regularize ? regularize_alpha(l.alpha_x) : l.alpha_x
    ay = regularize ? regularize_alpha(l.alpha_y) : l.alpha_y
    ior = l.inv_abbe > 0f0 ? _dispersed_ior(l.ior, l.inv_abbe, lambda.lambda[1]) : l.ior
    fresnel = lift(l.fresnel, table, lambda)
    return MicrofacetLobe{Mode, typeof(fresnel), C}(ax, ay, ior, l.energy, l.inv_abbe, fresnel)
end

@inline disperses(l::MicrofacetLobe) = l.inv_abbe > 0f0
@inline singular(l::MicrofacetLobe{<:Any, <:FresnelModel, CyclesGGX}) = l.alpha_x * l.alpha_y <= 2f-10
@inline singular(l::MicrofacetLobe) = trowbridge_reitz_effectively_smooth(l.alpha_x, l.alpha_y)
# pbrt's glass of no IOR contrast is a straight pass-through, at any roughness.
@inline singular(l::MicrofacetLobe{:glass, <:FresnelModel, <:Union{PbrtRadiance, PbrtImportance}}) =
    l.ior == 1f0 || trowbridge_reitz_effectively_smooth(l.alpha_x, l.alpha_y)
"""What refraction scales the light a lobe carries by: radiance by 1/η², under pbrt's convention."""
@inline refraction_scale(l::MicrofacetLobe{<:Any, <:FresnelModel, PbrtRadiance}) = 1f0 / l.ior^2
@inline refraction_scale(::MicrofacetLobe) = 1f0
@inline terminates_secondary(::MicrofacetLobe{<:Any, <:FresnelModel, <:Union{PbrtRadiance, PbrtImportance}}, ::UInt8) = true
@inline ends_dispersion_at_vertex(::MicrofacetLobe{<:Any, <:FresnelModel, <:Union{PbrtRadiance, PbrtImportance}}) = true

"""
What a sample of a microfacet lobe may take under `s` of the reflection and the refraction it
computes, pbrt's `pr` and `pt`: a glass by its reflectance and transmittance at the microfacet; a
reflection lobe its reflection; a thin glass the reflection it mirrors through the surface.
"""
@inline microfacet_choice(::MicrofacetLobe{:glass}, R, T, s::Scattering) =
    (allows_reflection(s) ? average(R) : 0f0, allows_transmission(s) ? average(T) : 0f0)
@inline microfacet_choice(::MicrofacetLobe{:reflection}, R, T, s::Scattering) = (allows_reflection(s) ? 1f0 : 0f0, 0f0)
@inline microfacet_choice(::MicrofacetLobe{:thin}, R, T, s::Scattering) = (allows_transmission(s) ? 1f0 : 0f0, 0f0)

@inline can_scatter(::MicrofacetLobe{:glass}, ::Scattering) = true
@inline can_scatter(::MicrofacetLobe{:thin}, s::Scattering) = allows_transmission(s)

@inline other_side(l::MicrofacetLobe{:glass, F, C}) where {F <: FresnelModel, C <: GGXConvention} =
    MicrofacetLobe{:glass, F, C}(l.alpha_x, l.alpha_y, 1f0 / l.ior, l.energy, l.inv_abbe, l.fresnel)
@inline adjoint(l::MicrofacetLobe{Mode, F, PbrtRadiance}) where {Mode, F <: FresnelModel} =
    MicrofacetLobe{Mode, F, PbrtImportance}(l.alpha_x, l.alpha_y, l.ior, l.energy, l.inv_abbe, l.fresnel)

@inline guide_albedo(::MicrofacetLobe{:reflection}, weight::Vec3f) = Vec3f(0)
@inline guide_albedo(l::MicrofacetLobe{:reflection, <:F82Tint}, weight::Vec3f) = weight .* l.fresnel.f0
@inline guide_albedo(::MicrofacetLobe{:reflection, <:NoFresnel}, weight::Vec3f) = weight

@inline function lobe_f(l::MicrofacetLobe{Mode}, wo::Vec3f, wi::Vec3f, lambda::Wavelengths,
                        s::Scattering = AnyScattering()) where {Mode}
    zero = (SpectralRadiance(0f0), 0f0)
    (wo[3] > 0f0 && !singular(l)) || return zero
    if Mode === :thin
        wi[3] < 0f0 || return zero
        wi = Vec3f(wi[1], wi[2], -wi[3])
    end
    transmission = wi[3] < 0f0
    (transmission && Mode !== :glass) && return zero
    H = transmission ? -(l.ior * wi + wo) : wo + wi
    len_h = norm(H)
    len_h > 0f0 || return zero
    H = H / len_h
    H[3] < 0f0 && (H = -H)
    cos_hi = dot(H, wo)
    (transmission && (cos_hi <= 0f0 || dot(H, wi) >= 0f0)) && return zero
    R, T = fresnel_rt(l.fresnel, cos_hi, l.ior, lambda)
    average(R) + average(T) > 0f0 || return zero
    pr, pt = microfacet_choice(l, R, T, s)
    lobe_pdf = pr + pt > 0f0 ? (transmission ? pt : pr) / (pr + pt) : 0f0
    D = trowbridge_reitz_d(H, l.alpha_x, l.alpha_y)
    lambda_o = trowbridge_reitz_lambda(wo, l.alpha_x, l.alpha_y)
    lambda_i = trowbridge_reitz_lambda(wi, l.alpha_x, l.alpha_y)
    common = D / wo[3] * (transmission ? (l.ior / len_h)^2 * abs(cos_hi * dot(H, wi)) : 0.25f0)
    pdf = common * lobe_pdf / (1f0 + lambda_o)
    f = (transmission ? T * refraction_scale(l) : R) * (common / (1f0 + lambda_o + lambda_i) * l.energy / abs(wi[3]))
    return (f, pdf)
end

"""Cycles' `bsdf_microfacet_sample`; a thin glass's as the reflection it is mirrored from."""
@inline function lobe_sample(l::MicrofacetLobe{Mode}, wo::Vec3f, u::Point2f, uc::Float32, lambda::Wavelengths,
                             s::Scattering = AnyScattering()) where {Mode}
    wo[3] > 0f0 || return NO_LOBE_SAMPLE
    is_singular = singular(l)
    H = is_singular ? Vec3f(0, 0, 1) : trowbridge_reitz_sample_wm(wo, u, l.alpha_x, l.alpha_y)
    cos_hi = dot(H, wo)
    R, T = fresnel_rt(l.fresnel, cos_hi, l.ior, lambda)
    average(R) + average(T) > 0f0 || return NO_LOBE_SAMPLE
    pr, pt = microfacet_choice(l, R, T, s)
    pr + pt > 0f0 || return NO_LOBE_SAMPLE
    pdf_reflect = pr / (pr + pt)
    refract = uc >= pdf_reflect
    if refract
        sin2_t = (1f0 - cos_hi^2) / l.ior^2
        sin2_t >= 1f0 && return NO_LOBE_SAMPLE
        cos_t = sqrt(1f0 - sin2_t)
        wi = -wo / l.ior + (cos_hi / l.ior - cos_t) * H
    else
        wi = 2f0 * cos_hi * H - wo
    end
    (wi[3] < 0f0) == refract || return NO_LOBE_SAMPLE
    is_singular = is_singular || (refract && abs(l.ior - 1f0) < 1f-4)
    eta = refract ? l.ior : 1f0
    out = Mode === :thin ? Vec3f(wi[1], wi[2], -wi[3]) : wi
    if is_singular
        coeff = refract ? T * refraction_scale(l) : R
        lobe_pdf = refract ? 1f0 - pdf_reflect : pdf_reflect
        flags = refract ? BXDF_SPECULAR_TRANSMISSION : BXDF_SPECULAR_REFLECTION
        return (out, coeff * (l.energy / max(abs(wi[3]), 1f-6)), lobe_pdf, flags, eta)
    end
    return (out, SpectralRadiance(0f0), 0f0, refract ? BXDF_GLOSSY_TRANSMISSION : BXDF_GLOSSY_REFLECTION, eta)
end

# The set-up of a microfacet lobe in RGB, as Cycles' `bsdf_microfacet_setup_fresnel_*` do it.

@inline rough_index(l::MicrofacetLobe) = sqrt(sqrt(l.alpha_x * l.alpha_y))

"""
Cycles' `bsdf_microfacet_estimate_albedo`, reflection and transmission: what the lobe returns of
light from `wo`, from the generalized Schlick tables, or from the Fresnel at the view angle for a thin film.
"""
@inline function estimate_albedo(lut, l::MicrofacetLobe{Mode}, mu::Float32) where {Mode}
    glass = Mode === :glass
    estimate(lut, l.fresnel, l, mu, glass)
end

@inline function estimate(lut, f::Union{F82Tint{Vec3f}, SchlickFresnel{Vec3f}}, l::MicrofacetLobe, mu::Float32, glass::Bool)
    if has_film(f)
        tint = f isa F82Tint ? f.tint : f.reflection_tint
        F = Vec3f(film_reflectance(typeof(f), l.ior, f.film_thickness, f.film_ior, mu, f.f0[1], tint[1], 610f0),
                  film_reflectance(typeof(f), l.ior, f.film_thickness, f.film_ior, mu, f.f0[2], tint[2], 550f0),
                  film_reflectance(typeof(f), l.ior, f.film_thickness, f.film_ior, mu, f.f0[3], tint[3], 465f0))
        transmission = f isa SchlickFresnel && glass ? f.transmission_tint .* (1f0 .- F) : Vec3f(0)
        return (tint .* F, transmission)
    end
    return estimate_without_film(lut, f, l, mu, glass)
end
@inline estimate(lut, f::FresnelModel, l::MicrofacetLobe, mu::Float32, glass::Bool) = estimate_without_film(lut, f, l, mu, glass)

@inline function estimate_without_film(lut, f::SchlickFresnel{Vec3f}, l::MicrofacetLobe, mu::Float32, glass::Bool)
    F = schlick_estimate(lut, f.f0, l, mu)
    return (f.reflection_tint .* F, glass ? f.transmission_tint .* (1f0 .- F) : Vec3f(0))
end
@inline function estimate_without_film(lut, f::DielectricFresnel{Vec3f}, l::MicrofacetLobe, mu::Float32, glass::Bool)
    F = l.ior > 1f0 ? schlick_estimate(lut, Vec3f(_f0_from_ior(l.ior)), l, mu) : Vec3f(fresnel_dielectric(mu, l.ior))
    return (f.reflection_tint .* F, glass ? f.transmission_tint .* (1f0 .- F) : Vec3f(0))
end
@inline function estimate_without_film(lut, f::F82Tint{Vec3f}, l::MicrofacetLobe, mu::Float32, glass::Bool)
    s = lookup_table_read_3d(lut, rough_index(l), mu, 0.5f0, Int32(LookupTable.GGX_GEN_SCHLICK_S), Int32(16), Int32(16), Int32(16))
    return (f.f0 .+ (1f0 .- f.f0) .* s, Vec3f(0))
end
@inline estimate_without_film(lut, ::NoFresnel{Vec3f}, l::MicrofacetLobe, mu::Float32, glass::Bool) =
    (Vec3f(1), glass ? Vec3f(1) : Vec3f(0))

@inline function schlick_estimate(lut, f0::Vec3f, l::MicrofacetLobe, mu::Float32)
    s = lookup_table_read_3d(lut, rough_index(l), mu, _ior_index(l.ior), Int32(LookupTable.GGX_GEN_SCHLICK_IOR_S),
                             Int32(16), Int32(16), Int32(16))
    return f0 .+ (1f0 .- f0) .* s
end

"""The Fresnel averaged over the hemisphere that the light scattering more than once sees, which darkens the energy compensation."""
@inline single_scatter_fresnel(f::F82Tint{Vec3f}, l::MicrofacetLobe, glass::Bool) = _f82_fss(f.f0, _f82_b(f.f0, f.tint))
@inline single_scatter_fresnel(f::SchlickFresnel{Vec3f}, l::MicrofacetLobe, glass::Bool) =
    glass ? f.transmission_tint :
    f.reflection_tint .* (f.f0 .+ (1f0 .- f.f0) .*
                          clamp((_dielectric_fss(l.ior) - _f0_from_ior(l.ior)) / (1f0 - _f0_from_ior(l.ior)), 0f0, 1f0))
@inline single_scatter_fresnel(::DielectricFresnel{Vec3f}, l::MicrofacetLobe, glass::Bool) = Vec3f(_dielectric_fss(l.ior))
@inline single_scatter_fresnel(f::NoFresnel{Vec3f}, l::MicrofacetLobe, glass::Bool) = f.color

"""
Cycles' `microfacet_ggx_preserve_energy`: scale the lobe up by what single scattering misses at
the view angle, and darken it by the Fresnel of the light that scatters more than once.
"""
@inline function preserve_energy(lut, w::Weighted{<:MicrofacetLobe{Mode}, Vec3f}, mu::Float32, fss::Vec3f) where {Mode}
    l = w.lobe
    rough = rough_index(l)
    if Mode === :glass
        inverse = l.ior < 1f0
        z = _ior_index(inverse ? 1f0 / l.ior : l.ior)
        e_offset = Int32(inverse ? LookupTable.GGX_GLASS_INV_E : LookupTable.GGX_GLASS_E)
        avg_offset = Int32(inverse ? LookupTable.GGX_GLASS_INV_EAVG : LookupTable.GGX_GLASS_EAVG)
        E = lookup_table_read_3d(lut, rough, mu, z, e_offset, Int32(16), Int32(16), Int32(16))
        E_avg = lookup_table_read_2d(lut, rough, z, avg_offset, Int32(16), Int32(16))
    else
        E = lookup_table_read_2d(lut, rough, mu, Int32(LookupTable.GGX_E), Int32(32), Int32(32))
        E_avg = lookup_table_read(lut, rough, Int32(LookupTable.GGX_EAVG), Int32(32))
    end
    missing = (1f0 - E) / max(E, 1f-6)
    energy = 1f0 + missing
    weight, sample_weight = w.weight, w.sample_weight
    if fss != Vec3f(1)
        fms = fss .* E_avg ./ (1f0 .- fss .* (1f0 - E_avg))
        darkening = (1f0 .+ fms .* missing) ./ energy
        weight = weight .* darkening
        sample_weight *= _avg(darkening)
    end
    return Weighted(with_energy(l, energy), weight, sample_weight, w.n, w.t)
end

"""
A microfacet lobe set up the way Cycles' `bsdf_microfacet_setup_fresnel_*` do: its share of the
samples scaled by its albedo seen from `wo`, at the cosine `mu` with its normal, and its energy
compensation; and what it reflects, which the layers under it give up.
"""
@inline function setup_fresnel(lut, w::Weighted{<:MicrofacetLobe{Mode}, Vec3f}, mu::Float32) where {Mode}
    refl, trans = estimate_albedo(lut, w.lobe, mu)
    w = Weighted(w.lobe, w.weight, w.sample_weight * _avg(refl .+ trans), w.n, w.t)
    return preserve_energy(lut, w, mu, single_scatter_fresnel(w.lobe.fresnel, w.lobe, Mode === :glass)), refl
end

# ── Thin dielectric ─────────────────────────────────────────────────────────

"""
pbrt-v4's `ThinDielectricBxDF`: a slab of no thickness, which reflects what its two interfaces and
the light between them reflect and lets the rest through unbent. Singular only.
"""
struct ThinDielectricLobe <: BSDFLobe
    eta::Float32
end

@inline ends_dispersion_at_vertex(::ThinDielectricLobe) = true

@inline lobe_f(::ThinDielectricLobe, wo::Vec3f, wi::Vec3f, ::Wavelengths) = (SpectralRadiance(0f0), 0f0)
@inline can_scatter(::ThinDielectricLobe, ::Scattering) = true

@inline function lobe_sample(l::ThinDielectricLobe, wo::Vec3f, u::Point2f, uc::Float32, ::Wavelengths)
    wo[3] > 0f0 || return NO_LOBE_SAMPLE
    R = fresnel_dielectric(wo[3], l.eta)
    T = 1f0 - R
    if R < 1f0
        R += T^2 * R / (1f0 - R^2)
        T = 1f0 - R
    end
    uc < R / (R + T) && return (Vec3f(-wo[1], -wo[2], wo[3]), SpectralRadiance(R / wo[3]), R / (R + T),
                                BXDF_SPECULAR_REFLECTION, 1f0)
    return (-wo, SpectralRadiance(T / wo[3]), T / (R + T), BXDF_SPECULAR_TRANSMISSION, 1f0)
end

# ── Sheen ───────────────────────────────────────────────────────────────────

"""Cycles' microfiber sheen, a linearly transformed cosine of coefficients `a`, `b`, in the frame that holds `wo` in its xz plane."""
struct SheenLobe <: BSDFLobe
    a::Float32
    b::Float32
end

@inline lift(l::SheenLobe, table, lambda, ::Bool) = l
@inline is_diffuse_lobe(::SheenLobe) = true
@inline function lobe_frame(::SheenLobe, n::Vec3f, t::Vec3f, wo::Vec3f)
    tangent = _unit(wo - dot(wo, n) * n, coordinate_system(n)[1])
    return n, tangent, cross(n, tangent)
end

@inline function sheen_value(l::SheenLobe, w::Vec3f)
    len2 = (l.a * w[1] + l.b * w[3])^2 + (l.a * w[2])^2 + w[3]^2
    return (1f0 / Float32(π)) * max(w[3], 0f0) * (l.a / len2)^2
end

@inline function lobe_f(l::SheenLobe, wo::Vec3f, wi::Vec3f, ::Wavelengths)
    wi[3] > 0f0 || return (SpectralRadiance(0f0), 0f0)
    val = sheen_value(l, wi)
    return (SpectralRadiance(val / wi[3]), val)
end

@inline function lobe_sample(l::SheenLobe, wo::Vec3f, u::Point2f, ::Float32, ::Wavelengths)
    r = sqrt(u[1])
    s, co = sincos(2f0 * Float32(π) * u[2])
    dx, dy = r * co, r * s
    dz = sqrt(max(1f0 - dx^2 - dy^2, 0f0))
    return (normalize(Vec3f(dx - dz * l.b, dy, dz * l.a)), SpectralRadiance(0f0), 0f0, BXDF_DIFFUSE_REFLECTION, 1f0)
end

"""Cycles' `bsdf_sheen_setup`: the LTC fit for the view angle and its albedo, which scales the lobe; none where the fit degenerates."""
@inline function sheen(lut, weight::Vec3f, n::Vec3f, roughness::Float32, wo::Vec3f)
    roughness = clamp(roughness, 1f-3, 1f0)
    mu = dot(n, wo)
    offset = Int32(LookupTable.SHEEN_LTC)
    a = lookup_table_read_2d(lut, mu, roughness, offset, Int32(32), Int32(32))
    b = lookup_table_read_2d(lut, mu, roughness, offset + Int32(1024), Int32(32), Int32(32))
    albedo = lookup_table_read_2d(lut, mu, roughness, offset + Int32(2048), Int32(32), Int32(32))
    (abs(a) < 1f-5 || albedo < 1f-5) && return Weighted(SheenLobe(1f0, 0f0), Vec3f(0), 0f0, n, Vec3f(0))
    return Weighted(SheenLobe(a, b), weight * albedo, n)
end

# ── Subsurface ──────────────────────────────────────────────────────────────

"""
Cycles' random walk subsurface closure, as `bssrdf_setup` makes it: the albedo, the mean free path
per wavelength, the anisotropy, and the GGX roughness and IOR the walk enters by. It scatters
nothing itself: `subsurface_scatter` walks it, and where the walk leaves the object it is the
diffuse lobe it `exited` as.
"""
struct SubsurfaceLobe{C} <: BSDFLobe
    albedo::C
    radius::C
    alpha::Float32
    ior::Float32
    anisotropy::Float32
    exited::Bool
end

SubsurfaceLobe(albedo::Vec3f, radius::Vec3f, alpha::Float32, ior::Float32, anisotropy::Float32) =
    SubsurfaceLobe{Vec3f}(albedo, radius, alpha, clamp(ior, 1.01f0, 3.8f0), clamp(anisotropy, -0.99f0, 0.99f0), false)

@inline lift(l::SubsurfaceLobe{Vec3f}, table, lambda, ::Bool) =
    SubsurfaceLobe{SpectralRadiance}(_uplift(table, _saturate3(l.albedo), lambda), _uplift_unbounded(table, l.radius, lambda),
                                     l.alpha, l.ior, l.anisotropy, false)
@inline is_diffuse_lobe(::SubsurfaceLobe) = true

@inline function lobe_f(l::SubsurfaceLobe, wo::Vec3f, wi::Vec3f, ::Wavelengths)
    (l.exited && wi[3] > 0f0) || return (SpectralRadiance(0f0), 0f0)
    return (SpectralRadiance(1f0 / Float32(π)), wi[3] / Float32(π))
end

@inline lobe_sample(l::SubsurfaceLobe, wo::Vec3f, u::Point2f, ::Float32, ::Wavelengths) =
    l.exited ? (cosine_sample_hemisphere(u), SpectralRadiance(0f0), 0f0, BXDF_DIFFUSE_REFLECTION, 1f0) : NO_LOBE_SAMPLE
