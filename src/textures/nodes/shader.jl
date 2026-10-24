# Ported from Blender Cycles v5.2.1: intern/cycles/kernel/svm/closure.h, displace.h and
# kernel/closure/bsdf_sheen.h, bsdf_oren_nayar.h, copyright 2011-2022 Blender Foundation,
# SPDX-License-Identifier: Apache-2.0; and kernel/closure/bsdf_microfacet.h, bsdf_util.h, adapted
# by Cycles from Open Shading Language, copyright 2009-2010 Sony Pictures Imageworks Inc., et al.
# All Rights Reserved, copyright 2011-2022 Blender Foundation, SPDX-License-Identifier: BSD-3-Clause.
# Modified: rewritten in Julia; the thin film is evaluated per wavelength by the Airy sum. See
# LICENSES/Apache-2.0.txt and LICENSES/BSD-3-Clause.txt.

# ============================================================================
# Shader programs: node programs that end in closures, as Cycles' SVM does
# ============================================================================
#
# A shader graph's BSDF nodes, Mix Shader and Add Shader are flattened on the
# host into one closure node per BSDF, each with the product of the mix factors
# above it as its weight. The program computes those weights and every input
# like any other value on the stack, and each closure instruction sets up what
# it describes the way Cycles' `svm_node_closure_bsdf` does: in RGB, with the
# energy compensation and layering from Cycles' lookup tables. The material
# lifts the result to the path's wavelengths and samples it as one BSDF.

module ClosureKind
const DIFFUSE     = UInt8(1)
const GLOSSY      = UInt8(2)
const METALLIC    = UInt8(3)
const TRANSLUCENT = UInt8(4)
const GLASS       = UInt8(5)
const SHEEN       = UInt8(6)
const TRANSPARENT = UInt8(7)
const EMISSION    = UInt8(8)
end

"""The lobes a closure list holds."""
module Lobe
const DIFFUSE     = UInt8(1)
const OREN_NAYAR  = UInt8(2)
const TRANSLUCENT = UInt8(3)
const MICROFACET  = UInt8(4)
const GLASS       = UInt8(5)
const SHEEN       = UInt8(6)
const THIN_GLASS  = UInt8(7)
const SUBSURFACE  = UInt8(8)
end

"""How a microfacet lobe's reflectance depends on the angle."""
module Fresnel
const NONE        = UInt8(0)
const F82_TINT    = UInt8(1)
const SCHLICK     = UInt8(2)
const DIELECTRIC  = UInt8(3)
end

"""The most closures a shader's list holds; a shader holds as many as its graph can produce, up to this."""
const MAX_CLOSURES = 8
const CLOSURE_WEIGHT_CUTOFF = 1f-5
const THIN_FILM_THICKNESS_CUTOFF = 0.1f0

"""
What a shader program reads besides the texture inputs: where the hit is seen from, the
position's screen-space differentials (for bump), the shading tangent and the geometric normal,
both facing outwards.
"""
struct ShaderContext
    tfc::TextureFilterContext
    wo::Vec3f
    dpdx::Vec3f
    dpdy::Vec3f
    dpdus::Vec3f
    ng::Vec3f
end

"""
One closure in RGB, set up. `weight` includes the darkening of the energy compensation, which
`energy` scales the single-scattering lobe up to. The colour slots depend on the lobe:

* microfacet and glass: `c1` is the reflectance at normal incidence, `c2` the reflection tint
  (`F82_TINT`: the edge tint), `c3` the transmission tint;
* Oren-Nayar: `c1` is the multiple-scattering term;

`p1`, `p2` are the sheen's LTC coefficients or Oren-Nayar's `a`, `b`, and for a microfacet lobe the
thin film's thickness and IOR. `t` is the tangent of an anisotropic lobe, zero for an isotropic one.
"""
struct RGBClosure
    lobe::UInt8
    fresnel::UInt8
    weight::Vec3f
    c1::Vec3f
    c2::Vec3f
    c3::Vec3f
    n::Vec3f
    t::Vec3f
    alpha_x::Float32
    alpha_y::Float32
    ior::Float32
    energy::Float32
    p1::Float32
    p2::Float32
    inv_abbe::Float32
    sample_weight::Float32
end

RGBClosure() = RGBClosure(0x00, 0x00, Vec3f(0), Vec3f(0), Vec3f(0), Vec3f(0), Vec3f(0, 0, 1), Vec3f(0),
                          0f0, 0f0, 1f0, 1f0, 0f0, 0f0, 0f0, 0f0)

_closure(lobe, weight, n; fresnel = Fresnel.NONE, c1 = Vec3f(1), c2 = Vec3f(1), c3 = Vec3f(1), t = Vec3f(0),
         alpha_x = 0f0, alpha_y = alpha_x, ior = 1f0, energy = 1f0, p1 = 0f0, p2 = 0f0, inv_abbe = 0f0,
         sample_weight = _avg(weight)) =
    RGBClosure(lobe, fresnel, weight, c1, c2, c3, n, t, alpha_x, alpha_y, ior, energy, p1, p2, inv_abbe, sample_weight)

"""The closures of a hit, the light it emits and how much of it is see-through."""
struct ClosureList{N}
    count::Int32
    closures::NTuple{N, RGBClosure}
    emission::Vec3f
    transparency::Vec3f
end

ClosureList{N}() where {N} = ClosureList(Int32(0), ntuple(_ -> RGBClosure(), Val(N)), Vec3f(0), Vec3f(0))

# Written and read by selects: a runtime index into a tuple is refused by the SPIR-V emitter.
@noinline function _push(cl::ClosureList{N}, c::RGBClosure) where {N}
    (cl.count >= N || _max(c.weight) <= CLOSURE_WEIGHT_CUTOFF || !(c.sample_weight > 0f0)) && return cl
    n = cl.count + Int32(1)
    return ClosureList(n, ntuple(k -> ifelse(Int32(k) == n, c, cl.closures[k]), Val(N)), cl.emission,
                       cl.transparency)
end

_with_emission(cl::ClosureList, e::Vec3f) = ClosureList(cl.count, cl.closures, cl.emission + e, cl.transparency)
_with_transparency(cl::ClosureList, t::Vec3f) = ClosureList(cl.count, cl.closures, cl.emission, cl.transparency + t)

@inline function _closure_at(closures::NTuple{N, T}, i::Int32) where {N, T}
    c = closures[1]
    Base.Cartesian.@nexprs 8 k -> (k <= N && (c = ifelse(i == Int32(k), closures[min(k, N)], c)))
    return c
end

@inline _max(v::Vec3f) = max(v[1], v[2], v[3])
@inline _pow3(v::Vec3f, e::Float32) = Vec3f(v[1]^e, v[2]^e, v[3]^e)

"""The specular layer's reflectance at normal incidence and IOR, as the Principled BSDF's `specular_ior_level` sets them."""
@inline function _specular_f0(ior::Float32, level::Float32)
    level = max(level, 0f0)
    f0 = _f0_from_ior(ior)
    return level == 0.5f0 ? f0 : f0 * 2f0 * level
end
@inline function _specular_eta(ior::Float32, level::Float32)
    max(level, 0f0) == 0.5f0 && return ior
    eta = _ior_from_f0(_specular_f0(ior, level))
    return ior < 1f0 ? 1f0 / eta : eta
end

"""
Cycles' random walk subsurface closure, as `bssrdf_setup` makes it: `c1` the albedo, `c2` the mean
free path per channel, `p1` the anisotropy, with the GGX entry roughness and IOR it refracts by.
"""
@inline _subsurface(weight::Vec3f, albedo::Vec3f, radius::Vec3f, n::Vec3f, alpha::Float32, ior::Float32,
                    anisotropy::Float32) =
    _closure(Lobe.SUBSURFACE, weight, n; c1 = albedo, c2 = radius, alpha_x = alpha, ior = clamp(ior, 1.01f0, 3.8f0),
             p1 = clamp(anisotropy, -0.99f0, 0.99f0))

"""Cycles' `closure_layering_weight`: what a layer of `albedo` leaves of `weight` for the layers under it."""
@inline function _layering(albedo::Vec3f, weight::Vec3f)
    ratio = Vec3f(ntuple(k -> weight[k] != 0f0 ? albedo[k] / weight[k] : 0f0, 3))
    return weight * clamp(1f0 - _max(ratio), 0f0, 1f0)
end
@inline _avg(v::Vec3f) = (v[1] + v[2] + v[3]) / 3f0
@inline _saturate3(v::Vec3f) = clamp.(v, 0f0, 1f0)
@inline _unit(n::Vec3f, fallback::Vec3f) = (l = norm(n); l > 1f-8 ? n / l : fallback)

"""Input `j` of the node at instruction `i`, which carries its inputs past `NODE_ARGS` in the `ARGS` instructions after it."""
@inline _pf(s, prog, i::Int32, j::Int) = _f(s, (@inbounds prog[i + Int32((j - 1) ÷ NODE_ARGS)]), (j - 1) % NODE_ARGS + 1)
@inline _pv(s, prog, i::Int32, j::Int) = _v(s, (@inbounds prog[i + Int32((j - 1) ÷ NODE_ARGS)]), (j - 1) % NODE_ARGS + 1)

"""Whether `wo` sees the back of the surface, as Cycles' `SD_BACKFACING`."""
@inline _backfacing(sc::ShaderContext) = dot(sc.wo, sc.ng) < 0f0

"""The geometric normal on the side `wo` is on: the frame Cycles' closures live in."""
@inline _facing_ng(sc::ShaderContext) = _backfacing(sc) ? -sc.ng : sc.ng

"""A closure's normal, unit and on the side `wo` is on, as Cycles flips `N` with `Ng` for a backfacing hit."""
@inline function _closure_normal(n::Vec3f, sc::ShaderContext)
    nn = _unit(n, sc.tfc.n)
    return _backfacing(sc) ? -nn : nn
end

"""Cycles' `ensure_valid_specular_reflection`: `n` tilted towards `ng` until the mirror of `wo` leaves the surface."""
@inline function _valid_reflection_normal(ng::Vec3f, wo::Vec3f, n::Vec3f)
    r = 2f0 * dot(n, wo) * n - wo
    iz = dot(wo, ng)
    threshold = min(0.9f0 * iz, 0.01f0)
    dot(ng, r) >= threshold && return n
    x = _unit(n - dot(n, ng) * ng, n)
    ix = dot(wo, x)
    a = ix^2 + iz^2
    b = 2f0 * (a + iz * threshold)
    c = (threshold + iz)^2
    root = sqrt(max(b^2 - 4f0 * a * c, 0f0))
    nz2 = ix < 0f0 ? 0.25f0 * (b + root) / a : 0.25f0 * (b - root) / a
    return sqrt(max(1f0 - nz2, 0f0)) * x + sqrt(max(nz2, 0f0)) * ng
end

@inline _valid_normal(n::Vec3f, sc::ShaderContext) = _valid_reflection_normal(_facing_ng(sc), sc.wo, n)

"""A tangent for an anisotropic lobe: `t` in the plane of `n`, rotated around it by `rotation` turns."""
@inline function _anisotropic_tangent(t::Vec3f, n::Vec3f, rotation::Float32, fallback::Vec3f)
    t = _unit(t - dot(t, n) * n, _unit(fallback - dot(fallback, n) * n, coordinate_system(n)[1]))
    rotation == 0f0 && return t
    s, c = sincos(rotation * 2f0 * Float32(π))
    return c * t + s * cross(n, t)
end

@inline _f0_from_ior(ior::Float32) = ((ior - 1f0) / (ior + 1f0))^2
@inline function _ior_from_f0(f0::Float32)
    r = sqrt(clamp(f0, 0f0, 0.99f0))
    return (1f0 + r) / (1f0 - r)
end

"""The real dielectric Fresnel remapped to go from 0 at normal incidence to 1 at grazing, as Cycles' generalized Schlick with a negative exponent."""
@inline function _schlick_shape(cos_i::Float32, ior::Float32)
    r0 = _f0_from_ior(ior)
    return clamp((fresnel_dielectric(cos_i, ior) - r0) / max(1f0 - r0, 1f-6), 0f0, 1f0)
end

"""Cycles' `fresnel_dielectric_Fss`: the average single-scattering Fresnel of a dielectric."""
@inline _dielectric_fss(eta::Float32) =
    eta < 1f0 ? 0.997118f0 + eta * (0.1014f0 - eta * (0.965241f0 + eta * 0.130607f0)) : (eta - 1f0) / (4.08567f0 + 1.00071f0 * eta)

@inline function _f82_b(f0::Vec3f, tint::Vec3f)
    f = 6f0 / 7f0
    f5 = f^5
    return (f0 .+ (1f0 .- f0) .* f5) .* (7f0 / (f5 * f)) .* (1f0 .- tint)
end
@inline _f82_fss(f0::Vec3f, b::Vec3f) = f0 .+ (1f0 .- f0) ./ 21f0 .- b ./ 126f0

# ── Lookup-table albedos ────────────────────────────────────────────────────

@inline _rough_index(c::RGBClosure) = sqrt(sqrt(c.alpha_x * c.alpha_y))
@inline _ior_index(ior::Float32) = sqrt(abs((ior - 1f0) / (ior + 1f0)))
@inline _mu_index(mu::Float32) = mu

"""
Cycles' `microfacet_ggx_preserve_energy`: scale the lobe up by what single scattering misses at
the view angle, and darken it by the Fresnel of the light that scatters more than once.
"""
@noinline function _preserve_energy(lut, c::RGBClosure, mu::Float32, fss::Vec3f)
    rough = _rough_index(c)
    y = _mu_index(mu)
    if c.lobe == Lobe.GLASS
        inverse = c.ior < 1f0
        z = _ior_index(inverse ? 1f0 / c.ior : c.ior)
        e_offset = Int32(inverse ? LookupTable.GGX_GLASS_INV_E : LookupTable.GGX_GLASS_E)
        avg_offset = Int32(inverse ? LookupTable.GGX_GLASS_INV_EAVG : LookupTable.GGX_GLASS_EAVG)
        E = lookup_table_read_3d(lut, rough, y, z, e_offset, Int32(16), Int32(16), Int32(16))
        E_avg = lookup_table_read_2d(lut, rough, z, avg_offset, Int32(16), Int32(16))
    else
        E = lookup_table_read_2d(lut, rough, y, Int32(LookupTable.GGX_E), Int32(32), Int32(32))
        E_avg = lookup_table_read(lut, rough, Int32(LookupTable.GGX_EAVG), Int32(32))
    end
    missing = (1f0 - E) / max(E, 1f-6)
    energy = 1f0 + missing
    weight, sample_weight = c.weight, c.sample_weight
    if fss != Vec3f(1)
        fms = fss .* E_avg ./ (1f0 .- fss .* (1f0 - E_avg))
        darkening = (1f0 .+ fms .* missing) ./ energy
        weight = weight .* darkening
        sample_weight *= _avg(darkening)
    end
    return RGBClosure(c.lobe, c.fresnel, weight, c.c1, c.c2, c.c3, c.n, c.t, c.alpha_x, c.alpha_y, c.ior, energy,
                      c.p1, c.p2, c.inv_abbe, sample_weight)
end

"""
Cycles' `bsdf_microfacet_estimate_albedo`, reflection and transmission: what the lobe returns of
light from `wo`, from the generalized Schlick tables, or from the Fresnel at the view angle for a thin film.
"""
@noinline function _estimate_albedo(lut, c::RGBClosure, mu::Float32)
    rough = _rough_index(c)
    y = _mu_index(mu)
    if c.p1 > THIN_FILM_THICKNESS_CUTOFF
        F = _rgb_fresnel(c, mu)
        return (c.c2 .* F, c.lobe == Lobe.GLASS ? c.c3 .* (1f0 .- F) : Vec3f(0))
    elseif c.fresnel == Fresnel.SCHLICK || (c.fresnel == Fresnel.DIELECTRIC && c.ior > 1f0)
        s = lookup_table_read_3d(lut, rough, y, _ior_index(c.ior), Int32(LookupTable.GGX_GEN_SCHLICK_IOR_S),
                                 Int32(16), Int32(16), Int32(16))
        f0 = c.fresnel == Fresnel.SCHLICK ? c.c1 : Vec3f(_f0_from_ior(c.ior))
        F = f0 .+ (1f0 .- f0) .* s
        return (c.c2 .* F, c.lobe == Lobe.GLASS ? c.c3 .* (1f0 .- F) : Vec3f(0))
    elseif c.fresnel == Fresnel.F82_TINT
        s = lookup_table_read_3d(lut, rough, y, 0.5f0, Int32(LookupTable.GGX_GEN_SCHLICK_S), Int32(16), Int32(16), Int32(16))
        return (c.c1 .+ (1f0 .- c.c1) .* s, Vec3f(0))
    elseif c.fresnel == Fresnel.DIELECTRIC
        F = Vec3f(fresnel_dielectric(mu, c.ior))
        return (c.c2 .* F, c.lobe == Lobe.GLASS ? c.c3 .* (1f0 .- F) : Vec3f(0))
    end
    return (Vec3f(1), c.lobe == Lobe.GLASS ? Vec3f(1) : Vec3f(0))
end

"""The lobe's Fresnel in RGB at the cosine `mu`, for an estimate: a thin film at three wavelengths."""
@inline function _rgb_fresnel(c::RGBClosure, mu::Float32)
    return Vec3f(_fresnel_channel(c, mu, c.c1[1], c.c2[1], 610f0), _fresnel_channel(c, mu, c.c1[2], c.c2[2], 550f0),
                 _fresnel_channel(c, mu, c.c1[3], c.c2[3], 465f0))
end

"""Set up a microfacet or glass closure's Fresnel the way Cycles' `bsdf_microfacet_setup_fresnel_*` do."""
@noinline function _setup_fresnel(lut, c::RGBClosure, mu::Float32)
    refl, trans = _estimate_albedo(lut, c, mu)
    c = RGBClosure(c.lobe, c.fresnel, c.weight, c.c1, c.c2, c.c3, c.n, c.t, c.alpha_x, c.alpha_y, c.ior, c.energy,
                   c.p1, c.p2, c.inv_abbe, c.sample_weight * _avg(refl .+ trans))
    fss = if c.fresnel == Fresnel.F82_TINT
        _f82_fss(c.c1, _f82_b(c.c1, c.c2))
    elseif c.fresnel == Fresnel.SCHLICK
        c.lobe == Lobe.GLASS ? c.c3 :
        c.c2 .* (c.c1 .+ (1f0 .- c.c1) .* clamp((_dielectric_fss(c.ior) - _f0_from_ior(c.ior)) / (1f0 - _f0_from_ior(c.ior)), 0f0, 1f0))
    elseif c.fresnel == Fresnel.DIELECTRIC
        Vec3f(_dielectric_fss(c.ior))
    else
        c.c1
    end
    return _preserve_energy(lut, c, mu, fss), refl
end

# ── Thin film ───────────────────────────────────────────────────────────────

"""
The reflectance at wavelength `lambda` (nm) of a substrate under a thin film, by the Airy sum of
the film's two interfaces, averaged over both polarizations. The substrate is a dielectric of
`eta3`, or with `k3 > 0` a conductor.
"""
@noinline function _thin_film_reflectance(cos_i::Float32, thickness::Float32, eta2::Float32, eta3::Float32,
                                        k3::Float32, lambda::Float32)
    sin2_i = max(1f0 - cos_i^2, 0f0)
    sin2_2 = sin2_i / eta2^2
    sin2_2 >= 1f0 && return 1f0
    cos_2 = sqrt(1f0 - sin2_2)
    n3 = Complex{Float32}(eta3, k3)
    cos_3 = sqrt(Complex{Float32}(1f0, 0f0) - Complex{Float32}(sin2_i, 0f0) / (n3 * n3))
    phase = Complex{Float32}(0f0, 4f0 * Float32(π) * eta2 * thickness * cos_2 / lambda)
    shift = exp(phase)
    r = 0f0
    for polarization in (0x00, 0x01)
        r12 = polarization == 0x00 ? (cos_i - eta2 * cos_2) / (cos_i + eta2 * cos_2) :
                                     (eta2 * cos_i - cos_2) / (eta2 * cos_i + cos_2)
        r23 = polarization == 0x00 ? (eta2 * cos_2 - n3 * cos_3) / (eta2 * cos_2 + n3 * cos_3) :
                                     (n3 * cos_2 - eta2 * cos_3) / (n3 * cos_2 + eta2 * cos_3)
        rr = (r12 + r23 * shift) / (1f0 + r12 * r23 * shift)
        r += abs2(rr)
    end
    return clamp(0.5f0 * r, 0f0, 1f0)
end

"""
A conductor's IOR and extinction at one wavelength from its reflectance at normal incidence and
at about 82°, as Cycles estimates them for a thin film (Gulbrandsen's artist-friendly Fresnel).
"""
@inline function _conductor_nk(f0::Float32, b::Float32)
    r = min(f0, 0.999f0)
    m = 1f0 - 1f0 / 7f0
    m5 = m^5
    g = clamp(f0 + (1f0 - f0) * m5 - b * (1f0 / 7f0) * m5 * m, 0f0, 1f0)
    sqrt_r = sqrt(r)
    n = (1f0 - g) * (1f0 + sqrt_r) / (1f0 - sqrt_r) + g * (1f0 - r) / (1f0 + r)
    k = sqrt(max((r * (n + 1f0)^2 - (n - 1f0)^2) / (1f0 - r), 0f0))
    return n, k
end

"""
A microfacet lobe's reflectance for one channel or wavelength with its thin film: the film's own
interference, and for a Principled specular layer the scaling Cycles applies towards `f0`.
"""
@inline function _film_fresnel(fresnel::UInt8, ior::Float32, thickness::Float32, film_ior::Float32, cos_i::Float32,
                               f0::Float32, tint::Float32, lambda::Float32)
    if fresnel == Fresnel.F82_TINT
        n, k = _conductor_nk(f0, _f82_b_channel(f0, tint))
        return _thin_film_reflectance(cos_i, thickness, film_ior, n, k, lambda)
    end
    F = _thin_film_reflectance(cos_i, thickness, film_ior, ior, 0f0, lambda)
    fresnel == Fresnel.SCHLICK || return F
    f0_real = _f0_from_ior(ior)
    (f0_real > 1f-5 && F != 1f0) || return F
    s = clamp((F - 1f0) / (f0_real - 1f0), 0f0, 1f0)
    return F * (1f0 + (f0 / f0_real - 1f0) * s)
end

@inline _fresnel_channel(c::RGBClosure, cos_i::Float32, f0::Float32, tint::Float32, lambda::Float32) =
    _film_fresnel(c.fresnel, c.ior, c.p1, c.p2, cos_i, f0, tint, lambda)

@inline function _f82_b_channel(f0::Float32, tint::Float32)
    f = 6f0 / 7f0
    f5 = f^5
    return (f0 + (1f0 - f0) * f5) * (7f0 / (f5 * f)) * (1f0 - tint)
end

# ── Sheen and Oren-Nayar ────────────────────────────────────────────────────

"""Cycles' `bsdf_sheen_setup`: the LTC fit for the view angle and its albedo, which scales the lobe."""
@noinline function _sheen(lut, weight::Vec3f, n::Vec3f, roughness::Float32, wo::Vec3f)
    roughness = clamp(roughness, 1f-3, 1f0)
    mu = dot(n, wo)
    offset = Int32(LookupTable.SHEEN_LTC)
    a = lookup_table_read_2d(lut, mu, roughness, offset, Int32(32), Int32(32))
    b = lookup_table_read_2d(lut, mu, roughness, offset + Int32(1024), Int32(32), Int32(32))
    albedo = lookup_table_read_2d(lut, mu, roughness, offset + Int32(2048), Int32(32), Int32(32))
    (abs(a) < 1f-5 || albedo < 1f-5) && return RGBClosure()
    return _closure(Lobe.SHEEN, weight * albedo, n; p1 = a, p2 = b)
end

@inline function _oren_nayar_g(cos_theta::Float32)
    cos_theta < 1f-6 && return (Float32(π) / 2f0 - 2f0 / 3f0) - cos_theta
    sin_theta = sqrt(max(1f0 - cos_theta^2, 0f0))
    theta = acos(clamp(cos_theta, -1f0, 1f0))
    return sin_theta * (theta - 2f0 / 3f0 - sin_theta * cos_theta) +
           2f0 / 3f0 * (sin_theta / cos_theta) * (1f0 - sin_theta^2 * sin_theta)
end

"""Cycles' energy-preserving Oren-Nayar (OpenPBR's), or Lambert where the roughness is almost zero."""
@noinline function _diffuse(weight::Vec3f, n::Vec3f, roughness::Float32, color::Vec3f, wo::Vec3f)
    roughness < 1f-5 && return _closure(Lobe.DIFFUSE, weight, n)
    sigma = clamp(roughness, 0f0, 1f0)
    a = 1f0 / (Float32(π) + sigma * (Float32(π) / 2f0 - 2f0 / 3f0))
    b = sigma * a
    albedo = _saturate3(color)
    e_avg = a * Float32(π) + ((2f0 * Float32(π) - 5.6f0) / 3f0) * b
    e_ms = (1f0 / Float32(π)) .* albedo .* albedo .* (e_avg / (1f0 - e_avg)) ./ (1f0 .- albedo .* (1f0 - e_avg))
    e_v = a * Float32(π) + b * _oren_nayar_g(max(dot(n, wo), 0f0))
    return _closure(Lobe.OREN_NAYAR, weight, n; c1 = e_ms .* (1f0 - e_v), p1 = a, p2 = b)
end

"""The transmission half of Cycles' thin subsurface: Translucent, or Oren-Nayar through the surface."""
@noinline function _diffuse_transmission(weight::Vec3f, n::Vec3f, roughness::Float32, color::Vec3f, wo::Vec3f)
    roughness < 1f-5 && return _closure(Lobe.TRANSLUCENT, weight, n)
    c = _diffuse(weight, n, roughness, color, wo)
    return RGBClosure(c.lobe, c.fresnel, c.weight, c.c1, c.c2, c.c3, -n, c.t, c.alpha_x, c.alpha_y, c.ior, c.energy,
                      c.p1, c.p2, c.inv_abbe, c.sample_weight)
end

"""
Cycles' `bsdf_thin_glass_setup`: a slab of glass of no thickness, which reflects what its two
interfaces and the light between them reflect, and lets the rest through, tinted by `color` along
its path in the slab, as a reflection mirrored to the far side and roughened for two refractions.
"""
@noinline function _thin_glass(lut, transmitted::Bool, w::Vec3f, n::Vec3f, alpha::Float32, ior::Float32,
                               reflection_tint::Vec3f, color::Vec3f, wo::Vec3f)
    cos_i = dot(n, wo)
    sin2_t = (1f0 - cos_i^2) / ior^2
    sin2_t >= 1f0 && return RGBClosure()
    cos_t = sqrt(1f0 - sin2_t)
    f0 = min.(_f0_from_ior(ior) .* reflection_tint, 1f0)
    r1 = f0 .+ (1f0 .- f0) .* _schlick_shape(cos_i, ior)
    t1 = 1f0 .- r1
    c = _pow3(max.(color, 0f0), 1f0 / max(cos_t, 1f-6))
    rc = r1 .* c
    T = c .* t1 .* t1 ./ max.(1f0 .- rc .* rc, 1f-6)
    if !transmitted
        R = r1 .+ T .* r1 .* c
        return first(_setup_fresnel(lut, _closure(Lobe.MICROFACET, R .* w, n; c1 = reflection_tint, alpha_x = alpha), cos_i))
    end
    transmission_alpha = clamp(alpha * 3.4f0 * (ior - 1f0) * (ior - 0.5f0)^2 / ior^3, 0f0, 1f0)
    transmission = _closure(Lobe.THIN_GLASS, T .* w, n; c1 = Vec3f(1), alpha_x = transmission_alpha)
    return first(_setup_fresnel(lut, transmission, cos_i))
end

# ── Closure nodes ───────────────────────────────────────────────────────────

@inline function _emit_closure(cl::ClosureList, s, ins::NodeInstr, sc::ShaderContext, lut)
    w = _f(s, ins, 1)
    kind = ins.mode[1]
    wo = sc.wo
    if kind == ClosureKind.DIFFUSE
        color = max.(_v(s, ins, 2), 0f0)
        n = _closure_normal(_v(s, ins, 4), sc)
        return _push(cl, _diffuse(w * color, n, _f(s, ins, 3), color, wo))
    elseif kind == ClosureKind.TRANSLUCENT
        return _push(cl, _closure(Lobe.TRANSLUCENT, w * max.(_v(s, ins, 2), 0f0), _closure_normal(_v(s, ins, 3), sc)))
    elseif kind == ClosureKind.GLOSSY || kind == ClosureKind.METALLIC
        metallic = kind == ClosureKind.METALLIC
        base = 2 + metallic
        roughness = clamp(_f(s, ins, base + 1), 0f0, 1f0)
        anisotropy = clamp(_f(s, ins, base + 2), -1f0, 1f0)
        n = _valid_normal(_closure_normal(_v(s, ins, base + 4), sc), sc)
        ax, ay, t = _anisotropic_alphas(roughness, anisotropy, _v(s, ins, base + 5), n, _f(s, ins, base + 3), sc)
        c = metallic ?
            _closure(Lobe.MICROFACET, Vec3f(w), n; fresnel = Fresnel.F82_TINT, c1 = _saturate3(_v(s, ins, 2)),
                     c2 = _saturate3(_v(s, ins, 3)), t, alpha_x = ax, alpha_y = ay) :
            _closure(Lobe.MICROFACET, w * max.(_v(s, ins, 2), 0f0), n; c1 = max.(_v(s, ins, 2), 0f0), t,
                     alpha_x = ax, alpha_y = ay)
        return _push(cl, first(_setup_fresnel(lut, c, dot(n, wo))))
    elseif kind == ClosureKind.GLASS
        color = max.(_v(s, ins, 2), 0f0)
        alpha = clamp(_f(s, ins, 3), 0f0, 1f0)^2
        ior = max(_f(s, ins, 4), 1f-5)
        n = _closure_normal(_v(s, ins, 5), sc)
        eta = _backfacing(sc) ? 1f0 / ior : ior
        c = _closure(Lobe.GLASS, Vec3f(w), n; fresnel = Fresnel.DIELECTRIC, c2 = color, c3 = color,
                     alpha_x = alpha, ior = eta)
        return _push(cl, first(_setup_fresnel(lut, c, dot(n, wo))))
    elseif kind == ClosureKind.SHEEN
        n = _closure_normal(_v(s, ins, 4), sc)
        return _push(cl, _sheen(lut, w * max.(_v(s, ins, 2), 0f0), n, _f(s, ins, 3), wo))
    elseif kind == ClosureKind.TRANSPARENT
        return _with_transparency(cl, w * _saturate3(_v(s, ins, 2)))
    elseif kind == ClosureKind.EMISSION
        return _with_emission(cl, w * max.(_v(s, ins, 2), 0f0) * _f(s, ins, 3))
    end
    return cl
end

"""
The alphas of a GGX lobe of `roughness` stretched by `anisotropy` along its tangent, as Cycles:
the tangent input where there is one, else the shading tangent, rotated by `rotation` turns.
"""
@inline function _anisotropic_alphas(roughness::Float32, anisotropy::Float32, tangent::Vec3f, n::Vec3f,
                                     rotation::Float32, sc::ShaderContext)
    alpha = roughness^2
    anisotropy == 0f0 && return alpha, alpha, Vec3f(0)
    aspect = sqrt(1f0 - abs(anisotropy) * 0.9f0)
    t = _anisotropic_tangent(tangent, n, rotation, sc.dpdus)
    anisotropy < 0f0 && (t = cross(n, t))
    return alpha / aspect, alpha * aspect, t
end

"""The inputs of a Principled BSDF, read off the stack before the closures are set up out of line."""
struct PrincipledInputs
    weight::Float32
    base::Vec3f
    metallic::Float32
    roughness::Float32
    ior::Float32
    alpha::Float32
    n::Vec3f
    diffuse_roughness::Float32
    subsurface::Float32
    specular_level::Float32
    specular_tint::Vec3f
    anisotropic::Float32
    rotation::Float32
    tangent::Vec3f
    transmission::Float32
    dispersion::Float32
    abbe::Float32
    coat_weight::Float32
    coat_roughness::Float32
    coat_ior::Float32
    coat_tint::Vec3f
    coat_n::Vec3f
    sheen_weight::Float32
    sheen_roughness::Float32
    sheen_tint::Vec3f
    emission::Vec3f
    film_thickness::Float32
    film_ior::Float32
    subsurface_anisotropy::Float32
    thin_wall::Float32
    subsurface_radius::Vec3f
    subsurface_scale::Float32
end

@inline _principled(cl::ClosureList, s, prog, i::Int32, ins::NodeInstr, sc::ShaderContext, lut) =
    _principled_closures(cl, PrincipledInputs(
        _f(s, ins, 1), _pv(s, prog, i, 2), _pf(s, prog, i, 3), _pf(s, prog, i, 4), _pf(s, prog, i, 5),
        _pf(s, prog, i, 6), _pv(s, prog, i, 7), _pf(s, prog, i, 8), _pf(s, prog, i, 9), _pf(s, prog, i, 14),
        _pv(s, prog, i, 15), _pf(s, prog, i, 16), _pf(s, prog, i, 17), _pv(s, prog, i, 18), _pf(s, prog, i, 19),
        _pf(s, prog, i, 20), _pf(s, prog, i, 21), _pf(s, prog, i, 22), _pf(s, prog, i, 23), _pf(s, prog, i, 24),
        _pv(s, prog, i, 25), _pv(s, prog, i, 26), _pf(s, prog, i, 27), _pf(s, prog, i, 28), _pv(s, prog, i, 29),
        _pv(s, prog, i, 30) * _pf(s, prog, i, 31), _pf(s, prog, i, 32), _pf(s, prog, i, 33), _pf(s, prog, i, 13),
        _pf(s, prog, i, 34), _pv(s, prog, i, 10), _pf(s, prog, i, 11)), sc, lut)

"""
Blender's Principled BSDF, as Cycles' `svm_node_closure_bsdf` lays it out: transparency first,
then sheen, coat and emission, which take from what is under them what they reflect, then the
metal, transmission and the specular layer over subsurface and diffuse.
"""
const _PRINCIPLED_LAYERS = Int32(9)

@inline function _principled_closures(cl::ClosureList, p::PrincipledInputs, sc::ShaderContext, lut)
    w = Vec3f(p.weight)
    alpha = clamp(p.alpha, 0f0, 1f0)
    if alpha < 1f0
        cl = _with_transparency(cl, w * (1f0 - alpha))
        w = w * alpha
    end
    for layer in Int32(1):_PRINCIPLED_LAYERS
        c, w = _principled_layer(layer, p, w, sc, lut)
        cl = _push(cl, c)
        layer == Int32(2) && _max(p.emission) > 0f0 && (cl = _with_emission(cl, max.(p.emission, 0f0) .* w))
    end
    return cl
end

"""
One layer of Blender's Principled BSDF, as Cycles' `svm_node_closure_bsdf` lays them out: the
sheen, coat, metal, transmission (two closures with a thin wall), specular, subsurface (two with a
thin wall) and diffuse closure for the weight `w` the layers above leave, and what this one leaves.
Out of line and one closure at a time, so no call holds the closure list.
"""
@noinline function _principled_layer(layer::Int32, p::PrincipledInputs, w::Vec3f, sc::ShaderContext, lut)
    none = (RGBClosure(), w)
    wo = sc.wo
    base = max.(p.base, 0f0)
    base_clamped = min.(base, 1f0)
    metallic = clamp(p.metallic, 0f0, 1f0)
    roughness = clamp(p.roughness, 0f0, 1f0)
    ior = max(p.ior, 1f-5)
    n = _closure_normal(p.n, sc)
    diffuse_roughness = clamp(p.diffuse_roughness, 0f0, 1f0)
    subsurface = clamp(p.subsurface, 0f0, 1f0)
    specular_tint = max.(p.specular_tint, 0f0)
    transmission = clamp(p.transmission, 0f0, 1f0)
    coat_weight = max(p.coat_weight, 0f0)
    coat_n = _closure_normal(p.coat_n, sc)
    film_thickness = p.film_thickness
    film_ior = film_thickness > THIN_FILM_THICKNESS_CUTOFF ? max(p.film_ior, 1f-5) : 0f0
    thin_wall = p.thin_wall > 0.5f0

    if layer == Int32(1)
        sheen_weight = max(p.sheen_weight, 0f0)
        sheen_weight > CLOSURE_WEIGHT_CUTOFF || return none
        sheen_n = _unit(n + clamp(coat_weight, 0f0, 1f0) * (coat_n - n), n)
        sheen = _sheen(lut, sheen_weight * max.(p.sheen_tint, 0f0) .* w, sheen_n, clamp(p.sheen_roughness, 0f0, 1f0), wo)
        return (sheen, _layering(sheen.weight, w))
    elseif layer == Int32(2)
        coat_weight > CLOSURE_WEIGHT_CUTOFF || return none
        coat_ior = max(p.coat_ior, 1f0)
        coat_vn = _valid_normal(coat_n, sc)
        coat = _closure(Lobe.MICROFACET, coat_weight * w, coat_vn; fresnel = Fresnel.DIELECTRIC,
                        alpha_x = clamp(p.coat_roughness, 0f0, 1f0)^2, ior = coat_ior)
        coat, refl = _setup_fresnel(lut, coat, dot(coat_vn, wo))
        w = _layering(coat.weight .* refl, w)
        tint = max.(p.coat_tint, 0f0)
        if tint != Vec3f(1)
            cos_nt = sqrt(max(1f0 - (1f0 - dot(wo, coat_vn)^2) / coat_ior^2, 1f-8))
            w = w .* (1f0 .+ coat_weight .* (_pow3(tint, 1f0 / cos_nt) .- 1f0))
        end
        return (coat, w)
    end

    vn = _valid_normal(n, sc)
    ax, ay, t = _anisotropic_alphas(roughness, clamp(p.anisotropic, 0f0, 1f0), p.tangent, vn, p.rotation, sc)
    if layer == Int32(3)
        metallic > CLOSURE_WEIGHT_CUTOFF || return none
        metal = _closure(Lobe.MICROFACET, metallic * w, vn; fresnel = Fresnel.F82_TINT, c1 = base_clamped,
                         c2 = min.(specular_tint, 1f0), t, alpha_x = ax, alpha_y = ay, p1 = film_thickness, p2 = film_ior)
        return (first(_setup_fresnel(lut, metal, dot(vn, wo))), w * (1f0 - metallic))
    elseif layer == Int32(4) || layer == Int32(5)
        transmission > CLOSURE_WEIGHT_CUTOFF || return none
        if thin_wall
            w_out = layer == Int32(4) ? w : w * (1f0 - transmission)
            return (_thin_glass(lut, layer == Int32(5), transmission * w, vn, roughness^2, ior, specular_tint,
                                base_clamped, wo), w_out)
        end
        layer == Int32(5) && return none
        abbe = max(p.abbe, 0f0)
        glass = _closure(Lobe.GLASS, transmission * w, vn; fresnel = Fresnel.SCHLICK,
                         c1 = min.(_f0_from_ior(ior) * specular_tint, 1f0), c2 = Vec3f(1), c3 = sqrt.(base_clamped),
                         alpha_x = roughness^2, ior = _backfacing(sc) ? 1f0 / ior : ior, p1 = film_thickness,
                         p2 = film_ior, inv_abbe = abbe > 0f0 ? clamp(p.dispersion, 0f0, 1f0) / abbe : 0f0)
        return (first(_setup_fresnel(lut, glass, dot(vn, wo))), w * (1f0 - transmission))
    elseif layer == Int32(6)
        eta = _specular_eta(ior, p.specular_level)
        f0 = _specular_f0(ior, p.specular_level)
        (eta != 1f0 || film_thickness > THIN_FILM_THICKNESS_CUTOFF) || return none
        specular = _closure(Lobe.MICROFACET, w, vn; fresnel = Fresnel.SCHLICK, c1 = min.(f0 * specular_tint, 1f0),
                            t, alpha_x = ax, alpha_y = ay, ior = eta, p1 = film_thickness, p2 = film_ior)
        specular, refl = _setup_fresnel(lut, specular, dot(vn, wo))
        return (specular, _layering(specular.weight .* refl, w))
    elseif layer == Int32(7) || layer == Int32(8)
        subsurface > CLOSURE_WEIGHT_CUTOFF || return none
        ss = base_clamped .* subsurface .* w
        if thin_wall
            a = p.subsurface_anisotropy
            layer == Int32(7) && return (_diffuse(clamp(0.5f0 * (1f0 - a), 0f0, 1f0) * ss, n, diffuse_roughness,
                                                  base_clamped, wo), w)
            return (_diffuse_transmission(clamp(0.5f0 * (1f0 + a), 0f0, 1f0) * ss, n, diffuse_roughness, base_clamped,
                                          wo), w)
        end
        layer == Int32(8) && return none
        radius = max.(p.subsurface_radius, 0f0) * max(p.subsurface_scale, 0f0)
        _max(radius) < 1f-8 && return (_diffuse(ss, n, diffuse_roughness, base_clamped, wo), w)
        return (_subsurface(ss, base_clamped, radius, n, roughness^2, _specular_eta(ior, p.specular_level),
                            p.subsurface_anisotropy), w)
    end
    return (_diffuse(base .* (1f0 - subsurface) .* w, n, diffuse_roughness, base, wo), w)
end

# ── Bump ────────────────────────────────────────────────────────────────────

"""
Blender's Bump node, as Cycles' `svm_node_set_bump`: the height at the hit and at the hit moved
by `filter_width` of each screen-space differential, turned into a normal.
"""
@inline function _bump(s, ins::NodeInstr, sc::ShaderContext)
    h_c, h_x, h_y = _f(s, ins, 1), _f(s, ins, 2), _f(s, ins, 3)
    strength = clamp(_f(s, ins, 4), 0f0, 1f0)
    scale = ins.mode[1] == 0x01 ? -_f(s, ins, 5) : _f(s, ins, 5)
    normal_in = _unit(_v(s, ins, 6), sc.tfc.n)
    filter_width = _f(s, ins, 7)
    dpdx, dpdy = _bump_differentials(sc)

    rx = cross(dpdy, normal_in)
    ry = cross(normal_in, dpdx)
    det = dot(dpdx, rx)
    surfgrad = (h_x - h_c) * rx + (h_y - h_c) * ry
    normal_out = filter_width * abs(det) * normal_in - scale * sign(det) * surfgrad
    l = norm(normal_out)
    l > 0f0 || return normal_in
    return _unit(strength * (normal_out / l) + (1f0 - strength) * normal_in, normal_in)
end

"""The differentials a bump shifts the hit by; along the shading tangents when the hit has none."""
@inline function _bump_differentials(sc::ShaderContext)
    dot(sc.dpdx, sc.dpdx) > 0f0 && dot(sc.dpdy, sc.dpdy) > 0f0 && return sc.dpdx, sc.dpdy
    t = _unit(sc.dpdus - dot(sc.dpdus, sc.tfc.n) * sc.tfc.n, coordinate_system(sc.tfc.n)[1])
    return BUMP_DEFAULT_DELTA * t, BUMP_DEFAULT_DELTA * cross(sc.tfc.n, t)
end

@inline function _shifted_leaf(op::UInt8, s, ins::NodeInstr, sc::ShaderContext)
    fw = ins.k[1]
    dpdx, dpdy = _bump_differentials(sc)
    tfc = sc.tfc
    op == NodeOp.POSITION_DX && return Vec3f(tfc.p) + fw * dpdx
    op == NodeOp.POSITION_DY && return Vec3f(tfc.p) + fw * dpdy
    op == NodeOp.UV_DX && return Vec3f(tfc.uv[1] + fw * tfc.dudx, tfc.uv[2] + fw * tfc.dvdx, 0f0)
    return Vec3f(tfc.uv[1] + fw * tfc.dudy, tfc.uv[2] + fw * tfc.dvdy, 0f0)
end

const _PRINCIPLED_ARGS = Int32(3)

"""
Run a compiled shader program at a hit: the closures it produces. Not inlined, for the reason
`eval_nodes` is not.
"""
@noinline function eval_shader(prog::AbstractVector{NodeInstr{F}}, textures, sc::ShaderContext, ::Val{N}) where {F, N}
    lut = lookup_tables(textures)
    s = _new_stack(Val(_stack_size(F)))
    cl = ClosureList{N}()
    i = Int32(1)
    n = length(prog) % Int32
    @inbounds while i <= n
        ins = prog[i]
        op = ins.op
        skip = Int32(0)
        if op == NodeOp.RESULT
            return cl
        elseif op == NodeOp.CLOSURE
            cl = _emit_closure(cl, s, ins, sc, lut)
        elseif op == NodeOp.PRINCIPLED
            cl = _principled(cl, s, prog, i, ins, sc, lut)
            skip = _PRINCIPLED_ARGS
        elseif op == NodeOp.BUMP
            s = _store(s, ins, _bump(s, ins, sc))
        elseif NodeOp.POSITION_DX <= op <= NodeOp.UV_DY
            s = _store(s, ins, _shifted_leaf(op, s, ins, sc))
        else
            s, skip = _exec(s, prog, i, ins, textures, sc.tfc)
        end
        i += Int32(1) + skip
    end
    return cl
end

@propagate_inbounds _th_shader(prog::AbstractVector{<:NodeInstr}, textures, sc, capacity::Val) = eval_shader(prog, textures, sc, capacity)
@propagate_inbounds _th_shader(prog, textures, sc, ::Val{N}) where {N} = ClosureList{N}()
