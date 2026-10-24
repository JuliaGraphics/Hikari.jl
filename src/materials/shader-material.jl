# Ported from Blender Cycles v5.2.1: intern/cycles/kernel/closure/
# bsdf_microfacet.h, bsdf_sheen.h, bsdf_oren_nayar.h. Copyright 2011-2022 Blender Foundation.
# SPDX-License-Identifier: Apache-2.0. Modified: rewritten in Julia, spectral. See
# LICENSES/Apache-2.0.txt.

# ============================================================================
# ShaderMaterial: a material that is a shader graph
# ============================================================================

"""
    ShaderMaterial(shader)

A material made of a shader graph, as Blender's Material Output takes one: BSDF nodes such as
[`principled_bsdf`](@ref TextureNodes.principled_bsdf), combined by `mix_shader` and
`add_shader`, with node textures and `bump` at any of their inputs. The graph is compiled to one
program, so graphs differ only in what the type says they use: transparency, subsurface
scattering and emission, whose code is compiled only into the kernels of shaders that have them,
and how many closures they can produce at once, which is what each hit holds room for.
"""
struct ShaderMaterial{ProgT, Transparent, Subsurface, Emissive, Capacity} <: Material
    program::ProgT
end

function ShaderMaterial(shader::ShaderNode)
    graph = ShaderGraph(shader)
    return ShaderMaterial{ShaderGraph, has_transparency(graph), has_subsurface(graph), has_emission(graph),
                          closure_capacity(graph)}(graph)
end

to_device_material(dhv, mat::ShaderMaterial{P, T, S, E, N}) where {P, T, S, E, N} =
    ShaderMaterial{TexHandle, T, S, E, N}(device_param(dhv, mat.program))

"""The closures `mat`'s program produces at the hit `sc`."""
@propagate_inbounds eval_shader_handle(textures, mat::ShaderMaterial{P, T, S, E, N}, sc::ShaderContext) where {P, T, S, E, N} =
    eval_shader_handle(textures, mat.program, sc, Val(N))

"""A closure at the path's wavelengths, and its share of the BSDF's samples."""
struct SpectralClosure
    lobe::UInt8
    fresnel::UInt8
    weight::SpectralRadiance
    c1::SpectralRadiance
    c2::SpectralRadiance
    c3::SpectralRadiance
    n::Vec3f
    t::Vec3f
    alpha_x::Float32
    alpha_y::Float32
    ior::Float32
    energy::Float32
    p1::Float32
    p2::Float32
    sample_weight::Float32
end

SpectralClosure() = SpectralClosure(0x00, 0x00, SpectralRadiance(0f0), SpectralRadiance(0f0), SpectralRadiance(0f0),
                                    SpectralRadiance(0f0), Vec3f(0, 0, 1), Vec3f(0), 0f0, 0f0, 1f0, 1f0, 0f0, 0f0, 0f0)

"""
The closures a shader produced at a hit, sampled and evaluated as one BSDF, and what the hit emits.
`Subsurface` and `Emissive` are the shader's, so a BSDF without them carries no code for them.
"""
struct ClosureBSDF{N, Subsurface, Emissive}
    count::Int32
    total_sample_weight::Float32
    closures::NTuple{N, SpectralClosure}
    lambda::Wavelengths
    emission::SpectralRadiance
    dispersive::Bool
end

@propagate_inbounds function shader_context(work, camera, samples_per_pixel::Int32)
    dpdx, dpdy = approximate_dp_dxy(work.pi, work.n, camera, samples_per_pixel)
    dudx, dudy, dvdx, dvdy = compute_uv_derivatives(work.dpdu, work.dpdv, dpdx, dpdy)
    tfc = TextureFilterContext(work.uv, dudx, dudy, dvdx, dvdy, work.face_idx, work.bary, work.pi, work.ns)
    return ShaderContext(tfc, work.wo, dpdx, dpdy, work.dpdus, work.n)
end

@propagate_inbounds eval_shader_handle(textures, h::TexHandle, sc::ShaderContext, capacity::Val) =
    Raycore.with_texture(_th_shader, textures, h.slot, h.idx, textures, sc, capacity)

"""
    material_bxdf(mat, table, textures, work, camera, samples_per_pixel, lambda, regularize)

The BSDF of `mat` at the hit `work`: `get_bxdf` with the hit's texture filter context, and for a
shader the closures its program produces there.
"""
@propagate_inbounds function material_bxdf(mat, table, textures, work, camera, samples_per_pixel::Int32,
                                           lambda::Wavelengths, regularize::Bool)
    tfc = compute_texture_filter_context(work, camera, samples_per_pixel)
    return get_bxdf(mat, table, textures, tfc, lambda, regularize)
end

@propagate_inbounds material_bxdf(mi::MediumInterface, table, textures, work, camera, samples_per_pixel::Int32,
                                  lambda::Wavelengths, regularize::Bool) =
    material_bxdf(mi.material, table, textures, work, camera, samples_per_pixel, lambda, regularize)

@propagate_inbounds function material_bxdf(mat::ShaderMaterial{P, T, S, E, N}, table, textures, work, camera,
                                           samples_per_pixel::Int32, lambda::Wavelengths,
                                           regularize::Bool) where {P, T, S, E, N}
    sc = shader_context(work, camera, samples_per_pixel)
    cl = eval_shader_handle(textures, mat, sc)
    return closure_bsdf(Val(S), Val(E), cl, table, lambda, regularize)
end

"""
    add_emission!(pixel_L, bsdf, work)

Add what the surface at `work` emits towards the path. Only a shader emits this way: the other
emitters are area lights, which the emitter kernel handles.
"""
@inline add_emission!(pixel_L, bsdf, work) = nothing
@propagate_inbounds function add_emission!(pixel_L, bsdf::ClosureBSDF{N, S, true}, work) where {N, S}
    is_black(bsdf.emission) && return nothing
    L = work.beta * bsdf.emission / average(work.r_u)
    accumulate_spectrum!(pixel_L, (work.pixel_index - Int32(1)) * Int32(4), L)
    return nothing
end

@inline _uplift(table, c::Vec3f, lambda) = uplift_rgb(table, RGBSpectrum(c[1], c[2], c[3]), lambda)
"""A positive colour that is not a reflectance, such as a mean free path, lifted by its shape and scaled back."""
@inline function _uplift_unbounded(table, c::Vec3f, lambda)
    m = _max(c)
    m > 0f0 || return SpectralRadiance(0f0)
    return _uplift(table, c / m, lambda) * m
end

"""OpenPBR's dispersion, as Cycles' `bsdf_glass_ior`: the Cauchy IOR at `lambda` nm through the d line's `ior`."""
@inline function _dispersed_ior(ior::Float32, inv_abbe::Float32, lambda::Float32)
    inside = ior < 1f0
    n = inside ? 1f0 / ior : ior
    fac = 1f0 / (1f0 / 0.4861f0^2 - 1f0 / 0.6563f0^2)
    B = (n - 1f0) * inv_abbe * fac
    A = n - B / 0.5876f0^2
    n = A + B / (lambda * 1f-3)^2
    return inside ? 1f0 / n : n
end

"""One closure lifted to the path's wavelengths. Not inlined: the closure list would hold eight copies."""
@noinline function _spectral_closure(c::RGBClosure, scale::Float32, sample_weight::Float32, table, lambda::Wavelengths,
                                     regularize::Bool)
    microfacet = c.lobe == Lobe.MICROFACET || c.lobe == Lobe.GLASS
    ax = regularize && microfacet ? regularize_alpha(c.alpha_x) : c.alpha_x
    ay = regularize && microfacet ? regularize_alpha(c.alpha_y) : c.alpha_y
    ior = c.inv_abbe > 0f0 ? _dispersed_ior(c.ior, c.inv_abbe, lambda.lambda[1]) : c.ior
    c2 = c.lobe == Lobe.SUBSURFACE ? _uplift_unbounded(table, c.c2, lambda) : _uplift(table, _saturate3(c.c2), lambda)
    c3 = _uplift(table, _saturate3(c.c3), lambda)
    return SpectralClosure(c.lobe, c.fresnel, _uplift(table, _saturate3(c.weight * scale), lambda),
                           _uplift(table, _saturate3(c.c1), lambda), c2,
                           c3, c.n, c.t, ax, ay, ior, c.energy, c.p1, c.p2,
                           sample_weight)
end

@propagate_inbounds function closure_bsdf(::Val{S}, ::Val{E}, cl::ClosureList{N}, table, lambda::Wavelengths,
                                          regularize::Bool) where {S, E, N}
    alpha = 1f0 - _avg(cl.transparency)
    scale = alpha > 1f-6 ? 1f0 / alpha : 0f0
    dispersive = reduce(|, ntuple(k -> Int32(k) <= cl.count && cl.closures[k].inv_abbe > 0f0, Val(N)))
    sample_weights = ntuple(k -> Int32(k) <= cl.count ? cl.closures[k].sample_weight : 0f0, Val(N))
    closures = ntuple(Val(N)) do k
        Int32(k) <= cl.count || return SpectralClosure()
        _spectral_closure(cl.closures[k], scale, sample_weights[k], table, lambda, regularize)
    end
    emission = E && _max(cl.emission) > 0f0 ?
               uplift_rgb_illuminant(table, RGBSpectrum(cl.emission[1], cl.emission[2], cl.emission[3]), lambda) / D65_PHOTOMETRIC :
               SpectralRadiance(0f0)
    return ClosureBSDF{N, S, E}(cl.count, sum(sample_weights), closures, lambda, emission, dispersive)
end

# ── Lobes, in a frame around the closure's own normal ──────────────────────

"""The reflectance and transmittance of a microfacet lobe at the cosine `cos_i` between the microfacet normal and `wo`."""
@inline function _fresnel(c::SpectralClosure, cos_i::Float32, lambda::Wavelengths)
    one = SpectralRadiance(1f0)
    c.fresnel == Fresnel.NONE && return (one, one)
    if c.p1 > THIN_FILM_THICKNESS_CUTOFF
        F = SpectralRadiance(ntuple(k -> _film_fresnel(c.fresnel, c.ior, c.p1, c.p2, cos_i, c.c1.data[k], c.c2.data[k],
                                                        lambda.lambda[k]), Val(4)))
        c.fresnel == Fresnel.F82_TINT && return (F, SpectralRadiance(0f0))
        return (c.c2 * F, c.c3 * (one - F))
    end
    if c.fresnel == Fresnel.F82_TINT
        f = 6f0 / 7f0
        f5 = f^5
        b = (c.c1 + (one - c.c1) * f5) * (one - c.c2) * (7f0 / (f5 * f))
        m = clamp(1f0 - cos_i, 0f0, 1f0)
        m5 = m^5
        return (clamp(c.c1 + (one - c.c1) * m5 - b * (cos_i * m5 * m), 0f0, 1f0), SpectralRadiance(0f0))
    end
    F_real = fresnel_dielectric(cos_i, c.ior)
    if c.fresnel == Fresnel.SCHLICK
        f0_real = _f0_from_ior(c.ior)
        s = clamp((F_real - f0_real) / max(1f0 - f0_real, 1f-6), 0f0, 1f0)
        F = c.c1 + (one - c.c1) * s
        return (c.c2 * F, c.c3 * (one - F))
    end
    return (c.c2 * F_real, c.c3 * (1f0 - F_real))
end

@inline _is_singular(c::SpectralClosure) = c.alpha_x * c.alpha_y <= 2f-10

@inline function _frame(c::SpectralClosure, wo::Vec3f)
    n = c.n
    if c.lobe == Lobe.SHEEN
        t = _unit(wo - dot(wo, n) * n, coordinate_system(n)[1])
        return n, t, cross(n, t)
    end
    c.t == Vec3f(0) && return (n, coordinate_system(n)...)
    return n, c.t, cross(n, c.t)
end

@inline function _sheen_value(c::SpectralClosure, w::Vec3f)
    a, b = c.p1, c.p2
    len2 = (a * w[1] + b * w[3])^2 + (a * w[2])^2 + w[3]^2
    return (1f0 / Float32(π)) * max(w[3], 0f0) * (a / len2)^2
end

"""Cycles' `bsdf_oren_nayar_get_intensity` over the cosine to the light."""
@inline function _oren_nayar(c::SpectralClosure, wo::Vec3f, wi::Vec3f)
    nl, nv = wi[3], max(wo[3], 0f0)
    t = dot(wi, wo) - nl * nv
    t > 0f0 && (t /= max(nl, nv) + floatmin(Float32))
    single = c.p1 + c.p2 * t
    e_l = c.p1 * Float32(π) + c.p2 * _oren_nayar_g(nl)
    return c.weight * (SpectralRadiance(single) + c.c1 * (1f0 - e_l))
end

"""
`f` (without the cosine) and pdf of closure `c` for the local directions `wo`, `wi`, as Cycles'
closures evaluate them.
"""
@noinline function _lobe_eval(c::SpectralClosure, wo::Vec3f, wi::Vec3f, lambda::Wavelengths)
    zero = (SpectralRadiance(0f0), 0f0)
    lobe = c.lobe
    lobe == Lobe.SUBSURFACE && return zero
    if lobe == Lobe.DIFFUSE || lobe == Lobe.OREN_NAYAR
        wi[3] > 0f0 || return zero
        f = lobe == Lobe.DIFFUSE ? c.weight * (1f0 / Float32(π)) : _oren_nayar(c, wo, wi)
        return (f, wi[3] / Float32(π))
    elseif lobe == Lobe.TRANSLUCENT
        wi[3] < 0f0 || return zero
        return (c.weight * (1f0 / Float32(π)), -wi[3] / Float32(π))
    elseif lobe == Lobe.SHEEN
        wi[3] > 0f0 || return zero
        val = _sheen_value(c, wi)
        return (c.weight * (val / wi[3]), val)
    end
    (wo[3] > 0f0 && !_is_singular(c)) || return zero
    if lobe == Lobe.THIN_GLASS
        wi[3] < 0f0 || return zero
        wi = Vec3f(wi[1], wi[2], -wi[3])
    end
    transmission = wi[3] < 0f0
    (transmission && lobe != Lobe.GLASS) && return zero
    H = transmission ? -(c.ior * wi + wo) : wo + wi
    len_h = norm(H)
    len_h > 0f0 || return zero
    H = H / len_h
    H[3] < 0f0 && (H = -H)
    cos_hi = dot(H, wo)
    (transmission && (cos_hi <= 0f0 || dot(H, wi) >= 0f0)) && return zero
    R, T = _fresnel(c, cos_hi, lambda)
    coeff = transmission ? T : R
    sum_rt = average(R) + average(T)
    sum_rt > 0f0 || return zero
    pdf_reflect = average(R) / sum_rt
    lobe_pdf = lobe == Lobe.GLASS ? (transmission ? 1f0 - pdf_reflect : pdf_reflect) : 1f0
    D = trowbridge_reitz_d(H, c.alpha_x, c.alpha_y)
    lambda_o = trowbridge_reitz_lambda(wo, c.alpha_x, c.alpha_y)
    lambda_i = trowbridge_reitz_lambda(wi, c.alpha_x, c.alpha_y)
    common = D / wo[3] * (transmission ? (c.ior / len_h)^2 * abs(cos_hi * dot(H, wi)) : 0.25f0)
    pdf = common * lobe_pdf / (1f0 + lambda_o)
    f = coeff * (common / (1f0 + lambda_o + lambda_i) * c.energy / abs(wi[3]))
    return (c.weight * f, pdf)
end

"""
A direction sampled from closure `c` in local space: `(wi, f, pdf, flags, eta)`, `f` and pdf for a
singular lobe only. `uc` picks between reflection and refraction.
"""
@noinline function _lobe_sample(c::SpectralClosure, wo::Vec3f, u::Point2f, uc::Float32, lambda::Wavelengths)
    none = (Vec3f(0, 0, 1), SpectralRadiance(0f0), 0f0, 0x00, 1f0)
    lobe = c.lobe
    if lobe == Lobe.DIFFUSE || lobe == Lobe.OREN_NAYAR || lobe == Lobe.TRANSLUCENT
        wi = cosine_sample_hemisphere(u)
        lobe == Lobe.TRANSLUCENT && return (Vec3f(wi[1], wi[2], -wi[3]), SpectralRadiance(0f0), 0f0, BXDF_DIFFUSE_TRANSMISSION, 1f0)
        return (wi, SpectralRadiance(0f0), 0f0, BXDF_DIFFUSE_REFLECTION, 1f0)
    elseif lobe == Lobe.SHEEN
        r = sqrt(u[1])
        s, co = sincos(2f0 * Float32(π) * u[2])
        dx, dy = r * co, r * s
        dz = sqrt(max(1f0 - dx^2 - dy^2, 0f0))
        wi = normalize(Vec3f(dx - dz * c.p2, dy, dz * c.p1))
        return (wi, SpectralRadiance(0f0), 0f0, BXDF_DIFFUSE_REFLECTION, 1f0)
    end
    wo[3] > 0f0 || return none
    if lobe == Lobe.THIN_GLASS
        wi, f, pdf, flags, eta = _microfacet_sample(c, wo, u, uc, lambda)
        return (Vec3f(wi[1], wi[2], -wi[3]), f, pdf, flags, eta)
    end
    return _microfacet_sample(c, wo, u, uc, lambda)
end

"""A microfacet or glass closure's sample, as Cycles' `bsdf_microfacet_sample`; a thin glass's as the reflection it is mirrored from."""
@inline function _microfacet_sample(c::SpectralClosure, wo::Vec3f, u::Point2f, uc::Float32, lambda::Wavelengths)
    none = (Vec3f(0, 0, 1), SpectralRadiance(0f0), 0f0, 0x00, 1f0)
    lobe = c.lobe
    singular = _is_singular(c)
    H = singular ? Vec3f(0, 0, 1) : trowbridge_reitz_sample_wm(wo, u, c.alpha_x, c.alpha_y)
    cos_hi = dot(H, wo)
    R, T = _fresnel(c, cos_hi, lambda)
    sum_rt = average(R) + average(T)
    sum_rt > 0f0 || return none
    pdf_reflect = lobe == Lobe.GLASS ? average(R) / sum_rt : 1f0
    refract = uc >= pdf_reflect
    if refract
        sin2_t = (1f0 - cos_hi^2) / c.ior^2
        sin2_t >= 1f0 && return none
        cos_t = sqrt(1f0 - sin2_t)
        wi = -wo / c.ior + (cos_hi / c.ior - cos_t) * H
    else
        wi = 2f0 * cos_hi * H - wo
    end
    (wi[3] < 0f0) == refract || return none
    singular = singular || (refract && abs(c.ior - 1f0) < 1f-4)
    eta = refract ? c.ior : 1f0
    if singular
        coeff = refract ? T : R
        lobe_pdf = refract ? 1f0 - pdf_reflect : pdf_reflect
        flags = refract ? BXDF_SPECULAR_TRANSMISSION : BXDF_SPECULAR_REFLECTION
        return (wi, c.weight * coeff * (c.energy / max(abs(wi[3]), 1f-6)), lobe_pdf, flags, eta)
    end
    return (wi, SpectralRadiance(0f0), 0f0, refract ? BXDF_GLOSSY_TRANSMISSION : BXDF_GLOSSY_REFLECTION, eta)
end

"""
`f` of closure `c` in Hikari's convention, which leaves the cosine to the integrator and takes it
with the shading normal `ns`: the lobe's own cosine, with its normal, over that one.
"""
@propagate_inbounds function _closure_f(c::SpectralClosure, wo::Vec3f, wi::Vec3f, ns::Vec3f, lambda::Wavelengths)
    n, t, b = _frame(c, wo)
    f, pdf = _lobe_eval(c, world_to_local(wo, n, t, b), world_to_local(wi, n, t, b), lambda)
    cos_s = abs(dot(wi, ns))
    cos_s > 1f-7 || return (SpectralRadiance(0f0), 0f0)
    return (f * (abs(dot(wi, n)) / cos_s), pdf)
end

@propagate_inbounds function _mixture(bsdf::ClosureBSDF, wo::Vec3f, wi::Vec3f, ns::Vec3f)
    f = SpectralRadiance(0f0)
    pdf = 0f0
    for i in Int32(1):bsdf.count
        c = _closure_at(bsdf.closures, i)
        fi, pi = _closure_f(c, wo, wi, ns, bsdf.lambda)
        f += fi
        pdf += pi * c.sample_weight
    end
    return f, pdf / bsdf.total_sample_weight
end

@propagate_inbounds function evaluate_bsdf_spectral(
    bsdf::ClosureBSDF, ::RGBToSpectrumTable, textures,
    wo::Vec3f, wi::Vec3f, n::Vec3f, dpdus::Vec3f, ::TextureFilterContext, ::Wavelengths,
    ::Bool = false,
)
    bsdf.total_sample_weight > 0f0 || return (SpectralRadiance(0f0), 0f0)
    return _mixture(bsdf, wo, wi, n)
end

@propagate_inbounds function sample_bsdf_spectral(
    bsdf::ClosureBSDF, ::RGBToSpectrumTable, textures,
    wo::Vec3f, n::Vec3f, dpdus::Vec3f, ::TextureFilterContext,
    ::Wavelengths, sample_u::Point2f, rng::Float32,
    ::Bool = false,
)
    total = bsdf.total_sample_weight
    total > 0f0 || return SpectralBSDFSample()

    target = rng * total
    chosen = Int32(1)
    before = 0f0
    acc = 0f0
    for i in Int32(1):bsdf.count
        sw = _closure_at(bsdf.closures, i).sample_weight
        if acc <= target
            chosen = i
            before = acc
        end
        acc += sw
    end
    c = _closure_at(bsdf.closures, chosen)
    uc = clamp((target - before) / max(c.sample_weight, 1f-12), 0f0, 0.99999994f0)

    cn, t, b = _frame(c, wo)
    wi_local, f_singular, pdf_singular, flags, eta = _lobe_sample(c, world_to_local(wo, cn, t, b), sample_u, uc,
                                                                   bsdf.lambda)
    flags == 0x00 && return SpectralBSDFSample()
    wi = normalize(local_to_world(wi_local, cn, t, b))
    side = dot(wo, n) * dot(wi, n) < 0f0 ? BXDF_TRANSMISSION : BXDF_REFLECTION
    flags = (flags & ~(BXDF_TRANSMISSION | BXDF_REFLECTION)) | side

    if is_specular(flags)
        cos_s = abs(dot(wi, n))
        cos_s > 1f-7 || return SpectralBSDFSample()
        f = f_singular * (abs(dot(wi, cn)) / cos_s)
        return SpectralBSDFSample(f, wi, pdf_singular * c.sample_weight / total, flags, eta, false, bsdf.dispersive)
    end
    f, pdf = _mixture(bsdf, wo, wi, n)
    pdf > 0f0 || return SpectralBSDFSample()
    return SpectralBSDFSample(f, wi, pdf, flags, eta, false, bsdf.dispersive && is_transmissive(flags))
end

# ── Alpha and guides ────────────────────────────────────────────────────────

@propagate_inbounds get_surface_alpha(::ShaderMaterial{P, false}, textures, tfc::TextureFilterContext) where {P} = 1f0
@propagate_inbounds function get_surface_alpha(mat::ShaderMaterial{P, true}, textures, tfc::TextureFilterContext) where {P}
    sc = ShaderContext(tfc, tfc.n, Vec3f(0), Vec3f(0), Vec3f(0), tfc.n)
    return clamp(1f0 - _avg(eval_shader_handle(textures, mat, sc).transparency), 0f0, 1f0)
end

@propagate_inbounds guide_material(mat::ShaderMaterial, materials, tfc, ::Val{D}) where {D} =
    guide_rgb(closure_albedo(eval_shader_handle(materials, mat,
                                                ShaderContext(tfc, tfc.n, Vec3f(0), Vec3f(0), Vec3f(0), tfc.n))))

"""The colour the closures reflect in all, for the denoiser's albedo guide."""
@inline function closure_albedo(cl::ClosureList)
    a = Vec3f(0)
    for i in Int32(1):cl.count
        c = _closure_at(cl.closures, i)
        a += c.lobe == Lobe.MICROFACET && c.fresnel == Fresnel.F82_TINT ? c.weight .* c.c1 :
             c.lobe == Lobe.MICROFACET && c.fresnel == Fresnel.NONE ? c.weight :
             c.lobe == Lobe.MICROFACET ? Vec3f(0) : c.weight
    end
    return RGBSpectrum(a[1], a[2], a[3])
end
