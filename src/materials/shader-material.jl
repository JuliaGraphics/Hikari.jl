# ============================================================================
# ShaderMaterial: a material that is a shader graph
# ============================================================================

"""
    ShaderMaterial(shader)

A material made of a shader graph, as Blender's Material Output takes one: BSDF nodes such as
[`principled_bsdf`](@ref TextureNodes.principled_bsdf), combined by `mix_shader` and
`add_shader`, with node textures and `bump` at any of their inputs. The graph is compiled to one
program, so every graph is the same material type.
"""
struct ShaderMaterial{ProgT} <: Material
    program::ProgT
end

ShaderMaterial(shader::ShaderNode) = ShaderMaterial(ShaderGraph(shader))

"""A closure with its colours at the path's wavelengths, and its share of the BSDF's samples."""
struct SpectralClosure
    lobe::UInt8
    fresnel::UInt8
    weight::SpectralRadiance
    f0::SpectralRadiance
    tint::SpectralRadiance
    n::Vec3f
    alpha::Float32
    ior::Float32
    sample_weight::Float32
end

SpectralClosure() = SpectralClosure(0x00, 0x00, SpectralRadiance(0f0), SpectralRadiance(0f0), SpectralRadiance(0f0),
                                    Vec3f(0, 0, 1), 0f0, 1f0, 0f0)

"""The closures a shader produced at a hit, sampled and evaluated as one BSDF."""
struct ClosureBSDF
    count::Int32
    total_sample_weight::Float32
    closures::NTuple{MAX_CLOSURES, SpectralClosure}
end

@propagate_inbounds function shader_context(work, camera, samples_per_pixel::Int32)
    dpdx, dpdy = approximate_dp_dxy(work.pi, work.n, camera, samples_per_pixel)
    dudx, dudy, dvdx, dvdy = compute_uv_derivatives(work.dpdu, work.dpdv, dpdx, dpdy)
    tfc = TextureFilterContext(work.uv, dudx, dudy, dvdx, dvdy, work.face_idx, work.bary, work.pi, work.ns)
    return ShaderContext(tfc, work.wo, dpdx, dpdy, work.dpdus, work.n)
end

@propagate_inbounds eval_shader_handle(textures, h::TexHandle, sc::ShaderContext) =
    Raycore.with_texture(_th_shader, textures, h.slot, h.idx, textures, sc)

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

@propagate_inbounds function material_bxdf(mat::ShaderMaterial, table, textures, work, camera,
                                           samples_per_pixel::Int32, lambda::Wavelengths, regularize::Bool)
    sc = shader_context(work, camera, samples_per_pixel)
    cl = eval_shader_handle(textures, mat.program, sc)
    return closure_bsdf(cl, table, lambda, sc.wo, regularize)
end

@inline _uplift(table, c::Vec3f, lambda) = uplift_rgb(table, RGBSpectrum(c[1], c[2], c[3]), lambda)
@inline _avg(c::Vec3f) = (c[1] + c[2] + c[3]) / 3f0

"""How much of the light from `wo` a closure returns, roughly: its share of the BSDF's samples."""
@inline function _sample_weight(c::RGBClosure, wo::Vec3f)
    w = _avg(c.weight)
    c.lobe == Lobe.MICROFACET || return w
    c.fresnel == Fresnel.CONSTANT && return w * _avg(c.f0)
    c.fresnel == Fresnel.F82_TINT && return w * max(_avg(c.f0), 0.05f0)
    return w * _avg(_dielectric_albedo(c.f0, c.ior, abs(dot(wo, c.n))))
end

@propagate_inbounds function closure_bsdf(cl::ClosureList, table, lambda::Wavelengths, wo::Vec3f, regularize::Bool)
    sample_weights = ntuple(k -> Int32(k) <= cl.count ? _sample_weight(cl.closures[k], wo) : 0f0, Val(MAX_CLOSURES))
    closures = ntuple(Val(MAX_CLOSURES)) do k
        c = cl.closures[k]
        Int32(k) <= cl.count || return SpectralClosure()
        alpha = regularize && c.lobe == Lobe.MICROFACET ? regularize_alpha(c.alpha) : c.alpha
        weight = _uplift(table, _saturate3(c.weight), lambda)
        f0 = c.lobe == Lobe.MICROFACET ? _uplift(table, c.f0, lambda) : SpectralRadiance(1f0)
        tint = c.fresnel == Fresnel.F82_TINT ? _uplift(table, c.tint, lambda) : SpectralRadiance(1f0)
        SpectralClosure(c.lobe, c.fresnel, weight, f0, tint, c.n, alpha, c.ior, sample_weights[k])
    end
    return ClosureBSDF(cl.count, sum(sample_weights), closures)
end

# ── Lobes, in a frame around the closure's own normal ──────────────────────

@inline function _fresnel(c::SpectralClosure, cos_i::Float32)
    c.fresnel == Fresnel.CONSTANT && return c.f0
    one = SpectralRadiance(1f0)
    if c.fresnel == Fresnel.F82_TINT
        f = 6f0 / 7f0
        f5 = f^5
        b = (c.f0 + (one - c.f0) * f5) * (one - c.tint) * (7f0 / (f5 * f))
        m = clamp(1f0 - cos_i, 0f0, 1f0)
        m5 = m^5
        return clamp(c.f0 + (one - c.f0) * m5 - b * (cos_i * m5 * m), 0f0, 1f0)
    end
    return c.f0 + (one - c.f0) * _schlick_shape(cos_i, c.ior)
end

@inline _is_delta(c::SpectralClosure) = c.lobe == Lobe.MICROFACET && trowbridge_reitz_effectively_smooth(c.alpha, c.alpha)

"""`f` (without the cosine) and pdf of closure `c` for the local directions `wo`, `wi`."""
@propagate_inbounds function _lobe_eval(c::SpectralClosure, wo::Vec3f, wi::Vec3f)
    if c.lobe == Lobe.DIFFUSE
        wi[3] > 0f0 || return (SpectralRadiance(0f0), 0f0)
        return (c.weight * (1f0 / Float32(π)), wi[3] / Float32(π))
    elseif c.lobe == Lobe.TRANSLUCENT
        wi[3] < 0f0 || return (SpectralRadiance(0f0), 0f0)
        return (c.weight * (1f0 / Float32(π)), -wi[3] / Float32(π))
    end
    (wi[3] > 0f0 && !_is_delta(c)) || return (SpectralRadiance(0f0), 0f0)
    wm = normalize(wo + wi)
    D = trowbridge_reitz_d(wm, c.alpha, c.alpha)
    G = trowbridge_reitz_g(wo, wi, c.alpha, c.alpha)
    F = _fresnel(c, abs(dot(wo, wm)))
    f = c.weight * F * (D * G / (4f0 * wo[3] * wi[3]))
    pdf = trowbridge_reitz_pdf(wo, wm, c.alpha, c.alpha) / (4f0 * abs(dot(wo, wm)))
    return (f, pdf)
end

"""A direction sampled from closure `c` in local space: `(wi, f, pdf, flags)`, with `f` and pdf only for a delta lobe."""
@propagate_inbounds function _lobe_sample(c::SpectralClosure, wo::Vec3f, u::Point2f)
    if c.lobe == Lobe.DIFFUSE || c.lobe == Lobe.TRANSLUCENT
        wi = cosine_sample_hemisphere(u)
        c.lobe == Lobe.TRANSLUCENT && (wi = Vec3f(wi[1], wi[2], -wi[3]))
        flags = c.lobe == Lobe.DIFFUSE ? BXDF_DIFFUSE_REFLECTION : BXDF_DIFFUSE_TRANSMISSION
        return (wi, SpectralRadiance(0f0), 0f0, flags)
    end
    if _is_delta(c)
        wi = Vec3f(-wo[1], -wo[2], wo[3])
        f = c.weight * _fresnel(c, wo[3]) * (1f0 / max(wi[3], 1f-6))
        return (wi, f, 1f0, BXDF_SPECULAR_REFLECTION)
    end
    wm = trowbridge_reitz_sample_wm(wo, u, c.alpha, c.alpha)
    wi = -wo + 2f0 * dot(wo, wm) * wm
    return (wi, SpectralRadiance(0f0), 0f0, BXDF_GLOSSY_REFLECTION)
end

@inline function _frame(n::Vec3f, wo::Vec3f)
    n = dot(n, wo) < 0f0 ? -n : n
    t, b = coordinate_system(n)
    return n, t, b
end

"""
`f` of closure `c` in Hikari's convention, which leaves the cosine to the integrator and takes it
with the shading normal `ns`: the lobe's own cosine, with its normal, over that one.
"""
@propagate_inbounds function _closure_f(c::SpectralClosure, wo::Vec3f, wi::Vec3f, ns::Vec3f)
    n, t, b = _frame(c.n, wo)
    f, pdf = _lobe_eval(c, world_to_local(wo, n, t, b), world_to_local(wi, n, t, b))
    cos_s = abs(dot(wi, ns))
    cos_s > 1f-7 || return (SpectralRadiance(0f0), 0f0)
    return (f * (abs(dot(wi, n)) / cos_s), pdf)
end

"""The reflection or transmission side `wi` is on, seen from `wo` across the shading normal."""
@inline _same_side(wo::Vec3f, wi::Vec3f, ns::Vec3f) = dot(wo, ns) * dot(wi, ns) > 0f0

@propagate_inbounds function _mixture(bsdf::ClosureBSDF, wo::Vec3f, wi::Vec3f, ns::Vec3f)
    f = SpectralRadiance(0f0)
    pdf = 0f0
    for i in Int32(1):bsdf.count
        c = _closure_at(bsdf.closures, i)
        reflects = c.lobe != Lobe.TRANSLUCENT
        reflects == _same_side(wo, wi, ns) || continue
        fi, pi = _closure_f(c, wo, wi, ns)
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
    acc = 0f0
    for i in Int32(1):bsdf.count
        acc += _closure_at(bsdf.closures, i).sample_weight
        acc <= target && (chosen = min(i + Int32(1), bsdf.count))
    end
    c = _closure_at(bsdf.closures, chosen)

    cn, t, b = _frame(c.n, wo)
    wi_local, f_delta, pdf_delta, flags = _lobe_sample(c, world_to_local(wo, cn, t, b), sample_u)
    wi = normalize(local_to_world(wi_local, cn, t, b))
    reflects = c.lobe != Lobe.TRANSLUCENT
    reflects == _same_side(wo, wi, n) || return SpectralBSDFSample()

    if is_specular(flags)
        cos_s = abs(dot(wi, n))
        cos_s > 1f-7 || return SpectralBSDFSample()
        f = f_delta * (abs(dot(wi, cn)) / cos_s)
        return SpectralBSDFSample(f, wi, pdf_delta * c.sample_weight / total, flags, 1f0)
    end
    f, pdf = _mixture(bsdf, wo, wi, n)
    pdf > 0f0 || return SpectralBSDFSample()
    return SpectralBSDFSample(f, wi, pdf, flags, 1f0)
end

@propagate_inbounds guide_material(mat::ShaderMaterial, materials, tfc, ::Val{D}) where {D} =
    guide_rgb(closure_albedo(eval_shader_handle(materials, mat.program,
                                                ShaderContext(tfc, tfc.n, Vec3f(0), Vec3f(0), Vec3f(0), tfc.n))))

"""The colour the closures reflect in all, for the denoiser's albedo guide."""
@inline function closure_albedo(cl::ClosureList)
    a = Vec3f(0)
    for i in Int32(1):cl.count
        c = _closure_at(cl.closures, i)
        a += c.lobe == Lobe.MICROFACET && c.fresnel != Fresnel.DIELECTRIC ? c.weight .* c.f0 :
             c.lobe == Lobe.MICROFACET ? Vec3f(0) : c.weight
    end
    return RGBSpectrum(a[1], a[2], a[3])
end
