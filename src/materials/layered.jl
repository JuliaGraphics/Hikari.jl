# Ported from pbrt-v4: src/pbrt/bxdfs.h, `LayeredBxDF`. Copyright (c) 1998-2020 Matt Pharr, Wenzel
# Jakob, and Greg Humphreys. SPDX-License-Identifier: Apache-2.0. Modified: rewritten in Julia over
# Hikari's BSDF lobes. See LICENSES/Apache-2.0.txt.

# ============================================================================
# Layered BSDF: an interface over another, a medium between them
# ============================================================================
#
# pbrt-v4's `LayeredBxDF` estimates the BSDF of a stack of two interfaces stochastically, by random
# walks between them. Its interfaces are pbrt's BxDFs, which answer for directions on either side and
# under a restriction to reflection or transmission; here they are tuples of weighted lobes in the
# stack's local frame (`n = z`, `t = x`), answering the same way: `wo` below an interface is mirrored
# to the side the lobes are set up for, with their IOR inverted (`other_side`), and a restriction is
# a `Scattering`. pbrt's `!mode`, the importance a walk carries back up, is `adjoint`.

const LOCAL_N = Vec3f(0, 0, 1)
const LOCAL_T = Vec3f(1, 0, 0)

"""A lobe of a layered BSDF's interface: weighted by `weight`, in the stack's local frame."""
@inline local_lobe(lobe::BSDFLobe, weight::SpectralRadiance, sample_weight::Float32 = 1f0) =
    Weighted(lobe, weight, sample_weight, LOCAL_N, LOCAL_T)

@inline mirror(v::Vec3f) = Vec3f(v[1], v[2], -v[3])
@inline other_side(w::Weighted) = Weighted(other_side(w.lobe), w.weight, w.sample_weight, w.n, w.t)
@inline adjoint(w::Weighted) = Weighted(adjoint(w.lobe), w.weight, w.sample_weight, w.n, w.t)
@inline adjoint(lobes::Tuple) = map(adjoint, lobes)

"""An interface's lobes and directions as seen from the side `wo` is on."""
@inline from_above(lobes::Tuple, wo::Vec3f, wi::Vec3f) =
    wo[3] < 0f0 ? (map(other_side, lobes), mirror(wo), mirror(wi), true) : (lobes, wo, wi, false)

"""pbrt's `BxDF::f` of an interface's lobes: for `wo` on either side."""
@propagate_inbounds function interface_f(lobes::Tuple, wo::Vec3f, wi::Vec3f, lambda::Wavelengths)
    l, o, i, _ = from_above(lobes, wo, wi)
    return first(mixture(l, o, i, LOCAL_N, lambda))
end

"""pbrt's `BxDF::PDF` of an interface's lobes under `s`."""
@propagate_inbounds function interface_pdf(lobes::Tuple, wo::Vec3f, wi::Vec3f, lambda::Wavelengths,
                                           s::Scattering = AnyScattering())
    l, o, i, _ = from_above(lobes, wo, wi)
    total = share(l, s)
    total > 0f0 || return 0f0
    return last(mixture(l, o, i, LOCAL_N, lambda, s)) / total
end

"""
pbrt's `BxDF::Sample_f` of an interface's lobes under `s`: `(ok, wi, f, pdf, reflection, specular)`,
`f` and `pdf` of the sample, `wi` on the side pbrt's would be.
"""
@propagate_inbounds function interface_sample(lobes::Tuple, wo::Vec3f, uc::Float32, u::Point2f, lambda::Wavelengths,
                                              s::Scattering = AnyScattering())
    l, o, _, flipped = from_above(lobes, wo, wo)
    none = (false, Vec3f(0, 0, 1), SpectralRadiance(0f0), 0f0, false, false)
    total = share(l, s)
    total > 0f0 || return none
    wi, f, pdf, flags, _, sw, _, _ = pick_sample(l, uc * total, 0f0, o, u, lambda, s)
    flags == 0x00 && return none
    if is_specular(flags)
        pdf *= sw / total
    else
        f, pdf = mixture(l, o, wi, LOCAL_N, lambda, s)
        pdf /= total
    end
    out = flipped ? mirror(wi) : wi
    return (pdf > 0f0 && wi[3] != 0f0, out, f, pdf, !is_transmissive(flags), is_specular(flags))
end

"""Whether an interface is singular: its every lobe is, as pbrt's `IsSpecular(Flags())`."""
@inline interface_specular(lobes::Tuple) = reduce(&, map(w -> lobe_singular(w.lobe), lobes); init = true)
@inline lobe_singular(l::MicrofacetLobe) = singular(l)
@inline lobe_singular(::ThinDielectricLobe) = true
@inline lobe_singular(::BSDFLobe) = false

"""pbrt's `Tr`: the transmittance of the layer's unit-density medium over height `dz` along `w`."""
@inline layer_tr(dz::Float32, w::Vec3f) = abs(dz) <= floatmin(Float32) ? 1f0 : exp(-abs(dz / w[3]))

"""
pbrt-v4's `LayeredBxDF`: the interface `top` over `bottom` (tuples of weighted lobes, pbrt's
convention, in the local frame), `thickness` apart, with a medium of single-scattering `albedo` and
Henyey-Greenstein asymmetry `g` between them (none for a black albedo); estimated by `n_samples`
random walks of at most `max_depth` bounces. `TwoSided` mirrors it for light from below, as pbrt's
coated materials are. In the shading frame of `n` and `t`. `dispersive` where the interface's IOR is
a spectrum, taken at the path's first wavelength: the path's others end where it meets the BSDF, as
pbrt's coated materials end them in `GetBxDF` (`terminates_at_vertex`).
"""
struct LayeredBSDF{TwoSided, Top <: Tuple, Bottom <: Tuple}
    top::Top
    bottom::Bottom
    n::Vec3f
    t::Vec3f
    thickness::Float32
    albedo::SpectralRadiance
    g::Float32
    max_depth::Int32
    n_samples::Int32
    lambda::Wavelengths
    dispersive::Bool
end

"""
    LayeredBSDF{TwoSided}(top, bottom, ns, dpdus, thickness, albedo, g, max_depth, n_samples, lambda; dispersive)

The layered BSDF of `top` over `bottom` in the shading frame of `ns` and `dpdus`.
"""
@inline function LayeredBSDF{S}(top::Tuple, bottom::Tuple, ns::Vec3f, dpdus::Vec3f, thickness::Float32,
                                albedo::SpectralRadiance, g::Float32, max_depth::Int32, n_samples::Int32,
                                lambda::Wavelengths; dispersive::Bool) where {S}
    return LayeredBSDF{S, typeof(top), typeof(bottom)}(top, bottom, ns, first(shading_frame(ns, dpdus)), thickness,
                                                        albedo, g, max_depth, n_samples, lambda, dispersive)
end

@inline terminates_at_vertex(b::LayeredBSDF) = b.dispersive

@inline has_medium(b::LayeredBSDF) = !is_black(b.albedo)
@inline two_sided(::LayeredBSDF{S}) where {S} = S
@inline local_frame(b::LayeredBSDF) = (b.n, b.t, cross(b.n, b.t))

@inline uniform(rng) = ((x, r) = pcg32_uniform_f32(rng); (min(x, 0.99999994f0), r))

"""pbrt-v4's `LayeredBxDF::f`, in the stack's local frame."""
@propagate_inbounds function layered_f(b::LayeredBSDF, wo::Vec3f, wi::Vec3f)
    lambda = b.lambda
    f = SpectralRadiance(0f0)
    if two_sided(b) && wo[3] < 0f0
        wo, wi = -wo, -wi
    end
    entered_top = two_sided(b) || wo[3] > 0f0
    same = wo[3] * wi[3] > 0f0
    # The exit interface is the bottom where the walk leaves on the side it entered from below, or
    # the other way round.
    exit_bottom = same ⊻ entered_top
    exit_z = exit_bottom ? 0f0 : b.thickness
    enter = entered_top ? b.top : b.bottom
    exit, non_exit = exit_bottom ? (b.bottom, b.top) : (b.top, b.bottom)
    exit_specular, non_exit_specular = interface_specular(exit), interface_specular(non_exit)
    same && (f += Float32(b.n_samples) * interface_f(enter, wo, wi, lambda))

    rng = pcg32_init(pbrt_hash(UInt64(0), wo), pbrt_hash(wi))
    for _ in 1:b.n_samples
        # Into the layers through the entrance interface, and the light's way out through the exit one.
        uc, rng = uniform(rng); u1, rng = uniform(rng); u2, rng = uniform(rng)
        ok, wos_wi, wos_f, wos_pdf, _, _ = interface_sample(enter, wo, uc, Point2f(u1, u2), lambda, OnlyTransmission())
        (ok && !is_black(wos_f)) || continue
        uc, rng = uniform(rng); u1, rng = uniform(rng); u2, rng = uniform(rng)
        ok, wis_wi, wis_f, wis_pdf, _, _ = interface_sample(adjoint(exit), wi, uc, Point2f(u1, u2), lambda, OnlyTransmission())
        (ok && !is_black(wis_f)) || continue

        beta = wos_f * abs(wos_wi[3]) / wos_pdf
        z = entered_top ? b.thickness : 0f0
        w = wos_wi
        for depth in 0:(Int(b.max_depth) - 1)
            if depth > 3 && max_component(beta) < 0.25f0
                q = max(0f0, 1f0 - max_component(beta))
                rr, rng = uniform(rng)
                rr < q && break
                beta = beta / (1f0 - q)
            end
            if !has_medium(b)
                z = z == b.thickness ? 0f0 : b.thickness
                beta = beta * layer_tr(b.thickness, w)
            else
                ue, rng = uniform(rng)
                dz = sample_exponential(ue, 1f0 / abs(w[3]))
                zp = w[3] > 0f0 ? z + dz : z - dz
                zp == z && continue
                if 0f0 < zp < b.thickness
                    # A scattering event in the medium: the light's way out through the exit, then onwards.
                    wt = exit_specular ? 1f0 : power_heuristic(1, wis_pdf, 1, hg_phase_pdf(b.g, dot(-w, -wis_wi)))
                    f += beta * b.albedo * hg_phase_pdf(b.g, dot(-w, -wis_wi)) * wt * layer_tr(zp - exit_z, wis_wi) *
                         wis_f / wis_pdf
                    p1, rng = uniform(rng); p2, rng = uniform(rng)
                    ps_wi, ps_p = sample_hg_phase_spectral(b.g, -w, Point2f(p1, p2))
                    (ps_p > 0f0 && ps_wi[3] != 0f0) || continue
                    beta = beta * b.albedo     # p / pdf is one for Henyey-Greenstein
                    w = ps_wi
                    z = zp
                    if ((z < exit_z && w[3] > 0f0) || (z > exit_z && w[3] < 0f0)) && !exit_specular
                        f_exit = interface_f(exit, -w, wi, lambda)
                        if !is_black(f_exit)
                            exit_pdf = interface_pdf(exit, -w, wi, lambda, OnlyTransmission())
                            f += beta * layer_tr(zp - exit_z, ps_wi) * f_exit * power_heuristic(1, ps_p, 1, exit_pdf)
                        end
                    end
                    continue
                end
                z = clamp(zp, 0f0, b.thickness)
            end
            if z == exit_z
                # The path straight through both interfaces, on the walk's first arrival at the exit.
                # pbrt's walk only reflects there, which loses no light through its coated materials,
                # whose bottoms are opaque, but all of it through a transmissive bottom; no other
                # strategy reaches this path, so it is added at weight one. It is zero for an opaque
                # bottom, so pbrt's materials are unchanged.
                if depth == 0 && !exit_specular
                    f += beta * interface_f(exit, -w, wi, lambda)
                end
                # Reflection at the exit interface.
                uc, rng = uniform(rng); u1, rng = uniform(rng); u2, rng = uniform(rng)
                ok, bs_wi, bs_f, bs_pdf, _, _ = interface_sample(exit, -w, uc, Point2f(u1, u2), lambda, OnlyReflection())
                (ok && !is_black(bs_f)) || break
                beta = beta * bs_f * abs(bs_wi[3]) / bs_pdf
                w = bs_wi
            else
                # Scattering at the other interface, with the light's way out along `wis` and along its own sample.
                if !non_exit_specular
                    wt = exit_specular ? 1f0 : power_heuristic(1, wis_pdf, 1, interface_pdf(non_exit, -w, -wis_wi, lambda))
                    f += beta * interface_f(non_exit, -w, -wis_wi, lambda) * abs(wis_wi[3]) * wt *
                         layer_tr(b.thickness, wis_wi) * wis_f / wis_pdf
                end
                uc, rng = uniform(rng); u1, rng = uniform(rng); u2, rng = uniform(rng)
                ok, bs_wi, bs_f, bs_pdf, _, _ = interface_sample(non_exit, -w, uc, Point2f(u1, u2), lambda, OnlyReflection())
                (ok && !is_black(bs_f)) || break
                beta = beta * bs_f * abs(bs_wi[3]) / bs_pdf
                w = bs_wi
                if !exit_specular
                    f_exit = interface_f(exit, -w, wi, lambda)
                    if !is_black(f_exit)
                        wt = non_exit_specular ? 1f0 :
                             power_heuristic(1, bs_pdf, 1, interface_pdf(exit, -w, wi, lambda, OnlyTransmission()))
                        f += beta * layer_tr(b.thickness, bs_wi) * f_exit * wt
                    end
                end
            end
        end
    end
    return f / Float32(b.n_samples)
end

"""
pbrt-v4's `LayeredBxDF::Sample_f`, in the stack's local frame: `(ok, wi, f, pdf, flags)`, the pdf
proportional only.
"""
@propagate_inbounds function layered_sample(b::LayeredBSDF, wo::Vec3f, uc::Float32, u::Point2f)
    lambda = b.lambda
    none = (false, Vec3f(0, 0, 1), SpectralRadiance(0f0), 0f0, 0x00)
    flip = false
    if two_sided(b) && wo[3] < 0f0
        wo = -wo
        flip = true
    end
    entered_top = two_sided(b) || wo[3] > 0f0
    ok, w, f, pdf, reflection, specular_path = interface_sample(entered_top ? b.top : b.bottom, wo, uc, u, lambda)
    (ok && !is_black(f)) || return none
    if reflection
        flags = specular_path ? BXDF_SPECULAR_REFLECTION : BXDF_GLOSSY_REFLECTION
        return (true, flip ? -w : w, f, pdf, flags)
    end
    rng = pcg32_init(pbrt_hash(UInt64(0), wo), pbrt_hash(uc, u))
    f = f * abs(w[3])
    z = entered_top ? b.thickness : 0f0
    for depth in 0:(Int(b.max_depth) - 1)
        rr_beta = max_component(f) / pdf
        if depth > 3 && rr_beta < 0.25f0
            q = max(0f0, 1f0 - rr_beta)
            rr, rng = uniform(rng)
            rr < q && return none
            pdf *= 1f0 - q
        end
        w[3] == 0f0 && return none
        if has_medium(b)
            ue, rng = uniform(rng)
            dz = sample_exponential(ue, 1f0 / abs(w[3]))
            zp = w[3] > 0f0 ? z + dz : z - dz
            zp == z && return none
            if 0f0 < zp < b.thickness
                p1, rng = uniform(rng); p2, rng = uniform(rng)
                ps_wi, ps_p = sample_hg_phase_spectral(b.g, -w, Point2f(p1, p2))
                (ps_p > 0f0 && ps_wi[3] != 0f0) || return none
                f = f * b.albedo * ps_p
                pdf *= ps_p
                specular_path = false
                w = ps_wi
                z = zp
                continue
            end
            z = clamp(zp, 0f0, b.thickness)
        else
            z = z == b.thickness ? 0f0 : b.thickness
            f = f * layer_tr(b.thickness, w)
        end
        uc2, rng = uniform(rng); u1, rng = uniform(rng); u2, rng = uniform(rng)
        ok, bs_wi, bs_f, bs_pdf, bs_reflection, bs_specular =
            interface_sample(z == 0f0 ? b.bottom : b.top, -w, uc2, Point2f(u1, u2), lambda)
        (ok && !is_black(bs_f)) || return none
        f = f * bs_f
        pdf *= bs_pdf
        specular_path &= bs_specular
        w = bs_wi
        if !bs_reflection
            out_reflection = wo[3] * w[3] > 0f0
            flags = specular_path ? (out_reflection ? BXDF_SPECULAR_REFLECTION : BXDF_SPECULAR_TRANSMISSION) :
                                    (out_reflection ? BXDF_GLOSSY_REFLECTION : BXDF_GLOSSY_TRANSMISSION)
            return (true, flip ? -w : w, f, pdf, flags)
        end
        f = f * abs(bs_wi[3])
    end
    return none
end

"""pbrt-v4's `LayeredBxDF::PDF`, in the stack's local frame: mixed with the uniform sphere's, as pbrt's."""
@propagate_inbounds function layered_pdf(b::LayeredBSDF, wo::Vec3f, wi::Vec3f)
    lambda = b.lambda
    if two_sided(b) && wo[3] < 0f0
        wo, wi = -wo, -wi
    end
    rng = pcg32_init(pbrt_hash(UInt64(0), wi), pbrt_hash(wo))
    entered_top = two_sided(b) || wo[3] > 0f0
    same = wo[3] * wi[3] > 0f0
    pdf_sum = 0f0
    same && (pdf_sum += Float32(b.n_samples) *
                        interface_pdf(entered_top ? b.top : b.bottom, wo, wi, lambda, OnlyReflection()))
    for _ in 1:b.n_samples
        if same
            # The TRT term: in through `t`, off `r`, out through `t`.
            r_iface, t_iface = entered_top ? (b.bottom, b.top) : (b.top, b.bottom)
            uc, rng = uniform(rng); u1, rng = uniform(rng); u2, rng = uniform(rng)
            ok_o, wos_wi, wos_f, wos_pdf, _, _ = interface_sample(t_iface, wo, uc, Point2f(u1, u2), lambda, OnlyTransmission())
            uc, rng = uniform(rng); u1, rng = uniform(rng); u2, rng = uniform(rng)
            ok_i, wis_wi, wis_f, wis_pdf, _, _ =
                interface_sample(adjoint(t_iface), wi, uc, Point2f(u1, u2), lambda, OnlyTransmission())
            if ok_o && !is_black(wos_f) && ok_i && !is_black(wis_f)
                if interface_specular(t_iface)
                    pdf_sum += interface_pdf(r_iface, -wos_wi, -wis_wi, lambda)
                else
                    uc, rng = uniform(rng); u1, rng = uniform(rng); u2, rng = uniform(rng)
                    ok_r, rs_wi, rs_f, rs_pdf, _, _ = interface_sample(r_iface, -wos_wi, uc, Point2f(u1, u2), lambda)
                    if ok_r && !is_black(rs_f)
                        if interface_specular(r_iface)
                            pdf_sum += interface_pdf(t_iface, -rs_wi, wi, lambda)
                        else
                            r_pdf = interface_pdf(r_iface, -wos_wi, -wis_wi, lambda)
                            pdf_sum += power_heuristic(1, wis_pdf, 1, r_pdf) * r_pdf
                            t_pdf = interface_pdf(t_iface, -rs_wi, wi, lambda)
                            pdf_sum += power_heuristic(1, rs_pdf, 1, t_pdf) * t_pdf
                        end
                    end
                end
            end
        else
            # The TT term: through the one interface and out through the other.
            to_iface, ti_iface = entered_top ? (b.top, b.bottom) : (b.bottom, b.top)
            uc, rng = uniform(rng); u1, rng = uniform(rng); u2, rng = uniform(rng)
            ok_o, wos_wi, wos_f, wos_pdf, wos_reflection, _ = interface_sample(to_iface, wo, uc, Point2f(u1, u2), lambda)
            (ok_o && !is_black(wos_f) && !wos_reflection) || continue
            uc, rng = uniform(rng); u1, rng = uniform(rng); u2, rng = uniform(rng)
            ok_i, wis_wi, wis_f, wis_pdf, wis_reflection, _ = interface_sample(adjoint(ti_iface), wi, uc, Point2f(u1, u2), lambda)
            (ok_i && !is_black(wis_f) && !wis_reflection) || continue
            if interface_specular(to_iface)
                pdf_sum += interface_pdf(ti_iface, -wos_wi, wi, lambda)
            elseif interface_specular(ti_iface)
                pdf_sum += interface_pdf(to_iface, wo, -wis_wi, lambda)
            else
                pdf_sum += (interface_pdf(to_iface, wo, -wis_wi, lambda) + interface_pdf(ti_iface, -wos_wi, wi, lambda)) / 2f0
            end
        end
    end
    return 0.9f0 * (pdf_sum / Float32(b.n_samples)) + 0.1f0 / (4f0 * Float32(π))
end

@propagate_inbounds function evaluate_bsdf_spectral(
    b::LayeredBSDF, ::RGBToSpectrumTable, textures,
    wo::Vec3f, wi::Vec3f, n::Vec3f, dpdus::Vec3f, ::TextureFilterContext, ::Wavelengths,
    ::Bool = false,
)
    ln, lt, lb = local_frame(b)
    o, i = world_to_local(wo, ln, lt, lb), world_to_local(wi, ln, lt, lb)
    (o[3] == 0f0 || i[3] == 0f0) && return (SpectralRadiance(0f0), 0f0)
    return (layered_f(b, o, i), layered_pdf(b, o, i))
end

@propagate_inbounds function sample_bsdf_spectral(
    b::LayeredBSDF, ::RGBToSpectrumTable, textures,
    wo::Vec3f, n::Vec3f, dpdus::Vec3f, ::TextureFilterContext,
    ::Wavelengths, sample_u::Point2f, uc::Float32,
    ::Bool = false,
)
    ln, lt, lb = local_frame(b)
    ok, wi, f, pdf, flags = layered_sample(b, world_to_local(wo, ln, lt, lb), uc, sample_u)
    ok || return SpectralBSDFSample()
    # The pdf is proportional only: the integrator asks `evaluate_bsdf_spectral` for the real one.
    return SpectralBSDFSample(f, normalize(local_to_world(wi, ln, lt, lb)), pdf, flags, 1f0, true, false)
end
