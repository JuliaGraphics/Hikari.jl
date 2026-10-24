# Ported from Blender Cycles v5.2.1: intern/cycles/kernel/integrator/subsurface.h and
# subsurface_random_walk.h. Copyright 2011-2022 Blender Foundation. SPDX-License-Identifier:
# Apache-2.0. Modified: rewritten in Julia, over the path's wavelengths instead of RGB channels.
# See LICENSES/Apache-2.0.txt.

# ============================================================================
# Random walk subsurface scattering
# ============================================================================

const BSSRDF_MAX_BOUNCES = 256
const SUBSURFACE_SIMILARITY_LEVEL = 9
const VOLUME_THROUGHPUT_EPSILON = 1f-6

"""
    subsurface_scatter(bsdf, work, accel, sample_idx) -> (bsdf, work)

Pick between a shader's BSDF and its subsurface closure, in proportion to their sample weights,
as Cycles' `surface_shader_bsdf_bssrdf_pick` does. Picking the BSDF leaves it without the
subsurface closure and the path's throughput divided by its chance; picking subsurface walks
inside the object to where the light leaves it, and returns a diffuse BSDF there with the work
item moved to that point, as Cycles' `subsurface_shader_data_setup`. A walk that does not get out
returns a BSDF of no closures, which ends the path.
"""
@inline subsurface_scatter(bsdf, work, accel, sample_idx) = (bsdf, work)

@propagate_inbounds function subsurface_scatter(bsdf::ClosureBSDF, work, accel, sample_idx::Int32)
    k = _subsurface_index(bsdf)
    k == Int32(0) && return (bsdf, work)
    sss = _closure_at(bsdf.closures, k)
    total = bsdf.total_sample_weight
    rng = pcg32_init(UInt64(work.pixel_index) << 32 | UInt64(reinterpret(UInt32, sample_idx)), UInt64(work.depth) + UInt64(0x5353))
    u, rng = pcg32_uniform_f32(rng)
    pick = sss.sample_weight / total
    if u >= pick
        rest = total - sss.sample_weight
        rest > 0f0 || return (_no_bsdf(bsdf), work)
        closures = ntuple(i -> Int32(i) == k ? _without_samples(bsdf.closures[i]) : bsdf.closures[i], Val(MAX_CLOSURES))
        reduced = ClosureBSDF(bsdf.count, rest, closures, bsdf.lambda, bsdf.emission, bsdf.dispersive)
        return (reduced, _with_beta(work, work.beta * (total / rest)))
    end
    return _random_walk(sss, bsdf, work, accel, rng, total / sss.sample_weight)
end

@inline function _subsurface_index(bsdf::ClosureBSDF)
    k = Int32(0)
    for i in Int32(1):bsdf.count
        _closure_at(bsdf.closures, i).lobe == Lobe.SUBSURFACE && (k = i)
    end
    return k
end

@inline _without_samples(c::SpectralClosure) =
    SpectralClosure(c.lobe, c.fresnel, c.weight, c.c1, c.c2, c.c3, c.n, c.t, c.alpha_x, c.alpha_y, c.ior, c.energy,
                    c.p1, c.p2, 0f0)

@inline _no_bsdf(bsdf::ClosureBSDF) = ClosureBSDF(Int32(0), 0f0, bsdf.closures, bsdf.lambda, bsdf.emission, false)

"""A diffuse BSDF of weight one around `n`: what leaves an object where a subsurface walk gets out of it."""
@inline function _exit_bsdf(bsdf::ClosureBSDF, n::Vec3f)
    exit = SpectralClosure(Lobe.DIFFUSE, Fresnel.NONE, SpectralRadiance(1f0), SpectralRadiance(1f0), SpectralRadiance(1f0),
                           SpectralRadiance(1f0), n, Vec3f(0), 0f0, 0f0, 1f0, 1f0, 0f0, 0f0, 1f0)
    closures = ntuple(i -> i == 1 ? exit : SpectralClosure(), Val(MAX_CLOSURES))
    return ClosureBSDF(Int32(1), 1f0, closures, bsdf.lambda, SpectralRadiance(0f0), false)
end

@inline _with_beta(w::VPMaterialEvalWorkItem, beta::SpectralRadiance) =
    VPMaterialEvalWorkItem(w.pi, w.ps, w.n, w.dpdu, w.dpdv, w.time, w.ns, w.dpdus, w.dpdvs, w.uv, w.face_idx, w.bary,
                           w.wo, w.material_idx, w.interface, w.lambda, w.pixel_index, beta, w.r_u, w.r_l, w.depth,
                           w.eta_scale, w.specular_bounce, w.any_non_specular_bounces, w.prev_intr_p, w.prev_intr_n,
                           w.current_medium)

@inline _at_exit(w::VPMaterialEvalWorkItem, geom, face_idx::UInt32, bary, beta::SpectralRadiance) =
    VPMaterialEvalWorkItem(geom.pi, geom.ps, geom.n, geom.dpdu, geom.dpdv, w.time, geom.ns, geom.dpdus, geom.dpdvs,
                           geom.uv, face_idx, bary, geom.ns, w.material_idx, w.interface, w.lambda, w.pixel_index, beta,
                           w.r_u, w.r_l, w.depth, w.eta_scale, false, true, w.prev_intr_p, w.prev_intr_n,
                           w.current_medium)

"""Cycles' `subsurface_entry_bounce`: a GGX refraction of the view into the surface."""
@inline function _subsurface_entry(c::SpectralClosure, wo::Vec3f, u::Point2f)
    n = c.n
    cos_ni = dot(n, wo)
    cos_ni > 0f0 || return (false, Vec3f(0))
    t, b = coordinate_system(n)
    local_h = trowbridge_reitz_sample_wm(world_to_local(wo, n, t, b), u, max(c.alpha_x, 1f-4), max(c.alpha_x, 1f-4))
    h = local_to_world(local_h, n, t, b)
    cos_hi = dot(h, wo)
    neta = 1f0 / c.ior
    dnp = max(sqrt(max(1f0 - neta^2 * (1f0 - cos_hi^2), 0f0)), 1f-7)
    return (true, normalize(-(neta * wo) + (neta * cos_hi - dnp) * h))
end

"""d'Eon's mapping from a subsurface colour and anisotropy to the single-scattering albedo, as Cycles."""
@inline function _walk_albedo(color::Float32, g::Float32)
    s = 4.20863f0 * color - sqrt(9.59217f0 + 41.6808f0 * color + 17.7126f0 * color^2) + 4.09712f0
    s2 = s * s
    return clamp((1f0 - s2) / (1f0 - g * s2), 0f0, 0.999999f0)
end

@inline _map4(f, a::SpectralRadiance) = SpectralRadiance(ntuple(k -> f(a.data[k]), Val(4)))
@inline _map4(f, a::SpectralRadiance, b::SpectralRadiance) = SpectralRadiance(ntuple(k -> f(a.data[k], b.data[k]), Val(4)))
@inline _maximum(a::SpectralRadiance) = max(a.data...)
@inline _minimum(a::SpectralRadiance) = min(a.data...)
@inline _dot4(a::SpectralRadiance, b::SpectralRadiance) = sum(ntuple(k -> a.data[k] * b.data[k], Val(4)))

@inline _phase_dwivedi(v::Float32, phase_log::Float32, cos_theta::Float32) = 1f0 / ((v - cos_theta) * phase_log)
@inline _sample_dwivedi(v::Float32, phase_log::Float32, u::Float32) = v - (v + 1f0) * exp(-u * phase_log)

@inline function _direction_from_cosine(d::Vec3f, cos_theta::Float32, u::Float32)
    sin_theta = sqrt(max(1f0 - cos_theta^2, 0f0))
    s, c = sincos(2f0 * Float32(π) * u)
    t, b = coordinate_system(d)
    return sin_theta * c * t + sin_theta * s * b + cos_theta * d
end

@inline function _hg_sample(d::Vec3f, g::Float32, u::Point2f)
    cos_theta = if abs(g) < 1f-3
        1f0 - 2f0 * u[1]
    else
        k = (1f0 - g^2) / (1f0 - g + 2f0 * g * u[1])
        (1f0 + g^2 - k^2) / (2f0 * g)
    end
    return _direction_from_cosine(d, clamp(cos_theta, -1f0, 1f0), u[2])
end

@inline _hg_pdf(cos_theta::Float32, g::Float32) =
    (1f0 / (4f0 * Float32(π))) * (1f0 - g^2) / max(1f0 + g^2 - 2f0 * g * cos_theta, 1f-12)^1.5f0

@inline _transmittance(sigma_t::SpectralRadiance, t::Float32) = SpectralRadiance(ntuple(k -> exp(-sigma_t.data[k] * t), Val(4)))

@inline function _walk_pdf(sigma_t::SpectralRadiance, t::Float32, hit::Bool)
    T = _transmittance(sigma_t, t)
    return hit ? T : sigma_t * T
end

"""
Cycles' `subsurface_random_walk`: scatter through the object from the entry, with distances by
the extinction of one wavelength chosen by its throughput, half the time guided towards the
surfaces it is between (Dwivedi), until a step reaches a surface.
"""
@noinline function _random_walk(c::SpectralClosure, bsdf::ClosureBSDF, work, accel, rng, pick_scale::Float32)
    fail = (_no_bsdf(bsdf), work)
    u1, rng = pcg32_uniform_f32(rng)
    u2, rng = pcg32_uniform_f32(rng)
    entered, d = _subsurface_entry(c, work.wo, Point2f(u1, u2))
    (entered && dot(d, _facing_geometric(work)) < 0f0) || return fail

    g = c.p1
    throughput = work.beta * c.weight * pick_scale
    alpha = _map4(a -> _walk_albedo(a, g), c.c1)
    sigma_t = _map4(r -> 1f0 / max(r, 1f-16), c.c2)
    throughput = _map4((t, a) -> a != 0f0 ? t / a : 0f0, throughput, c.c1)
    min_alpha = 0.2f0
    throughput = _map4((t, a) -> a < min_alpha ? t * a / min_alpha : t, throughput, alpha)
    alpha = _map4(a -> max(a, min_alpha), alpha)
    sigma_s = sigma_t * alpha
    max_alpha = _maximum(alpha)
    diffusion_length = 1f0 / sqrt(1f0 - max_alpha^(2.44294f0 - 0.0215813f0 * max_alpha + 0.578637f0 / max_alpha))
    diffusion_length == 1f0 && return fail
    phase_log = log((diffusion_length + 1f0) / (diffusion_length - 1f0))
    guided_fraction_org = 1f0 - max(0.5f0, abs(g)^0.125f0)
    sigma_s_star = sigma_s * (1f0 - g)
    sigma_t_star = sigma_t - sigma_s + sigma_s_star

    p0 = Point3f(work.pi)
    n = c.n
    p = p0
    have_opposite = false
    opposite_distance = 0f0
    for bounce in Int32(0):Int32(BSSRDF_MAX_BOUNCES - 1)
        similar = bounce > Int32(SUBSURFACE_SIMILARITY_LEVEL)
        anis = similar ? 0f0 : g
        guided_fraction = similar ? 0.75f0 : guided_fraction_org
        st = similar ? sigma_t_star : sigma_t
        ss = similar ? sigma_s_star : sigma_s

        weights = _map4((t, a) -> abs(t * a), throughput, alpha)
        wsum = sum(weights.data)
        wsum > 0f0 || return fail
        channel_pdf = weights / wsum
        rc, rng = pcg32_uniform_f32(rng)
        target = rc * wsum
        acc = 0f0
        channel = 4
        for ch in 1:4
            acc += weights.data[ch]
            if target < acc
                channel = ch
                break
            end
        end
        sample_sigma_t = st.data[channel]
        randt, rng = pcg32_uniform_f32(rng)

        backward_fraction = 0f0
        forward_factor = 0f0
        forward_stretch = 1f0
        backward_factor = 0f0
        backward_stretch = 1f0
        if bounce > Int32(0)
            ug, rng = pcg32_uniform_f32(rng)
            guided = ug < guided_fraction
            guide_backward = false
            if have_opposite
                x = clamp(dot(Vec3f(p - p0), -n), 0f0, opposite_distance)
                backward_fraction = 1f0 / (1f0 + exp((opposite_distance - 2f0 * x) / diffusion_length))
                ub, rng = pcg32_uniform_f32(rng)
                guide_backward = ub < backward_fraction
            end
            us1, rng = pcg32_uniform_f32(rng)
            us2, rng = pcg32_uniform_f32(rng)
            if guided
                cos_theta = _sample_dwivedi(diffusion_length, phase_log, us1)
                guide_backward && (cos_theta = -cos_theta)
                new_d = _direction_from_cosine(n, cos_theta, us2)
                hg = _hg_pdf(dot(d, new_d), anis)
            else
                new_d = _hg_sample(d, anis, Point2f(us1, us2))
                hg = _hg_pdf(dot(d, new_d), anis)
                cos_theta = dot(new_d, n)
            end
            d = new_d
            forward_factor = (1f0 / (2f0 * Float32(π))) * _phase_dwivedi(diffusion_length, phase_log, cos_theta) / hg
            backward_factor = (1f0 / (2f0 * Float32(π))) * _phase_dwivedi(diffusion_length, phase_log, -cos_theta) / hg
            forward_stretch = 1f0 - cos_theta / diffusion_length
            backward_stretch = 1f0 + cos_theta / diffusion_length
            guided && (sample_sigma_t *= guide_backward ? backward_stretch : forward_stretch)
        end

        t = -log(1f0 - randt) / sample_sigma_t
        t_max = bounce == Int32(0) ? max(t, 10f0 / _minimum(st)) : t
        origin = bounce == Int32(0) ? Point3f(p + d * 1f-4) : p
        found, primitive, t_hit, barycentric, _ = Raycore.closest_hit(accel, Raycore.Ray(o = origin, d = d, t_max = t_max))
        surface = found && t_hit <= t_max
        surface_t = bounce == Int32(0) ? t_hit + 1f-4 : t_hit
        if bounce == Int32(0) && surface
            have_opposite = true
            opposite_distance = dot(Vec3f(origin + t_hit * d - p0), -n)
        end
        hit = surface && surface_t < t
        hit && (t = surface_t)
        p = Point3f(p + t * d)

        pdf = _walk_pdf(st, t, hit)
        if bounce > Int32(0)
            guided_pdf = _walk_pdf(st * forward_stretch, t, hit)
            if have_opposite
                back_pdf = _walk_pdf(st * backward_stretch, t, hit)
                guided_pdf = guided_pdf * (forward_factor * (1f0 - backward_fraction)) + back_pdf * (backward_factor * backward_fraction)
            else
                guided_pdf = guided_pdf * forward_factor
            end
            pdf = pdf * (1f0 - guided_fraction) + guided_pdf * guided_fraction
        end
        transmittance = _transmittance(st, t)
        throughput = throughput * (hit ? transmittance : ss * transmittance) / _dot4(channel_pdf, pdf)
        if hit
            all(isfinite, throughput.data) || return fail
            geom = vp_compute_surface_geometry(primitive, barycentric, Point3f(p + d), -d, 1f0)
            exit_work = _at_exit(work, geom, primitive.metadata.primitive_index, SVector{3, Float32}(barycentric),
                                 throughput)
            return (_exit_bsdf(bsdf, geom.ns), exit_work)
        end
        _maximum(throughput) < VOLUME_THROUGHPUT_EPSILON && return fail
    end
    return fail
end

"""The geometric normal on the side the view is on."""
@inline _facing_geometric(w) = dot(w.wo, w.n) < 0f0 ? -w.n : w.n
