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

Pick between a shader's BSDF and its subsurface lobe, in proportion to their shares of the
samples, as Cycles' `surface_shader_bsdf_bssrdf_pick` does. Picking the BSDF leaves it without the
subsurface lobe and the path's throughput divided by its chance; picking subsurface walks inside
the object to where the light leaves it, and returns a diffuse BSDF there with the work item moved
to that point, as Cycles' `subsurface_shader_data_setup`. A walk that does not get out returns a
BSDF of no samples, which ends the path. A BSDF without a subsurface lobe is returned as it is.
"""
@inline subsurface_scatter(bsdf, work, accel, sample_idx) = (bsdf, work)

# The BSDF keeps its type through every outcome: picking the BSDF takes the subsurface lobe's share
# away, a walk that gets out turns that lobe into the diffuse lobe it exits by and takes every other
# lobe's share away, and a walk that does not leaves no share at all.
@propagate_inbounds function subsurface_scatter(bsdf::LobeBSDF{L}, work, accel, sample_idx::Int32) where {L}
    return subsurface_scatter(bsdf, Val(subsurface_slot(L)), work, accel, sample_idx)
end

@inline subsurface_scatter(bsdf::LobeBSDF, ::Val{0}, work, accel, sample_idx::Int32) = (bsdf, work)

@propagate_inbounds function subsurface_scatter(bsdf::LobeBSDF, ::Val{K}, work, accel, sample_idx::Int32) where {K}
    sss = bsdf.lobes[K]
    sss.sample_weight > 0f0 || return (bsdf, work)
    total = bsdf.total
    rng = pcg32_init(UInt64(work.pixel_index) << 32 | UInt64(reinterpret(UInt32, sample_idx)), UInt64(work.depth) + UInt64(0x5353))
    u, rng = pcg32_uniform_f32(rng)
    if u >= sss.sample_weight / total
        rest = total - sss.sample_weight
        rest > 0f0 || return (no_samples(typeof(bsdf.lobes), bsdf.lambda), work)
        lobes = with_lobe(bsdf.lobes, Val(K), without_samples, identity)
        return (LobeBSDF(lobes, rest, bsdf.lambda, bsdf.emission, bsdf.dispersive), _with_beta(work, work.beta * (total / rest)))
    end
    # The walk traces rays, so what is live across it is held through every traversal: the walk's
    # parameters and the wavelengths, not the lobes. The other slots come back blank, which no
    # evaluation reads, as they take no samples.
    params, lambda = WalkParams(sss), bsdf.lambda
    exited, exit_work, n = random_walk(params, work, accel, rng, total / sss.sample_weight)
    exited || return (no_samples(typeof(bsdf.lobes), lambda), work)
    lobes = with_lobe(typeof(bsdf.lobes), Val(K), exit_lobe(typeof(sss), params, n))
    return (LobeBSDF(lobes, 1f0, lambda, SpectralRadiance(0f0), false), exit_work)
end

"""The position of the last subsurface lobe in a BSDF's lobes, 0 for none: the one a walk starts from, as Cycles picks the last."""
@generated subsurface_slot(::Type{L}) where {L <: Tuple} =
    something(findlast(T -> T <: Weighted{<:SubsurfaceLobe}, fieldtypes(L)), 0)

"""The lobes with `f` applied to the `K`-th and `g` to every other, unrolled when compiled."""
@generated with_lobe(lobes::L, ::Val{K}, f, g) where {L <: Tuple, K} =
    Expr(:tuple, (k == K ? :(f(lobes[$k])) : :(g(lobes[$k])) for k in 1:fieldcount(L))...)

"""Lobes of the types `L` with `w` the `K`-th and every other blank."""
@generated with_lobe(::Type{L}, ::Val{K}, w) where {L <: Tuple, K} =
    Expr(:tuple, (k == K ? :w : :(blank($(fieldtype(L, k)))) for k in 1:fieldcount(L))...)

"""A value of type `T` with every field zero."""
@generated function blank(::Type{T}) where {T}
    isprimitivetype(T) && return :(zero($T))
    T <: Tuple && return Expr(:tuple, (:(blank($(fieldtype(T, i)))) for i in 1:fieldcount(T))...)
    return Expr(:new, T, (:(blank($(fieldtype(T, i)))) for i in 1:fieldcount(T))...)
end

@inline without_samples(w::Weighted) = Weighted(w.lobe, w.weight, 0f0, w.n, w.t)
"""A BSDF of lobes of the types `L` that takes no samples and reflects nothing."""
@inline no_samples(::Type{L}, lambda::Wavelengths) where {L <: Tuple} =
    LobeBSDF(blank(L), 0f0, lambda, SpectralRadiance(0f0), false)

"""What a random walk needs of the subsurface lobe it starts from."""
struct WalkParams
    n::Vec3f
    alpha::Float32
    ior::Float32
    anisotropy::Float32
    weight::SpectralRadiance
    albedo::SpectralRadiance
    radius::SpectralRadiance
end

WalkParams(w::Weighted{<:SubsurfaceLobe}) =
    WalkParams(w.n, w.lobe.alpha, w.lobe.ior, w.lobe.anisotropy, w.weight, w.lobe.albedo, w.lobe.radius)

"""The subsurface lobe a walk of `p` started from, as the diffuse lobe of weight one around `n` that it leaves the object by."""
@inline function exit_lobe(::Type{Weighted{L, C}}, p::WalkParams, n::Vec3f) where {L <: SubsurfaceLobe, C}
    return Weighted(L(p.albedo, p.radius, p.alpha, p.ior, p.anisotropy, true), SpectralRadiance(1f0), 1f0, n, Vec3f(0))
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
@inline function subsurface_entry(p::WalkParams, wo::Vec3f, u::Point2f)
    n = p.n
    cos_ni = dot(n, wo)
    cos_ni > 0f0 || return (false, Vec3f(0))
    t, b = coordinate_system(n)
    local_h = trowbridge_reitz_sample_wm(world_to_local(wo, n, t, b), u, max(p.alpha, 1f-4), max(p.alpha, 1f-4))
    h = local_to_world(local_h, n, t, b)
    cos_hi = dot(h, wo)
    neta = 1f0 / p.ior
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

Inlined. Outlined as a function of its own, Apple's GPU compiler crashed ("Internal compiler
error") linking the shading kernel of a subsurface material, on hardware and software traversal
alike, once the BSDF sampling after the walk was in the same kernel. Inlined measured the same
as outlined on a 7900 XTX (within 1.5%, both traversals).
"""
@inline function random_walk(c::WalkParams, work, accel, rng, pick_scale::Float32)
    fail = (false, work, Vec3f(0, 0, 1))
    u1, rng = pcg32_uniform_f32(rng)
    u2, rng = pcg32_uniform_f32(rng)
    entered, d = subsurface_entry(c, work.wo, Point2f(u1, u2))
    (entered && dot(d, _facing_geometric(work)) < 0f0) || return fail

    g = c.anisotropy
    throughput = work.beta * c.weight * pick_scale
    alpha = _map4(a -> _walk_albedo(a, g), c.albedo)
    sigma_t = _map4(r -> 1f0 / max(r, 1f-16), c.radius)
    throughput = _map4((t, a) -> a != 0f0 ? t / a : 0f0, throughput, c.albedo)
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
            return (true, exit_work, geom.ns)
        end
        _maximum(throughput) < VOLUME_THROUGHPUT_EPSILON && return fail
    end
    return fail
end

"""The geometric normal on the side the view is on."""
@inline _facing_geometric(w) = dot(w.wo, w.n) < 0f0 ? -w.n : w.n
