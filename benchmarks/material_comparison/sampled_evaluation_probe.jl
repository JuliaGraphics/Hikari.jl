module SampledEvaluationProbe
using Hikari, Mantle, JSON3, Statistics, Test
using LinearAlgebra: dot, normalize
using ..MaterialComparison
const H = Hikari

# Same sampling decisions as lobe_sample, retaining its Fresnel result for the
# selected term. Reconstruct the evaluation normal for now: below roughness .03,
# its z rounding changes the existing GGX D by factors of three. This isolates
# Fresnel reuse from that numerical difference rather than relaxing the test.
@inline function microfacet_sample(l::H.MicrofacetLobe{Mode}, wo, u, uc, lambda, s) where {Mode}
    wo[3] > 0f0 || return H.NO_LOBE_SAMPLE
    smooth = H.singular(l)
    h = smooth ? H.Vec3f(0, 0, 1) : H.microfacet_sample_wm(l, wo, u)
    cos_hi = dot(h, wo)
    r, t = H.fresnel_rt(l.fresnel, cos_hi, l.ior, lambda)
    H.average(r) + H.average(t) > 0f0 || return H.NO_LOBE_SAMPLE
    pr, pt = H.microfacet_choice(l, r, t, s)
    pr + pt > 0f0 || return H.NO_LOBE_SAMPLE
    pdf_reflect = pr / (pr + pt)
    refract = uc >= pdf_reflect
    if refract
        sin2_t = (1f0 - cos_hi^2) / l.ior^2
        sin2_t >= 1f0 && return H.NO_LOBE_SAMPLE
        cos_t = sqrt(1f0 - sin2_t)
        wi = -wo / l.ior + (cos_hi / l.ior - cos_t) * h
    else
        wi = 2f0 * cos_hi * h - wo
    end
    (wi[3] < 0f0) == refract || return H.NO_LOBE_SAMPLE
    smooth = smooth || (refract && abs(l.ior - 1f0) < 1f-4)
    eta = refract ? l.ior : 1f0
    out = Mode === :thin ? H.Vec3f(wi[1], wi[2], -wi[3]) : wi
    coeff = refract ? t * H.refraction_scale(l) : r
    choice_pdf = refract ? 1f0 - pdf_reflect : pdf_reflect
    if smooth
        flags = (refract || Mode === :thin) ? H.BXDF_SPECULAR_TRANSMISSION : H.BXDF_SPECULAR_REFLECTION
        return (out, coeff * (l.energy / max(abs(wi[3]), 1f-6)), choice_pdf, flags, eta)
    end
    eval_h = refract ? -(l.ior * wi + wo) : wo + wi
    len_h = H.norm(eval_h)
    len_h > 0f0 || return H.NO_LOBE_SAMPLE
    eval_h /= len_h
    eval_h[3] < 0f0 && (eval_h = -eval_h)
    eval_cos_hi = dot(eval_h, wo)
    common = H.microfacet_d(l, eval_h) / wo[3]
    if refract
        common *= (l.ior / len_h)^2 * abs(eval_cos_hi * dot(eval_h, wi))
    else
        common *= 0.25f0
    end
    lambda_o = H.microfacet_lambda(l, wo)
    lambda_i = H.microfacet_lambda(l, wi)
    value = coeff * (common / (1f0 + lambda_o + lambda_i) * l.energy / abs(wi[3]))
    pdf = common * choice_pdf / (1f0 + lambda_o)
    flags = (refract || Mode === :thin) ? H.BXDF_GLOSSY_TRANSMISSION : H.BXDF_GLOSSY_REFLECTION
    return (out, value, pdf, flags, eta)
end

@inline sampled_lobe(l::H.MicrofacetLobe, wo, u, uc, lambda, s) =
    microfacet_sample(l, wo, u, uc, lambda, s)
@inline sampled_lobe(l::H.BSDFLobe, wo, u, uc, lambda, s) = H.lobe_sample(l, wo, u, uc, lambda, s)

@inline function chosen_sample(lobes, count, target, wo, u, ns, lambda)
    before = 0f0
    @inbounds for j in 1:count
        w = lobes[j]
        sw = w.sample_weight
        after = before + sw
        if sw > 0f0 && (target < after || j == count)
            uc = clamp((target - before) / max(sw, 1f-12), 0f0, 0.99999994f0)
            n, t, b = H.lobe_frame(w.lobe, w.n, w.t, wo)
            local_wi, value, pdf, flags, eta = H.surface_apply(sampled_lobe, w.lobe,
                H.world_to_local(wo, n, t, b), u, uc, lambda, H.AnyScattering())
            wi = normalize(H.local_to_world(local_wi, n, t, b))
            value *= w.weight
            if !H.is_specular(flags)
                if w.lobe.kind in (3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20)
                    shadowing = H.bump_shadowing(H.is_diffuse_lobe(w.lobe), n, ns, wi)
                    cos_s = abs(dot(wi, ns))
                    if shadowing > 0f0 && cos_s > 1f-7
                        value *= shadowing * abs(dot(wi, n)) / cos_s
                    else
                        value, pdf = H.SpectralRadiance(0f0), 0f0
                    end
                else
                    value, pdf = H.weighted_f(w, wo, wi, ns, lambda)
                end
            end
            return wi, value, pdf, flags, eta, sw, n, H.terminates_secondary(w.lobe, flags), j
        end
        before = after
    end
    return (H.pick_sample((), target, before, wo, u, lambda)..., 0)
end

@inline function sample(bsdf, wo, ns, u, rng)
    total = bsdf.total
    total > 0f0 || return H.SpectralBSDFSample()
    lobes = bsdf.lobes
    wi, selected_f, selected_pdf, flags, eta, sw, cn, terminates, selected =
        chosen_sample(lobes.data, lobes.count, rng * total, wo, u, ns, bsdf.lambda)
    flags == 0x00 && return H.SpectralBSDFSample()
    side = dot(wo, ns) * dot(wi, ns) < 0f0 ? H.BXDF_TRANSMISSION : H.BXDF_REFLECTION
    flags = (flags & ~(H.BXDF_TRANSMISSION | H.BXDF_REFLECTION)) | side
    if H.is_specular(flags)
        cos_s = abs(dot(wi, ns))
        cos_s > 1f-7 || return H.SpectralBSDFSample()
        return H.SpectralBSDFSample(selected_f * (abs(dot(wi, cn)) / cos_s), wi,
            selected_pdf * sw / total, flags, eta, false, bsdf.dispersive && terminates)
    end
    value, pdf = H.SpectralRadiance(0f0), 0f0
    @inbounds for j in lobes.count:-1:1
        w = lobes.data[j]
        if w.sample_weight > 0f0
            f, p = j == selected ? (selected_f, selected_pdf) : H.weighted_f(w, wo, wi, ns, bsdf.lambda)
            value += f
            pdf += p * w.sample_weight
        end
    end
    pdf > 0f0 || return H.SpectralBSDFSample()
    return H.SpectralBSDFSample(value, wi, pdf / total, flags, eta, false, bsdf.dispersive && terminates)
end

function install!()
    before = Set(methods(H.sample_bsdf_spectral))
    @eval H @inline function sample_bsdf_spectral(
            bsdf::LobeBSDF{L}, ::RGBToSpectrumTable, textures,
            wo::Vec3f, n::Vec3f, dpdus::Vec3f, ::TextureFilterContext, ::Wavelengths,
            u::Point2f, rng::Float32, ::Bool = false) where {L <: SurfaceLobeSpan}
        $sample(bsdf, wo, n, u, rng)
    end
    return [m for m in methods(H.sample_bsdf_spectral) if !(m in before)]
end

function check_microfacets()
    lambda = H.Wavelengths((450f0, 510f0, 570f0, 630f0), (0.25f0, 0.25f0, 0.25f0, 0.25f0))
    @testset "sampled microfacet value and PDF match independent evaluation" begin
        for mode in (:reflection, :glass, :refraction), convention in (H.CyclesGGX(), H.CyclesBeckmann()),
                roughness in (0.02f0, 0.3f0, 0.8f0), ior in (0.6666667f0, 1.5f0),
                angle in (0f0, 0.5f0, 0.95f0), x in (0.1f0, 0.45f0, 0.8f0), uc in (0.001f0, 0.7f0)
            fresnel = H.DielectricFresnel(H.SpectralRadiance(1f0), H.SpectralRadiance(1f0))
            l = H.MicrofacetLobe{mode}(fresnel; alpha_x = roughness^2, alpha_y = roughness^2,
                ior, convention)
            wo = normalize(H.Vec3f(angle, 0, sqrt(1f0 - angle^2)))
            u = H.Point2f(x, 0.37f0)
            original = H.lobe_sample(l, wo, u, uc, lambda)
            actual = microfacet_sample(l, wo, u, uc, lambda, H.AnyScattering())
            @test actual[1] == original[1]
            @test actual[4:5] == original[4:5]
            actual[4] == 0x00 && continue
            expected = H.lobe_f(l, wo, actual[1], lambda, H.AnyScattering())
            @test all(isapprox.(actual[2].data, expected[1].data; rtol = 5f-4, atol = 2f-6))
            @test actual[3] ≈ expected[2] rtol = 5f-4 atol = 2f-6
        end
    end
    return nothing
end

function resources(fixture, output, label)
    result = Any[]
    seen = Set{UInt}()
    for plan in H.allplans(fixture.vp.state.plans), pass in plan.passes
        pass.pass.name == "shade" || continue
        for dispatch in pass.dispatches
            launch = dispatch.launch
            id = objectid(launch.pipeline)
            id in seen && continue
            push!(seen, id)
            prefix = output * "." * label * "_" * string(length(seen))
            write(prefix * ".spv", launch.compiled.spirv_bytes)
            ir = Mantle.pipeline_exec_ir(fixture.dev, launch.pipeline)
            ir === nothing && error("Enable pipeline executable properties before creating the device")
            for (i, entry) in enumerate(ir)
                write(prefix * "_" * string(i) * ".txt", entry.text)
            end
            push!(result, (; spirv_bytes = length(launch.compiled.spirv_bytes),
                native = Mantle.pipeline_exec_stats(fixture.dev, launch.pipeline),
                representations = [entry.name for entry in ir]))
        end
    end
    return result
end

function run(output; spec, repeats = 11, profile = false, names = ("mix_matte_metal", "mix_layered_glass"))
    rows = Any[]
    for settings in filter(row -> row.name in names, spec.cases)
        fixtures = Any[]
        added = Method[]
        try
            baseline = MaterialComparison.fixture(MaterialComparison.principled_material(settings, spec), spec)
            push!(fixtures, baseline)
            first_a = @timed Base.invokelatest(MaterialComparison.draw, baseline)
            expected = MaterialComparison.pixels(baseline)
            append!(added, install!())
            length(added) == 2 || error("Expected two temporary sampler methods, got $(length(added))")
            candidate = Base.invokelatest(MaterialComparison.fixture, MaterialComparison.principled_material(settings, spec), spec)
            push!(fixtures, candidate)
            first_b = @timed Base.invokelatest(MaterialComparison.draw, candidate)
            actual = MaterialComparison.pixels(candidate)
            @test all(isfinite, actual)
            @test actual ≈ expected rtol = 5f-5 atol = 2f-5
            for _ in 1:3, fixture in fixtures
                Base.invokelatest(MaterialComparison.draw, fixture)
            end
            times = (Float64[], Float64[])
            for i in 1:repeats, j in (isodd(i) ? (1, 2) : (2, 1))
                push!(times[j], @elapsed Base.invokelatest(MaterialComparison.draw, fixtures[j]))
            end
            @test MaterialComparison.pixels(baseline) == expected
            @test MaterialComparison.pixels(candidate) == actual
            record = (; material = settings.name, baseline_seconds = median(times[1]),
                candidate_seconds = median(times[2]), first_seconds = (first_a.time, first_b.time),
                julia_compile_seconds = (first_a.compile_time, first_b.compile_time),
                max_pixel_error = maximum(abs.(actual .- expected)), frame_seconds = times,
                resources = profile ? [resources(f, output, settings.name * "_" * string(j))
                    for (j, f) in enumerate(fixtures)] : nothing)
            push!(rows, record)
            mkpath(dirname(output))
            open(io -> JSON3.write(io, (; device = Mantle.devicename(baseline.dev), rows)), output, "w")
            println((; record.material, record.baseline_seconds, record.candidate_seconds, record.max_pixel_error))
            flush(stdout)
        finally
            foreach(Base.delete_method, added)
            foreach(MaterialComparison.release, fixtures)
        end
    end
    return nothing
end
end
