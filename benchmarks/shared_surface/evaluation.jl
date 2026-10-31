module SurfaceLobeEvaluationProbe
using Hikari, Mantle, KernelAbstractions, GeometryBasics, LinearAlgebra, Test
using GeometryBasics: Vec3f, Point2f
const MASK = Hikari.SURFACE_PRINCIPLED_MASK
@inline function unpack(values)
    terms = ntuple(Val(8)) do k
        offset = 27 * (k - 1)
        lobe = Hikari.SurfaceStoredLobe{true, MASK}(UInt32(values[offset + 1]), ntuple(i -> values[offset + 1 + i], Val(15)))
        return Hikari.Weighted(lobe, Hikari.SpectralRadiance(ntuple(i -> values[offset + 16 + i], Val(4))), values[offset + 21], Vec3f(ntuple(i -> values[offset + 21 + i], Val(3))), Vec3f(ntuple(i -> values[offset + 24 + i], Val(3))))
    end
    lambda = Hikari.Wavelengths(ntuple(i -> values[216 + i], Val(4)), ntuple(i -> values[220 + i], Val(4)))
    wo = normalize(Vec3f(values[225], values[226], values[227]))
    wi = normalize(Vec3f(values[228], values[229], values[230]))
    storage = Hikari.MVector{8, typeof(first(terms))}(Hikari.surface_empty(Val(8), Val(true), Val(MASK)))
    at = Int32(1)
    @inbounds for k in 1:8
        if terms[k].sample_weight > 0.0f0
            storage[at] = terms[k]
            at += Int32(1)
        end
    end
    return Hikari.SurfaceLobeSpan(Tuple(storage), at - Int32(1)), lambda, wo, wi
end
@inline function evaluate(lobes, lambda, wo, wi, ::Val{Stage}, rng) where {Stage}
    if Stage == 0
        return Hikari.weighted_f(lobes.data[1], wo, wi, Vec3f(0, 0, 1), lambda)
    elseif Stage == 1
        return Hikari.mixture(lobes.data, wo, wi, Vec3f(0, 0, 1), lambda)
    elseif Stage == 2
        return Hikari.mixture(lobes, wo, wi, Vec3f(0, 0, 1), lambda)
    elseif Stage == 4
        f, pdf = Hikari.mixture(lobes, wo, wi, Vec3f(0, 0, 1), lambda)
        sample = Hikari.pick_sample(lobes, rng * Hikari.share(lobes, Hikari.AnyScattering()), 0.0f0, wo, Point2f(0.1, 0.2), lambda)
        return Hikari.SpectralRadiance((f.data[1], pdf, sample[1][3], sample[2].data[1])), sample[3] + sample[5]
    end
    sampled = Hikari.pick_sample(lobes, rng * Hikari.share(lobes, Hikari.AnyScattering()), 0.0f0, wo, Point2f(0.1, 0.2), lambda)
    return Hikari.mixture(lobes, wo, sampled[1], Vec3f(0, 0, 1), lambda)
end
@inline function face_lobes(lobes, j)
    data = map(lobes.data) do w
        n = j <= 16 && w.lobe.kind in (5, 6) ? -w.n : w.n
        Hikari.Weighted(w.lobe, w.weight, w.sample_weight, n, w.t)
    end
    return Hikari.SurfaceLobeSpan(data, lobes.count)
end
@kernel inbounds = true function evaluation_kernel!(out, values, table, stage::Val{Stage}) where {Stage}
    j = @index(Global)
    if j <= 32
        lobes, lambda, wo, wi = unpack(values)
        lobes = face_lobes(lobes, j)
        wo = Vec3f(wo[1], wo[2], j <= 16 ? -wo[3] : wo[3])
        if Stage == 5
            color = Hikari._uplift(table, Vec3f(0.8, 0.5, 0.2), lambda)
            lobes = Hikari.SurfaceLobeSpan(map(w -> Hikari.Weighted(w.lobe, w.weight * color, w.sample_weight, w.n, w.t), lobes.data), lobes.count)
        end
        f, pdf = evaluate(lobes, lambda, wo, wi, Val(Stage == 5 ? 4 : Stage), Float32(j - 1) / 32.0f0)
        for k in 1:4
            out[j, k] = f.data[k]
        end
        out[j, 5] = pdf
    end
end
function inputs()
    spectrum = Hikari.SpectralRadiance
    coat = Hikari.MicrofacetLobe{:reflection}(Hikari.DielectricFresnel(spectrum(1), spectrum(1)); alpha_x = 0.03f0^2, ior = 1.5f0)
    specular = Hikari.MicrofacetLobe{:reflection}(Hikari.SchlickFresnel(spectrum(0.04f0), spectrum(1), spectrum(1), 0.0f0, 0.0f0); alpha_x = 0.3f0^2, ior = 1.5f0)
    diffuse = Hikari.DiffuseLobe(0.31830987f0, 0.0f0, spectrum(0))
    terms = (
        Hikari.Weighted(coat, spectrum(0.2f0), 0.008f0, Vec3f(0, 0, 1), Vec3f(0)),
        Hikari.Weighted(specular, spectrum(0.8f0), 0.04f0, Vec3f(0, 0, 1), Vec3f(0)),
        Hikari.Weighted(diffuse, spectrum(0.8f0), 0.76f0, Vec3f(0, 0, 1), Vec3f(0)),
    )
    values = zeros(Float32, 231)
    for (k, term) in enumerate(terms)
        w = Hikari.surface_store(term, Val(true), Val(MASK))
        fields = (Float32(w.lobe.kind), w.lobe.data..., w.weight.data..., w.sample_weight, w.n..., w.t...)
        values[(27 * (k - 1) + 1):(27 * k)] .= fields
    end
    values[217:224] .= (450, 510, 570, 630, 0.25, 0.25, 0.25, 0.25)
    values[225:230] .= (0.2, 0.1, 1, -0.3, 0.2, 1)
    values[231] = 3
    return values, terms
end
function reference(values, terms, stage, j)
    _, lambda, wo, wi = unpack(values)
    terms = map(w -> Hikari.Weighted(w.lobe, w.weight, w.sample_weight, j <= 16 && !(w.lobe isa Hikari.DiffuseLobe) ? -w.n : w.n, w.t), terms)
    if stage == 5
        color = Hikari._uplift(Hikari.get_srgb_table(), Vec3f(0.8, 0.5, 0.2), lambda)
        terms = map(w -> Hikari.Weighted(w.lobe, w.weight * color, w.sample_weight, w.n, w.t), terms)
        stage = 4
    end
    wo = Vec3f(wo[1], wo[2], j <= 16 ? -wo[3] : wo[3])
    if stage == 0
        return Hikari.weighted_f(first(terms), wo, wi, Vec3f(0, 0, 1), lambda)
    elseif stage == 4
        f, pdf = Hikari.mixture(terms, wo, wi, Vec3f(0, 0, 1), lambda)
        sample = Hikari.pick_sample(terms, Float32(j - 1) / 32.0f0 * Hikari.share(terms, Hikari.AnyScattering()), 0.0f0, wo, Point2f(0.1, 0.2), lambda)
        return Hikari.SpectralRadiance((f.data[1], pdf, sample[1][3], sample[2].data[1])), sample[3] + sample[5]
    elseif stage == 3
        total = Hikari.share(terms, Hikari.AnyScattering())
        wi = first(Hikari.pick_sample(terms, Float32(j - 1) / 32.0f0 * total, 0.0f0, wo, Point2f(0.1, 0.2), lambda))
    end
    return Hikari.mixture(terms, wo, wi, Vec3f(0, 0, 1), lambda)
end
function run(; stages = 0:3)
    dev = Mantle.Device()
    values, terms = inputs()
    gpu_values = Mantle.devicearray(dev, values)
    output = Mantle.devicearray(dev, Float32, (32, 5))
    fixture = nothing
    table = nothing
    try
        if 5 in stages
            fixture = Main.SurfaceProgramGPUProbe.fixture(Hikari.Principled(); resolution = 16, samples = 1)
            Main.SurfaceProgramGPUProbe.draw(fixture)
            table = fixture.vp.state.rgb2spec_table
        end
        for stage in stages
            println("Packed evaluation stage ", stage); flush(stdout)
            evaluation_kernel!(dev, 32)(output, gpu_values, table, Val(stage); ndrange = 32)
            Mantle.waitidle(dev)
            actual = Array(output)
            expected = zeros(Float32, 32, 5)
            for j in 1:32
                f, pdf = reference(values, terms, stage, j)
                expected[j, :] .= (f.data..., pdf)
            end
            println("GPU front=", actual[32, :], "; CPU=", expected[32, :])
            println("GPU back=", actual[1, :], "; CPU=", expected[1, :])
            @test all(isfinite, actual)
            @test actual ≈ expected rtol = 2.0f-5 atol = 2.0f-6
        end
    finally
        Mantle.unsafe_free!(gpu_values)
        Mantle.unsafe_free!(output)
        fixture === nothing || Main.SurfaceProgramGPUProbe.release(fixture)
    end
    return nothing
end
end
