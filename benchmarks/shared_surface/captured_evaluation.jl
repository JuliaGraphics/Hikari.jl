# Reproduce evaluation with the exact per-lane records from real program setup,
# removing program interpretation, RGB lookup and storage construction on GPU.
module CapturedSurfaceEvaluation
using Hikari, Mantle, KernelAbstractions, GeometryBasics, LinearAlgebra, Test
using GeometryBasics: Vec3f
using ..SurfaceProgramCountProbe: sample_values, wavelengths, reference
@kernel inbounds = true function evaluate!(out, records, values, ::Val{K}) where {K}
    j = @index(Global)
    if j <= 32
        base = 2 + (K - 1) * 27
        lobe = Hikari.SurfaceStoredLobe{true, Hikari.SURFACE_PRINCIPLED_MASK}(UInt32(records[j, base + 1]), ntuple(k -> records[j, base + 1 + k], Val(15)))
        w = Hikari.Weighted(lobe, Hikari.SpectralRadiance(ntuple(k -> records[j, base + 16 + k], Val(4))), records[j, base + 21], Vec3f(ntuple(k -> records[j, base + 21 + k], Val(3))), Vec3f(ntuple(k -> records[j, base + 24 + k], Val(3))))
        wo = normalize(Vec3f(0.2, 0.1, (Float32(j) - 16.5f0) / 16.0f0))
        f, pdf = Hikari.weighted_f(w, wo, normalize(Vec3f(-0.3, 0.2, 1)), Vec3f(0, 0, 1), wavelengths(values, j))
        for k in 1:4
            out[j, k] = f.data[k]
        end
        out[j, 5] = pdf
    end
end
function run(; terms = (1, 2, 3))
    material = Hikari.Principled(roughness = 0.3, coat_weight = 0.2)
    records, _ = reference(material)
    values = sample_values()
    dev = Mantle.Device()
    gpu_records = Mantle.devicearray(dev, records)
    gpu_values = Mantle.devicearray(dev, values)
    output = Mantle.devicearray(dev, Float32, (32, 5))
    try
        for k in terms
            println("Captured term ", k); flush(stdout)
            evaluate!(dev, 32)(output, gpu_records, gpu_values, Val(k); ndrange = 32)
            Mantle.waitidle(dev)
            actual = Array(output)
            expected = Main.SurfaceSingleTermProbe.expected(material, k)
            @test all(isfinite, actual)
            @test actual ≈ expected rtol = 2.0f-5 atol = 2.0f-6
        end
    finally
        Mantle.unsafe_free!(output)
        Mantle.unsafe_free!(gpu_values)
        Mantle.unsafe_free!(gpu_records)
    end
    return nothing
end
end
