module SurfaceProgramCountProbe
using Hikari, Mantle, KernelAbstractions, Raycore, GeometryBasics, LinearAlgebra, Test
using GeometryBasics: Point2f, Point3f, Vec3f
const P = Hikari.SurfaceProgramNodes{8, true, Hikari.SURFACE_PRINCIPLED_MASK, false}
@inline wavelengths() = Hikari.Wavelengths((450.0f0, 510.0f0, 570.0f0, 630.0f0), (0.25f0, 0.25f0, 0.25f0, 0.25f0))
@inline function wavelengths(values, j)
    return Hikari.Wavelengths(
        ntuple(k -> @inbounds(values[j + 32 * (k - 1)]), Val(4)),
        ntuple(k -> @inbounds(values[j + 32 * (k + 3)]), Val(4))
    )
end
sample_values() = vec(Float32[k <= 4 ? 450 + 60 * (k - 1) + 0.1f0 * j : 0.25f0 for j in 1:32, k in 1:8])
@inline record(w) = (Float32(w.lobe.kind), w.lobe.data..., w.weight.data..., w.sample_weight, w.n..., w.t...)
@inline function sample_metrics(bsdf, sc, lambda, j)
    wi = normalize(Vec3f(-0.3, 0.2, 1))
    f, pdf = Hikari.mixture(bsdf.lobes, sc.wo, wi, sc.tfc.n, lambda)
    sample = Hikari.pick_sample(bsdf.lobes, Float32(j - 1) / 32.0f0 * bsdf.total, 0.0f0, sc.wo, Point2f(0.1, 0.2), lambda)
    return (f.data[1], pdf, sample[1][3], sample[2].data[1], sample[3], sample[5])
end
function context(j)
    wo = normalize(Vec3f(0.2, 0.1, (Float32(j) - 16.5f0) / 16.0f0))
    tfc = Hikari.TextureFilterContext(Point2f(0.3, 0.6), Point3f(0.3, -0.2, 0.7), Vec3f(0, 0, 1))
    return Hikari.ShaderContext(tfc, wo, Vec3f(1.0f-3, 0, 0), Vec3f(0, 1.0f-3, 0), Vec3f(1, 0, 0), Vec3f(0, 0, 1), 1.0f0, Vec3f(0), Point2f(0))
end
function reference(material)
    store = Raycore.MultiTypeSet(KernelAbstractions.CPU())
    mat = Hikari.to_device_material(store, Hikari.ShaderMaterial(Hikari.shader_node(material)))
    textures = Raycore.StaticMultiTypeSet((), Tuple(T[a for a in store.texture_gpu_arrays if typeof(a) === T] for T in store.texture_order))
    prog = Hikari.closure_programs(textures)[mat.closures]
    expected = zeros(Float32, 32, 218)
    evaluated = zeros(Float32, 32, 6)
    source = Hikari.shader_programs(textures, mat.program)[mat.program.idx]
    M = typeof(mat).parameters[3]
    values = sample_values()
    for j in 1:32
        lambda = wavelengths(values, j)
        bsdf = Hikari.surface_program_bsdf(P, Hikari.GatheredStack(values, Int32(1), Int32(1)), prog, context(j), Hikari.LobeSetup(Hikari.lookup_tables(textures), false), Hikari.get_srgb_table(), lambda)
        expected[j, 1] = bsdf.lobes.count
        expected[j, 2] = bsdf.total
        for k in 1:8
            fields = record(bsdf.lobes.data[k])
            for c in 1:27
                expected[j, 2 + 27 * (k - 1) + c] = fields[c]
            end
        end
        rgb = Hikari.shader_lobes(M, source, textures, context(j), false)
        typed = Hikari.lobe_bsdf(M, rgb, Hikari.get_srgb_table(), lambda)
        evaluated[j, :] .= sample_metrics(typed, context(j), lambda, j)
    end
    return expected, evaluated
end
@kernel inbounds = true function count_kernel!(output, textures, values, table, program::Int32)
    j = @index(Global)
    if j <= 32
        lambda = wavelengths(values, j)
        prog = Hikari.closure_programs(textures)[program]
        stack = Hikari.GatheredStack(values, Int32(1), Int32(1))
        bsdf = Hikari.surface_program_bsdf(P, stack, prog, context(j), Hikari.LobeSetup(Hikari.lookup_tables(textures), false), table, lambda)
        output[j, 1] = Float32(bsdf.lobes.count)
        output[j, 2] = bsdf.total
        for k in 1:8
            fields = record(bsdf.lobes.data[k])
            for c in 1:27
                output[j, 2 + 27 * (k - 1) + c] = fields[c]
            end
        end
    end
end
@kernel inbounds = true function evaluate_kernel!(output, textures, values, table, program::Int32, ::Val{Stage}) where {Stage}
    j = @index(Global)
    if j <= 32
        sc = context(j); lambda = wavelengths(values, j)
        prog = Hikari.closure_programs(textures)[program]
        stack = Hikari.GatheredStack(values, Int32(1), Int32(1))
        bsdf = Hikari.surface_program_bsdf(P, stack, prog, sc, Hikari.LobeSetup(Hikari.lookup_tables(textures), false), table, lambda)
        for c in 1:6
            output[j, c] = 0.0f0
        end
        if Stage != 1
            f, pdf = Hikari.mixture(bsdf.lobes, sc.wo, normalize(Vec3f(-0.3, 0.2, 1)), sc.tfc.n, lambda)
            output[j, 1] = f.data[1]; output[j, 2] = pdf
        end
        if Stage != 0
            sample = Hikari.pick_sample(bsdf.lobes, Float32(j - 1) / 32.0f0 * bsdf.total, 0.0f0, sc.wo, Point2f(0.1, 0.2), lambda)
            output[j, 3] = sample[1][3]; output[j, 4] = sample[2].data[1]
            output[j, 5] = sample[3]; output[j, 6] = sample[5]
        end
    end
end
function run(material = Hikari.Principled(roughness = 0.3, coat_weight = 0.2); evaluate = false, stages = 0:2)
    R = Main.SurfaceProgramGPUProbe
    f = R.fixture(material; resolution = 16, samples = 1)
    output = nothing; values = nothing
    try
        R.draw(f)
        store = f.scene.materials
        mat = Raycore.@allowscalar first(first(store.static.data))
        output = Mantle.devicearray(f.dev, Float32, (32, 218))
        values = Mantle.devicearray(f.dev, sample_values())
        count_kernel!(f.dev, 32)(output, Hikari.closure_set(store.static), values, f.vp.state.rgb2spec_table, mat.closures; ndrange = 32)
        Mantle.waitidle(f.dev)
        actual = Array(output)
        println("counts=", actual[:, 1]); println("first slot=", actual[1, 1:29])
        @test all(0 .<= actual[:, 1] .<= 8)
        @test all(isfinite, actual)
        @test all(>(0), actual[:, 2])
        expected, expected_eval = reference(material)
        @test actual[:, 1] == expected[:, 1]
        @test actual[:, 3:27:218] == expected[:, 3:27:218]
        @test actual ≈ expected rtol = 2.0f-5 atol = 2.0f-6
        if evaluate
            for stage in stages
                println("Program evaluation stage ", stage); flush(stdout)
                evaluate_kernel!(f.dev, 32)(output, Hikari.closure_set(store.static), values, f.vp.state.rgb2spec_table, mat.closures, Val(stage); ndrange = 32)
                Mantle.waitidle(f.dev)
                evaluated = Array(output)[:, 1:6]
                println("evaluated=", evaluated)
                @test all(isfinite, evaluated)
                columns = stage == 0 ? (1:2) : stage == 1 ? (3:6) : (1:6)
                @test evaluated[:, columns] ≈ expected_eval[:, columns] rtol = 2.0f-5 atol = 2.0f-6
            end
        end
        return actual
    finally
        output === nothing || Mantle.unsafe_free!(output)
        values === nothing || Mantle.unsafe_free!(values)
        R.release(f)
    end
end
end
