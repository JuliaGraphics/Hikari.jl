module SurfaceProgramKernelMWE
using Hikari, Mantle, KernelAbstractions, Raycore, GeometryBasics, LinearAlgebra, Test, JSON3
using GeometryBasics: Point2f, Point3f, Vec3f
function reference()
    store = Raycore.MultiTypeSet(KernelAbstractions.CPU())
    mat = Hikari.to_device_material(store, Hikari.surface_material(Hikari.Principled()))
    textures = Raycore.StaticMultiTypeSet((), Tuple(T[a for a in store.texture_gpu_arrays if typeof(a) === T] for T in store.texture_order))
    prog = Hikari.shader_programs(textures, mat.program)[mat.program.idx]
    lambda = Hikari.Wavelengths((450.0f0, 510.0f0, 570.0f0, 630.0f0), (0.25f0, 0.25f0, 0.25f0, 0.25f0))
    wo = normalize(Vec3f(0.2, 0.1, 1))
    tfc = Hikari.TextureFilterContext(Point2f(0.3, 0.6), Point3f(0.3, -0.2, 0.7), Vec3f(0, 0, 1))
    sc = Hikari.ShaderContext(tfc, wo, Vec3f(1.0f-3, 0, 0), Vec3f(0, 1.0f-3, 0), Vec3f(1, 0, 0), Vec3f(0, 0, 1), 1.0f0, Vec3f(0), Point2f(0))
    M = typeof(mat).parameters[3]
    produced = Hikari.shader_lobes(M, prog, textures, sc, false)
    bsdf = Hikari.lobe_bsdf(M, produced, Hikari.get_srgb_table(), lambda)
    f, pdf = Hikari.mixture(bsdf.lobes, wo, normalize(Vec3f(-0.3, 0.2, 1)), tfc.n, lambda)
    return Float32[f.data[1], pdf, bsdf.total]
end
const P = Hikari.SurfaceProgramNodes{8, false, Hikari.SURFACE_PRINCIPLED_MASK, false}
@kernel inbounds = true function program_kernel!(out, textures, values, table, program::Int32)
    j = @index(Global)
    if j == 1
        lambda = Hikari.Wavelengths((450.0f0, 510.0f0, 570.0f0, 630.0f0), (0.25f0, 0.25f0, 0.25f0, 0.25f0))
        wo = normalize(Vec3f(0.2, 0.1, 1))
        tfc = Hikari.TextureFilterContext(Point2f(0.3, 0.6), Point3f(0.3, -0.2, 0.7), Vec3f(0, 0, 1))
        sc = Hikari.ShaderContext(tfc, wo, Vec3f(1.0f-3, 0, 0), Vec3f(0, 1.0f-3, 0), Vec3f(1, 0, 0), Vec3f(0, 0, 1), 1.0f0, Vec3f(0), Point2f(0))
        prog = Hikari.closure_programs(textures)[program]
        stack = Hikari.GatheredStack(values, Int32(1), Int32(1))
        bsdf = Hikari.surface_program_bsdf(P, stack, prog, sc, Hikari.LobeSetup(Hikari.lookup_tables(textures), false), table, lambda)
        f, pdf = Hikari.mixture(bsdf.lobes, wo, normalize(Vec3f(-0.3, 0.2, 1)), tfc.n, lambda)
        out[1] = f.data[1];out[2] = pdf;out[3] = bsdf.total
    end
end
function run()
    R = Main.SurfaceProgramGPUProbe
    fixture = R.fixture(Hikari.Principled(); resolution = 16, samples = 1)
    output = nothing;values = nothing
    try
        println("Preparing reference renderer");flush(stdout)
        R.draw(fixture)
        store = fixture.scene.materials
        material = Raycore.@allowscalar first(first(store.static.data))
        textures = Hikari.closure_set(store.static)
        output = Mantle.devicearray(fixture.dev, Float32, 3)
        values = Mantle.devicearray(fixture.dev, zeros(Float32, 1))
        println("Launching standalone shared-program kernel");flush(stdout)
        elapsed = @elapsed begin
            program_kernel!(fixture.dev, 32)(output, textures, values, fixture.vp.state.rgb2spec_table, material.closures; ndrange = 32)
            Mantle.waitidle(fixture.dev)
        end
        actual = Array(output)
        @test all(isfinite, actual)
        @test all(>(0), actual)
        expected = reference()
        @test actual ≈ expected rtol = 2.0f-5 atol = 2.0f-6
        println(JSON3.write((; seconds = elapsed, actual, expected)))
        return actual
    finally
        output === nothing||Mantle.unsafe_free!(output)
        values === nothing||Mantle.unsafe_free!(values)
        R.release(fixture)
    end
end
end
