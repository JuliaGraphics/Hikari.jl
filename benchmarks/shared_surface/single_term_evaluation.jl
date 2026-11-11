# Isolate a single stored term after the real runtime program setup. This keeps
# wavelength inputs dynamic while removing mixture traversal from the kernel.
module SurfaceSingleTermProbe
using Hikari, Mantle, KernelAbstractions, Raycore, GeometryBasics, LinearAlgebra, Test
using GeometryBasics: Vec3f
using ..SurfaceProgramCountProbe: P, context, wavelengths, sample_values
@kernel inbounds = true function term_kernel!(out, textures, values, table, program::Int32, ::Val{K}, ::Val{Mode}) where {K, Mode}
    j = @index(Global)
    if j <= 32
        sc = context(j)
        lambda = wavelengths(values, j)
        prog = Hikari.closure_programs(textures)[program]
        stack = Hikari.GatheredStack(values, Int32(1), Int32(1))
        bsdf = Hikari.surface_program_bsdf(P, stack, prog, sc, Hikari.LobeSetup(Hikari.lookup_tables(textures), false), table, lambda)
        w = bsdf.lobes.data[K]
        if Mode == 1
            w = Hikari.Weighted(Hikari.surface_restore(w.lobe, Val(5)), w.weight, w.sample_weight, w.n, w.t)
        elseif Mode == 2
            lambda = wavelengths()
        elseif Mode == 3
            w = Hikari.Weighted(Hikari.SurfaceStoredLobe{true, Hikari.SURFACE_PRINCIPLED_MASK}(UInt32(5), w.lobe.data), w.weight, w.sample_weight, w.n, w.t)
        end
        f, pdf = Hikari.weighted_f(w, sc.wo, normalize(Vec3f(-0.3, 0.2, 1)), sc.tfc.n, lambda)
        for k in 1:4
            out[j, k] = f.data[k]
        end
        out[j, 5] = pdf
    end
end
function expected(material, k)
    store = Raycore.MultiTypeSet(KernelAbstractions.CPU())
    mat = Hikari.to_device_material(store, Hikari.ShaderMaterial(Hikari.shader_node(material)))
    textures = Raycore.StaticMultiTypeSet((), Tuple(T[a for a in store.texture_gpu_arrays if typeof(a) === T] for T in store.texture_order))
    prog = Hikari.closure_programs(textures)[mat.closures]
    values = sample_values()
    result = zeros(Float32, 32, 5)
    for j in 1:32
        sc = context(j); lambda = wavelengths(values, j)
        bsdf = Hikari.surface_program_bsdf(P, Hikari.GatheredStack(values, Int32(1), Int32(1)), prog, sc, Hikari.LobeSetup(Hikari.lookup_tables(textures), false), Hikari.get_srgb_table(), lambda)
        f, pdf = Hikari.weighted_f(bsdf.lobes.data[k], sc.wo, normalize(Vec3f(-0.3, 0.2, 1)), sc.tfc.n, lambda)
        result[j, :] .= (f.data..., pdf)
    end
    return result
end
function run(; terms = (3, 1, 2), mode = 0)
    material = Hikari.Principled(roughness = 0.3, coat_weight = 0.2)
    r = Main.SurfaceProgramGPUProbe
    fixture = r.fixture(material; resolution = 16, samples = 1)
    output = values = nothing
    try
        r.draw(fixture)
        store = fixture.scene.materials
        mat = Raycore.@allowscalar first(first(store.static.data))
        output = Mantle.devicearray(fixture.dev, Float32, (32, 5))
        values = Mantle.devicearray(fixture.dev, sample_values())
        for k in terms
            println("Evaluate single term ", k, " mode ", mode); flush(stdout)
            term_kernel!(fixture.dev, 32)(output, Hikari.closure_set(store.static), values, fixture.vp.state.rgb2spec_table, mat.closures, Val(k), Val(mode); ndrange = 32)
            Mantle.waitidle(fixture.dev)
            actual = Array(output)
            @test all(isfinite, actual)
            @test actual ≈ expected(material, k) rtol = 2.0f-5 atol = 2.0f-6
        end
    finally
        output === nothing || Mantle.unsafe_free!(output)
        values === nothing || Mantle.unsafe_free!(values)
        r.release(fixture)
    end
    return nothing
end
end
