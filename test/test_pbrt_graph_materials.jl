module PbrtGraphTests
using Hikari, Raycore, Test, Mantle, GeometryBasics
using GeometryBasics: Point2f, Point3f, Vec3f
using KernelAbstractions: CPU
using LinearAlgebra: normalize
function run_exact()
    table = Hikari.get_srgb_table()
    wavelengths = Hikari.sample_wavelengths_stratified((0.11f0, 0.27f0, 0.53f0, 0.79f0))
    n, tangent = Vec3f(0, 0, 1), Vec3f(1, 0, 0)
    tfc = Hikari.TextureFilterContext(Point2f(0.5f0), Point3f(0), n)
    models = [
        Hikari.Diffuse(color = (0.2, 0.4, 0.7), roughness = 0.3),
        Hikari.Dielectric(ior = 1.5, roughness = (0.1, 0.3)),
        Hikari.Dielectric(ior = Hikari.AU_ETA_SPECTRUM, roughness = 0.1),
        Hikari.ThinDielectric(ior = 1.4),
        Hikari.Conductor(ior = (0.2, 0.3, 0.4), extinction = (3.1, 3.3, 3.7), roughness = (0.1, 0.2)),
        Hikari.Conductor(reflectance = (0.2, 0.4, 0.7), roughness = 0.1),
        Hikari.Gold(roughness = 0.1),
        Hikari.DiffuseTransmission(reflectance = (0.2, 0.3, 0.4), transmittance = (0.4, 0.2, 0.1)),
        Hikari.DiffuseTransmission(scale = Hikari.TextureNodes.xyz().x * 0.1f0 + 0.5f0),
        Hikari.DiffuseTransmission(scale = fill(0.5f0, 4, 4)),
        Hikari.CoatedDiffuse(reflectance = (0.2, 0.4, 0.7), roughness = 0.1, albedo = (0.1, 0.2, 0.1), g = 0.2),
        Hikari.CoatedConductor(interface_roughness = 0.1, conductor_roughness = 0.2),
        Hikari.CoatedConductor(interface_roughness = 0.1, reflectance = (0.3, 0.5, 0.7)),
        Hikari.CoatedDiffuseTransmission(roughness = 0.2, albedo = (0.1, 0.2, 0.1)),
    ]
    return @testset "native graph program preserves pbrt BSDFs" begin
        for m in models
            store = Raycore.MultiTypeSet(CPU())
            original = Hikari.to_device_material(store, m)
            graph = Hikari.to_device_material(store, Hikari.surface_material(m))
            textures = Raycore.StaticMultiTypeSet((), Tuple(T[a for a in store.texture_gpu_arrays if typeof(a) === T] for T in store.texture_order))
            M = typeof(graph).parameters[3]
            prog = Hikari.shader_programs(textures, graph.program)[graph.program.idx]
            for side in (-1.0f0, 1.0f0), regularize in (false, true)
                wo = normalize(Vec3f(0.3f0, 0.2f0, side))
                sc = Hikari.ShaderContext(tfc, wo, Vec3f(0), Vec3f(0), tangent, n, 1.0f0, Vec3f(0), Point2f(0))
                reference = Hikari.get_bxdf(original, table, textures, tfc, wavelengths, regularize, wo, n, tangent)
                bsdf = Hikari.shader_bsdf(M, prog, textures, sc, regularize, table, wavelengths)
                if m isa Hikari.Diffuse
                    wi = normalize(Vec3f(-0.1f0, 0.4f0, side))
                    f, pdf = Hikari.evaluate_bsdf_spectral(bsdf, table, textures, wo, wi, n, tangent, tfc, wavelengths)
                    old_f, old_pdf = Hikari.evaluate_bsdf_spectral(reference, table, textures, wo, wi, n, tangent, tfc, wavelengths)
                    @test all(isapprox.(f.data, old_f.data))
                    @test pdf ≈ old_pdf
                else
                    @test bsdf === reference
                end
                @test Hikari.terminates_at_vertex(bsdf) == Hikari.terminates_at_vertex(reference)
            end
        end
    end
end
function stored(m, store = Raycore.MultiTypeSet(CPU()))
    original = Hikari.to_device_material(store, m)
    graph = Hikari.to_device_material(store, Hikari.surface_material(m))
    textures = Raycore.StaticMultiTypeSet((), Tuple(T[a for a in store.texture_gpu_arrays if typeof(a) === T] for T in store.texture_order))
    return store, original, graph, textures
end
function evaluate_graph(mat, textures, sc, lambda; regularize = false)
    M = typeof(mat).parameters[3]
    prog = Hikari.shader_programs(textures, mat.program)[mat.program.idx]
    return Hikari.shader_bsdf(M, prog, textures, sc, regularize, Hikari.get_srgb_table(), lambda)
end
function run_bump()
    table = Hikari.get_srgb_table()
    lambda = Hikari.sample_wavelengths_stratified((0.1f0, 0.3f0, 0.5f0, 0.7f0))
    n = normalize(Vec3f(0.1, 0.2, 1))
    tangent = normalize(Vec3f(1, 0, -0.1))
    tfc = Hikari.TextureFilterContext(Point2f(0.2, 0.4), 0.002f0, 0.001f0, 0.001f0, 0.003f0, UInt32(1), Hikari.SVector{3, Float32}(0.2, 0.3, 0.5), Point3f(0.2, 0.3, 0.5), n)
    geom = Hikari.PbrtBumpGeometry(Vec3f(2, 0, 0.1), Vec3f(0.01, 0.02, 0.03), Vec3f(0.03, 0.02, 0.01))
    wo = normalize(Vec3f(0.1, 0.2, 1))
    sc = Hikari.ShaderContext(tfc, wo, Vec3f(0), Vec3f(0), tangent, Vec3f(0, 0, 1), 1.0f0, Vec3f(0), Point2f(0), geom)
    return @testset "pbrt displacement keeps the original frame and BSDF" begin
        heights = (
            Hikari.TextureNodes.xyz().x * 0.02f0 + Hikari.TextureNodes.uv().y * 0.03f0,
            fill(0.02f0, 4, 4), Hikari.CheckerboardTexture(2.0f0, 3.0f0, 0.0f0, 0.0f0, Hikari.to_texture(0.01f0), Hikari.to_texture(0.03f0)),
        )
        for height in heights, model in (Hikari.Dielectric, Hikari.CoatedDiffuse, Hikari.Conductor), mode in (:constant, :graph_normal, :stored_normal)
            store = Raycore.MultiTypeSet(CPU())
            roughness = mode === :constant ? 0.2f0 : Hikari.TextureNodes.normal().x * 0.1f0 + 0.2f0
            mode === :stored_normal && (roughness = Hikari.device_param(store, roughness))
            m = model(roughness = roughness, displacement = height)
            store, original, graph, textures = stored(m, store)
            bumped_n, bumped_t = Hikari.perturb_bump_frame(original.displacement, textures, n, geom.dpdu, Vec3f(0, 2, 0), geom.dndu, geom.dndv, sc.ng, tfc)
            bumped_tfc = Hikari.TextureFilterContext(tfc.uv, tfc.dudx, tfc.dudy, tfc.dvdx, tfc.dvdy, tfc.face_idx, tfc.bary, tfc.p, bumped_n)
            reference = Hikari.get_bxdf(original, table, textures, bumped_tfc, lambda, false, wo, bumped_n, bumped_t)
            actual = evaluate_graph(graph, textures, sc, lambda)
            @test actual isa Hikari.FramedShaderBSDF
            @test actual.bsdf === reference
            @test actual.normal == bumped_n && actual.tangent == bumped_t
            # Gathered programs read the same frame and parameters as fused programs.
            prog = Hikari.shader_programs(textures, graph.program)[graph.program.idx]
            albedo, guide_normal = Hikari.shader_guides(typeof(graph).parameters[3], prog, textures, sc, Val(0))
            expected_albedo = Hikari.guide_material(original, textures, bumped_tfc, Val(0))
            @test albedo == Vec3f(expected_albedo.r, expected_albedo.g, expected_albedo.b)
            @test guide_normal == bumped_n
            s, i = Hikari.run_values(prog, textures, sc)
            inputs = Hikari.ShaderInputs(zeros(Float32, max(graph.inputs, 1)), Int32[], Int32(1))
            Hikari.gather_inputs!(inputs, Int32(1), s, prog, i)
            closures = Hikari.closure_programs(textures)[graph.closures]
            M = typeof(graph).parameters[3]
            produced = Hikari.node_lobes(M, Hikari.GatheredStack(inputs.values, Int32(1), Int32(1)), closures, Int32(1), sc, Hikari.LobeSetup(Hikari.lookup_tables(textures), false))
            @test Hikari.lobe_bsdf(M, produced, table, lambda) === reference
        end
    end
end
function run_composition()
    table = Hikari.get_srgb_table()
    lambda = Hikari.sample_wavelengths_stratified((0.1f0, 0.3f0, 0.5f0, 0.7f0))
    n, t = Vec3f(0, 0, 1), Vec3f(1, 0, 0)
    wo, wi = normalize(Vec3f(0.1, 0.2, 1)), normalize(Vec3f(-0.2, 0.1, 1))
    context(p) = Hikari.ShaderContext(Hikari.TextureFilterContext(Point2f(0.3), p, n), wo, Vec3f(0), Vec3f(0), t, n, 1.0f0, Vec3f(0), Point2f(0))
    value(b, textures, sc) = Hikari.evaluate_bsdf_spectral(b, table, textures, wo, wi, n, t, sc.tfc, lambda)
    @testset "native models mix and add with spectral weights" begin
        a = Hikari.Conductor(reflectance = (0.2, 0.4, 0.7), roughness = 0.2)
        b = Hikari.DiffuseTransmission(reflectance = (0.3, 0.2, 0.1), transmittance = (0.1, 0.2, 0.4))
        for (material, wa, wb) in ((Hikari.Mix(a, b; factor = 0.3f0), 0.7f0, 0.3f0), (Hikari.Add(a, b), 1.0f0, 1.0f0))
            store, _, graph, textures = stored(material)
            aa, bb = Hikari.to_device_material(store, a), Hikari.to_device_material(store, b)
            sc = context(Point3f(0))
            ba = Hikari.get_bxdf(aa, table, textures, sc.tfc, lambda, false, wo, n, t)
            bb = Hikari.get_bxdf(bb, table, textures, sc.tfc, lambda, false, wo, n, t)
            actual = evaluate_graph(graph, textures, sc, lambda)
            fa, pa = value(ba, textures, sc)
            fb, pb = value(bb, textures, sc)
            f, pdf = value(actual, textures, sc)
            @test all(isapprox.(f.data, (wa * fa + wb * fb).data))
            @test pdf ≈ (wa * pa + wb * pb) / (wa + wb)
        end
    end
    return @testset "layered composition preserves its conditional estimator" begin
        a, b = Hikari.CoatedDiffuse(roughness = 0.1), Hikari.Conductor(roughness = 0.2)
        for (material, wa, wb) in ((Hikari.Mix(a, b; factor = 0.3f0), 0.7f0, 0.3f0), (Hikari.Add(a, b), 1.0f0, 1.0f0))
            store, _, graph, textures = stored(material)
            aa, bb = Hikari.to_device_material(store, a), Hikari.to_device_material(store, b)
            sc = context(Point3f(0))
            ba = Hikari.get_bxdf(aa, table, textures, sc.tfc, lambda, false, wo, n, t)
            bb = Hikari.get_bxdf(bb, table, textures, sc.tfc, lambda, false, wo, n, t)
            fa, _ = value(ba, textures, sc)
            fb, _ = value(bb, textures, sc)
            accumulated = Hikari.SpectralRadiance(0.0f0)
            selected = zeros(Int, 2)
            for k in 1:2000
                sc = context(Point3f(Float32(k) * 0.001f0, 0.2, 0.3))
                actual = evaluate_graph(graph, textures, sc, lambda)
                selected[actual.selected] += 1
                f, _ = value(actual, textures, sc)
                accumulated += f
                sample = Hikari.sample_bsdf_spectral(actual, table, textures, wo, n, t, sc.tfc, lambda, Point2f(0.2, 0.4), 0.1f0)
                if sample.pdf > 0.0f0
                    @test sample.pdf_is_proportional == (actual.selected == 1)
                end
            end
            @test all(>(100), selected)
            @test isapprox(selected[2] / sum(selected), wb / (wa + wb); atol = 0.035)
            @test all(isapprox.((accumulated / 2000.0f0).data, (wa * fa + wb * fb).data; rtol = 0.05))
        end
        # Emission must be present whichever scattering component was selected.
        luminous = Hikari.Add(Hikari.Mix(a, Hikari.Glass(); factor = 0.3), Hikari.Emission(strength = 0.4))
        _, _, graph, textures = stored(luminous)
        values = [evaluate_graph(graph, textures, context(Point3f(k * 0.01, 0.2, 0.3)), lambda).emission for k in 1:20]
        @test all(==(first(values)), values)
        @test !Hikari.is_black(first(values))
    end
end
function run_updates()
    store = Raycore.MultiTypeSet(CPU())
    material(x) = Hikari.surface_material(Hikari.Conductor(roughness = Hikari.CheckerboardTexture(5.0f0, 5.0f0, 0.0f0, 0.0f0, Float32(x), 0.3f0)))
    key = push!(store, material(0.1))
    original = store[key]
    arrays, layout = length(store.texture_gpu_arrays), store.layout
    return @testset "graph parameter leaves update without taking arrays" begin
        for x in (0.15, 0.2, 0.25)
            @test Raycore.store!(store, key, material(x)) == key
            @test typeof(store[key]) === typeof(original)
            @test store[key].program == original.program
            @test (length(store.texture_gpu_arrays), store.layout) == (arrays, layout)
        end
    end
end
function run_parameter_features()
    store = Raycore.MultiTypeSet(CPU())
    original = Hikari.Conductor(roughness = Hikari.CheckerboardTexture(5.0f0, 5.0f0, 0.0f0, 0.0f0, 0.1f0, 0.3f0))
    shader = Hikari.surface_material(Hikari.Mix(original, Hikari.Glass(); factor = 0.3))
    compiled = Hikari.compiled(shader.program)
    return @testset "texture leaf features survive graph composition" begin
        @test typeof(first(compiled.instructions)).parameters[1] & Hikari.NodeFeature.IMAGE != 0
        native = Hikari.to_device_material(store, original)
        mat = Hikari.to_device_material(store, shader)
        textures = Raycore.StaticMultiTypeSet((), Tuple(T[a for a in store.texture_gpu_arrays if typeof(a) === T] for T in store.texture_order))
        tfc = Hikari.TextureFilterContext(Point2f(0.3, 0.4), 0.02f0, 0.01f0, 0.01f0, 0.02f0)
        n, t = Vec3f(0, 0, 1), Vec3f(1, 0, 0)
        wo = normalize(Vec3f(0.1, 0.2, 1))
        sc = Hikari.ShaderContext(tfc, wo, Vec3f(0), Vec3f(0), t, n, 1.0f0, Vec3f(0), Point2f(0))
        M = typeof(mat).parameters[3]
        prog = Hikari.shader_programs(textures, mat.program)[mat.program.idx]
        s, i = Hikari.run_values(prog, textures, sc)
        terms, _, _ = Hikari.node_lobes(M, s, prog, i, sc, Hikari.LobeSetup(Hikari.lookup_tables(textures), false))
        lambda = Hikari.sample_wavelengths_stratified((0.1f0, 0.3f0, 0.5f0, 0.7f0))
        table = Hikari.get_srgb_table()
        @test Hikari.pbrt_term_bsdf(first(terms), table, lambda) ===
            Hikari.get_bxdf(native, table, textures, tfc, lambda, false, wo, n, t)
    end
end
function run_gpu_composition()
    return @testset "native and Cycles nodes render in one graph" for hw in (false, true)
        backend = Mantle.defaultbackend()
        scene = Hikari.Scene(; backend, hw_accel = hw)
        native = Hikari.CoatedDiffuse(reflectance = (0.4, 0.2, 0.1), roughness = 0.2)
        materials = (
            Hikari.Add(Hikari.Mix(native, Hikari.Glass(roughness = 0.2); factor = 0.3), Hikari.Emission(strength = 0.2)),
            Hikari.Mix(native, Hikari.SubsurfaceScattering(color = (0.8, 0.4, 0.2), scale = 0.1); factor = 0.5),
            Hikari.Mix(Hikari.Dielectric(ior = Hikari.AU_ETA_SPECTRUM, roughness = 0.2), Hikari.Diffuse(); factor = 0.5),
            Hikari.Mix(
                Hikari.Conductor(roughness = 0.2, displacement = Hikari.TextureNodes.xyz().x * 0.02f0),
                Hikari.Glossy(roughness = 0.3); factor = 0.5
            ),
        )
        for (i, material) in enumerate(materials)
            center = Point3f((i - 2.5f0) * 1.2f0, 0, 0)
            push!(scene, GeometryBasics.normal_mesh(GeometryBasics.Tesselation(GeometryBasics.Sphere(center, 0.5f0), 16)), material)
        end
        push!(scene, Hikari.PointLight(Point3f(0, -2, 4), Hikari.RGBSpectrum(50.0f0)))
        push!(scene, Hikari.AmbientLight(Hikari.RGBSpectrum(0.2f0)))
        Hikari.sync!(scene)
        film = Hikari.Film(backend, Hikari.Film(Point2f(64, 24)))
        camera = Hikari.PerspectiveCamera(Point3f(0, -5, 1), Point3f(0), film; fov = 26.0f0)
        vp = Hikari.VolPath(samples = 16, max_depth = 4, hw_accel = hw)
        try
            vp(scene, film, camera)
            img = Array(film.framebuffer)
            @test all(p -> isfinite(p.r) && isfinite(p.g) && isfinite(p.b), img)
            for columns in (5:12, 21:28, 37:44, 53:60)
                @test sum(p -> p.r + p.g + p.b, img[9:16, columns]) > 0.1
            end
        finally
            close(vp)
            Hikari.free!(film)
        end
    end
end
run_exact()
run_bump()
run_composition()
run_updates()
run_parameter_features()
run_gpu_composition()
end
