using Test
using Hikari
using Hikari.TextureNodes
using Hikari: TextureFilterContext, RGBSpectrum, device_param, NodeOp, DiffuseLobe, MicrofacetLobe, SubsurfaceLobe,
    F82Tint, SchlickFresnel, DielectricFresnel
using GeometryBasics
using GeometryBasics: Point2f, Point3f, Vec3f, Vec2f, GLTriangleFace, QuadFace
using Hikari: SVector
using Colors: RGB
using KernelAbstractions: CPU
using LinearAlgebra
using Raycore
using Mantle
using Statistics: mean

"""A description's shader, whatever `surface_material` would render it with."""
shader_material(description) = Hikari.ShaderMaterial(Hikari.shader_node(description))

static_textures(store) =
    Raycore.StaticMultiTypeSet((), Tuple(T[a for a in store.texture_gpu_arrays if typeof(a) === T] for T in store.texture_order))

"""What the shader program of the stored material `mat` produces at `sc`: its lobes, transparency and emission."""
function produced(store, mat::Hikari.ShaderMaterial{P, A, M}, sc) where {P, A, M}
    textures = static_textures(store)
    return Hikari.shader_lobes(M, Hikari.shader_programs(textures, mat.program)[mat.program.idx], textures, sc, false)
end

"""The lobes that take samples: the others are slots of the material's type no layer filled at this hit."""
present(lobes) = [w for w in lobes if w.sample_weight > 0]

function shade(shader; p = Point3f(0), n = Vec3f(0, 0, 1), wo = Vec3f(0, 0, 1),
               dpdx = Vec3f(1f-3, 0, 0), dpdy = Vec3f(0, 1f-3, 0), tangent_sign = 1f0)
    store = Raycore.MultiTypeSet(CPU())
    mat = Hikari.to_device_material(store, shader_material(shader))
    sc = Hikari.ShaderContext(TextureFilterContext(Point2f(0), p, n), wo, dpdx, dpdy, Vec3f(1, 0, 0), n, tangent_sign,
                              Vec3f(0), Point2f(0))
    return produced(store, mat, sc)
end
lobes(shader; kw...) = present(first(shade(shader; kw...)))

"""The lobes of `shader` bound to `mesh`, at face `face` and barycentrics `bary`, the uv there moving by `duv` per pixel."""
function lobes_on(mesh, shader; face = 1, bary = (1 / 3, 1 / 3, 1 / 3), n = Vec3f(0, 0, 1), wo = Vec3f(0, 0, 1),
                  uv = Point2f(0), duv = (0f0, 0f0, 0f0, 0f0), dpdx = Vec3f(1f-3, 0, 0), dpdy = Vec3f(0, 1f-3, 0))
    store = Raycore.MultiTypeSet(CPU())
    _, material = Hikari.bind_mesh(mesh, shader_material(shader))
    mat = Hikari.to_device_material(store, material)
    tfc = TextureFilterContext(uv, duv..., UInt32(face), SVector{3, Float32}(bary), Point3f(0), n)
    sc = Hikari.ShaderContext(tfc, wo, dpdx, dpdy, Vec3f(1, 0, 0), n, 1f0, Vec3f(0), Point2f(0))
    return present(first(produced(store, mat, sc)))
end

function program(shader)
    store = Raycore.MultiTypeSet(CPU())
    mat = Hikari.to_device_material(store, shader_material(shader))
    return Hikari.shader_programs(static_textures(store), mat.program)[mat.program.idx]
end

features(::Hikari.ShaderProgram{F}) where {F} = F & ~(UInt32(3) << Hikari.STACK_SHIFT)

# A shader is shaded in two kernels: the values kernel runs the program and gathers what the BSDF
# nodes read of it (`gather_inputs!`), the closure kernel runs the nodes from that and the closure
# program (`closure_program`), so a node edit compiles the small values kernel alone (13 s to 3-4 s
# a node edit on an RTX 4000 Ada). The two halves must produce what the program run whole does:
# every input the nodes read off the stack gathered, in the order the closure program numbered it.
@testset "the values and closure halves produce what one program does" begin
    split_shaders = [
        Hikari.Principled(base_color = noise(xyz()).color, roughness = wave(xyz()).fac * 0.5, metallic = checker(xyz()).fac,
                          normal = bump(noise(xyz(); scale = 4).fac), coat_weight = 1, coat_tint = Hikari.TextureNodes.magic(xyz()).color,
                          sheen_weight = 0.5, sheen_tint = voronoi(xyz()).color, emission_strength = xyz().x,
                          emission_color = gradient(xyz()).color, alpha = mix(0.3, 1.0, xyz().y)),
        Hikari.Principled(base_color = (0.8, 0.3, 0.2), transmission_weight = brick(xyz()).fac, roughness = 0.2,
                          subsurface_weight = 0.5, subsurface_radius = noise(xyz()).color),
        Hikari.Mix(Hikari.Diffuse(color = wave(xyz()).color, normal = normal_map((0.5, 0.75, 1.0))),
                   Hikari.Glossy(roughness = noise(xyz()).fac); factor = xyz().x),
        Hikari.Add(Hikari.Transparent(color = checker(xyz()).color), Hikari.Emission(color = (1, 0.5, 0.2), strength = xyz().z)),
        Hikari.Diffuse(color = (0.2, 0.4, 0.6)),
        Hikari.Refraction(color = noise(xyz()).color, roughness = wave(xyz()).fac,
                           ior = 1.3 + xyz().x, normal = bump(noise(xyz()).fac)),
        Hikari.Refraction(distribution = :ggx, roughness = noise(xyz()).fac),
        Hikari.Metallic(base_color = noise(xyz()).color, edge_tint = (0.9, 0.8, 0.7), roughness = 0.3,
                        thin_film_thickness = wave(xyz()).fac * 500, thin_film_ior = 1.33),
        Hikari.Mix(Hikari.Glass(color = (0.8, 0.9, 1.0), thin_film_thickness = noise(xyz()).fac * 400),
                   Hikari.Diffuse(color = (0.4, 0.6, 0.2)); factor = 0.6),
        Hikari.Mix(Hikari.Toon(color = noise(xyz()).color, size = wave(xyz()).fac, smooth = 0.2),
                   Hikari.Toon(component = :glossy, size = 0.3, smooth = noise(xyz()).fac); factor = 0.4),
        Hikari.Mix(Hikari.Principled(subsurface_weight = 1, subsurface_scale = 0.1),
                   Hikari.Principled(subsurface_weight = 1, subsurface_scale = noise(xyz()).fac * 0.02);
                   factor = wave(xyz()).fac),
        Hikari.SubsurfaceScattering(color = noise(xyz()).color, scale = wave(xyz()).fac * 0.1,
                                    radius = (1, 0.5, 0.25), roughness = 0.3, normal = bump(noise(xyz()).fac)),
    ]
    for shader in split_shaders
        store = Raycore.MultiTypeSet(CPU())
        mat = Hikari.to_device_material(store, shader_material(shader))
        M = typeof(mat).parameters[3]
        textures = static_textures(store)
        sc = Hikari.ShaderContext(TextureFilterContext(Point2f(0.3, 0.6), Point3f(0.3, -0.2, 0.7), Vec3f(0, 0, 1)),
                                  normalize(Vec3f(0.2, 0.1, 1)), Vec3f(1f-3, 0, 0), Vec3f(0, 1f-3, 0), Vec3f(1, 0, 0),
                                  Vec3f(0, 0, 1), 1f0, Vec3f(0), Point2f(0))
        whole = produced(store, mat, sc)
        # The hit in the middle of three, so a stride or an offset gone wrong reads a neighbour.
        capacity, hit = Int32(3), Int32(2)
        inputs = Hikari.ShaderInputs(fill(NaN32, max(Int(mat.inputs), 1) * capacity), zeros(Int32, capacity), capacity)
        prog = Hikari.shader_programs(textures, mat.program)[mat.program.idx]
        s, i = Hikari.run_values(prog, textures, sc)
        Hikari.gather_inputs!(inputs, hit, s, prog, i)
        @test count(!isnan, inputs.values) == mat.inputs
        closures = Hikari.closure_programs(textures)[mat.closures]
        halves = Hikari.node_lobes(M, Hikari.GatheredStack(inputs.values, capacity, hit), closures, Int32(1), sc,
                                   Hikari.LobeSetup(Hikari.lookup_tables(textures), false))
        @test halves == whole
    end
end

@testset "Shader materials" begin
    @testset "Toon's component fixes its lobe type" begin
        diffuse = only(lobes(Hikari.Toon(color = (0.2, 0.4, 0.6), size = 0.3, smooth = 0.2)))
        glossy = only(lobes(Hikari.Toon(component = :glossy)))
        @test diffuse.lobe isa Hikari.ToonLobe{false}
        @test diffuse.weight ≈ Vec3f(0.2, 0.4, 0.6)
        @test diffuse.lobe.size ≈ 0.3f0 * Float32(π) / 2f0
        @test diffuse.lobe.smooth ≈ 0.2f0 * Float32(π) / 2f0
        @test glossy.lobe isa Hikari.ToonLobe{true}
        @test_throws ArgumentError Hikari.shader_node(Hikari.Toon(component = :invalid))
    end
    @testset "Glass and Metallic take Blender's thin-film inputs" begin
        glass = Hikari.Glass(ior = 1.5, thin_film_thickness = 450, thin_film_ior = 1.33)
        front = only(lobes(glass)).lobe
        back = only(lobes(glass; wo = Vec3f(0, 0, -1))).lobe
        @test front.fresnel isa SchlickFresnel
        @test front.fresnel.film_thickness == 450f0
        @test front.fresnel.film_ior ≈ 1.33f0
        @test back.ior ≈ 1f0 / 1.5f0
        @test back.fresnel.film_ior ≈ 1.33f0 / 1.5f0
        metal = only(lobes(Hikari.Metallic(thin_film_thickness = 350, thin_film_ior = 1.4))).lobe
        @test metal.fresnel isa F82Tint
        @test metal.fresnel.film_thickness == 350f0
        @test metal.fresnel.film_ior ≈ 1.4f0
        @test Hikari.bsdf_nodes(Hikari.ShaderGraph(Hikari.shader_node(Hikari.Glass()))) == Tuple{Hikari.GlassBSDF}
        @test Hikari.bsdf_nodes(Hikari.ShaderGraph(Hikari.shader_node(Hikari.Metallic()))) == Tuple{Hikari.MetallicBSDF}
    end

    @testset "thin-film interfaces keep the substrate's Fresnel response" begin
        # With no IOR change at the top interface, the film leaves the F82
        # substrate response unchanged. Using physical conductor amplitudes
        # instead loses the artist's edge tint, even with the correct phase.
        for f0 in (0.2f0, 0.6f0, 0.9f0), tint in (0.5f0, 0.9f0), mu in (0.1f0, 0.4f0, 0.9f0)
            m = 1f0 - mu
            expected = clamp(f0 + (1f0 - f0) * m^5 - Hikari._f82_b_channel(f0, tint) * mu * m^6, 0f0, 1f0)
            actual = Hikari.film_reflectance(F82Tint, 1f0, 400f0, 1f0, mu, f0, tint, 550f0)
            @test actual ≈ expected atol = 2f-5
        end
        # A film matching the substrate is just the original dielectric,
        # including total internal reflection when viewed from inside.
        for mu in (0.1f0, 0.5f0, 0.9f0), ior in (1.5f0, 1f0 / 1.5f0)
            film_ior = ior > 1f0 ? 1.5f0 : 1f0
            actual = Hikari.film_reflectance(SchlickFresnel, ior, 550f0, film_ior, mu,
                                             Hikari._f0_from_ior(ior), 1f0, 550f0)
            @test actual ≈ Hikari.fresnel_dielectric(mu, ior) atol = 1f-5
        end
    end

    @testset "a diffuse BSDF is one diffuse lobe" begin
        w = only(lobes(Hikari.Diffuse(color = (0.2, 0.4, 0.6))))
        @test w.lobe isa DiffuseLobe
        @test w.weight ≈ Vec3f(0.2, 0.4, 0.6)
        @test w.n == Vec3f(0, 0, 1)
    end

    @testset "a refraction node retains its distribution and inputs" begin
        for (distribution, convention) in ((:beckmann, Hikari.CyclesBeckmann), (:ggx, Hikari.CyclesGGX))
            material = Hikari.Refraction(color = (0.3, 0.6, 0.9), roughness = 0.4, ior = 1.6; distribution)
            w = only(lobes(material))
            @test w.lobe isa MicrofacetLobe{:refraction, <:Hikari.NoFresnel, convention}
            @test w.lobe.alpha_x ≈ 0.16f0
            @test w.lobe.ior == 1.6f0
            @test w.lobe.energy == 1f0
            @test w.weight ≈ Vec3f(0.3, 0.6, 0.9)
            back = only(lobes(material; wo = Vec3f(0, 0, -1)))
            @test back.lobe.ior ≈ 1f0 / 1.6f0
        end
        @test_throws ArgumentError Hikari.shader_node(Hikari.Refraction(distribution = :other))
    end

    @testset "mix shader weighs its two shaders by its factor" begin
        red, blue = Hikari.Diffuse(color = (1, 0, 0)), Hikari.Diffuse(color = (0, 0, 1))
        @test [w.weight for w in lobes(Hikari.Mix(red, blue; factor = 0.25))] ≈ [Vec3f(0.75, 0, 0), Vec3f(0, 0, 0.25)]
        @test [w.weight for w in lobes(Hikari.Mix(red, blue; factor = xyz().x); p = Point3f(0.3, 0, 0))] ≈
              [Vec3f(0.7, 0, 0), Vec3f(0, 0, 0.3)]
        @test [w.weight for w in lobes(Hikari.Add(red, blue))] ≈ [Vec3f(1, 0, 0), Vec3f(0, 0, 1)]
        # Cycles clamps a Mix Shader's factor before propagating its weights.
        # Without it, a factor of 2 doubled the second shader's brightness.
        a, b = Hikari.Diffuse(color = (0.2, 0.1, 0.1)), Hikari.Diffuse(color = (0.1, 0.2, 0.3))
        for factor in (-1f0, 0f0, 0.4f0, 1f0, 2f0)
            fac = clamp(factor, 0f0, 1f0)
            expected = filter(w -> maximum(w) > 0, [Vec3f(a.color...) * (1f0 - fac), Vec3f(b.color...) * fac])
            for input in (factor, xyz().x)
                actual = lobes(Hikari.Mix(a, b; factor = input); p = Point3f(factor, 0, 0))
                @test [w.weight for w in actual] ≈ expected
            end
        end
    end

    @testset "a metallic principled BSDF is one F82 tint lobe" begin
        w = only(lobes(Hikari.Principled(base_color = (0.9, 0.6, 0.2), metallic = 1, roughness = 0.4)))
        @test w.lobe isa MicrofacetLobe{:reflection, <:F82Tint}
        @test w.lobe.fresnel.f0 ≈ Vec3f(0.9, 0.6, 0.2)
        @test w.lobe.alpha_x ≈ w.lobe.alpha_y ≈ 0.16f0
    end

    @testset "the specular layer of a principled BSDF takes its reflection from the diffuse" begin
        specular, diffuse = lobes(Hikari.Principled(base_color = (0.8, 0.8, 0.8), roughness = 0))
        @test specular.lobe isa MicrofacetLobe{:reflection, <:SchlickFresnel}
        @test specular.lobe.fresnel.f0 ≈ Vec3f(0.04)
        @test diffuse.lobe isa DiffuseLobe
        @test diffuse.weight ≈ Vec3f(0.8 * 0.96) rtol = 1e-3
    end

    @testset "a coat takes its reflection from the layers under it" begin
        coat, specular, diffuse = lobes(Hikari.Principled(base_color = (0.8, 0.8, 0.8), roughness = 0, coat_weight = 1,
                                                        coat_roughness = 0))
        @test coat.lobe isa MicrofacetLobe{:reflection, <:DielectricFresnel}
        @test coat.lobe.alpha_x == 0f0
        @test specular.weight ≈ Vec3f(0.96) rtol = 1e-3
        @test diffuse.weight ≈ Vec3f(0.8 * 0.96 * 0.96) rtol = 1e-3
    end

    @testset "subsurface is a random walk lobe with the radius scaled" begin
        ws = lobes(Hikari.Principled(base_color = (0.9, 0.6, 0.5), roughness = 0.4, subsurface_weight = 1,
                                   subsurface_radius = (1, 0.5, 0.25), subsurface_scale = 0.1))
        sss = only(filter(w -> w.lobe isa SubsurfaceLobe, ws))
        @test sss.lobe.albedo ≈ Vec3f(0.9, 0.6, 0.5)
        @test sss.lobe.radius ≈ Vec3f(0.1, 0.05, 0.025)
        @test sss.lobe.alpha ≈ 0.16f0
        @test !any(w -> w.lobe isa DiffuseLobe, ws)
    end

    @testset "standalone subsurface inputs and zero-radius channels" begin
        ws = lobes(Hikari.SubsurfaceScattering(color = (0.9, 0.6, 0.4), radius = (1, 0.5, 0.25),
                                               scale = 0.1, roughness = 0.3, ior = 1.6, anisotropy = 0.4))
        walk = only(ws)
        @test walk.lobe isa SubsurfaceLobe
        @test walk.lobe.radius ≈ Vec3f(0.1, 0.05, 0.025)
        @test walk.lobe.albedo ≈ Vec3f(0.9, 0.6, 0.4)
        @test walk.lobe.alpha == 0.3f0
        @test walk.lobe.ior == 1.6f0
        @test walk.lobe.anisotropy == 0.4f0
        @test walk.sample_weight ≈ 1.9f0
        @test Hikari.bsdf_nodes(Hikari.ShaderGraph(Hikari.shader_node(Hikari.SubsurfaceScattering(scale = 0)))) ==
              Tuple{Hikari.SubsurfaceBSDF{false}}
        tilted = normal_map((1, 0.5, 0.55); space = :world)
        constant = only(lobes(Hikari.SubsurfaceScattering(scale = 0, normal = tilted)))
        dynamic = only(lobes(Hikari.SubsurfaceScattering(scale = xyz().x, normal = tilted)))
        @test constant.n ≈ dynamic.n
        @test 2f0 * constant.n[3]^2 - 1f0 >= 0.0099f0
        @test constant.lobe isa DiffuseLobe
        @test !any(ins -> ins.op == NodeOp.NOISE,
                   program(Hikari.SubsurfaceScattering(scale = 0, radius = noise(xyz()).color)))
        for material in (
            Hikari.SubsurfaceScattering(color = (1, 1, 1), radius = (1, 0, 1), scale = 0.1),
            Hikari.Principled(base_color = (1, 1, 1), specular_ior_level = 0, subsurface_weight = 1,
                              subsurface_radius = (1, 0, 1), subsurface_scale = 0.1, diffuse_roughness = 1),
        )
            ws = lobes(material)
            @test length(ws) == 2
            walk = only(filter(w -> w.lobe isa SubsurfaceLobe, ws))
            diffuse = only(filter(w -> w.lobe isa DiffuseLobe, ws))
            @test walk.weight == Vec3f(1, 0, 1)
            @test diffuse.weight == Vec3f(0, 1, 0)
            @test diffuse.lobe.a == 1f0 / Float32(π)
            @test diffuse.lobe.b == 0f0
        end
        # Cycles clamps radius after multiplying it by scale.
        negative = only(filter(w -> w.lobe isa SubsurfaceLobe,
            lobes(Hikari.Principled(subsurface_weight = 1, subsurface_radius = (-1, -2, -3), subsurface_scale = -0.1))))
        @test negative.lobe.radius ≈ Vec3f(0.1, 0.2, 0.3)
    end

    @testset "the material type is the shader's BSDF nodes" begin
        nodes(::Hikari.ShaderMaterial{P, A, M}) where {P, A, M} = M
        transparent(::Hikari.ShaderMaterial{P, A}) where {P, A} = A !== Nothing
        material(shader) = shader_material(shader)
        P = Hikari.PrincipledBSDF
        @test nodes(material(Hikari.Principled())) == Tuple{P{false, false, false}}
        @test !transparent(material(Hikari.Principled()))
        @test transparent(material(Hikari.Principled(alpha = 0.5)))
        @test nodes(material(Hikari.Principled(subsurface_weight = 1))) == Tuple{P{false, true, false}}
        @test nodes(material(Hikari.Principled(subsurface_weight = 1, thin_wall = true))) == Tuple{P{true, true, false}}
        @test nodes(material(Hikari.Principled(emission_strength = xyz().x))) == Tuple{P{false, false, true}}
        mixed = material(Hikari.Mix(Hikari.Diffuse(), Hikari.Transparent(); factor = 0.5))
        @test nodes(mixed) == Tuple{Hikari.DiffuseBSDF, Hikari.TransparentBSDF}
        @test transparent(mixed)
    end

    # The lobes of a material are its nodes' slots, whatever a hit fills of them, so a shader
    # keeps every lobe it produces. The closure list this replaced held at most eight, and
    # dropped the rest of a graph of more.
    @testset "a shader holds every lobe its nodes produce" begin
        slots(shader) = length(first(shade(shader)))
        @test slots(Hikari.Principled()) == 6
        @test slots(Hikari.Principled(transmission_weight = 1, thin_wall = true)) == 7
        @test slots(Hikari.Principled(subsurface_weight = 1)) == 8
        @test length(lobes(Hikari.Principled(metallic = 0.5, coat_weight = 1))) == 4
        heavy = Hikari.Add(Hikari.Principled(base_color = (0.8, 0.3, 0.2), roughness = 0.4, metallic = 0.3, sheen_weight = 1,
                                           coat_weight = 1, transmission_weight = 0.5),
                           Hikari.Principled(base_color = (0.5, 0.5, 0.5), roughness = 0.2, sheen_weight = 0.5,
                                           coat_weight = 1))
        @test length(lobes(heavy)) == 10
    end

    @testset "a device shader reads only programs of its own type" begin
        store = Raycore.MultiTypeSet(CPU())
        plain = Hikari.to_device_material(store, shader_material(Hikari.Diffuse(color = (0.2, 0.4, 0.6))))
        noisy = Hikari.to_device_material(store, shader_material(Hikari.Diffuse(color = noise(xyz()).color)))
        @test features(plain.program) == 0
        @test features(noisy.program) == Hikari.NodeFeature.NOISE
        @test typeof(plain.program) != typeof(noisy.program)
        sc = Hikari.ShaderContext(TextureFilterContext(Point2f(0), Point3f(0), Vec3f(0, 0, 1)), Vec3f(0, 0, 1),
                                  Vec3f(0), Vec3f(0), Vec3f(1, 0, 0), Vec3f(0, 0, 1), 1f0, Vec3f(0), Point2f(0))
        @test only(first(produced(store, plain, sc))).weight ≈ Vec3f(0.2, 0.4, 0.6)
    end

    @testset "a shader program holds only the shader code it runs" begin
        store = Raycore.MultiTypeSet(CPU())
        shiny = Hikari.Principled(base_color = noise(xyz()).color, coat_weight = 1, normal = bump(noise(xyz()).fac))
        mat = Hikari.to_device_material(store, shader_material(shiny))
        N = Hikari.NodeFeature
        @test features(mat.program) == N.NOISE | N.BUMP
    end

    @testset "a lobe tilted off the smooth normal is shadowed as in Cycles" begin
        ns, tilted = Vec3f(0, 0, 1), normalize(Vec3f(1, 0, 1))
        @test Hikari.bump_shadowing(true, ns, ns, normalize(Vec3f(1, 0, 1))) == 1f0
        @test Hikari.bump_shadowing(true, tilted, ns, ns) == 1f0
        @test Hikari.bump_shadowing(true, tilted, ns, normalize(Vec3f(1, 0, 1))) ≈ 1 / (1 + (sqrt(1.125) - 1) / 2)
        @test Hikari.bump_shadowing(true, tilted, ns, normalize(Vec3f(-1, 0, 0.2))) == 0f0
        @test Hikari.bump_shadowing(false, tilted, ns, normalize(Vec3f(-1, 0, 0.2))) == 0f0
        @test Hikari.bump_shadowing(false, tilted, ns, normalize(Vec3f(1, 0, 1))) == 1f0
    end

    @testset "a tangent space normal map tilts the normal along the tangent frame" begin
        mapped(colour; kw...) = only(lobes(Hikari.Diffuse(normal = normal_map(colour; kw...)))).n
        @test mapped((0.5, 0.5, 1.0)) ≈ Vec3f(0, 0, 1)
        @test mapped((0.75, 0.5, 1.0)) ≈ normalize(Vec3f(0.5, 0, 1))
        @test mapped((0.5, 0.75, 1.0)) ≈ normalize(Vec3f(0, 0.5, 1))
        @test only(lobes(Hikari.Diffuse(normal = normal_map((0.5, 0.75, 1.0))); tangent_sign = -1f0)).n ≈
              normalize(Vec3f(0, -0.5, 1))
        @test mapped((0.5, 0.75, 1.0); invert_green = true) ≈ normalize(Vec3f(0, -0.5, 1))
        @test mapped((0.75, 0.5, 0.5); strength = 0.5) ≈ normalize(Vec3f(0.25, 0, 0.5))
        @test mapped((0.75, 0.5, 1.0); strength = 0) ≈ Vec3f(0, 0, 1)
    end

    @testset "a world space normal map is the normal its colour encodes" begin
        @test only(lobes(Hikari.Diffuse(normal = normal_map((1.0, 0.5, 0.5); space = :world)))).n ≈ Vec3f(1, 0, 0)
        @test only(lobes(Hikari.Diffuse(normal = normal_map((0.5, 0.5, 0.0); space = :blender_world)))).n ≈
              Vec3f(0, 0, 1)
    end

    @testset "a mesh attribute is interpolated over the hit's triangle" begin
        tri = GeometryBasics.Mesh(Point3f[(0, 0, 0), (1, 0, 0), (0, 1, 0)], [GLTriangleFace(1, 2, 3)];
                                  paint = Vec3f[(0.2, 0, 0), (0, 0.4, 0), (0, 0, 0.6)],
                                  color = [RGB{Float32}(1, 0, 0), RGB{Float32}(0.5, 0.5, 0.5), RGB{Float32}(0, 0, 1)])
        weight(shader; kw...) = only(lobes_on(tri, shader; kw...)).weight
        @test weight(Hikari.Diffuse(color = attribute(:paint).color); bary = (0.5, 0.25, 0.25)) ≈ Vec3f(0.1, 0.1, 0.15)
        @test weight(Hikari.Diffuse(color = attribute(:paint).fac); bary = (1, 0, 0)) ≈ Vec3f(0.2 / 3)
        @test weight(Hikari.Diffuse(color = attribute(:color).color); bary = (0, 1, 0)) ≈ Vec3f(0.21404114)
        @test isempty(lobes_on(tri, Hikari.Diffuse(color = attribute(:missing).color)))
    end

    @testset "a quad's second triangle reads its own corners" begin
        quad = GeometryBasics.Mesh(Point3f[(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [QuadFace(1, 2, 3, 4)];
                                   paint = Vec3f[(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)])
        corners = [only(lobes_on(quad, Hikari.Diffuse(color = attribute(:paint).color); face = 2, bary = b)).weight
                   for b in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
        @test Set(corners) == Set([Vec3f(1, 0, 0), Vec3f(0, 0, 1), Vec3f(1, 1, 1)])
    end

    @testset "generated and object coordinates are the mesh's own positions" begin
        tri = GeometryBasics.Mesh(Point3f[(1, 2, 3), (3, 2, 3), (1, 6, 4)], [GLTriangleFace(1, 2, 3)])
        coordinate(which) = only(lobes_on(tri, Hikari.Diffuse(color = getproperty(texture_coordinate(), which));
                                             bary = (0, 1, 0))).weight
        @test coordinate(:object) ≈ Vec3f(3, 2, 3)
        @test coordinate(:generated) ≈ Vec3f(1, 0, 0)
    end

    @testset "reflection coordinates mirror the view about the normal" begin
        reflected(wo) = only(lobes(Hikari.Diffuse(color = texture_coordinate().reflection * 0.5 + 0.5); wo)).weight
        @test reflected(Vec3f(0, 0, 1)) ≈ Vec3f(0.5, 0.5, 1)
        @test reflected(normalize(Vec3f(1, 0, 1))) ≈ Vec3f(0.5 - 0.5 / sqrt(2), 0.5, 0.5 + 0.5 / sqrt(2))
    end

    @testset "a bump on a linear attribute tilts the normal by its slope" begin
        tri = GeometryBasics.Mesh(Point3f[(0, 0, 0), (1, 0, 0), (0, 1, 0)], [GLTriangleFace(1, 2, 3)];
                                  uv = Vec2f[(0, 0), (1, 0), (0, 1)], height = Float32[0, 1, 0])
        c = only(lobes_on(tri, Hikari.Diffuse(normal = bump(attribute(:height).fac; distance = 0.5));
                             uv = Point2f(0.25, 0.25), duv = (1f-3, 0f0, 0f0, 1f-3)))
        @test c.n ≈ normalize(Vec3f(-0.5, 0, 1))
    end

    @testset "a bump on a linear height tilts the normal by its slope" begin
        c = only(lobes(Hikari.Diffuse(normal = bump(xyz().x; distance = 0.5))))
        @test c.n ≈ normalize(Vec3f(-0.5, 0, 1))
        c = only(lobes(Hikari.Diffuse(normal = bump(xyz().x; distance = 0.5, invert = true))))
        @test c.n ≈ normalize(Vec3f(0.5, 0, 1))
        c = only(lobes(Hikari.Diffuse(normal = bump(xyz().x; distance = 0.5, strength = 0))))
        @test c.n ≈ Vec3f(0, 0, 1)
    end

    # An alpha test ran the whole shader for the transparency it reads: a textured,
    # bumped, coated sheet ran its noise, the bump's three evaluations and every layer
    # of the Principled BSDF at each shadow ray crossing it.
    @testset "an alpha test runs only the shader's transparency" begin
        store = Raycore.MultiTypeSet(CPU())
        leaf = image_texture(fill(RGB{Float32}(0.3, 0.6, 0.2), 4, 4))
        sheet = Hikari.Principled(base_color = leaf, coat_weight = 1, normal = bump(noise(xyz()).fac),
                                alpha = xyz().x * leaf.y)
        shader = Hikari.Mix(sheet, Hikari.Transparent(color = (0.2, 0.4, 0.6)); factor = xyz().y)
        mat = Hikari.to_device_material(store, shader_material(shader))
        N = Hikari.NodeFeature
        @test features(mat.alpha) & (N.NOISE | N.BUMP) == 0
        # The image both programs read is stored once.
        @test count(a -> eltype(a) === RGBSpectrum && ndims(a) == 2, store.texture_gpu_arrays) == 1
        textures = static_textures(store)
        for (x, y) in ((0.25, 0.5), (0.75, 0.1), (-0.5, 0.9), (1.5, 0.3))
            p = Point3f(x, y, 0)
            tfc = TextureFilterContext(Point2f(0.5), p, Vec3f(0, 0, 1))
            sc = Hikari.ShaderContext(tfc, Vec3f(0, 0, 1), Vec3f(0), Vec3f(0), Vec3f(0), Vec3f(0, 0, 1), 1f0, Vec3f(0),
                                      Point2f(0))
            # What the transparency program lets through is what the whole shader does, colour and all:
            # the Transparent BSDF tints it.
            whole = clamp.(produced(store, mat, sc)[2], 0f0, 1f0)
            @test Hikari.surface_transparency(mat, textures, tfc) ≈ whole
        end
        opaque = Hikari.to_device_material(store, shader_material(Hikari.Principled()))
        @test opaque.alpha === nothing
    end

    @testset "a bump evaluates only what reads the position three times" begin
        fixed = noise(constant((0.3, 0.2, 0.1)))
        ops = [ins.op for ins in program(Hikari.Diffuse(normal = bump(xyz().x * fixed)))]
        @test count(==(NodeOp.NOISE), ops) == 1
        @test count(==(NodeOp.MATH), ops) == 3
    end

    @testset "a principled BSDF runs only the inputs its weights read" begin
        @test [ins.op for ins in program(Hikari.Principled())] ==
              [NodeOp.PRINCIPLED, NodeOp.ARGS, NodeOp.ARGS, NodeOp.ARGS, NodeOp.RESULT]
        @test count(==(NodeOp.POSITION), [ins.op for ins in program(Hikari.Principled(anisotropic = 0.5))]) == 1
    end

    # The constants a BSDF node reads are folded into its instructions: the default normal is the
    # shading normal, and a colour is in the data after the node, so neither is a value the program
    # computes and stores before the node reads it.
    @testset "a BSDF node reads its constants from its own instructions" begin
        @test [ins.op for ins in program(Hikari.Principled(subsurface_weight = 1))] ==
              [NodeOp.PRINCIPLED, NodeOp.ARGS, NodeOp.ARGS, NodeOp.ARGS, NodeOp.DATA, NodeOp.RESULT]
        @test [ins.op for ins in program(Hikari.Diffuse(color = (0.9, 0.6, 0.5)))] ==
              [NodeOp.CLOSURE, NodeOp.DATA, NodeOp.RESULT]
        tilted = normalize(Vec3f(0, 0.2, 1))
        w = only(lobes(Hikari.Diffuse(color = (0.9, 0.6, 0.5)); n = tilted))
        @test w.weight ≈ Vec3f(0.9, 0.6, 0.5)
        @test w.n ≈ tilted
    end

    @testset "a texture cannot use a bump" begin
        @test_throws "a texture cannot use `bump`" device_param(Raycore.MultiTypeSet(CPU()), bump(xyz().x))
    end

    @testset "renders as the Diffuse material ($(hw ? "hardware" : "software") tracing)" for hw in (false, true)
        function render(material)
            backend = Mantle.defaultbackend()
            scene = Hikari.Scene(; backend, hw_accel = hw)
            sphere = GeometryBasics.normal_mesh(GeometryBasics.Tesselation(GeometryBasics.Sphere(Point3f(0), 1f0), 32))
            push!(scene, sphere, material)
            push!(scene, Hikari.PointLight(Point3f(0, 0.5, 4), RGBSpectrum(40f0)))
            Hikari.sync!(scene)
            film = Hikari.Film(backend, Hikari.Film(Point2f(32, 32)))
            camera = Hikari.PerspectiveCamera(Point3f(0, 0, 4), Point3f(0), film; fov = 40f0)
            vp = Hikari.VolPath(samples = 16, max_depth = 3, hw_accel = hw)
            vp(scene, film, camera)
            img = Array(film.framebuffer)
            close(vp)
            return [mean(getfield(p, c) for p in img) for c in (:r, :g, :b)]
        end
        colour = (0.7, 0.3, 0.1)
        # With a normal, a Diffuse is a shader; without, it has its own kernel.
        shader, own = Hikari.Diffuse(color = colour, normal = normal()), Hikari.Diffuse(color = colour)
        @test Hikari.surface_material(shader) isa Hikari.ShaderMaterial && Hikari.surface_material(own) === own
        @test render(shader) ≈ render(own) rtol = 0.02
        @test render(Hikari.SubsurfaceScattering(color = colour, scale = 0)) ≈ render(shader) rtol = 0.02
        # Both constant and node-driven factors saturate before weighting;
        # a dim second closure must not be scaled past the selected shader.
        expected = render(shader)
        dim = Hikari.Diffuse(color = (0.1, 0.1, 0.1))
        for factor in (2f0, xyz().x * 0 + 2)
            @test render(Hikari.Mix(dim, shader; factor)) ≈ expected rtol = 0.02
        end
        for factor in (-1f0, xyz().x * 0 - 1)
            @test render(Hikari.Mix(shader, dim; factor)) ≈ expected rtol = 0.02
        end
    end

    # A subsurface material's shading kernel holds the random walk, which traces, and the
    # BSDF sampling after it. With the walk outlined, Apple's GPU compiler crashed linking
    # that kernel ("Internal compiler error") on hardware and software traversal alike, so a
    # subsurface material could not render on an M5 at all.
    @testset "a subsurface material renders ($(hw ? "hardware" : "software") tracing)" for hw in (false, true)
        backend = Mantle.defaultbackend()
        scene = Hikari.Scene(; backend, hw_accel = hw)
        sphere = GeometryBasics.normal_mesh(GeometryBasics.Tesselation(GeometryBasics.Sphere(Point3f(0), 1f0), 32))
        ground = GeometryBasics.normal_mesh(GeometryBasics.Rect3f(Point3f(-4, -4, -1.05), Vec3f(8, 8, 0.01)))
        push!(scene, sphere, Hikari.Principled(base_color = (0.9, 0.6, 0.5), roughness = 0.4, subsurface_weight = 1,
                                               subsurface_radius = (1, 0.5, 0.25), subsurface_scale = 0.2))
        push!(scene, ground, Hikari.Diffuse(color = (0.6, 0.6, 0.6)))
        push!(scene, Hikari.PointLight(Point3f(0, -2, 4), RGBSpectrum(40f0)))
        Hikari.sync!(scene)
        film = Hikari.Film(backend, Hikari.Film(Point2f(32, 24)))
        camera = Hikari.PerspectiveCamera(Point3f(0, -4, 1), Point3f(0), film; fov = 40f0)
        vp = Hikari.VolPath(samples = 16, max_depth = 4, hw_accel = hw)
        vp(scene, film, camera)
        img = Array(film.framebuffer)
        close(vp)
        @test all(p -> isfinite(p.r) && isfinite(p.g) && isfinite(p.b), img)
        # The sphere fills the middle of the frame; lit, it is not black there.
        @test mean(p.r + p.g + p.b for p in img[9:16, 13:20]) > 0.05
    end

    # A sheet opaque where x > 0 and cut out elsewhere, between a light straight above
    # and the ground the camera looks at from under the sheet: half the ground is in
    # its shadow. Shadow rays asked for the sheet's alpha with the uv alone, so the
    # program read its position at the origin, where the sheet is cut out, and the
    # whole ground was lit.
    @testset "an alpha over position casts its shadow ($(hw ? "hardware" : "software") tracing)" for hw in (false, true)
        backend = Mantle.defaultbackend()
        scene = Hikari.Scene(; backend, hw_accel = hw)
        ground = GeometryBasics.normal_mesh(GeometryBasics.Rect3f(Point3f(-4, -4, -0.01), Vec3f(8, 8, 0.01)))
        sheet = GeometryBasics.normal_mesh(GeometryBasics.Rect3f(Point3f(-4, -4, 1), Vec3f(8, 8, 0.01)))
        push!(scene, ground, Hikari.Diffuse(color = (0.8, 0.8, 0.8)))
        push!(scene, sheet, Hikari.Principled(base_color = (0.8, 0.8, 0.8), roughness = 1,
                                              alpha = greater(xyz().x, 0.0)))
        push!(scene, Hikari.PointLight(Point3f(0, 0, 4), RGBSpectrum(60f0)))
        Hikari.sync!(scene)
        film = Hikari.Film(backend, Hikari.Film(Point2f(32, 24)))
        camera = Hikari.PerspectiveCamera(Point3f(0, -3, 0.5f0), Point3f(0, 0, 0), film; fov = 50f0)
        vp = Hikari.VolPath(samples = 16, max_depth = 2, hw_accel = hw)
        vp(scene, film, camera)
        img = Array(film.framebuffer)
        close(vp)
        brightness(cols) = mean(p.r + p.g + p.b for p in img[:, cols])
        left, right = brightness(1:16), brightness(17:32)
        @test max(left, right) > 3 * min(left, right)
    end

    # Hardware tracing shaded the first surface a ray hit whatever its alpha: the
    # closest-hit shaders had no alpha test, so a cut-out region was a surface whose
    # lobes all weighed nothing and rendered black, where the software trace went
    # on through it to the wall behind. Only shadow rays, which loop over their
    # hits, saw the cutout.
    @testset "an alpha cutout shows what is behind it, as in software tracing" begin
        function halves(hw)
            backend = Mantle.defaultbackend()
            scene = Hikari.Scene(; backend, hw_accel = hw)
            wall = GeometryBasics.normal_mesh(GeometryBasics.Rect3f(Point3f(-4, -4, -2), Vec3f(8, 8, 0.01)))
            sheet = GeometryBasics.normal_mesh(GeometryBasics.Rect3f(Point3f(-4, -4, 0), Vec3f(8, 8, 0.01)))
            push!(scene, wall, Hikari.Diffuse(color = (0.8, 0.8, 0.8)))
            push!(scene, sheet, Hikari.Principled(base_color = (0.02, 0.02, 0.02), roughness = 1,
                                                  alpha = greater(xyz().x, 0.0)))
            push!(scene, Hikari.PointLight(Point3f(0, 0, 1), RGBSpectrum(30f0)))
            Hikari.sync!(scene)
            film = Hikari.Film(backend, Hikari.Film(Point2f(32, 24)))
            camera = Hikari.PerspectiveCamera(Point3f(0, 0, 4), Point3f(0, 0, 0), film; fov = 40f0)
            vp = Hikari.VolPath(samples = 16, max_depth = 2, hw_accel = hw)
            vp(scene, film, camera)
            img = Array(film.framebuffer)
            close(vp)
            brightness(cols) = mean(p.r + p.g + p.b for p in img[:, cols])
            return brightness(1:16), brightness(17:32)
        end
        sw, hw = halves(false), halves(true)
        @test max(sw...) > 2 * min(sw...)
        @test hw[1] ≈ sw[1] rtol = 0.1
        @test hw[2] ≈ sw[2] rtol = 0.1
    end

    # A parameter changed is a program copied into the slot the material has, so the
    # store takes no new array and the scene keeps its recorded plans. The update threw
    # before: Raycore rebuilt the material field by field from the new graph, which no
    # `ShaderMaterial` constructor takes.
    @testset "a shader material is updated in place" begin
        backend = Mantle.defaultbackend()
        scene = Hikari.Scene(; backend)
        sphere = GeometryBasics.normal_mesh(GeometryBasics.Tesselation(GeometryBasics.Sphere(Point3f(0), 1f0), 16))
        handle = push!(scene, sphere, Hikari.Principled(base_color = (0.8, 0.3, 0.2), roughness = 0.4))
        push!(scene, Hikari.PointLight(Point3f(0, 0.5, 4), RGBSpectrum(40f0)))
        Hikari.sync!(scene)
        store = scene.materials
        arrays, layout = length(store.texture_gpu_arrays), store.layout

        rougher = Hikari.Principled(base_color = (0.8, 0.3, 0.2), roughness = 0.6)
        Hikari.update_material!(scene, handle.interface, rougher)
        @test (length(store.texture_gpu_arrays), store.layout) == (arrays, layout)
        material = store[(Raycore.@allowscalar scene.media_interfaces[handle.interface]).material]
        F = only(typeof(material.program).parameters)
        stored = filter(a -> eltype(a) === Hikari.NodeInstr{F}, store.texture_gpu_arrays)[material.program.idx]
        @test Array(stored) == Hikari.link(Hikari.compile_shader(Hikari.ShaderGraph(Hikari.shader_node(rougher)).closures), ())

        # Other node families are another material type: a slot of its own, which the
        # interface is pointed at (`test_material_type_change.jl`).
        before = (Raycore.@allowscalar scene.media_interfaces[handle.interface]).material
        Hikari.update_material!(scene, handle.interface, Hikari.Principled(roughness = noise(xyz()).fac))
        @test (Raycore.@allowscalar scene.media_interfaces[handle.interface]).material.type_idx != before.type_idx

        # A transparent shader's transparency program is updated in place with it.
        cutout(edge) = Hikari.Principled(base_color = (0.8, 0.3, 0.2), alpha = greater(xyz().x, edge))
        sheet = push!(scene, sphere, cutout(0.0))
        Hikari.sync!(scene)
        arrays, layout = length(store.texture_gpu_arrays), store.layout
        Hikari.update_material!(scene, sheet.interface, cutout(0.5))
        @test (length(store.texture_gpu_arrays), store.layout) == (arrays, layout)
        material = store[(Raycore.@allowscalar scene.media_interfaces[sheet.interface]).material]
        F = only(typeof(material.alpha).parameters)
        stored = filter(a -> eltype(a) === Hikari.NodeInstr{F}, store.texture_gpu_arrays)[material.alpha.idx]
        @test Array(stored) == Hikari.link(Hikari.compile_shader(Hikari.transparency_graph(Hikari.ShaderGraph(
            Hikari.shader_node(cutout(0.5)))).closures), ())
    end

    # An update compiled the graph again, and compiling stored every image it read: each update
    # of a shader reading an image took a new array, kept the one it replaced, and changed the
    # store's layout, which drops every plan recorded against it. The image goes where the old
    # one is, and the colour and the transparency programs read it from one array.
    @testset "an image a shader reads is updated where it is" begin
        backend = Mantle.defaultbackend()
        scene = Hikari.Scene(; backend)
        sphere = GeometryBasics.normal_mesh(GeometryBasics.Tesselation(GeometryBasics.Sphere(Point3f(0), 1f0), 16))
        image(c) = fill(RGBSpectrum(c...), 8, 8)
        shader(c, r) = (img = image_texture(image(c));
                        Hikari.Principled(base_color = img, roughness = r, alpha = img.x))
        handle = push!(scene, sphere, shader((0.9, 0.2, 0.1), 0.4))
        push!(scene, Hikari.PointLight(Point3f(0, 0.5, 4), RGBSpectrum(40f0)))
        Hikari.sync!(scene)
        store = scene.materials
        images() = filter(a -> eltype(a) === RGBSpectrum, store.texture_gpu_arrays)
        @test length(images()) == 1
        arrays, layout = length(store.texture_gpu_arrays), store.layout
        Hikari.update_material!(scene, handle.interface, shader((0.1, 0.3, 0.8), 0.6))
        @test (length(store.texture_gpu_arrays), store.layout) == (arrays, layout)
        @test Array(only(images())) == image((0.1, 0.3, 0.8))
    end
end

# #47's kintsugi bowl, its graph as the PR built it: Voronoi edges over a noise-warped position drive
# the gold seams' colour, metallic, roughness, the coat and a bump, a color ramp of noise the glaze.
@testset "the kintsugi bowl of #47 renders" begin
    p = xyz()
    warped = p + (noise(p; scale = 1.2, detail = 4).color - 0.5) * 0.7
    gold = 1 - smoothstep(0.0, 0.045, voronoi(warped; scale = 1.1, feature = :distance_to_edge).distance)
    glaze = color_ramp(noise(warped; scale = 7, detail = 6).fac, 0.0 => (0.02, 0.08, 0.38), 1.0 => (0.35, 0.55, 0.85))
    bowl = Hikari.Principled(base_color = mix(glaze, (1.0, 0.74, 0.32), gold), metallic = gold,
                             roughness = mix(0.3, 0.16, gold), coat_weight = 1 - gold, normal = bump(0.0018 * gold; distance = 1))
    backend = Mantle.defaultbackend()
    scene = Hikari.Scene(; backend)
    push!(scene, GeometryBasics.normal_mesh(GeometryBasics.Tesselation(GeometryBasics.Sphere(Point3f(-1.1, 0, 0.45), 0.45f0), 64)), bowl)
    push!(scene, Hikari.PointLight(Point3f(1, -3, 4), RGBSpectrum(40f0)))
    push!(scene, Hikari.AmbientLight(RGBSpectrum(0.3f0)))
    Hikari.sync!(scene)
    film = Hikari.Film(backend, Hikari.Film(Point2f(96, 96)))
    # Where the graph puts a seam in view: it is a function of position.
    camera = Hikari.PerspectiveCamera(Point3f(0, -2.2, 1.2), Point3f(-1.1, 0, 0.45), film; up = Vec3f(0, 0, 1), fov = 35f0)
    vp = Hikari.VolPath(samples = 16, max_depth = 6)
    vp(scene, film, camera)
    img = Array(film.framebuffer)
    close(vp)
    @test all(c -> isfinite(c.r + c.g + c.b), img)
    sphere = img[30:80, 25:72]
    @test count(c -> c.b > 1.5f0 * c.r && c.b > 0.05f0, sphere) > length(sphere) ÷ 2    # the glaze
    @test count(c -> c.r > 1.5f0 * c.b && c.r > 0.05f0, sphere) > 0                     # a gold seam
end
