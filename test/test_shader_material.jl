using Test
using Hikari
using Hikari.TextureNodes
using Hikari: TextureFilterContext, RGBSpectrum, device_param, Lobe, Fresnel, NodeOp
using GeometryBasics
using GeometryBasics: Point2f, Point3f, Vec3f, Vec2f, GLTriangleFace, QuadFace
using Hikari: SVector
using Colors: RGB
using KernelAbstractions: CPU
using LinearAlgebra
using Raycore
using Mantle
using Statistics: mean

static_textures(store) =
    Raycore.StaticMultiTypeSet((), Tuple(T[a for a in store.texture_gpu_arrays if typeof(a) === T] for T in store.texture_order))

function closures(shader; p = Point3f(0), n = Vec3f(0, 0, 1), wo = Vec3f(0, 0, 1),
                  dpdx = Vec3f(1f-3, 0, 0), dpdy = Vec3f(0, 1f-3, 0), tangent_sign = 1f0)
    store = Raycore.MultiTypeSet(CPU())
    graph = Hikari.ShaderGraph(shader)
    handle = device_param(store, graph)
    sc = Hikari.ShaderContext(TextureFilterContext(Point2f(0), p, n), wo, dpdx, dpdy, Vec3f(1, 0, 0), n, tangent_sign,
                              Vec3f(0), Point2f(0))
    cl = Hikari.eval_shader_handle(static_textures(store), handle, sc, Val(Hikari.closure_capacity(graph)))
    return collect(cl.closures[1:cl.count])
end

"""The closures of `shader` bound to `mesh`, at face `face` and barycentrics `bary`, the uv there moving by `duv` per pixel."""
function closures_on(mesh, shader; face = 1, bary = (1 / 3, 1 / 3, 1 / 3), n = Vec3f(0, 0, 1), wo = Vec3f(0, 0, 1),
                     uv = Point2f(0), duv = (0f0, 0f0, 0f0, 0f0), dpdx = Vec3f(1f-3, 0, 0), dpdy = Vec3f(0, 1f-3, 0))
    store = Raycore.MultiTypeSet(CPU())
    _, material = Hikari.bind_mesh(mesh, Hikari.ShaderMaterial(shader))
    handle = device_param(store, material.program)
    tfc = TextureFilterContext(uv, duv..., UInt32(face), SVector{3, Float32}(bary), Point3f(0), n)
    sc = Hikari.ShaderContext(tfc, wo, dpdx, dpdy, Vec3f(1, 0, 0), n, 1f0, Vec3f(0), Point2f(0))
    cl = Hikari.eval_shader_handle(static_textures(store), handle, sc, Val(Hikari.closure_capacity(material.program)))
    return collect(cl.closures[1:cl.count])
end

function program(shader)
    store = Raycore.MultiTypeSet(CPU())
    handle = device_param(store, Hikari.ShaderGraph(shader))
    return static_textures(store).textures[handle.slot][handle.idx]
end

@testset "Shader materials" begin
    @testset "a diffuse BSDF is one diffuse closure" begin
        c = only(closures(diffuse_bsdf(color = (0.2, 0.4, 0.6))))
        @test c.lobe == Lobe.DIFFUSE
        @test c.weight ≈ Vec3f(0.2, 0.4, 0.6)
        @test c.n == Vec3f(0, 0, 1)
    end

    @testset "mix shader weighs its two shaders by its factor" begin
        red, blue = diffuse_bsdf(color = (1, 0, 0)), diffuse_bsdf(color = (0, 0, 1))
        @test [c.weight for c in closures(mix_shader(0.25, red, blue))] ≈ [Vec3f(0.75, 0, 0), Vec3f(0, 0, 0.25)]
        @test [c.weight for c in closures(mix_shader(xyz().x, red, blue); p = Point3f(0.3, 0, 0))] ≈
              [Vec3f(0.7, 0, 0), Vec3f(0, 0, 0.3)]
        @test [c.weight for c in closures(add_shader(red, blue))] ≈ [Vec3f(1, 0, 0), Vec3f(0, 0, 1)]
    end

    @testset "a metallic principled BSDF is one F82 tint closure" begin
        c = only(closures(principled_bsdf(base_color = (0.9, 0.6, 0.2), metallic = 1, roughness = 0.4)))
        @test (c.lobe, c.fresnel) == (Lobe.MICROFACET, Fresnel.F82_TINT)
        @test c.c1 ≈ Vec3f(0.9, 0.6, 0.2)
        @test c.alpha_x ≈ c.alpha_y ≈ 0.16f0
    end

    @testset "the specular layer of a principled BSDF takes its reflection from the diffuse" begin
        specular, diffuse = closures(principled_bsdf(base_color = (0.8, 0.8, 0.8), roughness = 0))
        @test (specular.lobe, specular.fresnel) == (Lobe.MICROFACET, Fresnel.SCHLICK)
        @test specular.c1 ≈ Vec3f(0.04)
        @test diffuse.lobe == Lobe.DIFFUSE
        @test diffuse.weight ≈ Vec3f(0.8 * 0.96) rtol = 1e-3
    end

    @testset "a coat takes its reflection from the layers under it" begin
        coat, specular, diffuse = closures(principled_bsdf(base_color = (0.8, 0.8, 0.8), roughness = 0, coat_weight = 1,
                                                           coat_roughness = 0))
        @test (coat.fresnel, coat.alpha_x) == (Fresnel.DIELECTRIC, 0f0)
        @test specular.weight ≈ Vec3f(0.96) rtol = 1e-3
        @test diffuse.weight ≈ Vec3f(0.8 * 0.96 * 0.96) rtol = 1e-3
    end

    @testset "subsurface is a random walk closure with the radius scaled" begin
        cs = closures(principled_bsdf(base_color = (0.9, 0.6, 0.5), roughness = 0.4, subsurface_weight = 1,
                                      subsurface_radius = (1, 0.5, 0.25), subsurface_scale = 0.1))
        sss = only(filter(c -> c.lobe == Lobe.SUBSURFACE, cs))
        @test sss.c1 ≈ Vec3f(0.9, 0.6, 0.5)
        @test sss.c2 ≈ Vec3f(0.1, 0.05, 0.025)
        @test sss.alpha_x ≈ 0.16f0
        @test !any(c -> c.lobe == Lobe.DIFFUSE, cs)
    end

    @testset "the material type says what a shader uses" begin
        flags(m::Hikari.ShaderMaterial{P, T, S, E}) where {P, T, S, E} = (T, S, E)
        @test flags(Hikari.ShaderMaterial(principled_bsdf())) == (false, false, false)
        @test flags(Hikari.ShaderMaterial(principled_bsdf(alpha = 0.5))) == (true, false, false)
        @test flags(Hikari.ShaderMaterial(principled_bsdf(subsurface_weight = 1))) == (false, true, false)
        @test flags(Hikari.ShaderMaterial(principled_bsdf(subsurface_weight = 1, thin_wall = true))) == (false, false, false)
        @test flags(Hikari.ShaderMaterial(principled_bsdf(emission_strength = xyz().x))) == (false, false, true)
        @test flags(Hikari.ShaderMaterial(mix_shader(0.5, diffuse_bsdf(), transparent_bsdf()))) == (true, false, false)
    end

    @testset "a shader holds room for as many closures as its graph can produce" begin
        capacity(shader) = Hikari.closure_capacity(Hikari.ShaderGraph(shader))
        @test capacity(principled_bsdf()) == 2
        @test capacity(principled_bsdf(metallic = 1)) == 1
        @test capacity(principled_bsdf(metallic = 1, coat_weight = 1)) == 2
        @test capacity(principled_bsdf(coat_weight = xyz().x, transmission_weight = 1, thin_wall = true)) == 5
        @test capacity(mix_shader(0.5, diffuse_bsdf(), transparent_bsdf())) == 1
        @test capacity(add_shader(principled_bsdf(sheen_weight = 1, subsurface_weight = 1, transmission_weight = 1),
                                  principled_bsdf(coat_weight = 1))) == 8
        @test length(closures(principled_bsdf(metallic = 0.5, coat_weight = 1))) == 4
    end

    @testset "a device shader reads only programs of its own type" begin
        store = Raycore.MultiTypeSet(CPU())
        plain = Hikari.to_device_material(store, Hikari.ShaderMaterial(diffuse_bsdf(color = (0.2, 0.4, 0.6))))
        noisy = Hikari.to_device_material(store, Hikari.ShaderMaterial(diffuse_bsdf(color = noise(xyz()).color)))
        @test typeof(plain.program) == Hikari.ShaderProgram{0x00000000}
        @test typeof(noisy.program) == Hikari.ShaderProgram{Hikari.NodeFeature.NOISE}
        sc = Hikari.ShaderContext(TextureFilterContext(Point2f(0), Point3f(0), Vec3f(0, 0, 1)), Vec3f(0, 0, 1),
                                  Vec3f(0), Vec3f(0), Vec3f(1, 0, 0), Vec3f(0, 0, 1), 1f0, Vec3f(0), Point2f(0))
        @test Hikari.eval_shader_handle(static_textures(store), plain, sc).closures[1].weight ≈ Vec3f(0.2, 0.4, 0.6)
    end

    @testset "a closure tilted off the smooth normal is shadowed as in Cycles" begin
        ns, tilted = Vec3f(0, 0, 1), normalize(Vec3f(1, 0, 1))
        @test Hikari._bump_shadowing(true, ns, ns, normalize(Vec3f(1, 0, 1))) == 1f0
        @test Hikari._bump_shadowing(true, tilted, ns, ns) == 1f0
        @test Hikari._bump_shadowing(true, tilted, ns, normalize(Vec3f(1, 0, 1))) ≈ 1 / (1 + (sqrt(1.125) - 1) / 2)
        @test Hikari._bump_shadowing(true, tilted, ns, normalize(Vec3f(-1, 0, 0.2))) == 0f0
        @test Hikari._bump_shadowing(false, tilted, ns, normalize(Vec3f(-1, 0, 0.2))) == 0f0
        @test Hikari._bump_shadowing(false, tilted, ns, normalize(Vec3f(1, 0, 1))) == 1f0
    end

    @testset "a tangent space normal map tilts the normal along the tangent frame" begin
        mapped(colour; kw...) = only(closures(diffuse_bsdf(normal = normal_map(colour; kw...)))).n
        @test mapped((0.5, 0.5, 1.0)) ≈ Vec3f(0, 0, 1)
        @test mapped((0.75, 0.5, 1.0)) ≈ normalize(Vec3f(0.5, 0, 1))
        @test mapped((0.5, 0.75, 1.0)) ≈ normalize(Vec3f(0, 0.5, 1))
        @test only(closures(diffuse_bsdf(normal = normal_map((0.5, 0.75, 1.0))); tangent_sign = -1f0)).n ≈
              normalize(Vec3f(0, -0.5, 1))
        @test mapped((0.5, 0.75, 1.0); invert_green = true) ≈ normalize(Vec3f(0, -0.5, 1))
        @test mapped((0.75, 0.5, 0.5); strength = 0.5) ≈ normalize(Vec3f(0.25, 0, 0.5))
        @test mapped((0.75, 0.5, 1.0); strength = 0) ≈ Vec3f(0, 0, 1)
    end

    @testset "a world space normal map is the normal its colour encodes" begin
        @test only(closures(diffuse_bsdf(normal = normal_map((1.0, 0.5, 0.5); space = :world)))).n ≈ Vec3f(1, 0, 0)
        @test only(closures(diffuse_bsdf(normal = normal_map((0.5, 0.5, 0.0); space = :blender_world)))).n ≈
              Vec3f(0, 0, 1)
    end

    @testset "a mesh attribute is interpolated over the hit's triangle" begin
        tri = GeometryBasics.Mesh(Point3f[(0, 0, 0), (1, 0, 0), (0, 1, 0)], [GLTriangleFace(1, 2, 3)];
                                  paint = Vec3f[(0.2, 0, 0), (0, 0.4, 0), (0, 0, 0.6)],
                                  color = [RGB{Float32}(1, 0, 0), RGB{Float32}(0.5, 0.5, 0.5), RGB{Float32}(0, 0, 1)])
        weight(shader; kw...) = only(closures_on(tri, shader; kw...)).weight
        @test weight(diffuse_bsdf(color = attribute(:paint).color); bary = (0.5, 0.25, 0.25)) ≈ Vec3f(0.1, 0.1, 0.15)
        @test weight(diffuse_bsdf(color = attribute(:paint).fac); bary = (1, 0, 0)) ≈ Vec3f(0.2 / 3)
        @test weight(diffuse_bsdf(color = attribute(:color).color); bary = (0, 1, 0)) ≈ Vec3f(0.21404114)
        @test isempty(closures_on(tri, diffuse_bsdf(color = attribute(:missing).color)))
    end

    @testset "a quad's second triangle reads its own corners" begin
        quad = GeometryBasics.Mesh(Point3f[(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [QuadFace(1, 2, 3, 4)];
                                   paint = Vec3f[(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)])
        corners = [only(closures_on(quad, diffuse_bsdf(color = attribute(:paint).color); face = 2, bary = b)).weight
                   for b in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
        @test Set(corners) == Set([Vec3f(1, 0, 0), Vec3f(0, 0, 1), Vec3f(1, 1, 1)])
    end

    @testset "generated and object coordinates are the mesh's own positions" begin
        tri = GeometryBasics.Mesh(Point3f[(1, 2, 3), (3, 2, 3), (1, 6, 4)], [GLTriangleFace(1, 2, 3)])
        coordinate(which) = only(closures_on(tri, diffuse_bsdf(color = getproperty(texture_coordinate(), which));
                                             bary = (0, 1, 0))).weight
        @test coordinate(:object) ≈ Vec3f(3, 2, 3)
        @test coordinate(:generated) ≈ Vec3f(1, 0, 0)
    end

    @testset "reflection coordinates mirror the view about the normal" begin
        reflected(wo) = only(closures(diffuse_bsdf(color = texture_coordinate().reflection * 0.5 + 0.5); wo)).weight
        @test reflected(Vec3f(0, 0, 1)) ≈ Vec3f(0.5, 0.5, 1)
        @test reflected(normalize(Vec3f(1, 0, 1))) ≈ Vec3f(0.5 - 0.5 / sqrt(2), 0.5, 0.5 + 0.5 / sqrt(2))
    end

    @testset "a bump on a linear attribute tilts the normal by its slope" begin
        tri = GeometryBasics.Mesh(Point3f[(0, 0, 0), (1, 0, 0), (0, 1, 0)], [GLTriangleFace(1, 2, 3)];
                                  uv = Vec2f[(0, 0), (1, 0), (0, 1)], height = Float32[0, 1, 0])
        c = only(closures_on(tri, diffuse_bsdf(normal = bump(attribute(:height).fac; distance = 0.5));
                             uv = Point2f(0.25, 0.25), duv = (1f-3, 0f0, 0f0, 1f-3)))
        @test c.n ≈ normalize(Vec3f(-0.5, 0, 1))
    end

    @testset "a bump on a linear height tilts the normal by its slope" begin
        c = only(closures(diffuse_bsdf(normal = bump(xyz().x; distance = 0.5))))
        @test c.n ≈ normalize(Vec3f(-0.5, 0, 1))
        c = only(closures(diffuse_bsdf(normal = bump(xyz().x; distance = 0.5, invert = true))))
        @test c.n ≈ normalize(Vec3f(0.5, 0, 1))
        c = only(closures(diffuse_bsdf(normal = bump(xyz().x; distance = 0.5, strength = 0))))
        @test c.n ≈ Vec3f(0, 0, 1)
    end

    @testset "a bump evaluates only what reads the position three times" begin
        fixed = noise(constant((0.3, 0.2, 0.1)))
        ops = [ins.op for ins in program(diffuse_bsdf(normal = bump(xyz().x * fixed)))]
        @test count(==(NodeOp.NOISE), ops) == 1
        @test count(==(NodeOp.MATH), ops) == 3
    end

    @testset "a principled BSDF runs only the inputs its weights read" begin
        @test [ins.op for ins in program(principled_bsdf())] ==
              [NodeOp.NORMAL, NodeOp.PRINCIPLED, NodeOp.ARGS, NodeOp.ARGS, NodeOp.ARGS, NodeOp.RESULT]
        @test count(==(NodeOp.POSITION), [ins.op for ins in program(principled_bsdf(anisotropic = 0.5))]) == 1
        @test count(==(NodeOp.CONST), [ins.op for ins in program(principled_bsdf(subsurface_weight = 1))]) == 1
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
        @test render(Hikari.ShaderMaterial(diffuse_bsdf(color = colour))) ≈ render(Hikari.Diffuse(Kd = colour)) rtol = 0.02
    end
end
