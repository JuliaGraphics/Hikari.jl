# What a material type's shading kernel takes of the scene's materials (`shade_set`).
#
# The kernel took the whole set. A new material type, or a shader of other node families,
# changed the set's type, so every shading kernel in the scene compiled again (35–120 s each on
# an RTX 4000 Ada, most of it NVIDIA's pipeline compile), and each grew by every other shader's
# program: a texture read compiled the interpreter of every node-family set in the scene. A
# voronoi node added to one material cost ten minutes. A kernel now takes a view of the slots it
# reads, so its type, and the kernel compiled for it, stay as they were.
using Test, Hikari, Raycore, Mantle, GeometryBasics
using Hikari.TextureNodes
using GeometryBasics: Point2f, Point3f, Vec3f, Rect3f, Sphere, Tesselation, normal_mesh

ss_key(scene, h) = (Raycore.@allowscalar scene.media_interfaces[h.interface]).material
ss_type(scene, h) = typeof(scene.materials[ss_key(scene, h)])
ss_view(scene, T) = typeof(Hikari.shade_set(scene.materials.static, T))
"""The element types of the texture slots a view of `scene`'s materials holds."""
ss_textures(scene, V) = [eltype(eltype(typeof(scene.materials.static.textures).parameters[i])) for i in V.parameters[2]]

@testset "shade_set" begin
    scene = Hikari.Scene(; backend = Mantle.defaultbackend())
    sphere = normal_mesh(Tesselation(Sphere(Point3f(0), 1.0f0), 8))
    plain = push!(scene, sphere, Hikari.Principled(base_color = (0.8, 0.2, 0.1), roughness = 0.3))
    # A pbrt model uses a shader value program, just like Principled.
    pbrt = push!(scene, sphere, Hikari.Conductor(roughness = noise(xyz()).fac))
    P, C = ss_type(scene, plain), ss_type(scene, pbrt)
    before = (ss_view(scene, P), ss_view(scene, C))

    @testset "other materials leave a kernel's view" begin
        push!(scene, sphere, Hikari.Principled(base_color = voronoi(xyz()).color))
        push!(
            scene, sphere, Hikari.Principled(
                base_color = noise(xyz()).color, emission_strength = 1,
                emission_color = (1, 1, 1)
            )
        )
        push!(scene, sphere, Hikari.Glass(ior = 1.5))
        @test length(scene.materials.static.data) == 4
        @test (ss_view(scene, P), ss_view(scene, C)) == before
    end

    @testset "a view reads its own shader program and the texture programs" begin
        F = Hikari.NodeFeature
        texture_program(E) = E <: Hikari.NodeInstr && E.parameters[1] & F.TEXTURE != 0
        shader_program(E) = E <: Hikari.NodeInstr && E.parameters[1] & F.TEXTURE == 0
        @test count(shader_program, ss_textures(scene, ss_view(scene, P))) == 1
        @test count(texture_program, ss_textures(scene, ss_view(scene, P))) == 0
        @test count(shader_program, ss_textures(scene, ss_view(scene, C))) == 1
        @test count(texture_program, ss_textures(scene, ss_view(scene, C))) == 0
    end

    # A shadow ray may pass a cut-out material anywhere, so every kernel reads its transparency.
    @testset "a cut-out material joins every view" begin
        cut = push!(scene, sphere, Hikari.Principled(base_color = (0.2, 0.2, 0.2), alpha = greater(xyz().x, 0.0)))
        Cut = ss_type(scene, cut)
        @test Hikari.may_cut(Cut) && !Hikari.may_cut(P)
        @test ss_key(scene, cut).type_idx in ss_view(scene, P).parameters[1]
        @test ss_key(scene, cut).type_idx in ss_view(scene, C).parameters[1]
    end

    # A view had no `lookup_tables` method and fell back to none, read as 1: the specular layer's
    # albedo, so every Principled rendered with a black diffuse base.
    @testset "a view reads the lookup tables" begin
        static = scene.materials.static
        @test (Raycore.@allowscalar Hikari.lookup_tables(Hikari.shade_set(static, P))) ===
            (Raycore.@allowscalar Hikari.lookup_tables(static))
    end

    @testset "a pbrt mix keeps a graph view" begin
        h = push!(scene, sphere, Hikari.Mix(Hikari.Gold(), Hikari.CoatedDiffuse(reflectance = (0.4, 0.4, 0.4)); factor = 0.3f0))
        M = ss_type(scene, h)
        @test M <: Hikari.ShaderMaterial
        @test all(T -> !(T <: Hikari.MixMaterial), map(eltype, scene.materials.static.data))
        @test Hikari.shade_set(scene.materials.static, M) isa Raycore.SetView
    end
end

# The trace kernel took the whole material set and a queue tuple typed per material type, so a
# new type, every node edit that changes a shader's families, compiled it again: the biggest
# kernel there is, 5 s in Lava and as much in the driver on an RTX 4000 Ada, for each edit. It
# takes the queues' flat store and the cut-out types now (`MaterialQueues`, `cut_set`).
@testset "a new material type leaves the trace kernel as it was" begin
    backend = Mantle.defaultbackend()
    dev = Hikari.mantle_device(backend)
    scene = Hikari.Scene(; backend)
    sphere(x) = normal_mesh(Tesselation(Sphere(Point3f(x, 0, 0.5), 0.4f0), 16))
    push!(scene, sphere(0), Hikari.Principled(base_color = (0.8, 0.2, 0.1), roughness = 0.3))
    push!(scene, Hikari.PointLight(Point3f(0, -2, 3), Hikari.RGBSpectrum(30.0f0)))
    Hikari.sync!(scene)
    film = Hikari.Film(backend, Hikari.Film(Point2f(16, 16)))
    camera = Hikari.PerspectiveCamera(Point3f(0, -3, 1), Point3f(0, 0, 0.5), film; up = Vec3f(0, 0, 1), fov = 40.0f0)
    vp = Hikari.VolPath(samples = 1, max_depth = 3)
    stage(name) = [
        (d.kernel, Mantle.devicetype(dev, d.args)) for pl in Hikari.allplans(vp.state.plans)
            for pp in pl.passes if pp.pass.name == name for d in pp.pass.dispatches
    ]
    vp(scene, film, camera)
    trace = stage("trace")
    @test !isempty(trace)
    push!(scene, sphere(1), Hikari.Principled(base_color = voronoi(xyz()).color))
    push!(scene, sphere(-1), Hikari.Glass(ior = 1.5))
    Hikari.sync!(scene)
    vp(scene, film, camera)
    @test length(scene.materials.static.data) == 2
    @test stage("trace") == trace
    # Glass is a Principled preset: all three shaders share the same BSDF kernel.
    @test length(unique(stage("shade"))) == 1
    # A cut-out material is read by the trace, so it does change it.
    push!(scene, sphere(2), Hikari.Principled(base_color = (0.5, 0.5, 0.5), alpha = greater(xyz().x, 2.0)))
    Hikari.sync!(scene)
    vp(scene, film, camera)
    @test stage("trace") != trace
    close(vp)
end

# A node edit that changed a shader's node families compiled its whole shading kernel again: ~27k
# instructions of which its program is ~3k, 13 s on an RTX 4000 Ada. The program runs in a values
# kernel of its own now, and the closure kernel of the shader's BSDF nodes, which does the rest,
# names no program: the edit compiles a values kernel alone.
@testset "a node edit compiles the values kernel alone" begin
    backend = Mantle.defaultbackend()
    dev = Hikari.mantle_device(backend)
    scene = Hikari.Scene(; backend)
    h = push!(
        scene, normal_mesh(Tesselation(Sphere(Point3f(0, 0, 0.5), 0.4f0), 16)),
        Hikari.Principled(base_color = wave(xyz()).color, roughness = 0.3)
    )
    push!(scene, Hikari.PointLight(Point3f(0, -2, 3), Hikari.RGBSpectrum(30.0f0)))
    Hikari.sync!(scene)
    film = Hikari.Film(backend, Hikari.Film(Point2f(16, 16)))
    camera = Hikari.PerspectiveCamera(Point3f(0, -3, 1), Point3f(0, 0, 0.5), film; up = Vec3f(0, 0, 1), fov = 40.0f0)
    vp = Hikari.VolPath(samples = 1, max_depth = 3)
    stage(name) = unique(
        [
            (d.kernel, Mantle.devicetype(dev, d.args)) for pl in Hikari.allplans(vp.state.plans)
                for pp in pl.passes if pp.pass.name == name for d in pp.pass.dispatches
        ]
    )
    vp(scene, film, camera)
    closures, values = stage("shade"), stage("shader-values")
    @test length(closures) == 1 && length(values) == 1
    Hikari.update_material!(scene, h, Hikari.Principled(base_color = voronoi(xyz()).color, roughness = 0.3))
    vp(scene, film, camera)
    @test stage("shade") == closures
    @test length(stage("shader-values")) == 2 && values[1] in stage("shader-values")
    # Images introduce a new array type in the scene. An opaque BSDF kernel
    # consumes gathered inputs and must not start depending on that array.
    push!(
        scene, normal_mesh(Tesselation(Sphere(Point3f(1, 0, 0.5), 0.4f0), 8)),
        Hikari.Principled(base_color = fill(Hikari.RGBSpectrum(0.4f0), 4, 4))
    )
    Hikari.sync!(scene)
    vp(scene, film, camera)
    @test stage("shade") == closures
    close(vp)
end

@testset "simple native graph edits reuse the shading kernel" begin
    backend = Mantle.defaultbackend()
    dev = Hikari.mantle_device(backend)
    scene = Hikari.Scene(; backend)
    checker(x) = Hikari.CheckerboardTexture(5.0f0, 5.0f0, 0.0f0, 0.0f0, x, 0.3f0)
    material(x) = Hikari.Conductor(roughness = x, reflectance = (0.3, 0.5, 0.7))
    h = push!(scene, normal_mesh(Tesselation(Sphere(Point3f(0), 0.7f0), 16)), material(checker(0.1f0)))
    push!(scene, Hikari.PointLight(Point3f(0, -2, 3), Hikari.RGBSpectrum(30.0f0)))
    Hikari.sync!(scene)
    film = Hikari.Film(backend, Hikari.Film(Point2f(16, 16)))
    camera = Hikari.PerspectiveCamera(Point3f(0, -3, 1), Point3f(0), film; fov = 40.0f0)
    vp = Hikari.VolPath(samples = 1, max_depth = 3)
    stage() = unique(
        [
            (d.kernel, Mantle.devicetype(dev, d.args)) for pl in Hikari.allplans(vp.state.plans)
                for pp in pl.passes if pp.pass.name == "shade" for d in pp.pass.dispatches
        ]
    )
    try
        vp(scene, film, camera)
        original = stage()
        @test length(original) == 1
        key = ss_key(scene, h)
        for roughness in (checker(0.2f0), xyz().x * 0.05f0 + 0.2f0, checker(0.15f0))
            Hikari.update_material!(scene, h, material(roughness))
            vp(scene, film, camera)
            @test ss_key(scene, h) == key
            @test stage() == original
        end
    finally
        close(vp)
        Hikari.free!(film)
    end
end

@testset "constant graphs skip values without changing the BSDF kernel" begin
    backend = Mantle.defaultbackend()
    dev = Hikari.mantle_device(backend)
    scene = Hikari.Scene(; backend)
    sphere(x) = normal_mesh(Tesselation(Sphere(Point3f(x, 0, 0.5), 0.4f0), 16))
    graph(m) = Hikari.ShaderMaterial(Hikari.shader_node(m))
    h = push!(scene, sphere(0), graph(Hikari.Diffuse(color = (0.7, 0.3, 0.2))))
    push!(scene, sphere(1), graph(Hikari.Diffuse(color = noise(xyz()).color)))
    push!(scene, Hikari.PointLight(Point3f(0, -2, 3), Hikari.RGBSpectrum(30.0f0)))
    Hikari.sync!(scene)
    film = Hikari.Film(backend, Hikari.Film(Point2f(16, 16)))
    camera = Hikari.PerspectiveCamera(Point3f(0, -3, 1), Point3f(0, 0, 0.5), film; fov = 40.0f0)
    vp = Hikari.VolPath(samples = 1, max_depth = 3)
    stage(name) = unique(
        [
            (d.kernel, Mantle.devicetype(dev, d.args)) for pl in Hikari.allplans(vp.state.plans)
                for pp in pl.passes if pp.pass.name == name for d in pp.pass.dispatches
        ]
    )
    try
        vp(scene, film, camera)
        shade, key = stage("shade"), ss_key(scene, h)
        width = Hikari.shader_input_layout(scene.materials).width
        @test length(shade) == 1
        @test length(stage("shader-values")) == 1
        Hikari.update_material!(scene, h, graph(Hikari.Diffuse(roughness = xyz().x)))
        vp(scene, film, camera)
        @test stage("shade") == shade
        @test length(stage("shader-values")) == 2
        @test ss_key(scene, h) == key
        @test Hikari.shader_input_layout(scene.materials).width == width
        Hikari.update_material!(scene, h, graph(Hikari.Diffuse(color = (0.7, 0.3, 0.2))))
        vp(scene, film, camera)
        @test stage("shade") == shade
        @test length(stage("shader-values")) == 1
        @test ss_key(scene, h) == key
    finally
        close(vp)
        Hikari.free!(film)
    end
end

# What the missing lookup tables did to an image: a Principled of no specular layer is its diffuse
# base, which rendered black. It renders as the Diffuse of its colour.
@testset "a Principled of no specular renders as its Diffuse" begin
    backend = Mantle.defaultbackend()
    function sphere_mean(material)
        scene = Hikari.Scene(; backend)
        push!(scene, normal_mesh(Tesselation(Sphere(Point3f(0, 0, 0.5), 0.5f0), 48)), material)
        push!(scene, Hikari.PointLight(Point3f(0, -2, 3), Hikari.RGBSpectrum(30.0f0)))
        push!(scene, Hikari.AmbientLight(Hikari.RGBSpectrum(0.2f0)))
        Hikari.sync!(scene)
        film = Hikari.Film(backend, Hikari.Film(Point2f(64, 64)))
        camera = Hikari.PerspectiveCamera(Point3f(0, -3, 0.5f0), Point3f(0, 0, 0.5f0), film; up = Vec3f(0, 0, 1), fov = 30.0f0)
        vp = Hikari.VolPath(samples = 64, max_depth = 4)
        vp(scene, film, camera)
        img = Array(film.framebuffer)
        close(vp)
        # The sphere fills the middle of the image.
        middle = img[25:40, 25:40]
        return sum(p -> p.r + p.g + p.b, middle) / length(middle)
    end
    principled = sphere_mean(Hikari.Principled(base_color = (0.6, 0.6, 0.6), specular_ior_level = 0))
    diffuse = sphere_mean(Hikari.Diffuse(color = (0.6, 0.6, 0.6)))
    @test diffuse > 0.1
    @test principled ≈ diffuse rtol = 0.03
end
