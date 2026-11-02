using Test, Hikari, Mantle, Raycore, KernelAbstractions, GeometryBasics
using Hikari.TextureNodes
using GeometryBasics: Point2f, Point3f, Vec3f, Sphere, Tesselation, normal_mesh

capacity_program(material) = Hikari.closure_program(Hikari.compiled(Hikari.surface_material(material).program))[1]
capacity_hint(material) = Hikari.closure_capacity_hint(first(capacity_program(material)))

@testset "closure storage bounds depend on program values" begin
    @test capacity_program(Hikari.Glass(roughness = 0.2)) ==
          capacity_program(Hikari.Principled(base_color = (1, 1, 1), roughness = 0.2, transmission_weight = 1))
    @test capacity_hint(Hikari.Glass()) == 1
    @test capacity_hint(Hikari.Glass(roughness = noise(xyz()).fac)) == 1
    @test capacity_hint(Hikari.Glass(thin_film_thickness = 400)) == 1
    @test capacity_hint(Hikari.Principled()) == 5
    @test capacity_hint(Hikari.Principled(metallic = checker(xyz()).fac, coat_weight = 1, sheen_weight = 1)) == 5
    for m in (Hikari.Principled(transmission_weight = 0.5),
              Hikari.Principled(transmission_weight = checker(xyz()).fac),
              Hikari.Principled(transmission_weight = 1, coat_weight = 1),
              Hikari.Principled(transmission_weight = 1, sheen_weight = 1),
              Hikari.Principled(transmission_weight = 1, metallic = 0.2),
              Hikari.Principled(subsurface_weight = 0.5),
              Hikari.Glass(thin_wall = true), Hikari.Diffuse(),
              Hikari.Mix(Hikari.Glass(), Hikari.Diffuse(); factor = 0.5))
        @test capacity_hint(m) == 0
    end
    # Only the reserved high bits change: output-width flags retain their meaning.
    for m in (Hikari.Glass(), Hikari.Principled())
        compiled = Hikari.compiled(Hikari.surface_material(m).program)
        first_bsdf = findfirst(ins -> ins.op == Hikari.NodeOp.PRINCIPLED, compiled.instructions)
        @test first(capacity_program(m)).outwide & UInt8((1 << Hikari.NODE_OUTS) - 1) ==
              compiled.instructions[first_bsdf].outwide
    end
    function queues(m)
        nodes = Hikari.bsdf_nodes(typeof(Hikari.surface_material(m)))
        M = Hikari.bsdf_kernel_nodes(nodes, Mantle.VulkanAPI())
        q = Hikari.MaterialQueues(Int32[], Int32[], Bool[], Int32(1))
        inputs = Hikari.ShaderInputs(Float32[], Int32[], Int32(1))
        return Hikari.shade_queues(Hikari.ClosureQueue{M}(q, 1, inputs, 0))
    end
    @test map(typeof, queues(Hikari.Glass())) === map(typeof, queues(Hikari.Principled()))
    @test map(q -> typeof(q).parameters[1].parameters[1], queues(Hikari.Glass())) == (1, 5, 8)
    for m in (Hikari.Diffuse(), Hikari.Principled(subsurface_weight = 0.5),
              Hikari.Add(Hikari.Principled(), Hikari.Principled()))
        @test length(queues(m)) == 1
    end
    store = Raycore.MultiTypeSet(KernelAbstractions.CPU())
    key = push!(store, Hikari.surface_material(Hikari.Glass(roughness = 0.2)))
    old = store[key]
    layout = store.layout
    arrays = length(store.texture_gpu_arrays)
    for (m, hint) in ((Hikari.Principled(), 5),
                      (Hikari.Principled(transmission_weight = 0.5), 0),
                      (Hikari.Glass(roughness = 0.2), 1))
        @test Raycore.store!(store, key, Hikari.surface_material(m)) == key
        @test typeof(store[key]) === typeof(old)
        @test store.layout == layout
        @test length(store.texture_gpu_arrays) == arrays
        instructions = filter(a -> eltype(a) === Hikari.NodeInstr{Hikari.NodeFeature.CLOSURES}, store.texture_gpu_arrays)
        @test Hikari.closure_capacity_hint(first(Array(instructions[store[key].closures]))) == hint
    end
end

function capacity_fixture(first_material; hw_accel = false)
    backend = Mantle.defaultbackend()
    scene = Hikari.Scene(; backend, hw_accel)
    first_handle = push!(scene, normal_mesh(Tesselation(Sphere(Point3f(-0.55, 0, 0), 0.6f0), 24)), first_material)
    # All three capacity categories share the same material slot in this scene.
    push!(scene, normal_mesh(Tesselation(Sphere(Point3f(0.55, 0.1, 0), 0.6f0), 24)),
          Hikari.Principled(transmission_weight = 0.5, roughness = 0.2))
    push!(scene, Hikari.PointLight(Point3f(-2, -2, 3), Hikari.RGBSpectrum(40f0)))
    Hikari.sync!(scene)
    film = Hikari.Film(backend, Hikari.Film(Point2f(128, 128)))
    camera = Hikari.PerspectiveCamera(Point3f(0, -4, 1), Point3f(0), film; fov = 40f0)
    vp = Hikari.VolPath(; samples = 8, max_depth = 4)
    return (; scene, film, camera, vp, first_handle, dev = Hikari.mantle_device(backend))
end
capacity_draw(f) = (f.vp(f.scene, f.film, f.camera); Mantle.waitidle(f.dev); nothing)
capacity_image(f) = [Float32(c) for p in Array(f.film.framebuffer) for c in (p.r, p.g, p.b)]
capacity_release(f) = (close(f.vp); Hikari.free!(f.film); nothing)

# Independent differential reference: the retained typed formulas used before
# shared storage. Only this fixture records those kernels.
function capacity_reference_draw(f)
    @eval Hikari.bsdf_kernel_nodes(::Type{M}, ::Mantle.Backend) where {M <: Tuple} = M
    method = which(Hikari.bsdf_kernel_nodes, Tuple{Type{Tuple{}}, Mantle.VulkanAPI})
    try
        return Base.invokelatest(capacity_draw, f)
    finally
        Base.delete_method(method)
    end
end

@testset "mixed storage capacities and in-place transitions keep recorded plans" begin
    material = Hikari.Glass(roughness = 0.2)
    f = capacity_fixture(material)
    try
        capacity_draw(f)
        plans = f.vp.state.plans
        arrays = length(f.scene.materials.texture_gpu_arrays)
        layout = f.scene.materials.layout
        for m in (Hikari.Principled(roughness = 0.3, metallic = 0.35, coat_weight = 0.4, sheen_weight = 0.25),
                  Hikari.Principled(transmission_weight = 0.5, roughness = 0.2), material)
            Hikari.update_material!(f.scene, f.first_handle.interface, m)
            capacity_draw(f)
            actual = capacity_image(f) # Keep the first post-update frame.
            capacity_draw(f)
            @test capacity_image(f) ≈ actual rtol = 5f-5 atol = 2f-5
            @test f.vp.state.plans === plans
            @test length(f.scene.materials.texture_gpu_arrays) == arrays
            @test f.scene.materials.layout == layout
            expected = capacity_fixture(m)
            try
                capacity_reference_draw(expected)
                reference = capacity_image(expected)
                @test all(isfinite, actual)
                @test maximum(actual) > 0f0
                @test actual ≈ reference rtol = 5f-5 atol = 2f-5
            finally
                capacity_release(expected)
            end
        end
    finally
        capacity_release(f)
    end
end
