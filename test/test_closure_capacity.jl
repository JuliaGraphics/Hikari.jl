using Test, Hikari, Mantle, Raycore, KernelAbstractions, GeometryBasics
using Hikari.TextureNodes
using GeometryBasics: Point2f, Point3f, Vec3f, Sphere, Tesselation, normal_mesh

capacity_program(material) = Hikari.closure_program(Hikari.compiled(Hikari.surface_material(material).program))[1]
capacity_hint(material) = Hikari.closure_capacity_hint(first(capacity_program(material)))

function capacity_queues(m)
    nodes = Hikari.bsdf_nodes(typeof(Hikari.surface_material(m)))
    M = Hikari.bsdf_kernel_nodes(nodes, Mantle.VulkanAPI())
    q = Hikari.MaterialQueues(Int32[], Int32[], Bool[], Int32(1))
    inputs = Hikari.ShaderInputs(Float32[], Int32[], Int32(1))
    return Hikari.shade_queues(Hikari.ClosureQueue{M}(q, 1, inputs, 0))
end

@testset "closure storage bounds depend on program values" begin
    @test capacity_program(Hikari.Glass(roughness = 0.2)) ==
          capacity_program(Hikari.Principled(base_color = (1, 1, 1), roughness = 0.2, transmission_weight = 1))
    @test capacity_hint(Hikari.Glass()) == 1
    @test capacity_hint(Hikari.Glass(roughness = noise(xyz()).fac)) == 1
    @test capacity_hint(Hikari.Glass(thin_film_thickness = 400)) == 1
    for distribution in (:multi_ggx, :ggx, :beckmann), metallic in (0,1)
        @test capacity_hint(Hikari.Principled(; metallic, distribution)) == 2
        @test capacity_hint(Hikari.Principled(; metallic, distribution,
            base_color = noise(xyz()).color, roughness = noise(xyz()).fac)) == 2
    end
    for m in (Hikari.Principled(metallic = 0.35),
              Hikari.Principled(metallic = checker(xyz()).fac),
              Hikari.Principled(coat_weight = 0.4), Hikari.Principled(sheen_weight = 0.25),
              Hikari.Principled(coat_weight = checker(xyz()).fac),
              Hikari.Principled(sheen_weight = checker(xyz()).fac))
        @test capacity_hint(m) == 5
    end
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
    @test map(typeof, capacity_queues(Hikari.Glass())) === map(typeof, capacity_queues(Hikari.Principled()))
    @test map(q -> typeof(q).parameters[1].parameters[1], capacity_queues(Hikari.Glass())) == (1, 2, 5, 16)
    for m in (Hikari.Diffuse(), Hikari.Principled(subsurface_weight = 0.5))
        @test length(capacity_queues(m)) == 1
    end
    store = Raycore.MultiTypeSet(KernelAbstractions.CPU())
    key = push!(store, Hikari.surface_material(Hikari.Glass(roughness = 0.2)))
    old = store[key]
    layout = store.layout
    arrays = length(store.texture_gpu_arrays)
    for (m, hint) in ((Hikari.Principled(), 2), (Hikari.Principled(metallic = 1), 2),
                      (Hikari.Principled(metallic = 0.35), 5),
                      (Hikari.Principled(coat_weight = 0.4), 5),
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

function capacity_fixture(first_material; hw_accel = false, single = false)
    backend = Mantle.defaultbackend()
    scene = Hikari.Scene(; backend, hw_accel)
    first_handle = push!(scene, normal_mesh(Tesselation(Sphere(Point3f(-0.55, 0, 0), 0.6f0), 24)), first_material)
    # All capacity categories share the same material slot in this scene.
    single || push!(scene, normal_mesh(Tesselation(Sphere(Point3f(0.55, 0.1, 0), 0.6f0), 24)),
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

@testset "storage capacities and in-place transitions keep recorded plans (single=$single)" for single in (false,true)
    material = Hikari.Glass(roughness = 0.2)
    f = capacity_fixture(material; single)
    try
        capacity_draw(f)
        plans = f.vp.state.plans
        partition = f.vp.state.shader_partition
        partition_allocations = length(f.vp.state.per_material_memory.owned)
        @test partition isa Hikari.ShaderCapacityPartition
        @test length(partition.items) == (single ? 1 : 4 * f.vp.state.material_queues.capacity)
        @test length(partition.sizes) == 4
        arrays = length(f.scene.materials.texture_gpu_arrays)
        layout = f.scene.materials.layout
        for (name,m) in (("opaque", Hikari.Principled(roughness = 0.3)),
                        ("metal", Hikari.Principled(roughness = 0.3, metallic = 1)),
                        ("mixed metal", Hikari.Principled(roughness = 0.3, metallic = 0.35)),
                        ("layered", Hikari.Principled(roughness = 0.3, metallic = 0.35, coat_weight = 0.4, sheen_weight = 0.25)),
                        ("partial transmission", Hikari.Principled(transmission_weight = 0.5, roughness = 0.2)),
                        ("glass",material))
            @testset "$name" begin
                Hikari.update_material!(f.scene, f.first_handle.interface, m)
                capacity_draw(f)
                actual = capacity_image(f) # Keep the first post-update frame.
                capacity_draw(f)
                @test capacity_image(f) ≈ actual rtol = 5f-5 atol = 2f-5
                @test f.vp.state.plans === plans
                @test f.vp.state.shader_partition === partition
                @test length(f.vp.state.per_material_memory.owned) == partition_allocations
                @test length(f.scene.materials.texture_gpu_arrays) == arrays
                @test f.scene.materials.layout == layout
                expected = capacity_fixture(m; single)
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
        end
    finally
        capacity_release(f)
    end
end

@testset "two-record storage preserves microfacet distributions" for distribution in (:ggx, :beckmann)
    material = Hikari.Principled(; distribution, roughness = 0.3, base_color = (0.5,0.4,0.3))
    actual = capacity_fixture(material)
    expected = capacity_fixture(material)
    try
        capacity_draw(actual)
        capacity_reference_draw(expected)
        image, reference = capacity_image(actual), capacity_image(expected)
        @test all(isfinite,image)
        @test maximum(image) > 0f0
        @test image ≈ reference rtol = 5f-5 atol = 2f-5
    finally
        capacity_release(actual)
        capacity_release(expected)
    end
end

# Local arrays below sixteen records render 3-8x slower on NVIDIA (first seen on an
# RTX 3070) with identical pixels, so the pattern is pinned for every driver.
@testset "array storage starts at sixteen records" begin
    records(m) = Hikari.bsdf_kernel_nodes(Hikari.bsdf_nodes(typeof(Hikari.surface_material(m))), Mantle.VulkanAPI()).parameters[1]
    principled = Hikari.Principled()
    layered = Hikari.Principled(metallic = 0.35, coat_weight = 0.4, sheen_weight = 0.25)
    @test records(Hikari.Diffuse()) == 1
    @test records(Hikari.Add(Hikari.Diffuse(), Hikari.Diffuse())) == 4
    @test records(principled) == 16
    @test records(Hikari.Glass()) == 16
    @test records(Hikari.Glass(thin_wall = true)) == 16
    @test records(Hikari.Principled(subsurface_weight = 0.5)) == 16
    @test records(Hikari.Mix(principled, Hikari.Diffuse(); factor = 0.5)) == 16
    @test records(Hikari.Mix(principled, layered; factor = 0.5)) == 16
    @test records(Hikari.Mix(Hikari.Mix(principled, layered; factor = 0.5), principled; factor = 0.5)) == 32
    # Only one Principled node has bounded kernels; everything else dispatches once.
    for m in (Hikari.Mix(principled, Hikari.Diffuse(); factor = 0.5), Hikari.Mix(principled, layered; factor = 0.5),
              Hikari.Principled(subsurface_weight = 0.5))
        @test length(capacity_queues(m)) == 1
        @test capacity_hint(m) == 0
    end
    @test length(capacity_queues(principled)) == 4
end
