

# Basic geometric types
gen_point3f() = Point3f(1.0f0, 2.0f0, 3.0f0)
gen_vec3f() = Vec3f(0.0f0, 0.0f0, 1.0f0)
gen_normal3f() = Trace.Normal3f(0.0f0, 0.0f0, 1.0f0)

# Rays
gen_ray() = Trace.Ray(o=Point3f(0.0f0, 0.0f0, -2.0f0), d=Vec3f(0.0f0, 0.0f0, 1.0f0))
gen_ray_differentials() = Trace.RayDifferentials(o=Point3f(0.0f0, 0.0f0, -2.0f0), d=Vec3f(0.0f0, 0.0f0, 1.0f0))

# Triangle
function gen_triangle()
    v1 = Point3f(0.0f0, 0.0f0, 0.0f0)
    v2 = Point3f(1.0f0, 0.0f0, 0.0f0)
    v3 = Point3f(0.0f0, 1.0f0, 0.0f0)
    n1 = Trace.Normal3f(0.0f0, 0.0f0, 1.0f0)
    uv1 = Point2f(0.0f0, 0.0f0)
    uv2 = Point2f(1.0f0, 0.0f0)
    uv3 = Point2f(0.0f0, 1.0f0)
    RayCaster.Triangle(
        SVector(v1, v2, v3),
        SVector(n1, n1, n1),
        SVector(Vec3f(NaN), Vec3f(NaN), Vec3f(NaN)),
        SVector(uv1, uv2, uv3),
        UInt32(1)
    )
end

# MaterialScene with simple geometry
function gen_material_scene()
    # Create a simple triangle mesh
    points = Point3f[(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    faces = TriangleFace{Int}[(1, 2, 3)]
    mesh = GeometryBasics.Mesh(points, faces)

    tri_mesh = Trace.TriangleMesh(mesh)

    # Create a simple matte material with textures
    kd = Trace.Texture(Trace.RGBSpectrum(0.5f0))
    sigma = Trace.Texture(0.0f0)
    material = Trace.MatteMaterial(kd, sigma)

    # Create MaterialScene using build_material_scene helper
    mesh_material_pairs = Tuple{RayCaster.TriangleMesh, typeof(material)}[(tri_mesh, material)]
    Trace.build_material_scene(mesh_material_pairs)
end

# Barycentric coordinates for triangle intersection
gen_bary_coords() = Point3f(0.33f0, 0.33f0, 0.34f0)

# ==================== Custom Test Macros ====================

"""
    @test_opt_alloc expr

Combined macro that tests both type stability (via @test_opt) and zero allocations.
This is equivalent to:
    @test_opt expr
    @test @allocated(expr) == 0
"""
macro test_opt_alloc(expr)
    return esc(quote
        $expr # warmup
        JET.@test_opt $expr
        @test @allocated($expr) == 0
    end)
end

# ==================== Type Stability Tests ====================

@testset "Type Stability: Interaction" begin
    p = Point3f(0, 0, 1)
    time = 0f0
    wo = Vec3f(0, 0, -1)
    n = Trace.Normal3f(0, 0, 1)

    it = Trace.Interaction(p, time, wo, n)

    @test it isa Trace.Interaction
    @test it.p == p
    @test it.time == time
    @test it.wo == wo
    @test it.n == n
end

@testset "Type Stability: ShadingInteraction" begin
    n = Trace.Normal3f(0, 0, 1)
    dpdu = Vec3f(1, 0, 0)
    dpdv = Vec3f(0, 1, 0)
    dndu = Trace.Normal3f(0)
    dndv = Trace.Normal3f(0)

    si = Trace.ShadingInteraction(n, dpdu, dpdv, dndu, dndv)

    @test si isa Trace.ShadingInteraction
    @test si.n == n
    @test si.∂p∂u == dpdu
    @test si.∂p∂v == dpdv
    @test si.∂n∂u == dndu
    @test si.∂n∂v == dndv
end

@testset "Type Stability: SurfaceInteraction" begin
    # Create a basic surface interaction using the constructor
    p = Point3f(0, 0, 1)
    time = 0f0
    wo = Vec3f(0, 0, -1)
    uv = Point2f(0.5, 0.5)
    dpdu = Vec3f(1, 0, 0)
    dpdv = Vec3f(0, 1, 0)
    dndu = Trace.Normal3f(0)
    dndv = Trace.Normal3f(0)

    si = Trace.SurfaceInteraction(p, time, wo, uv, dpdu, dpdv, dndu, dndv, false)

    @test si isa Trace.SurfaceInteraction
    @test si.core.p == p
    @test si.core.time == time
    @test si.core.wo == wo
    @test si.∂p∂u == dpdu
    @test si.∂p∂v == dpdv
    @test si.uv == uv
end

@testset "Type Stability: triangle_to_surface_interaction" begin
    triangle = gen_triangle()
    ray = gen_ray()
    bary = gen_bary_coords()

    # Test type stability
    @test_opt Trace.triangle_to_surface_interaction(triangle, ray, bary)
end

@testset "Type Stability: MaterialScene intersect!" begin
    ms = gen_material_scene()
    ray = gen_ray()
    @test_opt Trace.intersect!(ms, ray)
end
