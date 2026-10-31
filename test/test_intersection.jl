@testset "Ray-Bounds intersection" begin
    b = Trace.Bounds3(Point3f(1), Point3f(2))
    b_neg = Trace.Bounds3(Point3f(-2), Point3f(-1))
    r0 = Trace.Ray(o = Point3f(0), d = Vec3f(1, 0, 0))
    r1 = Trace.Ray(o = Point3f(0), d = Vec3f(1))
    ri = Trace.Ray(o = Point3f(1.5), d = Vec3f(1, 1, 0))

    r, t0, t1 = Trace.intersect(b, r1)
    @test r && t0 ≈ 1f0 && t1 ≈ 2f0
    r, t0, t1 = Trace.intersect(b, r0)
    @test !r && t0 ≈ 0f0 && t1 ≈ 0f0
    r, t0, t1 = Trace.intersect(b, ri)
    @test r && t0 ≈ 0f0 && t1 ≈ 0.5f0

    # Test intersection with precomputed direction reciprocal.
    inv_dir = 1f0 ./ r1.d
    dir_is_negative = Trace.is_dir_negative(r1.d)
    @test Trace.intersect_p(b, r1, inv_dir, dir_is_negative)
    @test !Trace.intersect_p(b_neg, r1, inv_dir, dir_is_negative)
end

@testset "Ray-Sphere insersection" begin
    # Sphere at the origin.
    core = Trace.ShapeCore(Trace.Transformation(), false)
    s = Trace.Sphere(core, 1f0, 360f0)

    r = Trace.Ray(o = Point3f(0, -2, 0), d = Vec3f(0, 1, 0))
    i, t, interaction = Trace.intersect(s, r, false)
    ip = Trace.intersect_p(s, r, false)
    @test i == ip
    @test t ≈ 1f0
    @test r(t) ≈ Point3f(0, -1, 0) # World intersection.
    @test interaction.core.p ≈ Point3f(0, -1, 0) # Object intersection.
    @test interaction.core.n ≈ Trace.Normal3f(0, -1, 0)
    @test norm(interaction.core.n) ≈ 1f0
    @test norm(interaction.shading.n) ≈ 1f0
    # Spawn new ray from intersection.
    spawn_direction = Vec3f(0, -1, 0)
    spawned_ray = Trace.spawn_ray(interaction, spawn_direction)
    @test spawned_ray.o ≈ Point3f(interaction.core.p)
    @test spawned_ray.d ≈ Vec3f(spawn_direction)
    i, t, interaction = Trace.intersect(s, spawned_ray, false)
    @test !i

    r = Trace.Ray(o = Point3f(0, 0, -2), d = Vec3f(0, 0, 1))
    i, t, interaction = Trace.intersect(s, r, false)
    ip = Trace.intersect_p(s, r, false)
    @test i == ip
    @test t ≈ 1f0
    @test r(t) ≈ Point3f(0, 0, -1) # World intersection.
    @test interaction.core.p ≈ Point3f(0, 0, -1) # Object intersection.
    @test interaction.core.n ≈ Trace.Normal3f(0, 0, -1)
    @test norm(interaction.core.n) ≈ 1f0
    @test norm(interaction.shading.n) ≈ 1f0

    # Test ray inside a sphere.
    r0 = Trace.Ray(o = Point3f(0), d = Vec3f(0, 1, 0))
    i, t, interaction = Trace.intersect(s, r0, false)
    @test i
    @test t ≈ 1f0
    @test r0(t) ≈ Point3f(0f0, 1f0, 0f0)
    @test interaction.core.n ≈ Trace.Normal3f(0, 1, 0)
    @test norm(interaction.core.n) ≈ 1f0
    @test norm(interaction.shading.n) ≈ 1f0

    # Test ray at the edge of the sphere.
    ray_at_edge = Trace.Ray(o = Point3f(0, -1, 0), d = Vec3f(0, -1, 0))
    i, t, interaction = Trace.intersect(s, ray_at_edge, false)
    @test i
    @test t ≈ 0f0
    @test ray_at_edge(t) ≈ Point3f(0, -1, 0)
    @test interaction.core.p ≈ Point3f(0, -1, 0)
    @test interaction.core.n ≈ Trace.Normal3f(0, -1, 0)

    # Translated sphere.
    core = Trace.ShapeCore(Trace.translate(Vec3f(0, 2, 0)), false)
    s = Trace.Sphere(core, 1f0, 360f0)
    r = Trace.Ray(o = Point3f(0, 0, 0), d = Vec3f(0, 1, 0))

    i, t, interaction = Trace.intersect(s, r, false)
    ip = Trace.intersect_p(s, r, false)
    @test i == ip
    @test t ≈ 1f0
    @test r(t) ≈ Point3f(0, 1, 0) # World intersection.
    @test interaction.core.p ≈ Point3f(0, 1, 0) # Object intesection.
    @test interaction.core.n ≈ Trace.Normal3f(0, -1, 0)
end

@testset "Test triangle" begin
    core = Trace.ShapeCore(Trace.translate(Vec3f(0, 0, 2)), false)
    triangles = Trace.create_triangle_mesh(
        core,
        1, UInt32[1, 2, 3],
        3, [Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(1, 1, 0)],
        [Trace.Normal3f(0, 0, -1), Trace.Normal3f(0, 0, -1), Trace.Normal3f(0, 0, -1)],
    )

    tv = Trace.vertices(triangles[1])
    a = norm(tv[1] - tv[2])^2 * 0.5f0
    @test Trace.area(triangles[1]) ≈ a

    target_wb = Trace.Bounds3(Point3f(0, 0, 2), Point3f(1, 1, 2))
    target_ob = Trace.Bounds3(Point3f(0, 0, 0), Point3f(1, 1, 0))
    @test Trace.object_bound(triangles[1]) ≈ target_ob
    @test Trace.world_bound(triangles[1]) ≈ target_wb

    # Test ray intersection.
    ray = Trace.Ray(o = Point3f(0, 0, -2), d = Vec3f(0, 0, 1))
    intersects_p = Trace.intersect_p(triangles[1], ray)
    intersects, t_hit, interaction = Trace.intersect(triangles[1], ray)
    @test intersects_p == intersects == true
    @test t_hit ≈ 4f0
    @test ray(t_hit) ≈ interaction.core.p ≈ Point3f(0, 0, 2)
    @test interaction.uv ≈ Point2f(0)
    @test interaction.core.n ≈ Trace.Normal3f(0, 0, -1)
    @test interaction.core.wo ≈ -ray.d
    # Test ray intersection (lower-left corner).
    ray = Trace.Ray(o = Point3f(1, 0.5, 0), d = Vec3f(0, 0, 1))
    intersects_p = Trace.intersect_p(triangles[1], ray)
    intersects, t_hit, interaction = Trace.intersect(triangles[1], ray)
    @test intersects_p == intersects == true
    @test t_hit ≈ 2f0
    @test ray(t_hit) ≈ interaction.core.p ≈ Point3f(1, 0.5, 2)
    @test interaction.uv ≈ Point2f(1, 0.5)
    @test interaction.core.n ≈ Trace.Normal3f(0, 0, -1)
    @test interaction.core.wo ≈ -ray.d
end

@testset "BVH" begin
    primitives = Trace.Primitive[]
    for i in 0:3:21
        core = Trace.ShapeCore(Trace.translate(Vec3f(i, i, 0)), false)
        sphere = Trace.Sphere(core, 1f0, 360f0)
        push!(primitives, Trace.GeometricPrimitive(sphere))
    end

    bvh = Trace.BVHAccel(primitives[1:4])
    bvh2 = Trace.BVHAccel(Trace.Primitive[primitives[5:end]..., bvh])
    @test Trace.world_bound(bvh) ≈ Trace.Bounds3(Point3f(-1f0), Point3f(10f0, 10f0, 1f0))
    @test Trace.world_bound(bvh2) ≈ Trace.Bounds3(Point3f(-1f0), Point3f(22f0, 22f0, 1f0))

    ray1 = Trace.Ray(o = Point3f(-2f0, 0f0, 0f0), d = Vec3f(1f0, 0f0, 0f0))
    ray2 = Trace.Ray(o = Point3f(0f0, 18f0, 0f0), d = Vec3f(1f0, 0f0, 0f0))

    intersects, interaction = Trace.intersect!(bvh2, ray1)
    @test intersects
    @test ray1.t_max ≈ 1f0
    @test ray1(ray1.t_max) ≈ Point3f(-1f0, 0f0, 0f0)
    @test interaction.core.p ≈ Point3f(-1f0, 0f0, 0f0)

    intersects, interaction = Trace.intersect!(bvh2, ray2)
    @test intersects
    @test ray2.t_max ≈ 17f0
    @test ray2(ray2.t_max) ≈ Point3f(17f0, 18f0, 0f0)
    @test interaction.core.p ≈ Point3f(17f0, 18f0, 0f0)
end

@testset "Test BVH with spheres in a single row" begin
    primitives = Trace.Primitive[]

    core = Trace.ShapeCore(Trace.Transformation(), false)
    sphere = Trace.Sphere(core, 1f0, 360f0)
    push!(primitives, Trace.GeometricPrimitive(sphere))

    core = Trace.ShapeCore(Trace.translate(Vec3f(0, 0, 4)), false)
    sphere = Trace.Sphere(core, 2f0, 360f0)
    push!(primitives, Trace.GeometricPrimitive(sphere))

    core = Trace.ShapeCore(Trace.translate(Vec3f(0, 0, 11)), false)
    sphere = Trace.Sphere(core, 4f0, 360f0)
    push!(primitives, Trace.GeometricPrimitive(sphere))

    bvh = Trace.BVHAccel(primitives)
    @test Trace.world_bound(bvh) ≈ Trace.Bounds3(
        Point3f(-4, -4, -1), Point3f(4, 4, 15),
    )

    ray = Trace.Ray(o = Point3f(0, 0, -2), d = Vec3f(0, 0, 1))
    intersects, interaction = Trace.intersect!(bvh, ray)
    @test intersects
    @test ray.t_max ≈ 1f0
    @test ray(ray.t_max) ≈ interaction.core.p

    ray = Trace.Ray(o = Point3f(1.5, 0, -2), d = Vec3f(0, 0, 1))
    intersects, interaction = Trace.intersect!(bvh, ray)
    @test intersects
    @test 2f0 < ray.t_max < 6f0
    @test ray(ray.t_max) ≈ interaction.core.p

    ray = Trace.Ray(o = Point3f(3, 0, -2), d = Vec3f(0, 0, 1))
    intersects, interaction = Trace.intersect!(bvh, ray)
    @test intersects
    @test 7f0 < ray.t_max < 15f0
    @test ray(ray.t_max) ≈ interaction.core.p
end
