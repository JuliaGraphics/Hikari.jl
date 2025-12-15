

# Basic geometric types
gen_point3f() = Point3f(1.0f0, 2.0f0, 3.0f0)
gen_vec3f() = Vec3f(0.0f0, 0.0f0, 1.0f0)
gen_normal3f() = Hikari.Normal3f(0.0f0, 0.0f0, 1.0f0)

# Rays
gen_ray() = Hikari.Ray(o=Point3f(0.0f0, 0.0f0, -2.0f0), d=Vec3f(0.0f0, 0.0f0, 1.0f0))
gen_ray_differentials() = Hikari.RayDifferentials(o=Point3f(0.0f0, 0.0f0, -2.0f0), d=Vec3f(0.0f0, 0.0f0, 1.0f0))

# Triangle
function gen_triangle()
    v1 = Point3f(0.0f0, 0.0f0, 0.0f0)
    v2 = Point3f(1.0f0, 0.0f0, 0.0f0)
    v3 = Point3f(0.0f0, 1.0f0, 0.0f0)
    n1 = Hikari.Normal3f(0.0f0, 0.0f0, 1.0f0)
    uv1 = Point2f(0.0f0, 0.0f0)
    uv2 = Point2f(1.0f0, 0.0f0)
    uv3 = Point2f(0.0f0, 1.0f0)
    Raycore.Triangle(
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

    tri_mesh = Hikari.TriangleMesh(mesh)

    # Create a simple matte material with textures
    kd = Hikari.Texture(Hikari.RGBSpectrum(0.5f0))
    sigma = Hikari.Texture(0.0f0)
    material = Hikari.MatteMaterial(kd, sigma)

    # Create MaterialScene using build_material_scene helper
    mesh_material_pairs = Tuple{Raycore.TriangleMesh, typeof(material)}[(tri_mesh, material)]
    Hikari.build_material_scene(mesh_material_pairs)
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
    n = Hikari.Normal3f(0, 0, 1)

    it = Hikari.Interaction(p, time, wo, n)

    @test it isa Hikari.Interaction
    @test it.p == p
    @test it.time == time
    @test it.wo == wo
    @test it.n == n
end

@testset "Type Stability: ShadingInteraction" begin
    n = Hikari.Normal3f(0, 0, 1)
    dpdu = Vec3f(1, 0, 0)
    dpdv = Vec3f(0, 1, 0)
    dndu = Hikari.Normal3f(0)
    dndv = Hikari.Normal3f(0)

    si = Hikari.ShadingInteraction(n, dpdu, dpdv, dndu, dndv)

    @test si isa Hikari.ShadingInteraction
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
    dndu = Hikari.Normal3f(0)
    dndv = Hikari.Normal3f(0)

    si = Hikari.SurfaceInteraction(p, time, wo, uv, dpdu, dpdv, dndu, dndv, false)

    @test si isa Hikari.SurfaceInteraction
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
    @test_opt Hikari.triangle_to_surface_interaction(triangle, ray, bary)
end

@testset "Type Stability: MaterialScene intersect!" begin
    ms = gen_material_scene()
    ray = gen_ray()
    @test_opt Hikari.intersect!(ms, ray)
end

# ==================== shade_material Test Generators ====================

# Generate a SurfaceInteraction for shading tests
function gen_surface_interaction()
    p = Point3f(0.25f0, 0.25f0, 0.0f0)
    time = 0f0
    wo = Vec3f(0f0, 0f0, -1f0)
    uv = Point2f(0.25f0, 0.25f0)
    dpdu = Vec3f(1f0, 0f0, 0f0)
    dpdv = Vec3f(0f0, 1f0, 0f0)
    dndu = Hikari.Normal3f(0f0)
    dndv = Hikari.Normal3f(0f0)
    Hikari.SurfaceInteraction(p, time, wo, uv, dpdu, dpdv, dndu, dndv, false)
end

# Generate a PointLight for shading tests
function gen_point_light()
    Hikari.PointLight(Point3f(0f0, 1f0, 0f0), Hikari.RGBSpectrum(1f0))
end

# Generate a Scene with a single UberMaterial type
function gen_scene_single_material()
    # Create matte material
    kd = Hikari.Texture(Hikari.RGBSpectrum(0.5f0))
    sigma = Hikari.Texture(0.0f0)
    material = Hikari.MatteMaterial(kd, sigma)

    # Create triangle mesh
    points = Point3f[(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    faces = TriangleFace{Int}[(1, 2, 3)]
    mesh = GeometryBasics.Mesh(points, faces)
    tri_mesh = Raycore.TriangleMesh(mesh)

    # Create MaterialScene
    ms = Hikari.MaterialScene([(tri_mesh, material)])

    # Create light and scene
    light = gen_point_light()
    Hikari.Scene([light], ms)
end

# Generate a Scene with multiple UberMaterial variants (matte, mirror, glass)
function gen_scene_multi_uber_materials()
    material_matte = Hikari.MatteMaterial(
        Hikari.Texture(Hikari.RGBSpectrum(0.5f0)),
        Hikari.Texture(0.0f0)
    )
    material_mirror = Hikari.MirrorMaterial(
        Hikari.Texture(Hikari.RGBSpectrum(1f0))
    )
    material_glass = Hikari.GlassMaterial(
        Hikari.Texture(Hikari.RGBSpectrum(1f0)),
        Hikari.Texture(Hikari.RGBSpectrum(1f0)),
        Hikari.Texture(0f0),
        Hikari.Texture(0f0),
        Hikari.Texture(1.5f0),
        true
    )

    # Create triangle mesh
    points = Point3f[(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    faces = TriangleFace{Int}[(1, 2, 3)]
    mesh = GeometryBasics.Mesh(points, faces)
    tri_mesh = Raycore.TriangleMesh(mesh)

    # Create MaterialScene with multiple materials
    ms = Hikari.MaterialScene([
        (tri_mesh, material_matte),
        (tri_mesh, material_mirror),
        (tri_mesh, material_glass)
    ])

    light = gen_point_light()
    Hikari.Scene([light], ms)
end

# Generate a Scene with mixed material types (UberMaterial + CloudVolume)
function gen_scene_mixed_material_types()
    # UberMaterial (matte)
    material_matte = Hikari.MatteMaterial(
        Hikari.Texture(Hikari.RGBSpectrum(0.5f0)),
        Hikari.Texture(0.0f0)
    )

    # CloudVolume material
    cloud_data = zeros(Float32, 10, 10, 10)
    cloud_data[4:6, 4:6, 4:6] .= 0.001f0
    cloud_material = Hikari.CloudVolume(cloud_data, (100.0, 100.0, 100.0))

    # Create triangle mesh
    points = Point3f[(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    faces = TriangleFace{Int}[(1, 2, 3)]
    mesh = GeometryBasics.Mesh(points, faces)
    tri_mesh = Raycore.TriangleMesh(mesh)

    # Create MaterialScene with both material types
    ms = Hikari.MaterialScene([
        (tri_mesh, material_matte),
        (tri_mesh, cloud_material)
    ])

    light = gen_point_light()
    Hikari.Scene([light], ms)
end

# ==================== shade_material Type Stability Tests ====================

@testset "Type Stability: shade_material" begin
    ray = gen_ray_differentials()
    si = gen_surface_interaction()
    beta = Hikari.RGBSpectrum(1f0)

    @testset "Single UberMaterial type" begin
        scene = gen_scene_single_material()
        materials = scene.aggregate.materials

        @test_opt Hikari.shade_material(materials, UInt8(1), UInt32(1), ray, si, scene, beta)
    end

    @testset "Multiple UberMaterial variants" begin
        scene = gen_scene_multi_uber_materials()
        materials = scene.aggregate.materials

        # Test each material variant
        @test_opt Hikari.shade_material(materials, UInt8(1), UInt32(1), ray, si, scene, beta)
        @test_opt Hikari.shade_material(materials, UInt8(1), UInt32(2), ray, si, scene, beta)
        @test_opt Hikari.shade_material(materials, UInt8(1), UInt32(3), ray, si, scene, beta)
    end

    @testset "Mixed material types (UberMaterial + CloudVolume)" begin
        scene = gen_scene_mixed_material_types()
        materials = scene.aggregate.materials

        # Test UberMaterial (type 1)
        @test_opt Hikari.shade_material(materials, UInt8(1), UInt32(1), ray, si, scene, beta)

        # Test CloudVolume (type 2)
        @test_opt Hikari.shade_material(materials, UInt8(2), UInt32(1), ray, si, scene, beta)
    end
end

@testset "Type Stability: shade (UberMaterial)" begin
    ray = gen_ray_differentials()
    si = gen_surface_interaction()
    beta = Hikari.RGBSpectrum(1f0)
    scene = gen_scene_single_material()

    # Get the material directly
    material = scene.aggregate.materials[1][1]

    @test_opt Hikari.shade(material, ray, si, scene, beta)
end
