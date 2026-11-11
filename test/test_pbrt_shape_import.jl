# Shapes from a pbrt file are where pbrt puts them, with pbrt's normals.
#
# Five ways the importer's meshes differed from pbrt-v4's, none of them visible in a
# scene of untransformed meshes without normals:
#
# * A `trianglemesh` had its points transformed and its `N` not: rotated, its shading
#   normals still pointed where the file wrote them, along the surface.
# * Normals were transformed by the transform itself; pbrt uses the inverse transpose,
#   the one that keeps them normal to a surface scaled unevenly.
# * `ReverseOrientation` rebuilt the mesh from its points alone, losing its normals and
#   uvs; pbrt negates the normals.
# * A transform that mirrors turns the winding over; pbrt turns the normal it makes from
#   the winding back (`transformSwapsHandedness`), Hikari did not.
# * `loopsubdiv` stopped at the last refinement, with normals averaged over the faces.
#   pbrt's `LoopSubdivide` pushes the vertices to the limit surface and gives them its
#   normal; and a boundary vertex took any two neighbours on the boundary, out of a
#   `Set`, where pbrt takes the two along boundary edges. The reflection of a light in
#   a subdivided gold dome sat a pixel off pbrt's, 2.8 % of the scene's energy.

using Test, Hikari, Mantle, GeometryBasics, Colors
using LinearAlgebra: dot, cross, normalize, norm
isdefined(@__MODULE__, :render_scene) || include(joinpath(@__DIR__, "pbrt", "suite.jl"))

shapes(text) = let pb = Hikari.parse_pbrt_string(text)
    [Hikari.build_pbrt_shape(s, pb) for s in pb.shapes]
end
mesh_of(text) = only(shapes(text))
winding_normal(mesh, face = 1) = let f = GeometryBasics.faces(mesh)[face], p = GeometryBasics.coordinates(mesh)
    normalize(cross(Vec3f(p[f[2]] - p[f[1]]), Vec3f(p[f[3]] - p[f[1]])))
end

@testset "pbrt shapes are placed as pbrt places them" begin
    @testset "a trianglemesh's normals turn with it" begin
        m = mesh_of("""
            AttributeBegin
            Rotate 90 1 0 0
            Shape "trianglemesh" "point3 P" [ -1 -1 0  1 -1 0  1 1 0 ] "integer indices" [ 0 1 2 ]
                "normal N" [ 0 0 1  0 0 1  0 0 1 ]
            AttributeEnd""")
        @test all(n -> Vec3f(n) ≈ Vec3f(0, -1, 0), m.normal)
    end

    @testset "normals stay normal to a surface scaled unevenly" begin
        # The plane x = y, its normal (1, -1, 0), stretched along x.
        m = mesh_of("""
            AttributeBegin
            Scale 4 1 1
            Shape "trianglemesh" "point3 P" [ 0 0 0  1 1 0  0 0 1 ] "integer indices" [ 0 1 2 ]
                "normal N" [ 1 -1 0  1 -1 0  1 -1 0 ]
            AttributeEnd""")
        p = GeometryBasics.coordinates(m)
        for n in m.normal
            @test abs(dot(Vec3f(n), normalize(Vec3f(p[2] - p[1])))) < 1f-6
            @test abs(dot(Vec3f(n), normalize(Vec3f(p[3] - p[1])))) < 1f-6
        end
    end

    @testset "ReverseOrientation negates the normals and keeps the uvs" begin
        text(reverse) = """
            AttributeBegin
            $(reverse ? "ReverseOrientation" : "")
            Shape "trianglemesh" "point3 P" [ 0 0 0  1 0 0  0 1 0 ] "integer indices" [ 0 1 2 ]
                "normal N" [ 0 0 1  0 0 1  0 0 1 ] "point2 uv" [ 0 0  1 0  0 1 ]
            AttributeEnd"""
        m, r = mesh_of(text(false)), mesh_of(text(true))
        @test [Vec3f(n) for n in r.normal] == [-Vec3f(n) for n in m.normal]
        @test r.uv == m.uv
        @test winding_normal(r) ≈ -winding_normal(m)
    end

    @testset "a mirroring transform keeps pbrt's orientation" begin
        text(transform) = """
            AttributeBegin
            $transform
            Shape "trianglemesh" "point3 P" [ 0 0 0  1 0 0  0 1 0 ] "integer indices" [ 0 1 2 ]
            AttributeEnd"""
        # pbrt's normal of the mirrored triangle is the unmirrored one's: +z.
        @test winding_normal(mesh_of(text(""))) ≈ Vec3f(0, 0, 1)
        @test winding_normal(mesh_of(text("Scale -1 1 1"))) ≈ Vec3f(0, 0, 1)
    end

    @testset "loopsubdiv moves a boundary vertex by its boundary edges only" begin
        # A flat 2×2 grid. The vertex in the middle of a side has its two neighbours along the
        # side symmetric about it, at every level and in the limit, so it stays where it is; the
        # side's corner vertices move inwards (pbrt's boundary rule rounds corners). A neighbour
        # across an interior edge, the next side's middle, would pull it in.
        m = mesh_of("""
            Shape "loopsubdiv" "integer levels" [3]
                "point3 P" [ -1 -1 0  0 -1 0  1 -1 0  -1 0 0  0 0 0  1 0 0  -1 1 0  0 1 0  1 1 0 ]
                "integer indices" [ 0 1 4  0 4 3   1 2 5  1 5 4   3 4 7  3 7 6   4 5 8  4 8 7 ]""")
        p = GeometryBasics.coordinates(m)
        for q in (Point3f(0, -1, 0), Point3f(-1, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0), Point3f(0, 0, 0))
            @test minimum(v -> norm(Vec3f(v - q)), p) < 1f-6
        end
        @test all(q -> abs(q[3]) < 1f-6, p)
        @test all(n -> abs(abs(Vec3f(n)[3]) - 1) < 1f-6, m.normal)
    end

    @testset "a mirror's reflection on a loop-subdivided dome is where pbrt's is" begin
        # The suite scene: a gold mirror subdivided four times reflects an area light. Every
        # pixel within 1 of pbrt's (the 2.8 % of the scene that was off was in pixels off by up
        # to 8, along the reflection's edge).
        name = "tex_conductor_smooth_curved_arealight"
        ref, ours = ensure_reference(name), render_scene(name; samples = REFERENCE_SPP)
        lum(c) = Float64(red(c)) + Float64(green(c)) + Float64(blue(c))
        @test maximum(abs.(lum.(ours) .- lum.(ref))) < 1
        @test compute_metrics(ref, ours).energy_ratio ≈ 1 rtol = 0.005
    end
end
