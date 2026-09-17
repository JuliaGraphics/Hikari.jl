using Test
using Hikari
using LinearAlgebra
using GeometryBasics

# `camera_transform` is a rigid motion, so every target along the view ray
# produces the same matrix and the same image. Nothing in a rendering test can
# therefore catch the distance being wrong -- which is how RayMakie shipped a
# camera whose target sat 1 unit ahead of the eye regardless of the scene.
#
# It is not cosmetic downstream: Makie's `Camera3D` scales pan by
# `2*norm(lookat-eye)/height*delta` and zoom by `lookat - zoom_step*viewdir`,
# while rotation is angular and ignores the distance entirely. On Crown (true
# distance 34.4, geometry ~90x43x100) the 1-unit target left a full pan drag
# moving the camera 0.14 world units and a scroll click 0.10 -- indistinguish-
# able from dead input, while rotation behaved normally.
@testset "pbrt camera LookAt distance" begin
    @testset "recorded from the file" begin
        scene_str = """
        LookAt 0 5.5 24
               0 11 -10
               0 1 0
        Camera "perspective" "float fov" 30
        WorldBegin
        Material "diffuse"
        Shape "sphere" "float radius" 1
        """
        pbrt = Hikari.parse_pbrt_string(scene_str)
        @test pbrt.camera_lookat_distance ≈ norm(Point3f(0, 11, -10) - Point3f(0, 5.5, 24))
        @test pbrt.camera_lookat_distance ≈ 34.44198f0 rtol = 1.0f-5
    end

    # Absent `LookAt`, 0 means "unknown" so the caller can fall back to scene
    # bounds. Returning 1 here would be indistinguishable from a real 1-unit
    # LookAt and would silently reintroduce the bug.
    @testset "absent LookAt reports 0, not a guess" begin
        scene_str = """
        Camera "perspective"
        WorldBegin
        Material "diffuse"
        Shape "sphere" "float radius" 1
        """
        pbrt = Hikari.parse_pbrt_string(scene_str)
        @test pbrt.camera_lookat_distance == 0.0f0
    end

    # `Transform` replaces the CTM wholesale, so a distance recorded by an
    # earlier `LookAt` does not describe the camera any more.
    @testset "Transform after LookAt clears the distance" begin
        scene_str = """
        LookAt 0 0 10   0 0 0   0 1 0
        Transform [ 1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1 ]
        Camera "perspective"
        WorldBegin
        Material "diffuse"
        Shape "sphere" "float radius" 1
        """
        pbrt = Hikari.parse_pbrt_string(scene_str)
        @test pbrt.camera_lookat_distance == 0.0f0
    end

    # The property that matters is that the distance describes the SAME camera
    # as `camera_transform`: the eye recovered from the matrix, advanced by the
    # distance along the view direction, must land on the file's target.
    #
    # Deliberately not pinned: two `LookAt`s before `WorldBegin`. pbrt-v4
    # CONCATENATES `LookAt` onto the CTM rather than replacing it (see the
    # comment on that branch in parser.jl), so `LookAt ... z 10` followed by
    # `LookAt ... z 500` puts the eye at z=510 and no single input distance
    # describes it. Real scenes issue one, and this asserts that case.
    @testset "distance is consistent with camera_transform" begin
        scene_str = """
        LookAt 3 4 12
               1 2 3
               0 1 0
        Camera "perspective"
        WorldBegin
        Material "diffuse"
        Shape "sphere" "float radius" 1
        """
        pbrt = Hikari.parse_pbrt_string(scene_str)
        ctw = inv(pbrt.camera_transform)
        eye = Point3f(ctw[1, 4], ctw[2, 4], ctw[3, 4])
        forward = normalize(Vec3f(-ctw[1, 3], -ctw[2, 3], -ctw[3, 3]))
        reconstructed = eye + forward * pbrt.camera_lookat_distance
        @test eye ≈ Point3f(3, 4, 12) rtol = 1.0f-4
        @test reconstructed ≈ Point3f(1, 2, 3) rtol = 1.0f-4
    end

    # Crown is the scene the bug was found on; pin it end to end.
    @testset "crown.pbrt" begin
        crown = "/sim/Programmieren/VulkanDev/RayDemo/Crown/crown.pbrt"
        if isfile(crown)
            pbrt = Hikari.parse_pbrt(crown)
            @test pbrt.camera_lookat_distance ≈ 34.44198f0 rtol = 1.0f-5
        else
            @test_skip "crown.pbrt not available"
        end
    end
end
