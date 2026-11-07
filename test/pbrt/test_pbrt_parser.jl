using Test
using Hikari
using GeometryBasics
using LinearAlgebra
using FileIO

const SCENES_DIR = joinpath(@__DIR__, "scenes")

# ============================================================================
# Tokenizer tests
# ============================================================================
@testset "PBRT Tokenizer" begin
    tokens = Hikari.tokenize("""
    # This is a comment
    LookAt 0 0 5  0 0 0  0 1 0
    Camera "perspective" "float fov" [45]
    Shape "sphere" "float radius" 1.5
    """)

    # Check token types
    @test tokens[1].type == Hikari.TOK_WORD
    @test tokens[1].value == "LookAt"
    @test tokens[2].type == Hikari.TOK_NUMBER
    @test tokens[2].value == "0"

    # Quoted strings
    cam_tok = findfirst(t -> t.value == "perspective", tokens)
    @test cam_tok !== nothing
    @test tokens[cam_tok].type == Hikari.TOK_STRING

    # Brackets
    lbracket = findfirst(t -> t.type == Hikari.TOK_LBRACKET, tokens)
    @test lbracket !== nothing

    # Escaped strings
    tokens2 = Hikari.tokenize("""Shape "plymesh" "string filename" "path/to/file.ply" """)
    file_tok = findfirst(t -> t.value == "path/to/file.ply", tokens2)
    @test file_tok !== nothing
end

# ============================================================================
# Parser tests
# ============================================================================
@testset "PBRT Parser" begin
    @testset "Basic scene structure" begin
        pbrt = Hikari.parse_pbrt_string("""
        LookAt 0 0 5  0 0 0  0 1 0
        Camera "perspective" "float fov" 45
        Film "rgb" "integer xresolution" 800 "integer yresolution" 600
        Integrator "volpath" "integer maxdepth" 12

        WorldBegin
        LightSource "point" "rgb I" [100 100 100]
        Material "diffuse" "rgb reflectance" [0.5 0.5 0.5]
        Shape "sphere" "float radius" 1
        """)

        @test pbrt.film !== nothing
        @test pbrt.film.type == "rgb"
        @test Hikari.pbrt_get_int(pbrt.film, "xresolution", 0) == 800
        @test Hikari.pbrt_get_int(pbrt.film, "yresolution", 0) == 600

        @test pbrt.camera !== nothing
        @test pbrt.camera.type == "perspective"
        @test Hikari.pbrt_get_float(pbrt.camera, "fov", 0.0) == 45.0

        @test pbrt.integrator !== nothing
        @test Hikari.pbrt_get_int(pbrt.integrator, "maxdepth", 0) == 12

        @test length(pbrt.shapes) == 1
        @test pbrt.shapes[1].entity.type == "sphere"
        @test Hikari.pbrt_get_float(pbrt.shapes[1].entity, "radius", 0.0) == 1.0

        @test length(pbrt.lights) == 1
        @test pbrt.lights[1].entity.type == "point"
    end

    @testset "Named materials" begin
        pbrt = Hikari.parse_pbrt_string("""
        WorldBegin
        MakeNamedMaterial "mymat" "string type" "diffuse"
            "rgb reflectance" [0.8 0.1 0.1]
        MakeNamedMaterial "glass" "string type" "dielectric"
            "float eta" 1.5

        NamedMaterial "mymat"
        Shape "sphere" "float radius" 1

        NamedMaterial "glass"
        Shape "sphere" "float radius" 0.5
        """)

        @test length(pbrt.named_materials) == 2
        @test haskey(pbrt.named_materials, "mymat")
        @test haskey(pbrt.named_materials, "glass")
        @test pbrt.named_materials["mymat"].type == "diffuse"
        @test pbrt.named_materials["glass"].type == "dielectric"

        @test pbrt.shapes[1].material_name == "mymat"
        @test pbrt.shapes[2].material_name == "glass"
    end

    @testset "Transforms" begin
        pbrt = Hikari.parse_pbrt_string("""
        WorldBegin
        AttributeBegin
            Translate 1 2 3
            Material "diffuse" "rgb reflectance" [1 0 0]
            Shape "sphere" "float radius" 0.5
        AttributeEnd
        """)

        transform = pbrt.shapes[1].transform
        # Column-major: translation in column 4
        @test transform[1, 4] ≈ 1f0
        @test transform[2, 4] ≈ 2f0
        @test transform[3, 4] ≈ 3f0
    end

    @testset "AttributeBegin/End scoping" begin
        pbrt = Hikari.parse_pbrt_string("""
        WorldBegin
        Material "diffuse" "rgb reflectance" [1 0 0]

        AttributeBegin
            Material "dielectric" "float eta" 1.5
            Shape "sphere" "float radius" 1
        AttributeEnd

        # Should revert to outer material
        Shape "sphere" "float radius" 0.5
        """)

        @test pbrt.shapes[1].material_inline.type == "dielectric"
        @test pbrt.shapes[2].material_inline.type == "diffuse"
    end

    @testset "Area lights" begin
        pbrt = Hikari.parse_pbrt_string("""
        WorldBegin
        AttributeBegin
            AreaLightSource "diffuse" "rgb L" [5 5 5]
            Material "diffuse" "rgb reflectance" [1 1 1]
            Shape "sphere" "float radius" 0.5
        AttributeEnd

        # Shape without area light
        Material "diffuse" "rgb reflectance" [0.5 0.5 0.5]
        Shape "sphere" "float radius" 1
        """)

        @test pbrt.shapes[1].area_light !== nothing
        @test pbrt.shapes[1].area_light.type == "diffuse"
        @test pbrt.shapes[2].area_light === nothing
    end

    @testset "LookAt matrix" begin
        m = Hikari.pbrt_lookat(Point3f(0, 0, 5), Point3f(0, 0, 0), Vec3f(0, 1, 0))
        # Camera at (0,0,5) looking at origin — should be a valid view matrix
        c2w = inv(m)
        eye = Point3f(c2w[1, 4], c2w[2, 4], c2w[3, 4])
        @test eye[1] ≈ 0f0 atol = 1e-5
        @test eye[2] ≈ 0f0 atol = 1e-5
        @test eye[3] ≈ 5f0 atol = 1e-4
    end

    @testset "Triangle mesh" begin
        pbrt = Hikari.parse_pbrt_string("""
        WorldBegin
        Material "diffuse" "rgb reflectance" [1 1 1]
        Shape "trianglemesh"
            "point3 P" [-1 -1 0  1 -1 0  1 1 0  -1 1 0]
            "integer indices" [0 1 2  0 2 3]
        """)

        @test length(pbrt.shapes) == 1
        @test pbrt.shapes[1].entity.type == "trianglemesh"
        P = Hikari.pbrt_get_floats(pbrt.shapes[1].entity, "P")
        @test length(P) == 12  # 4 vertices × 3 components
        indices = Hikari.pbrt_get_ints(pbrt.shapes[1].entity, "indices")
        @test length(indices) == 6  # 2 triangles × 3
    end
end

# ============================================================================
# Scene builder tests
# ============================================================================
@testset "PBRT Scene Builder" begin
    @testset "Material construction" begin
        entity = Hikari.PBRTEntity("Material", "diffuse",
            Dict("reflectance" => Hikari.PBRTParam(:rgb, "reflectance", [0.8, 0.2, 0.1])))
        mat = Hikari.build_pbrt_material(entity, Hikari.PBRTScene(
            nothing, nothing, Mat4f(I), nothing, nothing,
            Dict{String,Hikari.PBRTEntity}(), Dict{String,Hikari.PBRTEntity}(),
            Dict{String,Hikari.PBRTEntity}(), Dict{String,Mat4f}(),
            Hikari.PBRTShapeRecord[], Hikari.PBRTLightRecord[], "."))
        @test mat isa Hikari.Diffuse
    end

    @testset "Sphere tessellation" begin
        mesh = Hikari.tessellate_sphere(1f0; segments=16)
        coords = GeometryBasics.coordinates(mesh)
        # All vertices should be at radius 1 (±numerical error)
        for c in coords
            @test norm(Vec3f(c)) ≈ 1f0 atol = 1e-5
        end
    end

    @testset "Disk tessellation" begin
        mesh = Hikari.tessellate_disk(2f0; segments=32)
        coords = GeometryBasics.coordinates(mesh)
        # All vertices on z=0 plane
        for c in coords
            @test c[3] ≈ 0f0
        end
        # All non-center vertices at radius 2
        for c in coords[2:end]
            @test norm(Vec3f(c[1], c[2], 0)) ≈ 2f0 atol = 1e-5
        end
    end
end

# ============================================================================
# Full scene rendering tests (GPU)
# ============================================================================
@testset "PBRT Scene Rendering" begin
    backend = if isdefined(Main, :Lava) && isdefined(Lava, :LavaBackend)
        Mantle.defaultbackend()
    else
        KernelAbstractions.CPU()
    end

    function render_and_check(pbrt_file; samples=4, max_depth=5, min_nonzero_frac=0.01)
        path = joinpath(SCENES_DIR, pbrt_file)
        @test isfile(path)
        r = Hikari.load_pbrt(path; backend=backend, samples=samples, max_depth=max_depth)
        vp = Hikari.VolPath(samples=samples, max_depth=max_depth)
        vp(r.scene, r.film, r.camera)
        Hikari.postprocess!(r.film)
        fb = r.film.framebuffer
        nonzero = sum(p -> p.r > 0.001 || p.g > 0.001 || p.b > 0.001, fb)
        frac = nonzero / length(fb)
        @test frac > min_nonzero_frac
        return r.film
    end

    @testset "Point light + sphere" begin
        render_and_check("single_sphere_point.pbrt")
    end

    @testset "Distant light + sphere" begin
        render_and_check("single_sphere_distant.pbrt")
    end

    @testset "Area light + sphere" begin
        render_and_check("single_sphere_area.pbrt")
    end

    @testset "All materials" begin
        render_and_check("all_materials.pbrt"; max_depth=10)
    end

    @testset "Named materials" begin
        render_and_check("named_materials.pbrt")
    end

    @testset "Transforms" begin
        render_and_check("transforms.pbrt")
    end

    @testset "PLY mesh" begin
        render_and_check("plymesh.pbrt")
    end
end
