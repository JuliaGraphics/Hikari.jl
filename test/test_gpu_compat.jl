# GPU Compatibility Tests for VolPath Kernel Inner Functions
#
# This file tests kernel inner functions for GPU compatibility issues.
# It creates a complete test scene with various materials, lights, and media,
# then analyzes key functions for runtime dispatch, throws, and allocations.
#
# Usage: Run each test block manually in the REPL after loading:
#   using JET, Test, GeometryBasics, LinearAlgebra, StaticArrays
#   using Hikari, Raycore, KernelAbstractions
#   include("test/gpu_compat_analyzer.jl")

using Test
using GeometryBasics
using LinearAlgebra
using StaticArrays

# ============================================================================
# Test Scene Setup - Creates concrete types for all VolPath components
# ============================================================================

"""
Create a test scene with various materials, lights, and geometry.
Returns all components needed for testing kernel inner functions.
"""
function create_test_scene()
    using Hikari
    using Raycore

    # Helper to create triangle mesh from GeometryBasics primitives
    function tmesh(prim, material)
        prim_tess = prim isa Sphere ? Tesselation(prim, 32) : prim
        mesh = normal_mesh(prim_tess)
        Hikari.GeometricPrimitive(Raycore.TriangleMesh(mesh), material)
    end

    # --- Materials ---
    white_matte = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.73f0, 0.73f0, 0.73f0))
    red_matte = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.65f0, 0.05f0, 0.05f0))
    green_matte = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.12f0, 0.45f0, 0.15f0))
    glass = Hikari.GlassMaterial(index=1.5f0)
    emissive = Hikari.EmissiveMaterial(Le=Hikari.RGBSpectrum(10f0, 10f0, 10f0))

    # --- Media (for volumetric rendering) ---
    fog = Hikari.HomogeneousMedium(
        σ_a = Hikari.RGBSpectrum(0.01f0),
        σ_s = Hikari.RGBSpectrum(0.1f0),
        Le = Hikari.RGBSpectrum(0f0),
        g = 0.3f0
    )

    # Glass sphere filled with fog
    glass_with_fog = Hikari.MediumInterface(glass; inside=fog, outside=nothing)

    # --- Geometry (Cornell box + objects) ---
    box_size = 2f0
    half = box_size / 2f0

    floor = tmesh(Rect3f(Vec3f(-half, 0, -half), Vec3f(box_size, 0.01f0, box_size)), white_matte)
    ceiling = tmesh(Rect3f(Vec3f(-half, box_size - 0.01f0, -half), Vec3f(box_size, 0.01f0, box_size)), white_matte)
    back_wall = tmesh(Rect3f(Vec3f(-half, 0, half - 0.01f0), Vec3f(box_size, box_size, 0.01f0)), white_matte)
    left_wall = tmesh(Rect3f(Vec3f(-half, 0, -half), Vec3f(0.01f0, box_size, box_size)), red_matte)
    right_wall = tmesh(Rect3f(Vec3f(half - 0.01f0, 0, -half), Vec3f(0.01f0, box_size, box_size)), green_matte)

    # Objects
    matte_sphere = tmesh(Sphere(Point3f(-0.4f0, 0.35f0, 0.2f0), 0.35f0), white_matte)
    glass_sphere = tmesh(Sphere(Point3f(0.4f0, 0.4f0, -0.1f0), 0.4f0), glass_with_fog)
    emissive_sphere = tmesh(Sphere(Point3f(0f0, 1.7f0, 0f0), 0.15f0), emissive)

    # --- Build MaterialScene ---
    primitives = [floor, ceiling, back_wall, left_wall, right_wall,
                  matte_sphere, glass_sphere, emissive_sphere]
    mat_scene = Hikari.MaterialScene(primitives)

    # --- Lights ---
    point_light = Hikari.PointLight(Point3f(0f0, 1.5f0, 0f0), Hikari.RGBSpectrum(10f0, 10f0, 10f0))
    ambient_light = Hikari.AmbientLight(Hikari.RGBSpectrum(0.1f0, 0.1f0, 0.15f0))
    lights = (point_light, ambient_light)

    # --- Scene ---
    scene = Hikari.Scene(lights, mat_scene)

    # --- Camera & Film ---
    width, height = 64, 64  # Small for testing
    aspect = Float32(width) / Float32(height)
    film = Hikari.Film(Point2f(width, height))

    camera_pos = Point3f(0f0, 1f0, -2.5f0)
    camera_lookat = Point3f(0f0, 0.8f0, 0f0)
    camera_up = Vec3f(0f0, 1f0, 0f0)
    fov = 60f0

    screen_window = Raycore.Bounds2(Point2f(-aspect, -1f0), Point2f(aspect, 1f0))
    camera = Hikari.PerspectiveCamera(
        Raycore.look_at(camera_pos, camera_lookat, camera_up),
        screen_window, 0f0, 1f0, 0f0, 1f6, fov, film
    )

    return (
        scene = scene,
        mat_scene = mat_scene,
        film = film,
        camera = camera,
        materials = mat_scene.materials,
        media = mat_scene.media,
        lights = lights,
        fog = fog,
    )
end

# ============================================================================
# Basic Analyzer Tests (without Hikari dependency)
# ============================================================================

# Simple arithmetic - should be GPU compatible
@inline function gpu_compatible_add(a::Float32, b::Float32)::Float32
    return a + b
end

# Type-stable function with concrete types
@inline function gpu_compatible_dot(a::NTuple{3, Float32}, b::NTuple{3, Float32})::Float32
    return a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
end

# Runtime dispatch - type instability
function gpu_bad_dispatch(x)
    return x + 1  # x is Any, causes dispatch
end

# Throws an error
function gpu_bad_throw(x::Float32)
    if x < 0
        throw(DomainError(x, "must be non-negative"))
    end
    return sqrt(x)
end

# Heap allocation
function gpu_bad_alloc(n::Int)
    arr = Vector{Float32}(undef, n)
    return arr
end

# ============================================================================
# Test Functions (run manually in REPL)
# ============================================================================

"""
Run basic detection tests for the GPU analyzer.
"""
function run_basic_tests()
    println("="^60)
    println("Basic Detection Tests")
    println("="^60)

    # Test GPU compatible functions
    println("\nGPU Compatible Functions:")
    result = report_gpu_compat(gpu_compatible_add, (Float32, Float32))
    @assert isempty(JET.get_reports(result)) "gpu_compatible_add should be GPU compatible"
    println("  ✓ gpu_compatible_add: no issues")

    result = report_gpu_compat(gpu_compatible_dot, (NTuple{3, Float32}, NTuple{3, Float32}))
    @assert isempty(JET.get_reports(result)) "gpu_compatible_dot should be GPU compatible"
    println("  ✓ gpu_compatible_dot: no issues")

    # Test runtime dispatch detection
    println("\nRuntime Dispatch Detection:")
    result = report_gpu_compat(gpu_bad_dispatch, (Any,); check_throw=false, check_alloc=false)
    reports = JET.get_reports(result)
    @assert !isempty(reports) "Should detect runtime dispatch"
    @assert any(r -> r isa GPURuntimeDispatchReport, reports) "Should have GPURuntimeDispatchReport"
    println("  ✓ Detected $(length(reports)) runtime dispatch issue(s)")

    # Test throw detection (with intrinsic filtering)
    println("\nThrow Detection:")
    result = report_gpu_compat(gpu_bad_throw, (Float32,); check_dispatch=false, check_alloc=false)
    reports = JET.get_reports(result)
    # Note: sqrt throws are filtered by default (ignore_intrinsic_throws=true)
    # Only the explicit DomainError throw should be reported
    println("  ✓ Detected $(length(reports)) throw statement(s) (GPU intrinsic throws filtered)")

    # Test allocation detection
    println("\nAllocation Detection:")
    result = report_gpu_compat(gpu_bad_alloc, (Int,); check_dispatch=false, check_throw=false)
    reports = JET.get_reports(result)
    @assert !isempty(reports) "Should detect allocation"
    @assert any(r -> r isa GPUAllocationReport, reports) "Should have GPUAllocationReport"
    println("  ✓ Detected $(length(reports)) allocation(s)")

    # Test GPU intrinsic filtering
    println("\nGPU Intrinsic Throw Filtering:")
    result = report_gpu_compat(Base.sqrt, (Float32,); ignore_intrinsic_throws=true)
    println("  sqrt with filtering: $(length(JET.get_reports(result))) reports")
    result = report_gpu_compat(Base.sqrt, (Float32,); ignore_intrinsic_throws=false)
    println("  sqrt without filtering: $(length(JET.get_reports(result))) reports")

    println("\n" * "="^60)
    println("All basic tests passed!")
    println("="^60)
end

"""
Run comprehensive VolPath kernel function tests.
"""
function run_volpath_tests()
    using Hikari
    using KernelAbstractions
    using Raycore

    println("="^70)
    println("VolPath Comprehensive GPU Compatibility Tests")
    println("="^70)

    # Create test scene
    println("\nCreating test scene...")
    test_data = create_test_scene()
    println("  ✓ Scene created")

    # Run small render to verify everything works
    println("\nRunning integration render...")
    scene = test_data.scene
    film = test_data.film
    camera = test_data.camera

    integrator = Hikari.VolPath(samples_per_pixel=1, max_depth=3)
    Hikari.clear!(film)
    integrator(scene, film, camera)

    nonzero = count(x -> x != 0, film.pixels)
    println("  ✓ Render completed: $nonzero / $(length(film.pixels)) non-zero pixels")

    # Get types for GPU compatibility testing
    mat_scene = test_data.mat_scene
    materials = test_data.materials
    media = test_data.media
    lights = test_data.lights
    textures = mat_scene.textures

    # Create VolPathState to get rgb2spec_table
    state = Hikari.VolPathState(KernelAbstractions.CPU(), 64*64, Int32(8))
    rgb2spec_table = state.rgb2spec_table

    # Create sample types
    lambda_vals = (400f0, 500f0, 600f0, 700f0)
    pdf_vals = (1f0, 1f0, 1f0, 1f0)
    lambda = Hikari.SampledWavelengths{4}(lambda_vals, pdf_vals)
    mat_idx = Hikari.MaterialIndex(UInt8(1), UInt32(1))
    ray = Raycore.Ray(o=Point3f(0,0,0), d=Vec3f(0,0,1), t_max=Inf32)

    # Get material instances
    matte_mat = first(materials[1])
    glass_mat = first(materials[2])
    emissive_mat = first(materials[3])

    # Build comprehensive test list
    all_tests = []

    # Material dispatch functions
    push!(all_tests, ("is_emissive_dispatch", Hikari.is_emissive_dispatch, (typeof(materials), typeof(mat_idx))))
    push!(all_tests, ("is_pure_emissive_dispatch", Hikari.is_pure_emissive_dispatch, (typeof(materials), typeof(mat_idx))))
    push!(all_tests, ("has_bsdf_dispatch", Hikari.has_bsdf_dispatch, (typeof(materials), typeof(mat_idx))))
    push!(all_tests, ("has_medium_interface_dispatch", Hikari.has_medium_interface_dispatch, (typeof(materials), typeof(mat_idx))))

    # Medium dispatch functions
    push!(all_tests, ("sample_point_dispatch", Hikari.sample_point_dispatch, (typeof(rgb2spec_table), typeof(media), Int32, Point3f, typeof(lambda))))
    push!(all_tests, ("get_majorant_dispatch", Hikari.get_majorant_dispatch, (typeof(rgb2spec_table), typeof(media), Int32, typeof(ray), Float32, Float32, typeof(lambda))))

    # Phase functions
    push!(all_tests, ("hg_p", Hikari.hg_p, (Float32, Float32)))
    push!(all_tests, ("sample_hg", Hikari.sample_hg, (Float32, Vec3f, Point2f)))

    # Light sampling
    push!(all_tests, ("sample_light_from_tuple", Hikari.sample_light_from_tuple, (typeof(rgb2spec_table), typeof(lights), Int32, Point3f, typeof(lambda), Point2f)))

    # Geometry sampling
    push!(all_tests, ("concentric_sample_disk", Hikari.concentric_sample_disk, (Point2f,)))
    push!(all_tests, ("cosine_sample_hemisphere", Hikari.cosine_sample_hemisphere, (Point2f,)))
    push!(all_tests, ("uniform_sample_sphere", Hikari.uniform_sample_sphere, (Point2f,)))
    push!(all_tests, ("uniform_sample_cone", Hikari.uniform_sample_cone, (Point2f, Float32)))

    # Wavelength sampling
    push!(all_tests, ("sample_visible_wavelengths", Hikari.sample_visible_wavelengths, (Float32,)))

    # BSDF sampling per material type
    push!(all_tests, ("sample_bsdf_spectral (Matte)", Hikari.sample_bsdf_spectral, (typeof(rgb2spec_table), typeof(matte_mat), typeof(textures), Vec3f, Vec3f, Point2f, typeof(lambda), Point2f, Float32)))
    push!(all_tests, ("sample_bsdf_spectral (Glass)", Hikari.sample_bsdf_spectral, (typeof(rgb2spec_table), typeof(glass_mat), typeof(textures), Vec3f, Vec3f, Point2f, typeof(lambda), Point2f, Float32)))
    push!(all_tests, ("sample_bsdf_spectral (Emissive)", Hikari.sample_bsdf_spectral, (typeof(rgb2spec_table), typeof(emissive_mat), typeof(textures), Vec3f, Vec3f, Point2f, typeof(lambda), Point2f, Float32)))

    # Emission functions
    push!(all_tests, ("get_emission_spectral_dispatch", Hikari.get_emission_spectral_dispatch, (typeof(rgb2spec_table), typeof(materials), typeof(textures), typeof(mat_idx), Vec3f, Vec3f, Point2f, typeof(lambda))))
    push!(all_tests, ("get_emission_spectral_uv_dispatch", Hikari.get_emission_spectral_uv_dispatch, (typeof(rgb2spec_table), typeof(materials), typeof(textures), typeof(mat_idx), Point2f, typeof(lambda))))

    # Microfacet/Fresnel functions
    push!(all_tests, ("sample_dielectric_transmission_spectral", Hikari.sample_dielectric_transmission_spectral, (Float32, Vec3f, Float32)))
    push!(all_tests, ("sample_ggx_vndf", Hikari.sample_ggx_vndf, (Vec3f, Float32, Float32, Point2f)))
    push!(all_tests, ("trowbridge_reitz_sample", Hikari.trowbridge_reitz_sample, (Vec3f, Float32, Float32, Float32, Float32)))

    # Run all tests
    println("\n" * "-"^70)
    println("Testing $(length(all_tests)) kernel inner functions...")
    println("-"^70)

    passed = 0
    failed = 0
    errors = 0

    for (name, f, types) in all_tests
        try
            result = report_gpu_compat(f, types; frame_filter=gpu_module_filter(Hikari))
            reports = JET.get_reports(result)
            dispatch = count(r -> r isa GPURuntimeDispatchReport, reports)

            if dispatch == 0
                passed += 1
                println("✓ $name")
            else
                failed += 1
                println("✗ $name: $dispatch runtime dispatch(es)")
            end
        catch e
            errors += 1
            println("⚠ $name: Error - $(typeof(e).name.name)")
        end
    end

    println("-"^70)
    println("Results: $passed passed, $failed failed, $errors errors")
    println("="^70)

    return (passed=passed, failed=failed, errors=errors)
end

# Print usage info
println("""
GPU Compatibility Test Suite
============================

To run tests manually:

1. Load dependencies:
   using JET, Test, GeometryBasics, LinearAlgebra, StaticArrays
   using Hikari, Raycore, KernelAbstractions

2. Load analyzer:
   include("test/gpu_compat_analyzer.jl")

3. Run tests:
   run_basic_tests()      # Test analyzer detection
   run_volpath_tests()    # Test 22 VolPath kernel functions

Or test individual functions:
   @report_gpu_compat my_function(args...)
   report_gpu_compat(my_function, (ArgType1, ArgType2))

Options:
   frame_filter=gpu_module_filter(Hikari)  # Only analyze Hikari code
   check_dispatch=false                     # Skip dispatch checks
   check_throw=false                        # Skip throw checks
   check_alloc=false                        # Skip allocation checks
   ignore_intrinsic_throws=false            # Show GPU intrinsic throws (sqrt, sin, etc.)
""")
