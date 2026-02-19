# BOMEX Cloud Example for Hikari VolPath Integrator
#
# Renders real cloud simulation data (BOMEX LES) using volumetric path tracing.
# The BOMEX data contains sparse cloud fields from large-eddy simulation.
#
# Lighting setup inspired by pbrt-v4's Disney cloud scene:
# - Ambient sky light (blue)
# - Directional sun light

using Hikari
using Hikari: Transformation
using GeometryBasics
using GeometryBasics: Point3f, Vec3f, Point2f, normal_mesh
using Raycore
using LinearAlgebra
using FileIO

# Load BOMEX data using Oceananigans
using Oceananigans
const FieldTimeSeries = Oceananigans.FieldTimeSeries

const BOMEX_PATH = joinpath(@__DIR__, "..", "..", "..", "bomex_3d.jld2")

# =============================================================================
# Data Loading
# =============================================================================

function load_bomex_data(path=BOMEX_PATH)
    qlt = FieldTimeSeries(path, "qˡ")
    grid_extent = (2.0, 2.0, size(qlt)[3] / size(qlt)[1] * 2.0)
    return qlt, grid_extent
end

# =============================================================================
# Helper: Create mesh from GeometryBasics primitive
# =============================================================================

function tmesh(prim, material)
    prim = prim isa Sphere ? Tesselation(prim, 64) : prim
    mesh = normal_mesh(prim)
    Hikari.GeometricPrimitive(Raycore.TriangleMesh(mesh), material)
end

# =============================================================================
# BOMEX Cloud with Disney-style Lighting
# =============================================================================

function example_bomex_disney_lighting(;
    frame_idx::Int = 5,  # Frame 5 has good cloud structure
    extinction_scale::Float32 = 500000f0,  # Scale LWC (kg/kg) to extinction (~620 max)
    g::Float32 = 0.877f0,  # Forward scattering (Disney cloud value)
    samples::Int = 64,
    max_depth::Int = 32,
    width::Int = 512,
    height::Int = 512,
)
    println("\nBOMEX Cloud with Disney-style Lighting")
    println("=" ^ 50)

    # Load BOMEX data
    println("Loading BOMEX data...")
    qlt, grid_extent = load_bomex_data()
    println("  Data shape: ", size(qlt))

    # Extract density field - BOMEX qˡ is liquid water content in kg/kg
    # BOMEX values are ~0.001 kg/kg, very sparse
    # extinction_scale converts to physically reasonable extinction coefficients
    density_raw = interior(qlt[frame_idx])
    density_scaled = Float32.(density_raw .* extinction_scale)
    println("  Frame $frame_idx: $(count(x -> x > 0, density_scaled)) non-zero voxels")
    println("  LWC range: $(extrema(density_raw)) kg/kg")
    println("  Scaled extinction range: $(extrema(density_scaled))")

    # Rotate the volume 90° so the cloud layer faces the camera
    density_rotated = permutedims(density_scaled, (1, 3, 2))
    density_rotated = reverse(density_rotated, dims=2)
    println("  Rotated shape: ", size(density_rotated))

    # Scene materials
    white = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.73f0, 0.73f0, 0.73f0))
    red = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.65f0, 0.05f0, 0.05f0))
    green = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.12f0, 0.45f0, 0.15f0))

    # Box dimensions
    box_size = 2f0
    half = box_size / 2

    # Create walls (no ceiling/back for sky visibility)
    floor_prim = tmesh(Rect3f(Vec3f(-half, 0, -half), Vec3f(box_size, 0.01f0, box_size)), white)
    left_wall = tmesh(Rect3f(Vec3f(-half, 0, -half), Vec3f(0.01f0, box_size, box_size)), red)
    right_wall = tmesh(Rect3f(Vec3f(half - 0.01f0, 0, -half), Vec3f(0.01f0, box_size, box_size)), green)

    # Cloud cube bounds
    cloud_cube_size = 1.2f0
    cloud_cube_origin = Point3f(-cloud_cube_size/2, 0.3f0, -cloud_cube_size/2)
    cloud_cube_bounds = Raycore.Bounds3(
        cloud_cube_origin,
        cloud_cube_origin + Vec3f(cloud_cube_size, cloud_cube_size, cloud_cube_size)
    )

    # Create GridMedium with BOMEX data
    # Following Disney cloud: σ_a=0 (pure scattering), σ_s=1 (unit multiplier)
    # The density array already contains extinction coefficients
    cloud_medium = Hikari.GridMedium(
        density_rotated;
        σ_a = Hikari.RGBSpectrum(0f0),   # No absorption (like Disney cloud)
        σ_s = Hikari.RGBSpectrum(1f0),   # Unit scattering multiplier
        g = g,
        bounds = cloud_cube_bounds
    )

    # Transparent boundary (no refraction)
    transparent = Hikari.GlassMaterial(
        Kr = Hikari.RGBSpectrum(0f0),
        Kt = Hikari.RGBSpectrum(1f0),
        index = 1.0f0
    )
    cloud_material = Hikari.MediumInterface(transparent; inside=cloud_medium, outside=nothing)

    cloud_cube_prim = tmesh(
        Rect3f(cloud_cube_origin, Vec3f(cloud_cube_size, cloud_cube_size, cloud_cube_size)),
        cloud_material
    )

    # Build scene
    mat_scene = Hikari.MaterialScene([
        floor_prim, left_wall, right_wall,
        cloud_cube_prim
    ])

    # Disney cloud-style lighting:
    # - Ambient sky: RGB [0.03, 0.07, 0.23] (blue sky)
    # - Distant sun: direction [-0.5826, -0.7660, -0.2717], RGB [2.6, 2.5, 2.3]
    sun_direction = normalize(Vec3f(-0.5826f0, -0.7660f0, -0.2717f0))

    lights = (
        Hikari.AmbientLight(Hikari.RGBSpectrum(0.03f0, 0.07f0, 0.23f0)),
        Hikari.DirectionalLight(
            Transformation(),
            Hikari.RGBSpectrum(2.6f0, 2.5f0, 2.3f0),
            sun_direction
        ),
    )

    scene = Hikari.Scene(lights, mat_scene)

    # Camera and film
    resolution = Point2f(width, height)
    film = Hikari.Film(resolution)

    aspect = Float32(width) / Float32(height)
    screen_window = Raycore.Bounds2(Point2f(-aspect, -1f0), Point2f(aspect, 1f0))

    camera_pos = Point3f(0f0, 1f0, -3.5f0)
    camera_lookat = Point3f(0f0, 0.9f0, 0f0)
    camera_up = Vec3f(0f0, 1f0, 0f0)

    camera = Hikari.PerspectiveCamera(
        Hikari.look_at(camera_pos, camera_lookat, camera_up),
        screen_window, 0f0, 1f0, 0f0, 1f6, 40f0, film
    )

    # Render
    integrator = Hikari.VolPath(samples_per_pixel=samples, max_depth=max_depth)

    println("Rendering with $samples spp, max_depth=$max_depth...")
    Hikari.clear!(film)
    @time integrator(scene, film, camera)

    # Postprocess
    Hikari.postprocess!(film; exposure=1.0f0, tonemap=:aces, gamma=1.0f0)
    return film.postprocess
end

# =============================================================================
# BOMEX Cloud in Cornell Box (original setup with point light)
# =============================================================================

function example_bomex_cornell_box(;
    frame_idx::Int = 5,
    extinction_scale::Float32 = 500000f0,
    g::Float32 = 0.877f0,
    samples::Int = 64,
    max_depth::Int = 32,
    width::Int = 512,
    height::Int = 512,
)
    println("\nBOMEX Cloud in Cornell Box")
    println("=" ^ 50)

    # Load BOMEX data
    println("Loading BOMEX data...")
    qlt, grid_extent = load_bomex_data()
    println("  Data shape: ", size(qlt))

    density_raw = interior(qlt[frame_idx])
    density_scaled = Float32.(density_raw .* extinction_scale)
    println("  Frame $frame_idx: $(count(x -> x > 0, density_scaled)) non-zero voxels")
    println("  Scaled extinction range: $(extrema(density_scaled))")

    density_rotated = permutedims(density_scaled, (1, 3, 2))
    density_rotated = reverse(density_rotated, dims=2)

    # Cornell box materials
    white = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.73f0, 0.73f0, 0.73f0))
    red = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.65f0, 0.05f0, 0.05f0))
    green = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.12f0, 0.45f0, 0.15f0))

    box_size = 2f0
    half = box_size / 2

    floor_prim = tmesh(Rect3f(Vec3f(-half, 0, -half), Vec3f(box_size, 0.01f0, box_size)), white)
    ceiling_prim = tmesh(Rect3f(Vec3f(-half, box_size - 0.01f0, -half), Vec3f(box_size, 0.01f0, box_size)), white)
    back_wall = tmesh(Rect3f(Vec3f(-half, 0, half - 0.01f0), Vec3f(box_size, box_size, 0.01f0)), white)
    left_wall = tmesh(Rect3f(Vec3f(-half, 0, -half), Vec3f(0.01f0, box_size, box_size)), red)
    right_wall = tmesh(Rect3f(Vec3f(half - 0.01f0, 0, -half), Vec3f(0.01f0, box_size, box_size)), green)

    cloud_cube_size = 1.2f0
    cloud_cube_origin = Point3f(-cloud_cube_size/2, 0.3f0, -cloud_cube_size/2)
    cloud_cube_bounds = Raycore.Bounds3(
        cloud_cube_origin,
        cloud_cube_origin + Vec3f(cloud_cube_size, cloud_cube_size, cloud_cube_size)
    )

    cloud_medium = Hikari.GridMedium(
        density_rotated;
        σ_a = Hikari.RGBSpectrum(0f0),
        σ_s = Hikari.RGBSpectrum(1f0),
        g = g,
        bounds = cloud_cube_bounds
    )

    transparent = Hikari.GlassMaterial(
        Kr = Hikari.RGBSpectrum(0f0),
        Kt = Hikari.RGBSpectrum(1f0),
        index = 1.0f0
    )
    cloud_material = Hikari.MediumInterface(transparent; inside=cloud_medium, outside=nothing)

    cloud_cube_prim = tmesh(
        Rect3f(cloud_cube_origin, Vec3f(cloud_cube_size, cloud_cube_size, cloud_cube_size)),
        cloud_material
    )

    mat_scene = Hikari.MaterialScene([
        floor_prim, ceiling_prim, back_wall, left_wall, right_wall,
        cloud_cube_prim
    ])

    lights = (
        Hikari.PointLight(Point3f(0, 1.9f0, 0), Hikari.RGBSpectrum(20f0, 20f0, 20f0)),
    )

    scene = Hikari.Scene(lights, mat_scene)

    resolution = Point2f(width, height)
    film = Hikari.Film(resolution)

    aspect = Float32(width) / Float32(height)
    screen_window = Raycore.Bounds2(Point2f(-aspect, -1f0), Point2f(aspect, 1f0))

    camera_pos = Point3f(0f0, 1f0, -3.5f0)
    camera_lookat = Point3f(0f0, 0.9f0, 0f0)
    camera_up = Vec3f(0f0, 1f0, 0f0)

    camera = Hikari.PerspectiveCamera(
        Hikari.look_at(camera_pos, camera_lookat, camera_up),
        screen_window, 0f0, 1f0, 0f0, 1f6, 40f0, film
    )

    integrator = Hikari.VolPath(samples_per_pixel=samples, max_depth=max_depth)

    println("Rendering with $samples spp, max_depth=$max_depth...")
    Hikari.clear!(film)
    @time integrator(scene, film, camera)
    Hikari.postprocess!(film; exposure=1.5f0, tonemap=:aces, gamma=1.0f0)
    return film.postprocess
end

# =============================================================================
# Run Example
# =============================================================================

# Disney-style lighting (recommended)
img = example_bomex_disney_lighting(samples=64, max_depth=32)
