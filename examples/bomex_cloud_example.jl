# BOMEX Cloud Example for Hikari VolPath Integrator
#
# Renders real cloud simulation data (BOMEX LES) using volumetric path tracing.
# The BOMEX data contains sparse cloud fields from large-eddy simulation.

using Hikari
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
# BOMEX Cloud in Cornell Box
# =============================================================================

function example_bomex_cornell_box(;
    frame_idx::Int = 5,  # Frame 5 has good cloud structure
    extinction_scale::Float32 = 10000f0,  # Scale LWC (kg/kg) to extinction coefficient (~240 at max)
    single_scatter_albedo::Float32 = 5f0,  # Clouds: high scattering, some absorption for visibility
    g::Float32 = 10f0,  # Forward scattering (typical for clouds)
    samples::Int = 25,
    max_depth::Int = 16,
    width::Int = 512,
    height::Int = 512,
)
    println("\nBOMEX Cloud in Cornell Box")
    println("=" ^ 50)

    # Load BOMEX data
    println("Loading BOMEX data...")
    qlt, grid_extent = load_bomex_data()
    println("  Data shape: ", size(qlt))

    # Extract density field - BOMEX qˡ is liquid water content in kg/kg
    # Typical cloud LWC is 0.1-2 g/m³, BOMEX values are ~0.001 kg/kg
    # We scale by extinction_scale to get physically reasonable extinction coefficients
    density_raw = interior(qlt[frame_idx])
    density_scaled = Float32.(density_raw .* extinction_scale)
    println("  Frame $frame_idx: $(count(x -> x > 0, density_scaled)) non-zero voxels")
    println("  LWC range: $(extrema(density_raw)) kg/kg")
    println("  Scaled extinction range: $(extrema(density_scaled))")

    # Rotate the volume 90° so the cloud layer (originally at low z) faces the camera
    # Original data: clouds at z=14-32 out of 84 (near bottom of domain)
    # After rotation: clouds face the front (-z direction toward camera)
    # permutedims (x,y,z) -> (x,z,y) rotates around x-axis
    density_rotated = permutedims(density_scaled, (1, 3, 2))
    # Flip so clouds face camera (they were at low z, now should be at high y after rotation)
    density_rotated = reverse(density_rotated, dims=2)
    println("  Rotated shape: ", size(density_rotated))

    # Cornell box materials
    white = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.73f0, 0.73f0, 0.73f0))
    red = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.65f0, 0.05f0, 0.05f0))
    green = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.12f0, 0.45f0, 0.15f0))

    # Box dimensions (unit scale)
    box_size = 2f0
    half = box_size / 2

    # Create Cornell box walls
    floor_prim = tmesh(Rect3f(Vec3f(-half, 0, -half), Vec3f(box_size, 0.01f0, box_size)), white)
    ceiling_prim = tmesh(Rect3f(Vec3f(-half, box_size - 0.01f0, -half), Vec3f(box_size, 0.01f0, box_size)), white)
    back_wall = tmesh(Rect3f(Vec3f(-half, 0, half - 0.01f0), Vec3f(box_size, box_size, 0.01f0)), white)
    left_wall = tmesh(Rect3f(Vec3f(-half, 0, -half), Vec3f(0.01f0, box_size, box_size)), red)
    right_wall = tmesh(Rect3f(Vec3f(half - 0.01f0, 0, -half), Vec3f(0.01f0, box_size, box_size)), green)

    # Cloud cube in center of box
    cloud_cube_size = 1.2f0
    cloud_cube_origin = Point3f(-cloud_cube_size/2, 0.3f0, -cloud_cube_size/2)
    cloud_cube_bounds = Raycore.Bounds3(
        cloud_cube_origin,
        cloud_cube_origin + Vec3f(cloud_cube_size, cloud_cube_size, cloud_cube_size)
    )

    # Create GridMedium with BOMEX data
    # The density field now contains extinction coefficients directly (LWC * extinction_scale)
    # σ_s = albedo * σ_t, σ_a = (1 - albedo) * σ_t
    # Since density IS σ_t, we set σ_s=albedo and σ_a=(1-albedo) as multipliers
    cloud_medium = Hikari.GridMedium(
        density_rotated;
        σ_a = Hikari.RGBSpectrum(1f0 - single_scatter_albedo),  # Absorption = (1-albedo) * density
        σ_s = Hikari.RGBSpectrum(single_scatter_albedo),        # Scattering = albedo * density
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

    # Cloud cube primitive
    cloud_cube_prim = tmesh(
        Rect3f(cloud_cube_origin, Vec3f(cloud_cube_size, cloud_cube_size, cloud_cube_size)),
        cloud_material
    )

    # Build scene
    mat_scene = Hikari.MaterialScene([
        floor_prim, ceiling_prim, back_wall, left_wall, right_wall,
        cloud_cube_prim
    ])

    lights = (
        Hikari.PointLight(Point3f(0, 1.9f0, 0), Hikari.RGBSpectrum(20f0, 20f0, 20f0)),
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
    # Postprocess and save
    Hikari.postprocess!(film; exposure=1.5f0, tonemap=:aces, gamma=1.0f0)
    return film.postprocess
end

# =============================================================================
# Run Example
# =============================================================================

img = example_bomex_cornell_box(
    samples=200,
    max_depth=100
)
