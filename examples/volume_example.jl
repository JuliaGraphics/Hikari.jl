# Volume Rendering Example
# Demonstrates CloudVolume material with Whitted integrator using BOMEX LES cloud data

using Hikari
using Raycore
using GeometryBasics
using FileIO
using Colors
using LinearAlgebra: normalize

# =============================================================================
# Load BOMEX Cloud Data
# =============================================================================

# Load the BOMEX LES cloud data via Oceananigans
using Oceananigans
const FieldTimeSeries = Oceananigans.FieldTimeSeries

bomex_path = joinpath(@__DIR__, "..", "..", "..", "bomex_3d.jld2")
qˡt = FieldTimeSeries(bomex_path, "qˡ")
println("Loaded BOMEX data: $(size(qˡt))")

# Compute grid extent (normalized to ~2 units wide)
const grid_extent = (2.0, 2.0, size(qˡt)[3] / size(qˡt)[1] * 2.0)
println("Grid extent: $grid_extent")

# Find first frame with cloud data
function find_cloud_frame(qˡt)
    for i in 1:size(qˡt, 4)
        if maximum(interior(qˡt[i])) > 0
            return i
        end
    end
    return 1
end

# =============================================================================
# Scene Setup
# =============================================================================

function create_bomex_cloud_scene(qˡ_field)
    # Create cloud volume from BOMEX data
    cloud = Hikari.CloudVolume(
        Float32.(interior(qˡ_field));
        origin=Point3f(-1, -1, 0),
        extent=Vec3f(2, 2, Float32(grid_extent[3])),
        extinction_scale=10000f0,  # High value for sparse cloud data
        asymmetry_g=0.85f0,        # Forward scattering (realistic for clouds)
        single_scatter_albedo=0.99f0
    )
    println("Cloud: origin=$(cloud.origin), extent=$(cloud.extent)")

    # Cloud box geometry - convert to TriangleMesh
    cloud_box_geo = Rect3f(cloud.origin, cloud.extent)
    cloud_box_mesh = Raycore.TriangleMesh(normal_mesh(cloud_box_geo))

    # Create material scene
    mat_scene = Hikari.MaterialScene([
        (cloud_box_mesh, cloud),
    ])

    # SunSkyLight - provides both sun illumination and procedural sky background
    sun_dir = normalize(Vec3f(0.4f0, 0.5f0, 0.85f0))  # Sun high in sky
    sun_sky = Hikari.SunSkyLight(
        sun_dir,
        Hikari.RGBSpectrum(25f0);  # Sun intensity
        turbidity=2.5f0,           # Clear sky
        ground_albedo=Hikari.RGBSpectrum(0.8f0, 0.8f0, 0.9f0)
    )

    Hikari.Scene((sun_sky,), mat_scene), cloud
end

function create_camera_and_film(; width=1024, height=768)
    resolution = Point2f(width, height)
    film = Hikari.Film(resolution; filter=Hikari.LanczosSincFilter(Point2f(2, 2), 3f0))

    # Camera positioned to look at the cloud
    crop_bounds = Raycore.Bounds2(Point2f(0, 0), Point2f(1, 1))
    cam_pos = Point3f(2.0, 3.0, 0.0)
    look_at_pt = Point3f(0.0, 0.0, -0.3)  # Aim at cloud region
    up = Vec3f(0, 0, 1)
    cam_to_world = Raycore.look_at(cam_pos, look_at_pt, up)

    camera = Hikari.PerspectiveCamera(
        cam_to_world, crop_bounds,
        0f0, 1f0, 0f0, 1f6,
        45f0, film
    )

    film, camera
end

# =============================================================================
# Render
# =============================================================================

function render_bomex_cloud(; samples=8, max_depth=3, width=1024, height=768, frame=nothing, save_path=nothing)
    # Use specified frame or find first frame with cloud data
    frame_idx = isnothing(frame) ? find_cloud_frame(qˡt) : frame
    println("Using frame $frame_idx")

    scene, cloud = create_bomex_cloud_scene(qˡt[frame_idx])
    film, camera = create_camera_and_film(; width, height)

    integrator = Hikari.Whitted(; samples, max_depth)

    Hikari.clear!(film)
    println("Rendering BOMEX cloud frame $frame_idx with Whitted integrator ($(samples) spp)...")
    @time integrator(scene, film, camera)

    # Postprocess
    Hikari.postprocess!(film; exposure=1.0f0, tonemap=:aces, gamma=1.2f0)
    fb = film.postprocess

    if !isnothing(save_path)
        fb_clamped = map(c -> RGB(clamp(c.r, 0, 1), clamp(c.g, 0, 1), clamp(c.b, 0, 1)), fb)
        FileIO.save(save_path, fb_clamped)
        println("Saved to $save_path")
    end

    fb, film, camera, scene, cloud
end

# =============================================================================
# Run Example
# =============================================================================

# Render single frame (auto-selects first frame with cloud data)
fb, film, camera, scene, cloud = render_bomex_cloud(; samples=8, save_path="bomex_cloud.png")
