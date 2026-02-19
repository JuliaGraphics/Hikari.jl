# Medium Example for Hikari VolPath Integrator
#
# This example demonstrates volumetric rendering with participating media,
# following pbrt-v4's approach where surfaces define medium boundaries via MediumInterface.
#
# Key concepts:
# - MediumInterface wraps materials to define medium boundaries at surfaces
# - Use actual Medium objects: MediumInterface(glass; inside=fog, outside=nothing)
# - Media are automatically extracted from MediumInterface during scene construction
# - Camera rays start in vacuum by default
# - Rays enter/exit media only when crossing MediumInterface surfaces

using Hikari
using GeometryBasics
using GeometryBasics: Point3f, Vec3f, Point2f, normal_mesh, Tesselation
using Raycore
using LinearAlgebra

# =============================================================================
# Helper: Create mesh from GeometryBasics primitive
# =============================================================================

function tmesh(prim, material)
    prim = prim isa Sphere ? Tesselation(prim, 64) : prim
    mesh = normal_mesh(prim)
    Hikari.GeometricPrimitive(Raycore.TriangleMesh(mesh), material)
end

# =============================================================================
# Helper: Create a simple Cornell box scene
# =============================================================================

function create_cornell_box(; sphere_material=nothing)
    # Default materials
    white = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.73f0, 0.73f0, 0.73f0))
    red = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.65f0, 0.05f0, 0.05f0))
    green = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.12f0, 0.45f0, 0.15f0))

    sphere_mat = isnothing(sphere_material) ? white : sphere_material

    # Box dimensions
    box_size = 2f0
    half = box_size / 2

    # Create primitives using Rect3f
    floor = tmesh(Rect3f(Vec3f(-half, 0, -half), Vec3f(box_size, 0.01f0, box_size)), white)
    ceiling = tmesh(Rect3f(Vec3f(-half, box_size - 0.01f0, -half), Vec3f(box_size, 0.01f0, box_size)), white)
    back_wall = tmesh(Rect3f(Vec3f(-half, 0, half - 0.01f0), Vec3f(box_size, box_size, 0.01f0)), white)
    left_wall = tmesh(Rect3f(Vec3f(-half, 0, -half), Vec3f(0.01f0, box_size, box_size)), red)
    right_wall = tmesh(Rect3f(Vec3f(half - 0.01f0, 0, -half), Vec3f(0.01f0, box_size, box_size)), green)

    # Sphere in the middle
    sphere = tmesh(Sphere(Point3f(0, 0.5f0, 0), 0.4f0), sphere_mat)

    mat_scene = Hikari.MaterialScene([floor, ceiling, back_wall, left_wall, right_wall, sphere])

    # Area light on ceiling
    lights = (
        Hikari.PointLight(Point3f(0, 1.8f0, 0), Hikari.RGBSpectrum(15f0, 15f0, 15f0)),
    )

    Hikari.Scene(lights, mat_scene)
end

function create_camera(; width=512, height=512)
    resolution = Point2f(width, height)
    film = Hikari.Film(resolution)

    aspect = Float32(width) / Float32(height)
    screen_window = Raycore.Bounds2(Point2f(-aspect, -1f0), Point2f(aspect, 1f0))

    camera_pos = Point3f(0f0, 1f0, -3.5f0)
    camera_lookat = Point3f(0f0, 1f0, 0f0)
    camera_up = Vec3f(0f0, 1f0, 0f0)

    camera = Hikari.PerspectiveCamera(
        Hikari.look_at(camera_pos, camera_lookat, camera_up),
        screen_window, 0f0, 1f0, 0f0, 1f6, 40f0, film
    )

    film, camera
end

# =============================================================================
# Example 1: Glass sphere filled with fog
# =============================================================================
#
# The sphere surface uses MediumInterface to define the medium boundary.
# Camera rays start in vacuum. When a ray enters the glass sphere, it
# transitions into the fog medium. When it exits, it returns to vacuum.
#
# MediumInterface(glass; inside=fog, outside=nothing) means:
# - inside=fog: rays going INTO the sphere enter the fog medium
# - outside=nothing: rays going OUT OF the sphere enter vacuum

function example_foggy_glass_sphere()
    println("\nExample 1: Glass Sphere Filled with Fog")
    println("=" ^ 50)

    # Create fog medium
    fog = Hikari.HomogeneousMedium(
        σ_a = Hikari.RGBSpectrum(0.01f0),
        σ_s = Hikari.RGBSpectrum(0.5f0),   # Dense fog inside sphere
        Le = Hikari.RGBSpectrum(0f0),
        g = 0.3f0                           # Forward scattering
    )

    # Glass material wrapped with MediumInterface
    # inside=fog: fog medium inside the sphere
    # outside=nothing: vacuum outside
    glass = Hikari.GlassMaterial(
        Kr = Hikari.RGBSpectrum(1f0),
        Kt = Hikari.RGBSpectrum(1f0),
        index = 1.5f0
    )
    glass_with_fog = Hikari.MediumInterface(glass; inside=fog, outside=nothing)

    # Create scene with foggy glass sphere
    scene = create_cornell_box(sphere_material=glass_with_fog)
    film, camera = create_camera()

    integrator = Hikari.VolPath(samples_per_pixel=64, max_depth=16)

    # Render - media is automatically extracted from MediumInterface materials
    # gpu_film = Raycore.to_gpu(Array, film)
    # gpu_scene = Raycore.to_gpu(Array, scene)
    gpu_film = film
    gpu_scene = scene
    Hikari.clear!(gpu_film)

    println("Rendering glass sphere with internal fog...")
    println("(Camera in vacuum, fog only inside sphere via MediumInterface)")
    @time integrator(gpu_scene, gpu_film, camera)

    Hikari.postprocess!(gpu_film; exposure=1.0f0, tonemap=:aces, gamma=1.2f0)
    img = Array(gpu_film.postprocess)
    println("Done!")

    return img
end

# =============================================================================
# Example 2: Glowing sphere (emissive medium)
# =============================================================================
#
# A sphere filled with glowing emissive medium.
# The medium emission (Le) creates a volumetric glow effect.

function example_glowing_sphere()
    println("\nExample 2: Glowing Sphere (Emissive Medium)")
    println("=" ^ 50)

    # Emissive medium - glows orange
    glowing_medium = Hikari.HomogeneousMedium(
        σ_a = Hikari.RGBSpectrum(0.1f0),
        σ_s = Hikari.RGBSpectrum(0.05f0),
        Le = Hikari.RGBSpectrum(2f0, 0.8f0, 0.2f0),  # Orange glow
        g = 0.0f0
    )

    # Glass sphere containing the glowing medium
    glass = Hikari.GlassMaterial(index=1.3f0)
    glowing_glass = Hikari.MediumInterface(glass; inside=glowing_medium, outside=nothing)

    scene = create_cornell_box(sphere_material=glowing_glass)
    film, camera = create_camera()

    integrator = Hikari.VolPath(samples_per_pixel=64, max_depth=8)

    # gpu_film = Raycore.to_gpu(Array, film)
    # gpu_scene = Raycore.to_gpu(Array, scene)
    gpu_film = film
    gpu_scene = scene
    Hikari.clear!(gpu_film)

    println("Rendering glowing sphere...")
    @time integrator(gpu_scene, gpu_film, camera)

    Hikari.postprocess!(gpu_film; exposure=0.5f0, tonemap=:aces, gamma=1.2f0)
    img = Array(gpu_film.postprocess)
    println("Done!")

    return img
end

# =============================================================================
# Example 3: Clear glass sphere in foggy room
# =============================================================================
#
# This demonstrates the opposite case: fog fills the room, but the sphere
# contains clear air (vacuum).
#
# To achieve this, ALL room surfaces need MediumInterface(material, fog) to
# indicate fog on both sides, and the sphere uses inside=nothing, outside=fog.
#
# Note: For a true foggy room, we'd need an enclosing box with MediumInterface
# that starts rays in fog. This example shows the concept.

function example_foggy_room_clear_sphere()
    println("\nExample 3: Clear Glass Sphere in Foggy Room")
    println("=" ^ 50)

    # Light fog filling the room
    room_fog = Hikari.HomogeneousMedium(
        σ_a = Hikari.RGBSpectrum(0.001f0),
        σ_s = Hikari.RGBSpectrum(0.08f0),
        Le = Hikari.RGBSpectrum(0f0),
        g = 0.0f0
    )

    # Materials
    white = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.73f0))
    red = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.65f0, 0.05f0, 0.05f0))
    green = Hikari.MatteMaterial(Kd=Hikari.RGBSpectrum(0.12f0, 0.45f0, 0.15f0))

    # Walls embedded in fog (fog on both sides)
    white_in_fog = Hikari.MediumInterface(white, room_fog)
    red_in_fog = Hikari.MediumInterface(red, room_fog)
    green_in_fog = Hikari.MediumInterface(green, room_fog)

    # Glass sphere: vacuum inside, fog outside
    glass = Hikari.GlassMaterial(
        Kr = Hikari.RGBSpectrum(1f0),
        Kt = Hikari.RGBSpectrum(1f0),
        index = 1.5f0
    )
    clear_bubble = Hikari.MediumInterface(glass; inside=nothing, outside=room_fog)

    # Box dimensions
    box_size = 2f0
    half = box_size / 2

    # Create primitives with fog-embedded materials
    floor = tmesh(Rect3f(Vec3f(-half, 0, -half), Vec3f(box_size, 0.01f0, box_size)), white_in_fog)
    ceiling = tmesh(Rect3f(Vec3f(-half, box_size - 0.01f0, -half), Vec3f(box_size, 0.01f0, box_size)), white_in_fog)
    back_wall = tmesh(Rect3f(Vec3f(-half, 0, half - 0.01f0), Vec3f(box_size, box_size, 0.01f0)), white_in_fog)
    left_wall = tmesh(Rect3f(Vec3f(-half, 0, -half), Vec3f(0.01f0, box_size, box_size)), red_in_fog)
    right_wall = tmesh(Rect3f(Vec3f(half - 0.01f0, 0, -half), Vec3f(0.01f0, box_size, box_size)), green_in_fog)
    sphere = tmesh(Sphere(Point3f(0, 0.5f0, 0), 0.4f0), clear_bubble)

    mat_scene = Hikari.MaterialScene([floor, ceiling, back_wall, left_wall, right_wall, sphere])
    lights = (Hikari.PointLight(Point3f(0, 1.8f0, 0), Hikari.RGBSpectrum(15f0, 15f0, 15f0)),)
    scene = Hikari.Scene(lights, mat_scene)

    film, camera = create_camera()
    integrator = Hikari.VolPath(samples_per_pixel=128, max_depth=10)

    gpu_film = Raycore.to_gpu(Array, film)
    gpu_scene = Raycore.to_gpu(Array, scene)
    Hikari.clear!(gpu_film)

    println("Rendering foggy room with clear bubble...")
    println("(Fog on room surfaces via MediumInterface, vacuum inside sphere)")
    @time integrator(gpu_scene, gpu_film, camera)

    Hikari.postprocess!(gpu_film; exposure=1.0f0, tonemap=:aces, gamma=1.2f0)
    img = Array(gpu_film.postprocess)
    println("Done!")

    return img
end

# =============================================================================
# Example 4: Heterogeneous Cloud Sphere (Procedural Perlin Noise)
# =============================================================================
#
# A sphere filled with procedural cloud density using turbulent noise.
# This creates realistic wispy cloud structures with varying density.

"""
    perlin_noise_3d(x, y, z)

Simple 3D Perlin-like noise using sine functions for GPU compatibility.
"""
function perlin_noise_3d(x::Float32, y::Float32, z::Float32)
    # Hash function
    hash(n) = sin(n) * 43758.5453f0 - floor(sin(n) * 43758.5453f0)

    # Integer and fractional parts
    ix, iy, iz = floor(Int, x), floor(Int, y), floor(Int, z)
    fx, fy, fz = x - ix, y - iy, z - iz

    # Smoothstep
    ux = fx * fx * (3f0 - 2f0 * fx)
    uy = fy * fy * (3f0 - 2f0 * fy)
    uz = fz * fz * (3f0 - 2f0 * fz)

    # Hash corners
    n = Float32(ix + iy * 57 + iz * 113)
    a = hash(n)
    b = hash(n + 1f0)
    c = hash(n + 57f0)
    d = hash(n + 58f0)
    e = hash(n + 113f0)
    f = hash(n + 114f0)
    g = hash(n + 170f0)
    h = hash(n + 171f0)

    # Trilinear interpolation
    k0 = a
    k1 = b - a
    k2 = c - a
    k3 = e - a
    k4 = a - b - c + d
    k5 = a - c - e + g
    k6 = a - b - e + f
    k7 = -a + b + c - d + e - f - g + h

    return k0 + k1*ux + k2*uy + k3*uz + k4*ux*uy + k5*uy*uz + k6*uz*ux + k7*ux*uy*uz
end

"""
    fbm_3d(x, y, z; octaves=6, lacunarity=2.0, persistence=0.5)

Fractional Brownian Motion - layered noise for cloud-like turbulence.
"""
function fbm_3d(x::Float32, y::Float32, z::Float32;
                octaves::Int=6, lacunarity::Float32=2f0, persistence::Float32=0.5f0)
    value = 0f0
    amplitude = 1f0
    frequency = 1f0
    max_value = 0f0

    for _ in 1:octaves
        value += perlin_noise_3d(x * frequency, y * frequency, z * frequency) * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    end

    return value / max_value
end

"""
    generate_cloud_density(resolution; shape=:sphere, turbulence=0.5)

Generate a 3D density field for cloud rendering.
"""
function generate_cloud_density(resolution::Int; shape::Symbol=:sphere, turbulence::Float32=0.5f0)
    density = zeros(Float32, resolution, resolution, resolution)

    center = Float32(resolution) / 2f0
    radius = Float32(resolution) / 2f0 * 0.8f0  # 80% of half-size

    for k in 1:resolution, j in 1:resolution, i in 1:resolution
        # Normalized coordinates [-1, 1]
        x = (Float32(i) - center) / radius
        y = (Float32(j) - center) / radius
        z = (Float32(k) - center) / radius

        # Base shape
        if shape == :sphere
            dist = sqrt(x^2 + y^2 + z^2)
            base_density = max(0f0, 1f0 - dist)
        elseif shape == :box
            base_density = max(0f0, 1f0 - max(abs(x), abs(y), abs(z)))
        else
            base_density = 1f0
        end

        # Add turbulent displacement to coordinates
        noise_scale = 3f0
        nx = x * noise_scale
        ny = y * noise_scale
        nz = z * noise_scale

        # FBM turbulence
        turb = fbm_3d(nx, ny, nz; octaves=5, persistence=0.6f0)

        # Combine base shape with turbulence
        # The turbulence modulates both the density and slightly displaces the surface
        noise_val = turb * turbulence
        final_density = base_density * (0.5f0 + noise_val) + noise_val * 0.3f0

        # Apply a soft falloff at edges
        edge_falloff = smoothstep(1.2f0, 0.3f0, dist)
        final_density *= edge_falloff

        density[i, j, k] = clamp(final_density, 0f0, 1f0)
    end

    return density
end

# Smoothstep helper
smoothstep(edge0, edge1, x) = begin
    t = clamp((x - edge0) / (edge1 - edge0), 0f0, 1f0)
    t * t * (3f0 - 2f0 * t)
end

function example_heterogeneous_cloud()
    println("\nExample 4: Heterogeneous Cloud (Procedural Noise)")
    println("=" ^ 50)

    # Generate procedural cloud density
    println("Generating procedural cloud density (64³)...")
    resolution = 64
    density = generate_cloud_density(resolution; shape=:sphere, turbulence=2f0)
    println("  Max density: ", maximum(density))
    println("  Non-zero voxels: ", count(d -> d > 0.01f0, density))

    # Create GridMedium with the procedural density
    # The cloud will be placed in the sphere at (0, 0.5, 0) with radius 0.4
    # Map the density grid to a bounding box centered on the sphere
    sphere_center = Point3f(0f0, 0.5f0, 0f0)
    sphere_radius = 0.4f0
    cloud_bounds = Hikari.Bounds3(
        Point3f(sphere_center[1] - sphere_radius, sphere_center[2] - sphere_radius, sphere_center[3] - sphere_radius),
        Point3f(sphere_center[1] + sphere_radius, sphere_center[2] + sphere_radius, sphere_center[3] + sphere_radius)
    )

    cloud_medium = Hikari.GridMedium(
        density;
        σ_a = Hikari.RGBSpectrum(10f0),    # Absorption makes density visible
        σ_s = Hikari.RGBSpectrum(0.4f0),    # Moderate scattering
        g = 0.3f0,                           # Slight forward scattering
        bounds = cloud_bounds
    )

    # Transparent boundary (no refraction) to clearly see internal structure
    # Use index=1.0 to avoid glass distortion obscuring the cloud structure
    transparent = Hikari.GlassMaterial(
        Kr = Hikari.RGBSpectrum(0f0),
        Kt = Hikari.RGBSpectrum(1f0),
        index = 1.0f0
    )
    cloud_material = Hikari.MediumInterface(transparent; inside=cloud_medium, outside=nothing)

    # Create scene with cloud sphere in Cornell box
    scene = create_cornell_box(sphere_material=cloud_material)
    film, camera = create_camera()

    integrator = Hikari.VolPath(samples_per_pixel=128, max_depth=16)

    # Render
    Hikari.clear!(film)

    println("Rendering heterogeneous cloud in Cornell box...")
    println("(GridMedium with procedural turbulent noise inside glass sphere)")
    @time integrator(scene, film, camera)

    Hikari.postprocess!(film; exposure=1.0f0, tonemap=:aces, gamma=1.2f0)
    img = Array(film.postprocess)
    println("Done!")

    return img
end

# =============================================================================
# Run all examples
# =============================================================================

function run_all_examples()
    println("\n" * "=" ^ 60)
    println("Hikari Medium Examples")
    println("=" ^ 60 * "\n")

    results = Dict{String, Any}()

    results["foggy_glass"] = example_foggy_glass_sphere()
    results["glowing"] = example_glowing_sphere()
    results["fog_room_clear"] = example_foggy_room_clear_sphere()
    results["heterogeneous_cloud"] = example_heterogeneous_cloud()

    println("\n" * "=" ^ 60)
    println("All examples completed!")
    println("=" ^ 60)

    return results
end

# To run:
# include("examples/medium_example.jl")
# results = run_all_examples()
#
# Or run individual examples:
# img = example_foggy_glass_sphere()
# img = example_glowing_sphere()
# img = example_foggy_room_clear_sphere()
img = example_heterogeneous_cloud()
