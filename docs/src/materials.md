## Materials Showcase

This example demonstrates all the different material types available in Hikari, arranged in a scene to clearly show their properties.

### Available Materials

Hikari supports several physically-based materials:

- **MatteMaterial**: Diffuse surfaces with optional roughness (Oren-Nayar model)
- **MirrorMaterial**: Perfect specular reflection (can simulate polished metals)
- **GlassMaterial**: Transparent material with refraction and optional roughness
- **PlasticMaterial**: Combination of diffuse and glossy specular reflection

```@example materials
using GeometryBasics
using Hikari
using Raycore
using FileIO
using ImageShow

# Helper to convert primitives to Hikari meshes
function tmesh(prim, material)
    prim = prim isa Sphere ? Tesselation(prim, 64) : prim
    mesh = normal_mesh(prim)
    m = Raycore.TriangleMesh(mesh)
    return Hikari.GeometricPrimitive(m, material)
end

# Helper to create a sphere resting on the ground
LowSphere(radius, x, z) = Sphere(Point3f(x, radius, z), radius)

# ============================================
# Define all material types
# ============================================

# 1. MATTE MATERIALS - Diffuse reflection
# Smooth diffuse (Lambertian) - terracotta color
matte_smooth = Hikari.MatteMaterial(
    Hikari.ConstantTexture(Hikari.RGBSpectrum(0.8f0, 0.4f0, 0.3f0)),
    Hikari.ConstantTexture(0f0),  # σ=0 means perfectly smooth Lambertian
)
# Rough diffuse (Oren-Nayar) - chalk/clay appearance
matte_rough = Hikari.MatteMaterial(
    Hikari.ConstantTexture(Hikari.RGBSpectrum(0.85f0, 0.85f0, 0.8f0)),
    Hikari.ConstantTexture(60f0),  # σ=60° high roughness for chalky look
)

# 2. MIRROR/METAL MATERIALS - Perfect specular reflection
# Silver mirror
metal_silver = Hikari.MirrorMaterial(
    Hikari.ConstantTexture(Hikari.RGBSpectrum(0.95f0, 0.93f0, 0.88f0)),
)
# Gold metal - warm golden tint
metal_gold = Hikari.MirrorMaterial(
    Hikari.ConstantTexture(Hikari.RGBSpectrum(1.0f0, 0.78f0, 0.34f0)),
)
# Copper metal - reddish-orange metallic
metal_copper = Hikari.MirrorMaterial(
    Hikari.ConstantTexture(Hikari.RGBSpectrum(0.95f0, 0.64f0, 0.54f0)),
)

# 3. GLASS MATERIALS - Refraction and transparency
# Clear glass
glass_clear = Hikari.GlassMaterial(
    Hikari.ConstantTexture(Hikari.RGBSpectrum(1f0)),      # Reflectance
    Hikari.ConstantTexture(Hikari.RGBSpectrum(1f0)),      # Transmittance
    Hikari.ConstantTexture(0f0),                          # u_roughness
    Hikari.ConstantTexture(0f0),                          # v_roughness
    Hikari.ConstantTexture(1.5f0),                        # IOR (glass ≈ 1.5)
    true,
)
# Frosted glass - rough transmission for diffuse transparency
glass_frosted = Hikari.GlassMaterial(
    Hikari.ConstantTexture(Hikari.RGBSpectrum(1f0)),
    Hikari.ConstantTexture(Hikari.RGBSpectrum(1f0)),
    Hikari.ConstantTexture(0.15f0),                       # Moderate roughness for frosted look
    Hikari.ConstantTexture(0.15f0),
    Hikari.ConstantTexture(1.5f0),
    true,
)

# 4. PLASTIC MATERIALS - Diffuse + glossy specular
# Shiny red plastic
plastic_shiny = Hikari.PlasticMaterial(
    Hikari.ConstantTexture(Hikari.RGBSpectrum(0.8f0, 0.1f0, 0.1f0)),  # Diffuse red
    Hikari.ConstantTexture(Hikari.RGBSpectrum(0.4f0)),                 # Specular
    Hikari.ConstantTexture(0.02f0),                                    # Low roughness (shiny)
    true,
)
# Matte blue plastic
plastic_matte = Hikari.PlasticMaterial(
    Hikari.ConstantTexture(Hikari.RGBSpectrum(0.15f0, 0.3f0, 0.7f0)),
    Hikari.ConstantTexture(Hikari.RGBSpectrum(0.15f0)),
    Hikari.ConstantTexture(0.25f0),                                    # Higher roughness
    true,
)

# Floor material
floor_mat = Hikari.MatteMaterial(
    Hikari.ConstantTexture(Hikari.RGBSpectrum(0.85f0)),
    Hikari.ConstantTexture(0f0),
)

# ============================================
# Create scene with 3x3 grid of spheres
# ============================================
# Back row:   Matte Smooth | Matte Rough  | Glass Clear
# Middle row: Metal Silver | Metal Gold   | Metal Copper
# Front row:  Frosted Glass| Plastic Shiny| Plastic Matte

r = 0.38f0  # sphere radius
sx, sz = 1.0f0, 1.1f0  # spacing

# Back row (z = -sz)
s1 = tmesh(LowSphere(r, -sx, -sz), matte_smooth)
s2 = tmesh(LowSphere(r, 0f0, -sz), matte_rough)
s3 = tmesh(LowSphere(r, sx, -sz), glass_clear)

# Middle row (z = 0)
s4 = tmesh(LowSphere(r, -sx, 0f0), metal_silver)
s5 = tmesh(LowSphere(r, 0f0, 0f0), metal_gold)
s6 = tmesh(LowSphere(r, sx, 0f0), metal_copper)

# Front row (z = sz)
s7 = tmesh(LowSphere(r, -sx, sz), glass_frosted)
s8 = tmesh(LowSphere(r, 0f0, sz), plastic_shiny)
s9 = tmesh(LowSphere(r, sx, sz), plastic_matte)

# Ground plane
ground = tmesh(Rect3f(Vec3f(-3, 0, -3), Vec3f(6, 0.01, 6)), floor_mat)

primitives = [s1, s2, s3, s4, s5, s6, s7, s8, s9, ground]
mat_scene = Hikari.MaterialScene(primitives)

# HDRI Environment lighting with importance sampling
# world_radius should be ~2-3x the scene extent for efficient photon mapping
env_light = Hikari.EnvironmentLight(
    joinpath(@__DIR__, "..", "..", "..", "RadeonProRender", "assets", "studio026.exr");
    scale=Hikari.RGBSpectrum(0.5f0),  # Dimmed to let spotlights be visible
    rotation=0f0,
)

# Add spotlights for dramatic lighting and to show specular highlights
# Key light - warm spotlight from front-right
key_light = Hikari.SpotLight(
    Point3f(3f0, 4f0, 3f0),   # position
    Point3f(0f0, 0f0, 0f0),   # target (scene center)
    Hikari.RGBSpectrum(15f0, 14f0, 12f0),  # warm white
    35f0, 30f0                 # cone angles
)

# Fill light - cooler spotlight from front-left
fill_light = Hikari.SpotLight(
    Point3f(-3f0, 3f0, 2f0),
    Point3f(0f0, 0.2f0, 0f0),
    Hikari.RGBSpectrum(6f0, 7f0, 8f0),  # cool white
    40f0, 35f0
)

# Rim light - from behind to create edge highlights on metals
rim_light = Hikari.SpotLight(
    Point3f(0f0, 3f0, -3f0),
    Point3f(0f0, 0.3f0, 0f0),
    Hikari.RGBSpectrum(8f0),
    45f0, 40f0
)

scene = Hikari.Scene([env_light, key_light, fill_light, rim_light], mat_scene)

# Camera setup
resolution = Point2f(1024)
filter = Hikari.LanczosSincFilter(Point2f(1f0), 3f0)
film = Hikari.Film(
    resolution,
    Hikari.Bounds2(Point2f(0f0), Point2f(1f0)),
    filter, 1f0, 1f0,
)
screen = Hikari.Bounds2(Point2f(-1f0), Point2f(1f0))
camera = Hikari.PerspectiveCamera(
    Hikari.look_at(Point3f(0, 3.5, 5), Point3f(0, 0.2, 0), Vec3f(0, 1, 0)),
    screen, 0f0, 1f0, 0f0, 1f6, 40f0, film,
)

# Render with Wavefront path tracer
integrator = Hikari.WavefrontIntegrator(camera; max_depth=5, samples_per_pixel=16)
integrator(scene, film)
```

### Material Properties Reference

| Material | Key Parameters | Best For |
|----------|---------------|----------|<>
| **MatteMaterial** | `Kd` (color), `σ` (roughness 0-90°) | Diffuse surfaces: walls, cloth, paper, chalk |
| **MirrorMaterial** | `Kr` (reflectance color) | Polished metals: silver, gold, copper, chrome |
| **GlassMaterial** | `Kr`, `Kt`, roughness, `index` (IOR) | Glass, water, gems, ice, frosted surfaces |
| **PlasticMaterial** | `Kd`, `Ks`, `roughness` | Plastic, painted surfaces, ceramics |

### Metal Colors Reference

Common reflectance values for metals using MirrorMaterial:
- **Silver**: RGB(0.95, 0.93, 0.88)
- **Gold**: RGB(1.0, 0.78, 0.34)
- **Copper**: RGB(0.95, 0.64, 0.54)
- **Aluminum**: RGB(0.91, 0.92, 0.92)
- **Iron**: RGB(0.56, 0.57, 0.58)

### Index of Refraction (IOR) Reference

Common IOR values for GlassMaterial:
- Air: 1.0
- Water: 1.33
- Glass: 1.5
- Crystal: 1.6-2.0
- Diamond: 2.42
