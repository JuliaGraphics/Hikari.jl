using GeometryBasics, ImageShow
using LinearAlgebra
using Trace, FileIO, MeshIO
using TraceMakie, GLMakie

model = load(joinpath(@__DIR__, "..", "..", "src", "assets", "models", "caustic-glass.ply"))

glass = Trace.GlassMaterial(
    Trace.ConstantTexture(Trace.RGBSpectrum(1f0)),
    Trace.ConstantTexture(Trace.RGBSpectrum(1f0)),
    Trace.ConstantTexture(0f0),
    Trace.ConstantTexture(0f0),
    Trace.ConstantTexture(1.25f0),
    true,
)
plastic = Trace.PlasticMaterial(
    Trace.ConstantTexture(Trace.RGBSpectrum(0.6399999857f0, 0.6399999857f0, 0.6399999857f0)),
    Trace.ConstantTexture(Trace.RGBSpectrum(0.1000000015f0, 0.1000000015f0, 0.1000000015f0)),
    Trace.ConstantTexture(0.010408001f0),
    true,
)

scene = Scene(size=(1024, 1024); lights=[
    AmbientLight(RGBf(1, 1, 1)),
    PointLight(Vec3f(4, 4, 10), RGBf(150, 150, 150)),
    PointLight(Vec3f(-3, 10, 2.5), RGBf(60, 60, 60)),
    PointLight(Vec3f(0, 3, 0.5), RGBf(40, 40, 40))
])

cam3d!(scene)
mesh!(scene, model, material=glass)
mini, maxi = extrema(Rect3f(decompose(Point, model)))
floorrect = Rect3f(Vec3f(-10, mini[2], -10), Vec3f(20, -1, 20))
mesh!(scene, floorrect, material=plastic)
center!(scene)
update_cam!(scene, Vec3f(-1.6, 6.2, 0.2), Vec3f(-3.6, 2.5, 2.4), Vec3f(0, 1, 0))

# Render with SPPM
img = TraceMakie.render_sppm(scene; search_radius=0.075f0, max_depth=5, iterations=100)

# Save the result
using Colors
image_01 = map(c -> mapc(x -> clamp(x, 0f0, 1f0), c), img)
save(joinpath(@__DIR__, "caustic_glass.png"), image_01)
