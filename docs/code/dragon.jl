using Hikari, ImageShow, Colors, FileIO, LinearAlgebra, GeometryBasics
using TraceMakie, GLMakie

glass = Hikari.GlassMaterial(
    Hikari.ConstantTexture(Hikari.RGBSpectrum(1.0f0)),
    Hikari.ConstantTexture(Hikari.RGBSpectrum(1.0f0)),
    Hikari.ConstantTexture(0.0f0),
    Hikari.ConstantTexture(0.0f0),
    Hikari.ConstantTexture(1.25f0),
    true,
)
mirror = Hikari.MirrorMaterial(Hikari.ConstantTexture(Hikari.RGBSpectrum(1.0f0)))
plastic = Hikari.PlasticMaterial(
    Hikari.ConstantTexture(Hikari.RGBSpectrum(0.6399999857f0, 0.6399999857f0, 0.6399999857f0)),
    Hikari.ConstantTexture(Hikari.RGBSpectrum(0.1000000015f0, 0.1000000015f0, 0.1000000015f0)),
    Hikari.ConstantTexture(0.010408001f0),
    true,
)

dragon = load(joinpath(@__DIR__, "dragon.obj"))
scene = Scene(size=(1024, 1024); lights=[
    AmbientLight(RGBf(1, 1, 1)),
    PointLight(Vec3f(4, 4, 10), RGBf(150, 150, 150)),
    PointLight(Vec3f(-3, 10, 2.5), RGBf(60, 60, 60)),
    PointLight(Vec3f(0, 3, 0.5), RGBf(40, 40, 40))
])

cam3d!(scene)
points = dragon.position
mini, maxi = extrema(points)
sfactor = 1.0 / maximum(maxi .- mini)
normed = map(points) do p
    n = (p .- mini) .* sfactor
    Point3f(n[1], n[3], n[2])
end
dragon.position .= normed
fs = decompose(TriangleFace{Int}, dragon)
fsi = reinterpret(Int, fs)
bb = Rect3f(dragon.position[fsi])
xmin, ymin, zmin = minimum(bb)
mesh!(scene, dragon, material=glass)
mesh!(scene, Rect3f(Vec3f(-2, -2, zmin), Vec3f(4, 4, 0.01)), material=plastic)
update_cam!(scene, Vec3f(1.25, 1.08, 0.44), Vec3f(0.36, -0.22, -0.52), Vec3f(-0.29, -0.42, 0.85))

# Render with SPPM
img = TraceMakie.render_sppm(scene; search_radius=0.075f0, max_depth=5, iterations=500)

# Display and save the result
s = Scene(size=size(img))
image!(s, rotr90(img), space=:pixel)
display(s; scalefactor=1, px_per_unit=2)

image_01 = map(c -> mapc(x -> clamp(x, 0f0, 1f0), c), img)
save(joinpath(@__DIR__, "dragon.png"), image_01)
