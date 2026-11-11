# MWE: dielectric + area light segfault
# Reproduces a lavapipe crash when rendering a dielectric sphere
# with an area light after fixing the Dielectric BSDF f/pdf values.
#
# The crash does NOT happen with:
# - dielectric + point light
# - dielectric + ambient light
# - diffuse + area light
#
# Run: julia --project=/sim/Programmieren/VulkanDev dev/Hikari/test/mwe_dielectric_area_crash.jl

using Hikari, Lava, GeometryBasics
import KernelAbstractions as KA
import Adapt

ENV["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"
backend = Mantle.defaultbackend()

scene_file = joinpath(@__DIR__, "pbrt/scenes/mat_dielectric_light_area.pbrt")
r = Hikari.load_pbrt(scene_file; backend=backend, samples=1, max_depth=2)

println("Scene loaded. Rendering...")
vp = Hikari.VolPath(samples=1, max_depth=2)
vp(r.scene, r.film, r.camera)
Hikari.postprocess!(r.film)
println("OK — no crash")
