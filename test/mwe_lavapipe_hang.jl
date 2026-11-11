# MWE: lavapipe hangs when vp_compute_shading_tangents uses double-cross pattern
#
# The double-cross pattern (matching pbrt-v4 exactly):
#   ts = cross(ns, ss); ss = cross(ts, ns)
# causes lavapipe's LLVM JIT to hang on complex conductor material kernels.
#
# The Gram-Schmidt pattern (mathematically equivalent):
#   dpdus = dpdu - ns * dot(ns, dpdu); normalize; dpdvs = cross(ns, dpdus)
# works fine.
#
# To reproduce:
#   1. Apply the double-cross version of vp_compute_shading_tangents in intersection.jl
#   2. Run this script
#   3. Observe hang on mat_conductor_mirror_light_point (never completes)
#
# With the Gram-Schmidt version, both scenes complete in ~5s each.

using Hikari, Lava, GeometryBasics, FileIO
import KernelAbstractions as KA

ENV["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"
backend = Mantle.defaultbackend()

# This works (diffuse doesn't use shading tangents for specular reflection)
println("Rendering diffuse...")
Hikari.render_pbrt(
    joinpath(@__DIR__, "pbrt/scenes/mat_diffuse_light_point.pbrt");
    backend=backend, samples=1)
println("Diffuse OK")

# This hangs with double-cross, works with Gram-Schmidt
println("Rendering conductor_mirror (will hang if double-cross is active)...")
Hikari.render_pbrt(
    joinpath(@__DIR__, "pbrt/scenes/mat_conductor_mirror_light_point.pbrt");
    backend=backend, samples=1)
println("Conductor mirror OK")
