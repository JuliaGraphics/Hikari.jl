# MWE: isolate Lava SPIR-V bug triggered by dielectric f_val computation
# The computation `SpectralRadiance * (Float32 / max(Float32, 1f-6))` in a
# large kernel (3-material-type StaticMultiTypeSet) crashes both lavapipe and RADV.
#
# Run: julia --project=. dev/Hikari/test/mwe_lava_crash.jl

ENV["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"
using Hikari, Lava, GeometryBasics, FileIO
import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index, synchronize, allocate

backend = Mantle.defaultbackend()

# Minimal kernel that mimics the crashing pattern
@kernel function crash_kernel!(output, @Const(input), @Const(cos_vals), @Const(R_vals))
    i = @index(Global)
    sr = input[i]  # SpectralRadiance (4xFloat32)
    cos_i = cos_vals[i]
    R = R_vals[i]

    # This pattern crashes in the full evaluate_materials kernel:
    f_val = sr * (R / max(cos_i, 1f-6))

    output[i] = f_val
end

N = 256
sr_arr = KA.allocate(backend, Hikari.SpectralRadiance, N)
cos_arr = KA.allocate(backend, Float32, N)
R_arr = KA.allocate(backend, Float32, N)
out_arr = KA.allocate(backend, Hikari.SpectralRadiance, N)

# Fill with typical dielectric values
copyto!(sr_arr, fill(Hikari.SpectralRadiance(1f0), N))
copyto!(cos_arr, fill(0.95f0, N))
copyto!(R_arr, fill(0.04f0, N))

kernel! = crash_kernel!(backend)
kernel!(out_arr, sr_arr, cos_arr, R_arr; ndrange=N)
synchronize(backend)

result = Array(out_arr)
println("Result[1] = $(result[1])")
println("Expected: SpectralRadiance(0.04/0.95 ≈ $(0.04/0.95))")
println("PASS — the isolated kernel works. The bug requires the full evaluate_materials context.")
