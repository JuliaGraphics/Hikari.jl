# MWE: dump SPIR-V for the crashing 3-material kernel
# Run: julia --project=. dev/Hikari/test/mwe_dump_spirv.jl

ENV["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"

using Hikari, Lava, GeometryBasics, FileIO

# Monkey-patch Lava to dump SPIR-V before creating pipeline
# `_validate_spirv` was renamed to `validate_spirv` and is the compiler's.
const _original_validate = Lava.validate_spirv
let dump_counter = Ref(0)
    global function Lava.validate_spirv(spirv_bytes, label, source_map)
        dump_counter[] += 1
        path = "/tmp/spirv_dump_$(dump_counter[]).spv"
        write(path, spirv_bytes)
        println("  [DUMP] Wrote $(length(spirv_bytes)) bytes to $path (label: $label)")
        flush(stdout)
        return _original_validate(spirv_bytes, label, source_map)
    end
end

backend = Mantle.defaultbackend()
println("Backend ready")
flush(stdout)

# Load and render - should dump all compiled kernels
sf = joinpath(@__DIR__, "pbrt/scenes/mat_dielectric_light_area.pbrt")
r = Hikari.load_pbrt(sf; backend=backend, samples=1, max_depth=1)
println("Scene loaded, rendering max_depth=1 (should succeed)...")
flush(stdout)

vp = Hikari.VolPath(samples=1, max_depth=1)
vp(r.scene, r.film, r.camera)
println("max_depth=1 OK - all kernels compiled and dumped")
flush(stdout)

# Now list the dumps and validate each
println("\nValidating all dumped SPIR-V files:")
flush(stdout)
for f in sort(readdir("/tmp"; join=true))
    if startswith(basename(f), "spirv_dump_") && endswith(f, ".spv")
        sz = filesize(f)
        # Run spirv-val
        val_cmd = Lava.SPIRV_Tools_jll.spirv_val()
        p = run(pipeline(`$val_cmd --target-env vulkan1.3 --scalar-block-layout $f`; stderr=devnull); wait=false)
        wait(p)
        status = p.exitcode == 0 ? "VALID" : "INVALID"
        println("  $f ($sz bytes): $status")
        flush(stdout)
    end
end
