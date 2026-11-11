"""Summarize compiler-call, phase, JIT and pipeline timing without adding overlapping clocks."""
import csv
import json
import pathlib
import sqlite3
import sys

root = pathlib.Path(sys.argv[1])
summary = {"scope": "Application object/pipeline caches empty initially, then reused after a fresh process. Driver-private caches were not cleared.",
           "baseline_warning": "Historical pbrt runs on current dependencies with ray queries, not its original RT pipeline.",
           "clock_warning": "Julia JIT, compiler phases and compiler-call wall times overlap; do not add them.",
           "case_order": ["matte", "rough_glass", "rough_metal", "plastic", "layered_principled"],
           "rows": []}
for implementation, folder in (("original_hikari", "original"), ("typed_cycles", "typed"), ("current_cycles", "current_v2")):
    for state in ("cold", "restart"):
        directory = root / folder
        data = json.loads((directory / (state + ".json")).read_text())
        for row in data["rows"]:
            first = row["first_render"]
            shade = [v for v in first["compiler_lookups"] if "vp_shade_" in v["signature"]]
            record = {"material": row["material"], "implementation": implementation, "cache_state": state,
                      "shading_compiler_call_seconds": sum(v["seconds"] for v in shade),
                      "shading_spirv_misses": sum(v["misses"] for v in shade),
                      "all_compiler_calls_seconds": first["compiler_lookup_wall_seconds"],
                      "first_frame_seconds": first["wall_seconds"],
                      "host_jit_seconds": first["julia_compile_seconds"],
                      "host_recompile_seconds": first["julia_recompile_seconds"],
                      "pipeline_creation_seconds": first["pipeline_creation_wall_seconds"],
                      "setup_seconds": row["setup"]["wall_seconds"],
                      "spirv_cache_misses": first["spirv_cache"]["misses"],
                      "llvm_and_spirv_phase_seconds": {v["label"]: v["seconds"] for v in first["phase_table"] if v["group"] == "stage"},
                      "pipeline_cache_loaded_bytes": data["pipeline_cache_loaded_bytes"],
                      "source": str(directory / (state + ".json"))}
            summary["rows"].append(record)

def optix_trace(path):
    c = sqlite3.connect(path)
    result = {}
    for name in ("optixModuleCreateWithTasks", "optixTaskExecute", "optixPipelineCreate", "optixProgramGroupCreate"):
        rows = c.execute("select n.start,n.end from NVTX_EVENTS n left join StringIds s on n.textId=s.id "
                         "where coalesce(n.text,s.value)=? and n.end is not null order by n.start", (name,)).fetchall()
        if not rows:
            continue
        merged = []
        for start, end in rows:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        result[name] = {"calls": len(rows), "summed_thread_seconds": sum(b-a for a,b in rows)/1e9,
                        "wall_union_seconds": sum(b-a for a,b in merged)/1e9,
                        "wall_span_seconds": (max(b for a,b in rows)-min(a for a,b in rows))/1e9}
    result["cuda_module_load"] = c.execute("select count(*),sum(a.end-a.start)/1e9 from CUPTI_ACTIVITY_KIND_RUNTIME a "
                                          "join StringIds s on a.nameId=s.id where s.value='cuModuleLoadData'").fetchone()
    c.close()
    return result

summary["blender"] = {"scope": "OptiX compilation tasks and pipeline creation traced by Nsight. Ships precompiled device code; these exclude per-material CPU/SVM compilation, whose duration is not exposed by this GPU build."}
for state, filename in (("cold", "cycles.sqlite"), ("restart", "cycles_restart.sqlite")):
    path = root / "blender" / filename
    if path.exists():
        summary["blender"][state] = optix_trace(path)
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
keys = ["material", "implementation", "cache_state", "shading_compiler_call_seconds", "shading_spirv_misses",
        "all_compiler_calls_seconds", "first_frame_seconds", "host_jit_seconds", "pipeline_creation_seconds", "spirv_cache_misses"]
with (root / "timings.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(summary["rows"])
for row in summary["rows"]:
    print(row["cache_state"], row["material"], row["implementation"],
          "shade", round(row["shading_compiler_call_seconds"], 4),
          "all", round(row["all_compiler_calls_seconds"], 4), "first", round(row["first_frame_seconds"], 3))
print("Blender native compilation:", json.dumps(summary["blender"], indent=2))
