"""Summarize the recorded four-way benchmark without rerunning a renderer."""
import csv
import json
import pathlib
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = pathlib.Path(sys.argv[1])
output = root / "main"
hikari = json.loads((output / "hikari_cycles.json").read_text())
original = {r["material"]: r["variants"][0] for r in json.loads((output / "original_hikari.json").read_text())["rows"]}
cycles = {r["material"]: r for r in json.loads((output / "blender_cycles.json").read_text())["rows"]}
rng = np.random.default_rng(20261104)
summary = {"resolution": hikari["resolution"], "samples": hikari["samples"],
           "device": hikari["device"], "max_depth": hikari["max_depth"],
           "baseline_warning": "Historical pbrt uses current dependencies and ray queries, not the former RT pipeline. Regression versus the original fast renderer remains unverified.",
           "rows": []}
columns = ["Blender Cycles", "Historical pbrt / current runtime", "Typed Cycles Principled", "Current Cycles Principled"]
figure, axes = plt.subplots(len(hikari["rows"]), 4, figsize=(12, 15))

for i, row in enumerate(hikari["rows"]):
    name = row["material"]
    typed, current = row["variants"]
    old = original.get(name)
    images = []
    # Blender's pixel origin and the Hikari framebuffer's indexing differ.
    # This one mapping is fixed from the matte case and used for every row.
    n = hikari["resolution"]
    blender = np.fromfile(output / (name + "_cycles.f32"), dtype=np.float32).reshape(n, n, 3)
    images.append(blender.transpose(1, 0, 2)[:, ::-1])
    for variant in ("original_hikari", "typed_cycles", "current_cycles"):
        path = output / f"{name}_{variant}.f32"
        images.append(np.fromfile(path, dtype=np.float32).reshape(n, n, 3) if path.exists() else None)
    for j, (axis, image) in enumerate(zip(axes[i], images)):
        if i == 0:
            axis.set_title(columns[j], fontsize=10)
        if j == 0:
            axis.set_ylabel(name.replace("_", " "), fontsize=10)
        if image is None:
            axis.text(0.5, 0.5, "No equivalent model", ha="center", va="center")
        else:
            assert np.isfinite(image).all() and image.max() > 0
            axis.imshow(np.clip(image, 0, 1) ** (1 / 2.2))
        axis.set_xticks([])
        axis.set_yticks([])
    values = [cycles[name]["median_seconds"], old["median_seconds"] if old else None,
              typed["median_seconds"], current["median_seconds"]]
    record = {"material": name, **{key: value * 1000 if value is not None else None for key, value in zip(
        ["blender_ms", "historical_pbrt_current_runtime_ms", "typed_cycles_ms", "current_cycles_ms"], values)}}
    record["current_vs_typed_percent"] = (values[3] / values[2] - 1) * 100
    record["current_vs_historical_pbrt_current_runtime_percent"] = (values[3] / values[1] - 1) * 100 if old else None
    record["max_typed_current_pixel_error"] = current["max_error_against_typed_cycles"]
    if old:
        c, o = np.asarray(current["frame_seconds"]), np.asarray(old["frame_seconds"])
        boot = (np.median(rng.choice(c, (3000, len(c))), axis=1) /
                np.median(rng.choice(o, (3000, len(o))), axis=1) - 1) * 100
        record["current_vs_historical_pbrt_current_runtime_bootstrap_95_percent"] = np.quantile(boot, [0.025, 0.975]).tolist()
    summary["rows"].append(record)

summary["fresh_session_startup"] = {}
for name, filename in (("current", "hikari_cycles"), ("typed", "hikari_cycles"), ("original", "original_hikari")):
    directory = root / "startup" / name
    path = directory / (filename + ".json")
    if path.exists():
        variant = json.loads(path.read_text())["rows"][0]["variants"][0]
        summary["fresh_session_startup"][name] = {
            **{key: variant[key] for key in ("setup_seconds", "first_render_seconds_existing_caches", "spirv_cache", "llvm_cache")},
            **json.loads((directory / "package_load.json").read_text())}

figure.suptitle("RTX 4000 Ada: same scene, 512×512, 64 samples\nPlastic uses different pbrt/Cycles models; gamma applied only for display", fontsize=12)
figure.tight_layout(rect=(0, 0, 1, 0.96))
figure.savefig(root / "images.png", dpi=150)
plt.close(figure)
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
with (root / "timings.csv").open("w", newline="") as stream:
    keys = ["material", "blender_ms", "historical_pbrt_current_runtime_ms", "typed_cycles_ms", "current_cycles_ms",
            "current_vs_historical_pbrt_current_runtime_percent", "current_vs_typed_percent"]
    writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(summary["rows"])
print(json.dumps(summary, indent=2))
