"""Compare matching scene captures after read_cycles_images.py has loaded their EXRs."""
import json
import pathlib
import sys

import numpy as np


def image(path, resolution):
    data = np.fromfile(path, dtype=np.float32).reshape(resolution, resolution, 3)
    assert np.isfinite(data).all() and data.max() > 0, path
    return data


def object_regions(spec, resolution):
    eye = np.asarray(spec["camera"]["position"], dtype=float)
    forward = np.asarray(spec["camera"]["lookat"], dtype=float) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    tangent = np.tan(np.deg2rad(spec["camera"]["fov"]) / 2)
    yy, xx = np.mgrid[:resolution, :resolution]
    for position in spec.get("instances", [[0, 0, 0]]):
        relative = np.asarray(position) - eye
        depth = np.dot(relative, forward)
        cx = (0.5 + np.dot(relative, right) / (2 * tangent * depth)) * resolution
        cy = (0.5 - np.dot(relative, up) / (2 * tangent * depth)) * resolution
        # An inner disk avoids silhouettes and background dominating this check.
        radius = 0.85 * spec["sphere_radius"] * resolution / (2 * tangent * depth)
        yield (xx - cx) ** 2 + (yy - cy) ** 2 < radius ** 2


def compare(spec_path, output):
    spec = json.loads(spec_path.read_text())
    current = json.loads((output / "hikari_cycles.json").read_text())
    cycles = json.loads((output / "blender_cycles.json").read_text())
    n = current["resolution"]
    assert n == cycles["resolution"]
    assert current["samples"] == cycles["samples"] == spec["samples"]
    assert current["max_depth"] == cycles["max_depth"] == spec["max_depth"]
    assert current.get("environment_strength", 0) == cycles.get("environment_strength", 0) == spec.get("environment_strength", 0)
    references = {row["material"]: row for row in cycles["rows"]}
    settings = {row["name"]: row for row in spec["cases"]}
    assert set(references) == set(settings) == {row["material"] for row in current["rows"]}
    rows = []
    for row in current["rows"]:
        name = row["material"]
        candidate = next(v for v in row["variants"] if v["implementation"] == "current_cycles")
        reference = references[name]
        # The mapping is fixed by the original benchmark, not fitted per image.
        hikari = image(output / f"{name}_current_cycles.f32", n)[:, ::-1].transpose(1, 0, 2)[::-1]
        blender = image(output / f"{name}_cycles.f32", n)[::-1]
        mean_error = float(hikari.mean() / blender.mean() - 1)
        regions = []
        names = settings[name].get("materials", [settings[name]])
        masks = list(object_regions(spec, n))
        assert len(names) == len(masks)
        for material, mask in zip(names, masks):
            assert mask.any()
            regions.append({"material": material["name"], "pixels": int(mask.sum()),
                            "hikari_mean": float(hikari[mask].mean()),
                            "cycles_mean": float(blender[mask].mean()),
                            "relative_mean_error": float(hikari[mask].mean() / blender[mask].mean() - 1)})
        ratio = candidate["median_seconds"] / reference["median_seconds"]
        rows.append({"material": name, "hikari_ms": candidate["median_seconds"] * 1000,
                     "cycles_ms": reference["median_seconds"] * 1000, "current_over_cycles": ratio,
                     "within_ten_percent": ratio <= 1.1, "relative_image_mean_error": mean_error,
                     "regions": regions, "shade_signatures": len(candidate["shade_signatures"])})
    result = {"resolution": n, "samples": current["samples"], "max_depth": current["max_depth"],
              "device": current["device"], "rows": rows}
    (output / "scene_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    compare(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
