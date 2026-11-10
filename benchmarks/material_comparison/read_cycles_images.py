"""Read saved EXRs with Blender's loader, outside all render measurements."""
import bpy
import json
import numpy as np
import pathlib
import sys

output = pathlib.Path(sys.argv[sys.argv.index("--") + 1])
stats_only = "--stats-only" in sys.argv[sys.argv.index("--") + 2:]
result = {}
for path in sorted(output.glob("*_cycles.exr")):
    image = bpy.data.images.load(str(path), check_existing=False)
    width, height = image.size
    data = np.asarray(image.pixels[:], dtype=np.float32).reshape(height, width, 4)[..., :3]
    assert np.isfinite(data).all() and data.max() > 0, path
    if not stats_only:
        data.tofile(path.with_suffix(".f32"))
    result[path.stem] = {"mean": float(data.mean()), "max": float(data.max()),
                         "width": width, "height": height}
    bpy.data.images.remove(image)
(output / "cycles_image_values.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
