"""Blender side of the four-way material benchmark; no image writes in timed renders."""
import bpy
import json
import math
import mathutils
import pathlib
import statistics
import sys
import time

args = sys.argv[sys.argv.index("--") + 1:]
spec_path, output = pathlib.Path(args[0]), pathlib.Path(args[1])
backend = args[2].upper() if len(args) > 2 else "OPTIX"
device_name = args[3] if len(args) > 3 else "RTX 4000 Ada"
assert backend in {"OPTIX", "HIP"}, backend
spec = json.loads(spec_path.read_text())
output.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
preferences = bpy.context.preferences.addons["cycles"].preferences
preferences.compute_device_type = backend
if backend == "HIP":
    preferences.use_hiprt = True
preferences.get_devices_for_type(backend)
for device in preferences.devices:
    device.use = device.type == backend and device_name in device.name
devices = [(d.name, d.type) for d in preferences.devices if d.use]
assert len(devices) == 1, devices
scene.cycles.device = "GPU"
scene.cycles.samples = spec["samples"]
scene.cycles.use_adaptive_sampling = False
scene.cycles.use_denoising = False
scene.cycles.max_bounces = spec["max_depth"]
scene.cycles.min_light_bounces = 3
scene.cycles.diffuse_bounces = scene.cycles.glossy_bounces = spec["max_depth"]
scene.cycles.transmission_bounces = spec["max_depth"]
scene.cycles.volume_bounces = 0
scene.cycles.sample_clamp_direct = scene.cycles.sample_clamp_indirect = 0
scene.cycles.blur_glossy = 0
scene.cycles.pixel_filter_type = "BOX"
scene.cycles.filter_width = 1
scene.render.use_persistent_data = True
scene.render.resolution_x = scene.render.resolution_y = spec["resolution"]
scene.render.resolution_percentage = 100
scene.world = bpy.data.worlds.new("Uniform world")
scene.world.use_nodes = True
background = scene.world.node_tree.nodes["Background"]
background.inputs["Color"].default_value = (1, 1, 1, 1)
background.inputs["Strength"].default_value = spec.get("environment_strength", 0)
bpy.ops.wm.obj_import(filepath=str(output / "sphere.obj"), forward_axis="Y", up_axis="Z")
sphere = bpy.context.selected_objects[0]
camera_data = bpy.data.cameras.new("Camera")
camera_data.angle = math.radians(spec["camera"]["fov"])
camera = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera)
camera.location = spec["camera"]["position"]
direction = mathutils.Vector(spec["camera"]["lookat"]) - camera.location
camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
scene.camera = camera
light_data = bpy.data.lights.new("Point", "POINT")
light_data.energy = 4 * math.pi * spec["point_light"]["intensity"]
light_data.shadow_soft_size = 0
light = bpy.data.objects.new("Point", light_data)
scene.collection.objects.link(light)
light.location = spec["point_light"]["position"]
result = {"blender": bpy.app.version_string, "device": devices, "resolution": spec["resolution"],
          "backend": backend, "hardware_raytracing": backend == "OPTIX" or preferences.use_hiprt,
          "environment_strength": spec.get("environment_strength", 0),
          "samples": spec["samples"], "max_depth": spec["max_depth"], "rows": []}

def shader_node(tree, row):
    if "mix" in row:
        mix = row["mix"]
        node = tree.nodes.new("ShaderNodeMixShader")
        node.inputs[0].default_value = mix["factor"]
        tree.links.new(shader_node(tree, mix["a"]), node.inputs[1])
        tree.links.new(shader_node(tree, mix["b"]), node.inputs[2])
        return node.outputs[0]
    node = tree.nodes.new("ShaderNodeBsdfPrincipled")
    node.distribution = spec["distribution"].upper()
    for key, value in row["principled"].items():
        name = " ".join("IOR" if s == "ior" else s.capitalize() for s in key.split("_"))
        socket = next(s for s in node.inputs if s.name == name)
        socket.default_value = tuple(value) + (1,) if isinstance(value, list) else value
    return node.outputs[0]


def principled_material(row):
    material = bpy.data.materials.new(row["name"])
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(shader_node(tree, row), output.inputs["Surface"])
    return material

objects = [sphere]
for row in spec["cases"]:
    print("BENCHMARK_BEGIN", row["name"], flush=True)
    for previous in objects[1:]:
        bpy.data.objects.remove(previous, do_unlink=True)
    objects = [sphere]
    if "materials" in row:
        assert len(row["materials"]) == len(spec["instances"])
        for index, (settings, position) in enumerate(zip(row["materials"], spec["instances"])):
            obj = sphere if index == 0 else sphere.copy()
            if index:
                obj.data = sphere.data.copy()
                scene.collection.objects.link(obj)
                objects.append(obj)
            obj.location = position
            obj.data.materials.clear()
            obj.data.materials.append(principled_material(settings))
    else:
        sphere.location = (0, 0, 0)
        sphere.data.materials.clear()
        sphere.data.materials.append(principled_material(row))
    print("FIRST_RENDER_BEGIN", row["name"], flush=True)
    start = time.perf_counter()
    bpy.ops.render.render(write_still=False)
    first = time.perf_counter() - start
    print("FIRST_RENDER_END", row["name"], first, flush=True)
    for _ in range(spec["warmups"]):
        bpy.ops.render.render(write_still=False)
    times = []
    for _ in range(spec["repeats"]):
        start = time.perf_counter()
        bpy.ops.render.render(write_still=False)
        times.append(time.perf_counter() - start)
    image = bpy.data.images["Render Result"]
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    image.save_render(str(output / (row["name"] + "_cycles.exr")), scene=scene)
    record = {"material": row["name"], "first_render_seconds_existing_caches": first,
              "median_seconds": statistics.median(times), "frame_seconds": times}
    result["rows"].append(record)
    (output / "blender_cycles.json").write_text(json.dumps(result, indent=2))
    print("BENCHMARK", json.dumps(record), flush=True)
