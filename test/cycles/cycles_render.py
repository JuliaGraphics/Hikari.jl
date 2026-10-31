# Builds a scene.json of export_scene.jl in Blender and renders it with Cycles on the GPU, a linear EXR, timed.
#   blender -b -P cycles_render.py -- scene.json out.exr device runs
import bpy, json, sys, time, math, mathutils

args = sys.argv[sys.argv.index("--") + 1:]
scene_path, out_path, device = args[0], args[1], args[2]
runs = int(args[3]) if len(args) > 3 else 3
spec = json.load(open(scene_path))

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = device
prefs.get_devices()
for d in prefs.devices:
    d.use = d.type == device
print("cycles devices:", [(d.name, d.type, d.use) for d in prefs.devices])
cy = scene.cycles
cy.device = "GPU"
cy.samples = spec["samples"]
cy.use_adaptive_sampling = False
cy.use_denoising = False
cy.max_bounces = spec["max_bounces"]
cy.diffuse_bounces = cy.glossy_bounces = cy.transmission_bounces = spec["max_bounces"]
cy.transparent_max_bounces = spec["max_bounces"]
cy.volume_bounces = 0
cy.sample_clamp_direct = 0.0
cy.sample_clamp_indirect = 0.0
cy.blur_glossy = 0.0
cy.caustics_reflective = cy.caustics_refractive = True
cy.use_light_tree = True
scene.render.use_persistent_data = True
res = spec["resolution"]
scene.render.resolution_x = scene.render.resolution_y = res
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.render.image_settings.color_mode = "RGB"

world = bpy.data.worlds.new("world")
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
w = spec["world"]
bg.inputs["Color"].default_value = (w, w, w, 1.0)
bg.inputs["Strength"].default_value = 1.0
scene.world = world

def socket_name(key):
    return " ".join("IOR" if part == "ior" else part.capitalize() for part in key.split("_"))

def principled(inputs):
    mat = bpy.data.materials.new("principled")
    mat.use_nodes = True
    node = mat.node_tree.nodes["Principled BSDF"]
    node.subsurface_method = "RANDOM_WALK"
    for key, value in inputs.items():
        # By iteration: a socket the subsurface method leaves unavailable (Subsurface IOR under
        # Random Walk) is not found by name.
        sock = next(s for s in node.inputs if s.name == socket_name(key))
        if isinstance(value, list):
            value = tuple(value) + ((1.0,) if len(sock.default_value) == 4 else ())
        sock.default_value = value
    if inputs.get("anisotropic", 0) != 0:
        # Hikari's default tangent: radial around the world's Z axis, (-y, x, 0) of the position.
        nt = mat.node_tree
        geom = nt.nodes.new("ShaderNodeNewGeometry")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        neg = nt.nodes.new("ShaderNodeMath"); neg.operation = "MULTIPLY"; neg.inputs[1].default_value = -1.0
        comb = nt.nodes.new("ShaderNodeCombineXYZ")
        nt.links.new(geom.outputs["Position"], sep.inputs[0])
        nt.links.new(sep.outputs["Y"], neg.inputs[0])
        nt.links.new(neg.outputs[0], comb.inputs["X"])
        nt.links.new(sep.outputs["X"], comb.inputs["Y"])
        nt.links.new(comb.outputs[0], node.inputs["Tangent"])
    return mat

def diffuse(color):
    mat = bpy.data.materials.new("diffuse")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.remove(nt.nodes["Principled BSDF"])
    node = nt.nodes.new("ShaderNodeBsdfDiffuse")
    node.inputs["Color"].default_value = tuple(color) + (1.0,)
    nt.links.new(node.outputs[0], nt.nodes["Material Output"].inputs["Surface"])
    return mat

def emission(strength):
    mat = bpy.data.materials.new("emission")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.remove(nt.nodes["Principled BSDF"])
    node = nt.nodes.new("ShaderNodeEmission")
    node.inputs["Strength"].default_value = strength
    nt.links.new(node.outputs[0], nt.nodes["Material Output"].inputs["Surface"])
    return mat

def shader(nt, m):
    """The output socket of a node tree for a shader spec: a BSDF node, or a mix or sum of two."""
    if m["type"] == "bsdf":
        node = nt.nodes.new(m["node"])
        for key, value in m.get("inputs", {}).items():
            sock = next(s for s in node.inputs if s.name == key)
            if isinstance(value, list):
                value = tuple(value) + ((1.0,) if len(sock.default_value) == 4 else ())
            sock.default_value = value
        return node.outputs[0]
    if m["type"] == "mix":
        node = nt.nodes.new("ShaderNodeMixShader")
        node.inputs["Fac"].default_value = m["factor"]
        nt.links.new(shader(nt, m["a"]), node.inputs[1])
        nt.links.new(shader(nt, m["b"]), node.inputs[2])
        return node.outputs[0]
    if m["type"] == "add":
        node = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(shader(nt, m["a"]), node.inputs[0])
        nt.links.new(shader(nt, m["b"]), node.inputs[1])
        return node.outputs[0]
    raise ValueError(m["type"])

def node_material(m):
    mat = bpy.data.materials.new("nodes")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.remove(nt.nodes["Principled BSDF"])
    nt.links.new(shader(nt, m), nt.nodes["Material Output"].inputs["Surface"])
    return mat

def material(m):
    if m["type"] == "principled":
        return principled(m["inputs"])
    if m["type"] == "diffuse":
        return diffuse(m["color"])
    if m["type"] == "emission":
        return emission(m["strength"])
    return node_material(m)

for obj in spec["objects"]:
    bpy.ops.wm.obj_import(filepath=obj["obj"], forward_axis="Y", up_axis="Z")
    o = bpy.context.selected_objects[0]
    m = obj["material"]
    mat = material(m)
    o.data.materials.clear()
    o.data.materials.append(mat)

cam = spec["camera"]
cam_data = bpy.data.cameras.new("camera")
cam_data.sensor_fit = "AUTO"
cam_data.angle = math.radians(cam["fov"])
cam_obj = bpy.data.objects.new("camera", cam_data)
scene.collection.objects.link(cam_obj)
pos, look, up = (mathutils.Vector(cam[k]) for k in ("position", "lookat", "up"))
cam_obj.location = pos
forward = (look - pos).normalized()
right = forward.cross(up).normalized()
true_up = right.cross(forward)
cam_obj.matrix_world = mathutils.Matrix((
    (right.x, true_up.x, -forward.x, pos.x),
    (right.y, true_up.y, -forward.y, pos.y),
    (right.z, true_up.z, -forward.z, pos.z),
    (0, 0, 0, 1)))
scene.camera = cam_obj

# Warm up: kernels compile, the BVH is built and kept (persistent data).
cy.samples = 1
scene.render.filepath = out_path + ".warm.exr"
bpy.ops.render.render(write_still=True)
cy.samples = spec["samples"]
scene.render.filepath = out_path
times = []
for _ in range(runs):
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    times.append(time.time() - t0)
times.sort()
json.dump({"min": times[0], "median": times[len(times) // 2], "times": times}, open(out_path + ".time.json", "w"))
print("CYCLES TIMES", times)
