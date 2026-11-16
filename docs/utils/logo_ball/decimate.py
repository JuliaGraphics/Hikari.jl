import sys
import bmesh
import bpy

source, target, ratio = sys.argv[sys.argv.index("--") + 1:]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=source, forward_axis="Y", up_axis="Z")
obj = bpy.context.selected_objects[0]

mesh = bmesh.new()
mesh.from_mesh(obj.data)
bmesh.ops.remove_doubles(mesh, verts=mesh.verts, dist=1e-6)
mesh.to_mesh(obj.data)
mesh.free()

decimate = obj.modifiers.new("decimate", "DECIMATE")
decimate.ratio = float(ratio)
decimate.use_collapse_triangulate = True
bpy.context.view_layer.objects.active = obj
bpy.ops.object.modifier_apply(modifier=decimate.name)
bpy.ops.object.shade_smooth()

bpy.ops.wm.obj_export(filepath=target, forward_axis="Y", up_axis="Z", export_materials=False,
                      export_uv=False, export_normals=True, export_triangulated_mesh=True)
print("triangles", len(obj.data.polygons), "vertices", len(obj.data.vertices))
