"""
Runs INSIDE Blender (not imported by the app):
    blender -b --factory-startup --python blender_fbx_to_glb.py -- in.fbx out.glb

Blender's FBX importer applies the file's own unit scale (so a centimeter
FBX comes in at the right size in meters) and converts its axes to Blender's
Z-up; the glTF exporter then converts Z-up to glTF's Y-up. Cameras/lights are
not exported, so the GLB's bounds are the object's own geometry.
"""

import sys

import bpy

args = sys.argv[sys.argv.index("--") + 1 :]
src, dst = args[0], args[1]

bpy.ops.wm.read_factory_settings(use_empty=True)
try:
    bpy.ops.import_scene.fbx(filepath=src)
except Exception as exc:  # corrupt/unsupported FBX: report one readable line, not a traceback
    message = str(exc).strip().splitlines()[-1] if str(exc).strip() else type(exc).__name__
    print(f"TVASTA_CONVERT_ERROR: not a readable FBX file ({message})", file=sys.stderr)
    sys.exit(2)

meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
if not meshes:
    print("TVASTA_CONVERT_ERROR: FBX contains no mesh geometry", file=sys.stderr)
    sys.exit(2)

# Bake every static mesh's world transform (FBX unit scale, axis rotation,
# parent hierarchy) into its vertices, so the GLB has identity nodes. Three.js
# (bbox corners), trimesh in the pose service (exact vertices) and
# glb_inspect then all agree on the same bounds; with a rotated node they'd
# each compute a slightly different box and the MegaPose overlay would sit
# off-center. Skinned meshes are left alone — baking would break the rig.
static = [m for m in meshes if not any(mod.type == "ARMATURE" for mod in m.modifiers)]
if static:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in static:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = static[0]
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.make_single_user(object=True, obdata=True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

bpy.ops.export_scene.gltf(
    filepath=dst,
    export_format="GLB",
    export_yup=True,
    export_apply=True,  # bake modifiers into the exported mesh
    export_cameras=False,
    export_lights=False,
)
