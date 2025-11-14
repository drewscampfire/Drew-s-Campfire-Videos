import bpy
import os

# --- Configuration ---
image_folder_path = r""
collection_name = "ImportedImagePlanes"
offset_distance = 0 # Set to 0 if you want them all at the origin
allowed_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".tga"}

def import_images_to_collection(folder_path, coll_name, spacing):
    """
    Imports images from a folder as planes into a specified collection in Blender.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at path: {folder_path}")
        return None

    if coll_name in bpy.data.collections:
        target_collection = bpy.data.collections[coll_name]
        print(f"Using existing collection: '{coll_name}'")
    else:
        target_collection = bpy.data.collections.new(coll_name)
        bpy.context.scene.collection.children.link(target_collection)
        print(f"Created new collection: '{coll_name}'")

    image_files = []
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            _, ext = os.path.splitext(filename)
            if ext.lower() in allowed_extensions:
                image_files.append(filename)

    if not image_files:
        print(f"No image files found in folder: {folder_path}")
        return target_collection

    print(f"Found {len(image_files)} image files to import.")

    imported_count = 0
    current_x_offset = 0.0
    original_scene_collection = bpy.context.view_layer.active_layer_collection.collection

    for filename in image_files:
        file_list = [{'name': filename}]
        full_path = os.path.join(folder_path, filename)

        try:
            bpy.ops.image.import_as_mesh_planes(
                files=file_list,
                directory=folder_path,

                # choose one of these two:
                shader='SHADELESS',  # → completely flat, ignores lights
                # shader='EMISSION',           # → self-illuminating
                # emit_strength=1.0,           # needed only if using EMISSION

                # transparency & blending
                use_transparency=True,
                render_method='BLENDED',  # colored transparency

                # keep other defaults or adjust as needed
                interpolation='Linear',  # nearest-neighbor sampling
                extension='CLIP',
                alpha_mode='STRAIGHT',
                use_auto_refresh=True,
                relative=True,
                overwrite_material=True,

                # transform settings...
                size_mode='ABSOLUTE',
                height=1.0,
            )

            imported_obj = bpy.context.active_object

            if imported_obj:
                print(f"  Imported '{filename}' as object '{imported_obj.name}'")

                for coll in imported_obj.users_collection:
                     coll.objects.unlink(imported_obj)

                target_collection.objects.link(imported_obj)

                if spacing > 0:
                    imported_obj.location.x = current_x_offset
                    current_x_offset += spacing

                imported_count += 1
            else:
                print(f"  Warning: Could not get active object after importing '{filename}'. Skipping move/offset.")

        except Exception as e:
            print(f"Error importing file '{filename}': {e}")

    print(f"\nImport process finished.")
    print(f"Successfully imported {imported_count} images into collection '{coll_name}'.")

    return target_collection

# --- Run the main function ---
if __name__ == "__main__":
    if bpy.context.active_object and bpy.context.active_object.mode == 'EDIT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')

    created_collection = import_images_to_collection(
        image_folder_path,
        collection_name,
        offset_distance
    )

    if created_collection:
        print("Script finished successfully.")
    else:
        print("Script finished with errors (folder not found?).")