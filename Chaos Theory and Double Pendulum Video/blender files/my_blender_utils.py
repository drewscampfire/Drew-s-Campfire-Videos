import bpy
import sys


def clear_single_keyframes(obj, data_path, array_index):
    obj = bpy.data.objects[obj]
    action = obj.animation_data.action if obj.animation_data else None

    if action:
        fcurve = next(
            (f for f in action.fcurves if f.data_path == data_path and f.array_index == array_index),
            None
        )
        if fcurve:
            action.fcurves.remove(fcurve)


def delete_collection_and_objects(collection_name):
    collection = bpy.data.collections.get(collection_name)
    if collection:
        for subcollection in collection.children:
            for obj in subcollection.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
    else:
        print(f"Collection '{collection_name}' does not exist.")


def delete_all_objects_in_collection(collection_name):
    collection = bpy.data.collections.get(collection_name)
    if not collection:
        print(f"No collection found with the name {collection_name}")
        return

    for obj in collection.objects:
        bpy.data.objects.remove(obj)


def clearly_print(data):
    print(f"\n\nclearly printed: {data}\n\n")


def add_collection_to_scene(name):
    new_collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(new_collection)


def turn_off_visibility(objs):
    for obj in objs:
        obj.visible_diffuse = False
        obj.visible_glossy = False
        obj.visible_transmission = False
        obj.visible_volume_scatter = False
        obj.visible_shadow = False


def delete_collection_by_name(collection_name):
    def delete_collection_and_contents(collection):
        for sub_collection in collection.children:
            delete_collection_and_contents(sub_collection)
        for obj in collection.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(collection)

    collection = bpy.data.collections.get(collection_name)
    if collection:
        delete_collection_and_contents(collection)


def delete_collections_starting_with(prefix):
    def delete_collection_and_contents(collection):
        for sub_collection in collection.children:
            delete_collection_and_contents(sub_collection)
        for obj in collection.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(collection)

    for col in [c for c in bpy.data.collections if c.name.startswith(prefix)]:
        if col:
            delete_collection_and_contents(col)


def create_and_configure_existing_geom_modifier_on_object(obj, modifier_name: str, node_tree_name: str,
                                                          input_indices=None, input_values=None):
    def set_geom_node_input(modifier, input_index, value):
        input_name = f"Input_{input_index}"
        modifier[input_name] = value

    modifier = obj.modifiers.new(modifier_name, 'NODES')
    modifier.node_group = bpy.data.node_groups[node_tree_name].copy()

    for input_index, input_value in zip(input_indices, input_values):
        set_geom_node_input(modifier, input_index, input_value)

    return modifier


def clear_material_node_keyframes(obj):
    """
    Clear all keyframes from all nodes in all materials of the given object.
    """
    # Check if object exists and has materials
    if not obj or not obj.material_slots:
        print(f"No materials found on object: {obj.name if obj else 'None'}")
        return

    print(f"Processing object: {obj.name}")

    # Iterate through all material slots
    for material_slot in obj.material_slots:
        material = material_slot.material

        # Skip if material doesn't exist or doesn't use nodes
        if not material or not material.use_nodes:
            continue

        print(f"Processing material: {material.name}")

        # Get the node tree
        node_tree = material.node_tree

        # Skip if no animation data exists
        if not node_tree.animation_data or not node_tree.animation_data.action:
            print(f"No animation data found in material: {material.name}")
            continue

        fcurves = node_tree.animation_data.action.fcurves
        print(f"Found {len(fcurves)} fcurves")

        # Store fcurves to remove
        to_remove = []

        # First, list all fcurves and their paths
        for fc in fcurves:
            print(f"Found fcurve with path: {fc.data_path}")
            if "nodes" in fc.data_path:
                to_remove.append(fc)

        # Then remove them
        for fc in to_remove:
            print(f"Removing fcurve: {fc.data_path}")
            fcurves.remove(fc)

        # If no fcurves left, remove animation data
        if len(fcurves) == 0 and node_tree.animation_data:
            if node_tree.animation_data.action:
                node_tree.animation_data.action = None

        print(f"Finished processing material: {material.name}")


def clear_material_keyframes_in_range(start_frame, end_frame):
    """
    Clear all material node keyframes within the specified frame range for all materials in the scene.

    Args:
        start_frame (float): Start frame of the range to clear
        end_frame (float): End frame of the range to clear
    """
    # Get all materials in the scene
    materials = bpy.data.materials

    print(f"\n\nClearing keyframes between frames {start_frame} and {end_frame}")
    print(f"Found {len(materials)} materials in the scene")

    for material in materials:
        if not material.use_nodes or not material.node_tree:
            continue

        node_tree = material.node_tree

        # Skip if no animation data exists
        if not node_tree.animation_data or not node_tree.animation_data.action:
            continue

        print(f"\nProcessing material: {material.name}")
        fcurves = node_tree.animation_data.action.fcurves

        # Store fcurves to remove
        fcurves_to_remove = []

        # Process each fcurve
        for fc in fcurves:
            # Get indices of keyframes to remove
            indices_to_remove = []
            for i in range(len(fc.keyframe_points) - 1, -1, -1):  # Iterate backwards
                kf = fc.keyframe_points[i]
                frame = kf.co[0]
                if start_frame <= frame <= end_frame:
                    indices_to_remove.append(i)

            # If all keyframes should be removed, mark entire fcurve for removal
            if len(indices_to_remove) == len(fc.keyframe_points):
                fcurves_to_remove.append(fc)
            else:
                # Remove individual keyframes
                for index in indices_to_remove:  # Already in reverse order
                    try:
                        fc.keyframe_points.remove(fc.keyframe_points[index])
                    except Exception as e:
                        print(f"Warning: Could not remove keyframe at index {index}: {str(e)}")

        # Remove fcurves that had all keyframes in range
        for fc in fcurves_to_remove:
            try:
                print(f"Removing entire fcurve: {fc.data_path}")
                fcurves.remove(fc)
            except Exception as e:
                print(f"Warning: Could not remove fcurve: {str(e)}")

        # Clean up empty actions
        if len(node_tree.animation_data.action.fcurves) == 0:
            print(f"Removing empty action from material: {material.name}")
            node_tree.animation_data.action = None

        print(f"Finished processing material: {material.name}")

    # Refresh the viewport to show changes immediately
    for area in bpy.context.screen.areas:
        area.tag_redraw()

    # Force a frame change update
    current_frame = bpy.context.scene.frame_current
    bpy.context.scene.frame_set(current_frame)

    print("\nFinished clearing keyframes in range")


def clear_visibility_keyframes_in_range(start_frame, end_frame):
    """
    Clear all visibility keyframes within the specified frame range for all objects in the scene.

    Args:
        start_frame (float): Start frame of the range to clear
        end_frame (float): End frame of the range to clear
    """
    # Get all objects in the scene
    objects = bpy.data.objects

    print(f"Clearing visibility keyframes between frames {start_frame} and {end_frame}")
    print(f"Found {len(objects)} objects in the scene")

    for obj in objects:
        # Skip if no animation data exists
        if not obj.animation_data or not obj.animation_data.action:
            continue

        print(f"\nProcessing object: {obj.name}")
        fcurves = obj.animation_data.action.fcurves

        # Find the visibility fcurve
        visibility_fcurve = None
        for fc in fcurves:
            if fc.data_path == "visible_camera":
                visibility_fcurve = fc
                break

        if not visibility_fcurve:
            continue

        # Get indices of keyframes to remove
        indices_to_remove = []
        for i in range(len(visibility_fcurve.keyframe_points) - 1, -1, -1):  # Iterate backwards
            kf = visibility_fcurve.keyframe_points[i]
            frame = kf.co[0]
            if start_frame <= frame <= end_frame:
                indices_to_remove.append(i)

        # Remove keyframes
        for index in indices_to_remove:
            try:
                visibility_fcurve.keyframe_points.remove(visibility_fcurve.keyframe_points[index])
                print(f"Removed keyframe at frame {index} from {obj.name}")
            except Exception as e:
                print(f"Warning: Could not remove keyframe at index {index} from {obj.name}: {str(e)}")

        # If all keyframes were removed, remove the fcurve
        if len(visibility_fcurve.keyframe_points) == 0:
            fcurves.remove(visibility_fcurve)
            print(f"Removed empty visibility fcurve from {obj.name}")

        # Clean up empty actions
        if len(obj.animation_data.action.fcurves) == 0:
            print(f"Removing empty action from object: {obj.name}")
            obj.animation_data.action = None

        print(f"Finished processing object: {obj.name}")

    # Refresh the viewport to show changes immediately
    for area in bpy.context.screen.areas:
        area.tag_redraw()

    # Force a frame change update
    current_frame = bpy.context.scene.frame_current
    bpy.context.scene.frame_set(current_frame)

    print("\nFinished clearing visibility keyframes in range")


def clear_transform_keyframes_in_range(start_frame, end_frame):
    """
    Clear all transform and property keyframes (excluding materials and visibility)
    within the specified frame range for all objects in the scene.

    Args:
        start_frame (float): Start frame of the range to clear
        end_frame (float): End frame of the range to clear
    """
    # Get all objects in the scene
    objects = bpy.data.objects

    print(f"\nClearing transform keyframes between frames {start_frame} and {end_frame}")
    print(f"Found {len(objects)} objects in the scene")

    for obj in objects:
        # Skip if no animation data exists
        if not obj.animation_data or not obj.animation_data.action:
            continue

        print(f"\nProcessing object: {obj.name}")
        fcurves = obj.animation_data.action.fcurves

        # Store fcurves to remove
        fcurves_to_remove = []

        # Process each fcurve
        for fc in fcurves:
            # Skip material and visibility animations
            if "material" in fc.data_path or fc.data_path == "visible_camera":
                continue

            # Get indices of keyframes to remove
            indices_to_remove = []
            for i in range(len(fc.keyframe_points) - 1, -1, -1):  # Iterate backwards
                kf = fc.keyframe_points[i]
                frame = kf.co[0]
                if start_frame <= frame <= end_frame:
                    indices_to_remove.append(i)

            # If all keyframes should be removed, mark entire fcurve for removal
            if len(indices_to_remove) == len(fc.keyframe_points):
                fcurves_to_remove.append(fc)
            else:
                # Remove individual keyframes
                for index in indices_to_remove:  # Already in reverse order
                    try:
                        fc.keyframe_points.remove(fc.keyframe_points[index])
                        print(f"Removed keyframe at frame {index} from {obj.name} - {fc.data_path}")
                    except Exception as e:
                        print(f"Warning: Could not remove keyframe at index {index}: {str(e)}")

        # Remove fcurves that had all keyframes in range
        for fc in fcurves_to_remove:
            try:
                print(f"Removing entire fcurve from {obj.name}: {fc.data_path}")
                fcurves.remove(fc)
            except Exception as e:
                print(f"Warning: Could not remove fcurve: {str(e)}")

        # Clean up empty actions
        if len(obj.animation_data.action.fcurves) == 0:
            print(f"Removing empty action from object: {obj.name}")
            obj.animation_data.action = None

        print(f"Finished processing object: {obj.name}")

    # Refresh the viewport to show changes immediately
    for area in bpy.context.screen.areas:
        area.tag_redraw()

    # Force a frame change update
    current_frame = bpy.context.scene.frame_current
    bpy.context.scene.frame_set(current_frame)

    print("\nFinished clearing transform keyframes in range")


def clear_all_keyframes_in_range(start_frame, end_frame):
    clear_transform_keyframes_in_range(start_frame, end_frame)
    clear_material_keyframes_in_range(start_frame, end_frame)
    clear_visibility_keyframes_in_range(start_frame, end_frame)


def shift_all_keyframes(start_frame, end_frame, frame_offset):
    """
    Shifts all keyframes within the specified frame range by the given offset.
    Works on both object and material node animations.

    Args:
        start_frame (float): Start frame of the range to shift
        end_frame (float): End frame of the range to shift
        frame_offset (int): Number of frames to shift keyframes by (can be negative)
    """
    print(f"\nShifting keyframes between frames {start_frame} and {end_frame} by {frame_offset} frames")

    # Part 1: Shift object keyframes
    for obj in bpy.data.objects:
        if not obj.animation_data or not obj.animation_data.action:
            continue

        fcurves = obj.animation_data.action.fcurves

        for fc in fcurves:
            # Store original keyframes that need to be shifted
            keyframes_to_shift = []
            for kf in fc.keyframe_points:
                if start_frame <= kf.co[0] <= end_frame:
                    keyframes_to_shift.append(kf)

            # Shift the keyframes
            if keyframes_to_shift:
                for kf in keyframes_to_shift:
                    original_frame = kf.co[0]
                    kf.co[0] = original_frame + frame_offset
                    kf.handle_left[0] = kf.handle_left[0] + frame_offset
                    kf.handle_right[0] = kf.handle_right[0] + frame_offset

                # Update the FCurve after moving keyframes
                fc.update()

    # Part 2: Shift material node keyframes
    for material in bpy.data.materials:
        if not material.use_nodes or not material.node_tree:
            continue

        node_tree = material.node_tree
        if not node_tree.animation_data or not node_tree.animation_data.action:
            continue

        fcurves = node_tree.animation_data.action.fcurves

        for fc in fcurves:
            # Store original keyframes that need to be shifted
            keyframes_to_shift = []
            for kf in fc.keyframe_points:
                if start_frame <= kf.co[0] <= end_frame:
                    keyframes_to_shift.append(kf)

            # Shift the keyframes
            if keyframes_to_shift:
                for kf in keyframes_to_shift:
                    original_frame = kf.co[0]
                    kf.co[0] = original_frame + frame_offset
                    kf.handle_left[0] = kf.handle_left[0] + frame_offset
                    kf.handle_right[0] = kf.handle_right[0] + frame_offset

                # Update the FCurve after moving keyframes
                fc.update()

    # Refresh the viewport to show changes immediately
    for area in bpy.context.screen.areas:
        area.tag_redraw()

    # Force a frame change update
    current_frame = bpy.context.scene.frame_current
    bpy.context.scene.frame_set(current_frame)

    print("Finished shifting keyframes")





