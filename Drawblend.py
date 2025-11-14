import bpy
from mathutils import Euler, Vector
import numpy as np
import sys
from manim import DEGREES
import json
from mathutils import Vector, Euler

from chaos_theory_addon import Scene2_S


def serialize_vector(v):
    return [v.x, v.y, v.z]


def deserialize_vector(v):
    return Vector(v)


fps: int = bpy.context.scene.render.fps


class BlendScene:
    def __init__(self):
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

    def construct(self):
        # This is a placeholder. Subclasses should override this method.
        raise NotImplementedError

    def delete_keyframes_in_range(self, start_frame, end_frame, delete_materials=True, delete_actions=True,
                                  exclude_collections=[]):
        """Delete keyframes in the range start frame (inclusive) to end frame (exclusive)."""
        self.set_frame_number(1)
        actions = []
        if delete_actions:
            for obj in bpy.data.objects:
                if obj.animation_data and obj.animation_data.action:
                    if obj.users_collection:  # Check if the object is in a collection
                        if any(col.name in exclude_collections for col in obj.users_collection):
                            continue  # Skip this object
                    actions.append(obj.animation_data.action)
        if delete_materials:
            for mat in bpy.data.materials:
                if mat.node_tree and mat.node_tree.animation_data and mat.node_tree.animation_data.action:
                    # Assuming you don't put materials in collections, or if you do, you handle it here
                    actions.append(mat.node_tree.animation_data.action)
        for action in actions:
            for fcurve in action.fcurves:
                indices_to_remove = [i for i, kp in enumerate(fcurve.keyframe_points)
                                     if start_frame <= kp.co[0] < end_frame]
                for i in reversed(indices_to_remove):
                    fcurve.keyframe_points.remove(fcurve.keyframe_points[i])

    def set_frame_number(self, num):
        bpy.context.scene.frame_set(num)

    def compute_lagged_frame_start_and_end(self, frame_start, frame_end, number_of_animations, lag_ratio) -> list:
        """
        Calculate the start and end frames for each animation based on the given frame range, number of animations, and lag ratio.

        Parameters:
            frame_start (int): The starting frame of the animation sequence.
            frame_end (int): The ending frame of the animation sequence.
            number_of_animations (int): The number of animations in the sequence.
            lag_ratio (float): The ratio of lag between animations.

        Returns:
            list of tuples: R list of tuples representing the start and end frames for each animation.
        """
        total_runtime = frame_end - frame_start
        # Calculate individual animation duration
        animation_duration = total_runtime / (1 + lag_ratio * (number_of_animations - 1))

        # Calculate lag between animations
        lag = animation_duration * lag_ratio

        frames = []
        for i in range(number_of_animations):
            start_frame = i * lag
            end_frame = start_frame + animation_duration
            frames.append((start_frame + frame_start, end_frame + frame_start))

        return frames


class Blobject:
    def __init__(self, obj=None):
        self.obj = obj
        self.alpha_input = None
        self.geom_modifier = None
        obj_mat = None  # Declare obj_mat before the try block

        try:
            obj_mat = bpy.data.materials[f"{obj.name}"]
        except Exception as e:
            print(f"\nError: {e}; cannot find material for object {obj.name}.")

        # Ensure that obj_mat is not None before accessing it
        if obj_mat is not None:
            try:
                self.alpha_input = obj_mat.node_tree.nodes["Alpha"].outputs[0]
            except KeyError:
                raise KeyError(f"Object {self.obj.name} does not have an Alpha input.")
        else:
            # Handle the case where obj_mat is None (e.g., assign a default value or raise an exception)
            print(f"No material found for {self.obj.name}, skipping Alpha input assignment.")

        for modifier in self.obj.modifiers:
            if modifier.type == 'NODES':
                self.geom_modifier = modifier
                break  # Exit loop once the Geometry Nodes modifier is found

    def get_material(self, material_index):
        assert 0 <= material_index < len(self.obj.material_slots), "Material index out of range!"
        return self.obj.material_slots[material_index].material

    def get_current_location(self, based_on_previous_keyframe=False, frame_num=None):
        if not based_on_previous_keyframe:
            return self.obj.location
        else:
            return self.get_previous_keyframe_value(frame_num, "location")

    def get_current_rotation(self, based_on_previous_keyframe=False, frame_num=None):
        if not based_on_previous_keyframe:
            return self.obj.rotation_euler
        else:
            return self.get_previous_keyframe_value(frame_num, "rotation_euler")

    def get_current_scale(self, based_on_previous_keyframe=False, frame_num=None):
        if not based_on_previous_keyframe:
            return self.obj.scale
        else:
            print("prev")
            print(self.get_previous_keyframe_value(frame_num, "scale"))
            return self.get_previous_keyframe_value(frame_num, "scale")

    def get_previous_keyframe_value(self, frame_num: int, attribute: str):
        if attribute not in ["rotation_euler", "location", "scale"]:
            raise ValueError("Invalid attribute. Choose one of: rotation_euler, location, scale")

        values = []
        for i in range(3):  # Each attribute (location, rotation, scale) has 3 components (X, Y, Z)
            fcurve = self.obj.animation_data.action.fcurves.find(attribute, index=i)
            if not fcurve:
                values.append(getattr(self.obj, attribute)[i])
                continue

            # Get keyframes before frame_num
            previous_keyframes = [kf.co[1] for kf in fcurve.keyframe_points if kf.co[0] < frame_num]

            if not previous_keyframes:
                values.append(fcurve.evaluate(frame_num))
            else:
                values.append(previous_keyframes[-1])

        return tuple(values)

    def get_next_keyframe_value(self, frame_num: int, attribute: str):
        if attribute not in ["rotation_euler", "location", "scale"]:
            raise ValueError("Invalid attribute. Choose one of: rotation_euler, location, scale")

        values = []
        for i in range(3):  # Each attribute (location, rotation, scale) has 3 components (X, Y, Z)
            fcurve = self.obj.animation_data.action.fcurves.find(attribute, index=i)
            if not fcurve:
                values.append(getattr(self.obj, attribute)[i])
                continue

            # Get keyframes after frame_num and find the one with the smallest time greater than frame_num
            next_keyframes = [kf for kf in fcurve.keyframe_points if kf.co[0] > frame_num]

            if not next_keyframes:
                values.append(fcurve.evaluate(frame_num))
            else:
                next_keyframe = min(next_keyframes, key=lambda kf: kf.co[0])
                values.append(next_keyframe.co[1])

        return tuple(values)

    def animate_toggle_visibility(self, visibility_status: bool, frame_num: int):
        self.obj.visible_camera = visibility_status
        self.obj.keyframe_insert(data_path="visible_camera", frame=frame_num)

        return self

    # def animate_toggle_show_in(self, visibility_status: bool, frame_num: int, viewports=True, renders=True):
    #     assert viewports or renders, "Either viewports or renders must be True"
    #     if viewports:
    #         self.obj.hide_viewport = not visibility_status
    #         self.obj.keyframe_insert(data_path="hide_viewport", frame=frame_num)
    #     if renders:
    #         self.obj.hide_render = not visibility_status
    #         self.obj.keyframe_insert(data_path="hide_render", frame=frame_num)
    #
    #     return self

    def animate_alpha(self, alpha: float, frame_num: int):
        self.alpha_input.default_value = alpha
        self.alpha_input.keyframe_insert(data_path="default_value", frame=frame_num)

        return self

    def animate_location(self, location: tuple | list, frame_num: int):
        self.obj.location = location
        self.obj.keyframe_insert(data_path="location", index=0, frame=frame_num)
        self.obj.keyframe_insert(data_path="location", index=1, frame=frame_num)
        self.obj.keyframe_insert(data_path="location", index=2, frame=frame_num)

        return self

    def animate_rotation(self, rotation: tuple, frame_num: int):
        self.obj.rotation_euler = rotation
        self.obj.keyframe_insert(data_path="rotation_euler", index=0, frame=frame_num)
        self.obj.keyframe_insert(data_path="rotation_euler", index=1, frame=frame_num)
        self.obj.keyframe_insert(data_path="rotation_euler", index=2, frame=frame_num)

        return self

    def animate_scale(self, scale: tuple, frame_num: int):
        self.obj.scale = scale
        self.obj.keyframe_insert(data_path="scale", index=0, frame=frame_num)
        self.obj.keyframe_insert(data_path="scale", index=1, frame=frame_num)
        self.obj.keyframe_insert(data_path="scale", index=2, frame=frame_num)

        return self

    def animate_geom_modifier_input(self, input_name: str, value, frame_number: int):
        bpy.context.scene.frame_set(frame_number)
        self.geom_modifier[input_name] = value
        data_path = f'modifiers["{self.geom_modifier.name}"]["{input_name}"]'
        self.obj.keyframe_insert(data_path=data_path, frame=frame_number)

    def set_modifier_input(self, input_name: str, value):
        self.geom_modifier[input_name] = value

    def fade_in(
            self,
            start_frame: int,
            duration: float = None,
            end_frame: int = None,
            enable_toggle_visibility: bool = False
    ):
        if end_frame is None:
            if duration is not None:
                end_frame = start_frame + int(duration * fps)
            else:
                raise ValueError("Error: Please input either a duration or an end frame!")
        if enable_toggle_visibility:
            self.animate_toggle_visibility(True, start_frame)
        self.animate_alpha(0, start_frame)
        self.animate_alpha(1, end_frame)

        return self

    def fade_out(
            self,
            start_frame: int,
            duration: float = None,
            end_frame: int = None,
            enable_toggle_visibility: bool = False
    ):
        if end_frame is None:
            if duration is not None:
                end_frame = start_frame + int(duration * fps)
            else:
                raise ValueError("Error: Please input either a duration or an end frame!")
        if enable_toggle_visibility:
            self.animate_toggle_visibility(False, end_frame)
        self.animate_alpha(1, start_frame)
        self.animate_alpha(0, end_frame)

        return self

    def move_to(self,
                new_location: tuple,
                start_frame: int,
                duration: float = None,
                end_frame: int = None):
        if end_frame is None:
            if duration is not None:
                end_frame = start_frame + int(duration * fps)
            else:
                raise ValueError("Error: Please input either a duration or an end frame!")

        self.animate_location(tuple(self.get_current_location(True, start_frame)), start_frame)
        self.animate_location(new_location, end_frame)

        return self

    def shift_by(self, shift_amount: tuple, start_frame: int, duration: float = None, end_frame: int = None):
        if end_frame is None:
            if duration is not None:
                end_frame = start_frame + int(duration * fps)
            else:
                raise ValueError("Error: Please input either a duration or an end frame!")

        self.animate_location(self.get_current_location(True, start_frame), start_frame)
        self.animate_location(tuple(self.get_current_location() + Vector(shift_amount)), end_frame)

        return self

    def set_rotation_to(self, new_rotation_values: tuple, start_frame: int, duration: float = None, end_frame: int =
    None):
        if end_frame is None:
            if duration is not None:
                end_frame = start_frame + int(duration * fps)
            else:
                raise ValueError("Error: Please input either a duration or an end frame!")

        self.animate_rotation(tuple(self.get_current_rotation(True, start_frame)), start_frame)
        self.animate_rotation(new_rotation_values, end_frame)

        return self

    def shift__rotation_by(self, shift_amount: tuple, start_frame: int, duration: float = None, end_frame: int = None):
        init_rotation_euler = self.get_current_rotation()
        if end_frame is None:
            if duration is not None:
                end_frame = start_frame + int(duration * fps)
            else:
                raise ValueError("Error: Please input either a duration or an end frame!")

        self.animate_rotation(init_rotation_euler, start_frame)
        shifted_rotation = Euler([init_rotation_euler[i] + Euler(shift_amount)[i] for i in range(3)],
                                 init_rotation_euler.order)
        self.animate_rotation(tuple(shifted_rotation), end_frame)

        return self

    def set_scale_to(self, new_size: tuple, start_frame: int, duration: float = None, end_frame: int = None):
        """
        Don't use. It's too buggy.
        """
        if end_frame is None:
            if duration is not None:
                end_frame = start_frame + int(duration * fps)
            else:
                raise ValueError("Error: Please input either a duration or an end frame!")
        self.animate_scale(tuple(self.get_current_scale(True, start_frame)), start_frame)
        self.animate_scale(new_size, end_frame)

        return self

    # Scene Optimization Methods
    def hide_viewport(self, hide: bool = True):
        self.obj.hide_viewport = hide

    def get_object_index_in_collection(self):
        parent_collection = self.obj.users_collection[0]  # Assuming the object is in at least one collection
        try:
            index = parent_collection.objects[:].index(self.obj)
            return index
        except ValueError:
            return None

    def set_material_attribute(
            self,
            material_index: int,
            node_name: str,
            input_index: int,
            value: float
    ):
        assert 0 <= material_index < len(self.obj.material_slots), (f"Material index out of range! "
                                                                    f"len(obj.material_slots) = "
                                                                    f"{len(self.obj.material_slots)}")
        mat = self.get_material(material_index)
        mat.node_tree.nodes[node_name].inputs[input_index].default_value = value

    def animate_material_attribute(
            self,
            material_index: int,
            node_name: str,
            input_index: int,
            value: float,
            frame_num: int
    ):

        mat = self.get_material(material_index)
        self.set_material_attribute(material_index, node_name, input_index, value)
        mat.node_tree.nodes[node_name].inputs[input_index].keyframe_insert("default_value", frame=frame_num)

    def slide_material_attribute(
            self,
            material_index: int,
            node_name: str,
            input_index: int,
            init_value: float,
            final_value: float,
            start_frame: int,
            duration: float = None,
            end_frame: int = None
    ):

        if end_frame is None:
            if duration is not None:
                end_frame = start_frame + int(duration * fps)
            else:
                raise ValueError("Error: Please input either a duration or an end frame!")
        self.animate_material_attribute(material_index, node_name, input_index, init_value, start_frame)
        self.animate_material_attribute(material_index, node_name, input_index, final_value, end_frame)






