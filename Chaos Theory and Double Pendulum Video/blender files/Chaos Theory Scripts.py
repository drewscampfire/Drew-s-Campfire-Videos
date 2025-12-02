import bpy
import numpy as np
from manim import DEGREES
import importlib
import Drawblend
from mydebugger import inspect_shape_of_array

importlib.reload(Drawblend)
from Drawblend import *
import my_blender_utils
importlib.reload(my_blender_utils)
from my_blender_utils import *
from tqdm import tqdm

# region creating variables to reference to different collections in the scene
objects = {
    "First Pendulums Collection 1": bpy.data.collections["First Pendulums Collection 1"].objects,
    "Second Pendulums Collection 1": bpy.data.collections["Second Pendulums Collection 1"].objects,
    "First Bob Collection 1": bpy.data.collections["First Bob Collection 1"].objects,
    "Angle 1 Collection 1": bpy.data.collections["Angle 1 Collection 1"].objects,
    "Angle 2 Collection 1": bpy.data.collections["Angle 2 Collection 1"].objects,
    "Elbow 1 Collection 1": bpy.data.collections["Elbow 1 Collection 1"].objects,
    "Elbow 2 Collection 1": bpy.data.collections["Elbow 2 Collection 1"].objects,
    "First Vert Line Collection 1": bpy.data.collections["First Vert Line Collection 1"].objects,
    "Second Vert Line Collection 1": bpy.data.collections["Second Vert Line Collection 1"].objects
}


def turn_list_to_blobjects(list_of_objects) -> list[Blobject]:
    return [Blobject(obj) for obj in list_of_objects]


first_pendulum_objects = turn_list_to_blobjects(objects["First Pendulums Collection 1"])
second_pendulum_objects = turn_list_to_blobjects(objects["Second Pendulums Collection 1"])
# bob_1_objects = turn_list_to_blobjects(objects["First Bob Collection 1"])
angle_1_objects = turn_list_to_blobjects(objects["Angle 1 Collection 1"])
angle_2_objects = turn_list_to_blobjects(objects["Angle 2 Collection 1"])
elbow_1_objects = turn_list_to_blobjects(objects["Elbow 1 Collection 1"])
elbow_2_objects = turn_list_to_blobjects(objects["Elbow 2 Collection 1"])
vert_line_1_objects = turn_list_to_blobjects(objects["First Vert Line Collection 1"])
vert_line_2_objects = turn_list_to_blobjects(objects["Second Vert Line Collection 1"])

all_objects = [
    obj for obj_list in [
        first_pendulum_objects,
        second_pendulum_objects,
        # bob_1_objects,
        angle_1_objects,
        angle_2_objects,
        elbow_1_objects,
        elbow_2_objects,
        vert_line_1_objects,
        vert_line_2_objects
    ] for obj in obj_list
]

angle_visualizer_objects = [
    obj for obj_list in [
        angle_1_objects,
        angle_2_objects,
        elbow_1_objects,
        elbow_2_objects,
        vert_line_1_objects,
        vert_line_2_objects
    ] for obj in obj_list
]
pendulum_objects = [
    obj for obj_list in [
        first_pendulum_objects,
        second_pendulum_objects
    ] for obj in obj_list
]
all_colls = (angle_1_objects, elbow_1_objects, vert_line_1_objects, angle_2_objects, elbow_2_objects,
             vert_line_2_objects, first_pendulum_objects, second_pendulum_objects)

# endregion


# active_object = Blobject(bpy.context.active_object)
# selected_objects = turn_list_to_blobjects(bpy.context.selected_objects)


def create_meshgrid_like_2d_array(x_array, y_array):
    X, Y = np.meshgrid(x_array, y_array)

    return np.stack((X, Y), axis=-1).reshape(-1, 2)


def center_points_linspace(start: float, stop: float, num: int, retstep=False):
    d = (stop - start) / (2 * num)

    return np.linspace(start + d, stop - d, num, retstep=retstep)


class OperationOnSelectedObject(BlendScene):
    def __init__(self):
        selected_objects = bpy.context.selected_objects
        print(f"\nRunning OperationOnSelectedObject\n")

        if len(selected_objects) == 0:
            raise ValueError("Warning: No object selected.")
        elif len(selected_objects) > 1:
            raise ValueError("Warning: More than one object selected.")
        self.objs = turn_list_to_blobjects(selected_objects)
        super().__init__()

    def construct(self):
        pass


class Scene2A(BlendScene):
    def construct(self):
        # for obj in angle_visualizer_objects + first_pendulum_objects[1:] + second_pendulum_objects[1:]:
        #     if obj not in [elbow_1_objects[0], elbow_2_objects[0], vert_line_1_objects[0], vert_line_2_objects[0]]:
        #         obj.animate_alpha(0, 1)
        #         obj.animate_toggle_visibility(False, 1)
        # clear_material_keyframes_in_range(715, 1500)
        # clear_visibility_keyframes_in_range(715, 1500)
        # meshgrid_like_array = create_meshgrid_like_2d_array(
        #     center_points_linspace(-1, 1, 4),
        #     center_points_linspace(-0.5625, 0.5625, 4)[::-1]
        # )
        # meshgrid_like_array = np.insert(meshgrid_like_array, 1, 0, axis=1)
        # inspect_shape_of_array(meshgrid_like_array)
        # for i, obj in enumerate(first_pendulum_objects):
        #     start_frame = 715 + i * 30
        #     if obj is first_pendulum_objects[0]:
        #         obj.move_to(meshgrid_like_array[i], start_frame=start_frame, duration=0.5)
        #         continue
        #     obj.animate_location(location=meshgrid_like_array[i-1], frame_num=start_frame)
        #     obj.move_to(meshgrid_like_array[i], start_frame=start_frame, duration=0.5)
        # for col in all_colls:
        #     for i, obj in enumerate(col):
        #         if i == 0:
        #             continue
        #         start_frame = 715 + i * 30
        #         obj.fade_in(start_frame, 0.5, enable_toggle_visibility=True)
        # for obj in angle_visualizer_objects:
        #     obj.animate_toggle_visibility(True, 1300)
        #     obj.fade_out(1300, 2, enable_toggle_visibility=True)
        # clear_material_keyframes_in_range(2680, 3000)
        # clear_visibility_keyframes_in_range(2680, 3000)
        # for obj in angle_visualizer_objects:
        #     obj.fade_in(2660, 1, enable_toggle_visibility=True)
        clear_material_keyframes_in_range(2900, 3570)
        clear_visibility_keyframes_in_range(2900, 3570)
        clear_transform_keyframes_in_range(2900, 3599)
        for obj in vert_line_1_objects + elbow_1_objects + angle_1_objects + first_pendulum_objects:
            if obj in first_pendulum_objects:
                obj.animate_alpha(1, 2960)
                obj.animate_alpha(0.04, 2980)
            else:
                obj.animate_alpha(1, 2960)
                obj.animate_alpha(0.2, 2980)
        anim_frames_main = self.compute_lagged_frame_start_and_end(3000, 3272, 16, 0.5)
        anim_frames_main = np.asarray(anim_frames_main).astype(int).tolist()
        for i, (pendulum_2, angle_2_obj) in enumerate(zip(second_pendulum_objects, angle_2_objects)):
            self.set_frame_number(3000)
            angle_2_obj.animate_geom_modifier_input("Input_14", 0, anim_frames_main[i][0])
            angle_2_obj.animate_geom_modifier_input("Input_14", 2, anim_frames_main[i][0] + 10)

            prev_rot = pendulum_2.get_previous_keyframe_value(3000, "rotation_euler")
            next_rot = pendulum_2.get_next_keyframe_value(3800, "rotation_euler")
            pendulum_2.animate_rotation(prev_rot, anim_frames_main[i][0])
            pendulum_2.animate_rotation(next_rot, anim_frames_main[i][0] + 10)
            angle_2_obj.animate_scale((1, 1, 1), anim_frames_main[i][0])
            angle_2_obj.animate_scale((2, 2, 2), anim_frames_main[i][0] + 20)
            angle_2_obj.shift_by((-0.03, 0, 0.03), anim_frames_main[i][0], end_frame=anim_frames_main[i][0] + 20)
            angle_2_obj.animate_scale((2, 2, 2), 3350)
            angle_2_obj.animate_scale((1.2, 1.2, 1.2), 3410)
            angle_2_obj.shift_by((0.03, 0, -0.03), 3350, end_frame=3410)

        # clear_material_keyframes_in_range(3300, 3450)
        # clear_visibility_keyframes_in_range(3300, 3450)
        # for obj in angle_2_objects:
        #     obj.animate_scale((2, 2, 2), 3420)
        #     obj.animate_scale((1.1, 1.1, 1.1), 3480)
        for obj in angle_1_objects + vert_line_1_objects + elbow_1_objects:
            obj.animate_alpha(0.2, 3350)
            obj.animate_alpha(1, 3410)
        for obj in first_pendulum_objects:
            obj.animate_alpha(0.04, 3350)
            obj.animate_alpha(1, 3410)
        for obj in angle_visualizer_objects:
            obj.fade_out(3490, 1, enable_toggle_visibility=True)
        anim_frames_main = self.compute_lagged_frame_start_and_end(3587, 3682, 16, 0.06)
        for i, obj in enumerate(first_pendulum_objects):
            current_loc = obj.get_previous_keyframe_value(3600, "location")
            obj.animate_location(current_loc, anim_frames_main[i][0])
            obj.animate_location((0, -1 + i / 8, 0), anim_frames_main[i][1])
        # shift_all_keyframes(1750, 2550, 48)


class Scene2C(BlendScene):
    def construct(self):
        for i, (pen1, pen2) in enumerate(zip(first_pendulum_objects, second_pendulum_objects)):
            if i < 16:
                pass
            else:
                pen1.animate_toggle_visibility(False, 2484)
                pen2.animate_toggle_visibility(False, 2484)
                pen1.animate_toggle_visibility(True, 2485)
                pen2.animate_toggle_visibility(True, 2485)
        """
        # clear_all_keyframes_in_range(1, 2700)

        # move to 4 by 4 grid
        meshgrid_like_array = create_meshgrid_like_2d_array(
            center_points_linspace(-1, 1, 4),
            center_points_linspace(-0.5625, 0.5625, 4)[::-1]
        )
        meshgrid_like_array = np.insert(meshgrid_like_array, 1, 0, axis=1)
        for i, obj in enumerate(first_pendulum_objects[:16]):
            obj.animate_location(location=meshgrid_like_array[i], frame_num=1)

        # fade down of first pendulum objects
        for obj in first_pendulum_objects[:16] + angle_1_objects[:16] + elbow_1_objects[:16] + vert_line_1_objects[:16]:
            if obj in first_pendulum_objects[:16]:
                obj.animate_alpha(1, 270)
                obj.animate_alpha(0.04, 300)
            else:
                obj.animate_alpha(1, 270)
                obj.animate_alpha(0.2, 300)

        # animation reveal of 8 decimal planes of angle 2 objects
        anim_frames_main = self.compute_lagged_frame_start_and_end(800, 1004, 16, 0.5)
        anim_frames_main = np.asarray(anim_frames_main).astype(int).tolist()
        for i, (pendulum_2, angle_2_obj) in enumerate(zip(second_pendulum_objects[:16], angle_2_objects[:16])):
            angle_2_obj.set_modifier_input("Input_16", "{:.8f}".format(100.00000000 + 0.00000001 * i))
            angle_2_obj.animate_geom_modifier_input("Input_14", 0, anim_frames_main[i][0])
            angle_2_obj.animate_geom_modifier_input("Input_14", 8, anim_frames_main[i][0] + 10)
            angle_2_obj.animate_geom_modifier_input("Input_15", False, anim_frames_main[i][0])
            angle_2_obj.animate_geom_modifier_input("Input_15", True, anim_frames_main[i][0] + 10)

            current_rot = pendulum_2.get_current_rotation(frame_num=1)
            next_rot = pendulum_2.get_next_keyframe_value(3000, "rotation_euler")
            pendulum_2.animate_rotation(current_rot, anim_frames_main[i][0])
            pendulum_2.animate_rotation(next_rot, anim_frames_main[i][0] + 10)
            angle_2_obj.animate_scale((1, 1, 1), anim_frames_main[i][0])
            angle_2_obj.animate_scale((2, 2, 2), anim_frames_main[i][0] + 20)
            angle_2_obj.shift_by((0.025, 0, 0), anim_frames_main[i][0], end_frame=anim_frames_main[i][0] + 20)
            angle_2_obj.animate_scale((2, 2, 2), 1080)
            angle_2_obj.animate_scale((1.3, 1.3, 1.3), 1140)

        # fade up of first pendulum objects
        for obj in first_pendulum_objects[:16] + angle_1_objects[:16] + elbow_1_objects[:16] + vert_line_1_objects[:16]:
            if obj in first_pendulum_objects:
                obj.animate_alpha(0.04, 1080)
                obj.animate_alpha(1, 1140)
            else:
                obj.animate_alpha(0.2, 1080)
                obj.animate_alpha(1, 1140)

        # fadeout of angle visualizer objects
        for col in [angle_1_objects, angle_2_objects, elbow_1_objects, elbow_2_objects, vert_line_1_objects, vert_line_2_objects]:
            for obj in col[:16]:
                obj.animate_toggle_visibility(True, 2340)
                obj.fade_out(2340, end_frame=2400, enable_toggle_visibility=True)

        anim_frames_main = self.compute_lagged_frame_start_and_end(2400, 2500, 16, 0.1)
        anim_frames = self.compute_lagged_frame_start_and_end(2485, 2700, 984, 0.01)
        for i, obj in enumerate(first_pendulum_objects):
            if i < 16:
                obj.animate_location(meshgrid_like_array[i], anim_frames_main[i][0])
                obj.animate_location((0, -1 + i / 30, 0), anim_frames_main[i][1])
            else:
                obj.animate_location((0, -1.5 + i / 30, 10), anim_frames[i - 16][0])
                obj.animate_location((0, -1 + i / 30, 0), anim_frames[i - 16][1])
                """


class Scene2CX(BlendScene):
    def construct(self):
        clear_transform_keyframes_in_range(150, 350)

        print(f"\n\n len : {len(first_pendulum_objects)}\n\n")
        for i, obj in enumerate(first_pendulum_objects):
            obj.animate_location((0, -1 + i / 30, 0), 200)
            obj.animate_location((0, -1 + i / 267.589, 0), 340)


class Scene2CTS(BlendScene):
    def construct(self):
        for i, obj in enumerate(first_pendulum_objects):
            obj.animate_location((0, -1 + i / 12, 0), 1)
            obj.animate_location((0, -1 + i / 12, 0), 215)
            obj.animate_location((0, -1 + i / 192, 0), 360)


class Scene2D(BlendScene):
    def construct(self):
        for i, obj in enumerate(first_pendulum_objects):
            obj.animate_location((0, -1 + i / 267.589, 0), 1)


class S_edits(BlendScene):
    def construct(self):
        for i, obj in enumerate(first_pendulum_objects):
            if i >= 80:
                obj.hide_viewport(True)


class Scene2E(BlendScene):
    def construct(self):
        for i, obj in enumerate(first_pendulum_objects):
            obj.animate_location((0, -1 + i / 35, 0), 1)
        for obj in first_pendulum_objects + second_pendulum_objects:
            obj.obj.scale[1] = 0.55

        for obj in [first_pendulum_objects[-1], second_pendulum_objects[-1]]:
            obj.visible_camera = True
            obj.hide_viewport(False)


class Scene3A(BlendScene):
    def construct(self):
        for obj in first_pendulum_objects:
            obj.animate_rotation((0, 0, 0), 87)
            obj.animate_rotation((0, -90 * DEGREES, 0), 679)
        for obj in second_pendulum_objects:
            obj.animate_rotation((0, 0, 0), 87)
            obj.animate_rotation((0, -160 * DEGREES, 0), 679)
        for obj in angle_visualizer_objects:
            obj.animate_toggle_visibility(True, 1)
            obj.fade_out(690, end_frame=730, enable_toggle_visibility=True)
            obj.fade_in(1180, end_frame=1230, enable_toggle_visibility=True)


class Scene3S_6(BlendScene):
    def construct(self):
        shift_all_keyframes(1074, 1150, 1162 - 1080)


class Scene3B(BlendScene):
    def construct(self):
        # initializing the scene at frame 0
        self.set_frame_number(0)
        first_pendulum_objects[0].animate_location((-10.1, -10.4, 3.8), -100)


if __name__ == "__main__":
    # OperationOnSelectedObject().construct()
    Scene2E().construct()
