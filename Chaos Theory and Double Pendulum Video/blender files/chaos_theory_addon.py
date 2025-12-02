import bpy
from numpy import sin, cos
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit
from manim import DEGREES, ORIGIN
from tqdm import tqdm
import math
from mydebugger import timer
from my_blender_utils import turn_off_visibility, create_and_configure_existing_geom_modifier_on_object

bl_info = {
    "name": "addon as utils for my chaos theory video",
    "author": "Drew TR",
    "version": (1, 0),
    "blender": (3, 6, 1),
    "location": "View3D > Toolbar > Tracer",
    "description": "bunch of tools for my chaos theory video",
    "category": "3D View",
}

R = False
B = False
Scene2_S = False
Scene2_SS = False
CTS = False  # short-form chaos theory simulation
Scene2_U = False
Scene2_V = False
Scene2_V_30secs = True
A3 = False
B3 = False
S7 = False

fps = bpy.context.scene.render.fps
HINGE_LOCATION = -0.23527568578720093  # location of hinge from the top of the first pendulum rod
ANGLE_RADIUS = 0.08
hide_index = 100

if R:  # 16 double pendulums
    range_angle_1 = 90, 90
    range_angle_2 = 100, 115
if Scene2_S:  # 1000 double pendulums
    range_angle_1 = 90, 90
    range_angle_2 = 100, 100.00000999
    hide_index = 80
if Scene2_SS:
    range_angle_1 = 90, 90
    range_angle_2 = 100, 100.00000999
    hide_index = 80
if CTS:
    range_angle_1 = 90, 90
    range_angle_2 = 120, 120.999
    hide_index = 80
if Scene2_U:
    range_angle_1 = 65, 65
    range_angle_2 = 50, 75
if Scene2_V:
    range_angle_1 = 101, 101
    range_angle_2 = 113, 117  # 161, 164.92; 50 elements
if Scene2_V_30secs:
    range_angle_1 = 45, 65
    range_angle_2 = -80, -100
if A3:
    range_angle_1 = 90, 90
    range_angle_2 = 160, 160
if B3:
    range_angle_1 = 120, 130
    range_angle_2 = 150, 160
    HINGE_LOCATION = -0.73
if S7:
    range_angle_1 = 120, 130
    range_angle_2 = 150, 160


DOWN_VERTICAL = np.array([0, 0, -1])


source_collection = bpy.data.collections.get("BDP")
first_rodT = source_collection.objects["first pendulum"]
second_rodT = source_collection.objects["second pendulum"]
BOB1 = source_collection.objects["bob 1"]

bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)


def hide_viewport(*objs, hide: bool = True):
    for obj in objs:
        obj.hide_viewport = hide


def get_object_index_in_collection(obj):
    parent_collection = obj.users_collection[0]  # Assuming the object is in at least one collection
    try:
        index = parent_collection.objects[:].index(obj)
        return index
    except ValueError:
        return None


class DoublePendulum:
    g = 9.81
    if not B3:
        m1 = m2 = 1  # mass of the 2 bobs
        l1 = l2 = 1  # length of the 2 rods
    if B3:
        m1 = 3
        m2 = 1.5
        l1 = 0.25
        l2 = 0.2

    def __init__(
            self,
            ps1,
            t1,
            t2,
            t_span,
            rod_length=1.0,
    ):
        self.ps1 = np.array(ps1)
        self.t1 = t1
        self.t2 = t2
        self.t_span = t_span
        self.rod_length = rod_length

    @staticmethod
    @njit
    def hamilton_rhs(t1, t2,
                     p1, p2,
                     g,
                     m1, m2,
                     l1, l2):
        a = l1 * l2 * (m1 + m2 * sin(t1 - t2) ** 2)
        b = (p1 * p2 * sin(t1 - t2)) / a
        c = (m2 * (l2 * p1) ** 2 + (m1 + m2) * (l1 * p2) ** 2 - 2 * l1 * l2 * m2 * p1 * p2 * cos(t1 - t2)) * sin(
            2 * (t1 - t2)) / (2 * a ** 2)

        # RHSs of the Hamiltonian-derived first-order differential equations
        rhs_t1 = (l2 * p1 - l1 * p2 * cos(t1 - t2)) / (l1 * a)
        rhs_t2 = (l1 * (m1 + m2) * p2 - l2 * m2 * p1 * cos(t1 - t2)) / (l2 * m2 * a)
        rhs_p1 = -(m1 + m2) * g * l1 * sin(t1) - b + c
        rhs_p2 = -m2 * g * l2 * sin(t2) + b - c

        return np.array([
            rhs_t1, rhs_t2,
            rhs_p1, rhs_p2
        ])

    def funcs(self, t, r):
        t1, t2, p1, p2 = r

        return self.hamilton_rhs(t1, t2,
                                 p1, p2,
                                 self.g,
                                 self.m1, self.m2,
                                 self.l1, self.l2)

    def get_t_values(self):
        if R or B or Scene2_S or Scene2_SS or CTS or A3 or B3 or S7 or Scene2_V_30secs:
            return np.arange(0, self.t_span, 1 / fps)
        if Scene2_U:
            # Constants and Configuration
            time_to_speed_up = 400 / fps
            max_speed = 4 / fps
            max_num_of_frames = 1550
            speed_increment = 0.001  # Speed increase per step after time_to_speed_up

            # Initial phase with constant speed
            t_evals = list(np.arange(0, time_to_speed_up, 1 / fps))

            # Phase of increasing speed
            current_speed = 1 / fps
            t = time_to_speed_up
            while current_speed < max_speed and t < self.t_span:
                t_evals.append(t)
                t += current_speed
                current_speed += speed_increment

            # Phase with constant high speed
            while t < self.t_span and len(t_evals) < max_num_of_frames:
                t_evals.append(t)
                t += current_speed

            return t_evals
        if Scene2_V:
            # Constants and Configuration for Scene2_V
            time_to_speed_up = 807 / fps
            time_to_slow_down = 800 / fps
            regular_speed = 1 / fps
            max_speed = 3 / fps
            max_num_of_frames = 3150
            speed_increment = 0.001
            current_speed = regular_speed

            t_evals = list(np.arange(0, min(time_to_speed_up, self.t_span), regular_speed))
            t = time_to_speed_up

            # Phase of increasing speed
            while current_speed < max_speed and t < self.t_span and len(t_evals) < max_num_of_frames:
                t_evals.append(round(t, 8))  # Rounding to ensure precision
                t += current_speed
                current_speed += speed_increment

            # Phase with constant high speed
            while t < self.t_span and len(t_evals) < max_num_of_frames:
                t_evals.append(round(t, 8))
                t += current_speed

            # Sort the final list to guarantee ascending order
            t_evals = sorted(t_evals)

            return t_evals

    def get_angles(self):
        ans = solve_ivp(
            self.funcs,  # function to solve
            (0, self.t_span),  # time span
            (self.t1, self.t2, 0, 0),  # initial conditions
            t_eval=self.get_t_values(),  # time points
            method='DOP853',  # RK23 or RK45 or DOP853
            rtol=1e-10,
            atol=1e-12
        )
        theta_1 = self.normalize_angles_in_rads(np.array(ans.y[0]))
        theta_2 = self.normalize_angles_in_rads(np.array(ans.y[1]))

        return (
            self.change_first_element(theta_1, self.t1),  # list of theta_1 over time
            self.change_first_element(theta_2, self.t2),  # list of theta_2 over time
        )

    @staticmethod
    def normalize_angles_in_rads(angle_progression: np.ndarray) -> np.ndarray:
        x = np.fmod(angle_progression, 2 * np.pi)
        y = np.fmod(angle_progression, np.pi)

        return 2 * y - x

    @staticmethod
    def change_first_element(arr: np.ndarray, value: float) -> np.ndarray:
        arr[0] = value

        return arr


class DoublePendulumAnimator:
    def __init__(
            self,
    ):
        self.cohort_num = 1
        while f"ADPG {self.cohort_num}" in bpy.data.collections:
            self.cohort_num += 1

        def initialize_new_collections():
            # Generate unique parent collection name
            parent_name = f"ADPG {self.cohort_num}"

            # Create and link parent collection to scene
            parent_collection = bpy.data.collections.new(parent_name)
            bpy.context.scene.collection.children.link(parent_collection)

            # Define and create child collections
            child_names = ['First Pendulums Collection', 'Second Pendulums Collection',
                           'First Bob Collection', 'Angle 1 Collection', 'Angle 2 Collection', 'Elbow 1 Collection',
                           'Elbow 2 Collection', 'First Vert Line Collection', 'Second Vert Line Collection']

            for child_name in child_names:
                child_collection = bpy.data.collections.new(child_name + " " + str(self.cohort_num))
                parent_collection.children.link(child_collection)
                setattr(self, child_name.replace(" ", ""), child_collection)

        initialize_new_collections()

    @timer
    def chaos_demonstration(
            self,
            angle_1_range,
            angle_2_range,
            num_of_dp_systems,
            duration,
            starting_frame
    ):
        self.angle_1_range = angle_1_range
        self.angle_2_range = angle_2_range
        self.num_of_double_pendulums = num_of_dp_systems
        self.duration = duration
        self.starting_frame = starting_frame

        def simulate_double_pendulum(angle_pair, first_rod, second_rod, duration):
            frame_num = self.starting_frame

            def remap(angle_pair, precision: float):
                def linear_interpolation(x):
                    return 100.00 + precision * (x - 100)

                return linear_interpolation(angle_pair[0]), linear_interpolation(angle_pair[1])

            # duplicate for two_C_1; 1260 is the start frame for two_A_1 and 1650 for two_C_1
            if R:
                frame_num = 1850
                double_pendulum = DoublePendulum(
                    ORIGIN,
                    angle_pair[0] * DEGREES,
                    angle_pair[1] * DEGREES,
                    t_span=15
                )
                theta_1_array, theta_2_array = double_pendulum.get_angles()
                for theta_1, theta_2 in zip(-theta_1_array, -theta_2_array):
                    first_rod.rotation_euler[1] = theta_1
                    first_rod.keyframe_insert(data_path="rotation_euler", index=1, frame=frame_num)

                    second_rod.rotation_euler[1] = theta_2
                    second_rod.keyframe_insert(data_path="rotation_euler", index=1, frame=frame_num)

                    frame_num += 1

            if R:
                # 0.01 angle diff
                frame_start = 3850
                angle_pair_01 = remap(angle_pair, 0.01)
                double_pendulum = DoublePendulum(
                    ORIGIN,
                    90 * DEGREES,
                    angle_pair_01[1] * DEGREES,
                    25
                )
                theta_1_2_array, theta_2_2_array = double_pendulum.get_angles()
                frame_num = frame_start
                for theta_1, theta_2 in zip(-theta_1_2_array, -theta_2_2_array):
                    first_rod.rotation_euler[1] = theta_1
                    first_rod.keyframe_insert(data_path="rotation_euler", index=1, frame=frame_num)

                    second_rod.rotation_euler[1] = theta_2
                    second_rod.keyframe_insert(data_path="rotation_euler", index=1, frame=frame_num)

                    frame_num += 1

            # if R:
            #     # 0.000 000 01 diff
            #     frame_start = 8000
            #     angle_pair_0001 = remap(angle_pair, 1e-8)
            #     double_pendulum = DoublePendulum(
            #         ORIGIN,
            #         90 * DEGREES,
            #         angle_pair_0001[1] * DEGREES,
            #         1
            #     )
            #     theta_1_2_array, theta_2_2_array = double_pendulum.get_angles()
            #     frame_num = frame_start
            #     for theta_1, theta_2 in zip(-theta_1_2_array, -theta_2_2_array):
            #         first_rod.rotation_euler[1] = theta_1
            #         first_rod.keyframe_insert(data_path="rotation_euler", index=1, frame=frame_num)
            #
            #         second_rod.rotation_euler[1] = theta_2
            #         second_rod.keyframe_insert(data_path="rotation_euler", index=1, frame=frame_num)
            #
            #         frame_num += 1

            if Scene2_S or Scene2_SS:
                if Scene2_S:
                    frame_num = 3050
                elif Scene2_SS:
                    frame_num = 810
                double_pendulum = DoublePendulum(
                    ORIGIN,
                    angle_pair[0] * DEGREES,
                    angle_pair[1] * DEGREES,
                    duration  # 58
                )
                theta_1_array, theta_2_array = double_pendulum.get_angles()
                for theta_1, theta_2 in zip(-theta_1_array, -theta_2_array):
                    first_rod.rotation_euler[1] = theta_1
                    first_rod.keyframe_insert(data_path="rotation_euler", index=1, frame=frame_num)

                    second_rod.rotation_euler[1] = theta_2
                    second_rod.keyframe_insert(data_path="rotation_euler", index=1, frame=frame_num)

                    frame_num += 1

            if CTS:
                frame_num = 1500
                double_pendulum = DoublePendulum(
                    ORIGIN,
                    angle_pair[0] * DEGREES,
                    angle_pair[1] * DEGREES,
                    duration  # 58
                )
                theta_1_array, theta_2_array = double_pendulum.get_angles()
                for theta_1, theta_2 in zip(-theta_1_array, -theta_2_array):
                    first_rod.rotation_euler[1] = theta_1
                    first_rod.keyframe_insert(data_path="rotation_euler", index=1, frame=frame_num)

                    second_rod.rotation_euler[1] = theta_2
                    second_rod.keyframe_insert(data_path="rotation_euler", index=1, frame=frame_num)

                    frame_num += 1

            if Scene2_U:  # 1000 DPs
                frame_num = 500
                double_pendulum = DoublePendulum(
                    ORIGIN,
                    angle_pair[0] * DEGREES,
                    angle_pair[1] * DEGREES,
                    t_span=190)

                theta_1_array, theta_2_array = double_pendulum.get_angles()
                for theta_1, theta_2 in zip(-theta_1_array, -theta_2_array):
                    first_rod.rotation_euler[1] = theta_1
                    first_rod.keyframe_insert(data_path="rotation_euler", index=1, frame=frame_num)

                    second_rod.rotation_euler[1] = theta_2
                    second_rod.keyframe_insert(data_path="rotation_euler", index=1, frame=frame_num)

                    frame_num += 1

            if Scene2_V:  # 1000 DPs
                frame_num = 1550
                double_pendulum = DoublePendulum(
                    ORIGIN,
                    angle_pair[0] * DEGREES,
                    angle_pair[1] * DEGREES,
                    t_span=135)

                theta_1_array, theta_2_array = double_pendulum.get_angles()
                for theta_1, theta_2 in zip(-theta_1_array, -theta_2_array):
                    first_rod.rotation_euler[1] = theta_1
                    first_rod.keyframe_insert(data_path="rotation_euler", index=1, frame=frame_num)

                    second_rod.rotation_euler[1] = theta_2
                    second_rod.keyframe_insert(data_path="rotation_euler", index=1, frame=frame_num)

                    frame_num += 1

            if Scene2_V_30secs:  # 1000 DPs
                frame_num = 1
                double_pendulum = DoublePendulum(
                    ORIGIN,
                    angle_pair[0] * DEGREES,
                    angle_pair[1] * DEGREES,
                    t_span=70)

                theta_1_array, theta_2_array = double_pendulum.get_angles()
                for theta_1, theta_2 in zip(-theta_1_array, -theta_2_array):
                    first_rod.rotation_euler[1] = theta_1
                    first_rod.keyframe_insert(data_path="rotation_euler", index=1, frame=frame_num)

                    second_rod.rotation_euler[1] = theta_2
                    second_rod.keyframe_insert(data_path="rotation_euler", index=1, frame=frame_num)

                    frame_num += 1

            if A3 or B3 or S7:
                double_pendulum = DoublePendulum(
                    ORIGIN,
                    angle_pair[0] * DEGREES,
                    angle_pair[1] * DEGREES,
                    t_span=duration)

                theta_1_array, theta_2_array = double_pendulum.get_angles()
                for theta_1, theta_2 in zip(-theta_1_array, -theta_2_array):
                    first_rod.rotation_euler[1] = theta_1
                    first_rod.keyframe_insert(data_path="rotation_euler", index=1, frame=frame_num)

                    second_rod.rotation_euler[1] = theta_2
                    second_rod.keyframe_insert(data_path="rotation_euler", index=1, frame=frame_num)

                    frame_num += 1

            return theta_1_array, theta_2_array

        def reset_objects_in_collection(collection_name):
            collection = bpy.data.collections.get(collection_name)
            if collection:
                for obj in collection.objects:
                    obj.location = ORIGIN
                    obj.rotation_euler = ORIGIN
                    obj.animation_data_clear()

        def get_new_dp_parts(angle_1, angle_2, dp_num):
            def get_new_parts():
                def create_linked_duplicate(obj):
                    linked_duplicate = obj.copy()
                    if R or Scene2_U or Scene2_V or Scene2_V_30secs or A3 or S7 or B3:
                        linked_duplicate.data = obj.data.copy()
                    if Scene2_S:
                        if dp_num <= 16:
                            linked_duplicate.data = obj.data.copy()
                        else:
                            linked_duplicate.data = obj.data
                    if Scene2_SS or Scene2_U or Scene2_V or Scene2_V_30secs or CTS:
                        linked_duplicate.data = obj.data
                    linked_duplicate.animation_data_clear()

                    return linked_duplicate

                new_1st_rod = create_linked_duplicate(first_rodT)
                new_2nd_rod = create_linked_duplicate(second_rodT)
                new_bob1 = create_linked_duplicate(BOB1)

                new_1st_rod.name = f"first pen {self.cohort_num}_{dp_num}"
                new_2nd_rod.name = f"second pen {self.cohort_num}_{dp_num}"
                new_bob1.name = f"bob1 {self.cohort_num}_{dp_num}"

                turn_off_visibility([new_1st_rod, new_2nd_rod, new_bob1])

                return new_1st_rod, new_2nd_rod, new_bob1

            def assemble_dp_parts(first_rod, second_rod, bob1):
                bob1.parent = first_rod
                bob1.location.z = HINGE_LOCATION
                second_rod.constraints.new('COPY_LOCATION').target = bob1
                first_rod.rotation_euler[1] = angle_1 * DEGREES
                second_rod.rotation_euler[1] = angle_2 * DEGREES

                return first_rod, second_rod, bob1

            def put_in_their_collections(first_rod, second_rod, bob1):
                self.FirstPendulumsCollection.objects.link(first_rod)
                self.SecondPendulumsCollection.objects.link(second_rod)
                self.FirstBobCollection.objects.link(bob1)

            first_rod, second_rod, bob1 = assemble_dp_parts(*get_new_parts())

            put_in_their_collections(first_rod, second_rod, bob1)

            for obj in [first_rod, second_rod, bob1]:
                index_obj = get_object_index_in_collection(obj)
                if index_obj > hide_index:
                    if index_obj % 12 == 0:
                        continue
                    hide_viewport(obj)

            return first_rod, second_rod

        def create_angle_visualizer(index, first_rod, second_rod, angle_one_array, angle_two_array):
            # region Create the objects and assign pass_index
            angle_one_obj = bpy.data.objects.new(f"first rod angle {self.cohort_num}_{index}",
                                                 bpy.data.meshes.new(f"first rod angle {self.cohort_num}_{index}"))
            angle_two_obj = bpy.data.objects.new(f"second rod angle {self.cohort_num}_{index}",
                                                 bpy.data.meshes.new(f"second rod angle {self.cohort_num}_{index}"))
            elbow_one_obj = bpy.data.objects.new(f"first elbow {self.cohort_num}_{index}",
                                                 bpy.data.meshes.new(f"first elbow {self.cohort_num}_{index}"))
            elbow_two_obj = bpy.data.objects.new(f"second elbow {self.cohort_num}_{index}",
                                                 bpy.data.meshes.new(f"second elbow {self.cohort_num}_{index}"))
            vert_line_one_obj = bpy.data.objects.new(f"first vert line {self.cohort_num}_{index}",
                                                 bpy.data.meshes.new(f"first vert line {self.cohort_num}_{index}"))
            vert_line_two_obj = bpy.data.objects.new(f"second vert line {self.cohort_num}_{index}",
                                                 bpy.data.meshes.new(f"second vert line {self.cohort_num}_{index}"))
            turn_off_visibility(
                [angle_one_obj, angle_two_obj, elbow_one_obj, elbow_two_obj, vert_line_one_obj, vert_line_two_obj]
            )
            angle_one_obj.pass_index = index
            angle_two_obj.pass_index = index
            elbow_one_obj.pass_index = index
            elbow_two_obj.pass_index = index
            vert_line_one_obj.pass_index = index
            vert_line_two_obj.pass_index = index

            # region Copy the materials
            gold_rods_mat_copy_1 = bpy.data.materials.get("Yellow Rods").copy()
            gold_rods_mat_copy_2 = bpy.data.materials.get("Blue Rods").copy()
            gold_rods_mat_copy_1.name = f"{first_rod.name}"
            gold_rods_mat_copy_2.name = f"{second_rod.name}"
            first_rod.data.materials[0] = gold_rods_mat_copy_1
            second_rod.data.materials[0] = gold_rods_mat_copy_2

            first_rod_angle_mat = bpy.data.materials.get("First Rod Angle").copy()
            first_rod_angle_mat.name = f"{angle_one_obj.name}"
            second_rod_angle_mat = bpy.data.materials.get("Second Rod Angle").copy()
            second_rod_angle_mat.name = f"{angle_two_obj.name}"
            first_elbow_mat = bpy.data.materials.get("First Elbow").copy()
            first_elbow_mat.name = f"{elbow_one_obj.name}"
            second_elbow_mat = bpy.data.materials.get("Second Elbow").copy()
            second_elbow_mat.name = f"{elbow_two_obj.name}"
            first_vert_line_mat = bpy.data.materials.get("First Vert Line").copy()
            first_vert_line_mat.name = f"{vert_line_one_obj.name}"
            second_vert_line_mat = bpy.data.materials.get("Second Vert Line").copy()
            second_vert_line_mat.name = f"{vert_line_two_obj.name}"
            # endregion

            # region Link the objects to the collections
            self.Angle1Collection.objects.link(angle_one_obj)
            self.Elbow1Collection.objects.link(elbow_one_obj)
            self.Angle2Collection.objects.link(angle_two_obj)
            self.Elbow2Collection.objects.link(elbow_two_obj)
            self.FirstVertLineCollection.objects.link(vert_line_one_obj)
            self.SecondVertLineCollection.objects.link(vert_line_two_obj)
            # endregion

            for obj in [angle_one_obj, angle_two_obj, elbow_one_obj, elbow_two_obj, vert_line_one_obj, vert_line_two_obj]:
                if get_object_index_in_collection(obj) > hide_index:
                    hide_viewport(obj)

            # region configuring constraints
            child_of_constraint = angle_two_obj.constraints.new('CHILD_OF')
            child_of_constraint.target = second_rod
            angle_two_obj.location = second_rod.matrix_world.to_translation()

            # Set location influence
            child_of_constraint.use_location_x = True
            child_of_constraint.use_location_y = True
            child_of_constraint.use_location_z = True

            # Disable rotation and scale influence
            child_of_constraint.use_rotation_x = False
            child_of_constraint.use_rotation_y = False
            child_of_constraint.use_rotation_z = False
            child_of_constraint.use_scale_x = False
            child_of_constraint.use_scale_y = False
            child_of_constraint.use_scale_z = False
            # endregion

            # region Add Geom Nodes to objects
            create_and_configure_existing_geom_modifier_on_object(
                angle_one_obj,
                f'GeoNodesMod {index}',
                'Angle Visualizer',
                [6, 7, 13],
                [first_rod, index - 1, first_rod_angle_mat]
            )
            create_and_configure_existing_geom_modifier_on_object(
                angle_two_obj,
                f'GeoNodesMod {index}',
                'Angle Visualizer',
                [6, 7, 13],
                [second_rod, index - 1, second_rod_angle_mat]
            )
            create_and_configure_existing_geom_modifier_on_object(
                elbow_one_obj,
                f'ElbowNodes1',
                'Elbow Controller Geom',
                [7, 11, 13, 17],
                [index - 1, False, first_rod, first_elbow_mat]
            )
            create_and_configure_existing_geom_modifier_on_object(
                elbow_two_obj,
                f'ElbowNodes2',
                'Elbow Controller Geom',
                [7, 11, 13, 17],
                [index - 1, True, second_rod, second_elbow_mat]
            )
            create_and_configure_existing_geom_modifier_on_object(
                vert_line_one_obj,
                f'VertLineNodes1',
                'Add Vertical Dashes',
                [4, 6],
                [first_rod, first_vert_line_mat]
            )
            create_and_configure_existing_geom_modifier_on_object(
                vert_line_two_obj,
                f'VertLineNodes2',
                'Add Vertical Dashes',
                [4, 6],
                [second_rod, second_vert_line_mat]
            )
            create_and_configure_existing_geom_modifier_on_object(
                angle_one_obj,
                f'CopyLoc1',
                'copy location of another object',
                [2],
                [first_rod]
            )
            create_and_configure_existing_geom_modifier_on_object(
                elbow_one_obj,
                f'CopyLoc1',
                'copy location of another object',
                [2],
                [first_rod]
            )
            # endregion

        angle_1_inits = np.linspace(*self.angle_1_range, self.num_of_double_pendulums)
        angle_2_inits = np.linspace(*self.angle_2_range, self.num_of_double_pendulums)

        reset_objects_in_collection("BDP")

        for dp_num, (angle_1_init, angle_2_init) in tqdm(
                enumerate(zip(angle_1_inits, angle_2_inits), start=1),
                desc=f"Generating {self.num_of_double_pendulums} double pendulums",
        ):
            first_rod, second_rod = get_new_dp_parts(angle_1_init, angle_2_init, dp_num)
            first_rod.pass_index = dp_num
            second_rod.pass_index = dp_num
            theta_1_array, theta_2_array = simulate_double_pendulum(
                (angle_1_init, angle_2_init),
                first_rod,
                second_rod,
                self.duration,
            )
            bpy.context.scene.frame_set(1)
            if R or A3 or S7:
                create_angle_visualizer(dp_num, first_rod, second_rod, theta_1_array, theta_2_array)
            if Scene2_S and dp_num <= 16:
                create_angle_visualizer(dp_num, first_rod, second_rod, theta_1_array, theta_2_array)

        return self


class MainPanel(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "Double Pendulum Utilities"
    bl_idname = "PT_DOUBLE_PENDULUM_Utils"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "DP Tools"

    def draw(self, context):
        def create_row(label_text=None, label_icon=None, elements=[], condition=False):
            row = layout.row()
            row.label(text=label_text, icon=label_icon)
            for element in elements:
                if element.startswith('object.'):
                    row.operator(element)
                elif not hasattr(bpy.context.scene, element):
                    row.label(text=element)
                else:
                    row.prop(context.scene, element)
            if isinstance(condition, str):  # if condition is a property name
                row.enabled = not getattr(context.scene, condition)

        def draw_rows(row_configs):
            for config in row_configs:
                create_row(**config)

        active_obj = context.active_object
        active_obj_row = {
            "label_text": f"Active object: {active_obj.name}" if active_obj and active_obj.select_get() else "No active object",
            "label_icon": 'OBJECT_DATA' if active_obj and active_obj.select_get() else 'ERROR',
            "elements": ["object.add_tracer"]
        }
        row_configs = [
            {
                "label_text": "Parameters",
                "label_icon": 'COLLAPSEMENU',
                "elements": ["lifetime_in_frames", "tracer_start_frame", "tracer_end_frame"]
            },
            active_obj_row,
            {
                "label_text": "Parameters:",
                "label_icon": 'COLLAPSEMENU',
                "elements": ["num_of_double_pendulums", "DP_start_frame", "duration",
                             "type_manual_values"],
            },
            {
                "label_text": "Angle 1 range (in degrees)",
                "label_icon": 'TRACKING_REFINE_FORWARDS',
                "elements": ["angle_1_min", "angle_1_max"],
                "condition": "type_manual_values"
            },
            {
                "label_text": "Angle 2 range (in degrees)",
                "label_icon": 'TRACKING_REFINE_FORWARDS',
                "elements": ["angle_2_min", "angle_2_max"],
                "condition": "type_manual_values"
            },
            {
                "label_text": f"Create simulation from frame {context.scene.DP_start_frame} to "
                              f"{math.ceil(context.scene.DP_start_frame + context.scene.duration * fps - 1)}",
                "label_icon": "ADD",
                "elements": ["object.create_double_pendulums"],
            },
        ]

        layout = self.layout

        layout.label(text="APPLY TRACER TO SELECTED OBJECT", icon='ANIM')
        draw_rows(row_configs[:2])
        layout.label(text="")
        layout.label(text="GENERATE DOUBLE PENDULUMS", icon='DRIVER_ROTATIONAL_DIFFERENCE')
        draw_rows(row_configs[2:5])
        if context.scene.type_manual_values:
            row = layout.row()
            row.label(text="")  # empty label to push the other labels to the right
            row.label(text=f"Angle 1 min: {range_angle_1[0]}")
            row.label(text=f"Angle 1 max: {range_angle_1[1]}")
            row = layout.row()
            row.label(text="")  # empty label to push the other labels to the right
            row.label(text=f"Angle 2 min: {range_angle_2[0]}")
            row.label(text=f"Angle 2 max: {range_angle_2[1]}")
        draw_rows(row_configs[5:])


class AddTracerOperator(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "object.add_tracer"
    bl_label = "Add Tracer to Selected Object"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return len(context.selected_objects) == 1

    def execute(self, context):
        def get_global_coords_across_frames(start_frame, end_frame):
            obj = bpy.context.object
            current_frame = bpy.context.scene.frame_current

            points = []
            for frame in range(start_frame, end_frame + 1):
                bpy.context.scene.frame_set(frame)
                global_location = obj.matrix_world.to_translation()
                points.append(global_location)

            # Reset the current frame
            bpy.context.scene.frame_set(current_frame)

            return points

        def apply_geometry_nodes(obj):
            node_tree = bpy.data.node_groups["Tracer_Geom"]

            # Inputs
            node_tree.inputs[1].default_value = context.scene.lifetime_in_frames
            node_tree.inputs[2].default_value = context.scene.tracer_start_frame
            node_tree.inputs[3].default_value = context.scene.tracer_end_frame

            geo_nodes_mod = obj.modifiers.new(f"{bpy.context.object.name}_TracerModifier", 'NODES')
            geo_nodes_mod.node_group = node_tree

        def create_tracer(collection_name="Tracer Collection"):
            points_locations = get_global_coords_across_frames(context.scene.tracer_start_frame,
                                                               context.scene.tracer_end_frame)
            curve_data = bpy.data.curves.new(name="MyCurve", type='CURVE')
            spline = curve_data.splines.new(type='BEZIER')
            spline.bezier_points.add(len(points_locations) - 1)

            for i, point in enumerate(points_locations):
                spline.bezier_points[i].co = point
                spline.bezier_points[i].handle_left_type = 'AUTO'
                spline.bezier_points[i].handle_right_type = 'AUTO'

            spline.use_cyclic_u = False
            curve_object = bpy.data.objects.new(f"{bpy.context.object.name}_Tracer", curve_data)
            collection = bpy.data.collections.get(collection_name)
            collection.objects.link(curve_object)

            apply_geometry_nodes(collection.objects[-1])

        create_tracer()

        return {'FINISHED'}


class CreateDoublePendulumsOperator(bpy.types.Operator):
    bl_idname = "object.create_double_pendulums"
    bl_label = "Generate Double Pendulums"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
        if context.scene.type_manual_values:
            DoublePendulumAnimator().chaos_demonstration(
                range_angle_1,
                range_angle_2,
                context.scene.num_of_double_pendulums,
                context.scene.duration,
                context.scene.DP_start_frame
            )
        else:
            DoublePendulumAnimator().chaos_demonstration(
                [context.scene.angle_1_min, context.scene.angle_1_max],
                [context.scene.angle_2_min, context.scene.angle_2_max],
                context.scene.num_of_double_pendulums,
                context.scene.duration,
                context.scene.DP_start_frame
            )

        return {'FINISHED'}


def register():
    bpy.utils.register_class(AddTracerOperator)
    bpy.utils.register_class(CreateDoublePendulumsOperator)
    bpy.utils.register_class(MainPanel)

    def register_scene_properties():
        properties = {
            "tracer_start_frame": {"name": "Start Frame", "default": 1, "min": 1, "max": 10000},
            "tracer_end_frame": {"name": "End Frame", "default": 250, "min": 1, "max": 10000},
            "lifetime_in_frames": {"name": "Lifetime", "default": 20, "min": 1, "max": 250},
            "num_of_double_pendulums": {"name": "Count", "default": 10, "min": 1, "max": 20000},
            "DP_start_frame": {"name": "Start Frame", "default": 60, "min": 1, "max": 10000},
            "duration": {"name": "Duration", "default": 10.0, "min": 0.1, "max": 300.0},
            "angle_1_min": {"name": "Angle 1 min", "default": 30.0, "min": -180.0, "max": 180},
            "angle_1_max": {"name": "Angle 1 max", "default": 60.0, "min": -180.0, "max": 180.0},
            "angle_2_min": {"name": "Angle 2 min", "default": 120.0, "min": -180.0, "max": 180.0},
            "angle_2_max": {"name": "Angle 2 max", "default": 150.0, "min": -180.0, "max": 180.0},
            "type_manual_values": {"name": "Use Code To Set Angles (more accurate)", "default": False},
        }
        for prop, details in properties.items():
            if 'frame' in prop or prop == 'num_of_double_pendulums':
                setattr(bpy.types.Scene,
                        prop,
                        bpy.props.IntProperty(name=details["name"],
                                              default=details["default"],
                                              min=details["min"],
                                              max=details["max"]))
            elif 'angle' in prop:
                setattr(bpy.types.Scene,
                        prop,
                        bpy.props.FloatProperty(name=details["name"],
                                                default=details["default"],
                                                min=details["min"],
                                                max=details["max"],
                                                precision=5))
            elif prop == "type_manual_values":
                setattr(bpy.types.Scene,
                        prop,
                        bpy.props.BoolProperty(name=details["name"],
                                               default=details["default"]))
            else:
                setattr(bpy.types.Scene,
                        prop,
                        bpy.props.FloatProperty(name=details["name"],
                                                default=details["default"],
                                                min=details["min"],
                                                max=details["max"],
                                                precision=1))

    register_scene_properties()


def unregister():
    bpy.utils.unregister_class(MainPanel)
    bpy.utils.unregister_class(CreateDoublePendulumsOperator)
    bpy.utils.unregister_class(AddTracerOperator)

    def unregister_scene_properties():
        properties = [
            "tracer_start_frame",
            "tracer_end_frame",
            "lifetime_in_frames",
            "num_of_double_pendulums"
            "DP_start_frame",
            "duration",
            "angle_1_min",
            "angle_1_max",
            "angle_2_min",
            "angle_2_max",
            "type_manual_values",
        ]
        for prop in properties:
            delattr(bpy.types.Scene, prop)

    unregister_scene_properties()


if __name__ == "__main__":
    register()
