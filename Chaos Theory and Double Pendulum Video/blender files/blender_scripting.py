import bpy

from numpy import sin, cos
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit
from manim import DEGREES, ORIGIN

from mydebugger import timer
from my_blender_utils import clear_single_keyframes


scene = bpy.context.scene
fps = scene.render.fps


class SinglePendulum:
    g = 9.81
    l = g / (np.pi ** 2)

    def __init__(
            self,
            ps1,
            t,
            t_span,
            rod_length=1.0
    ):
        self.ps1 = np.array(ps1)
        self.t = t
        self.t_span = t_span
        self.rod_length = rod_length

    @staticmethod
    @njit
    def diffeq_rhs(t, w, g, l):
        return np.array([
            w,
            (-g / l) * np.sin(t)
        ])

    def funcs(self, t, r):
        t, w = r

        return self.diffeq_rhs(t, w, self.g, self.l)

    def get_t_values(self):
        return np.arange(0, self.t_span, 1 / fps)

    def get_angles(self):
        t_values = self.get_t_values()

        ans = solve_ivp(
            self.funcs,  # function to solve
            (0, self.t_span),  # time span
            (self.t, 0),  # initial conditions
            t_eval=t_values,  # time points
            method='DOP853',  # RK23 or RK45 or DOP853
            rtol=1e-10,
            atol=1e-12
        )

        return np.array(ans.y[0])


class DoublePendulum:
    g = 9.81
    m1 = m2 = 1  # mass of the 2 bobs
    l1 = l2 = 1  # length of the 2 rods

    def __init__(
            self,
            ps1,
            t1,
            t2,
            t_span,
            rod_length=1.0
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
        return np.arange(0, self.t_span, 1 / fps)

    def get_angles(self):
        t_values = self.get_t_values()

        ans = solve_ivp(
            self.funcs,  # function to solve
            (0, self.t_span),  # time span
            (self.t1, self.t2, 0, 0),  # initial conditions
            t_eval=t_values,  # time points
            method='DOP853',  # RK23 or RK45 or DOP853
            rtol=1e-10,
            atol=1e-12
        )

        return (
            np.array(ans.y[0]),
            np.array(ans.y[1])
        )


class SinglePendulumAnimator:
    def __init__(self, angle, duration, starting_frame):
        self.angle = angle
        self.duration = duration
        self.starting_frame = starting_frame
        clear_single_keyframes("single_pendulum", 'rotation_euler', 1)

    @timer
    def order_demonstration(self):
        angle_list = SinglePendulum(ORIGIN, self.angle, self.duration).get_angles()
        single_pendulum = bpy.data.objects['single_pendulum']
        clear_single_keyframes("single_pendulum", 'rotation_euler', 1)

        frame_num = self.starting_frame
        print(f"last angle: {-1*angle_list[-1]/DEGREES}")
        for angle in -1*angle_list:
            single_pendulum.rotation_euler[1] = angle
            single_pendulum.keyframe_insert(data_path="rotation_euler", index=1, frame=frame_num)
            frame_num += 1


class DPFirstScene:
    def __init__(
            self,
            angle_1,
            angle_2,
            first_rod,
            second_rod,
            duration,
            starting_frame
    ):
        self.angle_1 = angle_1
        self.angle_2 = angle_2
        self.first_rod = bpy.data.objects[first_rod]
        self.second_rod = bpy.data.objects[second_rod]
        print(self.first_rod, self.second_rod)
        self.duration = duration
        self.starting_frame = starting_frame
        clear_single_keyframes("second_pendulum", 'rotation_euler', 1)

    @timer
    def chaos_demonstration(self):
        def simulate_double_pendulum(first_rod, second_rod, duration, starting_frame_num):
            double_pendulum = DoublePendulum(
                ORIGIN,
                self.angle_1,
                self.angle_2,
                duration
            )
            theta_1_array, theta_2_array = double_pendulum.get_angles()
            frame_num = starting_frame_num
            for theta_1, theta_2 in zip(-1*theta_1_array, -1 * theta_2_array):
                first_rod.rotation_euler[1] = theta_1
                first_rod.keyframe_insert(data_path="rotation_euler", index=1, frame=frame_num)

                second_rod.rotation_euler[1] = theta_2
                second_rod.keyframe_insert(data_path="rotation_euler", index=1, frame=frame_num)

                frame_num += 1

        simulate_double_pendulum(
            self.first_rod,
            self.second_rod,
            self.duration,
            self.starting_frame)


if __name__ == "__main__":
    def run_sp():
        SinglePendulumAnimator(30 * DEGREES, duration=900/fps, starting_frame=1).order_demonstration()

    def run_dp_first_scene():
        DPFirstScene(
            angle_1=-80 * DEGREES,
            angle_2=160 * DEGREES,
            first_rod="single_pendulum",
            second_rod="second_pendulum",
            duration=50,
            starting_frame=1
        ).chaos_demonstration()

    # run_sp()
    run_dp_first_scene()

