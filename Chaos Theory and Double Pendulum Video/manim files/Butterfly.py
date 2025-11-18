import itertools
import subprocess
import time

from numba import njit
from numpy import sin, cos, floor
from scipy.integrate import solve_ivp

from manim import *
from manim.utils.color import rgba_to_color

import mydebugger as deb

fps = 24
config.quality = 'high_quality'  # ['fourk_quality', 'production_quality', 'high_quality', 'medium_quality',
# 'low_quality', 'example_quality']
config.frame_rate = fps
config.preview = True
config.disable_caching = True


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
            method='DOP853'  # RK23 or RK45 or DOP853
        )

        return (
            np.array(ans.y[0]),
            np.array(ans.y[1])
        )

    def get_coors(self):
        t_values = self.get_t_values()
        angle1, angle2 = self.get_angles()

        ps2_array = self.ps1 + np.column_stack(
            (
                np.sin(angle1),
                -np.cos(angle1),
                np.zeros_like(angle1)
            )
        ) * self.rod_length

        ps3_array = ps2_array + np.column_stack(
            (
                np.sin(angle2),
                -np.cos(angle2),
                np.zeros_like(angle2)
            )
        ) * self.rod_length

        return (
            t_values,
            ps2_array,
            ps3_array
        )

    def build_double_pendulum(self, rod_width_stroke, rod_color, tracker):
        def update_pendulums(mob, dt):
            tracker_value = int(tracker.get_value())

            mob.set_points_as_corners([
                self.ps1,
                ps2[tracker_value],
                ps3[tracker_value]
            ])

        ps2, ps3 = self.get_coors()[1:]

        double_pendulum = VMobject(
            stroke_width=rod_width_stroke,
            color=rod_color,
            joint_type=LineJointType.BEVEL
        ).set_points_as_corners([
            self.ps1,
            ps2[0],
            ps3[0]
        ])

        double_pendulum.add_updater(update_pendulums)

        return double_pendulum


class OptimizedDoublePendulum(DoublePendulum):
    def __init__(
            self,
            ps1,
            t1,
            t2,
            t_span,
            rod_length=1.0
    ):
        super().__init__(
            ps1,
            t1,
            t2,
            t_span,
            rod_length
        )

    @staticmethod
    @njit
    def hamilton_rhs(g,
                     t1, t2,
                     p1, p2):
        a = (1 + sin(t1 - t2) ** 2)
        b = (p1 * p2 * sin(t1 - t2)) / a
        c = (p1 ** 2 + 2 * p2 ** 2 - 2 * p1 * p2 * cos(t1 - t2)) * sin(2 * (t1 - t2)) / (2 * a ** 2)

        rhs_t1 = (p1 - p2 * cos(t1 - t2)) / a
        rhs_t2 = (2 * p2 - p1 * cos(t1 - t2)) / a
        rhs_p1 = -2 * g * sin(t1) - b + c
        rhs_p2 = -g * sin(t2) + b - c

        return np.array([
            rhs_t1,
            rhs_t2,
            rhs_p1,
            rhs_p2
        ])

    def funcs(self, t, r):
        t1, t2, p1, p2 = r

        return self.hamilton_rhs(
            self.g,
            t1, t2,
            p1, p2)

    def get_angles(self):
        t_values = self.get_t_values()

        ans = solve_ivp(
            self.funcs,  # function to solve
            (0, self.t_span),  # time span
            (self.t1, self.t2, 0, 0),  # initial conditions
            t_eval=t_values,  # time points
            method='DOP853')  # RK23 or RK45 or DOP853

        return (
            np.array(ans.y[0]),
            np.array(ans.y[1])
        )

    def get_coors(self):
        t_values = self.get_t_values()
        angle1, angle2 = self.get_angles()

        ps2_array = self.ps1 + np.column_stack((
            np.sin(angle1),
            -np.cos(angle1),
            np.zeros_like(angle1)
        )) * self.rod_length

        ps3_array = ps2_array + np.column_stack((
            np.sin(angle2),
            -np.cos(angle2),
            np.zeros_like(angle2)
        )) * self.rod_length

        return (
            t_values,
            ps2_array,
            ps3_array
        )

    def build_double_pendulum(self, rod_width_stroke, rod_color, tracker):
        def update_pendulums(mob, dt):
            tracker_value = int(tracker.get_value())

            mob.set_points_as_corners([
                self.ps1,
                ps2[tracker_value],
                ps3[tracker_value]
            ])

        ps2, ps3 = self.get_coors()[1:]
        double_pendulum = VMobject(
            stroke_width=rod_width_stroke,
            color=rod_color,
            joint_type=LineJointType.BEVEL
        ).set_points_as_corners([
            self.ps1,
            ps2[0],
            ps3[0]
        ])

        double_pendulum.add_updater(update_pendulums)

        return double_pendulum


class CustomCoordinateSystem:
    def __init__(
            self,
            x_range,
            y_range,
            ax_size
    ):
        self.x_min = x_range[0]
        self.x_max = x_range[1]
        self.y_min = y_range[0]
        self.y_max = y_range[1]
        self.ax_size = ax_size
        self.x_step = round((self.x_max - self.x_min) / 8)
        self.y_step = round((self.y_max - self.y_min) / 8)

    def get_axes_and_labels(self):
        ax = Axes(
            x_range=[self.x_min, self.x_max, self.x_step],
            y_range=[self.y_min, self.y_max, self.y_step],
            x_length=self.ax_size,
            y_length=self.ax_size,
            tips=False,
            axis_config={"include_numbers": True}
        )
        x_label = ax.get_x_axis_label(Tex(
            r"$\theta_1$"
        ), edge=RIGHT)
        y_label = ax.get_y_axis_label(Tex(
            r"$\theta_2$"
        ), edge=UR)

        return VGroup(ax,
                      x_label,
                      y_label)


class Lissajous:
    def __init__(
            self,
            angle_1,
            angle_2,
            duration,
            axes,
            thickness=0.8,
            color=LIGHT_PINK
    ):
        self.angle_1 = angle_1
        self.angle_2 = angle_2
        self.duration = duration
        self.axes = axes
        self.thickness = thickness
        self.color = color

    def get_points(self):
        dp = OptimizedDoublePendulum(
            ORIGIN,
            self.angle_1 * DEGREES,
            self.angle_2 * DEGREES,
            self.duration)
        angle_1, angle_2 = dp.get_angles()

        return np.column_stack((
            angle_1,
            angle_2,
            np.zeros(angle_1.shape)
        ))

    def get_normalized_points(self):
        def find_equivalents(a):
            x = np.fmod(a, 2 * np.pi)
            y = np.fmod(a, np.pi)

            return (2 * y - x) * (1 / DEGREES)

        return np.array([find_equivalents(point) for point in self.get_points()])

    def get_indices(self):
        normalized_points = self.get_normalized_points()
        x_values, y_values = normalized_points[:, 0], normalized_points[:, 1]
        indices1 = np.flatnonzero(np.abs(np.diff(x_values)) > 90) + 1
        indices2 = np.flatnonzero(np.abs(np.diff(y_values)) > 90) + 1
        result = np.concatenate([indices1, indices2])

        return np.unique(np.sort(result))

    def get_diffs(self):
        return np.diff(
            np.concatenate((
                [0],
                self.get_indices(),
                [(self.duration * fps)]
            ))
        )

    def get_fixed_normalized_points(self):
        normalized_points = self.get_normalized_points()
        indices = self.get_indices()
        for i in indices:
            a, b = normalized_points[i - 1], normalized_points[i]
            axis = 0 if abs(a[0]) > abs(a[1]) else 1
            direction = 1 if a[axis] > 0 else -1
            a[axis] = direction * 180
            b[axis] = -direction * 180

        return normalized_points

    def get_points_on_ax(self):
        fixed_normalized_points = self.get_fixed_normalized_points()

        return np.array([
            self.axes.c2p([point]) for point in fixed_normalized_points
        ])

    def get_lissa_parts(self):
        points_on_ax = self.get_points_on_ax()
        indices = self.get_indices()
        lissa_parts = []

        last_i = 0
        for i in indices:
            lissa_parts.append(
                VMobject()
                .set_points_smoothly(points_on_ax[last_i:i])
                .set_color(self.color)
                .set_stroke_width(self.thickness)
            )

            last_i = i

        lissa_parts.append(
            VMobject()
            .set_points_smoothly(points_on_ax[last_i:])
            .set_color(self.color)
            .set_stroke_width(self.thickness))

        return lissa_parts

    def get_whole_lissajous(self):
        return VMobject().add(
            *self.get_lissa_parts()
        )

    def get_contents(self):
        return (
            self.get_lissa_parts(),
            self.get_diffs(),
            self.get_points_on_ax(),
            self.get_indices()
        )

    def update_angles(self, angle_1, angle_2):
        self.angle_1 = angle_1
        self.angle_2 = angle_2


class SingleLissajous(Scene):
    def construct(self):
        angles = (40, 8)
        elapsed_time = 500
        duration = 10

        # for the coordinate system
        x_range = (-50, 50)
        y_range = (-50, 50)
        axes_size = 7.5
        dot_color = BLUE_C
        lissajous_thickness = 0.6

        coord_sys = CustomCoordinateSystem(
            x_range,
            y_range,
            axes_size
        ).get_axes_and_labels()
        axes = coord_sys[0]

        lissajous = Lissajous(
            angles[0],
            angles[1],
            elapsed_time,
            axes,
            lissajous_thickness
        )
        lissajous_parts, diffs, points_on_ax, indices = lissajous.get_contents()
        lissajous_animation = [
            Create(curve,
                   rate_func=linear,
                   run_time=(diff / fps) * (duration / elapsed_time)
                   )
            for curve, diff in zip(lissajous_parts, diffs)
        ]

        dot = Dot(
            points_on_ax[0],
            color=dot_color,
            fill_opacity=0.75
        )

        index, index_count = 0, 0

        def move_dot(dot):
            nonlocal index, index_count
            if index in indices:
                index_count += 1
            dot.move_to(
                lissajous_parts[index_count].get_end()
            )
            index += 1

        self.add(coord_sys)
        self.wait()
        self.add(dot)
        dot.add_updater(move_dot)
        self.play(AnimationGroup(*lissajous_animation, lag_ratio=1))
        dot.remove_updater(move_dot)
        self.wait()


class UpdatingLissajous(Scene):
    def construct(self):
        initial_angles = (5, 30)
        final_angles = (5, 35)
        elapsed_time = 60
        tracker_duration = 30

        # Coordinate system settings
        x_range = (-45, 45)
        y_range = (-45, 45)
        ax_size = 7.5

        # Double pendulum settings
        rod_length = 2
        rod_width_stroke = 2
        rod_color = BLUE_C
        rate = linear

        def get_pendulum_points(
                t1_val, t2_val,
                rod_length
        ):
            ps1 = np.array([0, 0, 0])
            ps2 = np.array([
                sin(t1_val) * rod_length,
                -cos(t1_val) * rod_length,
                0
            ])
            ps3 = ps2 + np.array([
                sin(t2_val) * rod_length,
                -cos(t2_val) * rod_length,
                0
            ])

            return ps1, ps2, ps3

        def update_double_pendulum(mob):
            angle_1, angle_2 = angle_pairs[int(tracker.get_value())]
            mob.set_points_as_corners(
                get_pendulum_points(
                    angle_1 * DEGREES,
                    angle_2 * DEGREES,
                    rod_length
                )
            )

        def update_lissajous(mob):
            angles_1, angle_2 = angle_pairs[int(tracker.get_value())]
            mob.become(Lissajous(
                angles_1,
                angle_2,
                elapsed_time,
                cs[0]
            ).get_whole_lissajous())

        cs = CustomCoordinateSystem(
            x_range,
            y_range,
            ax_size
        ).get_axes_and_labels()

        tracker = ValueTracker(0)
        angle_pairs = np.column_stack((
            np.linspace(
                initial_angles[0],
                final_angles[0],
                fps * elapsed_time
            ),
            np.linspace(
                initial_angles[1],
                final_angles[1],
                fps * elapsed_time
            )
        ))

        double_pendulum = VMobject(
            stroke_width=rod_width_stroke,
            color=rod_color,
            joint_type=LineJointType.BEVEL
        )
        angle1, angle2 = angle_pairs[int(tracker.get_value())]
        double_pendulum.set_points_as_corners(
            get_pendulum_points(
                angle1 * DEGREES,
                angle2 * DEGREES,
                rod_length
            )
        )
        double_pendulum.add_updater(update_double_pendulum)

        lissa = Lissajous(
            *initial_angles,
            elapsed_time,
            cs[0]
        ).get_whole_lissajous()
        lissa.add_updater(update_lissajous)

        self.add(double_pendulum, cs, lissa)
        self.wait()
        self.play(
            tracker.animate.set_value(len(angle_pairs) - 1),
            rate_func=rate,
            run_time=tracker_duration
        )
        self.wait()


class PendulumsSimulation(Scene):
    def construct(self):
        angle_1_range = -180, 180
        angle_2_range = -180, 180
        num_of_rows, num_of_columns = 30, 30
        elapsed_time = 10
        duration = 10

        axes_size = 7

        rod_width_stroke = 1
        rod_color = YELLOW_A

        coord_sys = CustomCoordinateSystem(
            (angle_1_range[0], angle_1_range[1]),
            (angle_2_range[0], angle_2_range[1]),
            axes_size
        ).get_axes_and_labels()
        axes = coord_sys[0]

        tracker = ValueTracker(0)

        x = np.linspace(
            angle_1_range[0],
            angle_1_range[1],
            num_of_rows + 2
        )[1:-1]
        y = np.linspace(
            angle_2_range[0],
            angle_2_range[1],
            num_of_columns + 2
        )[1:-1]
        rod_length = np.abs(
            axes.c2p(
                0, (y[1] - y[0]) / 2, 0
            )[1]
            - axes.c2p(
                0, 0, 0
            )[1]
        )

        grid_of_points = np.array(
            list(itertools.product(x, y)),
            dtype=[('x', float), ('y', float)]
        )
        total = len(grid_of_points)
        double_pendulums = []
        for i, point in enumerate(grid_of_points, 1):
            axes_coords = axes.c2p(point[0], point[1], 0)
            dp = OptimizedDoublePendulum(
                axes_coords,
                point[0] * DEGREES,
                point[1] * DEGREES,
                elapsed_time,
                rod_length=rod_length
            )
            moving_dp = dp.build_double_pendulum(rod_width_stroke, rod_color, tracker)
            double_pendulums.append(moving_dp)
            deb.progress_bar(i, total)

        self.add(coord_sys)
        self.wait()
        self.play(Create(VGroup(*double_pendulums)))
        self.wait()
        self.play(
            tracker.animate.set_value(fps * elapsed_time - 1),
            rate_func=linear,
            run_time=duration
        )
        self.wait()


class PixelPendulums:
    def __init__(self,
                 angle_1_range,
                 angle_2_range,
                 num_of_x_pixels,
                 num_of_y_pixels,
                 duration,
                 init_angle_1s,
                 init_angle_2s
                 ):
        self.angle_1_range = angle_1_range
        self.angle_2_range = angle_2_range
        self.num_of_x_pixels = num_of_x_pixels
        self.num_of_y_pixels = num_of_y_pixels
        self.duration = duration
        self.init_angle_1s = init_angle_1s
        self.init_angle_2s = init_angle_2s

    @staticmethod
    def zip_angles_list(angles_list):
        angles_list = np.degrees(angles_list)

        return (
            angles_list[:, 0].T,
            angles_list[:, 1].T
        )

    def get_angles_for_every_frame(self):
        """
        Takes in all initial configurations of the double pendulum in the grid and spits out 2 arrays containing the angles for each rod for every frame throughout the animation.

        :return: two numpy arrays of arrays containing angle values \n shape(fps*duration, num_of_x_pixels*num_of_y_pixels), shape(fps*duration, num_of_x_pixels*num_of_y_pixels)
        """
        angles_list = []
        i = 0
        total = (
                len(self.init_angle_1s)
                * len(self.init_angle_2s)
        )

        for angle2 in self.init_angle_2s:
            for angle1 in self.init_angle_1s:
                pendulum_angles = OptimizedDoublePendulum(
                    ORIGIN,
                    angle1 * DEGREES,
                    angle2 * DEGREES,
                    self.duration
                ).get_angles()
                angles_list.append(pendulum_angles)
                i += 1
                deb.progress_bar(i, total)

        return self.zip_angles_list(np.array(angles_list))


class PixelVisuals(Scene):
    def construct(self):
        angle_1_range = (-180, 180)
        angle_2_range = (-180, 180)
        num_of_x_pixels = 200
        num_of_y_pixels = 200
        duration = 10

        @njit
        def angle_to_color(angle_1, angle_2):  # must return values in the range [0, 255];
            """
            This function takes in two angles in degrees and returns the corresponding RGB color values in a list.

            :return: R list containing RGB color values in the range of [0, 255].
            """
            theta_1 = angle_1 * DEGREES
            theta_2 = angle_2 * DEGREES

            return np.array([
                floor((1 + cos(theta_2 + np.pi)) * 127.5),
                floor((1 + sin(theta_1 + np.pi) * sin(theta_2 + np.pi)) * 127.5),
                floor((1 + cos(theta_1 + np.pi)) * 127.5),
                255
            ], dtype=np.uint8)

        def get_pixel_grid(angle_1_for_one_frame, angle_2_for_one_frame):
            """
            This function takes in arrays of angles of all possible configurations of the double pendulum within the specified range of the grid at a single point in time.
            It reads the grid in the LEFT to RIGHT + UP to DOWN direction.

            :return: matrix containing RGBA values corresponding to the angles in the given arrays
            """
            pixel_grid = np.array([
                angle_to_color(angle_1, angle_2)
                for angle_1, angle_2 in zip(angle_1_for_one_frame, angle_2_for_one_frame)
            ])

            return np.array(pixel_grid, dtype=np.uint8)

        def get_pixel_grid_for_every_frame(angle_1_for_every_frame, angle_2_for_every_frame):
            """
            Returns a numpy array of pixel grid values array for each frame in the animation
            """
            pixel_grid_arrays = [
                get_pixel_grid(angle_1_for_one_frame,
                               angle_2_for_one_frame)
                for angle_1_for_one_frame, angle_2_for_one_frame in
                zip(angle_1_for_every_frame,
                    angle_2_for_every_frame)  # problem is prob here
            ]
            return np.array(pixel_grid_arrays)

        def get_temp_image():
            """
            :return: an ImageMobject() placeholder of RGBA values
            """
            temp_pixel_grid = pixel_grid_for_every_frame[0]

            return ImageMobject(
                temp_pixel_grid,
                scale_to_resolution=config.pixel_height * (1 / (1000 / num_of_x_pixels)),
                resampling_algorithm=RESAMPLING_ALGORITHMS['nearest']
            )

        def pixel_evolution(mob):
            frame_num = int(tracker.get_value())
            mob.pixel_array = pixel_grid_for_every_frame[frame_num]

        init_angle_1s = np.linspace(
            angle_1_range[0],
            angle_1_range[1],
            num_of_x_pixels
        )
        init_angle_2s = np.linspace(
            angle_2_range[1],
            angle_2_range[0],
            num_of_y_pixels
        )

        pixels_of_pendulums = PixelPendulums(
            angle_1_range,
            angle_2_range,
            num_of_x_pixels,
            num_of_y_pixels,
            duration,
            init_angle_1s,
            init_angle_2s
        )
        angle_1_for_every_frame, angle_2_for_every_frame = pixels_of_pendulums.get_angles_for_every_frame()

        pixel_grid_for_every_frame = get_pixel_grid_for_every_frame(
            angle_1_for_every_frame,
            angle_2_for_every_frame
        ).reshape((
            fps * duration,
            num_of_y_pixels,
            num_of_x_pixels,
            4
        ))

        tracker = ValueTracker(0)
        visualizer = get_temp_image()

        self.add(visualizer)
        visualizer.add_updater(pixel_evolution)
        self.wait()
        self.play(
            tracker.animate.set_value(fps * duration - 1),
            rate_func=linear,
            run_time=duration)
        self.wait()


class FlipVisuals(Scene):
    def construct(self):
        # visualizer inits
        angle_1_range = (-180, 180)
        angle_2_range = (-180, 180)
        num_of_x_pixels = 20
        num_of_y_pixels = 20
        nth_flip_to_evaluate = 4

        visualizer_elapsed_time = 24
        visualizer_transformation_duration = 6

        # color basis inits
        length_of_gradient_base = 900
        width_of_gradient_base = 150
        color_basis_transformation_duration = 2

        # other inits
        linearity = 1 / visualizer_elapsed_time
        dur_offset_of_arrow = (fps * visualizer_elapsed_time) / (fps * visualizer_elapsed_time - 1)

        init_angle_1s = np.linspace(angle_1_range[0], angle_1_range[1], num_of_x_pixels)
        init_angle_2s = np.linspace(angle_2_range[1], angle_2_range[0], num_of_y_pixels)

        visualizer_tracker = ValueTracker(0)
        color_basis_tracker = ValueTracker(0)

        r_gradient = (0, 255, 255, 0, 0, 0, 255)
        g_gradient = (0, 0, 255, 255, 255, 0, 0)
        b_gradient = (0, 0, 0, 0, 255, 255, 255)

        def is_complete_visualization(angle_1_range=angle_1_range, angle_2_range=angle_2_range):
            if angle_1_range == (-180, 180) and angle_2_range == (-180, 180):
                return True
            return False

        def get_linear_to_log_interp():
            def get_linear_interp(num_of_vals):
                return np.linspace(0, 1, num_of_vals)

            def get_logarithmic_interp(num_of_vals, linearity=linearity):
                offset = linearity * num_of_vals
                log_interp = np.geomspace(offset, num_of_vals, num_of_vals) - offset
                log_interp = 1 - log_interp / (num_of_vals - offset)

                return log_interp[::-1]

            linear_interp = get_linear_interp(length_of_gradient_base)
            log_interp = get_logarithmic_interp(length_of_gradient_base)
            linear_to_log_interp = np.array([
                np.linspace(lin, log, int(fps * color_basis_transformation_duration))
                for lin, log in zip(linear_interp, log_interp)
            ]).T

            return linear_to_log_interp

        def get_rgb_gradient(interpolate, for_visualizer=False):
            def piecewise_linear_interpolation(i, gradient_values):
                x_values = np.linspace(0, 1, 7)
                return np.interp(i, x_values, gradient_values)

            def get_logarithmic_interp(num_of_vals, linearity=linearity):
                offset = linearity * num_of_vals
                log_interp = np.geomspace(offset, num_of_vals, num_of_vals) - offset
                log_interp = log_interp / (num_of_vals - offset)

                return log_interp

            def piecewise_logarithmic_interpolation(i, gradient_values):
                x_values = get_logarithmic_interp(len(gradient_values))
                return np.interp(i, x_values, gradient_values)

            def interpolate_colors(gradient_values, interpolate):
                if not for_visualizer:
                    return [piecewise_linear_interpolation(i, gradient_values) for i in interpolate]

                return [piecewise_logarithmic_interpolation(i, gradient_values) for i in interpolate]

            r_values = interpolate_colors(r_gradient, interpolate)
            g_values = interpolate_colors(g_gradient, interpolate)
            b_values = interpolate_colors(b_gradient, interpolate)

            return np.array([
                np.array([r, g, b, 255], dtype=np.uint8)
                for r, g, b in zip(r_values, g_values, b_values)
            ])

        def get_color_base_mobject(interpolate):
            rgb_gradient = get_rgb_gradient(interpolate)
            rgb_gradient = rgb_gradient.reshape(
                (length_of_gradient_base, 1, 4)
            )
            rgb_gradient = np.tile(
                rgb_gradient,
                (1, width_of_gradient_base, 1)
            )[::-1]

            return ImageMobject(
                rgb_gradient,
                scale_to_resolution=config.pixel_height
            ).shift(5 * RIGHT)

        def get_indices_to_eval_color(angle_1_for_every_frame, angle_2_for_every_frame):
            def get_normalized_angles(angle_array):
                def find_equivalents(angle):
                    x = np.fmod(angle, 2 * 180.0)
                    y = np.fmod(angle, 180.0)

                    return 2 * y - x

                return np.array([
                    find_equivalents(angle) for angle in angle_array
                ])

            def compute_indices(angle_evolution, threshold=270):
                angle_evolution = get_normalized_angles(angle_evolution)
                angle_diffs = np.abs(np.diff(angle_evolution))

                return np.where(angle_diffs > threshold)[0] + 1

            def get_nth_flip(arr):
                if len(arr) < nth_flip_to_evaluate:
                    return None

                return arr[nth_flip_to_evaluate - 1]

            angle_1_evolutions = angle_1_for_every_frame.T
            angle_2_evolutions = angle_2_for_every_frame.T

            indices_to_eval_color = []
            for angle_1_evolution, angle_2_evolution in zip(angle_1_evolutions, angle_2_evolutions):
                indices1 = compute_indices(angle_1_evolution)
                indices2 = compute_indices(angle_2_evolution)

                pixel_eval = np.unique(np.sort(np.concatenate(
                    (indices1, indices2), axis=0
                )))
                pixel_eval = get_nth_flip(pixel_eval)
                indices_to_eval_color.append(pixel_eval)

            return np.array(indices_to_eval_color)

        def get_pixel_evolution(index):
            def modify_array_trail(input_array, flip_color):
                if index is not None:
                    input_array[index:, :] = flip_color
                return input_array

            def get_flip_color():
                if index is None:
                    return np.array([255, 255, 255, 255], dtype=np.uint8)

                return np.array(
                    get_rgb_gradient([index / (fps * visualizer_elapsed_time - 1)], for_visualizer=True)[0],
                    dtype=np.uint8
                )

            pixel_evolution = np.full(
                (fps * visualizer_elapsed_time, 4),
                [255, 255, 255, 255],
                dtype=np.uint8
            )

            return modify_array_trail(pixel_evolution, get_flip_color())

        def get_flip_visuals_for_every_frame(angle_1_for_every_frame, angle_2_for_every_frame):
            def set_border_of_2d_array(array, value=0):
                array[0, :] = value  # Set top border
                array[-1, :] = value  # Set bottom border
                array[:, 0] = value  # Set left border
                array[:, -1] = value  # Set right border

                return array

            indices_to_eval_color = get_indices_to_eval_color(angle_1_for_every_frame, angle_2_for_every_frame)
            if is_complete_visualization():
                indices_to_eval_color = set_border_of_2d_array(
                    indices_to_eval_color.reshape((num_of_y_pixels, num_of_x_pixels))
                ).flatten()

            all_pixel_evolutions = [get_pixel_evolution(idx_flip) for idx_flip in indices_to_eval_color]
            all_pixel_evolutions = np.array(all_pixel_evolutions, dtype=np.uint8).reshape(
                (num_of_y_pixels,
                 num_of_x_pixels,
                 fps * visualizer_elapsed_time,
                 4)
            )

            return np.transpose(all_pixel_evolutions, (2, 0, 1, 3))

        def get_temp_image_for_color_basis():
            temp_pixel_grid = np.zeros(
                (length_of_gradient_base, width_of_gradient_base, 4),
                dtype=np.uint8
            )
            return ImageMobject(
                temp_pixel_grid,
                scale_to_resolution=config.pixel_height
            ).shift(5 * RIGHT)

        def get_temp_image_for_visualizer():
            temp_pixel_grid = np.full(
                (num_of_y_pixels, num_of_x_pixels, 4),
                255,
                dtype=np.uint8
            )

            return ImageMobject(
                temp_pixel_grid,
                scale_to_resolution=config.pixel_height * (1 / (1000 / num_of_x_pixels)),
                resampling_algorithm=RESAMPLING_ALGORITHMS['nearest']
            )

        def pixel_evolution(mob):
            frame_num = round(visualizer_tracker.get_value())
            mob.pixel_array = all_flip_visuals_for_every_frame[frame_num]

        def color_basis_evolution(mob):
            frame_num = round(color_basis_tracker.get_value())
            mob.become(all_color_bases[frame_num])

        def update_arrow_position(mob):
            val = (visualizer_tracker.get_value()
                   * dur_offset_of_arrow) / (fps * visualizer_elapsed_time)
            mob.next_to(
                color_basis.get_bottom(),
                LEFT,
                buff=MED_SMALL_BUFF * 2
            )
            mob.shift(
                (color_basis.get_top() - color_basis.get_bottom()) * val
            )

            rgba = get_rgb_gradient([val], for_visualizer=True)[0] / 255
            mob.set_color(rgba_to_color(rgba))

        # building the color basis gradient on the right side
        all_color_bases = [get_color_base_mobject(interpolate) for interpolate in get_linear_to_log_interp()]
        color_basis = get_temp_image_for_color_basis()
        color_basis.add_updater(color_basis_evolution)

        # building the flip visuals itself
        pixels_of_pendulums = PixelPendulums(
            angle_1_range,
            angle_2_range,
            num_of_x_pixels,
            num_of_y_pixels,
            visualizer_elapsed_time,
            init_angle_1s,
            init_angle_2s
        )
        angle_1_for_every_frame, angle_2_for_every_frame = pixels_of_pendulums.get_angles_for_every_frame()
        all_flip_visuals_for_every_frame = get_flip_visuals_for_every_frame(
            angle_1_for_every_frame, angle_2_for_every_frame)

        visualizer = get_temp_image_for_visualizer()
        visualizer.add_updater(pixel_evolution)

        pointer = Arrow(
            ORIGIN,
            RIGHT,
            color=BLACK
        ).next_to(
            color_basis.get_bottom(),
            LEFT,
            buff=MED_SMALL_BUFF * 2
        )
        pointer.add_updater(update_arrow_position)

        self.add(color_basis)
        self.wait()
        self.play(
            color_basis_tracker.animate.set_value(fps * color_basis_transformation_duration - 1),
            rate_func=linear,
            run_time=color_basis_transformation_duration
        )
        self.wait()
        self.add(visualizer, pointer)
        self.wait()
        self.play(
            visualizer_tracker.animate.set_value(fps * visualizer_elapsed_time - 1),
            rate_func=linear,
            run_time=visualizer_transformation_duration)
        self.wait()


if __name__ == "__main__":
    Scenes = [
        "SingleLissajous",
        "UpdatingLissajous",
        "PendulumsSimulation",
        "PixelVisuals",
        "FlipVisuals"
    ]
    scene_to_render = Scenes[0]

    a = time.perf_counter()

    print(' '.join(['manim', 'Butterfly.py', scene_to_render]))
    subprocess.run(['manim', 'Butterfly.py', scene_to_render])

    b = time.perf_counter()

    print(f"\nRendering the '{scene_to_render}' scene took {b - a:.2f} seconds.")

