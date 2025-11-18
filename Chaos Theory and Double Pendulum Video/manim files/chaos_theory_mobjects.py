from __future__ import annotations

from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL

from manim_config import *
from chaos_theory_base_classes import *
from color_utils_chaos_theory import *
from mydebugger import timer, delete_memmap_files
from tqdm import tqdm


main_cs = DynamicAxes(
    x_range=(-180, 180),
    y_range=(-180, 180),
    x_length=pixel_length / SCENE_PIXELS,
    y_length=pixel_length / SCENE_PIXELS,
    x_is_in_degrees=True,
    y_is_in_degrees=True,
    font_size_x=18,
    font_size_y=18,
    use_constant_tick_length=True,
    x_line_to_number_buff=0.125,
    y_line_to_number_buff=0.125,
    tick_length=0.01
)


class DoublePendulum:
    def __init__(
            self,
            angle_pair: tuple[float, float],  # angles in degrees
    ):
        self.init_angle_1 = angle_pair[0]
        self.init_angle_2 = angle_pair[1]
        self.length_1: float = 1
        self.length_2: float = 1
        self.double_pendulum: DoublePendulumConstructor | None = None
        self.data_factory: ComputeDoublePendulumSimulation | None = None
        self.rate_func = None

    def create_double_pendulum(
            self,
            length_1,
            length_2,
            first_rod_color=FIRST_ROD_COLOR,
            second_rod_color=SECOND_ROD_COLOR,
            **kwargs
    ) -> UnionDoublePendulumConstructor:
        self.length_1 = length_1
        self.length_2 = length_2
        self.double_pendulum = UnionDoublePendulumConstructor(
            length_1=self.length_1,
            length_2=self.length_2,
            angle_pair=(self.init_angle_1, self.init_angle_2),
            first_rod_color=first_rod_color,
            second_rod_color=second_rod_color,
            **kwargs
        )

        return self.double_pendulum

    def get_data_factory(
            self,
            duration: float,
            rate_func: Callable[[float], float] = linear,
            override_fps: int | None = None,
    ):
        self.rate_func = rate_func
        self.data_factory = ComputeDoublePendulumSimulation(
            angle_pair=(self.init_angle_1, self.init_angle_2),
            t_span=duration,
            t_eval_rate_func=rate_func,
            override_fps=override_fps
        )
        return self.data_factory

    def copy(self):
        return DoublePendulum(angle_pair=(self.init_angle_1, self.init_angle_2))


class DoublePendulumConstructor(VGroup, metaclass=ConvertToOpenGL):  # breaks down when either angle is 0 degrees
    def __init__(
            self,
            length_1: float,
            length_2: float,
            angle_pair: tuple[float, float] | np.ndarray,
            first_rod_color=FIRST_ROD_COLOR,
            second_rod_color=SECOND_ROD_COLOR,
            include_bobs: bool = True,
            rod_stroke_width: float | None = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        assert length_1 == length_2, "lengths must be equal"
        min_length = 0.06
        self.length_1 = max(length_1, min_length)
        self.length_2 = max(length_2, min_length)
        self._angle_pair = angle_pair
        self.bobs_radius = max(self.length_1 / 7, 0.015)
        self.first_rod_color = first_rod_color
        self.second_rod_color = second_rod_color
        self.rod_stroke_width = max(rod_stroke_width if rod_stroke_width else self.length_1 * 10, 1.25)
        self.include_bobs = include_bobs
        self.angle_visualizer: AngleVisualizer | None = None

        self.rod1 = Line(ORIGIN,
                         np.array([0, -self.length_1, 0]),
                         stroke_width=self.rod_stroke_width,
                         color=self.first_rod_color,
                         name="rod1")
        self.rod2 = Line(np.array([0, -self.length_1, 0]),
                         np.array([0, -self.length_1 - self.length_2, 0]),
                         stroke_width=self.rod_stroke_width,
                         color=self.second_rod_color,
                         name="rod2")
        self.add(self.rod2, self.rod1)
        if self.include_bobs:
            self.bob1 = Circle(radius=self.bobs_radius,
                               fill_color=self.first_rod_color,
                               fill_opacity=1,
                               stroke_width=0,
                               name="bob1").move_to(self.rod1.get_end())
            self.bob2 = Circle(radius=self.bobs_radius,
                               fill_color=self.second_rod_color,
                               fill_opacity=1,
                               stroke_width=0,
                               name="bob2").move_to(self.rod2.get_end())
            self.add(self.bob1, self.bob2)
        self.update_angles()

    @property
    def angle_pair(self):
        return self._angle_pair

    @angle_pair.setter
    def angle_pair(self, new_angle_pair: tuple[float, float]):
        self._angle_pair = new_angle_pair
        self.update_angles()
        if self.angle_visualizer:
            self.angle_visualizer.update_angle_visualizer()

    def update_angles(self):
        angle_1 = self._angle_pair[0]
        angle_2 = self._angle_pair[1]
        angle_offset = 270 * DEGREES

        def rotate_rod(rod, angle):
            rod.rotate(-rod.get_angle() + angle * DEGREES + angle_offset, about_point=rod.get_start())

        rotate_rod(self.rod1, angle_1)
        rotate_rod(self.rod2, angle_2)

        self.rod2.move_to(self.rod1.get_end() + self.rod2.get_vector() / 2)
        if self.include_bobs:
            self.bob1.move_to(self.rod1.get_end())
            self.bob2.move_to(self.rod2.get_end())

    def create_angle_visualizer(self, angle_buff: float = 0.1):
        if self.angle_visualizer is None:
            self.angle_visualizer = AngleVisualizer(self, angle_buff)
        return self.angle_visualizer

    def set_stroke_width_of_rods(self, new_stroke_width: float):
        self.rod1.set_stroke_width(new_stroke_width)
        self.rod2.set_stroke_width(new_stroke_width)

        return self

    def unlink_angle_visualizer(self):
        assert self.angle_visualizer is not None, "no angle visualizer to unlink"
        self.angle_visualizer = None


class UnionDoublePendulumConstructor(DoublePendulumConstructor):
    def __init__(
            self,
            length_1: float,
            length_2: float,
            angle_pair: tuple[float, float] | np.ndarray,
            first_rod_color=FIRST_ROD_COLOR,
            second_rod_color=SECOND_ROD_COLOR,
            **kwargs
    ):
        super().__init__(
            length_1,
            length_2,
            angle_pair,
            first_rod_color,
            second_rod_color,
            **kwargs)

        self.pen1 = self.get_first_pen()
        self.pen2 = self.get_second_pen()
        for mob in [self.rod1, self.rod2, self.bob1, self.bob2]:
            mob.set_opacity(0)
        self.add(self.pen2, self.pen1)

    def get_pen(self, rod: Line, bob: Circle):
        pen = Union(bob, line_to_rectangle(rod), name="pendulum").set_color(rod.color).set_opacity(1)
        pen.set_stroke(width=0, opacity=0)

        return pen

    def get_first_pen(self):
        return self.get_pen(self.rod1, self.bob1)

    def get_second_pen(self):
        return self.get_pen(self.rod2, self.bob2)

    def update_pens(self):
        new_pen1 = self.get_first_pen()
        new_pen2 = self.get_second_pen()
        self.pen1.become(new_pen1)
        self.pen2.become(new_pen2)

    @property
    def angle_pair(self):
        return self._angle_pair

    @angle_pair.setter
    def angle_pair(self, new_angle_pair: tuple[float, float]):
        self._angle_pair = new_angle_pair
        self.update_angles()
        self.update_pens()
        if self.angle_visualizer:
            self.angle_visualizer.update_angle_visualizer()


class AngleVisualizer(VGroup, metaclass=ConvertToOpenGL):
    def __init__(
            self,
            double_pendulum: DoublePendulumConstructor,
            angle_buff: float = 0.1,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.double_pendulum = double_pendulum
        self.angle_buff = angle_buff
        self.length_1 = self.double_pendulum.length_1
        self.length_2 = self.double_pendulum.length_2
        self.theta_1 = self.double_pendulum.angle_pair[0]
        self.theta_2 = self.double_pendulum.angle_pair[1]
        self.rod1 = self.double_pendulum.rod1
        self.rod2 = self.double_pendulum.rod2
        self.rod_stroke_width = self.double_pendulum.rod_stroke_width
        self.first_rod_color = self.double_pendulum.first_rod_color
        self.second_rod_color = self.double_pendulum.second_rod_color
        self.vertical_dash_line_length_1 = self.length_1 / 1.5
        self.vertical_dash_line_length_2 = self.length_2 / 1.5
        self.angle_radius_1 = max(self.length_1 / 4.5, 0.2)
        self.angle_radius_2 = max(self.length_2 / 4.5, 0.2)
        self.dashed_line_1 = DashedLine(
            self.rod1.get_start(),
            self.rod1.get_start() + DOWN * self.vertical_dash_line_length_1,
            dash_length=self.length_1 / 25,
            dashed_ratio=0.5,
            stroke_width=self.rod_stroke_width / 7,
            stroke_opacity=1,
            name="dashed_line_1"
        ).set_color(self.first_rod_color)
        self.dashed_line_2 = DashedLine(
            self.rod1.get_end(),
            self.rod1.get_end() + DOWN * self.vertical_dash_line_length_2,
            dash_length=self.length_1 / 25,
            dashed_ratio=0.5,
            stroke_width=self.rod_stroke_width / 7,
            stroke_opacity=1,
            name="dashed_line_2"
        ).set_color(self.second_rod_color)
        self.angle_1_arc = self.get_angle_1_arc()
        self.angle_2_arc = self.get_angle_2_arc()
        self.angle_1_value = self.get_angle_1_value()
        self.angle_2_value = self.get_angle_2_value()
        self.angle_visualizer_list = [
            self.angle_1_arc,
            self.angle_2_arc,
            self.dashed_line_1,
            self.dashed_line_2,
            self.angle_1_value,
            self.angle_2_value
        ]
        self.add(*self.angle_visualizer_list)
        self.double_pendulum.angle_visualizer = self

    def get_angle_1_arc(self):
        return Arc(
            self.angle_radius_1,
            self.dashed_line_1.get_angle(),
            self.theta_1 * DEGREES,
            arc_center=self.rod1.get_start(),
            stroke_width=self.rod_stroke_width / 5,
            color=self.first_rod_color,
            stroke_opacity=1,
            name="angle_1_arc"
        )

    def get_angle_2_arc(self):
        return Arc(
            self.angle_radius_2,
            self.dashed_line_2.get_angle(),
            self.theta_2 * DEGREES,
            arc_center=self.rod2.get_start(),
            stroke_width=self.rod_stroke_width / 5,
            color=self.second_rod_color,
            stroke_opacity=1,
            name="angle_2_arc"
        )

    def get_angle_value(self, angle_arc, theta, length, rod_color, buff: float = 0) -> Tex:
        point = angle_arc.point_from_proportion(0.5)
        label_text = f"{theta:.{0}f}$^{{\\circ}}$"

        a = Tex(
            label_text,
            name="angle_value"
        ).scale(max(length / 3, 0.4))

        a.move_to(
            point + self.get_label_offset(a, theta, buff)
        ).set_color(rod_color).set_stroke(BLACK, 0.25, opacity=1)

        return a

    @staticmethod
    def get_label_offset(num: Mobject, angle: float, buff: float = 0) -> np.ndarray:
        a = num.width / 2
        b = num.height / 2
        max = np.sqrt(a**2 + b**2)
        angle = (angle / 2) * DEGREES

        def segment_length():
            if np.abs(np.tan(angle)) <= a / b:
                return np.abs(b / np.cos(angle))
            else:
                return np.abs(a / np.sin(angle))

        norm = segment_length() + buff + (max - segment_length()) / 1.5
        angle = angle + 3 * PI / 2
        return np.array([norm * np.cos(angle), norm * np.sin(angle), 0])

    def get_angle_1_value(self):
        return self.get_angle_value(
            self.angle_1_arc,
            self.theta_1,
            self.length_1,
            FIRST_ROD_COLOR,
            buff=self.angle_buff
        )

    def get_angle_2_value(self):
        return self.get_angle_value(
            self.angle_2_arc,
            self.theta_2,
            self.length_2,
            SECOND_ROD_COLOR,
            buff=self.angle_buff
        )

    @staticmethod
    def is_angle_negative(angle):
        if angle >= 0:
            return True
        else:
            return False

    def update_angle_visualizer(self):
        self.theta_1 = self.double_pendulum.angle_pair[0]
        self.theta_2 = self.double_pendulum.angle_pair[1]
        self.dashed_line_1.move_to(self.rod1.get_start() + self.dashed_line_1.get_vector() / 2)
        self.dashed_line_2.move_to(self.rod2.get_start() + self.dashed_line_2.get_vector() / 2)
        self.angle_1_arc.become(self.get_angle_1_arc())
        self.angle_2_arc.become(self.get_angle_2_arc())
        if self.angle_1_value:
            self.angle_1_value.become(self.get_angle_1_value())
        if self.angle_2_value:
            self.angle_2_value.become(self.get_angle_2_value())

    def unlink_angle_values(self):
        self.remove(self.angle_1_value, self.angle_2_value)
        self.angle_1_value = None
        self.angle_2_value = None


class DoublePendulumGhosts(VGroup, metaclass=ConvertToOpenGL):
    def __init__(
            self,
            main_dp: DoublePendulum,
            override_fps: int | None = None,
            duration: float = 1.25,
            max_opac: float = 0.075,  # R 0.075
            opacity_rate_func: Callable[[float], float] = slow_into,
            **kwargs
    ):
        super().__init__(
            **kwargs
        )
        self.main_dp = main_dp
        self.override_fps = override_fps
        self.duration = duration
        self.max_opacity = max_opac
        self.opacity_rate_func = opacity_rate_func
        self.first_dp_ghosts = VGroup()
        self.second_dp_ghosts = VGroup()
        self.create_ghosts()

    def create_ghosts(self):
        data_factory = self.main_dp.copy().get_data_factory(
            self.duration,
            override_fps=200 if self.override_fps is None else self.override_fps  # use 200 in final render
        )
        # Get angle data
        angle_1_values = data_factory.normalized_angle_1_progression_in_degrees
        angle_2_values = data_factory.normalized_angle_2_progression_in_degrees

        # Calculate the step size for indexing
        num_of_ghosts = len(angle_1_values)
        for i in range(num_of_ghosts):
            ghost = self.main_dp.double_pendulum.copy()

            ghost.angle_pair = (angle_1_values[i], angle_2_values[i])
            opacity = (1 - self.opacity_rate_func(i / num_of_ghosts)) * self.max_opacity

            pen1: VMobject = ghost.bob1.set_color(ghost.rod1.color).set_opacity(opacity).set_stroke(width=0, opacity=0)
            pen2: VMobject = ghost.bob2.set_color(ghost.rod2.color).set_opacity(opacity).set_stroke(width=0, opacity=0)

            self.first_dp_ghosts.add(pen1)
            self.second_dp_ghosts.add(pen2)

        self.add(self.first_dp_ghosts, self.second_dp_ghosts)


class TableOfDoublePendulums(VGroup, metaclass=ConvertToOpenGL):
    def __init__(
            self,
            cs: DynamicAxes,
            row_count: int = 30,
            column_count: int = 30,
            first_rod_color: ManimColor = FIRST_ROD_COLOR,
            second_rod_color: ManimColor = SECOND_ROD_COLOR,
            is_sparse: bool = False,
            **kwargs
    ):
        self._angle_pair_to_highlight: tuple[float, float] | None = None
        self.index: int | None = None
        super().__init__(**kwargs)
        assert column_count % 2 == 0, "Column count must be even"

        self.cs = cs
        self.row_count = row_count
        self.column_count = column_count

        def zigzag_combine(arr1, arr2):
            result = []
            for num1 in arr1:
                for num2 in arr2:
                    result.append((num1, num2))
            return result

        angle_1_range = cs.dyn_x_range
        angle_2_range = cs.dyn_y_range
        self.angle_1_values, self.x_retstep = center_points_linspace(
            angle_1_range[0],
            angle_1_range[1],
            row_count,
            True
        )
        self.angle_2_values, self.y_retstep = center_points_linspace(
            angle_2_range[0],
            angle_2_range[1],
            column_count,
            True
        )
        self.x_retstep *= cs.x_length / (cs.dyn_x_range[1] - cs.dyn_x_range[0])
        self.y_retstep *= cs.y_length / (cs.dyn_y_range[1] - cs.dyn_y_range[0])
        self.rod_length = min(
            self.x_retstep - 0.03,
            self.y_retstep - 0.03,
        ) / (2 if not is_sparse else 4)

        self.angle_pairs = np.array(zigzag_combine(self.angle_1_values, self.angle_2_values))
        self.dp_masters = []

        for angle_pair in tqdm(self.angle_pairs, desc="Processing angle pairs in TableOfDoublePendulums"):
            loc = cs.coords_to_point([angle_pair[0], angle_pair[1], 0])
            dp_master = DoublePendulum(angle_pair)
            dp = dp_master.create_double_pendulum(
                self.rod_length,
                self.rod_length,
                first_rod_color=first_rod_color,
                second_rod_color=second_rod_color,
                # include_bobs=False if is_sparse else True,
            ).shift(loc)
            self.dp_masters.append(dp_master)
            self.add(dp)

    @property
    def angle_pair_to_highlight(self):
        return self._angle_pair_to_highlight

    @angle_pair_to_highlight.setter
    def angle_pair_to_highlight(self, angle_pair: tuple[float, float]):
        def get_highlighted_version(double_pendulum: UnionDoublePendulumConstructor) -> VGroup:
            double_pendulum = double_pendulum.copy()
            double_pendulum.scale(
                2,
                about_point=double_pendulum.rod2.get_start()
            ).set_opacity(1)
            double_pendulum.rod1.stroke_width = double_pendulum.rod1.stroke_width * 1.5
            double_pendulum.rod2.stroke_width = double_pendulum.rod2.stroke_width * 1.5
            double_pendulum.bob1.radius = double_pendulum.bob1.radius * 0.85
            double_pendulum.bob2.radius = double_pendulum.bob2.radius * 0.85
            double_pendulum.update_pens()

            return double_pendulum

        new_index = self.get_index(angle_pair)

        # Check if this is the first time highlighting or if the highlighted pendulum is different
        if self.index is None or self.index != new_index:
            # Restore the previous pendulum if it exists
            if self.index is not None:
                self.submobjects[self.index].restore()

            # Update the index and highlight the new pendulum
            self.index = new_index
            dp = self.submobjects[self.index]
            dp.set_opacity(1)
            dp.save_state()
            dp.become(get_highlighted_version(dp))

            # Set the opacity of all other pendulums to 0.5
            for i, dp in enumerate(self.submobjects):
                if i != self.index:
                    dp.set_opacity(0.1)

    def add_reflect_nums(self, indices: set[int]) -> None:
        num_of_dps = len(self.submobjects)
        indices_copy = indices.copy()
        for index in indices_copy:
            indices.add(num_of_dps - index - 1)


    def get_stable_dps(self) -> VGroup:
        stable_points = [(60, 70), (-60, -70)]
        unstable_override_points = [(-140, 142), (140, -142)]
        distance_threshold_squared = 35 ** 2
        override_distance_threshold_squared = 142 ** 2
        stable_indices = {750, 751, 752, 753, 754, 791, 792, 793, 794, 795, 796, 992, 1031, 1032, 1111, 1142,
                          1143, 1183, 1184, 1228, 1229, 1141, 1267, 1268, 1269, 1308, 1309, 1349}
        unstable_indices = {332, 501, 542, 583, 624, 665, 707, 748, 749, 747, 755, 838, 837, 1099, 1226}
        self.add_reflect_nums(stable_indices)
        self.add_reflect_nums(unstable_indices)
        print(f"Stable indices: {stable_indices}")
        print(f"Unstable indices: {unstable_indices}")

        stable_dps_list = []

        for i, dp in enumerate(self.submobjects):
            angle_1, angle_2 = dp._angle_pair

            if i in unstable_indices:
                continue
            if i in stable_indices:
                stable_dps_list.append(dp)
                continue

            # Calculate squared distances to stable and override points
            distance_1_squared = (angle_1 - stable_points[0][0]) ** 2 + (angle_2 - stable_points[0][1]) ** 2
            distance_2_squared = (angle_1 - stable_points[1][0]) ** 2 + (angle_2 - stable_points[1][1]) ** 2
            override_distance_1_squared = (angle_1 - unstable_override_points[0][0]) ** 2 + (angle_2 - unstable_override_points[0][1]) ** 2
            override_distance_2_squared = (angle_1 - unstable_override_points[1][0]) ** 2 + (angle_2 - unstable_override_points[1][1]) ** 2

            # Skip adding to stable list if it meets the override unstable condition
            if (override_distance_1_squared <= override_distance_threshold_squared or
                override_distance_2_squared <= override_distance_threshold_squared):
                continue

            # Add to stable list if it meets the stable conditions
            if (3 * math.cos(math.radians(angle_1)) + math.cos(math.radians(angle_2)) > 2 or
                distance_1_squared <= distance_threshold_squared or
                distance_2_squared <= distance_threshold_squared):
                stable_dps_list.append(dp)

        return VGroup(*stable_dps_list)

    def get_unstable_dps(self) -> VGroup:
        stable_dps_set = set(self.get_stable_dps().submobjects)
        unstable_dps_list = [dp for dp in self.submobjects if dp not in stable_dps_set]

        return VGroup(*unstable_dps_list)

    def get_index(self, angle_pair: tuple[float, float]):
        def find_closest(sorted_array, target: float):
            sorted_array = np.array(sorted_array)
            diff_array = np.abs(sorted_array - target)
            closest_index = np.argmin(diff_array)

            return closest_index

        angle_1_index = find_closest(self.angle_1_values, angle_pair[0])
        angle_2_index = find_closest(self.angle_2_values, angle_pair[1])

        return self.row_count * (angle_1_index + 1) + angle_2_index - self.column_count

    def get_list_of_columns(self):
        columns = [self.submobjects[i:i + self.row_count] for
                   i in range(0,  len(self.submobjects), self.row_count)]
        return columns


class BlockyPixels(TableOfDoublePendulums):
    def __init__(
            self,
            cs: DynamicAxes,
            row_count: int,
            column_count: int,
            color_func_index: int = 0,
            overlap: float = 0.01,
            stroke_width: float = 1,
            **kwargs
    ):
        super().__init__(
            cs,
            row_count,
            column_count,
            **kwargs
        )
        self.remove(*self.submobjects)
        self.create_rectangles(color_func_index, stroke_width, overlap)

    def create_rectangles(self, color_func_index=0, stroke_width=0.0, overlap=0.0):
        """Non-parallel version that's faster for this specific use case"""
        for angle_pair in tqdm(
                self.angle_pairs,
                desc="Processing angle pairs in BlockyPixels class"
        ):
            loc = self.cs.coords_to_point([angle_pair[0], angle_pair[1], 0])
            rect = Rectangle(
                turn_angles_to_color(*angle_pair, color_func_index=color_func_index),
                width=self.x_retstep + overlap,
                height=self.y_retstep + overlap,
                fill_opacity=1,
                stroke_opacity=1,
                stroke_width=stroke_width,
                stroke_color=BLACK,
                name=f"blocky_pixel at {angle_pair}"
            ).shift(loc)
            self.add(rect)


class PixelGridAnglesComputation:
    def __init__(
            self,
            angle_1_domain: tuple[float, float],
            angle_2_domain: tuple[float, float],
            width_pixel_num: int,
            height_pixel_num: int,
    ):
        self.angle_1_values = self.get_angle_values(angle_1_domain, width_pixel_num)
        self.angle_2_values = self.get_angle_values(angle_2_domain, height_pixel_num)
        self.width_pixel_num = width_pixel_num
        self.height_pixel_num = height_pixel_num
        self.num_pixels_per_frame = width_pixel_num * height_pixel_num
        self.color_func = None
        self.angle_pairs = self.get_angle_pairs_from_two_lists(self.angle_1_values, self.angle_2_values)

    @staticmethod
    def get_angle_values(angle_domain: tuple[float, float], num: int) -> np.ndarray:
        return center_points_linspace(angle_domain[0], angle_domain[1], num=num)

    @staticmethod
    def get_angle_pairs_from_two_lists(angle_1_values: np.ndarray, angle_2_values: np.ndarray) -> np.ndarray:
        return np.dstack(
            np.meshgrid(angle_1_values, angle_2_values[::-1])
        ).reshape(-1, 2)

    def get_static_initial_visuals_int_data(self, color_func: Callable) -> np.ndarray:
        """
        :return: numpy array with shape (height, width, 4)
        """
        self.color_func = color_func
        pixel_data_for_static_frame = self.color_func(torch.tensor(self.angle_pairs, device=device)).reshape(
            (self.height_pixel_num, self.width_pixel_num, 4)
        )
        pixel_int_data_for_static_frame = (pixel_data_for_static_frame * 255).cpu().numpy().astype(np.uint8)

        del pixel_data_for_static_frame
        torch.cuda.empty_cache()

        return pixel_int_data_for_static_frame

    @timer
    def get_batched_angle_pairs(
            self,
            old_x_range: tuple[float, float],
            new_x_range: tuple[float, float],
            old_y_range: tuple[float, float],
            new_y_range: tuple[float, float],
            num_of_frames: int,
            zoom_rate_func: Callable,
            zoom_flip: bool = False
    ) -> np.ndarray:
        zoom_x_values = get_zoom_range_values(
            old_x_range,
            new_x_range,
            num_of_frames,
            zoom_rate_func
        )
        zoom_y_values = get_zoom_range_values(
            old_y_range,
            new_y_range,
            num_of_frames,
            zoom_rate_func
        )
        if not zoom_flip:
            file_name = get_unique_filename("extra")
        else:
            file_name = dp_data_file_dir + r"\scene7_8_batched.dat"
        file_exists = os.path.exists(file_name)
        print(f"Outputting to temporary/unique file: {file_name}")
        batch_angles_pairs_mmap = np.memmap(
            file_name,
            dtype=np.float64,
            mode='w+' if not file_exists else 'r+',
            shape=(num_of_frames * self.width_pixel_num * self.height_pixel_num,
                   2)
        )
        if file_exists:
            return batch_angles_pairs_mmap

        for i, (x_range, y_range) in tqdm(
                enumerate(zip(zoom_x_values, zoom_y_values)),
                total=len(zoom_x_values),
                desc="Processing Batched Angle Pairs"
        ):
            angle_1_values = self.get_angle_values(x_range, self.width_pixel_num)
            angle_2_values = self.get_angle_values(y_range, self.height_pixel_num)
            grid_x, grid_y = np.meshgrid(angle_1_values, angle_2_values[::-1])
            new_angle_pairs = np.dstack([grid_x, grid_y]).reshape(-1, 2)

            start_idx = i * new_angle_pairs.shape[0]
            batch_angles_pairs_mmap[start_idx:start_idx + new_angle_pairs.shape[0]] = new_angle_pairs
            if i % (num_of_frames // 80) == 0:
                batch_angles_pairs_mmap.flush()
        batch_angles_pairs_mmap.flush()

        return batch_angles_pairs_mmap

    def get_zoom_anim_data_pixel_visuals(
            self,
            old_x_range: tuple[float, float],
            new_x_range: tuple[float, float],
            old_y_range: tuple[float, float],
            new_y_range: tuple[float, float],
            duration: int,
            zoom_rate_func: Callable,
            color_func: Callable,
    ) -> np.ndarray:
        """
        :return: numpy array with shape (fps * duration, height, width, 4)
        """
        self.color_func = color_func
        num_of_frames = int(duration * fps)

        angle_pairs = self.get_batched_angle_pairs(
            old_x_range, new_x_range, old_y_range, new_y_range, num_of_frames, zoom_rate_func
        )

        pixel_int_data_mmap = np.memmap(
            get_unique_filename("extra"),
            dtype=np.uint8,
            mode='w+',
            shape=(num_of_frames, self.height_pixel_num, self.width_pixel_num, 4)
        )
        for i in tqdm(
            range(num_of_frames),
            total=num_of_frames,
            desc="Processing Zoom Anim Data Pixel Visuals"
        ):
            angle_pairs_chunk = angle_pairs[i * self.num_pixels_per_frame: (i + 1) * self.num_pixels_per_frame]
            pixel_data = self.color_func(torch.tensor(angle_pairs_chunk, device=device)).cpu().numpy()
            pixel_int_data_chunk = (pixel_data * 255).astype(np.uint8).reshape(
                (self.height_pixel_num, self.width_pixel_num, 4)
            )
            pixel_int_data_mmap[i] = pixel_int_data_chunk
        pixel_int_data_mmap.flush()

        return pixel_int_data_mmap

    def get_direct_int_rgba_data(self, color_func: Callable, elapsed_time: float, use_existing_dat: str | None =
    None, skip_processing: bool = False) \
            -> (
            np.ndarray):
        return OptimizedForPixelGridComputation(
            self.angle_pairs,
            elapsed_time,
            linear
        ).pixel_visuals_data(color_func, self.width_pixel_num, self.height_pixel_num,
                             use_existing_dat=use_existing_dat, skip_processing=skip_processing)

    @staticmethod
    def get_flip_indices(angle_progression: np.ndarray | None = None, jump_threshold: float = 180) -> np.ndarray:
        diff = np.abs(np.diff(angle_progression))
        jump = np.where(diff > jump_threshold)

        return np.asarray(jump)

    def get_flips_index_data(self, duration: float, rate_func: Callable = linear, **flip_visuals_kwargs) -> np.ndarray:
        return OptimizedForPixelGridComputation(
            self.angle_pairs,
            duration,
            rate_func
        ).flip_visuals_index_data(**flip_visuals_kwargs)

    def get_default_white_data(self, duration: float) -> np.memmap:
        white_data_mmap = np.memmap(
            get_unique_filename('white_default'),
            dtype=np.uint8,
            mode='w+',
            shape=(self.num_pixels_per_frame, int(duration * fps), 4)
        )
        chunk_size = self.num_pixels_per_frame // 40
        for i in range(0, self.num_pixels_per_frame, chunk_size):
            end_index = min(i + chunk_size, self.num_pixels_per_frame)
            white_data_mmap[i:end_index, :, :] = 255
            if i % 4 == 0:
                white_data_mmap.flush()

        white_data_mmap.flush()

        return white_data_mmap

    def get_flips_animation_data(
            self,
            color_tracker: ColorTracker,
            duration: float,
            skip_processing: bool = False,
            use_existing_dat: str | None = None
    ) -> np.ndarray:
        index_data = self.get_flips_index_data(
            duration,
            linear,
            skip_processing=skip_processing,
            use_existing_dat=use_existing_dat
        )
        if use_existing_dat is not None:
            file_name = os.path.join(
                dp_data_file_dir, f"{use_existing_dat + '_reshape_flips_' + str(fps) + quality}.dat"
            )
        else:
            file_name = get_unique_filename(base_name='flip_index_data', directory=dp_data_file_dir)

        file_exists = os.path.exists(file_name)
        print(f"file_name for get_flips_animation_data: {file_name}")
        flips_mmap = np.memmap(
            file_name,
            dtype=np.uint8,
            mode='w+' if not file_exists else 'r',
            shape=(int(duration * fps), self.height_pixel_num, self.width_pixel_num, 4)
        )
        if file_exists:
            return flips_mmap

        color_data_for_every_frame = self.get_default_white_data(duration)
        for dp_index, color_index in tqdm(
                enumerate(index_data),
                total=len(index_data),
                desc="Processing get_flips_animation_data"
        ):
            if color_index == -1:
                continue
            color_data_for_every_frame[dp_index][color_index:] = color_to_int_rgba(
                color_tracker.alpha_to_color(color_index / (int(duration * fps) - 1))
            )
            if dp_index % 50000 == 0:
                color_data_for_every_frame.flush()
                gc.collect()

        color_data_for_every_frame.flush()
        del index_data

        X, Y = self.height_pixel_num, self.width_pixel_num
        for x in tqdm(range(X), desc="Processing flips"):
            chunk = color_data_for_every_frame[x * Y: (x + 1) * Y, :, :]
            chunk = chunk.transpose(1, 0, 2)
            chunk = np.ascontiguousarray(chunk)
            flips_mmap[:, x, :, :] = chunk
            if x % 50 == 0:
                flips_mmap.flush()
                gc.collect()

        flips_mmap.flush()

        return flips_mmap

    def get_zooming_flips_animation_data(
            self,
            flip_visuals: FlipStaticVisuals,
            color_tracker: ColorTracker,
            new_x_range: tuple[float, float],
            new_y_range: tuple[float, float],
            anim_duration: float,
            zoom_rate_func: Callable = smoothstep,
            use_existing_dat: str | None = None,
            skip_processing: bool = False
    ) -> np.ndarray:
        print(f"\n----------> running get_zooming_flip_animation_data()")

        elapsed_time = color_tracker.elapsed_time
        num_of_frames = int(anim_duration * fps)
        batched_angle_pairs = self.get_batched_angle_pairs(
            (flip_visuals.cs.dyn_x_range[0], flip_visuals.cs.dyn_x_range[1]),
            new_x_range,
            (flip_visuals.cs.dyn_y_range[0], flip_visuals.cs.dyn_y_range[1]),
            new_y_range,
            num_of_frames,
            zoom_rate_func,
            zoom_flip=True,
        )
        print(f"shape of batched_angle_pairs: {batched_angle_pairs.shape}")
        print(f"size of batched_angle_pairs in GB: {batched_angle_pairs.nbytes / (1024 ** 3)}")

        indices_data = OptimizedForPixelGridComputation(
            batched_angle_pairs,
            elapsed_time
        ).flip_visuals_index_data(use_existing_dat=use_existing_dat, skip_processing=skip_processing)
        delete_memmap_files("extra")

        print(f"num of frames: {num_of_frames}, num of pixels per frame: {self.num_pixels_per_frame}")
        indices_data = indices_data.reshape(num_of_frames, self.num_pixels_per_frame)
        print(f"successful here")

        file_name = os.path.join(dp_data_file_dir, f"{use_existing_dat + '_color_' + str(fps) + quality}.dat")
        file_exists = os.path.exists(file_name)
        all_color_data_mmap = np.memmap(
            file_name,
            dtype=np.uint8,
            mode='w+' if not file_exists else 'r+',
            shape=(int(anim_duration * fps),
                   self.width_pixel_num * self.height_pixel_num,
                   4)
        )
        # -- REGION: ALTER AFTER PROCESSING IS FINISHED -- #
        if skip_processing: # use if not skip_processing or just delete it entirely
            pass
        else:
            white_frame_template = np.full((self.num_pixels_per_frame, 4), 255, dtype=np.uint8)
            for i, index_data in tqdm(
                enumerate(indices_data),
                desc="Processing Color Data",
                total=len(indices_data)
            ):
                white_frame = white_frame_template.copy()
                flipped_pixel_indices = np.where(index_data != -1)[0]

                color_indices_flipped = index_data[flipped_pixel_indices]
                alpha_norm_flipped = color_indices_flipped.astype(np.float32) / (int(elapsed_time * fps) - 1)
                flip_colors_rgba = np.array([
                    color_to_int_rgba(color_tracker.alpha_to_color(alpha))
                    for alpha in alpha_norm_flipped
                ], dtype=np.uint8)
                white_frame[flipped_pixel_indices] = flip_colors_rgba

                all_color_data_mmap[i] = white_frame

                if i % (len(indices_data) // 10) == 0:
                    all_color_data_mmap.flush()
            all_color_data_mmap.flush()
        # -- END REGION -- #

        del all_color_data_mmap
        gc.collect()
        zoom_animation_memmap = np.memmap(
            file_name,
            dtype=np.uint8,
            mode='r',
            shape=(num_of_frames, self.height_pixel_num, self.width_pixel_num, 4)
        )
        # if skip_processing:
        #     pass
        # else:
        #     for row in tqdm(range(self.height_pixel_num), desc="Reshaping zoom animation data"):
        #         start_index = row * self.width_pixel_num
        #         end_index = (row + 1) * self.width_pixel_num
        #
        #         zoom_animation_memmap[:, row, :, :] = all_color_data_mmap[:, start_index:end_index, :]
        #
        #         if row % 50 == 0:
        #             zoom_animation_memmap.flush()
        #             gc.collect()
        #
        #     zoom_animation_memmap.flush()
        # del all_color_data_mmap
        # delete_memmap_files("unreshape_zoom")

        return zoom_animation_memmap


class PixelStaticVisuals(ImageMobject):
    def __init__(
            self,
            cs: DynamicAxes,
            width_pixel_num: int,
            height_pixel_num: int,
            color_func: Callable,  # must be TCF
            **kwargs
    ):
        self.cs = cs
        self.width_pixel_num = width_pixel_num
        self.height_pixel_num = height_pixel_num
        self.angle_1_domain = (cs.dyn_x_range[0], cs.dyn_x_range[1])
        self.angle_2_domain = (cs.dyn_y_range[0], cs.dyn_y_range[1])
        self.width_pixel_num = width_pixel_num
        self.height_pixel_num = height_pixel_num
        self.color_func = color_func
        scale_to_resolution = height_pixel_num * config.frame_height / cs.y_length

        self.data_computation = PixelGridAnglesComputation(
            self.angle_1_domain,
            self.angle_2_domain,
            width_pixel_num,
            height_pixel_num,
        )

        int_rgba = self.data_computation.get_static_initial_visuals_int_data(self.color_func)

        super().__init__(
            int_rgba,
            scale_to_resolution=scale_to_resolution,
            **kwargs
        )

        del int_rgba
        torch.cuda.empty_cache()

        self.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        self.shift(self.cs.location)


class CrispPixelStaticVisuals(PixelStaticVisuals):
    def __init__(
            self,
            cs: DynamicAxes,
            color_func: Callable,
            **kwargs
    ):
        width_pixel_num = int(cs.x_length * SCENE_PIXELS)
        height_pixel_num = int(cs.y_length * SCENE_PIXELS)
        print(f"----------> pixel dimension of CrispPixelStaticVisuals is {width_pixel_num} x {height_pixel_num} \n"
              f"            with a total of {width_pixel_num * height_pixel_num:,} pixels")
        super().__init__(
            cs,
            width_pixel_num,
            height_pixel_num,
            color_func,
            **kwargs
        )


class FlipStaticVisuals(ImageMobject):
    def __init__(
            self,
            cs: DynamicAxes,
            width_pixel_num: int,
            height_pixel_num: int,
            **kwargs
    ):
        self.cs = cs
        self.angle_1_domain = (cs.dyn_x_range[0], cs.dyn_x_range[1])
        self.angle_2_domain = (cs.dyn_y_range[0], cs.dyn_y_range[1])
        self.width_pixel_num = width_pixel_num
        self.height_pixel_num = height_pixel_num
        self.color_tracker = None
        self.pixel_computation = PixelGridAnglesComputation(
            self.angle_1_domain,
            self.angle_2_domain,
            width_pixel_num,
            height_pixel_num,
        )

        int_rgba = self.pixel_computation.get_static_initial_visuals_int_data(TorchColorFuncs.white)
        scale_to_resolution = height_pixel_num * config.frame_height / cs.y_length
        super().__init__(
            int_rgba,
            scale_to_resolution=scale_to_resolution,
            **kwargs
        )
        self.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])

    def turn_to_last_image(self, color_tracker: ColorTracker, **flip_visuals_kwargs):
        self.color_tracker = color_tracker
        flips_animation_data = self.pixel_computation.get_flips_animation_data(
            color_tracker, color_tracker.elapsed_time, **flip_visuals_kwargs
        )
        self.pixel_array = flips_animation_data[-1].copy()

        del flips_animation_data
        delete_memmap_files("white_default")

        return self


class CrispFlipStaticVisuals(FlipStaticVisuals):
    def __init__(
            self,
            cs: DynamicAxes,
            **kwargs,
    ):
        width_pixel_num = int(cs.x_length * SCENE_PIXELS)
        height_pixel_num = int(cs.y_length * SCENE_PIXELS)
        print(f"----------> pixel dimension of CrispFlipStaticVisuals is {width_pixel_num} x {height_pixel_num} \n"
              f"            with a total of {width_pixel_num * height_pixel_num:,} pixels")
        super().__init__(
            cs,
            width_pixel_num,
            height_pixel_num,
            **kwargs
        )


class ColorTracker(Group):
    def __init__(
            self,
            colors: Sequence[ParsableManimColor],
            elapsed_time: float = 10,
            time_step: int = 1,
            is_horizontal: bool = False,
            width_units: float = 1,
            length_units: float = 7,
            color_rate_func: Callable[[float], float] = linear,
            radius: float = 0.15,
            **kwargs
    ):
        self.colors = colors
        self.elapsed_time = elapsed_time
        self.time_step = time_step
        self.is_horizontal = is_horizontal
        self.width_units = width_units
        self.length_units = length_units
        self.radius = radius
        self._color_rate_func = color_rate_func

        self.length_in_pixels = int(length_units * SCENE_PIXELS)
        self.width_in_pixels = int(width_units * SCENE_PIXELS)

        self.color_gradient = None

        self.color_basis = self.create_color_basis()
        self.arrow = self.create_arrow()
        self.time_labels_and_ticks = self.create_time_labels_and_ticks()

        super().__init__(
            self.color_basis,
            self.arrow,
            self.time_labels_and_ticks,
            **kwargs
        )
        self._alpha_progression = 0.0
        self.update_arrow()

    def create_color_basis(self) -> ImageMobject:
        self.color_gradient = color_gradient_with_rate_func(
            self.colors,
            self.length_in_pixels,
            self._color_rate_func
        )
        if self.is_horizontal:
            pixel_row = np.asarray([color_to_int_rgba(col) for col in self.color_gradient], dtype=np.uint8)
            pixel_array = np.tile(pixel_row[np.newaxis, :, :], (self.width_in_pixels, 1, 1))
        else:
            pixel_column = np.asarray([color_to_int_rgba(col) for col in self.color_gradient[::-1]], dtype=np.uint8)
            pixel_array = np.tile(pixel_column[:, np.newaxis, :], (1, self.width_in_pixels, 1))
        color_basis = ImageMobject(pixel_array, config.pixel_height, name="color_basis")
        color_basis.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])

        return color_basis

    def create_time_labels_and_ticks(self) -> VGroup:
        time_labels_and_ticks = VGroup()

        for t in np.arange(0, self.elapsed_time + 1e-5, self.time_step):
            col = self.alpha_to_color(t / self.elapsed_time)
            pos = self.alpha_to_pos(t / self.elapsed_time)
            label = Tex(
                round(t),
                font_size=20,
                color=col,
            )
            tick = Line(UP * 0.05, DOWN * 0.05, stroke_width=2, color=col) \
                if self.is_horizontal else Line(LEFT * 0.05, RIGHT * 0.05, stroke_width=2, color=col)

            if self.is_horizontal:
                label.next_to(self.color_basis, DOWN, buff=0.14)
                move_relative_to(tick, tick.get_start(), self.color_basis.get_bottom() + UP * 0.01)
                label.set_x(pos)
                tick.set_x(pos)
            else:
                label.next_to(self.color_basis, RIGHT, buff=0.14)
                move_relative_to(tick, tick.get_start(), self.color_basis.get_right() + LEFT * 0.01)
                label.set_y(pos)
                tick.set_y(pos)

            time_labels_and_ticks.add(label, tick)

        return time_labels_and_ticks

    def create_arrow(self) -> VGroup:
        arrow_specs = {
            "fill_opacity": 1,
            "stroke_width": 0,
        }
        circle = Circle(self.radius, **arrow_specs)
        shaft = Rectangle(height=self.radius / 1.5, width=self.radius * 5, **arrow_specs)
        tip = Triangle(radius=(self.radius / 1.5) / (math.sqrt(3)), **arrow_specs)

        if self.is_horizontal:
            tip.rotate(-np.pi)
            shaft.rotate(np.pi / 2)
            arrow = VGroup(circle, shaft, tip).arrange(DOWN, -0.01)
            return arrow.next_to(self.color_basis, UP, buff=0.02)
        else:
            tip.rotate(-np.pi / 2)
            arrow = VGroup(circle, shaft, tip, name="arrow of color tracker").arrange(RIGHT, -0.01)
            return arrow.next_to(self.color_basis, LEFT, buff=0.02)

    @property
    def alpha_progression(self):
        return self._alpha_progression

    @alpha_progression.setter
    def alpha_progression(self, new_alpha: float):
        self._alpha_progression = new_alpha
        self.update_arrow()

    def update_arrow(self):
        if self.is_horizontal:
            self.arrow.set_x(self.alpha_to_pos(self._alpha_progression))
        else:
            self.arrow.set_y(self.alpha_to_pos(self._alpha_progression))
        self.arrow.set_color(self.alpha_to_color(self._alpha_progression))

    @property
    def color_rate_func(self):
        return self._color_rate_func

    @color_rate_func.setter
    def color_rate_func(self, new_func):
        self._color_rate_func = new_func
        self.color_basis.become(self.create_color_basis().move_to(self.color_basis.get_center()))
        self.time_labels_and_ticks.become(self.create_time_labels_and_ticks())

    def alpha_to_pos(self, alpha: float) -> float:
        displacement = alpha * self.length_units

        if self.is_horizontal:
            return self.color_basis.get_left()[0] + displacement
        else:
            return self.color_basis.get_bottom()[1] + displacement

    def alpha_to_color(self, alpha: float) -> ManimColor:
        assert alpha <= 1, "alpha must be <= 1"

        return self.color_gradient[int(alpha * (len(self.color_gradient) - 1))]


class ComputeFlipsForSingleDP(PixelGridAnglesComputation):
    def __init__(
            self,
            main_dp: DoublePendulum,
            elapsed_time: float,
    ):
        self.main_dp = main_dp
        super().__init__(
            (self.main_dp.init_angle_1, self.main_dp.init_angle_1),
            (self.main_dp.init_angle_2, self.main_dp.init_angle_2),
            1,
            1,
        )
        self._elapsed_time = elapsed_time

    def get_flip_indices_for_single_dp(self) -> tuple[np.ndarray, np.ndarray]:
        compute = ComputeDoublePendulumSimulation(
            (self.main_dp.init_angle_1, self.main_dp.init_angle_2),
            self._elapsed_time
        )
        angle_1_progression = compute.normalized_angle_1_progression_in_degrees
        angle_2_progression = compute.normalized_angle_2_progression_in_degrees
        angle_1_flips_indices = self.get_flip_indices(angle_1_progression)
        angle_2_flips_indices = self.get_flip_indices(angle_2_progression)

        return angle_1_flips_indices, angle_2_flips_indices


class PixelInset(Group):
    def __init__(
            self,
            pixel_visual: PixelStaticVisuals | FlipStaticVisuals,
            x_range: tuple[float, float],
            y_range: tuple[float, float],
            x_length: float,
            y_length: float,
            location: np.ndarray,
            inset_line_dirs: Sequence,
            font_size_x: float = 16,
            font_size_y: float = 16,
            inset_color: ManimColor = BLACK,
            inset_stroke_width: float = 1,
            include_rect_inset_pixel_visual: bool = True,
            include_hexagons: bool = False,
            include_return_cs: bool = False,
            line_to_number_buff: float = 0.125,
            **kwargs
    ):
        """**kwargs to be passed to DynamicAxes"""
        self.pixel_visual = pixel_visual
        self.cs = pixel_visual.cs
        assert self.cs.dyn_x_range[0] <= x_range[0] < x_range[1] <= self.cs.dyn_x_range[1], (
            "x_range must be within the dynamic x_range")
        assert self.cs.dyn_y_range[0] <= y_range[0] < y_range[1] <= self.cs.dyn_y_range[1], (
            "y_range must be within the dynamic y_range")
        self.inset_line_dirs = inset_line_dirs
        self.inset_color = inset_color
        self.include_return_cs = include_return_cs

        self.inset_cs = DynamicAxes(
            x_range=x_range,
            y_range=y_range,
            x_length=x_length,
            y_length=y_length,
            x_is_in_degrees=self.cs.x_is_in_degrees,
            y_is_in_degrees=self.cs.y_is_in_degrees,
            font_size_x=font_size_x,
            font_size_y=font_size_y,
            use_constant_tick_length=True,
            x_line_to_number_buff=line_to_number_buff,
            y_line_to_number_buff=line_to_number_buff,
            tick_length=0.01,
            add_background_rectangle=False,
            **kwargs
        ).shift(location)

        if isinstance(self.pixel_visual, PixelStaticVisuals):
            self.inset_pixel_visual = CrispPixelStaticVisuals(self.inset_cs,
                                                              self.pixel_visual.color_func)
        else:
            self.inset_pixel_visual = CrispFlipStaticVisuals(self.inset_cs).shift(location)

        self.rect_pixel_visual = self.create_inset_rectangle().set_stroke(inset_color, inset_stroke_width, 1).set_fill(
            inset_color, 0.15)
        self.inset_lines = VGroup(name="inset_lines")
        for direction in self.inset_line_dirs:
            self.inset_lines.add(self.create_inset_line(direction))
        self.inset_lines.set_stroke(inset_color, inset_stroke_width, 1)

        self.rect_inset_pixel_visual = SurroundingRectangle(
            self.inset_pixel_visual,
            buff=0,
            fill_opacity=0,
            stroke_width=inset_stroke_width,
            stroke_color=inset_color,
            name="rect_inset_pixel_visual"
        )

        objs = [self.rect_pixel_visual, self.inset_lines]
        if include_hexagons:
            hexagon = self.get_hexagon()
            objs.append(hexagon)
        if include_rect_inset_pixel_visual:
            objs.append(self.rect_inset_pixel_visual)
        objs.append(self.inset_pixel_visual)
        if self.include_return_cs:
            objs.insert(0, self.inset_cs)

        super().__init__(*objs)

    def create_inset_rectangle(self) -> Polygon:
        return Polygon(
            self.cs.coords_to_point((self.inset_cs.x_range[0], self.inset_cs.y_range[1])),
            self.cs.coords_to_point((self.inset_cs.x_range[1], self.inset_cs.y_range[1])),
            self.cs.coords_to_point((self.inset_cs.x_range[1], self.inset_cs.y_range[0])),
            self.cs.coords_to_point((self.inset_cs.x_range[0], self.inset_cs.y_range[0])),
            name="rect_on_axes"
        )

    def create_inset_line(self, corner: np.ndarray) -> Line:
        return Line(self.rect_pixel_visual.get_corner(corner), self.inset_pixel_visual.get_corner(corner), name="inset_line")

    def get_hexagon(self) -> Difference:
        hexagon = Polygon(
            self.rect_inset_pixel_visual.get_corner(self.inset_line_dirs[0]),
            self.rect_inset_pixel_visual.get_corner(self.inset_line_dirs[1]),
            self.inset_pixel_visual.get_corner(self.inset_line_dirs[1]),
            self.inset_pixel_visual.get_corner(self.inset_line_dirs[0]),
        )
        hexagon = Difference(hexagon, self.rect_inset_pixel_visual)
        hexagon = Difference(hexagon, SurroundingRectangle(self.pixel_visual, buff=0), name="hexagon").set_stroke(WHITE, 0, 0)
        hexagon.set_fill(self.inset_color, 0.2)

        return hexagon

    @override_animation(FadeIn)
    def create_then_fadein(self, scene: Scene, shift: np.ndarray):
        for obj in self.submobjects:
            if isinstance(obj, Difference) or isinstance(obj, DynamicAxes):
                scene.play(FadeIn(obj, shift=shift, scale=0.5))
            elif isinstance(obj, ImageMobject):
                rect_copy = self.rect_pixel_visual.copy()
                scene.add(rect_copy)
                scene.play(FadeReplacementTransform(rect_copy, obj, run_time=3, rate_func=rush_from))
            else:
                scene.play(Create(obj))


class InsetScaffold(Group):
    def __init__(
            self,
            visuals: PixelStaticVisuals | FlipStaticVisuals,
            x_range: Tuple[float, float],
            y_range: Tuple[float, float],
            x_length: float,
            y_length: float,
            location: np.ndarray = ORIGIN,
            inset_line_dirs: Sequence = (UL, DL),
            font_size_x: float = 12,
            font_size_y: float = 12,
            inset_color: ManimColor = DARK_BROWN,
            inset_stroke_width: float = 2,
            include_image: bool = True,
            include_hexagons: bool = True,
            include_return_cs: bool = False,
            line_to_number_buff: float = 0.125,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.visuals = visuals
        self.main_axes = visuals.cs
        assert self.main_axes.dyn_x_range[0] <= x_range[0] < x_range[1] <= self.main_axes.dyn_x_range[1], (
            "x_range must be within the dynamic x_range")
        assert self.main_axes.dyn_y_range[0] <= y_range[0] < y_range[1] <= self.main_axes.dyn_y_range[1], (
            "y_range must be within the dynamic y_range")

        self.cs_island = DynamicAxes(
            x_range=x_range,
            y_range=y_range,
            x_length=x_length,
            y_length=y_length,
            x_is_in_degrees=self.main_axes.x_is_in_degrees,
            y_is_in_degrees=self.main_axes.y_is_in_degrees,
            font_size_x=font_size_x,
            font_size_y=font_size_y,
            use_constant_tick_length=True,
            x_line_to_number_buff=line_to_number_buff,
            y_line_to_number_buff=line_to_number_buff,
            tick_length=0.01,
        ).shift(location)

        if isinstance(self.visuals, PixelStaticVisuals):
            self.inset_image = CrispPixelStaticVisuals(
                self.cs_island,
                self.visuals.color_func
            )
        else:
            self.inset_image = CrispFlipStaticVisuals(
                self.cs_island
            ).shift(location)

        self.rect_on_main_axes = self.main_axes.create_rect_on_axes(x_range, y_range).set_stroke(
            inset_color, inset_stroke_width, 1)

        self.inset_lines = self._create_inset_lines(inset_line_dirs, inset_color, inset_stroke_width)

        self.pixel_visuals_rect = SurroundingRectangle(
            self.inset_image,
            buff=0,
            fill_opacity=0,
            stroke_width=inset_stroke_width,
            stroke_color=inset_color
        )
        if include_image:
            self.add(self.inset_image)
        self.add(self.rect_on_main_axes, self.pixel_visuals_rect, self.inset_lines)
        if include_hexagons:
            self.low_alpha_hexagon = self._get_low_alpha_hexagon(inset_line_dirs)
            self.add(self.low_alpha_hexagon)
        if include_return_cs:
            self.add(self.cs_island)

    def _create_inset_line(self, corner):
        """Create a line connecting the inset rectangle to the inset visualization."""
        return Line(self.rect_on_main_axes.get_corner(corner), self.inset_image.get_corner(corner))

    def _create_inset_lines(self, inset_line_dirs, inset_color, inset_stroke_width):
        """Create all inset connecting lines."""
        inset_lines = VGroup()
        for direction in inset_line_dirs:
            inset_lines.add(self._create_inset_line(direction))
        return inset_lines.set_stroke(inset_color, inset_stroke_width, 1)

    def _get_low_alpha_hexagon(self, inset_line_dirs):
        """Create a low-alpha hexagon connection between the main and inset visualizations."""
        hexagon = Polygon(
            self.rect_on_main_axes.get_corner(inset_line_dirs[0]),
            self.rect_on_main_axes.get_corner(inset_line_dirs[1]),
            self.inset_image.get_corner(inset_line_dirs[1]),
            self.inset_image.get_corner(inset_line_dirs[0]),
        )
        hexagon = Difference(hexagon, self.rect_on_main_axes)
        hexagon = Difference(hexagon, SurroundingRectangle(self.inset_image, buff=0),
                             name="hexagon").set_stroke(WHITE, 0, 0)
        hexagon.set_fill(BLACK, 0.11)  # 0.11

        return hexagon

    @override_animation(Create)
    def _create_override(self, shift: np.ndarray = ORIGIN, scale: float = 1.0):
        anims = [Create(self.rect_on_main_axes, run_time=0.75)]

        inset_line_anims = [Create(line, run_time=0.5) for line in self.inset_lines]
        if hasattr(self, 'low_alpha_hexagon'):
            inset_line_anims.append(FadeIn(self.low_alpha_hexagon, run_time=0.75))

        anims.append(AnimationGroup(*inset_line_anims))
        anims.append(Create(self.pixel_visuals_rect, run_time=0.75))

        if hasattr(self, 'cs_island') and self.cs_island in self.submobjects:
            anims.append(Create(self.cs_island, run_time=0.5))

        if self.inset_image in self.submobjects:
            anims.append(FadeIn(self.inset_image, run_time=0.5))

        return AnimationGroup(*anims, lag_ratio=0.9)


def center_points_linspace(start: float, stop: float, num: int, retstep: bool = False) -> np.ndarray:
    d = (stop - start) / (2 * num)

    return np.linspace(start + d, stop - d, num, retstep=retstep)


def torch_center_points_linspace(start: float, stop: float, num: int, retstep: bool = False):
    d = (stop - start) / (2 * num)
    linspace = torch.linspace(start + d, stop - d, num)

    if retstep:
        step = (stop - start) / (num - 1)
        return linspace, step

    return linspace