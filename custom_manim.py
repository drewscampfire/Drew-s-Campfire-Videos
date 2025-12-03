from __future__ import annotations

import math
import random
from manim import *
from manim_config import fps, use_background, anticipate, dp_data_file_dir
import audioop
from manim.utils.paths import path_along_arc, path_along_circles
import types


from typing import Tuple, Sequence, Iterable, Literal, Callable, Generator, List, Optional
import torch

from functools import wraps, partial
from scipy.optimize import fsolve
from manim_config import FIRST_ROD_COLOR, SECOND_ROD_COLOR, file_type, device
import os
import inspect


SCENE_PIXELS = config.pixel_height / config.frame_height
STROKE_WIDTHS = config.frame_height / config.pixel_height
UPPER_LEFT_CORNER = np.array((-config.frame_width / 2, config.frame_height / 2, 0))
LOWER_RIGHT_CORNER = np.array((config.frame_width / 2, -config.frame_height / 2, 0))
UPPER_RIGHT_CORNER = np.array((config.frame_width / 2, config.frame_height / 2, 0))
LOWER_LEFT_CORNER = np.array((-config.frame_width / 2, -config.frame_height / 2, 0))


def play_timeline(scene, timeline):
    """
    Plays a timeline of animations on a given scene.

    Args:
        scene (Scene): The scene to play the animations on.
        timeline (dict): R dictionary where the keys are the times at which the animations should start,
            and the values are the animations to play at that time. The values can be a single animation
            or an iterable of animations.

    Notes:
        Each animation in the timeline can have a different duration, so several animations can be
        running in parallel. If the value for a given time is an iterable, all the animations
        in the iterable are started at once (although they can end at different times depending
        on their run_time)
        The method returns when all animations have finished playing.

    Example:
        timeline = {
            0: Create(progress_bar, run_time=12, rate_func=linear),
            1: Create(sq, run_time=10),
            2: [
                Create(c, run_time=4),
                Create(tri, run_time=2)
            ],
            9: Write(txt.next_to(progress_bar, UP, buff=1), run_time=3)
        }
        play_timeline(self, timeline)

    Returns:
        None
    """
    previous_t = 0
    ending_time = 0
    for t, anims in sorted(timeline.items()):
        to_wait = t - previous_t
        if to_wait > 0:
            scene.wait(to_wait)
        previous_t = t
        if not isinstance(anims, Iterable):
            anims = [anims]
        for anim in anims:
            turn_animation_into_updater(anim)
            scene.add(anim.mobject)
            ending_time = max(ending_time, t + anim.run_time)
    if ending_time > t:
        scene.wait(ending_time - t)


class ComplexScene(Scene):
    _registered_subscenes = []

    def add_background(self):
        if use_background:
            background = ImageMobject("background_image.png", name="scene_background").set_opacity(config.background_opacity)
            background.scale_to_fit_width(config.frame_width)
            self.add(background.set_z_index(-100))

    def play_subscenes(self: ComplexScene):
        for method, params in self._registered_subscenes:
            method_name = method.__qualname__.split('.')[1]
            print(f"\n{'#' * 50} running run {method_name} with {fps} fps "
                  f"and dims {config.pixel_height}x{config.pixel_width}\n")

            class_name = method.__qualname__.split(".")[0]
            if class_name != self.__class__.__name__:
                continue
            self.next_section(**params)
            method(self)

    def wait(self, duration=1, stop_condition=None, frozen_frame=False):
        """
        Custom wait method for this project.
        Defaults to frozen_frame=False to ensure complete PNG sequences
        for video editing.
        """
        # Call the original wait method from the parent Scene class
        super().wait(duration=duration, stop_condition=stop_condition, frozen_frame=frozen_frame)

    @classmethod
    def run(cls: ComplexScene, *args, **section_params):
        def wrapper(method, self, *m_args, **m_kwargs):
            return method(self, *m_args, **m_kwargs)

        def decorator(method):
            @wraps(method)
            def wrapped(self, *m_args, **m_kwargs):
                return wrapper(method, self, *m_args, **m_kwargs)
            cls._registered_subscenes.append((wrapped, section_params))
            config.output_file = f"{method.__qualname__.split('.')[1]}.{file_type}"
            return wrapped
        if args and callable(args[0]):
            return decorator(args[0])
        else:
            return decorator

    def skip(*args, **kwargs):
        kwargs.pop("skip_animations", None)

        return ComplexScene.run(*args, skip_animations=True, **kwargs)

    def ignore(*args, **kwargs):
        pass


class ComplexAnimation(Animation):
    def extra_animate(
            self,
            delay: float = 0.0,
            duration: float = 1.0,
            shift: np.ndarray = None,
            move: np.ndarray = None,
            scale: Optional[float] = None,
            rate_func: Callable[[float], float] = smooth,
            more_mobjects: Optional[List[Mobject]] = None,
    ) -> ComplexAnimation:
        if shift is not None and move is not None:
            raise ValueError("shift and move cannot be specified simultaneously")
        more_mobjects = more_mobjects or []
        if more_mobjects and hasattr(self, '_extra_animations'):
            raise ValueError("Cannot add more_mobjects more than once in extra_animate")

        if not hasattr(self, '_extra_animations'):
            self._extra_animations = []
            self._original_interpolate = self.interpolate_mobject
            self.group_mobjects = Group(self.mobject, *more_mobjects)
            self.group_mobjects_centers = [m.get_center() for m in self.group_mobjects.submobjects]
            self.group_mobjects_widths = [m.width for m in self.group_mobjects.submobjects]

            def wrapped_interpolate(self, alpha):
                self._original_interpolate(alpha)
                current_time = alpha * self.run_time

                if not hasattr(self, '_initialized'):
                    self._original_center = self.group_mobjects.get_center()
                    self._initialized = True

                sorted_anims = sorted(self._extra_animations, key=lambda anim: anim['delay'])
                center = self._original_center.copy()
                scale_factor = 1.0

                for anim in sorted_anims:
                    progress = np.clip((current_time - anim['delay']) / anim['duration'], 0, 1)
                    anim_alpha = anim['rate_func'](progress)

                    if anim['move'] is not None:
                        center = center * (1 - anim_alpha) + anim['move'] * anim_alpha
                    elif anim['shift'] is not None:
                        center += anim['shift'] * anim_alpha
                    if anim['scale'] is not None:
                        target_scale = anim['scale'] * scale_factor
                        scale_factor = scale_factor * (1 - anim_alpha) + target_scale * anim_alpha

                for i, mob in enumerate(self.group_mobjects.submobjects):
                    new_loc = center + (self.group_mobjects_centers[i] - self._original_center) * scale_factor
                    mob.move_to(new_loc)
                    mob.scale_to_fit_width(self.group_mobjects_widths[i] * scale_factor)

            self.interpolate_mobject = types.MethodType(wrapped_interpolate, self)

        self._extra_animations.append({
            'delay': delay,
            'duration': duration,
            'shift': shift,
            'move': move,
            'scale': scale,
            'rate_func': rate_func,
        })

        return self

    def finish(self):
        super().finish()
        cleanup_attrs = [
            '_extra_animations',
            '_initialized',
            '_original_center',
            'group_mobjects',
            'group_mobjects_centers',
            'group_mobjects_widths',
        ]
        for attr in cleanup_attrs:
            if hasattr(self, attr):
                delattr(self, attr)
        if hasattr(self, '_original_interpolate'):
            self.interpolate_mobject = self._original_interpolate
            del self._original_interpolate


class DNText(Text):
    def __init__(self, text="random", font='Jokerman', **kwargs):
        super().__init__(text, font=font, **kwargs)


def font_to_mob_class(font: str):
    def constructor(*args, **kwargs):
        return DNText(*args, font=font, **kwargs)
    return constructor


class DynamicAxes(VGroup, CoordinateSystem):
    """
    A generally more powerful version of the Axes Mobject.
    """
    def __init__(
        self,
        x_range: tuple[float, float] = (-180, 180),
        y_range: tuple[float, float] = (-180, 180),
        x_length: float = round(config.frame_width) - 2,
        y_length: float = round(config.frame_height) - 2,
        x_is_in_degrees: bool = False,
        y_is_in_degrees: bool = False,
        font_size_x: float = 24,
        font_size_y: float = 24,
        include_zero_lines: bool = False,
        use_constant_tick_length: bool = False,
        x_line_to_number_buff: float = 0.25,
        y_line_to_number_buff: float = 0.25,
        is_simplified_axis_ticks: bool = False,
        x_axis_color: ManimColor = FIRST_ROD_COLOR,
        y_axis_color: ManimColor = SECOND_ROD_COLOR,
        labeled_values_for_x_override: list | None = None,
        labeled_values_for_y_override: list | None = None,
        tick_length: float = 0.02,
    ) -> None:
        VGroup.__init__(self)
        CoordinateSystem.__init__(self, x_range, y_range, x_length, y_length)
        self.dyn_x_range = x_range
        self.dyn_y_range = y_range
        self.x_is_in_degrees = x_is_in_degrees
        self.y_is_in_degrees = y_is_in_degrees
        self.font_size_x = font_size_x
        self.font_size_y = font_size_y
        self.decimal_precision_x = None
        self.decimal_precision_y = None
        self.include_zero_lines = include_zero_lines
        self.use_constant_tick_length = use_constant_tick_length
        self.x_line_to_number_buff = x_line_to_number_buff
        self.y_line_to_number_buff = y_line_to_number_buff
        self.is_simplified_axis_ticks = is_simplified_axis_ticks
        self.x_axis_color = x_axis_color
        self.y_axis_color = y_axis_color
        self.labelled_values_for_x_override = labeled_values_for_x_override
        self.labelled_values_for_y_override = labeled_values_for_y_override
        self.tick_length = tick_length
        self.bg_rectangle = None

        self.axis_configs = {
            "include_numbers": False,
            "include_ticks": False,
            "exclude_origin_tick": False,
            "stroke_width": 2,
            "include_tip": False,
        }
        self.insets = []

        self.x_axis, self.x_ticks, self.x_labels = self.get_x_axis_ticks_labels()
        self.y_axis, self.y_ticks, self.y_labels = self.get_y_axis_ticks_labels()

        self.x_zero_line = None
        self.y_zero_line = None
        self.x_axis_label : None | VMobject = None
        self.y_axis_label : None | VMobject = None

        self.bg_rectangle = None
        if self.include_zero_lines:
            if x_range[0] < 0 < x_range[1]:
                self.x_zero_line = self.get_x_zero_line()
                self.add(self.x_zero_line)
            if y_range[0] < 0 < y_range[1]:
                self.y_zero_line = self.get_y_zero_line()
                self.add(self.y_zero_line)

        self.add(
            self.x_ticks,
            self.y_ticks,
            self.x_axis,
            self.y_axis,
            self.x_labels,
            self.y_labels
        )
        self.axis_components = VGroup(
            self.x_axis,
            self.y_axis,
            self.x_ticks,
            self.y_ticks,
            self.x_labels,
            self.y_labels
        )

    def get_background_rectangle(self) -> Polygon:
        bg_rect = Polygon(
            self.c2p((self.dyn_x_range[0], self.dyn_y_range[0], 0)),
            self.c2p((self.dyn_x_range[1], self.dyn_y_range[0], 0)),
            self.c2p((self.dyn_x_range[1], self.dyn_y_range[1], 0)),
            self.c2p((self.dyn_x_range[0], self.dyn_y_range[1], 0)),
            color=WHITE,  # WHITE
            fill_opacity=0.05,  # 0.05
            stroke_width=1,
            stroke_color=BLACK,
            stroke_opacity=0.5,
            name="bg_rectangle of DynamicAxes"
        )
        self.bg_rectangle = bg_rect

        return self.bg_rectangle

    def get_x_zero_line(self):
        assert self.dyn_x_range[0] < 0 < self.dyn_x_range[1], f"wrong x range: {self.dyn_x_range}"
        assert self.include_zero_lines, "include_zero_lines must be True"

        return DashedLine(
            start=self.coords_to_point([0, self.dyn_y_range[0], 0]),
            end=self.coords_to_point([0, self.dyn_y_range[1], 0]),
            color=self.x_axis_color,
            dash_length=0.1,
            stroke_width=1,
            name="x_zero_line of DynamicAxes"
        )

    def get_y_zero_line(self):
        assert self.dyn_y_range[0] < 0 < self.dyn_y_range[1], f"wrong y range: {self.dyn_y_range}"
        assert self.include_zero_lines, "include_zero_lines must be True"

        return DashedLine(
            start=self.coords_to_point([self.dyn_x_range[0], 0, 0]),
            end=self.coords_to_point([self.dyn_x_range[1], 0, 0]),
            color=self.y_axis_color,
            dash_length=0.1,
            stroke_width=1,
            name="y_zero_line of DynamicAxes"
        )

    @staticmethod
    def get_labels(
            numberline: NumberLine,
            dict_values: dict[float, str | float | VMobject],
            direction: Sequence[float] = None,
            buff: float | None = None,
            font_size: float | None = None,
    ) -> VGroup:
        direction = numberline.label_direction if direction is None else direction
        buff = numberline.line_to_number_buff if buff is None else buff
        font_size = numberline.font_size if font_size is None else font_size

        labels = VGroup(name="labels of DynamicAxes")
        for x, label in dict_values.items():
            label = Tex(label)

            if hasattr(label, "font_size"):
                label.font_size = font_size
            else:
                raise AttributeError(f"{label} is not compatible with add_labels.")
            label.next_to(numberline.number_to_point(x), direction=direction, buff=buff)
            labels.add(label)

        return labels

    def _get_values_to_add_label_to(
            self,
            axis_range: tuple[float, float],
            axis: Literal["x", "y"],
            arange_threshold=1e-12,
            tolerance=1e-9
    ) -> tuple[list[float], list[float]]:

        def smallest_divisible(a, s):
            if a % s < tolerance:
                return a
            else:
                return ((a // s) + 1) * s

        def get_vals(mini, maxi, step) -> list[float]:
            assert mini < maxi, f"mini {mini} must be smaller than maxi {maxi}"
            mini_adjusted = mini - arange_threshold
            maxi_adjusted = maxi + arange_threshold

            if maxi <= 0:
                new_maxi = smallest_divisible(abs(maxi), step)
                # Generate values directly from maxi to mini with negative step
                return np.arange(-new_maxi, mini_adjusted, -step).tolist()[::-1]

            elif mini >= 0:
                new_mini = smallest_divisible(mini, step)
                # This case is already correct, generate ascending values from mini to maxi
                return np.arange(new_mini, maxi_adjusted, step).tolist()

            elif mini < 0 < maxi:
                # Handle both negative and positive ranges
                # Generate negative values starting from mini and positive values up to maxi
                mini_vals = np.arange(0, mini_adjusted, -step)[::-1]
                maxi_vals = np.arange(0, maxi_adjusted, step)[1:]  # Skip 0 to avoid duplicate
                return np.concatenate((mini_vals, maxi_vals)).tolist()

            raise ValueError("must not reach this point")

        min_val, max_val = axis_range
        assert min_val < max_val, "values in axis_range must be in ascending order"

        span = max_val - min_val
        span = round(span, 8)
        expo = math.floor(math.log10(span))
        divisor = 10 ** expo
        quotient = span / divisor

        if expo >= 1:
            if axis == "x":
                self.decimal_precision_x = 0
            else:
                self.decimal_precision_y = 0
        else:
            if axis == "x":
                self.decimal_precision_x = int(abs(expo)) + 1
            else:
                self.decimal_precision_y = int(abs(expo)) + 1

        # special case handling
        if axis_range == (-180, 180) or axis_range == (-180.0, 180.0):
            vals = np.linspace(-180, 180, 9) if not self.is_simplified_axis_ticks \
                else np.linspace(-180, 180, 5)
            return vals.tolist(), [-180, 0, 180]

        if quotient < 1 - tolerance:
            step = 0.1 * divisor
        elif quotient < 2 - tolerance:
            step = 0.2 * divisor
        elif quotient < 5 - tolerance:
            step = 0.5 * divisor
        elif quotient <= 10:
            step = divisor
        else:
            raise ValueError(f"must not reach this part, quotient: {quotient}, tolerance: {tolerance}, span: {span}, "
                             f"divisor: {divisor}, expo: {expo}, min_val: {min_val}, max_val: {max_val}")

        return get_vals(min_val, max_val, step), get_vals(min_val, max_val, divisor * 10)

    @staticmethod
    def _deci_count(num: float):
        num = float(f"{num:.12f}")
        # Convert the number to a string
        num_str = str(num)

        # Check if there's a decimal point
        if '.' in num_str:
            # Split the number on the decimal point
            integer_part, decimal_part = num_str.split('.')

            # Strip trailing zeros from the decimal part and count the remaining digits
            return len(decimal_part.rstrip('0'))
        else:
            # If there's no decimal point, return 0
            return 0

    def get_proportional_buff_factor(self, axis: Literal["x", "y"]) -> float:
        def buff_func(x):
            return 0.1 * x + 0.3
        if axis == "x":
            return buff_func(self.x_length)
        elif axis == "y":
            return buff_func(self.y_length)
        else:
            raise ValueError("axis must be 'x' or 'y'")

    def get_x_axis_ticks_labels(self):
        x_axis = NumberLine(
            self.dyn_x_range,
            self.x_length,
            font_size=self.font_size_x,
            line_to_number_buff=self.x_line_to_number_buff,
            name="x_axis",
            **self.axis_configs
        )
        x_axis.shift(DOWN * self.y_length / 2)

        x_num_labels, next_x_num_labels = self._get_values_to_add_label_to(self.dyn_x_range, "x")
        if self.labelled_values_for_x_override is not None:
            x_num_labels, next_x_num_labels = self.labelled_values_for_x_override, []

        if self.x_is_in_degrees:
            dict_values_x = {
                i: f"{i:.{min(self.decimal_precision_x, self._deci_count(i))}f}$^{{\\circ}}$"
                for i in x_num_labels
            }
        else:
            dict_values_x = {
                i: rf"{i:.{min(self.decimal_precision_x, self._deci_count(i))}f}"
                for i in x_num_labels
            }

        x_labels = self.get_labels(
            x_axis,
            dict_values_x,
            DOWN * self.get_proportional_buff_factor("x"),
            font_size=self.font_size_x,
        )

        longer_tick_length = self.tick_length * 2 if not self.use_constant_tick_length else self.tick_length
        x_ticks = VGroup(
            *[x_axis.get_tick(i, 7.15 * longer_tick_length if i in next_x_num_labels
            else 7.15 * self.tick_length) for i in x_num_labels], name="x_ticks"
        )
        if self.dyn_x_range[0] == x_num_labels[0]:
            x_ticks.submobjects[0].scale(0.5).shift(DOWN * 7.15 * self.tick_length / 2)

        x_axis.set_color(self.x_axis_color)
        x_ticks.set_color(self.x_axis_color)
        x_labels.set_color(self.x_axis_color)

        return x_axis, x_ticks, x_labels

    def get_y_axis_ticks_labels(self):
        y_axis = NumberLine(
            self.dyn_y_range,
            self.y_length,
            rotation=PI / 2,
            label_direction=LEFT,
            font_size=self.font_size_y,
            line_to_number_buff=self.y_line_to_number_buff,
            name="y_axis",
            **self.axis_configs
        )
        y_axis.shift(LEFT * self.x_length / 2)

        y_num_labels, next_y_num_labels = self._get_values_to_add_label_to(self.dyn_y_range, "y")
        if self.labelled_values_for_y_override is not None:
            y_num_labels, next_y_num_labels = self.labelled_values_for_y_override, []

        if self.y_is_in_degrees:
            dict_values_y = {
                i: f"{i:.{min(self.decimal_precision_y, self._deci_count(i))}f}$^{{\\circ}}$"
                for i in y_num_labels
            }
        else:
            dict_values_y = {
                i: rf"{i:.{min(self.decimal_precision_y, self._deci_count(i))}f}"
                for i in y_num_labels
            }

        y_labels = self.get_labels(
            y_axis,
            dict_values_y,
            LEFT * self.get_proportional_buff_factor("y"),
            font_size=self.font_size_y
        )

        longer_tick_length = self.tick_length * 2 if not self.use_constant_tick_length else self.tick_length
        y_ticks = VGroup(
            *[y_axis.get_tick(i, 7.15 * longer_tick_length if i in next_y_num_labels
              else 7.15 * self.tick_length) for i in y_num_labels], name="y_ticks"
        )
        if self.dyn_y_range[0] == y_num_labels[0]:
            y_ticks.submobjects[0].scale(0.5).shift(LEFT * 7.15 * self.tick_length / 2)

        y_axis.set_color(self.y_axis_color)
        y_ticks.set_color(self.y_axis_color)
        y_labels.set_color(self.y_axis_color)

        return y_axis, y_ticks, y_labels

    def coords_to_point(self, coords: Sequence) -> np.ndarray:
        """
        coords can be a 1D or 2D array
        """
        coords = np.asarray(coords)
        if coords.ndim == 1:
            point = np.array([
                self.x_axis.number_to_point(coords[0])[0],
                self.y_axis.number_to_point(coords[1])[1],
                0
            ])

            return point
        elif coords.ndim == 2:
            points = []
            for coord in coords:
                x = self.x_axis.number_to_point(coord[0])[0]
                y = self.y_axis.number_to_point(coord[1])[1]
                points.append([x, y, 0])

            return np.asarray(points)
        else:
            raise ValueError("coords must be 1D or 2D array")

    def get_axes(self):
        return self.x_axis, self.y_axis

    def _update_axis_range(self,
                           axis_numberline: NumberLine,
                           new_range: tuple[float, float],
                           axis: Literal["x", "y"],):
        assert axis == "x" or axis == "y", "axis must be 'x' or 'y'"
        axis_numberline.x_range = (new_range[0], new_range[1], 1)
        new_axis, new_ticks, new_labels = self.get_x_axis_ticks_labels() if axis == "x" else self.get_y_axis_ticks_labels()

        if axis == "x":
            if self.include_zero_lines:
                if new_range[0] < 0 < new_range[1]:
                    new_x_zero_line = self.get_x_zero_line()
                    if self.x_zero_line in self.submobjects:
                        self.x_zero_line.become(new_x_zero_line)
                    else:
                        self.add(new_x_zero_line)
                else:
                    if self.x_zero_line in self.submobjects:
                        self.remove(self.x_zero_line)
                        self.x_zero_line.become(VMobject())

            self.x_axis.become(new_axis)
            self.x_ticks.become(new_ticks)
            self.x_labels.become(new_labels)
        elif axis == "y":
            if self.include_zero_lines:
                if new_range[0] < 0 < new_range[1]:
                    new_y_zero_line = self.get_y_zero_line()
                    if self.y_zero_line in self.submobjects:
                        self.y_zero_line.become(new_y_zero_line)
                    else:
                        self.add(new_y_zero_line)
                else:
                    if self.y_zero_line in self.submobjects:
                        self.remove(self.y_zero_line)
                        self.y_zero_line.become(VMobject())
            self.y_axis.become(new_axis)
            self.y_ticks.become(new_ticks)
            self.y_labels.become(new_labels)
        else:
            raise ValueError("axis must be 'x' or 'y'")

    def update_x_range(self, new_x_range: tuple[float, float]):
        self.dyn_x_range = new_x_range
        self._update_axis_range(
            self.x_axis,
            new_x_range,
            "x")

    def update_y_range(self, new_y_range: tuple[float, float]):
        self.dyn_y_range = new_y_range
        self._update_axis_range(
            self.y_axis,
            new_y_range,
            "y")

    def get_explicit_plot(
            self,
            func: Callable[[float], float],
            num_of_points: int = 1000,
            x_range: tuple[float, float] | None = None,
            use_smoothing: bool = True,
            stroke_width: int = 2,
            color: ManimColor = YELLOW,
            **kwargs) -> Plotter:
        if x_range is None:
            x_range = self.dyn_x_range[:2]
        x_values = np.linspace(x_range[0], x_range[1], num_of_points)

        points = []
        for x in x_values:
            try:
                y = func(x)
                points.append([x, y, 0])
            except ValueError:
                # Skip the point if func(x) raises ValueError
                pass

        points = np.array(points)

        return Plotter(
            points,
            self,
            use_smoothing,
            stroke_width,
            [color],
            **kwargs
        )

    @property
    def location(self) -> np.ndarray:
        mid_x_coord = (self.dyn_x_range[0] + self.dyn_x_range[1]) / 2
        mid_y_coord = (self.dyn_y_range[0] + self.dyn_y_range[1]) / 2
        return self.coords_to_point((mid_x_coord, mid_y_coord))

    def create_rect_on_axes(self, x_range: tuple[float, float], y_range: tuple[float, float]):
        return Polygon(
                    self.coords_to_point((x_range[0], y_range[1])),
                    self.coords_to_point((x_range[1], y_range[1])),
                    self.coords_to_point((x_range[1], y_range[0])),
                    self.coords_to_point((x_range[0], y_range[0])),
                )

    @override_animation(Create)
    def _create_override(self, shift: np.ndarray = ORIGIN, scale: float = 1.0, y_axis_first: bool = False,
                         slow_factor: float = 1.0):
        anims = []
        if self.bg_rectangle:
            anims.append(FadeIn(self.bg_rectangle, shift=shift, scale=scale,
                                rate_func=anticipate, run_time=1
                                ))

        x_kwargs = {"shift": UP * 0.2, "scale": 0.9, "run_time": 0.5}
        y_kwargs = {"shift": RIGHT * 0.2, "scale": 0.9, "run_time": 0.5}

        # Define axis creation functions with the new component order
        def create_x_axis_anims():
            x_anims = []
            # 1. Main Axis Line
            x_anims.append(Create(self.x_axis, run_time=slow_factor * self.x_axis.length / 10, rate_func=linear))

            # 2. Axis Label
            if self.x_axis_label:
                x_anims.append(FadeIn(self.x_axis_label, rate_func=smootherstep, **x_kwargs))

            # 3. Ticks and Numerical Labels
            x_anims_group = []
            for x_tick, x_label in zip(self.x_ticks, self.x_labels):
                x_anims_group.append(FadeIn(Group(x_tick, x_label), rate_func=create_overshoot_func(), **x_kwargs))
            x_anims.append(AnimationGroup(*x_anims_group, lag_ratio=0.2))

            # 4. Zero Line
            if self.x_zero_line:
                x_anims.append(Write(self.x_zero_line, rate_func=linear, run_time=slow_factor * self.x_zero_line.get_length() / 10))
            return x_anims

        def create_y_axis_anims():
            y_anims = []
            # 1. Main Axis Line
            y_anims.append(Create(self.y_axis, run_time=slow_factor * self.y_axis.length / 10, rate_func=linear))

            # 2. Axis Label
            if self.y_axis_label:
                y_anims.append(FadeIn(self.y_axis_label, rate_func=smootherstep, **y_kwargs))

            # 3. Ticks and Numerical Labels
            y_anims_group = []
            for y_tick, y_label in zip(self.y_ticks, self.y_labels):
                y_anims_group.append(FadeIn(Group(y_tick, y_label), rate_func=create_overshoot_func(), **y_kwargs))
            y_anims.append(AnimationGroup(*y_anims_group, lag_ratio=0.2))

            # 4. Zero Line
            if self.y_zero_line:
                y_anims.append(Write(self.y_zero_line, rate_func=linear, run_time=slow_factor * self.y_zero_line.get_length() /
                                                                                  10))
            return y_anims

        # Add animations based on the y_axis_first parameter
        if y_axis_first:
            anims.extend(create_y_axis_anims())
            anims.extend(create_x_axis_anims())
        else:  # Default: x-axis first
            anims.extend(create_x_axis_anims())
            anims.extend(create_y_axis_anims())

        return AnimationGroup(*anims, lag_ratio=0.9)


class Plotter(VGroup):
    def __init__(self,
                 points: np.ndarray,
                 axes: DynamicAxes,
                 use_smoothing: bool = True,
                 stroke_wid: float = 1,
                 col: Sequence[ManimColor] | None = None,
                 stroke_opa: float = 1.0,
                 start_dot: VMobject | None = None,
                 tracer: VMobject | None = None,
                 use_dots: bool = True,
                 **kwargs):
        super().__init__(tolerance_for_point_equality=0, **kwargs)
        self.graph = VMobject(tolerance_for_point_equality=0)
        self.graph.set_stroke(width=stroke_wid, opacity=stroke_opa)
        self.converted_graph = VGroup(tolerance_for_point_equality=0)

        self.passed_points = points
        self.axes = axes
        self.use_smoothing = use_smoothing
        self.stroke_wid = stroke_wid
        self.stroke_opa = stroke_opa
        self.col = col if col else [LIGHT_BROWN, DARK_BROWN]
        self.start_index = 0
        self.end_index = len(self.passed_points)
        self.use_dots = use_dots

        self.add(self.converted_graph)
        if self.use_dots:
            self.start_dot = start_dot if start_dot \
                else HollowDot(
                0.075,
                0.7,
                self.col[0],
                self.col[0],
                0.15,
                1
            )
            self.tracer = tracer if tracer \
                else Circle(
                self.start_dot.radius * self.start_dot.hole_ratio,
                self.col[0],
                stroke_width=1, fill_opacity=1, stroke_color=1, stroke_opacity=1
            )
        self.update_axes(self.axes)
        if self.use_dots:
            self.add(self.start_dot, self.tracer)
            self.start_dot.move_to(self.axes.c2p(self.passed_points[0]))

    def update_plotter(self):
        self.graph.clear_points()
        # First, filter points based on the axes' x_range and y_range
        x_min, x_max = self.axes.dyn_x_range[:2]
        y_min, y_max = self.axes.dyn_y_range[:2]

        passed_points_copy = np.array([
            point for point in self.passed_points[self.start_index:self.end_index]
            if x_min <= point[0] <= x_max and y_min <= point[1] <= y_max
        ])
        temp_num_of_points = len(passed_points_copy)

        def get_wrapped_indices():
            indices = []
            x_width = self.axes.dyn_x_range[1] - self.axes.dyn_x_range[0]
            y_height = self.axes.dyn_y_range[1] - self.axes.dyn_y_range[0]

            for i in range(1, temp_num_of_points):
                a = passed_points_copy[i - 1]
                b = passed_points_copy[i]

                dx = abs(b[0] - a[0])
                dy = abs(b[1] - a[1])

                # Check if the segment wraps around x or y axis
                if (x_width * 0.5 <= dx <= x_width * 1.2) or \
                        (y_height * 0.5 <= dy <= y_height * 1.2):
                    indices.append(i)

            return indices

        wrapped_indices = get_wrapped_indices()

        #  align endpoints of subpaths to axes
        for index in wrapped_indices:
            a, b = passed_points_copy[index - 1: index + 1]
            axis = 0 if abs(b[0] - a[0]) > abs(b[1] - a[1]) else 1  # 0 for x_axis, 1 for y_axis
            average_of_the_other_axis = (a[axis ^ 1] + b[axis ^ 1]) / 2
            if axis == 0:
                a[axis] = self.axes.dyn_x_range[1 if a[axis] > 0 else 0]
                b[axis] = self.axes.dyn_x_range[1 if b[axis] > 0 else 0]
            else:
                a[axis] = self.axes.dyn_y_range[1 if a[axis] > 0 else 0]
                b[axis] = self.axes.dyn_y_range[1 if b[axis] > 0 else 0]
            a[axis ^ 1] = b[axis ^ 1] = average_of_the_other_axis

        point_coordinates_on_axes = self.axes.coords_to_point(passed_points_copy)

        extended_indices = [0] + wrapped_indices + [temp_num_of_points]
        if temp_num_of_points:
            for i, index in enumerate(extended_indices[:-1]):
                next_index = extended_indices[i + 1]
                if next_index - index <= 1:
                    # Skip short segments
                    continue
                self.graph.start_new_path(point_coordinates_on_axes[index])
                self.graph.add_points_as_corners(point_coordinates_on_axes[index + 1:next_index])

        if self.use_smoothing:
            self.graph.make_smooth()

        if self.use_dots:
            self.tracer.move_to(self.axes.c2p(self.passed_points[self.end_index - 1]))
            self.tracer.set_fill(interpolate_colors(self.col, t=self.end_index / len(self.passed_points)))
        self.converted_graph.become(split_vmobject_and_put_in_vgroup(
            self.graph,
            self.use_smoothing,
            self.col,
            len(self.passed_points)
        ))

    def update_axes(self, new_axes: DynamicAxes):
        self.axes = new_axes
        self.update_plotter()

    def set_indices(self, start_index: int, end_index: int):
        self.start_index = start_index
        self.end_index = end_index
        self.update_plotter()

    def revert_tracer(self):
        assert self.use_dots, "Cannot revert without dots"

        self.tracer.move_to(self.axes.c2p(self.passed_points[0]))
        self.tracer.set_fill(self.col[0])


class HollowDot(VGroup):
    def __init__(self,
                 radius: float = 1.0,
                 hole_ratio=0.5,
                 col: ManimColor = PINK,
                 middle_color: ManimColor = YELLOW,
                 inner_opacity: float = 1,
                 outer_opacity: float = 1,
                 **kwargs):
        super().__init__(stroke_width=0, **kwargs)
        self.radius = radius
        self.hole_ratio = hole_ratio
        self.outer = Annulus(inner_radius=radius * hole_ratio, outer_radius=radius,
                             color=col, stroke_width=1, stroke_color=BLACK, fill_opacity=outer_opacity)
        self.hole = Circle(radius=radius * hole_ratio, stroke_width=1, stroke_color=BLACK,
                           color=middle_color, fill_opacity=inner_opacity)
        if middle_color:
            self.add(self.hole)
        self.add(self.outer)


class MeasureLabel(VGroup):
    def __init__(
            self,
            label: Text | Tex | MathTex,
            start: np.ndarray,
            end: np.ndarray,
            label_position: float = 0.5,
            label_frame: bool = False,
            label_color: ManimColor = WHITE,
            frame_fill_color: ManimColor | None = None,
            frame_fill_opacity: float = 1,
            include_wings: bool = False,
            wings_length: float = 0.2,
            label_buff: float = SMALL_BUFF,
            **kwargs
    ):
        """
        Create a labeled measurement line with wings at both ends.

        Parameters:
        - label: Text | Tex | MathTex
            The label to be displayed.
        - start: np.ndarray
            The starting point of the main line.
        - end: np.ndarray
            The ending point of the main line.
        - **kwargs
            Additional arguments passed to Line().
        """
        super().__init__()

        main_line = Line(start, end, **kwargs)
        label.move_to(main_line.point_from_proportion(label_position))
        line1 = Line(
            start,
            get_closest_corner(SurroundingRectangle(label, buff=label_buff), start)
        ).match_style(main_line)
        line2 = Line(
            get_closest_corner(SurroundingRectangle(label, buff=label_buff), end),
            end
        ).match_style(main_line)

        self.add(line1, line2, label)

        if label_frame:
            box_frame = SurroundingRectangle(
                label, buff=0.05, color=label_color, stroke_width=0.5
            )
            self.add(box_frame)
            if frame_fill_color:
                box = BackgroundRectangle(
                    label,
                    buff=0.05,
                    color=frame_fill_color,
                    fill_opacity=frame_fill_opacity,
                    stroke_width=0.5,
                )
                self.add(box)

        if include_wings:
            direction_vector = end - start
            perpendicular_vector = np.array([-direction_vector[1], direction_vector[0], 0])
            perpendicular_vector /= np.linalg.norm(perpendicular_vector)
            for point in [start, end]:
                wing_start = point + wings_length * perpendicular_vector
                wing_end = point - wings_length * perpendicular_vector
                wing_line = Line(wing_start, wing_end).match_style(main_line)
                self.add(wing_line)


class Move(Transform):
    def __init__(self, mobject: Mobject, new_location: np.ndarray, **kwargs):
        end = new_location
        super().__init__(mobject, mobject.copy().move_to(end), **kwargs)


class MoveRelative(Transform):
    def __init__(
            self,
            mobject: Mobject,
            obj_point: np.ndarray,
            new_location: np.ndarray,
            **kwargs
    ):
        end = mobject.get_center() + new_location - obj_point
        super().__init__(mobject, mobject.copy().move_to(end), **kwargs)


class Shift(Transform):
    def __init__(self, mobject: Mobject, shift_amount: np.ndarray, **kwargs):
        end = mobject.get_center() + shift_amount
        super().__init__(mobject, mobject.copy().move_to(end), **kwargs)


class ManualScaleAnimation(Animation):
    def __init__(
            self,
            mobject: Mobject,
            scale_times: dict,
            rate_func: Callable[[float], float] = linear,
            run_time: float = 5,
            **kwargs
    ):
        scale_times.setdefault(0, 1)
        times = sorted(scale_times.keys())
        assert all(t1 < t2 for t1, t2 in zip(times, times[1:])), \
            "Keys in scale_times must be in increasing order."

        super().__init__(mobject, run_time=run_time, rate_func=linear, **kwargs)
        self.times = times
        self.sizes = [scale_times[time] for time in self.times]
        self.scale_rate_func = rate_func
        self.current_scale = 1
        self.total_time = 0  # Initialize total time
        self.about_point = mobject.get_center() + 0.1 * UP

    def interpolate_mobject(self, alpha: float) -> None:
        current_time = alpha * self.run_time
        for i, time in enumerate(self.times):
            if current_time < time:
                break
        else:
            self.mobject.scale(self.sizes[-1] / self.current_scale, about_point=self.about_point)
            self.current_scale = self.sizes[-1]
            return

        if i == 0:
            start_time, start_size = 0, self.sizes[0]
        else:
            start_time, start_size = self.times[i - 1], self.sizes[i - 1]

        end_time, end_size = self.times[i], self.sizes[i]
        t = (current_time - start_time) / (end_time - start_time)
        current_size = interpolate(start_size, end_size, t)
        try:
            current_size = apply_rate_func_to_scalar_within_bounds(
                current_size,
                self.scale_rate_func,
                (start_size, end_size)
            )
        except AssertionError:
            print("AssertionError in ManualScaleAnimation.interpolate_mobject")
            print(f"current_size: {current_size}, start_size: {start_size}, end_size: {end_size}, t: {t}")

        self.mobject.scale(current_size / self.current_scale, about_point=self.about_point)
        self.current_scale = current_size

    def update(self, dt):
        self.total_time += dt
        alpha = self.total_time / self.run_time
        self.interpolate_mobject(alpha)


class FadeReplacementTransform(Animation):
    """
        A transform animation that fades out the original object and fades in the target object.

        Note:
            This animation should only be applied to objects with full opacity.
            Applying this animation to objects with partial opacity may result in unexpected behavior.
    """
    def __init__(
            self,
            start_mobject: Mobject,
            target_mobject: Mobject,
            path_arc: float = 0,
            path_arc_axis: np.ndarray = OUT,
            path_arc_centers: np.ndarray = None,
            **kwargs
    ) -> None:
        # Store path_arc parameters
        self.path_arc = path_arc
        self.path_arc_axis = path_arc_axis
        self.path_arc_centers = path_arc_centers
        self.path_func = None
        if path_arc != 0 or path_arc_centers is not None:
            if path_arc_centers is not None:
                self.path_func = path_along_circles(
                    path_arc,
                    path_arc_centers,
                    path_arc_axis,
                )
            else:
                self.path_func = path_along_arc(
                    arc_angle=path_arc,
                    axis=path_arc_axis,
                )

        self.start_mobject = start_mobject
        self.target_mobject = target_mobject
        self.starting_target = None

        self.transform_anims = []
        self._init_transforms()

        self.mobject_group = Group(start_mobject, self.starting_target)
        super().__init__(self.mobject_group, **kwargs)

    def _create_transform_for_mob(self, mob1: Mobject, mob2: Mobject) -> Animation:
        conditions = [
            ("path_func", self.path_func, self.path_func is not None),
            ("path_arc", self.path_arc, self.path_arc != 0),
            ("path_arc_axis", self.path_arc_axis, self.path_arc_axis is not OUT),
            ("path_arc_centers", self.path_arc_centers, self.path_arc_centers is not None),
        ]
        transform_kwargs = {key: value for key, value, cond in conditions if cond}

        return Transform(mob1, mob2, rate_func=linear, **transform_kwargs)

    @staticmethod
    def create_faded_copy(mob_to_copy: Mobject, dest_mob: Mobject) -> Mobject:
        faded_copy = mob_to_copy.copy()
        faded_copy.move_to(dest_mob)
        faded_copy.stretch_to_fit_width(dest_mob.width)
        faded_copy.stretch_to_fit_height(dest_mob.height)
        if isinstance(faded_copy, Group):
            [m.set_opacity(0) for m in faded_copy.submobjects]
        elif isinstance(faded_copy, VMobject):
            faded_copy.set_opacity(0)

        return faded_copy

    def _init_transforms(self):
        faded_start = self.create_faded_copy(self.start_mobject, self.target_mobject)
        self.transform_anims.append(
            self._create_transform_for_mob(self.start_mobject, faded_start)
        )

        self.starting_target = self.create_faded_copy(self.target_mobject, self.start_mobject)
        self.transform_anims.append(
            self._create_transform_for_mob(self.starting_target, self.target_mobject)
        )

    def begin(self) -> None:
        for anim in self.transform_anims:
            anim.begin()
        super().begin()

    def interpolate_mobject(self, alpha: float):
        alpha = self.rate_func(alpha)

        # Update all transform animations
        for anim in self.transform_anims:
            anim.interpolate(alpha)

        # # Handle non-VMobject fading
        # for i, mob in enumerate(self.mobject_group.submobjects):
        #     if not isinstance(mob, VMobject):
        #         mob.set_opacity(1 - alpha if i == 0 else alpha)

    def clean_up_from_scene(self, scene: Scene) -> None:
        super().clean_up_from_scene(scene)
        scene.remove(self.mobject_group)
        for anim in self.transform_anims:
            anim.clean_up_from_scene(scene)

        scene.add(self.target_mobject)


def get_closest_corner(mobject: Mobject, point: np.ndarray) -> np.ndarray:
    critical_points = [
        mobject.get_critical_point(UL),
        mobject.get_critical_point(UR),
        mobject.get_critical_point(DL),
        mobject.get_critical_point(DR),
        mobject.get_critical_point(UP),
        mobject.get_critical_point(DOWN),
        mobject.get_critical_point(LEFT),
        mobject.get_critical_point(RIGHT)
    ]

    closest_crit_point = min(
        critical_points,
        key=lambda crit_point: np.linalg.norm(crit_point - point)
    )

    return closest_crit_point


def split_vmobject_and_put_in_vgroup(
        vmob: VMobject,
        use_smoothing: bool = False,
        col_gradient: Sequence[ManimColor] = (BLUE, GREEN),
        override_vmob_curve_count: int | None = None
) -> VGroup:
    vgroup = VGroup().match_style(vmob)
    if override_vmob_curve_count is None:
        vmobs_count = vmob.get_num_curves() + len(vmob.get_subpaths())
    else:
        vmobs_count = override_vmob_curve_count

    points_count = 0
    for subpath in vmob.get_subpaths():
        n = len(subpath)
        if n % 4 == 0:
            for i in range(0, n, 4):
                segment = VMobject()
                if use_smoothing:
                    segment.set_points_smoothly(subpath[i:i + 4])
                else:
                    segment.set_points_as_corners(subpath[i:i + 4])
                segment.set_stroke(
                    interpolate_colors(col_gradient, t=(points_count/4) / vmobs_count),
                    vmob.stroke_width,
                    vmob.stroke_opacity)
                vgroup.add(segment)
                points_count += 4
        else:
            raise ValueError(
                f"Subpath {subpath} does not contain a complete set of points for Bezier curves. "
            )

    return vgroup


def line_to_rectangle(line: Line) -> Rectangle:
    """
    Convert a Line object into a Rectangle object with a given stroke width.

    Args:
    line (Line): The Line object to convert.
    stroke_width (float): The stroke width to simulate with the rectangle's height.

    Returns:
    Rectangle: R Rectangle object representing the line with the specified stroke width.
    """
    line_angle = line.get_angle()

    rectangle = Rectangle(
        width=line.get_length(),
        height=line.stroke_width / 100,
        stroke_width=0,
        fill_opacity=1
    )
    rectangle.set_color(line.get_color())

    rectangle.move_to(line.get_center())
    rectangle.rotate(line_angle)

    return rectangle


def move_relative_to(obj: Mobject, obj_point: np.ndarray, new_location: np.ndarray):
    obj.shift(new_location - obj_point)


def create_accelerating_function(slope, crit):
    if not 0 <= crit <= 1:
        raise ValueError("crit must be between 0 and 1")

    # Calculate the value of the function at the critical point using the linear part
    linear_value_at_crit = slope * crit

    # Define the function to find the coefficients of the quadratic polynomial
    def find_coefficients(vars):
        a, b, c = vars
        # Constraints
        eq1 = a * crit**2 + b * crit + c - linear_value_at_crit  # Continuity: f(crit) = linear part at crit
        eq2 = 2 * a * crit + b - slope                           # Smooth derivative transition: f'(crit) = slope
        eq3 = a * 1**2 + b * 1 + c - 1                           # f(1) = 1

        return [eq1, eq2, eq3]

    a, b, c = fsolve(find_coefficients, [0, 0, linear_value_at_crit])

    def generated_function(t):
        if t <= crit:
            return slope * t
        else:
            return a * t**2 + b * t + c

    return generated_function


def create_piecewise_linear_accelerating_function(slope, crit):
    if not 0 <= crit <= 1:
        raise ValueError("crit must be between 0 and 1")

    linear_value_at_crit = slope * crit
    new_slope = (1 - linear_value_at_crit) / (1 - crit)

    def generated_function(t):
        if t <= crit:
            return slope * t
        else:
            return new_slope * (t - crit) + linear_value_at_crit

    return generated_function


def anti_over(n: float = 0.5) -> Callable[[float], float]:
    P0 = np.array([0, 0])
    P1 = np.array([n, -n])
    P2 = np.array([1 - n, 1 + n])
    P3 = np.array([1, 1])

    def bezier_func(t: float) -> float:
        point = (
            (1 - t) ** 3 * P0 +
            3 * (1 - t) ** 2 * t * P1 +
            3 * (1 - t) * t ** 2 * P2 +
            t ** 3 * P3
        )
        return float(point[1])  # Return only the y-component as a float

    return bezier_func


def apply_rate_func_to_scalar_within_bounds(
        scalar: float,
        rate_func: Callable[[float], float],
        bounds: Tuple[float, float],
        tolerance: float = 1e-13
) -> float:
    lower_bound, upper_bound = min(bounds), max(bounds)

    # Check if scalar is within bounds, allowing for small tolerance
    if not (lower_bound - tolerance <= scalar <= upper_bound + tolerance):
        raise ValueError(f"scalar {scalar} is not within bounds {bounds}")

    scalar = max(lower_bound, min(scalar, upper_bound))

    span = upper_bound - lower_bound
    if span < tolerance:
        return lower_bound

    return rate_func((scalar - lower_bound) / span) * span + lower_bound


def apply_rate_func_to_linear_sequence(
        sequence: np.ndarray,
        rate_func: Callable[[float], float],
        tolerance: float = 1e-13
) -> np.ndarray:
    bounds = (float(sequence[0]), float(sequence[-1]))
    return np.array([
        apply_rate_func_to_scalar_within_bounds(val, rate_func, bounds, tolerance)
        for val in sequence
    ])


def get_zoom_range_values(
    old_range: Tuple[float, float],
    new_range: Tuple[float, float],
    num_of_frames: int,
    zoom_rate_func: Callable[[float], float]
) -> np.ndarray:
    assert old_range[0] < old_range[1], "old_range must be in ascending order"
    assert new_range[0] < new_range[1], "new_range must be in ascending order"

    old_mean, new_mean = np.mean(old_range), np.mean(new_range)

    old_width, new_width = old_range[1] - old_range[0], new_range[1] - new_range[0]
    range_prog = np.geomspace(old_width, new_width, num_of_frames)
    step_ratio = range_prog[1] / range_prog[0]
    range_prog = np.exp(apply_rate_func_to_linear_sequence(np.log(range_prog), reverse_rate_func(zoom_rate_func)))

    def get_center_point(k: float, epsilon: float = 1e-13) -> float:
        if abs(step_ratio - 1.0) < epsilon:
            return old_mean + (new_mean - old_mean) * ((k - 1) / (num_of_frames - 1))
        else:
            d = (new_mean - old_mean) / (1 - step_ratio ** (num_of_frames - 1))
            return old_mean + d * (1 - step_ratio ** (k - 1))

    coord_values = []
    for k, width in enumerate(range_prog, start=1):
        k_adjusted = apply_rate_func_to_scalar_within_bounds(k, zoom_rate_func, (1, num_of_frames))
        center = get_center_point(k_adjusted)
        coord_values.append([center - width / 2, center + width / 2])

    coord_values[0] = old_range
    coord_values[-1] = new_range

    return np.array(coord_values)


def get_repeat_rate_func(rate_func: Callable[[float], float], num_repeats: int) -> Callable[[float], float]:
    def new_rate_func(t):
        segment_index = int(t * num_repeats)
        if segment_index == num_repeats:
            segment_index = num_repeats - 1

        segment_t = (t * num_repeats) - segment_index

        return rate_func(segment_t)

    return new_rate_func


def apply_gradient_along_path(
        obj: VMobject,
        colors: Sequence[ManimColor],
        density=250,
        stroke_width: float = 5,
        overlap: float = 0.1
) -> VMobject:
    result = VMobject(color=ORANGE)
    colors = color_gradient(colors, density + 1)
    for i in range(1, density + 1):
        alpha = min((i + overlap) / density, 1)
        prev_alpha = max((i - 1 - overlap) / density, 0)
        color = colors[i]
        subpath = obj.get_subcurve(prev_alpha, alpha)
        subpath.set_stroke(color, stroke_width, 1)
        result.add(subpath)

    return result


def interpolate_colors(color_list: Sequence[ManimColor], t: float) -> ManimColor:
    if t < 0 or t > 1:
        raise ValueError(f"t {t} must be between 0 and 1")

    if len(color_list) == 1:
        return color_list[0]

    if t == 0:
        return color_list[0]
    if t == 1:
        return color_list[-1]

    num_intervals = len(color_list) - 1
    interval_size = 1 / num_intervals
    index = int(t / interval_size)
    local_t = (t - index * interval_size) / interval_size

    return interpolate_color(color_list[index], color_list[index + 1], local_t)


def tensor_bezier(
        t: torch.Tensor,
        P0: torch.Tensor,
        P1: torch.Tensor,
        P2: torch.Tensor,
        P3: torch.Tensor
) -> torch.Tensor:
    assert torch.min(t) >= 0 and torch.max(t) <= 1, "t values should be within [0, 1]"

    return (1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1 + 3 * (1 - t) * t ** 2 * P2 + t ** 3 * P3


def distance_squared(t, P0, P1, P2, P3, a, b) -> torch.Tensor:
    point = tensor_bezier(t, P0, P1, P2, P3)
    return (point[:, 0] - a)**2 + (point[:, 1] - b)**2


def create_word_keymap(source_text, target_text):
    source_words = source_text.split()
    target_words = target_text.split()

    key_map = {source_words[i]: target_words[i] for i in range(min(len(source_words), len(target_words)))}

    return key_map


def get_unique_filename(base_name='all_solutions', extension='.dat', directory=dp_data_file_dir) -> str:
    counter = 1
    filename = os.path.join(directory, f"{base_name}{extension}")
    while os.path.exists(filename):
        filename = os.path.join(directory, f"{base_name}_{counter}{extension}")
        counter += 1
    return filename


def turn_array_into_mmap(array: np.ndarray, name: str = 'extra') -> np.ndarray:
    filename = get_unique_filename(name)
    array.tofile(filename)

    return np.memmap(filename, dtype=array.dtype, mode='r+', shape=array.shape)


def reverse_rate_func(rate_func: Callable[[float], float]) -> Callable[[float], float]:
    def reversed_func(t: float) -> float:
        return 1 - rate_func(1 - t)

    return reversed_func


def change_pitch(sound, pitch_factor):
    """Change the pitch of an audio file by changing its speed."""
    samples = np.array(sound.get_array_of_samples())
    new_sample_rate = int(sound.frame_rate * pitch_factor)
    sample_width = sound.sample_width
    channels = sound.channels
    samples_as_bytes = samples.tobytes()

    converted_audio, _ = audioop.ratecv(samples_as_bytes, sample_width, channels,
                                      sound.frame_rate, new_sample_rate, None)
    sound_with_new_pitch = sound._spawn(data=converted_audio)
    return sound_with_new_pitch.set_frame_rate(new_sample_rate)


def z_index_inspector(mobs: List[Mobject], indent_level: int = 0, scene_vars: dict = None, max_items: int = 12) -> None:
    if scene_vars is None:
        scene_vars = {id(val): name for name, val in inspect.currentframe().f_back.f_locals.items()}

    if indent_level == 0:
        print("\nScene Mobjects:")
        print("-" * 50)
        format_str = "{:<6} | {:<30} | {:<20}"
        print(format_str.format("Z-Index", "Variable Name", "Type"))
        print("-" * 50)

    format_str = "{:<6} | {:<30} | {:<20}"
    indent = "  " * indent_level

    limited_mobs = mobs[:max_items]
    total_mobs = len(mobs)

    for mob in limited_mobs:
        mob_type = mob.__class__.__name__[:20]
        var_name = scene_vars.get(id(mob), mob.name)[:30]

        print(indent + format_str.format(mob.z_index, var_name, mob_type))

        if isinstance(mob, (Group, VGroup)) and mob.submobjects:
            z_index_inspector(mob.submobjects, indent_level + 1, scene_vars, max_items)

    if total_mobs > max_items:
        print(indent + format_str.format("...", f"(+{total_mobs - max_items} more items)", "..."))

    if indent_level == 0:
        print("\n")


class HermiteSpline:
    """
    A class that handles both single-segment and multi-segment Hermite spline interpolation.
    """
    def __init__(self, points, tangents, times=None):
        """
        Initialize with multiple points and tangents.
        For single segment interpolation, just provide two points and two tangents.
        """
        points = np.array(points)
        assert len(points) >= 2, "Need at least 2 points"

        if tangents is None:
            tangents = np.zeros_like(points)
        else:
            tangents = np.array(tangents)
            assert len(points) == len(tangents), "Must have same number of points and tangents"

        self.points = points
        self.tangents = tangents

        if times is None:
            self.times = np.linspace(0, 1, len(points))
        else:
            assert len(times) == len(points), "Must have same number of points and times"
            self.times = np.array(times)

    @staticmethod
    def _basis_funcs(t):
        """Calculate all Hermite basis functions for given t"""
        h00 = 2 * t ** 3 - 3 * t ** 2 + 1
        h10 = t ** 3 - 2 * t ** 2 + t
        h01 = -2 * t ** 3 + 3 * t ** 2
        h11 = t ** 3 - t ** 2
        return h00, h10, h01, h11

    @staticmethod
    def interpolate_segment(p0, p1, m0, m1, t):
        """Interpolate a single segment between two points"""
        h00, h10, h01, h11 = HermiteSpline._basis_funcs(t)
        return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

    def _find_segment(self, t_val):
        """Find which segment contains t_val"""
        idx = np.searchsorted(self.times, t_val, side='right') - 1
        return np.clip(idx, 0, len(self.times) - 2)

    def _normalize_t(self, t_val, segment):
        """Normalize t_val to [0,1] within its segment"""
        t0, t1 = self.times[segment], self.times[segment + 1]
        return (t_val - t0) / (t1 - t0)

    def interpolate(self, t_val):
        """Interpolate at given t_val(s)"""
        t_val = np.clip(t_val, 0, 1)
        scalar_input = np.isscalar(t_val)
        t_val = np.atleast_1d(t_val)
        result = np.zeros_like(t_val, dtype=float)

        for i, t in enumerate(t_val):
            segment = self._find_segment(t)
            norm_t = self._normalize_t(t, segment)

            p0 = self.points[segment]
            p1 = self.points[segment + 1]
            m0 = self.tangents[segment]
            m1 = self.tangents[segment + 1]

            dt = self.times[segment + 1] - self.times[segment]
            scaled_m0 = m0 * dt
            scaled_m1 = m1 * dt

            result[i] = self.interpolate_segment(p0, p1, scaled_m0, scaled_m1, norm_t)

        return result[0] if scalar_input else result


def create_spline_func(points: list, tangents: list | None = None, times: list | None = None):
    """Helper function to create a spline interpolation function"""
    spline = HermiteSpline(points, tangents, times)

    return spline.interpolate


def create_antiover_func(anticipation: float = 0.5, overshoot: float = 0.5) -> callable:
    """Generate an anticipation-overshoot interpolation function"""
    assert anticipation >= 0 and overshoot >= 0, "anticipation and overshoot must be non-negative"

    return create_spline_func([0, 1], [-anticipation, -overshoot])


def create_overshoot_func(
        first_overshoot_value=0.05,
        time_of_first_overshoot=0.9,
        n_oscillations=7,
):
    """
    Creates a 'bouncy overshoot' spline that:
      - Starts at 0 (time=0),
      - First overshoot at 'time_of_first_overshoot',
      - Then flips above/below 1, halving the overshoot each time,
      - Ends exactly at 1 (time=1),
      - Times converge to 1 by halving the gap after the first overshoot.

    Tangents are computed using a Catmull-Rom scheme with the given tension.
    """
    # --- Points (y-values) ---
    points = [0, 1 + first_overshoot_value]
    error = first_overshoot_value
    sign = 1
    for _ in range(n_oscillations - 1):
        error /= 2
        sign = -sign
        points.append(1 + sign * error)
    points.append(1)  # Ensure final point is exactly 1

    # --- Times (x-values) ---
    times = [0, time_of_first_overshoot]
    last_time = time_of_first_overshoot
    for _ in range(n_oscillations - 1):
        last_time += (1 - last_time) / 2
        times.append(last_time)
    times.append(1)

    # --- Compute Tangents using Catmull-Rom with tension ---
    n = len(points)
    tangents = [0] * n
    # Endpoints: one-sided differences.
    tangents[0] = 0
    tangents[-1] = (points[-1] - points[-2]) / (times[-1] - times[-2])
    # Interior points.
    for i in range(1, n - 1):
        dt = times[i + 1] - times[i - 1]
        if dt != 0:
            if i != 1:
                tangents[i] = 1 * (points[i + 1] - points[i - 1]) / dt
            else:
                tangents[i] = first_overshoot_value / (time_of_first_overshoot - 1)
        else:
            tangents[i] = 0

    return create_spline_func(points=points, tangents=tangents, times=times)


def steep_slow_into(t: float) -> float:
    return np.sqrt(1 - (1 - t) ** 3)


def create_arc_path(start: np.ndarray, end: np.ndarray, arc_angle: float, num_points: int=50) -> VMobject:
    p_fn = path_along_arc(arc_angle)
    points = [p_fn(start, end, alpha) for alpha in np.linspace(0, 1, num_points)]
    path_vm = VMobject(stroke_color=YELLOW)
    path_vm.set_points_smoothly(points)
    return path_vm











