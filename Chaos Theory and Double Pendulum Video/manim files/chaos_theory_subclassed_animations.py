from __future__ import annotations

import warnings

from chaos_theory_mobjects import *
from chaos_theory_base_classes import *
from custom_manim import *


class ReleaseDoublePendulum(Animation):
    def __init__(
            self,
            double_pendulum_master: DoublePendulum,
            angle_visualizer: AngleVisualizer | None = None,
            duration: float | None = None,
            run_time: float | None = None,
            rate_func: Callable[[float], float] = linear,
            increasing_speed: bool = False,
            **kwargs
    ):
        assert double_pendulum_master.double_pendulum.angle_visualizer is angle_visualizer, \
            "angle_visualizer must be the same as double_pendulum_master.double_pendulum.angle_visualizer"
        super().__init__(
            mobject=double_pendulum_master.double_pendulum if angle_visualizer is None else VGroup(
                angle_visualizer, double_pendulum_master.double_pendulum),
            run_time=run_time if run_time else duration,
            **kwargs
        )
        self.angle_visualizer = angle_visualizer
        if increasing_speed:
            assert duration and run_time and duration >= run_time, ("duration and run_time must be provided if "
                                                                    "increasing_speed is True")
        data_factory = double_pendulum_master.get_data_factory(duration, rate_func)
        self.angle_1_values = data_factory.normalized_angle_1_progression_in_degrees
        self.angle_2_values = data_factory.normalized_angle_2_progression_in_degrees

        if len(self.angle_1_values) - 1 != int(fps * duration):
            raise ValueError(f"length of angle_1_values ({len(self.angle_1_values) - 1}) does not match fps ({fps}) * "
                             f"duration ({duration})")
        self.num_of_frames = len(self.angle_1_values) - 1

    def interpolate_mobject(self, alpha: float):
        alpha = min(alpha, 1)
        angle_index = round(alpha * self.num_of_frames)
        if not self.angle_visualizer:
            self.mobject.angle_pair = (
                self.angle_1_values[angle_index],
                self.angle_2_values[angle_index]
            )
        else:
            self.mobject.submobjects[1].angle_pair = (
                self.angle_1_values[angle_index],
                self.angle_2_values[angle_index]
            )


class DrawPlot(Animation):
    def __init__(
            self,
            plot: Plotter,
            run_time: float | None = None,
            # rate_func: Callable[[float], float] = linear,
            **kwargs
    ):
        self.axes = plot.axes
        self.num_of_points = plot.graph.get_num_curves() + len(plot.graph.get_subpaths())
        print(f"self.num_of_points: {self.num_of_points}")

        super().__init__(
            mobject=plot,
            run_time=run_time,
            rate_func=linear,
            introducer=True,
            **kwargs
        )
        # self.plot_rate_func = rate_func

    def interpolate_mobject(self, alpha: float) -> None:
        # alpha = apply_rate_func_to_scalar_within_bounds(alpha, self.plot_rate_func, (0, 1))
        # current_frame = int(alpha * (self.num_of_points - 1)) + 1
        # print(f"current_frame: {current_frame}")
        end_index = round((self.num_of_points - 1) * alpha) + 1
        # print(f"end_index = {end_index}")
        self.mobject.set_indices(0, end_index)


class RescaleAxes(Animation):
    def __init__(
            self,
            axes: DynamicAxes,
            new_x_range: tuple[float, float] | None = None,
            new_y_range: tuple[float, float] | None = None,
            run_time: float = 5,
            rate_func: Callable[[float], float] = linear,
            **kwargs
    ):
        if new_x_range is None and new_y_range is None:
            new_x_range = axes.dyn_x_range[0], axes.dyn_x_range[1]

        super().__init__(
            mobject=axes,
            run_time=run_time,
            rate_func=rate_func,
            **kwargs
        )
        self.axes = axes
        self.new_x_range = new_x_range
        self.new_y_range = new_y_range
        self.num_of_frames = int(run_time * fps)

        if self.new_x_range:
            self.x_values = get_zoom_range_values(
                self.mobject.dyn_x_range,
                self.new_x_range,
                self.num_of_frames,
                self.rate_func
            )
        if self.new_y_range:
            self.y_values = get_zoom_range_values(
                self.mobject.dyn_y_range,
                self.new_y_range,
                self.num_of_frames,
                self.rate_func
            )

    def interpolate_mobject(self, alpha: float) -> None:
        current_frame = int(alpha * (self.num_of_frames - 1)) + 1
        if self.new_x_range:
            self.axes.update_x_range(tuple(self.x_values[current_frame - 1]))
        if self.new_y_range:
            self.axes.update_y_range(tuple(self.y_values[current_frame - 1]))


class RescaleAxesWithPlots(RescaleAxes):
    def __init__(
            self,
            axes: DynamicAxes,
            plots: list[Plotter],
            new_x_range: tuple[float, float] | None = None,
            new_y_range: tuple[float, float] | None = None,
            run_time: float = 5,
            rate_func: Callable[[float], float] = smooth,
            **kwargs
    ):
        super().__init__(
            axes,
            new_x_range,
            new_y_range,
            run_time,
            rate_func,
            **kwargs
        )
        self.plots = plots
        self.mobject = VGroup(self.axes, *self.plots)
        for plot in self.plots:
            assert plot.axes is self.axes, "All plots' axes instances must be the axes passed to this class"

    def interpolate_mobject(self, alpha: float) -> None:
        current_frame = int(alpha * (self.num_of_frames - 1)) + 1
        if self.new_x_range:
            self.mobject[0].update_x_range(tuple(self.x_values[current_frame - 1]))
        if self.new_y_range:
            self.mobject[0].update_y_range(tuple(self.y_values[current_frame - 1]))

        for plot in self.mobject[1:]:
            plot.update_axes(self.mobject[0])


class ReleaseGhosts(ReleaseDoublePendulum):
    def __init__(
            self,
            ghosts: DoublePendulumGhosts,
            angle_visualizer: AngleVisualizer,
            **kwargs
    ):
        assert ghosts.main_dp.double_pendulum.angle_visualizer is angle_visualizer, \
            "Ghosts must have the same angle visualizer"
        super().__init__(
            ghosts.main_dp,
            angle_visualizer,
            ghosts.duration,
            **kwargs
        )
        self.ghosts = ghosts
        self.mobject = VGroup()

    def interpolate_mobject(self, alpha: float):
        current_frame = int(alpha * (self.num_of_frames * (self.ghosts.override_fps / fps) - 1)) + 1
        self.mobject.remove(*self.mobject.submobjects)
        self.mobject.add(*self.ghosts.first_dp_ghosts[0:current_frame])
        self.mobject.add(*self.ghosts.second_dp_ghosts[0:current_frame])


class ManualDoublePendulumAnimation(Animation):
    def __init__(
            self,
            double_pendulum: DoublePendulumConstructor,
            new_angle_pair: tuple[float, float],
            run_time: float = 1,
            rate_func: Callable[[float], float] = rate_functions.smootherstep,
            **kwargs
    ):
        super().__init__(
            mobject=double_pendulum if not double_pendulum.angle_visualizer
            else VGroup(double_pendulum.angle_visualizer, double_pendulum),
            run_time=run_time,
            **kwargs
        )

        self.double_pendulum = double_pendulum
        self.old_angle_pair = double_pendulum._angle_pair
        self.num_of_frames = int(run_time * fps)

        angle_1_values = np.linspace(self.old_angle_pair[0], new_angle_pair[0], self.num_of_frames)
        angle_2_values = np.linspace(self.old_angle_pair[1], new_angle_pair[1], self.num_of_frames)
        angle_1_values = apply_rate_func_to_linear_sequence(angle_1_values, rate_func)
        angle_2_values = apply_rate_func_to_linear_sequence(angle_2_values, rate_func)
        self.angle_values_progression = list(zip(angle_1_values, angle_2_values))

    def interpolate_mobject(self, alpha: float):
        alpha = min(alpha, 1)
        current_frame = int(alpha * (self.num_of_frames - 1)) + 1
        if not self.mobject[1].angle_visualizer:
            self.mobject.angle_pair = self.angle_values_progression[current_frame - 1]
        else:
            self.mobject[1].angle_pair = self.angle_values_progression[current_frame - 1]


class AnimateGhostsWithPlot(Animation):
    def __init__(
            self,
            ghosts: DoublePendulumGhosts,
            plot: Plotter,
            new_angle_pair: tuple[float, float],
            run_time: float = 1,
            scene: Scene | None = None,
            rate_func: Callable[[float], float] = smoothstep,
            use_rate_func_of_plot: bool = False,
            use_spiral_angle_progression: bool = False,
            **kwargs
    ):
        super().__init__(
            mobject=VGroup(
                ghosts.main_dp.double_pendulum.angle_visualizer,
                ghosts.main_dp.double_pendulum,
                ghosts,
                plot
            ),
            run_time=run_time,
            **kwargs
        )
        self.ghosts = ghosts
        self.plot = plot
        self.new_angle_pair = new_angle_pair
        self.scene = scene
        self.main_dp = ghosts.main_dp
        self.duration = ghosts.duration
        self.old_angle_pair = self.ghosts.main_dp.double_pendulum.angle_pair
        self.num_of_frames = int(run_time * fps)

        if not use_spiral_angle_progression:
            angle_1_values = np.linspace(self.old_angle_pair[0], self.new_angle_pair[0], self.num_of_frames)
            angle_2_values = np.linspace(self.old_angle_pair[1], self.new_angle_pair[1], self.num_of_frames)
            angle_1_values = apply_rate_func_to_linear_sequence(angle_1_values, rate_func)
            angle_2_values = apply_rate_func_to_linear_sequence(angle_2_values, rate_func)
            self.angle_values_progression = list(zip(angle_1_values, angle_2_values))
        else:
            self.angle_values_progression = self.get_spiral_progression(rate_func)

        self.list_of_main_dps = []
        self.list_of_angle_visualizers = []
        self.list_of_ghosts = []
        self.list_of_plots = []

        total_iterations = len(self.angle_values_progression)

        for i, angle_pair in tqdm(enumerate(self.angle_values_progression), total=total_iterations,
                                  desc="Generating DP, AV, Ghosts, and Plot"):
            main_dp = DoublePendulum(angle_pair)
            dp = main_dp.create_double_pendulum(
                self.main_dp.double_pendulum.length_1,
                self.main_dp.double_pendulum.length_2
            )
            move_relative_to(dp, dp.rod1.get_start(), self.main_dp.double_pendulum.rod1.get_start())
            av = dp.create_angle_visualizer()
            ghosts = DoublePendulumGhosts(
                main_dp,
                ghosts.override_fps,
                self.duration,
                self.ghosts.max_opacity,
                self.ghosts.opacity_rate_func
            )
            dp_duration = len(self.plot.passed_points) / fps
            points = main_dp.get_data_factory(
                dp_duration,
                self.main_dp.rate_func if use_rate_func_of_plot else linear).get_points_for_plotting_two_angles()
            plot = Plotter(
                points,
                self.plot.axes,
                self.plot.use_smoothing,
                self.plot.stroke_wid,
                self.plot.col,
                self.plot.stroke_opa,
            )

            self.list_of_main_dps.append(main_dp)
            self.list_of_angle_visualizers.append(av)
            self.list_of_ghosts.append(ghosts)
            self.list_of_plots.append(plot)

    def get_spiral_progression(self, rate_func: Callable) -> list[tuple]:
        start_point = np.array(self.old_angle_pair)
        end_point = np.array(self.new_angle_pair)
        radius_vector = start_point - end_point
        initial_radius = np.linalg.norm(radius_vector)

        num_points = self.num_of_frames
        num_revolutions = 1.6
        t_values = np.linspace(0, 0.95, num_points)
        t_values = apply_rate_func_to_linear_sequence(t_values, rate_func)

        angle_pairs = []

        # Generate spiral points
        for t in t_values:  # Exclude first and last points
            # Radius that decreases with time
            current_radius = initial_radius * (1 - t)

            # Angle that increases with time
            angle = 2 * PI * num_revolutions * t

            # Calculate rotation matrix
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])

            # Rotate and scale the radius vector
            normalized_radius_vector = radius_vector / np.linalg.norm(radius_vector)

            # Calculate point relative to end_point (center)
            point = end_point + current_radius * (rotation_matrix @ normalized_radius_vector)

            angle_pairs.append(tuple(point))

        return angle_pairs

    def interpolate_mobject(self, alpha: float):
        current_frame = int(alpha * (self.num_of_frames - 1))
        dp = self.list_of_main_dps[current_frame].double_pendulum
        av = self.list_of_angle_visualizers[current_frame]
        ghosts = self.list_of_ghosts[current_frame]
        plot = self.list_of_plots[current_frame]

        new_mobject = VGroup(av, dp, ghosts, plot)

        self.mobject[1].angle_pair = dp._angle_pair
        self.mobject.become(new_mobject)


class SweepToCreateDoublePendulums(Animation):
    def __init__(
            self,
            scene: Scene,
            dp_sweeper: UnionDoublePendulumConstructor,
            dp_sweeper_av: AngleVisualizer,
            table_of_double_pendulums: TableOfDoublePendulums,
            duration_for_each_transform: float = 1.5,
            apply_finish: bool = True,
            **kwargs
    ):
        run_time = len(table_of_double_pendulums.submobjects) / fps
        print(f"\n\n----------> run time of SweepToCreateDoublePendulums is {run_time:.2f} s")
        super().__init__(
            VGroup(dp_sweeper_av, dp_sweeper),
            run_time=len(table_of_double_pendulums.submobjects) / fps,
            rate_func=linear,
            **kwargs
        )
        self.scene = scene
        self.table_of_double_pendulums = table_of_double_pendulums
        self.duration_for_each_transform = duration_for_each_transform
        self.apply_finish = apply_finish
        self.num_of_frames = len(self.table_of_double_pendulums.submobjects)
        self.copies = []

    def get_copies_of_sweeper(self):
        return self.copies

    def interpolate_mobject(self, alpha: float):
        current_frame = int(alpha * (self.num_of_frames - 1)) + 1
        self.mobject.submobjects[1].angle_pair = self.table_of_double_pendulums.submobjects[current_frame - 1]._angle_pair
        copy_of_current_sweeper = VGroup(*self.mobject.submobjects[1][4:6]).copy().set_opacity(0.025)
        self.scene.add(copy_of_current_sweeper)
        self.copies.append(copy_of_current_sweeper)
        turn_animation_into_updater(Transform(
            copy_of_current_sweeper,
            self.table_of_double_pendulums.submobjects[current_frame - 1][4:6],
            run_time=self.duration_for_each_transform,
            rate_func=linear
        ))

    def finish(self) -> None:
        self.interpolate(1)
        if self.suspend_mobject_updating and self.mobject is not None:
            self.mobject.resume_updating()
        if self.apply_finish:
            self.mobject.submobjects[1].angle_pair = (180, 180)


class ReleaseTableOfDoublePendulums(Animation):
    def __init__(
            self,
            table_of_double_pendulums: TableOfDoublePendulums,
            duration: float,
            rate_func: Callable[[float], float] = linear,
            **kwargs
    ):
        super().__init__(
            mobject=table_of_double_pendulums,
            run_time=duration,
            **kwargs
        )
        self.duration = duration

        self.computation = OptimizedDoublePendulumComputation(
            table_of_double_pendulums.angle_pairs,
            duration,
            rate_func
        )
        self.angle_1_prog_list = self.computation.normalized_angle_1_progression_in_degrees
        self.angle_2_prog_list = self.computation.normalized_angle_2_progression_in_degrees
        self.num_of_frames = self.angle_1_prog_list.shape[1]

    def interpolate_mobject(self, alpha: float):
        current_frame = int(alpha * (self.num_of_frames - 1)) + 1
        angle_1_values = self.angle_1_prog_list[:, current_frame - 1]
        angle_2_values = self.angle_2_prog_list[:, current_frame - 1]

        for dp, angle1, angle2 in zip(self.mobject.submobjects, angle_1_values, angle_2_values):
            dp.angle_pair = (angle1, angle2)


class ReleaseTableWithManualAnimations(ReleaseTableOfDoublePendulums):
    def __init__(
            self,
            table_of_double_pendulums: TableOfDoublePendulums,
            unstable_dps: VGroup,
            stable_dps: VGroup,
            duration: float,  # must be > 10.5
            rate_func: Callable[[float], float] = linear,
            **kwargs
    ):
        super().__init__(
            table_of_double_pendulums,
            duration,
            rate_func,
            **kwargs
        )
        self.unstable_dps = unstable_dps
        self.stable_dps = stable_dps

        # Save initial states
        self.initial_unstable_opacity = 1.0
        self.initial_stable_opacity = 1.0

        # Set initial opacity to 1
        self.unstable_dps.set_opacity(1)
        self.stable_dps.set_opacity(1)

        self.start = 7.5
        self.fade_out_unstable_duration = 0.75  # Duration for fading out unstable pendulums in seconds
        self.linger_unstable_low_duration = 3  # Duration for lingering at low opacity in seconds
        self.restore_unstable_duration = 0.75  # Duration for restoring unstable pendulums in seconds
        self.linger_after_unstable_duration = 0.5  # Duration to linger after restoring unstable pendulums
        self.fade_out_stable_duration = 0.75  # Duration for fading out stable pendulums in seconds
        self.linger_stable_low_duration = 3  # Duration for lingering at low opacity in seconds
        self.restore_stable_duration = 0.75  # Duration for restoring stable pendulums in seconds
        self.low_opacity = 0.1

        # Assertions to ensure the parameters are valid
        assert 0 <= self.start < self.duration, "Start time must be within the duration of the animation."
        total_phase_time = (self.fade_out_unstable_duration + self.linger_unstable_low_duration +
                            self.restore_unstable_duration + self.linger_after_unstable_duration +
                            self.fade_out_stable_duration + self.linger_stable_low_duration +
                            self.restore_stable_duration)
        assert total_phase_time <= self.duration, "The sum of all phase durations must not exceed the total duration."

    def interpolate_mobject(self, alpha: float):
        super().interpolate_mobject(alpha)
        total_duration = self.duration
        current_time = alpha * total_duration

        # Calculate phase start times
        fade_out_unstable_start = self.start
        linger_unstable_low_start = fade_out_unstable_start + self.fade_out_unstable_duration
        restore_unstable_start = linger_unstable_low_start + self.linger_unstable_low_duration
        linger_after_unstable_start = restore_unstable_start + self.restore_unstable_duration
        fade_out_stable_start = linger_after_unstable_start + self.linger_after_unstable_duration
        linger_stable_low_start = fade_out_stable_start + self.fade_out_stable_duration
        restore_stable_start = linger_stable_low_start + self.linger_stable_low_duration
        end_animation_start = restore_stable_start + self.restore_stable_duration

        # Animate unstable_dps to set opacity
        if fade_out_unstable_start <= current_time < linger_unstable_low_start:
            progress = (current_time - fade_out_unstable_start) / self.fade_out_unstable_duration
            self.unstable_dps.set_opacity(1 - (1 - self.low_opacity) * progress)  # Decreasing opacity from 1 to low_opacity

        # Linger at low opacity
        elif linger_unstable_low_start <= current_time < restore_unstable_start:
            self.unstable_dps.set_opacity(self.low_opacity)  # Set opacity to low_opacity

        # Gradually restore unstable_dps opacity
        elif restore_unstable_start <= current_time < linger_after_unstable_start:
            progress = (current_time - restore_unstable_start) / self.restore_unstable_duration
            self.unstable_dps.set_opacity(self.low_opacity + (1 - self.low_opacity) * progress)  # Increasing opacity from low_opacity to 1

        # Linger after restoring unstable
        elif linger_after_unstable_start <= current_time < fade_out_stable_start:
            self.unstable_dps.set_opacity(1)  # Set opacity to 1

        # Animate stable_dps to set opacity
        if fade_out_stable_start <= current_time < linger_stable_low_start:
            progress = (current_time - fade_out_stable_start) / self.fade_out_stable_duration
            self.stable_dps.set_opacity(1 - (1 - self.low_opacity) * progress)  # Decreasing opacity from 1 to low_opacity

        # Linger at low opacity
        elif linger_stable_low_start <= current_time < restore_stable_start:
            self.stable_dps.set_opacity(self.low_opacity)  # Set opacity to low_opacity

        # Gradually restore stable_dps opacity
        elif restore_stable_start <= current_time < end_animation_start:
            progress = (current_time - restore_stable_start) / self.restore_stable_duration
            self.stable_dps.set_opacity(self.low_opacity + (1 - self.low_opacity) * progress)  # Increasing opacity from low_opacity to 1


class TrackDoublePendulumWithTable(ReleaseDoublePendulum):
    def __init__(
            self,
            double_pendulum_master: DoublePendulum,
            table: TableOfDoublePendulums,
            angle_visualizer: AngleVisualizer,
            duration: float | None = None,
            run_time: float | None = None,
            rate_func: Callable[[float], float] = linear,
            increasing_speed: bool = False,
            **kwargs
    ):
        super().__init__(
            double_pendulum_master,
            angle_visualizer,
            duration,
            run_time,
            rate_func,
            increasing_speed,
            **kwargs
        )
        self.table = table
        self.mobject = VGroup(double_pendulum_master.double_pendulum, angle_visualizer, table)

    def interpolate_mobject(self, alpha: float):
        current_frame = int(alpha * (self.num_of_frames - 1)) + 1
        new_angle_pair = (
            self.angle_1_values[current_frame - 1],
            self.angle_2_values[current_frame - 1]
        )
        self.mobject[0].angle_pair = new_angle_pair
        if self.angle_visualizer:
            self.mobject[1].update_angle_visualizer()

        self.mobject[2].angle_pair_to_highlight = new_angle_pair


class TurnTableIntoBlocks(AnimationGroup):
    def __init__(
            self,
            table: TableOfDoublePendulums,
            blocky_pixel: BlockyPixels,
            duration: float,
            lag_ratio: float,
            **kwargs
    ):
        animations = []
        for dp, block in zip(table.submobjects, blocky_pixel.submobjects):
            animation = FadeReplacementTransform(dp, block, rate_func=rush_into)
            animations.append(animation)

        super().__init__(
            *animations,
            run_time=duration,
            lag_ratio=lag_ratio,
            **kwargs
        )

    def finish(self) -> None:
        super().finish()


class PixelVisualizationAnimation(ComplexAnimation):
    def __init__(
            self,
            pixel_static_visuals: PixelStaticVisuals,
            duration: float = 10,
            use_existing_dat: str | None = None,
            skip_processing: bool = False,
            **kwargs
    ):
        super().__init__(
            pixel_static_visuals,
            run_time=duration,
            **kwargs
        )
        self.pixel_static_visuals = pixel_static_visuals
        self.cs = self.pixel_static_visuals.cs
        self.color_func = pixel_static_visuals.color_func
        self.data_computation = pixel_static_visuals.data_computation

        self.int_rgba = self.data_computation.get_direct_int_rgba_data(self.color_func, duration, use_existing_dat, skip_processing)

        del self.data_computation
        torch.cuda.empty_cache()

        self.num_of_frames = int(fps * duration)

        assert self.num_of_frames == self.int_rgba.shape[0], "something is wrong"

    def interpolate_mobject(self, alpha: float):
        current_frame = int(alpha * (self.num_of_frames - 1)) + 1
        new_image = ImageMobject(
            self.int_rgba[current_frame - 1].copy(),
            scale_to_resolution=self.pixel_static_visuals.height_pixel_num * config.frame_height / self.cs.y_length
        )
        self.mobject.become(
            new_image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"]).shift(self.cs.location)
        )


class TrackDoublePendulumWithPixelVisuals(ReleaseDoublePendulum):
    def __init__(
            self,
            double_pendulum_master: DoublePendulum,
            pixel_visuals: PixelStaticVisuals,
            angle_visualizer: AngleVisualizer | None,
            tracker: Square,
            static_tracker_copy: Square,
            duration: float | None = None,
            run_time: float | None = None,
            rate_func: Callable[[float], float] = linear,
            increasing_speed: bool = False,
            **kwargs
    ):
        super().__init__(
            double_pendulum_master,
            angle_visualizer,
            duration,
            run_time,
            rate_func,
            increasing_speed,
            **kwargs
        )
        self.pixel_visuals = pixel_visuals
        self.color_func = pixel_visuals.color_func
        self.cs = pixel_visuals.cs
        self.tracker = tracker

        if self.angle_visualizer:
            self.mobject = VGroup(
                double_pendulum_master.double_pendulum,
                angle_visualizer,
                self.tracker,
                static_tracker_copy
            )
        else:
            self.mobject = VGroup(
                double_pendulum_master.double_pendulum,
                self.tracker,
                static_tracker_copy
            )

    def interpolate_mobject(self, alpha: float):
        current_frame = int(alpha * (self.num_of_frames - 1)) + 1
        new_angle_pair = (
            self.angle_1_values[current_frame - 1],
            self.angle_2_values[current_frame - 1]
        )
        new_color = rgba_to_color(self.color_func(torch.from_numpy(np.array([new_angle_pair])))[0].tolist())

        self.mobject[0].angle_pair = new_angle_pair
        if self.angle_visualizer:
            self.mobject[1].update_angle_visualizer()
            self.mobject[2].set_fill(new_color).move_to(
                self.cs.coords_to_point(new_angle_pair)
            )
            self.mobject[3].set_fill(new_color)
        else:
            self.mobject[1].set_fill(new_color).move_to(
                self.cs.coords_to_point(new_angle_pair)
            )
            self.mobject[2].set_fill(new_color)


class TrackWithPixelVisualsForTable(TrackDoublePendulumWithPixelVisuals):
    def __init__(
            self,
            double_pendulum_master: DoublePendulum,
            pixel_visuals: PixelStaticVisuals,
            angle_visualizer: AngleVisualizer | None,
            tracker: Square,
            static_tracker_copy: Square,
            duration: float | None = None,
            run_time: float | None = None,
            rate_func: Callable[[float], float] = linear,
            increasing_speed: bool = False,
            **kwargs
    ):
        super().__init__(
            double_pendulum_master,
            pixel_visuals,
            angle_visualizer,
            tracker,
            static_tracker_copy,
            duration,
            run_time,
            rate_func,
            increasing_speed,
            **kwargs
        )
        self.mobject = VGroup(self.tracker, static_tracker_copy)

    def interpolate_mobject(self, alpha: float):
        current_frame = int(alpha * (self.num_of_frames - 1)) + 1
        new_angle_pair = (
            self.angle_1_values[current_frame - 1],
            self.angle_2_values[current_frame - 1]
        )
        new_color = rgba_to_color(self.color_func(torch.tensor([new_angle_pair]))[0].tolist())

        self.mobject[0].set_fill(new_color).move_to(
            self.cs.coords_to_point(new_angle_pair)
        )
        self.mobject[1].set_fill(new_color)


class TrackTableOfDoublePendulumsWithPixelVisuals(AnimationGroup):
    def __init__(
            self,
            table: TableOfDoublePendulums,
            pixel_visual: PixelStaticVisuals,
            tracker_group: VGroup,
            static_tracker_copy_group: VGroup,
            duration: float,
            **kwargs
    ):
        assert len(table.dp_masters) == len(tracker_group.submobjects) == len(static_tracker_copy_group.submobjects), \
            "table, tracker_group, and static_tracker_copy_group must have the same number of submobjects"

        track_anims = []
        for i, (dp_master, tracker, static_tracker_copy) in enumerate(tqdm(
                zip(table.dp_masters, tracker_group.submobjects, static_tracker_copy_group.submobjects),
                total=len(table.dp_masters),
                desc="Processing scene 6.4 animation"
        )):
            track_anims.append(TrackWithPixelVisualsForTable(
                dp_master,
                pixel_visual,
                None,
                tracker,
                static_tracker_copy,
                duration
            ))

        super().__init__(
            *track_anims,
            run_time=duration,
            **kwargs
        )


class ZoomPixelVisuals(RescaleAxes, ComplexAnimation):
    def __init__(
            self,
            pixel_visuals: PixelStaticVisuals,
            new_x_range: tuple[float, float] | None = None,
            new_y_range: tuple[float, float] | None = None,
            run_time: float = 5,
            zoom_rate_func: Callable[[float], float] = smooth,
            **kwargs
    ):
        super().__init__(
            pixel_visuals.cs,
            new_x_range,
            new_y_range,
            run_time,
            zoom_rate_func,
            **kwargs
        )
        self.mobject = pixel_visuals
        self.zoom_animation_data = pixel_visuals.data_computation.get_zoom_anim_data_pixel_visuals(
            (pixel_visuals.cs.dyn_x_range[0], pixel_visuals.cs.dyn_x_range[1]),
            new_x_range,
            (pixel_visuals.cs.dyn_y_range[0], pixel_visuals.cs.dyn_y_range[1]),
            new_y_range,
            run_time,
            zoom_rate_func,
            pixel_visuals.color_func
        )

    def interpolate_mobject(self, alpha: float) -> None:
        current_frame = int(alpha * (self.num_of_frames - 1)) + 1
        if self.new_x_range:
            self.axes.update_x_range(tuple(self.x_values[current_frame - 1]))
        if self.new_y_range:
            self.axes.update_y_range(tuple(self.y_values[current_frame - 1]))

        self.mobject.pixel_array = self.zoom_animation_data[current_frame - 1]

    def clean_up_from_scene(self, scene: Scene) -> None:
        super().clean_up_from_scene(scene)
        self.mobject.angle_1_domain = (self.axes.dyn_x_range[0], self.axes.dyn_x_range[1])
        self.mobject.angle_2_domain = (self.axes.dyn_y_range[0], self.axes.dyn_y_range[1])
        self.mobject.data_computation = PixelGridAnglesComputation(
            self.mobject.angle_1_domain,
            self.mobject.angle_2_domain,
            self.mobject.width_pixel_num,
            self.mobject.height_pixel_num
        )
        self.mobject.data_computation.color_func = self.mobject.color_func
        delete_memmap_files("all_solutions")


class IndicateFlip(ReleaseDoublePendulum):
    def __init__(
            self,
            double_pendulum_master: DoublePendulum,
            scene: Scene,
            angle_visualizer: AngleVisualizer | None = None,
            duration: float | None = None,
            run_time: float | None = None,
            rate_func: Callable[[float], float] = linear,
            increasing_speed: bool = False,
            **kwargs
    ):
        super().__init__(
            double_pendulum_master,
            angle_visualizer,
            duration,
            run_time,
            rate_func,
            increasing_speed,
            **kwargs
        )
        self.main_dp = double_pendulum_master
        self.scene = scene
        flip_compute = ComputeFlipsForSingleDP(self.main_dp, run_time if run_time else duration)
        self.angle_1_flip_indices, self.angle_2_flip_indices = flip_compute.get_flip_indices_for_single_dp()
        self.texts = []
        self.flip1 = ImageMobject("flip1.png", scale_to_resolution=1080 * 6)
        self.flip2 = ImageMobject("flip2.png", scale_to_resolution=1080 * 6)

    def add_flip_indicator(self, frame: int, angle_flip_indices: np.ndarray, bob, flip_image):
        if frame - 2 in angle_flip_indices:
            flip = flip_image.copy()
            self.scene.add(flip)
            turn_animation_into_updater(FadeOut(flip, scale=0.1, run_time=1, rate_func=rush_into))
            flip.add_updater(lambda obj: obj.shift(bob.get_corner(UP) - (obj.get_center() + DOWN * 0.3)))
            self.texts.append(flip)

    def interpolate_mobject(self, alpha: float):
        current_frame = int(alpha * (self.num_of_frames - 1)) + 1
        self.mobject[1].angle_pair = (
            self.angle_1_values[current_frame - 1],
            self.angle_2_values[current_frame - 1]
        )
        if self.angle_visualizer:
            self.mobject[0].update_angle_visualizer()

        self.add_flip_indicator(current_frame, self.angle_1_flip_indices, self.main_dp.double_pendulum.bob1, self.flip1)
        self.add_flip_indicator(current_frame, self.angle_2_flip_indices, self.main_dp.double_pendulum.bob2, self.flip2)

    def clean_up_from_scene(self, scene: Scene) -> None:
        super().clean_up_from_scene(scene)
        for text in self.texts:
            text.clear_updaters()


class TransformRateFuncOfColorTracker(Animation):
    def __init__(
            self,
            color_tracker: ColorTracker,
            new_color_rate_func: Callable[[float], float],
            duration: float = 2,
            rate_func: Callable[[float], float] = smooth,
            **kwargs
    ):
        self.old_color_rate_func = color_tracker.color_rate_func
        self.new_color_rate_func = new_color_rate_func

        super().__init__(
            mobject=color_tracker,
            run_time=duration,
            rate_func=rate_func,
            **kwargs
        )

    def get_interpolated_rate_func(self, a) -> Callable[[float], float]:
        def interpolated_rate_func(t):
            return self.old_color_rate_func(t) * (1 - a) + self.new_color_rate_func(t) * a

        return interpolated_rate_func

    def interpolate_mobject(self, alpha: float) -> None:
        alpha = self.rate_func(alpha)
        self.mobject.color_rate_func = self.get_interpolated_rate_func(alpha)


class RunColorTracker(Animation):
    def __init__(
            self,
            color_tracker: ColorTracker,
            duration: float = 10,
            set_alpha_prog_to: float = 1,
            rate_func: Callable[[float], float] = linear,
            **kwargs
    ):
        assert 0 <= set_alpha_prog_to <= 1, "set_alpha_prog_to must be in [0, 1]"
        super().__init__(
            color_tracker,
            run_time=duration,
            rate_func=rate_func,
            **kwargs
        )
        self.init_alpha_prog = color_tracker.alpha_progression
        self.set_alpha_prog_to = set_alpha_prog_to

    def interpolate_mobject(self, alpha: float) -> None:
        alpha_span = self.rate_func(alpha) * (self.set_alpha_prog_to - self.init_alpha_prog)
        self.mobject.alpha_progression = alpha_span + self.init_alpha_prog


class FlipPixelVisuals(ComplexAnimation):
    def __init__(
            self,
            flip_visual: FlipStaticVisuals,
            color_tracker: ColorTracker,
            duration: float = 14,
            **flip_visuals_kwargs
    ):
        assert color_tracker.elapsed_time == duration, "must match duration and elapsed_time"

        super().__init__(
            flip_visual,
            run_time=duration,
        )
        self.num_of_frames = int(duration * fps)
        self.num_of_dps = flip_visual.height_pixel_num * flip_visual.width_pixel_num

        self.flip_visual = flip_visual
        self.color_tracker = color_tracker
        self.scale_to_res = flip_visual.height_pixel_num * config.frame_height / flip_visual.cs.y_length
        self.data_computation = self.flip_visual.pixel_computation

        self.animation_data = self.data_computation.get_flips_animation_data(
            color_tracker,
            duration,
            **flip_visuals_kwargs
        )
        delete_memmap_files("white_default")

    def interpolate_mobject(self, alpha: float):
        current_frame = int(alpha * (self.num_of_frames - 1)) + 1
        self.mobject.pixel_array = self.animation_data[current_frame - 1].copy()

    def clean_up_from_scene(self, scene: Scene) -> None:
        super().clean_up_from_scene(scene)
        del self.animation_data
        delete_memmap_files("reshape_flips")


class FlipBlockyPixels(Animation):
    def __init__(
            self,
            blocky_pixels: BlockyPixels,
            color_tracker: ColorTracker,
            duration: float = 14,
            **flip_visuals_kwargs
    ):
        assert color_tracker.elapsed_time == duration, "must match duration and elapsed_time"

        super().__init__(
            blocky_pixels,
            run_time=duration,
        )
        self.num_of_frames = int(duration * fps)
        self.num_of_dps = blocky_pixels.row_count * blocky_pixels.column_count

        self.blocky_pixels = blocky_pixels
        self.color_tracker = color_tracker
        self.scale_to_res = blocky_pixels.row_count * config.frame_height / blocky_pixels.cs.y_length
        self.data_computation = PixelGridAnglesComputation(
            blocky_pixels.cs.dyn_x_range,
            blocky_pixels.cs.dyn_y_range,
            blocky_pixels.row_count,
            blocky_pixels.column_count
        )

        self.animation_data = self.data_computation.get_flips_animation_data(
            color_tracker,
            duration,
            **flip_visuals_kwargs
        )[:, ::-1, :, :]
        d1, d2, d3, d4 = self.animation_data.shape
        self.animation_data = self.animation_data.transpose(0, 2, 1, 3).reshape(d1, d2 * d3, d4)

    def interpolate_mobject(self, alpha: float):
        current_frame = int(alpha * (self.num_of_frames - 1)) + 1
        for rect, col in zip(self.mobject.submobjects, self.animation_data[current_frame - 1]):
            rect.set_fill(rgba_to_color(col))

    def clean_up_from_scene(self, scene: Scene) -> None:
        super().clean_up_from_scene(scene)
        del self.animation_data
        delete_memmap_files("white_default")


class ReleaseTableOfDoublePendulumsUntilFlip(ReleaseTableOfDoublePendulums):
    def __init__(
            self,
            table_of_double_pendulums: TableOfDoublePendulums,
            duration: float,
            **kwargs
    ):
        super().__init__(
            table_of_double_pendulums,
            duration,
            **kwargs
        )

        angle_1_domain: tuple[float, float] = table_of_double_pendulums.cs.dyn_x_range
        angle_2_domain: tuple[float, float] = table_of_double_pendulums.cs.dyn_y_range
        pixel_computation = PixelGridAnglesComputation(
            angle_1_domain,
            angle_2_domain,
            table_of_double_pendulums.row_count,
            table_of_double_pendulums.column_count,
        )
        self.flip_index_data = pixel_computation.get_flips_index_data(duration).reshape(
            table_of_double_pendulums.row_count,
            table_of_double_pendulums.column_count)
        self.flip_index_data = np.rot90(self.flip_index_data, k=-1).ravel()
        for i, flip_index in enumerate(self.flip_index_data.tolist()):
            if flip_index == -1:
                continue
            if abs(self.angle_2_prog_list[i][flip_index] - self.angle_2_prog_list[i][flip_index - 1]) > 180:
                self.angle_2_prog_list[i][flip_index:] = 180
                self.angle_1_prog_list[i][flip_index:] = self.angle_1_prog_list[i][flip_index]
            elif abs(self.angle_1_prog_list[i][flip_index] - self.angle_1_prog_list[i][flip_index - 1]) > 180:
                self.angle_1_prog_list[i][flip_index:] = 180
                self.angle_2_prog_list[i][flip_index:] = self.angle_2_prog_list[i][flip_index]
            else:
                warnings.warn("Unexpected case in ReleaseTableOfDoublePendulumsUntilFlip")

    def interpolate_mobject(self, alpha: float):
        current_frame = int(alpha * (self.num_of_frames - 1)) + 1
        angle_1_values = self.angle_1_prog_list[:, current_frame - 1]
        angle_2_values = self.angle_2_prog_list[:, current_frame - 1]

        for i, (dp, angle1, angle2, flip_index) in enumerate(zip(
                self.mobject.submobjects,
                angle_1_values,
                angle_2_values,
                self.flip_index_data
        )):
            dp.angle_pair = (angle1, angle2)
            # if current_frame == flip_index:
            #     dp.set_fill_color(BLACK)


class FlipVisualization(AnimationGroup):
    def __init__(
            self,
            flip_visual: FlipStaticVisuals | BlockyPixels,
            color_tracker: ColorTracker,
            table: TableOfDoublePendulums | None = None,
            duration: float = 13,
            **flip_visuals_kwargs
    ):
        assert color_tracker.elapsed_time == duration, "must match duration and elapsed_time"

        animations = []
        if table:
            row_check = table.row_count == (flip_visual.height_pixel_num if isinstance(flip_visual, FlipStaticVisuals)
                                            else flip_visual.row_count)
            col_check = table.column_count == (flip_visual.width_pixel_num if isinstance(flip_visual, FlipStaticVisuals)
                                               else flip_visual.column_count)

            assert row_check, "must match row and height"
            assert col_check, "must match column and width"

            table_anim = ReleaseTableOfDoublePendulumsUntilFlip(table, duration)
            animations.append(table_anim)

        tracker_anim = RunColorTracker(color_tracker, duration)
        animations.append(tracker_anim)

        if isinstance(flip_visual, FlipStaticVisuals):
            flip_visual_anim = FlipPixelVisuals(flip_visual, color_tracker, duration, **flip_visuals_kwargs)
            self.flip_anim = flip_visual_anim
            animations.append(self.flip_anim)
        elif isinstance(flip_visual, BlockyPixels):
            flip_visual_anim = FlipBlockyPixels(flip_visual, color_tracker, duration, **flip_visuals_kwargs)
            self.flip_anim = None
            animations.append(flip_visual_anim)
        else:
            raise ValueError("flip_visual must be FlipStaticVisuals or BlockyPixels")

        super().__init__(*animations)

    def clean_up_from_scene(self, scene: Scene) -> None:
        super().clean_up_from_scene(scene)
        del self.flip_anim
        delete_memmap_files("white_default")


class ZoomFlipVisuals(RescaleAxes):
    def __init__(
            self,
            flip_visual: FlipStaticVisuals,
            new_x_range: tuple[float, float],
            new_y_range: tuple[float, float],
            duration: float = 5,
            zoom_rate_func: Callable[[float], float] = smootherstep,
            use_existing_dat: str | None = None,
            skip_processing: bool = False,
            **kwargs
    ):
        assert flip_visual.color_tracker, "flip visual must be set to its last image and must contain a color tracker"

        self.axes = flip_visual.cs
        self.color_tracker = flip_visual.color_tracker
        super().__init__(
            self.axes,
            new_x_range,
            new_y_range,
            duration,
            zoom_rate_func,
            **kwargs
        )
        self.mobject = flip_visual
        self.new_x_range = new_x_range
        self.new_y_range = new_y_range

        self.zoom_animation_data = self.mobject.pixel_computation.get_zooming_flips_animation_data(
            self.mobject,
            self.color_tracker,
            self.new_x_range,
            self.new_y_range,
            self.run_time,
            zoom_rate_func,
            use_existing_dat,
            skip_processing
        )
        delete_memmap_files('white_default', to_print=True)
        delete_memmap_files('flip_index_data', to_print=True)

    def interpolate_mobject(self, alpha: float):
        current_frame = int(alpha * (self.num_of_frames - 1))
        self.mobject.pixel_array = self.zoom_animation_data[current_frame].copy()
        if self.new_x_range:
            self.axes.update_x_range(tuple(self.x_values[current_frame]))
        if self.new_y_range:
            self.axes.update_y_range(tuple(self.y_values[current_frame]))

    def clean_up_from_scene(self, scene: Scene) -> None:
        super().clean_up_from_scene(scene)
        del self.zoom_animation_data

