import subprocess

from chaos_theory_subclassed_animations import *
from color_utils_chaos_theory import *
from color_utils_chaos_theory import TorchColorFuncs as TCF
from custom_manim import *
from mydebugger import timer, delete_memmap_files, trim_audio


def get_dp_graph(
        cs: DynamicAxes,
        num_of_points: int,
        x_range: tuple[float, float],
        stroke_width: int = 2,
        color: ManimColor = YELLOW,
):
    graph = cs.get_explicit_plot(
        lambda x: math.acos(-3 * math.cos(x * math.pi / 180) + 2) * 180 / math.pi,
        num_of_points=num_of_points,
        x_range=x_range,
        use_smoothing=False,

    )
    graph2 = cs.get_explicit_plot(
        lambda x: -math.acos(-3 * math.cos(x * math.pi / 180) + 2) * 180 / math.pi,
        num_of_points=num_of_points,
        x_range=x_range,
        use_smoothing=False
    )

    return Plotter(
        np.concatenate((graph.passed_points, graph2.passed_points[::-1], [graph.passed_points[0]]), axis=0),
        cs,
        True,
        stroke_wid=stroke_width,
        col=[color],
        use_dots=False
    )


def get_standard_cs(x_range: tuple[float, float] = (-180, 180), y_range: tuple[float, float] = (-180, 180)):
    return DynamicAxes(
        x_range=x_range,
        y_range=y_range,
        x_length=pixel_length / SCENE_PIXELS,
        y_length=pixel_length / SCENE_PIXELS,
        x_is_in_degrees=True,
        y_is_in_degrees=True,
        font_size_x=20,
        font_size_y=20,
        include_zero_lines=False,
        use_constant_tick_length=True,
        x_line_to_number_buff=0.125,
        y_line_to_number_buff=0.125,
        tick_length=0.0125
    )


class Scene4(ComplexScene):
    run = ComplexScene.run
    skip = ComplexScene.skip
    ignore = ComplexScene.ignore

    def setup(self):
        self.add(NumberPlane())
        self.slope = 0.6
        self.crit = 2 / 3

        self.RAINBOW = [
            PURE_RED,
            rgb_to_color([255, 165, 0]),
            YELLOW,
            PURE_GREEN,
            rgb_to_color([0, 255, 255]),
            PURE_BLUE,
            rgb_to_color([255, 0, 255]),
        ]
        self.LIGHT_RAINBOW = [*[interpolate_color(col, WHITE, 0.9) for col in self.RAINBOW]]
        self.cs = DynamicAxes(
            x_range=(-180, 180),
            y_range=(-180, 180),
            x_length=10,
            y_length=6,
            x_is_in_degrees=True,
            y_is_in_degrees=True,
            font_size_x=30,
            font_size_y=30,
        )
        # self.main_dp = DoublePendulum(
        #     angle_pair=(30, 60)
        # )
        self.main_dp = DoublePendulum(
            angle_pair=(30, 60)
        )
        self.dp = self.main_dp.create_double_pendulum(2.5, 2.5)
        self.time_elapsed = 20

        points = self.main_dp.get_data_factory(self.time_elapsed).get_points_for_plotting_angle1_time()
        points2 = self.main_dp.get_data_factory(self.time_elapsed).get_points_for_plotting_angle2_time()
        both_points = self.main_dp.get_data_factory(100).get_points_for_plotting_two_angles()
        self.plot = Plotter(
            points, self.cs, True, 2, [FIRST_ROD_COLOR], use_dots=False)
        self.plot2 = Plotter(
            points2, self.cs, True, 2, [SECOND_ROD_COLOR], use_dots=False)
        self.plot3 = Plotter(
            both_points, self.cs, True, 1, [ORANGE], stroke_opa=0.45, use_dots=True
        )
        for plot in [self.plot, self.plot2, self.plot3]:
            plot.set_z_index(2)

        self.ghosts = DoublePendulumGhosts(
            self.main_dp,
        )

    def construct(self):
        self.clear()
        self.add_background()
        self.play_subscenes()

    @ignore
    def scene4_1_dp(self):
        self.dp.angle_pair = (0, 0)
        self.dp.shift(4.1 * DOWN)

        self.play(Move(self.dp, ORIGIN, rate_func=anticipate, run_time=1))
        av = self.dp.create_angle_visualizer()
        av.angle_1_value.set_z_index(1)
        av.angle_2_value.set_z_index(1)
        self.add(av)
        self.play(ManualDoublePendulumAnimation(self.dp, (30, 0), 1, smootherstep))
        self.play(ManualDoublePendulumAnimation(self.dp, (30, 60), 2, smootherstep))
        self.play(ReleaseDoublePendulum(self.main_dp, av, 20))

    @ignore
    def scene4_1_longer_entrance(self):
        self.dp.angle_pair = (0, 0)
        self.dp.shift(4.1 * DOWN)

        self.play(Move(self.dp, ORIGIN, rate_func=anticipate, run_time=5 / 3))

    @ignore
    def scene4_1_dp_for_hilbert(self):
        main_dp = DoublePendulum(
            angle_pair=(120, 100)
        )
        dp = main_dp.create_double_pendulum(3, 3, WHITE, WHITE).shift(2.5 * UP)

        self.add(dp)
        self.play(ReleaseDoublePendulum(main_dp, None, 38))

    @ignore
    def scene4_1_plot(self):
        self.cs = DynamicAxes(
            x_range=(0, 20),
            y_range=(-90, 90),
            x_length=12,
            y_length=1.5,
            x_is_in_degrees=False,
            y_is_in_degrees=True,
            font_size_x=20,
            font_size_y=20,
            include_zero_lines=True,
            use_constant_tick_length=True,
            x_line_to_number_buff=0.12,
            y_line_to_number_buff=0.32,
            x_axis_color=AMBER_ORANGE,
            y_axis_color=AMBER_ORANGE,
            labeled_values_for_x_override=np.arange(0, 20.1, 1).tolist(),
            labeled_values_for_y_override=[-90, -45, 0, 45, 90],
            tick_length=0.01

        )
        time_label = Text(r"time (s)", color=self.cs.x_axis_color
                          ).scale(0.4).next_to(self.cs.x_axis.get_end(), RIGHT, SMALL_BUFF * 1.5)
        y_label = Text(r"angle", color=self.cs.y_axis_color
                       ).scale(0.4).next_to(self.cs.y_axis.get_end(), UP, SMALL_BUFF * 1.5)
        self.cs.x_axis_label = time_label
        self.cs.y_axis_label = y_label

        Group(self.cs, time_label, y_label).next_to(DOWN * 4, UP, buff=0.05)
        bg_rect = self.cs.get_background_rectangle()

        self.plot.update_axes(self.cs)
        self.plot2.update_axes(self.cs)

        self.play(Create(self.cs, shift=UP, y_axis_first=True))
        self.play(DrawPlot(self.plot, run_time=self.time_elapsed),
                  DrawPlot(self.plot2, run_time=self.time_elapsed))

    @ignore
    def scene4_2_dp(self):
        self.dp.angle_pair = (0, 0)
        self.dp.shift(4.1 * DOWN)

        self.play(Move(self.dp, ORIGIN, rate_func=anticipate, run_time=1))
        av = self.dp.create_angle_visualizer()
        av.angle_1_value.set_z_index(1)
        av.angle_2_value.set_z_index(1)
        self.add(av)
        self.play(ManualDoublePendulumAnimation(self.dp, (30, 0), 1, smootherstep))
        self.play(ManualDoublePendulumAnimation(self.dp, (30, 60), 2, smootherstep))
        self.play(ReleaseDoublePendulum(
            self.main_dp,
            av,
            100,
        ))

    @ignore
    def scene4_2_both_angles_plot(self):
        self.cs = DynamicAxes(
            x_range=(-90, 90),
            y_range=(-90, 90),
            x_length=7,
            y_length=7,
            x_is_in_degrees=True,
            y_is_in_degrees=True,
            font_size_x=22,
            font_size_y=22,
            include_zero_lines=True,
            x_line_to_number_buff=0.12,
            y_line_to_number_buff=0.12,
            labeled_values_for_x_override=[-90, -60, -30, 0, 30, 60, 90],
            labeled_values_for_y_override=[-90, -60, -30, 0, 30, 60, 90],
            tick_length=0.013
        )
        bg_rect = self.cs.get_background_rectangle()
        both_points = self.main_dp.get_data_factory(100).get_points_for_plotting_two_angles()
        plot3 = Plotter(
            both_points,
            self.cs,
            True,
            1,
            [AMBER_ORANGE],
            0.4,
            use_dots=True
        ).set_z_index(2)
        plot3.tracer.move_to(plot3.start_dot)
        self.play(Create(self.cs, shift=LEFT * 3))
        self.play(FadeIn(VGroup(plot3.start_dot, plot3.tracer)))
        self.play(DrawPlot(plot3, run_time=100))

    @staticmethod
    def nine_grid(color: ManimColor = WHITE, stroke_width: float = 5) -> VMobject:
        width = config.frame_width - stroke_width * STROKE_WIDTHS
        height = config.frame_height - stroke_width * STROKE_WIDTHS
        return Rectangle(
            width=width,
            height=height,
            grid_xstep=width / 3,
            grid_ystep=height / 3,
            color=color,
            stroke_width=stroke_width
        )

    def get_nine(
            self,
            angle_pairs: Sequence[tuple[float, float]],
            duration: float,
            run_time: float,
            plot_color: ManimColor = WHITE,
            is_chaos=False,
    ):
        x, y = config.frame_width, config.frame_height
        x_interval, y_interval = x / 3, y / 3
        center_ax_loc = (7 / 48) * (x / 2) * RIGHT + 0.0625 * UP
        dp_loc_offset = 1.225 * LEFT
        U, V = np.meshgrid(np.array([-1, 0, 1]), np.array([1, 0, -1]))
        coordinate_pairs = np.column_stack((U.ravel(), V.ravel()))

        ax_locations = []
        dp_locations = []
        for coordinate_pair in coordinate_pairs:
            ax_location = center_ax_loc + x_interval * coordinate_pair[0] * RIGHT + y_interval * coordinate_pair[1] * UP
            ax_locations.append(ax_location)

            dp_location = dp_loc_offset + x_interval * coordinate_pair[0] * RIGHT + y_interval * coordinate_pair[1] * UP
            dp_locations.append(dp_location)
        axes = []
        for ax_location in ax_locations:
            ax = DynamicAxes(
                x_range=(-90, 90) if not is_chaos else (-180, 180),
                y_range=(-90, 90) if not is_chaos else (-180, 180),
                x_length=2.3,
                y_length=2.3,
                x_is_in_degrees=True,
                y_is_in_degrees=True,
                font_size_x=13,
                font_size_y=13,
                include_zero_lines=True,
                x_line_to_number_buff=0.175,
                y_line_to_number_buff=0.15,
                labeled_values_for_x_override=[-90, -45, 0, 45, 90] if not is_chaos else [-180, -90, 0, 90, 180],
                labeled_values_for_y_override=[-90, -45, 0, 45, 90] if not is_chaos else [-180, -90, 0, 90, 180],
                tick_length=0.008
            ).shift(ax_location)
            bg_rect = ax.get_background_rectangle()
            axes.append(ax)
        main_dp_list = []
        dp_list = []
        for dp_location in dp_locations:
            main_dp = DoublePendulum((0, 0))
            main_dp_list.append(main_dp)
            dp = main_dp.create_double_pendulum(
                0.7 if not is_chaos else 0.675,
                0.7 if not is_chaos else 0.675,
            )
            dp_list.append(dp)
            dp.move_to(dp_location - dp_loc_offset)

        self.play(AnimationGroup(
            *[FadeIn(dp, shift=UP * 1.5, scale=0.2, rate_func=anticipate)
              for dp in dp_list],
            run_time=1,
            lag_ratio=0
        ))
        self.play(*[AnimationGroup(
            Create(ax, shift=LEFT * 2, scale=0.25, slow_factor=2.0),
            MoveRelative(dp,
                         dp.rod1.get_start(),
                         dp_location + (0.5 if not is_chaos else 0.2) * UP,
                         rate_func=anticipate, )
        )
            for i, (ax, dp, dp_location) in enumerate(zip(axes, dp_list, dp_locations))
        ])

        av_list = []
        plot_list = []
        manual_1_list = []
        fadereplace_list = []
        for i, (main_dp, dp, ax) in enumerate(zip(main_dp_list, dp_list, axes)):
            av = dp.create_angle_visualizer(0.05)
            av.angle_1_value.set_z_index(1)
            av.angle_2_value.set_z_index(1)
            av_list.append(av)
            manual_1_list.append(ManualDoublePendulumAnimation(dp, (angle_pairs[i][0], 0), run_time=1))

        self.play(*manual_1_list)
        manual_2_list = []
        for i, (main_dp, dp, ax) in enumerate(zip(main_dp_list, dp_list, axes)):
            manual_2_list.append(ManualDoublePendulumAnimation(dp, angle_pairs[i], run_time=1))

        self.play(*manual_2_list)
        for i, (main_dp, dp, ax) in enumerate(zip(main_dp_list, dp_list, axes)):
            main_dp.init_angle_1, main_dp.init_angle_2 = angle_pairs[i]
            points = main_dp.get_data_factory(duration).get_points_for_plotting_two_angles()
            plotter = Plotter(
                points,
                ax,
                True,
                1,
                [plot_color],
                stroke_opa=0.4 if is_chaos else 0.3,
            )
            plotter.revert_tracer()
            plot_list.append(plotter)
            fadereplace_list.append(FadeReplacementTransform(
                av_list[i][-2:].copy(),
                VGroup(plotter.start_dot, plotter.tracer),
                run_time=2
            ))

        self.play(*fadereplace_list)

        anim_list = []
        for i, (plot, av) in enumerate(zip(plot_list, av_list)):
            anim_list.append(ReleaseDoublePendulum(
                main_dp_list[i],
                av_list[i],
                duration,
                run_time,
            ))
            anim_list.append(
                DrawPlot(plot, run_time=run_time))
        self.wait(1)
        self.play(*anim_list)

    @ignore
    def scene4_2_many_small_angle_plots(self):
        small_angle_pairs = [
            (50, 10),
            (-30, 40),
            (80, 70),
            (-50, -15),
            (-10, 50),
            (-5, -50),
            (45, 45),
            (-10, 90),
            (-40, -40)
        ]
        self.play(Create(self.nine_grid(AMBER_ORANGE), run_time=2.5))
        self.get_nine(small_angle_pairs,
                      40, 40,  # FR: 40, 40
                      AMBER_ORANGE, is_chaos=False)

    @ignore
    def scene4_3_many_big_plots(self):
        big_angle_pairs = [
            (175, 120),
            (-130, 150),
            (110, 45),
            (-160, -45),
            (-45, 125),
            (-50, -130),
            (-150, 50),
            (50, -155),
            (-175, -175)
        ]
        self.play(Create(self.nine_grid(AMBER_ORANGE), run_time=2.5))
        self.get_nine(big_angle_pairs,
                      40, 40,  # FR: 40, 40
                      AMBER_ORANGE, is_chaos=True)

    @ignore
    def scene4_4_ghosts(self):
        cs = DynamicAxes(
            x_range=(-180, 180),
            y_range=(-180, 180),
            x_length=pixel_length / SCENE_PIXELS,
            y_length=pixel_length / SCENE_PIXELS,
            x_is_in_degrees=True,
            y_is_in_degrees=True,
            font_size_x=20,
            font_size_y=20,
            include_zero_lines=True,
            use_constant_tick_length=True,
            x_line_to_number_buff=0.125,
            y_line_to_number_buff=0.125,
            tick_length=0.015
        ).set_z_index(2)
        bg_rect = cs.get_background_rectangle()
        main_dp = DoublePendulum((0, 0))
        dp = main_dp.create_double_pendulum(1.5, 1.5).move_to(ORIGIN)
        av = dp.create_angle_visualizer()
        av.angle_1_value.set_z_index(1)
        av.angle_2_value.set_z_index(1)
        dp_title = Text(
            "INITIAL POSITION",
            fill_color=AMBER_ORANGE,
            font="Montserrat",
            weight=MEDIUM
        ).scale(0.5)
        plot_title = Text(
            "30-SECOND PLOT",
            fill_color=AMBER_ORANGE,
            font="Montserrat",
            weight=MEDIUM
        ).scale(0.35).next_to(bg_rect, UP, buff=0.05)
        Group(cs, bg_rect, plot_title).next_to((config.frame_width / 2) * RIGHT, LEFT, 0.05)

        self.play(FadeIn(dp, shift=4 * UP, rate_func=anticipate))
        self.play(MoveRelative(dp, dp.rod1.get_start(), 4 * LEFT, run_time=1, rate_func=anticipate)
                  , FadeIn(bg_rect, cs, shift=3 * LEFT, scale=0.25, rate_func=anticipate))
        dp_title.next_to(dp.rod1.get_start()[0] * RIGHT + UP * 3.825, DOWN, buff=0.1)
        self.play(ManualDoublePendulumAnimation(dp, (-135, 0), run_time=1))
        self.play(ManualDoublePendulumAnimation(dp, (-135, 90), run_time=1))
        main_dp.init_angle_1 = dp.angle_pair[0]
        main_dp.init_angle_2 = dp.angle_pair[1]
        points = main_dp.get_data_factory(
            30  # FR: 30  # test with 6
        ).get_points_for_plotting_two_angles()
        plot = Plotter(points, cs, True, 1,
                       col=[AMBER_ORANGE],
                       stroke_opa=0.35).set_z_index(-1)
        plot.revert_tracer()
        self.play(FadeReplacementTransform(
            av[-2:].copy(),
            VGroup(plot.start_dot, plot.tracer),
            run_time=1.5,

        ))
        ghosts = DoublePendulumGhosts(main_dp,
                                      200,  # FR: 200  # test with 5
                                      duration=1.2,
                                      max_opac=0.075  # FR: 0.075  # test with 0.5
                                      )
        self.play(FadeIn(dp_title, shift=UP * 5, scale=2, run_time=1),
                  Wiggle(VGroup(plot.start_dot, plot.tracer), 2, run_time=1))
        self.play(DrawPlot(plot, main_dp.data_factory.t_span),
                  ReleaseGhosts(ghosts, av))
        """
        self.play(AnimateGhostsWithPlot(
            ghosts,
            plot,
            (120, 90),
            13,
            self
        ))
        # self.add_sound("Just Watch.mp3")
        self.play(AnimateGhostsWithPlot(  # music begins here
            ghosts,
            plot,
            (0, 0),
            4.5,
            self,
            rate_func=create_spline_func([0, 1], [0, 1], [0, 1])
        ))
        self.play(AnimateGhostsWithPlot(  # 16 sec interesting order
            ghosts,
            plot,
            (-60, -40),
            10.75,
            self,
            rate_func=create_spline_func([0, 1], [1, 0], [0, 1])
        ))
        self.play(AnimateGhostsWithPlot(
            ghosts,
            plot,
            (42, -89),
            7.75,
            self,
        ))
        self.play(AnimateGhostsWithPlot(  # 16 sec interesting chaos
            ghosts,
            plot,
            (179.8, -179.8),
            5,
            self,
        ))
        self.play(AnimateGhostsWithPlot(
            ghosts,
            plot,
            (-175.05, -10.26),
            5,
            self,
        ))
        self.play(AnimateGhostsWithPlot(
            ghosts,
            plot,
            (101, 115),
            6.5,
            self,
        ))
        self.play(AnimateGhostsWithPlot(  # 16 sec visiting islands
            ghosts,
            plot,
            (99, 115),
            15.5,
            self,
        ))

        self.play(AnimateGhostsWithPlot(  # 15 sec spiral from far reaches to (0, 0)
            ghosts,
            plot,
            (-20, -35),
            16,
            self,
            rate_func=linear,
            use_spiral_angle_progression=True
        ))
        self.play(AnimateGhostsWithPlot(  # 13 secs ending
            ghosts,
            plot,
            (180, -90),
            16,
            self,
            rate_func=create_spline_func([0, 1], [0, 1], [0, 1])
        ))
        """

    @ignore
    def scene4_4_ghosts_plot_title(self):
        cs = DynamicAxes(
            x_range=(-180, 180),
            y_range=(-180, 180),
            x_length=pixel_length / SCENE_PIXELS,
            y_length=pixel_length / SCENE_PIXELS,
            x_is_in_degrees=True,
            y_is_in_degrees=True,
            font_size_x=20,
            font_size_y=20,
            include_zero_lines=True,
            use_constant_tick_length=True,
            x_line_to_number_buff=0.125,
            y_line_to_number_buff=0.125,
            tick_length=0.015
        ).set_z_index(2)
        bg_rect = cs.get_background_rectangle()
        main_dp = DoublePendulum((0, 0))
        dp = main_dp.create_double_pendulum(1.5, 1.5).move_to(ORIGIN)
        av = dp.create_angle_visualizer()
        av.angle_1_value.set_z_index(1)
        av.angle_2_value.set_z_index(1)
        dp_title = Text(
            "INITIAL POSITION",
            fill_color=AMBER_ORANGE,
            font="Montserrat",
            weight=MEDIUM
        ).scale(0.5)
        plot_title = Text(
            "30-SECOND PLOT",
            fill_color=AMBER_ORANGE,
            font="Montserrat Medium",
        ).scale(0.35).next_to(bg_rect, UP, buff=0.05)
        Group(cs, bg_rect, plot_title).next_to((config.frame_width / 2) * RIGHT, LEFT, 0.05)

        self.play(FadeIn(plot_title, shift=UP * 5, scale=2, run_time=1))

    @ignore
    def scene4_4_ghosts_testing(self):
        self.add_sound(trim_audio("begin_again_3.wav", 502, 80))
        cs = DynamicAxes(
            x_range=(-180, 180),
            y_range=(-180, 180),
            x_length=pixel_length / SCENE_PIXELS,
            y_length=pixel_length / SCENE_PIXELS,
            x_is_in_degrees=True,
            y_is_in_degrees=True,
            font_size_x=20,
            font_size_y=20,
            include_zero_lines=True,
            use_constant_tick_length=True,
            x_line_to_number_buff=0.125,
            y_line_to_number_buff=0.125,
            tick_length=0.015
        ).set_z_index(2)
        bg_rect = cs.get_background_rectangle()
        main_dp = DoublePendulum((0, 0))
        dp = main_dp.create_double_pendulum(1.5, 1.5).set_z_index(1).move_to(ORIGIN)
        av = dp.create_angle_visualizer()
        dp_title = Text(
            "INITIAL POSITION",
            fill_color=LIGHT_BROWN,
            font="Montserrat Medium",
        ).scale(0.5)
        plot_title = Text(
            "50-SECOND PLOT",
            fill_color=LIGHT_BROWN,
            font="Montserrat Medium",
        ).scale(0.35).next_to(bg_rect, UP, buff=0.05)
        Group(cs, bg_rect, plot_title).next_to((config.frame_width / 2) * RIGHT, LEFT, 0.05)

        self.wait()
        self.play(FadeIn(dp, shift=2 * UP, scale=0.2))
        self.play(MoveRelative(
            dp,
            dp.rod1.get_start(),
            4 * LEFT,
            run_time=1
        ), Create(cs, shift=3 * LEFT, scale=0.25))
        dp_title.next_to(dp.rod1.get_start()[0] * RIGHT + UP * 4, DOWN, buff=0.1)
        self.wait()
        self.play(ManualDoublePendulumAnimation(dp, (-31, 0), run_time=2))
        self.play(ManualDoublePendulumAnimation(dp, (-31, -150), run_time=1.5))
        main_dp.init_angle_1 = dp.angle_pair[0]
        main_dp.init_angle_2 = dp.angle_pair[1]
        points = main_dp.get_data_factory(50).get_points_for_plotting_two_angles()
        plot = Plotter(points, cs, True, 1,
                       col=[ManimColor("#FFBF00"), ManimColor("#FFA500"), ManimColor("#FF8000")],
                       stroke_opa=0.5).set_z_index(-1)
        plot.revert_tracer()
        self.play(FadeReplacementTransform(
            av[-2:].copy(),
            VGroup(plot.start_dot, plot.tracer),
            run_time=3,

        ))
        ghosts = DoublePendulumGhosts(main_dp,
                                      10,  # FR: 200
                                      max_opac=0.4
                                      )
        self.wait()
        self.play(FadeIn(dp_title, shift=UP * 5, scale=2, run_time=1.5),
                  Wiggle(VGroup(plot.start_dot, plot.tracer), 2, run_time=1.5))
        self.wait()
        self.play(DrawPlot(plot, main_dp.data_factory.t_span),
                  ReleaseGhosts(ghosts, av),
                  FadeIn(plot_title, shift=UP * 5, scale=2, run_time=1.5))
        self.wait()
        # """
        self.play(AnimateGhostsWithPlot(
            ghosts,
            plot,
            (60, 5),
            1,
            self
        ))
        self.wait()


class Scene5(ComplexScene):
    run = ComplexScene.run
    skip = ComplexScene.skip
    ignore = ComplexScene.ignore

    def setup(self):
        self.add(NumberPlane())

    def construct(self):
        self.clear()
        self.add_background()
        self.play_subscenes()

    @ignore
    def scene5_1_sweep(self):
        cs = DynamicAxes(
            x_range=(-180, 180),
            y_range=(-180, 180),
            x_length=pixel_length / SCENE_PIXELS,
            y_length=pixel_length / SCENE_PIXELS,
            x_is_in_degrees=True,
            y_is_in_degrees=True,
            font_size_x=20,
            font_size_y=20,
            include_zero_lines=False,
            use_constant_tick_length=True,
            x_line_to_number_buff=0.125,
            y_line_to_number_buff=0.125,
            tick_length=0.015
        ).shift(3 * RIGHT)
        cs_bg_rect = cs.get_background_rectangle()
        table_of_dps = TableOfDoublePendulums(cs,
                                              40, 40  # (40, 40)
                                              )
        table_title = Text(
            "ALL POSSIBLE INITIAL POSITIONS",
            fill_color=AMBER_ORANGE,
            font="Montserrat",
            weight=MEDIUM
        ).scale(0.35).next_to(cs_bg_rect, UP, buff=0.05).set_opacity(0)
        cs_group = Group(cs_bg_rect, cs, table_title, table_of_dps).next_to(
            (config.frame_width / 2) * RIGHT, LEFT, 0.05)

        release_table_run_time = 48  # R: 48

        stable_dps = table_of_dps.get_stable_dps()
        unstable_dps = table_of_dps.get_unstable_dps()
        dp_sweeper = UnionDoublePendulumConstructor(
            1.5,
            1.5,
            (0, 0),
        ).move_to(ORIGIN)
        dp_sweeper_av = dp_sweeper.create_angle_visualizer()
        dp_sweeper_av.angle_1_value.set_z_index(1)
        dp_sweeper_av.angle_2_value.set_z_index(1)

        self.play(FadeIn(dp_sweeper, shift=UP * 4, rate_func=anticipate))
        self.play(MoveRelative(dp_sweeper, dp_sweeper.rod1.get_start(), LEFT * 4, rate_func=anticipate),
                  FadeIn(cs_bg_rect, cs, shift=LEFT * 3, scale=0.25, rate_func=anticipate))
        self.add(dp_sweeper_av)
        self.play(ManualDoublePendulumAnimation(dp_sweeper, (-180, 0), 1.5))
        self.play(ManualDoublePendulumAnimation(dp_sweeper, (-180, -180), 1.5))
        sweep_anim = SweepToCreateDoublePendulums(
            self,
            dp_sweeper,
            dp_sweeper_av,
            table_of_dps,
        )
        self.play(sweep_anim)
        self.wait(4)
        copies_of_sweeper = VGroup(*sweep_anim.get_copies_of_sweeper())
        cs_group.add(copies_of_sweeper)
        self.play(FadeOut(Group(dp_sweeper_av, dp_sweeper), shift=LEFT * 3, rate_func=anticipate),
                  Move(cs_group, ORIGIN, rate_func=anticipate))
        table_title.set_opacity(1)
        self.play(FadeIn(table_title, shift=UP * 5, scale=2, run_time=2))
        # """
        self.remove(copies_of_sweeper)
        self.play(ReleaseTableWithManualAnimations(
            table_of_dps,
            unstable_dps,
            stable_dps,
            release_table_run_time
        ))
        # """

    @ignore
    def scene5_1_test_boundaries(self):
        cs = DynamicAxes(
            x_range=(-180, 180),
            y_range=(-180, 180),
            x_length=pixel_length / SCENE_PIXELS,
            y_length=pixel_length / SCENE_PIXELS,
            x_is_in_degrees=True,
            y_is_in_degrees=True,
            font_size_x=20,
            font_size_y=20,
            include_zero_lines=False,
            use_constant_tick_length=True,
            x_line_to_number_buff=0.125,
            y_line_to_number_buff=0.125,
            tick_length=0.015
        ).shift(3 * RIGHT)
        cs_bg_rect = cs.get_background_rectangle()
        table_of_dps = TableOfDoublePendulums(cs,
                                              40, 40  # (40, 40)
                                              )
        table_title = Text(
            "ALL POSSIBLE INITIAL POSITIONS",
            fill_color=AMBER_ORANGE,
            font="Montserrat Medium",
        ).scale(0.35).next_to(cs_bg_rect, UP, buff=0.05).set_opacity(0)
        cs_group = Group(cs_bg_rect, cs, table_title, table_of_dps).next_to(
            (config.frame_width / 2) * RIGHT, LEFT, 0.05)

        release_table_run_time = 48  # R: 48

        stable_dps = table_of_dps.get_stable_dps()
        unstable_dps = table_of_dps.get_unstable_dps()
        dp_sweeper = UnionDoublePendulumConstructor(
            1.5,
            1.5,
            (0, 0),
        ).move_to(ORIGIN)
        dp_sweeper_av = dp_sweeper.create_angle_visualizer()
        dp_sweeper_av.angle_1_value.set_z_index(1)
        dp_sweeper_av.angle_2_value.set_z_index(1)

        self.next_section(skip_animations=True)
        self.play(FadeIn(dp_sweeper, shift=UP * 4, rate_func=anticipate))
        self.play(MoveRelative(dp_sweeper, dp_sweeper.rod1.get_start(), LEFT * 4, rate_func=anticipate),
                  FadeIn(cs_bg_rect, cs, shift=LEFT * 3, scale=0.25, rate_func=anticipate))
        self.add(dp_sweeper_av)
        self.play(ManualDoublePendulumAnimation(dp_sweeper, (-180, 0), 1.5))
        self.play(ManualDoublePendulumAnimation(dp_sweeper, (-180, -180), 1.5))
        sweep_anim = SweepToCreateDoublePendulums(
            self,
            dp_sweeper,
            dp_sweeper_av,
            table_of_dps,
        )
        self.play(sweep_anim)
        self.wait(4)
        copies_of_sweeper = VGroup(*sweep_anim.get_copies_of_sweeper())
        cs_group.add(copies_of_sweeper)
        self.play(FadeOut(Group(dp_sweeper_av, dp_sweeper), shift=LEFT * 3, rate_func=anticipate),
                  Move(cs_group, ORIGIN, rate_func=anticipate))
        table_title.set_opacity(1)
        self.play(FadeIn(table_title, shift=UP * 5, scale=2, run_time=2))
        self.remove(copies_of_sweeper)

        self.next_section()
        for i, dp in tqdm(enumerate(table_of_dps.submobjects), desc="Processing labels",
                          total=len(table_of_dps.submobjects)):
            label = DecimalNumber(i, 0).scale(0.15).move_to(dp.rod1.get_start())
            self.add(label)
        self.play(ReleaseTableWithManualAnimations(
            table_of_dps,
            unstable_dps,
            stable_dps,
            13
        ))

    @ignore
    def scene5_1_clean(self):
        cs = DynamicAxes(
            x_range=(-180, 180),
            y_range=(-180, 180),
            x_length=pixel_length / SCENE_PIXELS,
            y_length=pixel_length / SCENE_PIXELS,
            x_is_in_degrees=True,
            y_is_in_degrees=True,
            font_size_x=20,
            font_size_y=20,
            include_zero_lines=False,
            use_constant_tick_length=True,
            x_line_to_number_buff=0.125,
            y_line_to_number_buff=0.125,
            tick_length=0.015
        ).shift(3 * RIGHT)
        cs_bg_rect = cs.get_background_rectangle()
        table_of_dps = TableOfDoublePendulums(cs,
                                              40, 40  # (40, 40)
                                              )
        table_title = Text(
            "ALL POSSIBLE INITIAL POSITIONS",
            fill_color=AMBER_ORANGE,
            font="Montserrat",
            weight=MEDIUM
        ).scale(0.35).next_to(cs_bg_rect, UP, buff=0.05).set_opacity(0)
        cs_group = Group(cs_bg_rect, cs, table_title, table_of_dps).next_to(
            (config.frame_width / 2) * RIGHT, LEFT, 0.05)

        release_table_run_time = 48  # R: 48

        stable_dps = table_of_dps.get_stable_dps()
        unstable_dps = table_of_dps.get_unstable_dps()
        dp_sweeper = UnionDoublePendulumConstructor(
            1.5,
            1.5,
            (0, 0),
        ).move_to(ORIGIN)
        dp_sweeper_av = dp_sweeper.create_angle_visualizer()
        dp_sweeper_av.angle_1_value.set_z_index(1)
        dp_sweeper_av.angle_2_value.set_z_index(1)

        self.play(FadeIn(dp_sweeper, shift=UP * 4, rate_func=anticipate))
        self.play(MoveRelative(dp_sweeper, dp_sweeper.rod1.get_start(), LEFT * 4, rate_func=anticipate),
                  FadeIn(cs_bg_rect, cs, shift=LEFT * 3, scale=0.25, rate_func=anticipate))
        self.add(dp_sweeper_av)
        self.play(ManualDoublePendulumAnimation(dp_sweeper, (-180, 0), 1.5))
        self.play(ManualDoublePendulumAnimation(dp_sweeper, (-180, -180), 1.5))
        sweep_anim = SweepToCreateDoublePendulums(
            self,
            dp_sweeper,
            dp_sweeper_av,
            table_of_dps,
        )
        self.play(sweep_anim)
        self.wait(4)
        copies_of_sweeper = VGroup(*sweep_anim.get_copies_of_sweeper())
        cs_group.add(copies_of_sweeper)
        self.play(FadeOut(Group(dp_sweeper_av, dp_sweeper), shift=LEFT * 3, rate_func=anticipate),
                  Move(cs_group, ORIGIN, rate_func=anticipate))
        table_title.set_opacity(1)
        self.play(FadeIn(table_title, shift=UP * 5, scale=2, run_time=2))
        self.remove(copies_of_sweeper)

        self.play(ReleaseTableOfDoublePendulums(
            table_of_dps,
            release_table_run_time
        ))

    @ignore
    def scene5_1_test(self):
        cs = DynamicAxes(
            x_range=(-180, 180),
            y_range=(-180, 180),
            x_length=pixel_length / SCENE_PIXELS,
            y_length=pixel_length / SCENE_PIXELS,
            x_is_in_degrees=True,
            y_is_in_degrees=True,
            font_size_x=20,
            font_size_y=20,
            include_zero_lines=False,
            use_constant_tick_length=True,
            x_line_to_number_buff=0.125,
            y_line_to_number_buff=0.125,
            tick_length=0.015
        ).shift(3 * RIGHT).set_z_index(2)
        cs_bg_rect = cs.get_background_rectangle()
        table_of_dps = TableOfDoublePendulums(cs, 40, 40)  # (40, 40)
        table_title = Text(
            "ALL POSSIBLE INITIAL POSITIONS",
            fill_color=LIGHT_BROWN,
            font="Montserrat Medium",
        ).scale(0.35)
        table_title.next_to(cs_bg_rect, UP, buff=0.05).set_opacity(0)
        cs_group = Group(cs, cs_bg_rect, table_title, table_of_dps).next_to((config.frame_width / 2) * RIGHT, LEFT,
                                                                            0.05)

        release_table_run_time = 25  # R: 48

        stable_dps = table_of_dps.get_stable_dps()
        unstable_dps = table_of_dps.get_unstable_dps()
        dp_sweeper = UnionDoublePendulumConstructor(
            1.5,
            1.5,
            (0, 0),
        ).move_to(ORIGIN)
        dp_sweeper_av = dp_sweeper.create_angle_visualizer().set_z_index(-1)

        self.wait()
        self.add(cs, cs_bg_rect, table_title, table_of_dps)
        self.wait()

    @ignore
    def scene5_for_bg(self):
        self.clear()
        cs = DynamicAxes(
            x_range=(22, 34),
            y_range=(112, 124),
            x_length=14.22,
            y_length=8,
            x_is_in_degrees=True,
            y_is_in_degrees=True,
            font_size_x=20,
            font_size_y=20,
            use_constant_tick_length=True
        ).shift(0.5 * LEFT + 0.5 * UP)
        table_of_dps = TableOfDoublePendulums(cs, 4, 4, WHITE, WHITE, is_sparse=False)
        self.add(table_of_dps)

    @ignore
    def scene5_for_bg(self):
        self.clear()
        self.add(NumberPlane().set_color(WHITE))

    @ignore
    def tenmins_intro(self):
        self.clear()
        cs = DynamicAxes(
            x_range=(-35, -30),
            y_range=(-166, -161),
            x_length=pixel_length / SCENE_PIXELS,
            y_length=pixel_length / SCENE_PIXELS,
            x_is_in_degrees=True,
            y_is_in_degrees=True,
            font_size_x=18,
            font_size_y=18,
            use_constant_tick_length=True,
            line_to_number_buff=0.125,
            tick_length=0.01,
            labeled_values_for_x_override=np.arange(-35, -29, 1).tolist(),
            labeled_values_for_y_override=np.arange(-166, -160, 1).tolist(),
        ).set_z_index(1).move_to(ORIGIN).shift(2.5 * RIGHT)
        dp_sweeper = UnionDoublePendulumConstructor(
            1.5,
            1.5,
            (0, 0),
        ).move_to(ORIGIN).set_z_index(1)
        dp_sweeper_av = dp_sweeper.create_angle_visualizer()
        dp_group = Group(dp_sweeper, dp_sweeper_av)

        self.play(FadeIn(dp_sweeper, shift=UP * 3, scale=0.25))

        self.play(MoveRelative(dp_sweeper, dp_sweeper.rod1.get_start(), LEFT * 4 + UP * 2),
                  FadeIn(cs, shift=LEFT * 4, scale=0.5))

        self.add(dp_sweeper_av)
        self.play(MoveRelative(dp_group, dp_sweeper.rod1.get_start(), LEFT * 5.5, run_time=2),
                  ManualDoublePendulumAnimation(dp_sweeper, (31, 161), 2, smooth))

        self.wait(2)
        table_of_dps = TableOfDoublePendulums(cs, 40, 40)
        sweep_anim = SweepToCreateDoublePendulums(
            self,
            dp_sweeper,
            dp_sweeper_av,
            table_of_dps,
            1,
            False
        )
        self.play(sweep_anim)
        self.wait(2)
        self.remove(*sweep_anim.get_copies_of_sweeper())
        self.play(
            FadeOut(dp_group, shift=LEFT * 3.5),
            Move(Group(cs, table_of_dps), ORIGIN),
            run_time=1
        )
        self.wait(2)

    @ignore
    def tenmins(self):
        self.clear()
        cs = DynamicAxes(
            x_range=(90, 91),
            y_range=(120, 121),
            x_length=config.frame_width,
            y_length=config.frame_height,
            x_is_in_degrees=True,
            y_is_in_degrees=True,
            font_size_x=18,
            font_size_y=18,
            use_constant_tick_length=True,
            line_to_number_buff=0.125,
            tick_length=0.01
        ).set_z_index(1)
        table_of_dps = TableOfDoublePendulums(cs, 10, 10)

        random.seed(1)
        indices = [34, 35, 42, 43, 44, 45, 54, 55, 56, 57, 64, 65]
        for i in range(len(table_of_dps.submobjects)):
            dp: UnionDoublePendulumConstructor = table_of_dps.submobjects[i]
            dp.bob1.scale(1.5)
            dp.bob2.scale(1.5)
            dp.set_stroke_width_of_rods(7).update_pens()
            if i in indices:
                dp.set_color([GREEN_A, PURE_GREEN, GREEN_E])
            else:
                table_of_dps.submobjects[i].angle_pair = (random.uniform(-70, -25), random.uniform(-160, -115))
                dp.set_color([RED_A, PURE_RED, RED_E])
            dp.scale(1.9, about_point=dp.rod1.get_start())

        table_of_dps.move_to(ORIGIN).scale(1.3)

        # group = VGroup(cs, table_of_dps).move_to(ORIGIN)

        # self.add(group)
        self.add(table_of_dps)
        vgroup = VGroup(*[table_of_dps.submobjects[i] for i in indices])
        vgroup.move_to(ORIGIN)
        # anim = ReleaseTableOfDoublePendulums(
        #     table_of_dps,
        #     75
        # )
        # anim.interpolate(1)

    @ignore
    def all_possible_intro(self):
        self.clear()
        cs = DynamicAxes(
            x_range=(-180, 180),
            y_range=(-180, 180),
            x_length=pixel_length / SCENE_PIXELS,
            y_length=pixel_length / SCENE_PIXELS,
            x_is_in_degrees=True,
            y_is_in_degrees=True,
            font_size_x=18,
            font_size_y=18,
            use_constant_tick_length=True,
            line_to_number_buff=0.125,
            tick_length=0.01
        ).set_z_index(1).move_to(ORIGIN).shift(2.5 * RIGHT)
        dp_sweeper = UnionDoublePendulumConstructor(
            # 1.5,
            # 1.5,
            4, 4,
            (0, 0),
        ).move_to(ORIGIN)
        dp_sweeper_av = dp_sweeper.create_angle_visualizer().set_z_index(-1)
        dp_group = Group(dp_sweeper, dp_sweeper_av)

        self.add(dp_group)
        anim = ManualDoublePendulumAnimation(dp_sweeper, (99, 113), 2, smooth)
        anim.interpolate(1)
        dp_group.move_to(ORIGIN)

        self.wait()
        self.play(FadeIn(dp_sweeper, shift=UP * 3, scale=0.25))
        self.play(MoveRelative(dp_sweeper, dp_sweeper.rod1.get_start(), LEFT * 4 + UP * 2),
                  FadeIn(cs, shift=LEFT * 4, scale=0.5))

        self.add(dp_sweeper_av)
        self.play(MoveRelative(dp_group, dp_sweeper.rod1.get_start(), LEFT * 5.5, run_time=2),
                  ManualDoublePendulumAnimation(dp_sweeper, (99, 113), 2, smooth))

        self.wait(2)
        table_of_dps = TableOfDoublePendulums(cs, 40, 40)
        sweep_anim = SweepToCreateDoublePendulums(
            self,
            dp_sweeper,
            dp_sweeper_av,
            table_of_dps,
            1,
            False
        )
        self.play(sweep_anim)
        self.wait(2)
        self.remove(*sweep_anim.get_copies_of_sweeper())
        self.play(
            FadeOut(dp_group, shift=LEFT * 3.5),
            Move(Group(cs, table_of_dps), ORIGIN),
            run_time=1
        )
        self.wait(2)

    @ignore
    def all_possible(self):
        self.clear()
        cs = DynamicAxes(
            x_range=(-34, -29),
            y_range=(-153, -148),
            x_length=pixel_length / SCENE_PIXELS,
            y_length=pixel_length / SCENE_PIXELS,
            x_is_in_degrees=True,
            y_is_in_degrees=True,
            font_size_x=18,
            font_size_y=18,
            use_constant_tick_length=True,
            tick_length=0.01
        ).set_z_index(1).move_to(ORIGIN).shift(2.5 * RIGHT)
        table_of_dps = TableOfDoublePendulums(cs, 40, 40)

        objs = Group(cs, table_of_dps).move_to(ORIGIN)

        self.add(objs)
        self.wait()
        self.play(ReleaseTableOfDoublePendulums(table_of_dps, 60))


def get_four_insets(visuals: PixelStaticVisuals | FlipStaticVisuals, include_image: bool) -> list[InsetScaffold]:
    inset_side_length = 3.8
    outgrowth_ul = InsetScaffold(
        visuals,
        (60, 135),
        (45, 120),
        inset_side_length,
        inset_side_length,
        2 * UP + 1.1 * RIGHT,
        inset_line_dirs=[UL, DL],
        include_image=include_image
    )
    island_ur = InsetScaffold(
        outgrowth_ul.inset_image,
        (99, 103),
        (113, 117),
        inset_side_length,
        inset_side_length,
        2 * UP + 5.1 * RIGHT,
        inset_line_dirs=[UL, DL],
        include_image=include_image
    )
    island_dl = InsetScaffold(
        visuals,
        (-50, -15),
        (-170, -135),
        inset_side_length,
        inset_side_length,
        2 * DOWN + 1.1 * RIGHT,
        inset_line_dirs=[UL, DL],
        include_image=include_image
    )
    island_dr = InsetScaffold(
        island_dl.inset_image,
        (-36, -30),
        (-166, -160),
        inset_side_length,
        inset_side_length,
        2 * DOWN + 5.1 * RIGHT,
        inset_line_dirs=[UL, DL],
        include_image=include_image
    )

    return [outgrowth_ul, island_ur, island_dl, island_dr]


scale_value = 5.6 * SCENE_PIXELS / pixel_length
shift_value = 3.81 * LEFT + UP * 0.95


class Scene6(ComplexScene):
    run = ComplexScene.run
    skip = ComplexScene.skip
    ignore = ComplexScene.ignore

    @staticmethod
    def get_tracker_and_static_tracker_copy(
            angle_1: float,
            angle_2: float,
            axes1: DynamicAxes,
            pixel_image: PixelStaticVisuals,
            static_tracker_location: np.ndarray,
            tracker_side_length: float,
            static_tracker_size: float,
            stroke_width: float = 1,
    ) -> tuple[Square, Square]:
        color_fill = rgba_to_color(pixel_image.color_func(torch.tensor([[angle_1, angle_2]]))[0].tolist())
        tracker = Square(
            side_length=tracker_side_length,
            stroke_color=BLACK,
            stroke_width=1,
            fill_opacity=1,
            fill_color=color_fill
        ).move_to(
            axes1.coords_to_point((angle_1, angle_2))
        )
        static_tracker_copy = Square(
            static_tracker_size,
            stroke_color=BLACK,
            stroke_width=stroke_width,
            fill_opacity=1,
            fill_color=color_fill
        ).move_to(static_tracker_location)

        return tracker, static_tracker_copy

    # endregion

    def setup(self):
        pass

    def construct(self):
        self.add_background()
        self.play_subscenes()

    @ignore
    def scene6_1_revisit_and_6_2(self):
        self.wait()
        # region 6.1 and 6.2 objects
        cs = DynamicAxes(
            x_range=(-180, 180),
            y_range=(-180, 180),
            x_length=(pixel_length / SCENE_PIXELS) * (6 / 7),
            y_length=(pixel_length / SCENE_PIXELS) * (6 / 7),
            x_is_in_degrees=True,
            y_is_in_degrees=True,
            font_size_x=20 * (6 / 7),
            font_size_y=20 * (6 / 7),
            include_zero_lines=False,
            use_constant_tick_length=True,
            x_line_to_number_buff=0.125 * (6 / 7),
            y_line_to_number_buff=0.125 * (6 / 7),
            tick_length=0.015 * (6 / 7)
        ).shift(LEFT * 3.4).set_z_index(2)
        cs_bg_rect = cs.get_background_rectangle().set_z_index(0)
        cs_group = Group(cs, cs_bg_rect)
        cs2 = DynamicAxes(
            x_range=(-180, 180),
            y_range=(-180, 180),
            x_length=pixel_length / SCENE_PIXELS,
            y_length=pixel_length / SCENE_PIXELS,
            x_is_in_degrees=True,
            y_is_in_degrees=True,
            font_size_x=20,
            font_size_y=20,
            include_zero_lines=False,
            use_constant_tick_length=True,
            x_line_to_number_buff=0.125,
            y_line_to_number_buff=0.125,
            tick_length=0.015
        ).set_z_index(2)
        cs2_bg_rect = cs2.get_background_rectangle().set_z_index(0)
        cs2_group = Group(cs2, cs2_bg_rect)
        move_relative_to(cs2_group, cs2_bg_rect.get_center(), ORIGIN)

        table_of_dps = TableOfDoublePendulums(cs2, 40, 40).set_z_index(2)  # R: (40, 40)
        table_title = Tex(
            "ALL POSSIBLE ", "INITIAL ", "POSITIONS",
            fill_color=AMBER_ORANGE,
            tex_template=get_font_for_tex("Montserrat Medium")
        ).scale(0.5).next_to(cs2_bg_rect, UP, buff=0.05).set_z_index(3)
        table_title2 = Tex(
            "ALL POSSIBLE", r"\ ", "POSITIONS",
            fill_color=AMBER_ORANGE,
            tex_template=get_font_for_tex("Montserrat Medium")
        ).scale(0.5).next_to(cs2_bg_rect, UP, buff=0.05).set_z_index(3)
        Group(cs2_group, table_title, table_title2, table_of_dps).move_to(ORIGIN)
        dp_master = DoublePendulum((0, 0))
        dp = dp_master.create_double_pendulum(1.5, 1.5).set_z_index(4)
        move_relative_to(dp, dp.rod1.get_start(), LEFT * 4)
        # endregion

        # region 6.1 and 6.2 animations
        self.play(FadeIn(cs2_bg_rect, cs2, table_title, table_of_dps, scale=0, rate_func=smootherstep))
        """
        self.play(ReplacementTransform(table_title, table_title2, run_time=1, rate_func=smooth))
        self.play(
            MoveRelative(
                Group(cs2_group, table_title2, table_of_dps),
                cs2_bg_rect.get_center(),
                RIGHT * 2.95,
                rate_func=anticipate),
            FadeIn(dp, shift=RIGHT * 2.95, rate_func=anticipate),
        )
        dp_visualizer = dp.create_angle_visualizer().set_z_index(3)
        dp_visualizer.angle_1_value.set_z_index(5)
        dp_visualizer.angle_2_value.set_z_index(5)
        self.add(dp_visualizer)
        self.play(ManualDoublePendulumAnimation(dp, (80, 0), 1))
        self.play(ManualDoublePendulumAnimation(dp, (80, 20), 0.5))
        self.wait(0.5)
        dp_master.init_angle_1 = dp.angle_pair[0]
        dp_master.init_angle_2 = dp.angle_pair[1]

        table_of_dps.save_state()
        self.play(TrackDoublePendulumWithTable(
            dp_master,
            table_of_dps,
            dp_visualizer,
            duration=13  # 13 seconds
        ))
        self.play(
            Restore(table_of_dps, run_time=2, rate_func=anticipate),
            FadeOut(dp, dp_visualizer, shift=DOWN * 3, run_time=2, rate_func=anticipate)
        )
        table_of_dps.set_z_index(2)
        self.play(Group(cs2_group, table_of_dps, table_title2).animate.shift(LEFT * 2.95),
                  run_time=1,
                  rate_func=anticipate)
        # endregion

        # region scene 6.3
        blocky_table = BlockyPixels(
            table_of_dps.cs,
            table_of_dps.row_count,
            table_of_dps.column_count,
            color_func_index=0,
            overlap=0,
            stroke_width=0.5,
            name="blocky_table of TurnTableIntoBlocks"
        ).set_z_index(1)
        lag = 8 / (table_of_dps.row_count * table_of_dps.column_count * 4)
        self.play(TurnTableIntoBlocks(table_of_dps, blocky_table, 5, lag))
        for block in blocky_table.submobjects:
            self.remove(block)
        self.add(blocky_table)
        # endregion

        # region scene 6.3 mobjects
        dp_master = DoublePendulum((dp_master.init_angle_1, dp_master.init_angle_2))
        dp = dp_master.create_double_pendulum(1.5, 1.5).shift(LEFT * 4).set_z_index(4)
        av = dp.create_angle_visualizer().set_z_index(3)
        av.angle_1_value.set_z_index(5)
        av.angle_2_value.set_z_index(5)
        pixel_image_1 = BlockyPixels(
            cs2,
            80,  # R: 80
            80,  # R: 80
            0,
            0,
            0.25
        ).set_z_index(1)
        pixel_image_1_2 = BlockyPixels(
            cs2,
            160,  # R: 160
            160,  # R: 160
            0,
            0,+
            0.125
        ).set_z_index(1)
        pixel_image_1_3 = BlockyPixels(
            cs2,
            320,  # R: 320
            320,  # R: 320
            0,
            0,
            0.0625
        ).set_z_index(1)
        pixel_image_2 = PixelStaticVisuals(
            cs2,
            pixel_length,
            pixel_length,
            TCF.torus_smooth_gradient,
        ).set_z_index(1)
        # endregion

        # region scene 6.3 animations
        pixel_group = Group(pixel_image_1, pixel_image_1_2, pixel_image_1_3, pixel_image_2).shift(RIGHT * 11)
        self.wait(2)
        self.add(pixel_group)
        shift_kwargs = {
            "shift_amount": LEFT * 11,
            "run_time": 2,
            "rate_func": slow_into
        }
        self.play(Shift(pixel_image_1, **shift_kwargs))
        self.remove(blocky_table)
        self.wait()
        self.play(Shift(pixel_image_1_2, **shift_kwargs))
        self.remove(pixel_image_1)
        self.wait()
        self.play(Shift(pixel_image_1_3, **shift_kwargs))
        self.remove(pixel_image_1_2)
        self.wait()
        self.play(Shift(pixel_image_2, **shift_kwargs))
        self.remove(pixel_image_1_3)
        self.wait(2)
        self.play(
            FadeIn(Group(av, dp), shift=RIGHT * 4.85),
            Group(cs2_group, pixel_image_2, table_title2).animate.shift(3 * RIGHT),
            run_time=1, rate_func=anticipate
        )
        tracker, static_tracker_copy = self.get_tracker_and_static_tracker_copy(
            dp_master.init_angle_1,
            dp_master.init_angle_2,
            cs2,
            pixel_image_2,
            LEFT * 4,
            0.2,
            3,
            stroke_width=1
        )
        tracker_kwargs = {
            "path_arc": -PI / 5,
            "run_time": 2.5,
            "rate_func": smootherstep
        }
        self.play(FadeReplacementTransform(
            VGroup(*av.copy().submobjects[4:6]),
            tracker.set_z_index(2),
            **tracker_kwargs
        ))
        self.play(FadeReplacementTransform(
            tracker.copy(),
            static_tracker_copy.set_z_index(2),
            **tracker_kwargs
        ))
        pixel_ratio = static_tracker_copy.side_length * (config.frame_height / pixel_length)
        turn_animation_into_updater(
            ManualScaleAnimation(
                static_tracker_copy,
                {6: 1, 15: pixel_ratio},
                smooth,
                run_time=27
            )
        )
        self.play(TrackDoublePendulumWithPixelVisuals(
            dp_master,
            pixel_image_2,
            av,
            tracker,
            static_tracker_copy,
            12,
            suspend_mobject_updating=False
        ))
        table_title = Tex(
            "ALL POSSIBLE ", "INITIAL ", "POSITIONS",
            fill_color=AMBER_ORANGE,
            tex_template=get_font_for_tex("Montserrat Medium")
        ).scale(0.5 * (6/7)).next_to(cs.bg_rectangle, UP, buff=0.05 * (6/7))
        new_table_of_dps = TableOfDoublePendulums(cs, table_of_dps.row_count, table_of_dps.column_count).set_z_index(7)
        self.play(
            FadeOut(Group(dp, av, static_tracker_copy, tracker), shift=DOWN * 8),
            FadeIn(Group(cs_group, new_table_of_dps, table_title), shift=DOWN * 8),
            Group(cs2_group, pixel_image_2, table_title2).animate.scale(6/7).shift(0.8 * RIGHT + 0.01 * UP),
            run_time=2,
            rate_func=anticipate
        )
        # endregion

        # region scene 6.4 mobjects
        for submob in cs2.submobjects:
            submob.set_z_index(6)

        tracker_group = VGroup()
        static_tracker_copy_group = VGroup()
        dp_to_tracker_anims = []
        tracker_to_static_tracker_copy_anims = []
        anims_kwargs = {
            "path_arc": -PI / 5,
            "run_time": 4,
            "rate_func": smoothererstep
        }
        for i, dp in enumerate(tqdm(new_table_of_dps.submobjects, desc="Moving DPs to the right")):
            tracker, static_tracker_copy = self.get_tracker_and_static_tracker_copy(
                dp._angle_pair[0],
                dp._angle_pair[1],
                cs2,
                pixel_image_2,
                cs.coords_to_point((dp._angle_pair[0], dp._angle_pair[1])),
                0.06,
                new_table_of_dps.x_retstep,
                0.5
            )
            tracker.set_z_index(8)
            static_tracker_copy.set_z_index(8)
            tracker_group.add(tracker)
            static_tracker_copy_group.add(static_tracker_copy)
            dp_to_tracker_anims.append(
                FadeReplacementTransform(dp.copy(), tracker, **anims_kwargs)
            )
            tracker_to_static_tracker_copy_anims.append(
                FadeReplacementTransform(tracker.copy(), static_tracker_copy, **anims_kwargs)
            )
        # endregion

        # region scene 6.4 animations
        self.play(AnimationGroup(*dp_to_tracker_anims, lag_ratio=0))
        cs.set_z_index(9)
        self.play(AnimationGroup(*tracker_to_static_tracker_copy_anims, lag_ratio=0),
                  FadeOut(new_table_of_dps, run_time=4),
                  )
        self.play(TrackTableOfDoublePendulumsWithPixelVisuals(
            new_table_of_dps,
            pixel_image_2,
            tracker_group,
            static_tracker_copy_group,
            30
        ))
        # endregion
        """

    @ignore
    def scene6_5_high_qual_pixelvisuals(self):
        label_kwargs = {"color": AMBER_ORANGE, "label_color": AMBER_ORANGE, "stroke_width": 4}
        text_kwargs = {"color": AMBER_ORANGE}
        width = MeasureLabel(
            label=Text("3840 px", **text_kwargs).scale(0.5),
            start=UPPER_LEFT_CORNER,
            end=UPPER_RIGHT_CORNER,
            **label_kwargs
        ).next_to(UP * config.frame_height / 2, DOWN, buff=SMALL_BUFF)
        height = MeasureLabel(
            label=Text("2160 px", **text_kwargs).scale(0.5),
            start=UPPER_LEFT_CORNER,
            end=LOWER_LEFT_CORNER,
            **label_kwargs
        ).next_to(LEFT * config.frame_width / 2, RIGHT, buff=SMALL_BUFF)
        fourk = Text("4K", font="Montserrat Medium", **text_kwargs).scale(2.25)
        cs = DynamicAxes(
            x_range=(-180, 180),
            y_range=(-180, 180),
            x_length=pixel_length / SCENE_PIXELS,
            y_length=pixel_length / SCENE_PIXELS,
            x_is_in_degrees=True,
            y_is_in_degrees=True,
            font_size_x=20,
            font_size_y=20,
            include_zero_lines=False,
            use_constant_tick_length=True,
            x_line_to_number_buff=0.125,
            y_line_to_number_buff=0.125,
            tick_length=0.015
        ).set_z_index(2)
        cs_bg_rect = cs.get_background_rectangle()
        cs_width = MeasureLabel(
            label=Text("2000 px", **text_kwargs).scale(0.4),
            start=cs_bg_rect.get_corner(UL),
            end=cs_bg_rect.get_corner(UR),
            **label_kwargs
        ).next_to(cs_bg_rect.get_corner(UP), DOWN, buff=SMALL_BUFF).set_z_index(2)
        cs_height = MeasureLabel(
            label=Text("2000 px", **text_kwargs).scale(0.4),
            start=cs_bg_rect.get_corner(UL),
            end=cs_bg_rect.get_corner(DL),
            **label_kwargs
        ).next_to(cs_bg_rect.get_corner(LEFT), RIGHT, buff=SMALL_BUFF).set_z_index(2)
        table_title = Text(
            "ALL POSSIBLE INITIAL POSITIONS",
            fill_color=AMBER_ORANGE,
            font="Montserrat Medium",
        ).scale(0.35).next_to(cs_bg_rect, UP, buff=0.05)
        pixel_visual = CrispPixelStaticVisuals(cs, TCF.torus_smooth_gradient)
        self.play(AnimationGroup(
            FadeIn(fourk, scale=16, run_time=1.5, rate_func=smoothererstep),
            FadeIn(width, shift=UP * config.frame_height, rate_func=slow_into),
            FadeIn(height, shift=LEFT * config.frame_width, rate_func=slow_into),
            lag_ratio=0.5
        ))
        self.play(AnimationGroup(
            FadeOut(height, shift=RIGHT * config.frame_width, rate_func=rush_into),
            FadeOut(width, shift=DOWN * config.frame_height, rate_func=rush_into),
            FadeOut(fourk, shift=LEFT * config.frame_width / 2, rate_func=rush_into),
            FadeIn(cs_width, shift=UP * cs_bg_rect.height, rate_func=slow_into),
            FadeIn(cs_height, shift=LEFT * cs_bg_rect.width, rate_func=slow_into),
            lag_ratio=0.5,
        ))
        self.play(FadeIn(pixel_visual, shift=LEFT * 11, rate_func=slow_into, run_time=1.5))
        self.play(FadeIn(cs, rate_func=smoothererstep))
        self.play(FadeIn(table_title, shift=UP * 5, scale=2, run_time=2))
        self.play(AnimationGroup(
            FadeOut(cs_height, shift=RIGHT * cs_bg_rect.width, rate_func=rush_into),
            FadeOut(cs_width, shift=DOWN * cs_bg_rect.height, rate_func=rush_into),
            lag_ratio=0.5
        ))
        # self.play(PixelVisualizationAnimation(
        #     pixel_visual,
        #     60,  # R: 60
        #     "scene_6_5",
        #     False
        # ))

    @ignore
    def scene6_5_high_qual_pixelvisuals_testing(self):
        label_kwargs = {"color": YELLOW, "label_color": YELLOW, "stroke_width": 4}
        text_kwargs = {"color": YELLOW}
        width = MeasureLabel(
            label=Text("3840 px", **text_kwargs).scale(0.5),
            start=UPPER_LEFT_CORNER,
            end=UPPER_RIGHT_CORNER,
            **label_kwargs
        ).next_to(UP * config.frame_height / 2, DOWN, buff=SMALL_BUFF)
        height = MeasureLabel(
            label=Text("2160 px", **text_kwargs).scale(0.5),
            start=UPPER_LEFT_CORNER,
            end=LOWER_LEFT_CORNER,
            **label_kwargs
        ).next_to(LEFT * config.frame_width / 2, RIGHT, buff=SMALL_BUFF)
        fourk = Text("4K", font="Montserrat Medium", **text_kwargs).scale(2.25)
        cs = DynamicAxes(
            x_range=(-180, 180),
            y_range=(-180, 180),
            x_length=pixel_length / SCENE_PIXELS,
            y_length=pixel_length / SCENE_PIXELS,
            x_is_in_degrees=True,
            y_is_in_degrees=True,
            font_size_x=20,
            font_size_y=20,
            include_zero_lines=False,
            use_constant_tick_length=True,
            x_line_to_number_buff=0.125,
            y_line_to_number_buff=0.125,
            tick_length=0.015
        ).set_z_index(2)
        cs_bg_rect = cs.get_background_rectangle()
        cs_width = MeasureLabel(
            label=Text("2000 px", **text_kwargs).scale(0.4),
            start=cs_bg_rect.get_corner(UL),
            end=cs_bg_rect.get_corner(UR),
            **label_kwargs
        ).next_to(cs_bg_rect.get_corner(UP), DOWN, buff=SMALL_BUFF).set_z_index(2)
        cs_height = MeasureLabel(
            label=Text("2000 px", **text_kwargs).scale(0.4),
            start=cs_bg_rect.get_corner(UL),
            end=cs_bg_rect.get_corner(DL),
            **label_kwargs
        ).next_to(cs_bg_rect.get_corner(LEFT), RIGHT, buff=SMALL_BUFF).set_z_index(2)
        table_title = Text(
            "ALL POSSIBLE INITIAL POSITIONS",
            fill_color=AMBER_ORANGE,
            font="Montserrat Medium",
        ).scale(0.35).next_to(cs_bg_rect, UP, buff=0.05)
        pixel_visual = CrispPixelStaticVisuals(cs, TCF.torus_smooth_gradient)
        self.add_sound(trim_audio("begin_again_3.wav", 786, 72))
        self.wait()
        self.play(AnimationGroup(
            FadeIn(fourk, scale=16, run_time=1.5, rate_func=smoothererstep),
            FadeIn(width, shift=UP * config.frame_height, rate_func=rush_from),
            FadeIn(height, shift=LEFT * config.frame_width, rate_func=rush_from),
            lag_ratio=0.5
        ))
        self.wait()
        self.play(AnimationGroup(
            FadeOut(height, shift=RIGHT * config.frame_width, rate_func=rush_into),
            FadeOut(width, shift=DOWN * config.frame_height, rate_func=rush_into),
            FadeOut(fourk, shift=LEFT * config.frame_width / 2, rate_func=rush_into),
            FadeIn(cs_width, shift=UP * cs_bg_rect.height, rate_func=rush_from),
            FadeIn(cs_height, shift=LEFT * cs_bg_rect.width, rate_func=rush_from),
            lag_ratio=0.5,
        ))
        self.play(Create(cs, scale=0.1))
        self.play(FadeIn(pixel_visual, scale=0.1))
        self.play(FadeIn(table_title, shift=UP * 5, scale=2, run_time=2))
        self.wait(4)
        self.play(AnimationGroup(
            FadeOut(cs_height, shift=RIGHT * cs_bg_rect.width, rate_func=rush_into),
            FadeOut(cs_width, shift=DOWN * cs_bg_rect.height, rate_func=rush_into),
            lag_ratio=0.5
        ))
        self.wait(2)
        self.play(PixelVisualizationAnimation(
            pixel_visual,
            50,  # R: 60
            "scene_6_5_testing",
            False
        ))
        self.wait()

    @ignore
    def scene6_6_more_color_layouts(self):
        color_funcs = [TCF.torus_smooth_gradient, TCF.boundary_highlight, TCF.divide_by_quadrant,
                       TCF.checkerboard, TCF.radial_center_gradient, TCF.cat]
        color_funcs_names = ["torus", "boundary_highlight", "quadrant", "checkerboard", "radial", "cat"]
        groups = []
        pixel_visuals = []
        for color_func in color_funcs:
            cs = DynamicAxes(
                x_range=(-180, 180),
                y_range=(-180, 180),
                x_length=3.5,
                y_length=3.5,
                x_is_in_degrees=True,
                y_is_in_degrees=True,
                font_size_x=15,
                font_size_y=15,
                use_constant_tick_length=True,
                x_line_to_number_buff=0.13,
                y_line_to_number_buff=0.13,
                tick_length=0.008,
                is_simplified_axis_ticks=True
            ).set_z_index(1)
            bg_rect = cs.get_background_rectangle()
            table_title = Tex(
                "ALL POSSIBLE ", "INITIAL ", "POSITIONS",
                fill_color=AMBER_ORANGE,
                font_size=16,
                tex_template=get_font_for_tex("Montserrat Medium")
            ).next_to(bg_rect, UP, buff=0.05)
            pixel_visual = CrispPixelStaticVisuals(cs, color_func)
            groups.append(Group(cs, pixel_visual, table_title))
            pixel_visuals.append(pixel_visual)

        Group(*groups).arrange_in_grid(2, 3, buff=SMALL_BUFF)
        intro_cs = DynamicAxes(
            x_range=(-180, 180),
            y_range=(-180, 180),
            x_length=pixel_length / SCENE_PIXELS,
            y_length=pixel_length / SCENE_PIXELS,
            x_is_in_degrees=True,
            y_is_in_degrees=True,
            font_size_x=20,
            font_size_y=20,
            include_zero_lines=False,
            use_constant_tick_length=True,
            x_line_to_number_buff=0.125,
            y_line_to_number_buff=0.125,
            tick_length=0.015
        ).set_z_index(2)
        intro_pixel_visual = CrispPixelStaticVisuals(intro_cs, TCF.torus_smooth_gradient).set_z_index(1)
        self.add(intro_pixel_visual)
        self.wait(1)
        self.play(
            AnimationGroup(
                intro_pixel_visual.animate.move_to(pixel_visuals[0].get_center()).scale(3.5 / intro_cs.x_length),
                FadeIn(groups[0]),
                lag_ratio=0.5
            )
        )
        self.play(AnimationGroup(
            *[FadeIn(group, shift=LEFT * 3, rate_func=smootherstep) for group in groups[1:]],
            lag_ratio=0.5
        ))
        self.remove(intro_pixel_visual)
        self.add(*groups)
        self.play(*[PixelVisualizationAnimation(pixel_visual, 60, "scene6_6_" + name, False)
                    for pixel_visual, name in zip(pixel_visuals, color_funcs_names)])

    def get_scene6_ending_setup(self):
        self.cs = get_standard_cs().set_z_index(2)
        self.cs_bg_rect = self.cs.get_background_rectangle()
        self.table_title = Tex(
            "ALL POSSIBLE ", "INITIAL ", "POSITIONS",
            fill_color=AMBER_ORANGE,
            font_size=20,
            tex_template=get_font_for_tex("Montserrat Medium")
        ).next_to(self.cs_bg_rect, UP, buff=0.05)
        self.axes_group = Group(self.cs, self.table_title)
        self.pixel_visual = CrispPixelStaticVisuals(self.cs, TCF.torus_smooth_gradient)
        self.graph = get_dp_graph(
            self.cs,
            750,
            (-80, 80),
            3,
            color=DARK_BROWN
        ).set_z_index(20)

        self.pixel_viz_duration = 90

        """
        Scene 6 clips ending keys:
        1 - axes and title moving NW with dp_graph animation
        2 - pixel visual in middle with scale-shift animation
        3 - outgrowth inset animation
        4 - island inset animation
        5 - outgrowth pixel visual animation
        6 - island1 pixel visual animation
        7 - island2 pixel visual animation
        8 - island3 pixel visual animation
        """

    @ignore
    def scene6_ending_1(self):
        self.get_scene6_ending_setup()
        self.add(self.axes_group)
        self.play(FadeIn(self.graph))
        self.play(FadeOut(self.graph))
        self.play(self.axes_group.animate.scale(scale_value, about_point=ORIGIN  # start at 25 seconds
                                                ).shift(shift_value), rate_func=smoothererstep)

    @ignore
    def scene6_ending_2(self):
        self.get_scene6_ending_setup()
        self.add(self.pixel_visual)
        # self.play(PixelVisualizationAnimation(
        #     self.pixel_visual, 90, "scene6_7", False
        # ).extra_animate(
        #     22, 1, shift=shift_value, scale=scale_value, rate_func=smoothererstep
        # )
        # )

    @ignore
    def scene6_ending_3(self):  # first subscene must be decorated with '@skip'
        self.clear()

        insets = get_four_insets(self.pixel_visual, include_image=False)
        self.play(Create(insets[0]))

    @ignore
    def scene6_ending_4(self):  # first subscene must be decorated with '@skip'
        self.clear()

        insets = get_four_insets(self.pixel_visual, include_image=False)
        for inset in insets[1:]:
            self.play(Create(inset))

    @ignore
    def scene6_ending_5(self):  # first 2 subscenes must be decorated with '@skip'
        self.clear()

        insets = get_four_insets(self.pixel_visual, include_image=True)
        out_growth_ul = insets[0]
        self.add(out_growth_ul[0])
        self.play(PixelVisualizationAnimation(
            out_growth_ul[0], 90, "scene6_8_0th_pixelviz_", False
        ))

    @ignore
    def scene6_ending_6(self):  # first 2 subscenes must be decorated with '@skip'
        self.clear()

        insets = get_four_insets(self.pixel_visual, include_image=True)
        island_ur = insets[1]
        self.add(island_ur[0])
        self.play(PixelVisualizationAnimation(
            island_ur[0], 90, "scene6_8_1th_pixelviz_", False
        ))

    @ignore
    def scene6_ending_7(self):  # first 2 subscenes must be decorated with '@skip'
        self.clear()

        insets = get_four_insets(self.pixel_visual, include_image=True)
        island_ll = insets[2]
        self.add(island_ll[0])
        self.play(PixelVisualizationAnimation(
            island_ll[0], 90, "scene6_8_2th_pixelviz_", False
        ))

    @ignore
    def scene6_ending_8(self):  # first 2 subscenes must be decorated with '@skip'
        self.clear()

        insets = get_four_insets(self.pixel_visual, include_image=True)
        island_lr = insets[3]
        self.add(island_lr[0])
        self.play(PixelVisualizationAnimation(
            island_lr[0], 90, "scene6_8_3th_pixelviz_", False
        ))

    @ignore
    def scene6_ending_8_substitute(self):
        cs = get_standard_cs().set_z_index(2)
        cs_bg_rect = cs.get_background_rectangle()
        table_title = Tex(
            "ALL POSSIBLE ", "INITIAL ", "POSITIONS",
            fill_color=AMBER_ORANGE,
            font_size=20,
            tex_template=get_font_for_tex("Montserrat Medium")
        ).next_to(cs_bg_rect, UP, buff=0.05)
        pixel_visual = CrispPixelStaticVisuals(cs, TCF.torus_smooth_gradient)
        orig_group = Group(pixel_visual, cs, table_title)

        self.add(orig_group)
        self.play(orig_group.animate.scale(
            scale_value, about_point=ORIGIN
        ).shift(shift_value))
        cs.x_length *= scale_value
        cs.y_length *= scale_value
        cs_bg_rect.scale(scale_value).shift(shift_value)

        pixel_visual.replace(CrispPixelStaticVisuals(cs, TCF.torus_smooth_gradient).shift(shift_value))
        insets = get_four_insets(pixel_visual, True)
        insets.append(InsetScaffold(
            insets[2].inset_image,
            (-34, -29),
            (-153, -148),
            3.8,
            3.8,
            2 * DOWN + 5.1 * RIGHT,
            [UL, DL],
            include_image=True
        ))

        for i, inset in enumerate(insets):
            for submob in inset.submobjects[1:]:
                submob.set_z_index(3)
            if i <= 3:
                self.play(FadeIn(inset, run_time=0.5, scale=0.3, shift=LEFT * 4))
        use_existing_list = ["scene6_8_1st_", "scene6_8_2nd_", "scene6_8_3rd_", "scene6_8_4th_",
                             "scene6_8_5th_", "scene6_8_sub_"]
        pixel_visualize_kwargs = {
            "duration": 70,
            "skip_processing": False
        }
        pixel_visual_anims = []
        for inset, ued in zip([pixel_visual, *insets[:-2]], use_existing_list[:-2]):
            pixel_visual_anims.append(
                PixelVisualizationAnimation(
                    inset[0],
                    use_existing_dat=ued,
                    **pixel_visualize_kwargs
                )
            )
        fourth_island_anim = PixelVisualizationAnimation(
            insets[-2][0],
            use_existing_dat=use_existing_list[-2],
            **pixel_visualize_kwargs
        )
        fourth_island_anim.extra_animate(55, 2, shift=8 * RIGHT, more_mobjects=insets[-2])
        pixel_visual_anims.append(fourth_island_anim)

        self.add(insets[-1].shift(4 * DOWN))
        sub_island_anim = PixelVisualizationAnimation(
            insets[-1][0],
            use_existing_dat=use_existing_list[-1],
            **pixel_visualize_kwargs
        )
        sub_island_anim.extra_animate(56, 1, shift=4 * UP, more_mobjects=insets[-1])
        pixel_visual_anims.append(sub_island_anim)

        self.play(AnimationGroup(*pixel_visual_anims))

    @ignore
    def scene6_for_sub_island(self):
        self.add_sound(trim_audio("begin_again_3.wav", 1181, 179))
        self.add_background()

        self.axes_group.add(self.pixel_visual)
        self.axes_group.scale(scale_value, about_point=ORIGIN).shift(shift_value)
        self.cs.x_length *= scale_value
        self.cs.y_length *= scale_value
        self.cs.x_axis.length *= scale_value
        self.cs.y_axis.length *= scale_value

        self.axes_group.remove(self.pixel_visual)
        pixel_visual = CrispPixelStaticVisuals(self.cs, TCF.torus_smooth_gradient)
        self.axes_group.add(pixel_visual)
        insets = get_four_insets(pixel_visual, include_image=True)
        insets.append(InsetScaffold(
            insets[2].inset_image,
            (-34, -29),
            (-153, -148),
            3.8,
            3.8,
            2 * DOWN + 5.1 * RIGHT,
            [UL, DL],
            include_image=True
        ))

        use_existing_list = ["scene6_sub_1st_", "scene6_sub_2nd_",
                             "scene6_sub_3rd_", "scene6_sub_4th_",
                             "scene6_sub_5th_", "scene6_sub_sub_"]
        pixel_visualize_kwargs = {
            "duration": 70,
            "skip_processing": True
        }
        pixel_visual_anims = [PixelVisualizationAnimation(
            pixel_visual,
            use_existing_dat=use_existing_list[0],
            **pixel_visualize_kwargs
        )]
        self.add(self.axes_group)

        for i, inset in enumerate(insets):
            for submob in inset.submobjects[1:]:
                submob.set_z_index(3)
            if i <= 2:
                self.add(inset)
                pixel_visual_anims.append(PixelVisualizationAnimation(
                    inset[0],
                    use_existing_dat=use_existing_list[i + 1],
                    **pixel_visualize_kwargs
                ))
            elif i == 3:  # 4th inset
                self.add(inset)
                anim = PixelVisualizationAnimation(
                    inset[0],
                    use_existing_dat=use_existing_list[i + 1],  # uses list[4] -> "scene6_sub_5th_"
                    **pixel_visualize_kwargs
                )
                anim.extra_animate(delay=50, duration=2, shift=8 * RIGHT, more_mobjects=inset.submobjects)
                pixel_visual_anims.append(anim)

            elif i == 4:  # 5th inset
                self.add(inset.shift(4 * DOWN))
                anim = PixelVisualizationAnimation(
                    inset[0],
                    use_existing_dat=use_existing_list[i + 1],  # uses list[5] -> "scene6_sub_sub_"
                    **pixel_visualize_kwargs
                )
                anim.extra_animate(delay=51, duration=1, shift=4 * UP, more_mobjects=inset.submobjects)
                pixel_visual_anims.append(anim)
            else:
                raise ValueError("must not reach here")

        self.wait(8)
        self.play(AnimationGroup(*pixel_visual_anims))
        self.wait()

    @ignore
    def four_quadrants_intro(self):
        text_duration = 0.75
        cs = main_cs.copy().shift(RIGHT * 3.1)
        cs.remove(cs.bg_rectangle)
        table_count = 32
        table_dps = TableOfDoublePendulums(cs, table_count, table_count).set_z_index(2)
        label_kwargs = {"color": WHITE, "stroke_width": 3}
        label_text_kwargs = {"color": WHITE}
        quad_pixel_visual = CrispPixelStaticVisuals(cs, TCF.divide_by_quadrant).set_z_index(-1)

        # LABELS
        cs_width = MeasureLabel(
            label=Text("2000 px", **label_text_kwargs).scale(0.4),
            start=cs.bg_rectangle.get_corner(UL),
            end=cs.bg_rectangle.get_corner(UR),
            **label_kwargs
        ).next_to(cs.bg_rectangle.get_corner(UP), DOWN, buff=SMALL_BUFF).set_z_index(2)
        cs_height = MeasureLabel(
            label=Text("2000 px", **label_text_kwargs).scale(0.4),
            start=cs.bg_rectangle.get_corner(UL),
            end=cs.bg_rectangle.get_corner(DL),
            **label_kwargs
        ).next_to(cs.bg_rectangle.get_corner(LEFT), RIGHT, buff=SMALL_BUFF).set_z_index(2)

        texts_kwargs = {"line_spacing": 0.75}

        # Define the common colors for "first", "second", and "both angles"
        colored_text_kwargs = {
            "t2c": {
                "first": FIRST_ROD_COLOR,
                "second": SECOND_ROD_COLOR,
            },
            "t2g": {
                "both angles": [FIRST_ROD_COLOR, SECOND_ROD_COLOR]
            }
        }
        MANGO_YELLOW = rgb_to_color([1 * 255, 0.8 * 255, 0.15 * 255])
        APPLE_GREEN = rgb_to_color([0.5 * 255, 0.9 * 255, 0 * 255])

        # Create the list of texts, and use dictionary merging for indices 3, 4, 5, and 6
        texts = [
            Text("This is a 2000 by 2000 pixel grid.", **texts_kwargs),  # index 0
            Text("Every pixel represents a double pendulum\nwith unique angles.", **texts_kwargs),  # index 1
            Text("This is what some of those double\npendulums look like.", **texts_kwargs),  # index 2
            # Index 3
            Text(
                "If both angles are positive,\nit is represented by a yellow pixel.",
                **texts_kwargs,
                **{**colored_text_kwargs,
                   "t2c": {**colored_text_kwargs["t2c"], "yellow pixel": MANGO_YELLOW}}
            ),
            # Index 4
            Text(
                "If the first angle is negative\nand the second is positive,\nit is represented by a blue pixel.",
                **texts_kwargs,
                **{**colored_text_kwargs, "t2c": {**colored_text_kwargs["t2c"], "blue pixel": PURE_BLUE}}
            ),
            # Index 5
            Text(
                "If both angles are negative,\nit is represented by a green pixel.",
                **texts_kwargs,
                **{**colored_text_kwargs, "t2c": {**colored_text_kwargs["t2c"], "green pixel": APPLE_GREEN}}
            ),
            # Index 6
            Text(
                "If the first angle is positive\nand the second is negative,\nit is represented by a red pixel.",
                **texts_kwargs,
                **{**colored_text_kwargs, "t2c": {**colored_text_kwargs["t2c"], "red pixel": PURE_RED}}
            ),
            Text(
                "These 4 million double pendulums in pixel\nform represent all possible initial positions\nof "
                "the double pendulum.",
                t2c={"all possible initial positions": LIGHT_BROWN},
                **texts_kwargs
            ),  # index 7
            Text("Before running this simulation, let's also\nzoom in on some interesting areas.", **texts_kwargs),
            # index 8
            Text("Now, let's release all of them at once.", font_size=42, font="Montserrat Medium",
                 color=WHITE, stroke_width=1, stroke_opacity=1, **texts_kwargs)
            # index 9
        ]

        for i, text in enumerate(texts):
            if i == 9:
                pass
            elif i in [3, 4, 5, 6]:
                text.scale(0.35).move_to(LEFT * 4)
            else:
                text.scale(0.375).move_to(LEFT * 4)

        Group(texts[3], texts[4], texts[5], texts[6]
              ).arrange(DOWN, 1.2, aligned_edge=LEFT).move_to(LEFT * 5)
        center_group = Group(quad_pixel_visual, cs)

        dp_sweeper = UnionDoublePendulumConstructor(
            1.5,
            1.5,
            (0, 0),
        )
        move_relative_to(dp_sweeper, dp_sweeper.rod1.get_start(), LEFT * 4)
        dp_sweeper_av = dp_sweeper.create_angle_visualizer().set_z_index(-1)

        self.play(FadeIn(center_group, shift=RIGHT * 8, scale=0.5))
        self.play(AnimationGroup(
            FadeIn(cs_width, shift=UP * cs.bg_rectangle.height, rate_func=rush_from),
            FadeIn(cs_height, shift=LEFT * cs.bg_rectangle.width, rate_func=rush_from),
            lag_ratio=0.5,
        ))
        self.play(AddTextLetterByLetter(texts[0], run_time=text_duration))
        self.wait(2)
        self.play(FadeOut(texts[0]))
        self.play(AddTextLetterByLetter(texts[1], run_time=text_duration))
        self.wait(2)
        self.play(
            AnimationGroup(
                FadeOut(texts[1]),
                AnimationGroup(
                    FadeOut(cs_height, shift=RIGHT * cs.bg_rectangle.width, rate_func=rush_into),
                    FadeOut(cs_width, shift=DOWN * cs.bg_rectangle.height, rate_func=rush_into),
                    lag_ratio=0.5
                ), FadeIn(dp_sweeper, shift=UP * 3, scale=0.25),
                lag_ratio=0.5
            )
        )

        self.add(dp_sweeper_av)
        self.play(ManualDoublePendulumAnimation(dp_sweeper, (-180, -180), 2, smooth))
        self.wait()
        sweep_anim = SweepToCreateDoublePendulums(
            self,
            dp_sweeper,
            dp_sweeper_av,
            table_dps,
            1,
        )
        self.play(sweep_anim)
        self.wait()
        self.play(AnimationGroup(
            FadeOut(Group(dp_sweeper, dp_sweeper_av), shift=UP * 3),
            AddTextLetterByLetter(texts[2], run_time=text_duration),
            lag_ratio=0.5
        ))
        self.wait(3)
        self.play(FadeOut(texts[2]))

        angles = [(45, 60), (-90, 135), (-60, -150), (45, -50)]
        dp_group = []
        dp_av_group = []
        for i in range(4):
            dp = UnionDoublePendulumConstructor(1, 1, angles[i]).next_to(texts[i + 3], RIGHT)
            dp.angle_pair = (0, 0)
            dp_av = dp.create_angle_visualizer()
            dp_group.append(dp)
            dp_av_group.append(dp_av)
        locations = [cs.coords_to_point(angle) for angle in angles]
        for i in range(4):
            self.play(AddTextLetterByLetter(texts[i + 3], run_time=text_duration * 1.5))
            self.wait(0.5)
            self.play(FadeIn(dp_group[i].set_z_index(1), shift=UP * 2.5, scale=0.3))
            self.add(dp_av_group[i])
            self.play(
                ManualDoublePendulumAnimation(dp_group[i], (angles[i][0], 0), 1, smooth)
            )
            self.play(
                ManualDoublePendulumAnimation(dp_group[i], angles[i], 1, smooth)
            )
            dp_copy = dp_group[i].copy()
            self.play(dp_copy.animate.scale((table_dps.rod_length / dp_copy.length_1) * 4).move_to(locations[i]),
                      run_time=1.5)
            self.play(ShrinkToCenter(dp_copy, rate_func=rush_from, run_time=1.5))
            self.wait(0.5)
        self.wait()
        self.play(AnimationGroup(
            FadeOut(texts[3], texts[4], texts[5], texts[6], *dp_group, *dp_av_group,
                    *sweep_anim.get_copies_of_sweeper(), shift=6 * RIGHT, run_time=2),
            AddTextLetterByLetter(texts[7], run_time=text_duration * 2),
            lag_ratio=0.5
        ))
        self.wait(2)
        table_title = Text(
            "all possible initial positions",
            fill_color=LIGHT_BROWN,
            font="Montserrat Medium",
        ).scale(0.375).next_to(cs.bg_rectangle, UP, buff=0.0125)
        self.play(ReplacementTransform(texts[7][48:75].copy(), table_title, run_time=2),
                  FadeOut(texts[7]))
        center_group.add(table_title)
        self.play(AddTextLetterByLetter(texts[8], run_time=text_duration))
        self.wait(2)
        self.play(AnimationGroup(
            FadeOut(texts[8], run_time=1.5),
            Move(center_group, LEFT * 2.985, run_time=1.5),
            lag_ratio=0.5
        ))
        center_group = Group(cs, quad_pixel_visual, table_title).move_to(LEFT * 2.985)

        # INSETS
        outgrowth_ur = PixelInset(
            quad_pixel_visual,
            (60, 140),
            (70, 120),
            5.5,
            5.5 * (5 / 8),
            2 * UP + 4.17 * RIGHT,
            [UL, DL],
            16, 16,
            include_return_cs=True,
            line_to_number_buff=0.15,
            inset_color=DARK_BROWN,
            inset_stroke_width=1.5,
            # labeled_values_for_y_override=np.arange(80, 121, 10).tolist()
        )
        island_dr = PixelInset(
            outgrowth_ur.inset_pixel_visual,
            (98, 104),
            (113, 117),
            5.5,
            5.5 * (2 / 3),
            1.87 * DOWN + 4.17 * RIGHT,
            [UL, UR],
            16, 16,
            include_return_cs=True,
            line_to_number_buff=0.15,
            inset_color=DARK_BROWN,
            inset_stroke_width=1.5,
            labeled_values_for_x_override=np.arange(98, 105, 1).tolist(),
            labeled_values_for_y_override=np.arange(113, 118, 1).tolist(),
        )
        FadeIn(outgrowth_ur, scene=self, shift=LEFT * 2)
        self.wait(0.5)
        FadeIn(island_dr, scene=self, shift=LEFT * 2)
        self.wait(2)
        self.play(FadeIn(texts[9].set_z_index(11), scale=0.3, shift=3 * UP, run_time=2))
        self.wait(2)
        self.play(FadeOut(texts[9]))
        self.add(center_group, outgrowth_ur, island_dr)
        self.wait()
        # duration_of_pixel_vis = 25
        # self.play(AnimationGroup(
        #     PixelVisualizationAnimation(quad_pixel_visual, duration_of_pixel_vis),
        #     *[PixelVisualizationAnimation(inset.inset_pixel_visual, duration_of_pixel_vis)
        #       for inset in [outgrowth_ur, island_dr]]))
        # self.wait()

    @ignore
    def four_quadrants(self):
        text_duration = 0.75
        cs = main_cs.copy().shift(RIGHT * 3.1)
        cs.remove(cs.bg_rectangle)
        table_count = 32
        table_dps = TableOfDoublePendulums(cs, table_count, table_count).set_z_index(2)
        label_kwargs = {"color": WHITE, "stroke_width": 3}
        label_text_kwargs = {"color": WHITE}
        quad_pixel_visual = CrispPixelStaticVisuals(cs, TCF.torus_smooth_gradient).set_z_index(-1)
        """

        #LABELS
        cs_width = MeasureLabel(
            label=Text("2000 px", **label_text_kwargs).scale(0.4),
            start=cs.bg_rectangle.get_corner(UL),
            end=cs.bg_rectangle.get_corner(UR),
            **label_kwargs
        ).next_to(cs.bg_rectangle.get_corner(UP), DOWN, buff=SMALL_BUFF).set_z_index(2)
        cs_height = MeasureLabel(
            label=Text("2000 px", **label_text_kwargs).scale(0.4),
            start=cs.bg_rectangle.get_corner(UL),
            end=cs.bg_rectangle.get_corner(DL),
            **label_kwargs
        ).next_to(cs.bg_rectangle.get_corner(LEFT), RIGHT, buff=SMALL_BUFF).set_z_index(2)

        texts_kwargs = {"line_spacing": 0.75}

        # Define the common colors for "first", "second", and "both angles"
        colored_text_kwargs = {
            "t2c": {
                "first": FIRST_ROD_COLOR,
                "second": SECOND_ROD_COLOR,
            },
            "t2g": {
                "both angles": [FIRST_ROD_COLOR, SECOND_ROD_COLOR]
            }
        }
        MANGO_YELLOW = rgb_to_color([1 * 255, 0.8 * 255, 0.15 * 255])
        APPLE_GREEN = rgb_to_color([0.5 * 255, 1 * 255, 0.2 * 255])

        # Create the list of texts, and use dictionary merging for indices 3, 4, 5, and 6
        texts = [
            Text("This is a 2000 by 2000 pixel grid.", **texts_kwargs),  # index 0
            Text("Every pixel represents a double pendulum\nwith unique angles.", **texts_kwargs),  # index 1
            Text("This is what some of those double\npendulums look like.", **texts_kwargs),  # index 2
            # Index 3
            Text(
                "If both angles are positive,\nit is represented by a yellow pixel.",
                **texts_kwargs,
                **{**colored_text_kwargs,
                   "t2c": {**colored_text_kwargs["t2c"], "yellow": MANGO_YELLOW}}
            ),
            # Index 4
            Text(
                "If the first angle is negative\nand the second is positive,\nit is represented by a blue pixel.",
                **texts_kwargs,
                **{**colored_text_kwargs, "t2c": {**colored_text_kwargs["t2c"], "blue": PURE_BLUE}}
            ),
            # Index 5
            Text(
                "If both angles are negative,\nit is represented by a green pixel.",
                **texts_kwargs,
                **{**colored_text_kwargs, "t2c": {**colored_text_kwargs["t2c"], "green": APPLE_GREEN}}
            ),
            # Index 6
            Text(
                "If the first angle is positive\nand the second is negative,\nit is represented by a red pixel.",
                **texts_kwargs,
                **{**colored_text_kwargs, "t2c": {**colored_text_kwargs["t2c"], "red": PURE_RED}}
            ),
            Text(
                "These 4 million double pendulums in pixel\nform represent all possible initial positions\nof "
                "the double pendulum.",
                t2c={"all possible initial positions": LIGHT_BROWN},
                **texts_kwargs
            ),  # index 7
            Text("Before running this simulation, let's also\nzoom in on some interesting areas.", **texts_kwargs),
            # index 8
            Text("Now, let's release all of them at once.", font_size=50, font="Montserrat Medium", **texts_kwargs)  # index 9
        ]

        for i, text in enumerate(texts):
            if i == 9:
                pass
            elif i in [3, 4, 5, 6]:
                text.scale(0.35).move_to(LEFT * 4 + UP * 0.5)
            else:
                text.scale(0.375).move_to(LEFT * 4 + UP * 0.5)

        Group(texts[3], texts[4], texts[5], texts[6]
                               ).arrange(DOWN, 1.2, aligned_edge=LEFT).move_to(LEFT * 5)
        center_group = Group(quad_pixel_visual, cs)

        dp_sweeper = UnionDoublePendulumConstructor(
            1.5,
            1.5,
            (0, 0),
        )
        move_relative_to(dp_sweeper, dp_sweeper.rod1.get_start(), LEFT * 4)
        dp_sweeper_av = dp_sweeper.create_angle_visualizer().set_z_index(-1)

        self.play(FadeIn(center_group, shift=RIGHT * 8, scale=0.5))
        self.play(AnimationGroup(
            FadeIn(cs_width, shift=UP * cs.bg_rectangle.height, rate_func=rush_from),
            FadeIn(cs_height, shift=LEFT * cs.bg_rectangle.width, rate_func=rush_from),
            lag_ratio=0.5,
        ))
        self.play(AddTextLetterByLetter(texts[0], run_time=text_duration))
        self.wait(2)
        self.play(FadeOut(texts[0]))
        self.play(AddTextLetterByLetter(texts[1], run_time=text_duration))
        self.wait(2)
        self.play(
            AnimationGroup(
                FadeOut(texts[1]),
                AnimationGroup(
                    FadeOut(cs_height, shift=RIGHT * cs.bg_rectangle.width, rate_func=rush_into),
                    FadeOut(cs_width, shift=DOWN * cs.bg_rectangle.height, rate_func=rush_into),
                    lag_ratio=0.5
                ), FadeIn(dp_sweeper, shift=UP * 3, scale=0.25),
                lag_ratio=0.5
            )
        )

        self.add(dp_sweeper_av)
        self.play(ManualDoublePendulumAnimation(dp_sweeper, (-180, -180), 2, smooth))
        self.wait()
        sweep_anim = SweepToCreateDoublePendulums(
            self,
            dp_sweeper,
            dp_sweeper_av,
            table_dps,
            1,
        )
        self.play(sweep_anim)
        self.wait()
        self.play(AnimationGroup(
            FadeOut(Group(dp_sweeper, dp_sweeper_av), shift=UP * 3),
            AddTextLetterByLetter(texts[2], run_time=text_duration),
            lag_ratio=0.5
        ))
        self.wait(3)
        self.play(FadeOut(texts[2]))

        angles = [(45, 60), (-90, 135), (-60, -150), (45, -50)]
        dp_group = []
        dp_av_group = []
        for i in range(4):
            dp = UnionDoublePendulumConstructor(1, 1, angles[i]).next_to(texts[i + 3], RIGHT)
            dp.angle_pair = (0, 0)
            dp_av = dp.create_angle_visualizer()
            dp_group.append(dp)
            dp_av_group.append(dp_av)
        locations = [cs.coords_to_point(angle) for angle in angles]
        for i in range(4):
            self.play(AddTextLetterByLetter(texts[i + 3], run_time=text_duration * 1.5))
            self.wait(0.5)
            self.play(FadeIn(dp_group[i].set_z_index(1), shift=UP * 2.5, scale=0.3))
            self.add(dp_av_group[i])
            self.play(
                ManualDoublePendulumAnimation(dp_group[i], angles[i], 2, smooth)
            )
            dp_copy = dp_group[i].copy()
            self.play(dp_copy.animate.scale((table_dps.rod_length / dp_copy.length_1) * 4).move_to(locations[i]),
                                            run_time=1.5)
            self.play(ShrinkToCenter(dp_copy, rate_func=rush_from, run_time=1.5))
            self.wait(0.5)

        self.play(AnimationGroup(
            FadeOut(texts[3], texts[4], texts[5], texts[6], *dp_group, *dp_av_group,
                    *sweep_anim.get_copies_of_sweeper(), shift=6 * RIGHT, run_time=2),
            AddTextLetterByLetter(texts[7], run_time=text_duration * 2),
            lag_ratio=0.5
        ))
        self.wait(3)
        """
        table_title = Text(
            "all possible initial positions",
            fill_color=LIGHT_BROWN,
            font="Montserrat Medium",
        ).scale(0.375).next_to(cs.bg_rectangle, UP, buff=0.0125)
        """
        self.play(ReplacementTransform(texts[7][48:75].copy(), table_title, run_time=2),
                  FadeOut(texts[7]))
        center_group.add(table_title)
        self.play(AddTextLetterByLetter(texts[8], run_time=text_duration))
        self.wait(2)
        self.play(AnimationGroup(
            FadeOut(texts[8], run_time=1.5),
            Move(center_group, LEFT * 2.95, run_time=1.5),
            lag_ratio=0.5
        ))
        """
        center_group = Group(cs, quad_pixel_visual, table_title).move_to(LEFT * 2.985)

        # INSETS
        outgrowth_ur = PixelInset(
            quad_pixel_visual,
            (60, 140),
            (70, 120),
            5.5,
            5.5 * (5 / 8),
            2 * UP + 4.17 * RIGHT,
            [UL, DL],
            16, 16,
            include_return_cs=True,
            line_to_number_buff=0.15,
            inset_color=DARK_BROWN,
            inset_stroke_width=1.5,
            # labeled_values_for_y_override=np.arange(80, 121, 10).tolist()
        )
        island_dr = PixelInset(
            outgrowth_ur.inset_pixel_visual,
            (98, 104),
            (113, 117),
            5.5,
            5.5 * (2 / 3),
            1.87 * DOWN + 4.17 * RIGHT,
            [UL, UR],
            16, 16,
            include_return_cs=True,
            line_to_number_buff=0.15,
            inset_color=DARK_BROWN,
            inset_stroke_width=1.5,
            labeled_values_for_x_override=np.arange(98, 105, 1).tolist(),
            labeled_values_for_y_override=np.arange(113, 118, 1).tolist(),
        )
        """
        FadeIn(outgrowth_ur, scene=self)
        self.wait(0.5)
        FadeIn(island_dr, scene=self)
        self.wait(3)
        self.play(FadeIn(texts[9].set_z_index(11), scale=0.3, shift=3 * UP, run_time=2))
        self.wait(2)
        self.play(FadeOut(texts[9]))
        self.add(center_group, outgrowth_ur, island_dr)
        self.wait()
        """
        self.add(center_group, outgrowth_ur, island_dr)
        # self.wait()
        duration_of_pixel_vis = 520  # 520

        animation1 = PixelVisualizationAnimation(quad_pixel_visual, duration_of_pixel_vis, "anim1", True)
        animation2 = PixelVisualizationAnimation(outgrowth_ur.inset_pixel_visual, duration_of_pixel_vis, "anim2", True)
        animation3 = PixelVisualizationAnimation(island_dr.inset_pixel_visual, duration_of_pixel_vis, "anim3", True)
        self.play(AnimationGroup(animation1, animation2, animation3))

        self.wait()

    @ignore
    def testingdrive(self):
        cs = main_cs.copy()
        cs.remove(cs.bg_rectangle)
        table_count = 32
        quad_pixel_visual = CrispPixelStaticVisuals(cs, TCF.divide_by_quadrant).set_z_index(-1)
        self.add(quad_pixel_visual)
        anim = PixelVisualizationAnimation(quad_pixel_visual, 20, "example_anim")
        anim.interpolate(1163 / 1200)


class Scene7(ComplexScene):
    run = ComplexScene.run
    skip = ComplexScene.skip
    ignore = ComplexScene.ignore

    def setup(self):
        pass

    def construct(self):
        self.add_background()
        self.play_subscenes()

    @staticmethod
    def create_vert_color_tracker():
        return ColorTracker(
            RAINBOW,
            30,
            5,
            False,
            0.8,
            pixel_length / SCENE_PIXELS,
            color_rate_func=linear
        ).shift(6 * RIGHT)

    @staticmethod
    def get_new_rainbow():
        return RAINBOW[:1] + [interpolate_color(RAINBOW[0], RAINBOW[1], 0.5)] + RAINBOW[1:]

    @ignore
    def scene7_1(self):
        cs7_1 = get_standard_cs().set_z_index(2)
        cs7_2 = get_standard_cs((99, 103), (113, 117)).set_z_index(2)
        cs7_2_destination = cs7_2.copy().scale(6 / 7).move_to(RIGHT * config.frame_width / 4)
        cs7_2_bg_rect = cs7_2.get_background_rectangle()
        cs7_2_destination_bg_rect = cs7_2_destination.get_background_rectangle()

        cs_bg_rect = cs7_1.get_background_rectangle()
        table_title = Text(
            "ALL POSSIBLE INITIAL POSITIONS",
            fill_color=AMBER_ORANGE,
            font="Montserrat Medium",
        ).scale(0.35).next_to(cs_bg_rect, UP, buff=0.05)
        pixel_visual = CrispPixelStaticVisuals(cs7_1, TCF.torus_smooth_gradient)

        stroke_tracker = ValueTracker(1)
        rect_on_axes = cs7_1.create_rect_on_axes(
            cs7_2.dyn_x_range, cs7_2.dyn_y_range).set_stroke(DARK_BROWN, stroke_tracker.get_value()).set_z_index(1)

        def update_rect_on_axes(rect):
            new_rect = cs7_1.create_rect_on_axes(
                cs7_2.dyn_x_range, cs7_2.dyn_y_range).set_stroke(
                DARK_BROWN, stroke_tracker.get_value()
            ).set_z_index(1)
            rect.become(new_rect)

        self.play(FadeIn(cs7_1, table_title, pixel_visual, scale=0, rate_func=smootherstep))
        self.play(AnimationGroup(
            FocusOn(rect_on_axes.get_center()),
            Create(rect_on_axes),
            lag_ratio=0.75
        ))
        rect_on_axes.add_updater(update_rect_on_axes)
        self.play(ZoomPixelVisuals(
            pixel_visual,
            cs7_2.dyn_x_range,
            cs7_2.dyn_y_range,
            10, smoothstep
        ), FadeOut(table_title, run_time=4),
            ApplyMethod(stroke_tracker.set_value, 4, run_time=10, rate_func=rush_into)
        )
        rect_on_axes.clear_updaters()
        self.play(FadeOut(rect_on_axes, run_time=1))
        # self.add(cs7_2)
        # self.play(Group(cs7_1, pixel_visual).animate.scale(6/7).move_to(LEFT * config.frame_width / 4),
        #          Transform(cs7_2, cs7_2_destination, path_arc=-PI/6),
        #          Transform(cs7_2_bg_rect, cs7_2_destination_bg_rect, path_arc=-PI/6),
        #           run_time=2
        #           )
        # cs7_1.x_length *= 6 / 7
        # cs7_1.y_length *= 6 / 7
        # table_of_dps = TableOfDoublePendulums(cs7_2, 40, 40)  # R: (40, 40)
        # pixel_visual.replace(CrispPixelStaticVisuals(cs7_1, TCF.torus_smooth_gradient))
        # self.play(FadeIn(table_of_dps, scale=0.25, run_time=2, rate_func=smoothererstep))
        # self.play(PixelVisualizationAnimation(
        #     pixel_visual, 60, "scene7_1_", False
        # ),
        #     ReleaseTableOfDoublePendulums(table_of_dps, 60))

    @ignore
    def scene7_2_flip_show(self):
        main_dp = DoublePendulum((0, 0))
        dp = main_dp.create_double_pendulum(1.8, 1.8).shift(UP)
        av = dp.create_angle_visualizer()
        av.angle_1_value.set_z_index(1)
        av.angle_2_value.set_z_index(1)
        self.play(FadeIn(dp, shift=UP * 4, rate_func=anticipate))
        self.add(av)
        main_dp.init_angle_1 = 105
        main_dp.init_angle_2 = 152
        self.play(ManualDoublePendulumAnimation(dp, (main_dp.init_angle_1, 0), 1))
        self.play(ManualDoublePendulumAnimation(dp, (main_dp.init_angle_1, main_dp.init_angle_2), 1))
        self.play(IndicateFlip(main_dp, self, av, 40))  # R: 40

    @ignore
    def scene7_3_colortracker_intro(self):
        cs = get_standard_cs().set_z_index(2)
        cs_bg_rect = cs.get_background_rectangle()

        pixel_num = 20  # R: 40

        table = TableOfDoublePendulums(cs, pixel_num, pixel_num).set_z_index(1)
        table2 = TableOfDoublePendulums(cs, pixel_num, pixel_num,
                                        GOLD_C, BLUE_D
                                        ).set_z_index(1)
        table_title = Text(
            "ALL POSSIBLE INITIAL POSITIONS",
            fill_color=AMBER_ORANGE,
            font="Montserrat Medium",
        ).scale(0.35).next_to(cs_bg_rect, UP, buff=0.05)
        color_tracker = self.create_vert_color_tracker()
        flip_visual = BlockyPixels(
            cs,
            pixel_num,
            pixel_num,
            color_func_index=6
        )
        self.next_section(skip_animations=True)
        self.play(FadeIn(VGroup(table, cs, cs_bg_rect, table_title), scale=0, shift=UP * 3, rate_func=smootherstep))
        self.play(
            FadeIn(flip_visual, run_time=2),
            ReplacementTransform(table, table2, run_time=2)
        )
        self.play(FadeIn(color_tracker, shift=3 * LEFT, rate_func=smootherstep))
        self.play(RunColorTracker(color_tracker, 2, 5 / color_tracker.elapsed_time, smoothererstep))
        self.play(RunColorTracker(color_tracker, 2, 15 / color_tracker.elapsed_time, smoothererstep))
        self.play(RunColorTracker(color_tracker, 2, 0, smoothererstep))
        self.next_section(skip_animations=False)
        self.play(FlipVisualization(
            flip_visual,
            color_tracker,
            table2,
            color_tracker.elapsed_time,
            target_flip_number=4
        ))

    @ignore  # MAX_GB = 3.5
    def scene7_4_high_quality(self):
        cs = get_standard_cs().set_z_index(2)
        cs_bg_rect = cs.get_background_rectangle().set_z_index(-5)
        table_title = Text(
            "ALL POSSIBLE INITIAL POSITIONS",
            fill_color=AMBER_ORANGE,
            font="Montserrat Medium",
        ).scale(0.35).next_to(cs_bg_rect, UP, buff=0.05)
        color_tracker = self.create_vert_color_tracker().set_z_index(1)
        flip_visual = CrispFlipStaticVisuals(cs).set_z_index(0)

        label_kwargs = {"color": AMBER_ORANGE, "label_color": AMBER_ORANGE, "stroke_width": 4}
        tex_kwargs = {"tex_template": get_font_for_tex("Montserrat Medium"), "color": AMBER_ORANGE}
        cs_width = MeasureLabel(
            label=Tex("2000 px", font_size=24, **tex_kwargs),
            start=cs_bg_rect.get_corner(UL),
            end=cs_bg_rect.get_corner(UR),
            **label_kwargs
        ).next_to(cs_bg_rect.get_corner(UP), DOWN, buff=SMALL_BUFF).set_z_index(2)
        cs_height = MeasureLabel(
            label=Tex("2000 px", font_size=24, **tex_kwargs),
            start=cs_bg_rect.get_corner(UL),
            end=cs_bg_rect.get_corner(DL),
            **label_kwargs
        ).next_to(cs_bg_rect.get_corner(LEFT), RIGHT, buff=SMALL_BUFF).set_z_index(2)
        down_arrow = Arrow(
            start=color_tracker.color_basis.get_top(),
            end=color_tracker.color_basis.get_bottom(),
            buff=0.1,
            stroke_width=20,
            color=WHITE
        ).set_opacity(0.5)

        self.add(cs_bg_rect, cs, table_title, color_tracker)
        self.wait()
        self.play(FadeIn(flip_visual, shift=LEFT * 11, rate_func=slow_into))
        # self.play(AnimationGroup(
        #     FadeIn(cs_width, shift=UP * cs.bg_rectangle.height, rate_func=rush_from),
        #     FadeIn(cs_height, shift=LEFT * cs.bg_rectangle.width, rate_func=rush_from),
        #     lag_ratio=0.5
        # ))
        # self.play(AnimationGroup(
        #     FadeOut(cs_height, shift=RIGHT * cs.bg_rectangle.width, rate_func=rush_into),
        #     FadeOut(cs_width, shift=DOWN * cs.bg_rectangle.height, rate_func=rush_into),
        #     lag_ratio=0.5
        # ))
        # self.wait()
        # self.play(AnimationGroup(
        #     GrowArrow(down_arrow, run_time=3),
        #     TransformRateFuncOfColorTracker(color_tracker, slow_into, 3),
        #     lag_ratio=0.25
        # ))
        # self.play(FadeOut(down_arrow))
        # self.play(FlipVisualization(
        #     flip_visual,
        #     color_tracker,
        #     duration=color_tracker.elapsed_time,
        #     use_existing_dat='scene7_4_slow_into',
        #     skip_processing=False
        # ))

    @ignore
    def scene7_4_high_quality_testing(self):
        cs = get_standard_cs().set_z_index(2)
        cs_bg_rect = cs.get_background_rectangle()
        table_title = Text(
            "ALL POSSIBLE INITIAL POSITIONS",
            fill_color=AMBER_ORANGE,
            font="Montserrat Medium",
        ).scale(0.35).next_to(cs_bg_rect, UP, buff=0.05)
        color_tracker = self.create_vert_color_tracker()
        color_tracker.color_rate_func = slow_into
        flip_visual = CrispFlipStaticVisuals(cs)
        self.add(cs, table_title, color_tracker, flip_visual)
        self.play(FlipVisualization(
            flip_visual,
            color_tracker,
            duration=color_tracker.elapsed_time,
            use_existing_dat='scene7_4_testing_',
            skip_processing=False,
            target_flip_number=1,
        ))
        self.wait()

    @run
    def multiple_flips_viz2(self):
        color_tracker = ColorTracker(
            RAINBOW,
            70,  # FR: 70
            10,
            False,
            0.5,
            (pixel_length / SCENE_PIXELS) + 0.25,
            color_rate_func=steep_slow_into,  # FR: steep_slow_into
            arrow_length_level=4,
        ).shift(6.5 * RIGHT).set_z_index(-7)

        use_existing_list = ["viz1st_", "viz_2nd_", "viz_3rd_", "viz_4th_", "viz_5th_", "viz_6th_"]
        flips_visualize_kwargs = {
            "color_tracker": color_tracker,
            "table": None,
            "duration": color_tracker.elapsed_time,
            "skip_processing": False,
        }
        group = Group()
        flip_anims = []
        for i in range(1, 7):
            cs = DynamicAxes(
                x_range=(-180, 180),
                y_range=(-180, 180),
                x_length=3.6,
                y_length=3.6,
                x_is_in_degrees=True,
                y_is_in_degrees=True,
                font_size_x=15,
                font_size_y=15,
                include_zero_lines=False,
                use_constant_tick_length=True,
                x_line_to_number_buff=0.1225,
                y_line_to_number_buff=0.1225,
                labeled_values_for_x_override=[-180, -90, 0, 90, 180],
                labeled_values_for_y_override=[-180, -90, 0, 90, 180],
                tick_length=0.008
            ).set_z_index(5)
            cs_bg_rect = cs.get_background_rectangle().set_z_index(-5)
            suffix = ["st", "nd", "rd", "th", "th", "th"][i - 1]
            title = Text(
                f"ALL POSSIBLE INITIAL POSITIONS ({i}{suffix} flip)",
                fill_color=AMBER_ORANGE,
                font="Montserrat",
                weight=MEDIUM
            ).scale(0.2).next_to(cs_bg_rect, UP, buff=0.018)
            flip_visual = CrispFlipStaticVisuals(cs)
            group.add(Group(cs, title, flip_visual))

            flip_anims.append(FlipVisualization(
                flip_visual,
                target_flip_number=i,
                use_existing_dat=use_existing_list[i - 1],
                **flips_visualize_kwargs
            ))

        group.arrange_in_grid(2, 3, buff=SMALL_BUFF / 2).shift(LEFT * 0.75)

        self.add(color_tracker, group)
        self.next_section(skip_animations=True)
        self.wait()
        self.play(AnimationGroup(*flip_anims))
        pixel_visual_group = Group()
        for gr in group:
            pixel_visual_group.add(gr.submobjects[2])
        pixel_visual_group.arrange_in_grid(2, 3, buff=SMALL_BUFF / 2).move_to(ORIGIN)
        self.next_section(skip_animations=False)
        self.remove(*self.mobjects)
        self.add(pixel_visual_group)
        self.wait()

    @ignore
    def scene7_5_more_gradients_and_insets(self):
        cs = get_standard_cs().set_z_index(2)
        cs_bg_rect = cs.get_background_rectangle()
        table_title = Tex(
            "ALL POSSIBLE ", "INITIAL ", "POSITIONS",
            fill_color=AMBER_ORANGE,
            font_size=20,
            tex_template=get_font_for_tex("Montserrat Medium")
        ).next_to(cs_bg_rect, UP, buff=0.05)
        flip_vis_inset = CrispFlipStaticVisuals(cs)
        orig_group = Group(flip_vis_inset, cs, table_title)

        self.add(orig_group)
        self.play(orig_group.animate.scale(
            scale_value, about_point=ORIGIN
        ).shift(shift_value))
        cs.x_length *= scale_value
        cs.y_length *= scale_value
        cs_bg_rect.scale(scale_value).shift(shift_value)

        flip_vis_inset.replace(CrispFlipStaticVisuals(cs).shift(shift_value))
        insets = get_four_insets(flip_vis_inset, True)
        insets.append(InsetScaffold(
            insets[2].inset_image,
            (-34, -29),
            (-153, -148),
            3.8,
            3.8,
            2 * DOWN + 5.1 * RIGHT,
            [UL, DL],
            include_image=True
        ))

        color_tracker = ColorTracker(
            self.get_new_rainbow(),
            70,
            10,
            True,
            0.5,
            cs_bg_rect.length_over_dim(0),
            color_rate_func=steep_slow_into,
            radius=0.125
        ).next_to(cs_bg_rect, DOWN, 0.375).set_z_index(20)

        for i, inset in enumerate(insets):
            for submob in inset.submobjects[1:]:
                submob.set_z_index(3)
            if i <= 3:
                self.play(FadeIn(inset, run_time=0.5, scale=0.3, shift=LEFT * 4))
        self.play(FadeIn(color_tracker, shift=RIGHT * 7, run_time=1))
        use_existing_list = ["scene7_5_1st_", "scene7_5_2nd_", "scene7_5_3rd_", "scene7_5_4th_",
                             "scene7_5_5th_", "scene7_5_sub_"]
        flips_visualize_kwargs = {
            "color_tracker": color_tracker,
            "table": None,
            "duration": color_tracker.elapsed_time,
            "skip_processing": False
        }
        flip_anims = []
        for inset, ued in zip([flip_vis_inset, *insets[:-2]], use_existing_list[:-2]):
            flip_anims.append(
                FlipVisualization(
                    inset[0],
                    use_existing_dat=ued,
                    **flips_visualize_kwargs
                )
            )
        fourth_island_anim = FlipVisualization(
            insets[-2][0],
            use_existing_dat=use_existing_list[-2],
            **flips_visualize_kwargs
        )
        fourth_island_anim.flip_anim.extra_animate(55, 2, shift=8 * RIGHT, more_mobjects=insets[-2])
        flip_anims.append(fourth_island_anim)

        self.add(insets[-1].shift(4 * DOWN))
        sub_island_anim = FlipVisualization(
            insets[-1][0],
            use_existing_dat=use_existing_list[-1],
            **flips_visualize_kwargs
        )
        sub_island_anim.flip_anim.extra_animate(56, 1, shift=4 * UP, more_mobjects=insets[-1])
        flip_anims.append(sub_island_anim)

        self.play(AnimationGroup(*flip_anims))

        vert_color_tracker = ColorTracker(
            self.get_new_rainbow(),
            color_tracker.elapsed_time,
            10,
            False,
            0.8,
            pixel_length / SCENE_PIXELS,
            color_rate_func=color_tracker.color_rate_func
        ).shift(6 * RIGHT).set_z_index(2)
        vert_color_tracker.remove(vert_color_tracker.arrow)

        self.play(AnimationGroup(
            FadeOut(color_tracker, shift=LEFT * 5, scale=0.3),
            FadeOut(insets[4], shift=RIGHT * 8.5, scale=0.3),
            FadeOut(insets[3], shift=RIGHT * 8.5, scale=0.3),
            FadeOut(insets[2], shift=RIGHT * 8.5, scale=0.3),
            FadeOut(insets[1], shift=RIGHT * 8.5, scale=0.3),
            FadeOut(insets[0], shift=RIGHT * 8.5, scale=0.3),
            orig_group.animate.scale(1 / scale_value, about_point=cs_bg_rect.get_center()).shift(-shift_value),
            FadeIn(vert_color_tracker, shift=LEFT * 2),
            lag_ratio=0.1,
            run_time=2
        ))

    @ignore  # 30 fps
    def scene7_8_zooming_checking_fractal2_just_flip_visuals(self):
        cs = get_standard_cs().set_z_index(2)
        cs_bg_rect = cs.get_background_rectangle()
        table_title = Tex(
            "ALL POSSIBLE ", "INITIAL ", "POSITIONS",
            fill_color=AMBER_ORANGE,
            font_size=20,
            tex_template=get_font_for_tex("Montserrat Medium")
        ).next_to(cs_bg_rect, UP, buff=0.05)
        color_tracker = ColorTracker(
            self.get_new_rainbow(),
            70,  # R: 70
            10,
            False,
            0.8,
            pixel_length / SCENE_PIXELS,
            color_rate_func=steep_slow_into,
        ).shift(6 * RIGHT).set_z_index(2)
        color_tracker.remove(color_tracker.arrow)

        flip_visual = CrispFlipStaticVisuals(
            cs
        ).turn_to_last_image(color_tracker, use_existing_dat="scene7_8_ending_image_", skip_processing=True)
        zoom_flip_params = [
            {  # index 0
                "new_x_range": (103.2744, 103.2745),
                "new_y_range": (116.3204, 116.3205),
                "duration": 42,
                "skip_processing": True
            }
        ]
        use_existing_list = [
            "scene7_8_0th_", "scene7_8_1st_",
            "scene7_8_2nd_",
        ]

        self.add(cs, color_tracker, flip_visual, table_title)
        turn_animation_into_updater(FadeOut(table_title, run_time=4))
        for i, params in enumerate(zoom_flip_params):
            self.play(ZoomFlipVisuals(
                flip_visual,
                zoom_rate_func=linear,
                use_existing_dat=use_existing_list[i],
                **params
            ))

    @ignore
    def scene7_8_testing(self):
        cs = get_standard_cs(
            (103.2744, 103.2745),
            (116.3204, 116.3205)
        ).set_z_index(2)
        cs_bg_rect = cs.get_background_rectangle()
        table_title = Tex(
            "ALL POSSIBLE ", "INITIAL ", "POSITIONS",
            fill_color=AMBER_ORANGE,
            font_size=20,
            tex_template=get_font_for_tex("Montserrat Medium")
        ).next_to(cs_bg_rect, UP, buff=0.05)
        color_tracker = ColorTracker(
            self.get_new_rainbow(),
            70,  # R: 70
            10,
            False,
            0.8,
            pixel_length / SCENE_PIXELS,
            color_rate_func=steep_slow_into,
        ).shift(6 * RIGHT).set_z_index(2)
        color_tracker.remove(color_tracker.arrow)

        flip_visual = CrispFlipStaticVisuals(
            cs
        ).turn_to_last_image(color_tracker, use_existing_dat="scene7_8_testing_", skip_processing=False)

        self.add(cs, table_title, color_tracker, flip_visual)
        self.wait()


if __name__ == "__main__":

    scenes = [
        "Scene4",  # 0
        "Scene5",  # 1
        "Scene6",  # 2
        "Scene7",  # 3
    ]


    @timer
    def run_manim_scene(index):
        scene_to_render = scenes[index]
        if not use_background:
            print(' '.join(['\nmanim', 'chaos_theory.py', scene_to_render, '-t']))
            subprocess.run(['manim', 'chaos_theory.py', scene_to_render, '-t'])
        else:
            print(' '.join(['\nmanim', 'chaos_theory.py', scene_to_render]))
            subprocess.run(['manim', 'chaos_theory.py', scene_to_render])


    run_manim_scene(3)

    print(f"leftover memmaps: (should return nothing)")
    delete_memmap_files(
        'all_solutions',
        'extra',
        'white_default',
        'reshape_flips',
        'unreshape_zoom'
    )
