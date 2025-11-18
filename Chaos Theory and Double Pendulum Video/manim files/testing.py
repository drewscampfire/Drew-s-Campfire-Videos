from __future__ import annotations

import subprocess
from manim import *
from custom_manim import *
from mydebugger import *

def get_font_for_tex(font_name: str) -> TexTemplate:
    """
    Creates a Manim TexTemplate configured to use a specific system font
    for both regular text (Tex) and mathematical content (MathTex).
    """
    tex_template = TexTemplate(
        tex_compiler="xelatex",
        output_format='.xdv',
    )
    complete_preamble = rf"""
\usepackage{{fontspec}}
\usepackage{{unicode-math}}
\setmainfont{{{font_name}}}
\setmathfont{{{font_name}}}
"""
    tex_template.add_to_preamble(complete_preamble)
    return tex_template

config.tex_template = get_font_for_tex("IBM Plex Sans")
Text.set_default(font="Montserrat Medium")


class Test4(ComplexScene):
    run = ComplexScene.run
    skip = ComplexScene.skip
    ignore = ComplexScene.ignore

    def setup(self):
        self.add_background()
        self.play_subscenes()

    @run
    def construct(self):
        # Create mobjects for comparison
        tex1 = Tex(r"1 2 3 4 5 6 7 8 9 0")
        tex2 = MathTex(r"1 2 3 4 5 6 7 8 9 0")
        numline1 = NumberLine(
            x_range=[-5, 5, 1],
            length=10,
            color=BLUE,
            include_numbers=True,
            label_constructor=Tex
        )
        numline2 = NumberLine(
            x_range=[-5, 5, 1],
            length=10,
            color=RED,
            include_numbers=True,
        )

        # Position mobjects
        tex1.move_to(UP * 3)
        tex2.move_to(UP * 2)
        numline1.move_to(UP * 0.5)
        numline2.move_to(DOWN * 1.5)

        # Create labels
        tex1_label = Text("Tex", font_size=24).next_to(tex1, LEFT)
        tex2_label = Text("MathTex", font_size=24).next_to(tex2, LEFT)
        numline1_label = Text("NumberLine + Tex", font_size=20).next_to(numline1, UP)
        numline2_label = Text("NumberLine + MathTex", font_size=20).next_to(numline2, DOWN)

        # Add everything to scene
        self.add(tex1, tex2, numline1, numline2)
        self.add(tex1_label, tex2_label, numline1_label, numline2_label)
        self.wait(2)

    @ignore
    def test_scene_2(self):

        tex_string = Tex(r"1 2 3 4 5 6 7 8 9 0", color=ORANGE)

        mathtex_string = MathTex(r"1 2 3 4 5 6 7 8 9 0", color=YELLOW)

        numline_tex = NumberLine(
            x_range=[-5, 5, 1],
            length=6,
            include_numbers=True,
            label_constructor=Tex, color=ORANGE, font_size=36)

        numline_mathtex = NumberLine(
            x_range=[-5, 5, 1],
            length=6,
            include_numbers=True,
            label_constructor=MathTex, color=YELLOW, font_size=36)

        label_tl = Tex("Tex String").scale(0.8)
        label_tr = Tex("MathTex String").scale(0.8)
        label_bl = Tex("NumberLine with Tex").scale(0.8)
        label_br = Tex("NumberLine with MathTex").scale(0.8)


        # Group each object with its label
        group_tl = VGroup(label_tl, tex_string).arrange(DOWN, buff=0.4)
        group_tr = VGroup(label_tr, mathtex_string).arrange(DOWN, buff=0.4)
        group_bl = VGroup(label_bl, numline_tex).arrange(DOWN, buff=0.4)
        group_br = VGroup(label_br, numline_mathtex).arrange(DOWN, buff=0.4)

        grid = VGroup(group_tl, group_tr, group_bl, group_br).arrange_in_grid(
            rows=2, cols=2, buff=1.5
        )


        main_title = Tex("Comparison: ", r"$\LaTeX$ (Tex)", " vs ", r"Ka$\TeX$ (MathTex)").to_edge(UP)
        main_title[1].set_color(ORANGE)
        main_title[3].set_color(YELLOW)
        self.play(Write(main_title))
        self.wait(0.5)

        self.play(
            FadeIn(group_tl, shift=RIGHT),
            FadeIn(group_bl, shift=RIGHT),
            run_time=1.5
        )
        self.wait(1)

        self.play(
            FadeIn(group_tr, shift=LEFT),
            FadeIn(group_br, shift=LEFT),
            run_time=1.5
        )
        self.wait(1.5)


        highlight_tex_str = SurroundingRectangle(tex_string[0][2], color=RED, buff=0.05)
        highlight_mathtex_str = SurroundingRectangle(mathtex_string[0][2], color=RED, buff=0.05)
        highlight_tex_line = SurroundingRectangle(numline_tex.get_number_mobject(2), color=RED, buff=0.05)
        highlight_mathtex_line = SurroundingRectangle(numline_mathtex.get_number_mobject(2), color=RED, buff=0.05)

        highlights = VGroup(
            highlight_tex_str, highlight_mathtex_str,
            highlight_tex_line, highlight_mathtex_line
        )

        summary_text = Tex(
            r"Notice the subtle font difference. \textbf{MathTex} is faster and is the standard for math.",
            font_size=32
        ).to_edge(DOWN)

        self.play(Create(highlights))
        self.play(Write(summary_text))
        self.wait(4)

    @ignore
    def test_scene_3(self):
        # self.add(NumberPlane())
        image = ImageMobject("white_cat.png").scale(0.2).shift(LEFT * 3 + UP)
        square = SurroundingRectangle(image, buff=0).set_color(BLUE).shift(DOWN)
        start = image.get_center()
        end = start + RIGHT * 6   + DOWN * 2  # moving from A to B
        square_2 = square.copy().move_to(end + DOWN).set_z_index(2)
        arc = create_arc_path(start, end, PI / 2)  # arc_angle=PI for a half-circle arc

        self.add(image, arc, square)
        self.play(MoveAlongPath(image, arc, run_time=3),
                  ReplacementTransform(square, square_2, path_arc=PI / 2, run_time=3))
        self.wait()

class Test5(Scene):
    def construct(self):
        self.wait(1)



if __name__ == "__main__":
    scenes = [
        "Test1",  # 0
        "Test2",  # 1
        "Test3",  # 2
        "Test4",  # 3
        "Test5",  # 4
    ]

    @timer
    def run_manim_scene(index):
        """
        Executes a specified Manim scene using a provided index. The function retrieves the scene
        from a predefined list, constructs the required command-line statement for rendering,
        and invokes the Manim rendering process via a subprocess. Designed for automated control
        of scene rendering in a Manim animation workflow.

        Args:
            index (int): The zero-based index of the scene to render, corresponding to the
            list of available scenes.

        Raises:
            IndexError: Raised if the provided index is out of range for the list of scenes.
            FileNotFoundError: Raised if the Manim executable or input script file is not
            found in the execution environment.
        """
        scene_to_render = scenes[index]
        print(' '.join(['\nmanim', 'testing.py', scene_to_render]))
        subprocess.run(['manim', 'testing.py', scene_to_render])

    run_manim_scene(4)
