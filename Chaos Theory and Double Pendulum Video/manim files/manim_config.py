import torch
from manim import *

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    print(f"\nNo GPU found!\n")
    device = None
fps = 24
quality = 'high_quality'  # ['fourk_quality', 'high_quality', 'medium_quality']
MAX_GB = 3.5
config.quality = quality
config.frame_rate = fps
config.preview = True
config.disable_caching = True
config.zero_pad = 6

use_remote_instance = False
if use_remote_instance:
    dp_data_file_dir = r'/home/user/Desktop/dp_data_files' # for remote instance
    Text.set_default(font="Montserrat", weight=MEDIUM)
else:
    config.media_dir = r"C:\Users\Andrew Tres Reyes\OneDrive\Desktop\Manim Media\Chaos_Theory"  # for my system
    dp_data_file_dir = r'D:\dp_data_files'  # for my system
    Text.set_default(font="Montserrat Medium")

file_type = 'png'  # ['mp4', 'png']
if file_type == 'png':
    config.format = "png"
    config.save_pngs = True
    use_background = False
elif file_type == 'mp4':
    use_background = True
else:
    raise ValueError(f"Invalid file type: {file_type}")

FIRST_ROD_COLOR = ManimColor("#ffd77b")
SECOND_ROD_COLOR = ManimColor("#7ba3ff")
AMBER_ORANGE = ManimColor("#FFBF00")  # Amber Orange
DARK_PURPLE = ManimColor("#520380")  # Orchid Purple
anticipate = rate_functions.ease_in_out_back


if quality == 'fourk_quality':
    rtol = 1e-9
    atol = 1e-11
    pixel_length = 2000
    # chunk_size = 5500
elif quality == 'high_quality':
    rtol = 1e-7
    atol = 1e-9
    pixel_length = 1000
elif quality == 'medium_quality':
    rtol = 1e-6
    atol = 1e-8
    pixel_length = int((2000 / 2160) * config.pixel_height)
elif quality == 'low_quality':
    rtol = 1e-5
    atol = 1e-7
    pixel_length = int((2000 / 2160) * config.pixel_height)
else:
    rtol = 1e-9
    atol = 1e-11
    pixel_length = 1280


def get_font_for_tex(font_name: str) -> TexTemplate:
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


config.tex_template = get_font_for_tex("Linux Biolinum")

"""
Z-indexing rules:
# Be careful about setting the z-indices of VGroup, Group, or Mobjects with submobjects
# never use negative z_indices
# a VGroup with submobjects with different z-indices lead to unpredictable behavior; never do this
"""