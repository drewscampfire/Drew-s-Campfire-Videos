# Drew's Campfire Code

This is the source code for the animations on [Drew's Campfire](https://www.youtube.com/@drewscampfire), a channel dedicated to making clear and novel math and science visualizations.

This project goes beyond standard Manim usage by implementing Manim-like functionality in Blender (a traditionally 3D animation tool), enabling programmatic 3D animation in the same spirit that Manim brings to 2D.

---

## Architecture

```
Drew-s-Campfire-Videos/
├── src/                              # Reusable code libraries
│   ├── drawblend.py                  # My Blender scripting tool
│   ├── custom_manim.py               # Supplementary Manim functionality
│   └── debug_utils.py                
│
└── videos/
    └── double_pendulum/              
        ├── blender_files/            
        ├── manim_files/              # my Manim scenes for the double pendulum videos are here
        ├── chaos_theory_base_classes.py
        └── tensor_rate_functions.py
```

### `src/drawblend.py` - Scripting Tool for Blender

`drawblend.py` provides two classes, `BlendScene` and `Blobject`, that together expose a Python API for driving Blender programmatically, essentially bringing Manim-style programmatic animation to Blender's 3D environment. (Note: this requires manual setup for each scene in Blender).

**`BlendScene`** is the base class for Blender scenes, mirroring the role that `Scene` plays in Manim. It provides:

- `delete_keyframes_in_range` - clears all object and material keyframes in a given frame range, with the option to exclude specific collections. I usually used this to non-destructively recompute animation data during iterative development.
- `compute_lagged_frame_start_and_end` - given a total frame range and a lag ratio, computes the per-object start and end frames for a staggered group animation, equivalent to Manim's `lag_ratio` in `AnimationGroup`. (I just really like the result of using AnimationGroup in Manim, so I recreated it in Blender).

**`Blobject`** (Blender object) wraps a `bpy.data.objects` entry and exposes a chainable API for keyframing any object property. (Each method returns `self` so calls can be chained.) I also intentionally designed method names to be as close to Manim's as possible:

- `fade_in` / `fade_out` - keyframes the object's material alpha from 0 to 1 or vice versa over a given duration
- `move_to` / `shift_by` - keyframes the object's location
- `animate_geom_modifier_input` -- keyframes a named input on a Geometry Nodes modifier, useful for animating mesh properties
- `slide_material_attribute` - keyframes any shader node input between two values over a frame range, useful for animating material properties like color and roughness

### `src/custom_manim.py` - Custom Mobjects, Animation Classes and Scene Functionality

This is my supplementary library for Manim CE. (I owe a lot to the old Manim Discord server, which unfortunately has been compromised by a bad actor.) Three systems I've developed here (among others) are:

#### `ComplexScene` - Better Organization of Subscenes

`ComplexScene` extends Manim's `Scene` with a subscene system that splits a single scene class into independently renderable methods. Each method is registered using one of three decorators:

- **`@run`** - marks a ComplexScene method to be rendered. It is executed in declaration order when `play_subscenes()` is called in the `construct` method.
- **`@skip`** - identical to `@run` but passes `skip_animations=True`, causing the section to advance to its finished state instantly. Useful for skipping already-finished sections during development without removing them.
- **`@ignore`** - excludes a method entirely from `play_subscenes()`. Useful for keeping work-in-progress or archived subscenes in the file without rendering them.

Here is what it looks like in practice:

```python
class ExampleScene(ComplexScene):
    def construct(self):
        self.play_subscenes()

    @run
    def subscene1(self): ...

    @skip
    def subscene2(self): ...

    @ignore
    def subscene3(self): ...
```


#### `play_anims` - Precise Multi-Track Animation 

Manim is notoriously inflexible when it comes to starting a new animation when at least one other animation is already running. `play_anims` solves this like so:

```python
# the keys are the start times in seconds and the values are animations or lists of animations; self is a ComplexScene subclass

play_anims(self, {
    0:  Create(progress_bar, run_time=12, rate_func=linear),
    1:  FadeIn(title, run_time=2),
    4:  [
        Create(graph, run_time=8),
        Write(equation, run_time=1),
    ],
    9:  FadeOut(title, run_time=1),
})
```

Animations are converted to updaters and run in parallel with precise timing. 

#### `DynamicAxes` - A More Powerful `Axes` Mobject

This extends Manim's built-in `Axes`:

- **Automatic tick spacing** — tick density and axes labels are derived from the range of the y- or x-axis, so it works consistently whether the range is `(-180, 180)` or `(0.001, 0.002)`
- **Zero-crossing dashed lines** — when `include_zero_lines` is set to True, dashed reference lines at x=0 and y=0 are added automatically, but only when the range actually straddles zero
- **Label overrides** — lets you specify exactly which values in the axes get tick labels

---

## License

The contents of this repository are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).