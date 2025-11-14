import numpy as np
import os
import math
from tqdm import tqdm
import time  # Added for potential timing
from typing import Callable, Sequence, Tuple  # Added Tuple


try:
    from manim.constants import DEGREES, PI
    from manim.utils.rate_functions import linear # Assuming only linear is needed based on __main__


    print("Successfully imported constants and rate functions from Manim.")
except ImportError as e:
    print(f"Failed to import from Manim: {e}")
    print("Falling back to manual definitions.")
    PI = np.pi
    DEGREES = np.pi / 180.0

    from functools import wraps

    def unit_interval(function):
        @wraps(function)
        def wrapper(t, *args, **kwargs):
            t = np.clip(t, 0, 1)
            return function(t, *args, **kwargs)
        return wrapper

    @unit_interval
    def linear(t):
        return t

    # Add smooth, smoothstep etc. here ONLY IF get_zoom_range_values uses them
    # and the Manim import might fail. Since __main__ uses linear, only define that for fallback.

# --- Constants ---
fps = 30

def center_points_linspace(start: float, stop: float, num: int, retstep: bool = False) -> np.ndarray:
    d = (stop - start) / (2 * num)

    return np.linspace(start + d, stop - d, num, retstep=retstep)

def reverse_rate_func(rate_func: Callable[[float], float]) -> Callable[[float], float]:
    def reversed_func(t: float) -> float:
        return 1 - rate_func(1 - t)

    return reversed_func

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

    # Clamp scalar to bounds
    scalar = max(lower_bound, min(scalar, upper_bound))

    span = upper_bound - lower_bound
    if span < tolerance:
        return lower_bound  # or upper_bound, they're essentially the same

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

class PixelGridAnglesComputation:
    def __init__(
            self,
            angle_1_domain: tuple[float, float],
            angle_2_domain: tuple[float, float],
            width_pixel_num: int,
            height_pixel_num: int,
    ):
        self.angle_1_values, self.x_step = self.get_angle_values(angle_1_domain, width_pixel_num, retstep=True)
        self.angle_2_values, self.y_step = self.get_angle_values(angle_2_domain, height_pixel_num, retstep=True)
        self.width_pixel_num = width_pixel_num
        self.height_pixel_num = height_pixel_num
        self.num_pixels_per_frame = width_pixel_num * height_pixel_num
        self.angle_pairs = self.get_angle_pairs_from_two_lists(self.angle_1_values, self.angle_2_values)

    @staticmethod
    def get_angle_values(angle_domain: tuple[float, float], num: int, retstep: bool = False) -> np.ndarray | Tuple[np.ndarray, float]:
        return center_points_linspace(angle_domain[0], angle_domain[1], num=num, retstep=retstep)

    @staticmethod
    def get_angle_pairs_from_two_lists(angle_1_values: np.ndarray, angle_2_values: np.ndarray) -> np.ndarray:
        grid_x, grid_y = np.meshgrid(angle_1_values, angle_2_values[::-1])

        return np.dstack([grid_x, grid_y]).reshape(-1, 2)

    def get_batched_angle_pairs(
            self,
            old_x_range: tuple[float, float],
            new_x_range: tuple[float, float],
            old_y_range: tuple[float, float],
            new_y_range: tuple[float, float],
            num_of_frames: int,
            zoom_rate_func: Callable,
            output_filename: str, # Explicit filename parameter
            output_dir: str = "/data"  # Default to /data as requested
    ) -> np.memmap:
        """Generates and saves angle pairs for a zoom animation to a specified file."""
        print(f"Running get_batched_angle_pairs for {num_of_frames} frames.")
        print(f"Targeting {self.width_pixel_num}x{self.height_pixel_num} pixels per frame.")
        print(f"Output directory: {output_dir}")
        print(f"Output filename: {output_filename}")

        zoom_x_values = get_zoom_range_values(
            old_x_range, new_x_range, num_of_frames, zoom_rate_func
        )
        zoom_y_values = get_zoom_range_values(
            old_y_range, new_y_range, num_of_frames, zoom_rate_func
        )

        # --- File Path Handling ---
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                print(f"Created directory: {output_dir}")
            except OSError as e:
                print(f"Error creating directory {output_dir}: {e}. Please ensure permissions.")
                raise

        file_path = os.path.join(output_dir, output_filename) # Use the provided filename

        # --- Memmap Setup ---
        file_exists = os.path.exists(file_path)
        total_angle_pairs = num_of_frames * self.num_pixels_per_frame
        expected_shape = (total_angle_pairs, 2)
        expected_dtype = np.float64
        mode = 'w+' # Default to create/overwrite

        if file_exists:
            try:
                existing_mmap = np.memmap(file_path, dtype=expected_dtype, mode='r')
                if existing_mmap.shape == expected_shape:
                    print(f"File {file_path} exists with correct shape. Opening in 'r+' mode.")
                    mode = 'r+'
                    del existing_mmap
                else:
                    print(f"Warning: Existing file {file_path} has shape {existing_mmap.shape}, expected {expected_shape}. Recreating with 'w+'.")
                    del existing_mmap
            except Exception as e:
                print(f"Error checking existing file {file_path}: {e}. Recreating with 'w+'.")

        batch_angles_pairs_mmap = np.memmap(
            file_path,
            dtype=expected_dtype,
            mode=mode,
            shape=expected_shape
        )

        for i in tqdm(
                range(num_of_frames),
                total=num_of_frames,
                desc="Processing Batched Angle Pairs"
        ):
            x_r, y_r = zoom_x_values[i], zoom_y_values[i]
            current_angle_1_values = self.get_angle_values(tuple(x_r), self.width_pixel_num)
            current_angle_2_values = self.get_angle_values(tuple(y_r), self.height_pixel_num)
            new_angle_pairs = self.get_angle_pairs_from_two_lists(
                current_angle_1_values, current_angle_2_values
            )
            start_idx = i * self.num_pixels_per_frame
            end_idx = start_idx + self.num_pixels_per_frame
            batch_angles_pairs_mmap[start_idx:end_idx] = new_angle_pairs

            if i > 0 and i % (max(1, num_of_frames // 50)) == 0:
                batch_angles_pairs_mmap.flush()

        batch_angles_pairs_mmap.flush()
        print(f"Finished populating {file_path}")
        return batch_angles_pairs_mmap

if __name__ == "__main__":
    start_time = time.time()
    print("Starting vast.ai job script (Manim imports allowed).")

    # Parameters
    cs_initial_x_range = (-180.0, 180.0)
    cs_initial_y_range = (-180.0, 180.0)
    target_x_range = (103.2744, 103.2745)
    target_y_range = (116.3204, 116.3205)
    duration_secs = 42
    num_frames = int(duration_secs * fps)

    width_px = 2000
    height_px = 2000

    output_data_dir = "/data"
    output_file = "scene7_8_batched.dat"

    # Create computation object
    computation = PixelGridAnglesComputation(
        angle_1_domain=cs_initial_x_range,
        angle_2_domain=cs_initial_y_range,
        width_pixel_num=width_px,
        height_pixel_num=height_px
    )

    print(f"Output directory set to: {output_data_dir}")
    print(f"Output filename set to: {output_file}")

    batched_data_mmap = computation.get_batched_angle_pairs(
        old_x_range=cs_initial_x_range,
        new_x_range=target_x_range,
        old_y_range=cs_initial_y_range,
        new_y_range=target_y_range,
        num_of_frames=num_frames,
        zoom_rate_func=linear,  # Using linear as determined before
        output_filename=output_file, # Pass the desired filename
        output_dir=output_data_dir
    )

    memmap_filepath = batched_data_mmap.filename
    del batched_data_mmap

    end_time = time.time()
    print(f"Job finished in {end_time - start_time:.2f} seconds.")
    print(f"Data successfully written to {memmap_filepath}")
    print("You can now transfer this file using rclone or other methods.")