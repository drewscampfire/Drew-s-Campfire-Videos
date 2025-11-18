from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
import numpy as np
from math import sin, cos
import math
import torch
from scipy.integrate import solve_ivp
from torchdiffeq import odeint
from typing import Callable
from tqdm import tqdm
import os
import time
import argparse

parser = argparse.ArgumentParser(
    description="Compute double pendulum flip indices for Vast.ai. Allows GPU VRAM limit specification."
)
parser.add_argument(
    '--max_gb',
    type=float,
    default=4.5,
    help="Maximum GPU VRAM in GB to use for batch size calculations (e.g., 4.5, 10.0, 22.0)."
)
cli_args = parser.parse_args() # Parse command-line arguments

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    raise ValueError("GPU not found!")

fps = 30
QUALITY_STR = 'fourk_quality'
MAX_GB = cli_args.max_gb # Original was 4.5
print(f"INFO: Configured MAX_GB (VRAM limit for batching) to: {MAX_GB} GB") # Added for confirmation
rtol = 1e-9
atol = 1e-11
DEGREES = np.pi / 180
DATA_DIR = '.'


def linear(t: float) -> float:
    return t

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"TIMER: {func.__name__}() took {end_time - start_time:.2f} seconds.")
        return result
    return wrapper

def get_indices_with_all_zeroes(array: np.memmap, chunk_size: int = 50000) -> list[int]:
    results = []
    total_cols = array.shape[1]

    for i in tqdm(range(0, total_cols, chunk_size), desc="Processing chunks"):
        end_idx = min(i + chunk_size, total_cols)
        chunk = torch.from_numpy(array[:, i:end_idx, :]).to(device)
        zero_mask = torch.all(torch.all(chunk == 0, dim=0), dim=1)
        del chunk
        zero_indices = torch.where(zero_mask)[0] + i
        del zero_mask
        results.extend(zero_indices.cpu().tolist())
        torch.cuda.empty_cache()

    return results


@dataclass
class ComputeDoublePendulumSimulation:
    angle_pair: tuple[float, float]  # angles in DEGREES
    t_span: float
    t_eval_rate_func: Callable[[float], float] = linear
    override_fps: int | None = None
    g: float = 9.81
    m1: float = 1  # mass of the 2 bobs
    m2: float = 1
    l1: float = 1  # length of the 2 rods
    l2: float = 1
    _cached_angle_values_in_rad: tuple[np.ndarray, np.ndarray] = field(default=None, init=False, repr=False)

    def hamilton_rhs(self, t1, t2, p1, p2):
        a = self.l1 * self.l2 * (self.m1 + self.m2 * sin(t1 - t2) ** 2)
        b = (p1 * p2 * sin(t1 - t2)) / a
        c = (self.m2 * (self.l2 * p1) ** 2 +
             (self.m1 + self.m2) * (self.l1 * p2) ** 2 -
             2 * self.l1 * self.l2 * self.m2 * p1 * p2 * cos(t1 - t2)) * sin(2 * (t1 - t2)) / (2 * a ** 2)

        # RHSs of the Hamiltonian-derived first-order differential equations
        rhs_t1 = (self.l2 * p1 - self.l1 * p2 * cos(t1 - t2)) / (self.l1 * a)
        rhs_t2 = (self.l1 * (self.m1 + self.m2) * p2 - self.l2 * self.m2 * p1 * cos(t1 - t2)) / (self.l2 * self.m2 * a)
        rhs_p1 = - (self.m1 + self.m2) * self.g * self.l1 * sin(t1) - b + c
        rhs_p2 = - self.m2 * self.g * self.l2 * sin(t2) + b - c

        return np.array([rhs_t1, rhs_t2, rhs_p1, rhs_p2])

    def funcs(self, t, r):
        t1, t2, p1, p2 = r
        return self.hamilton_rhs(t1, t2, p1, p2)

    def get_t_values(self):
        step = 1 / (fps if self.override_fps is None else self.override_fps)
        time_array = np.arange(0, self.t_span + 1e-12, step)
        normalized_array = time_array / self.t_span
        transformed_array = np.array([self.t_eval_rate_func(t) for t in normalized_array]) * self.t_span
        return transformed_array

    def angle_values_in_rads(self) -> tuple[np.ndarray, np.ndarray]:
        # --- Debug: Print dtypes before calling solve_ivp ---
        init_cond_np = np.array((self.angle_pair[0] * DEGREES,
                                 self.angle_pair[1] * DEGREES, 0, 0))
        t_eval_np = self.get_t_values()
        print(
            "solve_ivp call:",
            f"initial conditions dtype = {init_cond_np.dtype}",
            f"t_eval dtype = {t_eval_np.dtype}"
        )
        # --------------------------------------------------------

        ans = solve_ivp(
            self.funcs,  # function to solve
            (0, self.t_span),  # time span
            (self.angle_pair[0] * DEGREES,
             self.angle_pair[1] * DEGREES, 0, 0),  # initial conditions
            t_eval=self.get_t_values(),  # time points
            method='DOP853',  # method choice
            rtol=rtol,
            atol=atol
        )
        return ans.y[0], ans.y[1]

    @cached_property
    def get_cached_angle_values_in_rads(self):
        return self.angle_values_in_rads()

    @staticmethod
    def change_first_element(arr: np.ndarray, value: float) -> np.ndarray:
        arr[0] = value
        return arr

    @property
    def angle_1_progression_in_rads(self):
        return self.get_cached_angle_values_in_rads[0]

    @property
    def angle_2_progression_in_rads(self):
        return self.get_cached_angle_values_in_rads[1]

    @staticmethod
    def normalize_angles_in_rads(angle_progression: np.ndarray) -> np.ndarray:
        x = np.fmod(angle_progression, 2 * np.pi)
        y = np.fmod(angle_progression, np.pi)
        return 2 * y - x

    @property
    def normalized_angle_1_progression_in_rads(self):
        angles = self.normalize_angles_in_rads(self.get_cached_angle_values_in_rads[0])
        return self.change_first_element(angles, self.angle_pair[0] * DEGREES)

    @property
    def normalized_angle_2_progression_in_rads(self):
        angles = self.normalize_angles_in_rads(self.get_cached_angle_values_in_rads[1])
        return self.change_first_element(angles, self.angle_pair[1] * DEGREES)

    @property
    def normalized_angle_1_progression_in_degrees(self) -> np.ndarray:
        angles = self.normalized_angle_1_progression_in_rads / DEGREES
        return self.change_first_element(angles, self.angle_pair[0])

    @property
    def normalized_angle_2_progression_in_degrees(self):
        angles = self.normalized_angle_2_progression_in_rads / DEGREES
        return self.change_first_element(angles, self.angle_pair[1])

    def get_points_for_plotting_two_angles(self):  # unit in degrees
        arr1 = self.normalized_angle_1_progression_in_degrees
        arr2 = self.normalized_angle_2_progression_in_degrees
        return np.column_stack((arr1, arr2, np.zeros(arr1.shape[0])))

    def get_points_for_plotting_angle1_time(self):  # unit in degrees
        arr1 = self.get_t_values()
        arr2 = self.normalized_angle_1_progression_in_degrees
        return np.column_stack((arr1, arr2, np.zeros(arr1.shape[0])))

    def get_points_for_plotting_angle2_time(self):  # unit in degrees
        arr1 = self.get_t_values()
        arr2 = self.normalized_angle_2_progression_in_degrees
        return np.column_stack((arr1, arr2, np.zeros(arr1.shape[0])))


class OptimizedDoublePendulumComputation(ComputeDoublePendulumSimulation):
    def __init__(
            self,
            angle_pairs: np.ndarray,
            t_span: float,
            t_eval_rate_func: Callable[[float], float] = linear
    ):
        self.angle_pair = (90, 90)  # remove if not used
        self.angle_pairs = torch.from_numpy(angle_pairs)
        self.angle_1_values = self.angle_pairs[:, 0]
        self.angle_2_values = self.angle_pairs[:, 1]
        self.t_span = t_span
        self.t_eval_rate_func = t_eval_rate_func
        self.num_of_dps = self.angle_pairs.size(0)
        self.g = 9.81

    def hamilton_rhs(self, t1, t2, p1, p2):
        t_diff = t1 - t2
        a = (1 + torch.sin(t_diff) ** 2)
        b = (p1 * p2 * torch.sin(t_diff)) / a
        c = (p1 ** 2 + 2 * p2 ** 2 - 2 * p1 * p2 * torch.cos(t_diff)) * torch.sin(2 * t_diff) / (2 * a ** 2)
        t1_dh = (p1 - p2 * torch.cos(t_diff)) / a
        t2_dh = (2 * p2 - p1 * torch.cos(t_diff)) / a
        p1_dh = -2 * self.g * torch.sin(t1) - b + c
        p2_dh = -self.g * torch.sin(t2) + b - c
        return t1_dh, t2_dh, p1_dh, p2_dh

    def funcs(self, t, r):
        t1, t2, p1, p2 = r.unbind(dim=-1)
        return torch.stack(self.hamilton_rhs(t1, t2, p1, p2), dim=-1)

    def get_t_values(self):
        return torch.arange(0, self.t_span, 1 / fps, dtype=torch.float64)

    @timer
    def angle_values_in_rads(self) -> tuple[np.ndarray, np.ndarray]:
        print(
            f"\n----------> running {self.__class__.__name__} by simulating {len(self.angle_pairs):,} double pendulums lasting {self.t_span} seconds each")
        print(
            f"size of final return data(int_rgba): {4 * len(self.angle_pairs) * fps * self.t_span / 2 ** 20:.2f} MB\n    reminder: CPU RAM is only 16 GB")

        all_initial_conditions = torch.column_stack((
            self.angle_1_values * DEGREES,
            self.angle_2_values * DEGREES,
            torch.zeros(self.num_of_dps),
            torch.zeros(self.num_of_dps)
        ))
        n_timesteps = len(self.get_t_values())  # Number of time steps in the ODE solution
        state_dim, bytes_per_elem = 4, 8  # 4 state variables, 8 bytes per element (float64)
        max_num_of_dp_per_batch = int(MAX_GB * 1024 ** 3) // (n_timesteps * state_dim * bytes_per_elem)
        num_of_batches = math.ceil(all_initial_conditions.shape[0] / max_num_of_dp_per_batch)

        # --- Debug: Compute and print dtypes before the for-loop ---
        t_values = self.get_t_values()
        print(
            "ODEINT call:",
            f"\nall_initial_conditions dtype = {all_initial_conditions.dtype}",
            f"\nt_eval dtype = {t_values.dtype}"
        )
        # --------------------------------------------------------------------

        list_of_data_arrays = []
        for i in tqdm(range(num_of_batches), desc="Processing angle_values_in_rads"):
            if i == num_of_batches - 1:
                init_conditions = all_initial_conditions[i * max_num_of_dp_per_batch:]
            else:
                init_conditions = all_initial_conditions[i * max_num_of_dp_per_batch: (i + 1) * max_num_of_dp_per_batch]

            solution = odeint(
                self.funcs,  # function to solve
                init_conditions.to(device),  # initial conditions (tensor)
                t_values.to(device),  # time points (tensor)
                method='dopri8',
                rtol=rtol,
                atol=atol
            )[:, :, :2]
            solution = solution.cpu().numpy().astype(np.float32).transpose((2, 1, 0))
            list_of_data_arrays.append(solution)

            del solution
            torch.cuda.empty_cache()

        all_solutions = np.concatenate(list_of_data_arrays, axis=1).astype(np.float32)
        return all_solutions[0], all_solutions[1]

    @cached_property
    def get_cached_angle_values_in_rads(self):
        return self.angle_values_in_rads()

    def get_data_for_pixel_grid(self) -> np.ndarray:
        array1 = self.normalized_angle_1_progression_in_degrees
        array2 = self.normalized_angle_2_progression_in_degrees
        return np.stack((array1.T, array2.T), axis=2)

    @staticmethod
    def change_first_element(arr: np.ndarray, value: float) -> np.ndarray:
        return arr


class OptimizedForPixelGridComputation(OptimizedDoublePendulumComputation):
    def __init__(
            self,
            angle_pairs: np.ndarray,
            t_span: float,
            t_eval_rate_func: Callable[[float], float] = linear
    ):
        super().__init__(angle_pairs, t_span, t_eval_rate_func)

    @staticmethod
    def _normalize_angles_in_rads_inplace(angle_progression: torch.Tensor) -> None:
        term_fmod_pi = torch.fmod(angle_progression, torch.pi)
        angle_progression.fmod_(2 * torch.pi)
        angle_progression.mul_(-1)
        angle_progression.add_(term_fmod_pi, alpha=2)
        del term_fmod_pi

    @timer
    def flip_visuals_index_data(
            self,
            jump_threshold: float = 180,
    ) -> np.ndarray:
        # --- Revised Print Statement ---
        num_dps_input = self.angle_pairs.shape[0]
        print(f"\n----------> Running flip_visuals_index_data:")
        print(f"              Input angle pairs shape: ({num_dps_input:,}, 2)")
        print(f"              Simulation time per pendulum (if run): {self.t_span} seconds")
        # --- End Revised Print Statement ---

        file_name = os.path.join(DATA_DIR, "scene7_8_0th__flip_30fourk_quality.dat")

        try:
            index_data_mmap = np.memmap(
                file_name,
                dtype=np.int16,
                mode='r+',
                shape=(self.num_of_dps,)
            )
        except:
            raise ValueError(f"File {file_name} does not exist")

        t_values = self.get_t_values()
        n_timesteps = len(t_values)  # Number of time steps in the ODE solution
        # 4 state variables; 8 bytes per element (float64)
        max_num_of_dp_per_batch = int(MAX_GB * 1024 ** 3) // (n_timesteps * 4 * 8)
        num_of_batches = math.ceil(self.num_of_dps / max_num_of_dp_per_batch)

        print(f"\nExpected batch processing with each batch up to {max_num_of_dp_per_batch} double pendulums")

        for i in tqdm(range(num_of_batches), desc="Processing flip_visuals_index_data"):
            start_index = i * max_num_of_dp_per_batch
            end_index = min((i + 1) * max_num_of_dp_per_batch, self.num_of_dps)

            batch_data = index_data_mmap[start_index:end_index]
            if not np.any(batch_data == 0):
                print(f"\nskipping batch starting at index {start_index} (already processed)")
                continue
            else:
                print(f"\nprocessing batch starting at index {start_index}")

            init_conditions = torch.column_stack((
                self.angle_1_values[start_index:end_index] * DEGREES,
                self.angle_2_values[start_index:end_index] * DEGREES,
                torch.zeros(end_index - start_index, dtype=torch.float64),
                torch.zeros(end_index - start_index, dtype=torch.float64)
            )).to(device)

            with torch.no_grad():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

                solution_ode = odeint(
                    self.funcs,
                    init_conditions,
                    t_values.to(device),
                    method='dopri8',
                    rtol=rtol,
                    atol=atol,
                )
                end_event.record()
                torch.cuda.synchronize()
                elapsed_odeint_ms = start_event.elapsed_time(end_event)
                print(f"Odeint call for batch {i} took {elapsed_odeint_ms / 1000.0:.2f} seconds")

                del init_conditions
                torch.cuda.empty_cache()

                angles_rad_f32 = solution_ode[:, :, :2].to(torch.float32)
                del solution_ode
                torch.cuda.empty_cache()

                OptimizedForPixelGridComputation._normalize_angles_in_rads_inplace(angles_rad_f32)
                torch.cuda.empty_cache()

                normalized_angles_deg = angles_rad_f32 / DEGREES
                del angles_rad_f32
                torch.cuda.empty_cache()

                permuted_angles_deg = normalized_angles_deg.permute(1, 0, 2)
                del normalized_angles_deg
                torch.cuda.empty_cache()

                angle_diffs = torch.diff(permuted_angles_deg, dim=1)
                del permuted_angles_deg
                torch.cuda.empty_cache()

                angle_diffs.abs_()

                angle_mask_bool = angle_diffs > jump_threshold
                del angle_diffs
                torch.cuda.empty_cache()

                true_indices = angle_mask_bool.to(torch.int8).argmax(dim=1)

                all_false_mask = ~angle_mask_bool.any(dim=1)  # This line can remain as is
                del angle_mask_bool
                torch.cuda.empty_cache()

                true_indices[all_false_mask] = torch.iinfo(true_indices.dtype).max  # Max value for no flip

                indices = torch.minimum(true_indices[:, 0], true_indices[:, 1]) + 1
                del true_indices
                torch.cuda.empty_cache()

                combined_all_false = all_false_mask[:, 0] & all_false_mask[:, 1]
                indices[combined_all_false] = -1
                del all_false_mask, combined_all_false
                torch.cuda.empty_cache()

                indices_np = indices.to(torch.int16).cpu().numpy()
                del indices
                torch.cuda.empty_cache()

                index_data_mmap[start_index:end_index] = indices_np
                index_data_mmap.flush()

        return index_data_mmap


if __name__ == "__main__":
    print("Starting Double Pendulum Flip Index Computation for Vast.ai...")
    t_span_simulation = 70.0  # seconds

    path_to_batched_angle_pairs = os.path.join(DATA_DIR, "scene7_8_batched.dat")

    try:
        num_zoom_frames = 42 * fps  # from zoom_flip_params[0]["duration"]
        pixels_per_frame = 2000 * 2000  # from CrispFlipStaticVisuals -> pixel_length
        expected_rows = num_zoom_frames * pixels_per_frame

        angle_pairs_np_mmap = np.memmap(
            path_to_batched_angle_pairs,
            dtype=np.float64,
            mode='r',
            shape=(expected_rows, 2)
        )
        print(f"Successfully memmapped angle_pairs_np_mmap with shape {angle_pairs_np_mmap.shape}")

    except Exception as e:
        print(f"Error loading or shaping {path_to_batched_angle_pairs}: {e}")
        print("Please ensure the file exists and its shape is correctly estimated for memmap.")
        exit(1)

    print("Instantiating OptimizedForPixelGridComputation...")
    computation_instance = OptimizedForPixelGridComputation(
            angle_pairs_np_mmap,
            t_span_simulation
        )

    print("Calling flip_visuals_index_data...")
    output_mmap = computation_instance.flip_visuals_index_data()
    print("You can now copy this .dat file back from the vast.ai instance.")

