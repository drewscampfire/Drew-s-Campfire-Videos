import time
from typing import Optional

import numpy as np
import torch
import os
import glob
from pydub import AudioSegment
import shutil
from tqdm import tqdm
from manim_config import device, dp_data_file_dir


# PYTHON GENERAL DEBUGGER


def inspect_shape_of_array(*args):
    """
    Prints the shape, size in bytes, and size in GB for a list of arrays.

    Accepts both numpy arrays (np.ndarray) and torch tensors.

    Args:
        *args: A list of arguments in the format:
            (array1, array2, ..., arrayN, var_name1, var_name2, ..., var_nameN),
            where each array has a corresponding name.
    """
    arrays = args[:len(args) // 2]
    var_names = args[len(args) // 2:]

    assert len(arrays) == len(var_names), "Each array must have a corresponding name"
    assert len(arrays) > 0, "At least one array and name must be provided"

    for arr, name in zip(arrays, var_names):
        print(f"\nname: {name}")
        print(f"shape: {arr.shape}")

        # Determine the total number of bytes based on the available attributes.
        if hasattr(arr, 'nbytes'):
            nbytes = arr.nbytes
        elif hasattr(arr, 'element_size') and hasattr(arr, 'numel'):
            nbytes = arr.element_size() * arr.numel()
        else:
            raise TypeError("Unsupported array type: must be a numpy.ndarray or a torch.Tensor")

        print(f"nbytes: {nbytes:,}")
        print(f"size in GB: {nbytes / (1024 ** 3):.6f} GB")


def get_time(*times):
    time_values = [*times]
    time_elapse = np.diff(np.array(time_values))
    print(f"\n")
    for i, time in enumerate(time_elapse):
        print(f"{i+2}-{i+1}: {time: .4f}")


def progress_bar(progress, total, description: str | None = None):
    """
        Prints a progress bar to visualize the progress of a task.

        Args:
            progress (int): The current progress value.
            total (int): The total number of iterations or steps.
            description (str, optional): The description of the progress bar. Defaults to "no description".

        Returns:
            None

        Example:
            total_iterations = 100
            for i in range(total_iterations):
                progress_bar(i+1, total_iterations)
                # Do some work here
                time.sleep(0.1)  # Simulate some work being done
        """
    percent = 100 * (progress / total)
    bar = '#' * int(percent / 2) + '-' * (50 - int(percent / 2))
    if description is not None:
        print(f"\r{percent:.2f}% |{bar}| {progress}/{total} | {description}", end="\r")
    else:
        print(f"\r{percent:.2f}% |{bar}| {progress}/{total}", end="\r")
    if progress == total:
        print()


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f'TTTTTTTTTTTTTTTTTTTTTT Processing Update: {func.__name__}() took {end - start:.2f} seconds to execute\n')
        return result
    return wrapper


def print_array(array: torch.Tensor | np.ndarray, array_name: str | None = None):
    if isinstance(array, torch.Tensor):
        num_elements = array.numel()
        element_size = array.element_size()
        shape = tuple(array.shape)
        dtype = str(array.dtype)
    elif isinstance(array, np.ndarray):
        num_elements = array.size
        element_size = array.itemsize
        shape = array.shape
        dtype = str(array.dtype)
    elif isinstance(array, list):
        array = np.asarray(array)
        num_elements = array.size
        element_size = array.itemsize
        shape = array.shape
        dtype = str(array.dtype)
    else:
        raise TypeError("Input must be a torch.Tensor, numpy.ndarray or list")

    total_memory_bytes = num_elements * element_size
    total_memory_mb = total_memory_bytes / (1024 ** 2)

    array_name = f"'{array_name}'" if array_name else f"'array'"

    print(f'PPPPPPPPPPPPPPPPPPPPPP {array_name} with shape {shape} dtype {dtype} is using {total_memory_mb:.2f} MB of '
          f'memory.')


def delete_memmap_files(*base_names: str, to_print: bool = True, target_directory: str = dp_data_file_dir):
    if not os.path.isdir(target_directory):
        if to_print:
            print(f"Warning: Target directory '{target_directory}' not found. No files will be deleted.")
        return

    for base_name in base_names:
        pattern = os.path.join(target_directory, f'{base_name}*.dat')
        matching_files = glob.glob(pattern)

        if not matching_files and to_print:
            print(f"----- No files found matching pattern: {pattern}")

        for file_path in matching_files:
            try:
                os.remove(file_path)
                if to_print:
                    print(f"XXXXXXXXXXXXXXX Deleted file: {file_path}")
            except OSError as e:
                if to_print:
                    print(f"Error deleting {file_path}: {e}")
            except Exception as e:
                if to_print:
                    print(f"An unexpected error occurred while deleting {file_path}: {e}")


def print_disk_info(drive="Scene2_S:\\"):
    try:
        total, used, free = shutil.disk_usage(drive)
        total_gb = total // (2**30)
        free_gb = free // (2**30)
        used_gb = used // (2**30)
        percent_free = (free / total) * 100

        print(f"Devices and drives")
        print(f"  OS ({drive[0]}:)")
        print(f"    {free_gb} GB free of {total_gb} GB")
        print(f"\nAdditional Information:")
        print(f"  Used Space: {used_gb} GB")
        print(f"  Percent Free: {percent_free:.2f}%")
    except Exception as e:
        print(f"An error occurred: {e}")


def delete_all_files_in_folder(folder_path, file_type="*"):
    # Find all files in the specified folder with the given file type
    files = glob.glob(os.path.join(folder_path, f"*.{file_type}"))

    # Delete each file
    for file_path in files:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Could not delete {file_path}: {e}")


def trim_audio(
        input_file: str,
        start_time: float,
        duration: Optional[float] = None,
        output_folder: str = (
                r""
                r"\Chaos Theory and Double Pendulum Video\manim files\wav files"
        )
) -> str:
    audio = AudioSegment.from_mp3(input_file)

    #  audio length in seconds
    audio_length_sec = len(audio) / 1000

    assert start_time >= 0, "start_time must be non-negative."
    assert start_time < audio_length_sec, "start_time must be less than the total audio length."

    if duration is not None:
        assert duration > 0, "duration must be positive."
        assert start_time + duration <= audio_length_sec, (
            "The sum of start_time and duration exceeds the total audio length."
        )

    start_ms = start_time * 1000

    if duration is not None:
        end_ms = (start_time + duration) * 1000
        trimmed_audio = audio[start_ms:end_ms]
    else:
        trimmed_audio = audio[start_ms:]

    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "trimmed_audio.mp3")
    trimmed_audio.export(output_file, format="mp3")

    return output_file


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









