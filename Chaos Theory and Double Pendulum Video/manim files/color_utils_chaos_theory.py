import torch

import numpy as np
from manim import *
from numpy import sin, cos, tan
from numba import njit
import math
from typing import Sequence, Callable
from custom_manim import *
from enum import Enum


RAINBOW = [
    PURE_RED,
    rgb_to_color([255, 165, 0]),  # Orange
    YELLOW,
    PURE_GREEN,
    rgb_to_color([0, 255, 255]),  # Cyan
    PURE_BLUE,
    rgb_to_color([255, 40, 255]),  # Magenta
]


class NumpyColorFuncs:
    @staticmethod
    def torus_smooth_gradient(angle_pairs: np.ndarray, alpha=1.0) -> np.ndarray:
        angle_1, angle_2 = angle_pairs[:, 0], angle_pairs[:, 1]
        theta_1 = angle_1 * DEGREES
        theta_2 = angle_2 * DEGREES

        g = (1 + np.tanh(np.sin(theta_1) * np.sin(theta_2))) * 0.5

        theta_1 = np.abs(theta_1) / PI
        theta_2 = np.abs(theta_2) / PI

        r = -theta_1 ** 2 + 1
        b = -theta_2 ** 2 + 1

        return np.stack([r, g, b, np.full_like(r, alpha)], axis=-1)

    @staticmethod
    def divide_by_quadrant(angle_pairs: np.ndarray, alpha=1.0) -> np.ndarray:
        theta_1 = angle_pairs[:, 0] * DEGREES
        theta_2 = angle_pairs[:, 1] * DEGREES

        r = np.zeros_like(theta_1)
        g = np.zeros_like(theta_1)
        b = np.zeros_like(theta_1)

        # Quadrant 1 (RED)
        mask = (theta_1 >= 0) & (theta_2 >= 0)
        r[mask] = 1

        # Quadrant 4 (GREEN)
        mask = (theta_1 >= 0) & (theta_2 < 0)
        g[mask] = 1

        # Quadrant 2 (BLUE)
        mask = (theta_1 < 0) & (theta_2 >= 0)
        b[mask] = 1

        # Quadrant 3 (YELLOW)
        mask = (theta_1 < 0) & (theta_2 < 0)
        r[mask] = 1
        g[mask] = 1
        b[mask] = 0

        return np.stack([r, g, b, np.full_like(r, alpha)], axis=-1)

    @staticmethod
    def annulus(angle_pairs: np.ndarray, alpha=1.0) -> np.ndarray:
        r = np.pi / 10
        theta_1 = angle_pairs[:, 0] * DEGREES
        theta_2 = angle_pairs[:, 1] * DEGREES
        circle_sum = theta_1 ** 2 + theta_2 ** 2

        inner_radius = np.pi / 2 - r
        outer_radius = np.pi / 2 + r

        inner_mask = circle_sum < inner_radius
        annulus_mask = (inner_radius <= circle_sum) & (circle_sum <= outer_radius)

        col = np.ones((angle_pairs.shape[0], 4))  # Default white background
        col[inner_mask] = color_to_rgba(PINK, alpha)
        col[annulus_mask] = [0, 0, 0, alpha]

        return col

    @staticmethod
    def checkerboard(angle_pairs: np.ndarray, alpha=1.0) -> np.ndarray:
        side_length = 8
        theta_1 = angle_pairs[:, 0] + 180
        theta_2 = angle_pairs[:, 1] + 180

        index_1 = (theta_1 // (360 / side_length)).astype(int)
        index_2 = (theta_2 // (360 / side_length)).astype(int)

        mask = (index_1 + index_2) % 2 == 0
        col = np.zeros((angle_pairs.shape[0], 4))
        col[mask] = [1, 1, 1, alpha]  # White color for even sum
        col[~mask] = [0, 0, 0, alpha]  # Black color for odd sum

        return col

    @staticmethod
    def grid_pattern(angle_pairs: np.ndarray, alpha=1.0, grid_size=16, line_thickness=0.1) -> np.ndarray:
        theta_1 = (angle_pairs[:, 0] * DEGREES + np.pi) % (2 * np.pi)
        theta_2 = (angle_pairs[:, 1] * DEGREES + np.pi) % (2 * np.pi)

        cell_width = (2 * np.pi) / grid_size
        half_line_thickness = (line_thickness / 2) * cell_width

        distance_to_edge_1 = np.minimum(theta_1 % cell_width, cell_width - (theta_1 % cell_width))
        distance_to_edge_2 = np.minimum(theta_2 % cell_width, cell_width - (theta_2 % cell_width))

        mask = (distance_to_edge_1 < half_line_thickness) | (distance_to_edge_2 < half_line_thickness)

        col = np.ones((angle_pairs.shape[0], 4))  # Default white background
        col[mask] = [0, 0, 0, alpha]  # Black grid lines

        return col

    @staticmethod
    def stripes(angle_pairs: np.ndarray, alpha=1.0) -> np.ndarray:
        num_of_alternating_stripes = 20
        theta_1 = angle_pairs[:, 0] + 180

        index = (theta_1 // (360 / num_of_alternating_stripes)).astype(int)
        mask = index % 2 == 0

        col = np.zeros((angle_pairs.shape[0], 4))
        col[mask] = [1, 1, 1, alpha]  # White for even indices
        col[~mask] = [0, 0, 0, alpha]  # Black for odd indices

        return col

    @staticmethod
    def radial_center_gradient(angle_pairs: np.ndarray, alpha=1.0) -> np.ndarray:
        def get_constant_color(normalized_distance: np.ndarray) -> np.ndarray:
            num_colors = len(RAINBOW)
            scaled_index = np.floor(normalized_distance * num_colors).astype(int)
            scaled_index = np.clip(scaled_index, 0, num_colors - 1)

            constant_colors = np.array([color_to_rgb(RAINBOW[i]) for i in range(num_colors)])
            return constant_colors[scaled_index]

        theta_1 = angle_pairs[:, 0] * DEGREES
        theta_2 = angle_pairs[:, 1] * DEGREES

        distance = np.sqrt(theta_1 ** 2 + theta_2 ** 2)
        max_distance = np.sqrt(2) * np.pi  # Maximum distance on a [-pi, pi] x [-pi, pi] grid
        normalized_distance = distance / max_distance
        colors = get_constant_color(normalized_distance)
        colors_with_alpha = np.concatenate([colors, np.full((colors.shape[0], 1), alpha)], axis=-1)

        return colors_with_alpha

    @staticmethod
    def cat(angle_pairs: np.ndarray, alpha=1.0) -> np.ndarray:
        cat_pixel_array = ImageMobject("cat.png").pixel_array[::-1, :, :]
        h, w, _ = cat_pixel_array.shape

        # Normalize angles to pixel indices
        angle_1_normalized = (angle_pairs[:, 0] + 180) / 360  # Normalize to range [0, 1]
        angle_2_normalized = (angle_pairs[:, 1] + 180) / 360  # Normalize to range [0, 1]

        # Convert to pixel indices
        x_index = (angle_1_normalized * (w - 1)).astype(int)
        y_index = (angle_2_normalized * (h - 1)).astype(int)

        return (cat_pixel_array[y_index, x_index] / 255).astype(np.float64)

    @staticmethod
    def white(angle_pairs: np.ndarray, alpha=1.0) -> np.ndarray:
        return np.full((angle_pairs.shape[0], 4), [1, 1, 1, alpha])


class TorchColorFuncs:
    @staticmethod
    def torus_smooth_gradient(angle_pairs: torch.Tensor, alpha=1.0) -> torch.Tensor:
        angle_1, angle_2 = angle_pairs[:, 0], angle_pairs[:, 1]
        theta_1 = angle_1 * DEGREES
        theta_2 = angle_2 * DEGREES

        g = g = (1 + torch.tanh(torch.sin(theta_1) * torch.sin(theta_2))) * 0.5

        theta_1 = torch.abs(theta_1) / PI
        theta_2 = torch.abs(theta_2) / PI

        r = -theta_1 ** 2 + 1
        b = -theta_2 ** 2 + 1

        return torch.stack([r, g, b, torch.full_like(r, alpha)], dim=-1)

    @staticmethod
    def divide_by_quadrant(angle_pairs: torch.Tensor, alpha=1.0) -> torch.Tensor:
        theta_1, theta_2 = angle_pairs.T * DEGREES

        # Define the RGB values for each quadrant
        rgb_values = [
            (0, 0, 1),  # blue for upper left
            (1, 1, 0),  # yellow for upper right
            (0, 1, 0),  # green for lower left
            (1, 0, 0)  # red for lower right
        ]

        # Masks for each quadrant
        quadrants = [
            (theta_1 < 0) & (theta_2 >= 0),  # Upper left
            (theta_1 >= 0) & (theta_2 >= 0),  # Upper right
            (theta_1 < 0) & (theta_2 < 0),  # Lower left
            (theta_1 >= 0) & (theta_2 < 0)  # Lower right
        ]

        # Stack RGB channels by applying masks
        r, g, b = (sum(mask * rgb[i] for mask, rgb in zip(quadrants, rgb_values)) for i in range(3))

        return torch.stack([r, g, b, torch.full_like(g, alpha)], dim=-1)

    @staticmethod
    def annulus(angle_pairs: torch.Tensor, alpha=1.0) -> torch.Tensor:
        r = np.pi / 10
        theta_1 = angle_pairs[:, 0] * DEGREES
        theta_2 = angle_pairs[:, 1] * DEGREES
        circle_sum = theta_1 ** 2 + theta_2 ** 2

        inner_radius = np.pi / 2 - r
        outer_radius = np.pi / 2 + r

        inner_mask = circle_sum < inner_radius
        annulus_mask = (inner_radius <= circle_sum) & (circle_sum <= outer_radius)

        col = torch.ones((angle_pairs.shape[0], 4))  # Default white background
        col[inner_mask] = torch.tensor(color_to_rgba(PINK, alpha))
        col[annulus_mask] = torch.tensor([0, 0, 0, alpha])

        return col

    @staticmethod
    def checkerboard(angle_pairs: torch.Tensor, alpha=1.0) -> torch.Tensor:
        side_length = 8
        theta_1 = angle_pairs[:, 0] + 180
        theta_2 = angle_pairs[:, 1] + 180

        index_1 = (theta_1 // (360 / side_length)).int()
        index_2 = (theta_2 // (360 / side_length)).int()

        mask = (index_1 + index_2) % 2 == 0
        col = torch.zeros((angle_pairs.shape[0], 4))
        col[mask] = torch.tensor([1, 1, 1, alpha])  # White color for even sum
        col[~mask] = torch.tensor([0, 0, 0, alpha])  # Black color for odd sum

        return col

    @staticmethod
    def grid_pattern(angle_pairs: torch.Tensor, alpha=1.0, grid_size=16, line_thickness=0.1) -> torch.Tensor:
        theta_1 = (angle_pairs[:, 0] * DEGREES + np.pi) % (2 * np.pi)
        theta_2 = (angle_pairs[:, 1] * DEGREES + np.pi) % (2 * np.pi)

        cell_width = (2 * np.pi) / grid_size
        half_line_thickness = (line_thickness / 2) * cell_width

        distance_to_edge_1 = torch.minimum(theta_1 % cell_width, cell_width - (theta_1 % cell_width))
        distance_to_edge_2 = torch.minimum(theta_2 % cell_width, cell_width - (theta_2 % cell_width))

        mask = (distance_to_edge_1 < half_line_thickness) | (distance_to_edge_2 < half_line_thickness)

        col = torch.ones((angle_pairs.shape[0], 4))  # Default white background
        col[mask] = torch.tensor([0, 0, 0, alpha])  # Black grid lines

        return col

    @staticmethod
    def stripes(angle_pairs: torch.Tensor, alpha=1.0) -> torch.Tensor:
        num_of_alternating_stripes = 20
        theta_1 = angle_pairs[:, 0] + 180

        index = (theta_1 // (360 / num_of_alternating_stripes)).int()
        mask = index % 2 == 0

        col = torch.zeros((angle_pairs.shape[0], 4))
        col[mask] = torch.tensor([1, 1, 1, alpha])  # White for even indices
        col[~mask] = torch.tensor([0, 0, 0, alpha])  # Black for odd indices

        return col

    @staticmethod
    def square_onion(angle_pairs: torch.Tensor, alpha=1.0) -> torch.Tensor:
        def get_constant_color(normalized_distance: torch.Tensor) -> torch.Tensor:
            num_colors = len(RAINBOW)
            scaled_index = torch.floor(normalized_distance * num_colors).int()
            scaled_index = torch.clamp(scaled_index, 0, num_colors - 1).to(device)

            colors_array = np.array([color_to_rgb(RAINBOW[i]) for i in range(num_colors)])
            constant_colors = torch.from_numpy(colors_array).to(device)

            return constant_colors[scaled_index]

        theta_1 = angle_pairs[:, 0] * DEGREES
        theta_2 = angle_pairs[:, 1] * DEGREES

        # Use L∞ norm (max norm) for squares instead of L2 (Euclidean) for circles
        distance = torch.maximum(torch.abs(theta_1), torch.abs(theta_2))

        # Maximum distance on a [-pi, pi] x [-pi, pi] grid using L∞ norm is pi
        max_distance = np.pi
        normalized_distance = distance / max_distance

        colors = get_constant_color(normalized_distance)
        colors_with_alpha = torch.cat(
            [colors, torch.full((colors.shape[0], 1), alpha, device=device)],
            dim=-1)

        return colors_with_alpha

    @staticmethod
    def boundary_highlight(angle_pairs: torch.Tensor, alpha=1.0, threshold=10.0,
                           highlight_color=(1.0, 0, 0)) -> torch.Tensor:
        """
        Colors angles that are close to ±180 degrees with a highlight color.

        Parameters:
        angle_pairs: Tensor of shape (N, 2) containing angle pairs in degrees
        alpha: Opacity value
        threshold: Angle threshold in degrees - how close to ±180 to trigger highlighting
        highlight_color: RGB color tuple for the highlight

        Returns:
        Tensor of shape (N, 4) with RGBA values
        """
        # Default background color (white)
        result = torch.ones((angle_pairs.shape[0], 4), dtype=torch.float32)

        # Find angles close to ±180 in either dimension
        angle_1 = angle_pairs[:, 0]
        angle_2 = angle_pairs[:, 1]

        # Calculate distance from ±180 for each angle
        dist_1 = torch.minimum(torch.abs(angle_1 - 180.0), torch.abs(angle_1 + 180.0))
        dist_2 = torch.minimum(torch.abs(angle_2 - 180.0), torch.abs(angle_2 + 180.0))

        # Create mask for points close to boundary in either dimension
        boundary_mask = (dist_1 < threshold) | (dist_2 < threshold)

        # Apply highlight color to boundary points
        result[boundary_mask, 0] = highlight_color[0]
        result[boundary_mask, 1] = highlight_color[1]
        result[boundary_mask, 2] = highlight_color[2]

        # Set alpha channel
        result[:, 3] = alpha

        return result

    @staticmethod
    def radial_center_gradient(angle_pairs: torch.Tensor, alpha=1.0) -> torch.Tensor:
        def get_constant_color(normalized_distance: torch.Tensor) -> torch.Tensor:
            num_colors = len(RAINBOW)
            scaled_index = torch.floor(normalized_distance * num_colors).int()
            scaled_index = torch.clamp(scaled_index, 0, num_colors - 1).to(device)

            colors_array = np.array([color_to_rgb(RAINBOW[i]) for i in range(num_colors)])
            constant_colors = torch.from_numpy(colors_array).to(device)

            return constant_colors[scaled_index]

        theta_1 = angle_pairs[:, 0] * DEGREES
        theta_2 = angle_pairs[:, 1] * DEGREES

        distance = torch.sqrt(theta_1 ** 2 + theta_2 ** 2)
        max_distance = np.sqrt(2) * np.pi  # Maximum distance on a [-pi, pi] x [-pi, pi] grid
        normalized_distance = distance / max_distance
        colors = get_constant_color(normalized_distance)
        colors_with_alpha = torch.cat(
            [colors, torch.full((colors.shape[0], 1), alpha, device=device)],
            dim=-1)

        return colors_with_alpha

    @staticmethod
    def cat(angle_pairs: torch.Tensor, alpha=1.0) -> torch.Tensor:
        cat_pixel_array = ImageMobject("white_cat.png").pixel_array[::-1, :, :]
        cat_pixel_array = torch.from_numpy(cat_pixel_array.copy()).to(device)
        h, w, _ = cat_pixel_array.shape

        angle_1_normalized = (angle_pairs[:, 0] + 180) / 360  # Normalize to range [0, 1]
        angle_2_normalized = (angle_pairs[:, 1] + 180) / 360  # Normalize to range [0, 1]

        # Convert to pixel indices
        angle_1_normalized = (angle_1_normalized * (w - 1)).int()
        angle_2_normalized = (angle_2_normalized * (h - 1)).int()

        return cat_pixel_array[angle_2_normalized, angle_1_normalized] / 255

    @staticmethod
    def white(angle_pairs: torch.Tensor, alpha=1.0) -> torch.Tensor:
        result = torch.ones((angle_pairs.shape[0], 4), dtype=torch.float32)
        result[:, 3] = alpha

        return result


# other color-related functions


def turn_angles_to_color(
        angle_1: float,
        angle_2: float,
        color_func_index: int = 0,
        alpha: float = 1.0
) -> ManimColor:
    """
    Return a ManimColor generated by applying a color function to the given angles.

    Args:
        angle_1 (float): The first angle.
        angle_2 (float): The second angle.
        color_func_index (int): The index of the color function to apply.
        alpha (float, optional): The alpha value for the color. Defaults to 1.0.

    Returns:
        ManimColor: The color generated by applying the color function to the angles.

    color_func indices:
        0: torus_smooth_gradient
        1: divide_by_quadrant
    """
    color_funcs = [
        NumpyColorFuncs.torus_smooth_gradient,  # 0
        NumpyColorFuncs.divide_by_quadrant,  # 1
        NumpyColorFuncs.annulus,  # 2
        NumpyColorFuncs.checkerboard,  # 3
        NumpyColorFuncs.stripes,  # 4
        NumpyColorFuncs.cat,  # 5
        NumpyColorFuncs.white,  # 6
    ]
    _rgba = color_funcs[color_func_index](angle_pairs=np.array([[angle_1, angle_2]]), alpha=alpha)

    assert _rgba.shape == (1, 4), f"shape of _rgba: {_rgba.shape} must be (1, 4)"

    return rgba_to_color(_rgba[0])


def color_gradient_with_rate_func(
        reference_colors: Sequence[ManimColor],
        length_of_output: int,
        rate_func: Callable[[float], float]
) -> list[ManimColor]:
    if len(reference_colors) == 1:
        return [ManimColor(reference_colors[0])] * length_of_output

    rgbs = [color_to_rgb(col) for col in reference_colors]
    alphas = np.linspace(0, len(rgbs) - 1, length_of_output)
    alphas = np.asarray(apply_rate_func_to_linear_sequence(alphas, rate_func))

    floors = alphas.astype("int")
    alphas_mod1 = alphas % 1

    # End edge case
    alphas_mod1[-1] = 1
    floors[-1] = len(rgbs) - 2
    return [
        rgb_to_color((rgbs[i] * (1 - alpha)) + (rgbs[i + 1] * alpha))
        for i, alpha in zip(floors, alphas_mod1)
    ]

