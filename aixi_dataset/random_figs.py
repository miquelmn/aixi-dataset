"""Module containing multiple drawing functions.

Written by: Miquel Miró Nicolau (UIB), 2022
"""

import math
import random
import numpy as np

from aixi_dataset import drawing

random.seed(42)


def circle(
    image: np.ndarray,
    grid_shape: tuple[int, int],
    grid_position: tuple[int, int],
    displacement: tuple[int, int],
    value: int,
) -> tuple[np.ndarray, float]:
    """Draws circles in the grid.

    Args:
        image: NumPy array representing the image to draw.
        grid_shape: Tuple of integers representing the shape of the grid.
        grid_position: Tuple of integers representing the position into the grid.
        displacement: Tuple of integers representing the displacement respecting the grid cell.
        value: Integer representing the value to draw.

    Returns:
        NumPy array representing the image with the circles drawn.
        Area of the circle drawn.
    """
    grid_x, grid_y = grid_position
    dx, dy = displacement

    radius = random.randint(5, max(int((min(grid_shape) - max(dx, dy)) // 2), 6))

    image = drawing.circle(
        image.shape,
        (grid_y + dy + (radius // 2), grid_x + dx + (radius // 2)),
        radius,
        value,
    )

    return image, math.pi * (radius**2)


def square(
    image: np.ndarray,
    grid_shape: tuple[int, int],
    grid_position: tuple[int, int],
    displacement: tuple[int, int],
    value: int,
) -> tuple[np.ndarray, int]:
    """Draws squares in the grid.

    Args:
        image: NumPy array representing the image to draw.
        grid_shape: Tuple of integers representing the shape of the grid.
        grid_position: Tuple of integers representing the position into the grid.
        displacement: Tuple of integers representing the displacement respecting the grid cell.
        value: Integer representing the value to draw.

    Returns:
        NumPy array representing the image with the squares drawn.
        Area of the square drawn.
    """
    dx, dy = displacement
    grid_x, grid_y = grid_position

    if dx < (grid_shape[0] // 2) or dy < (grid_shape[1] // 2):
        min_distance = max((dx, dy))
    else:
        min_distance = min((dx, dy))

    side = random.randint(5, max(max(grid_shape) - min_distance, 6))

    pos = (max((dx + grid_x - (side // 2)), 0), max((dy + grid_y - (side // 2)), 0))
    image = drawing.square(image.shape, pos, side, value)

    return image, side**2


def cross(
    image: np.ndarray,
    grid_shape: tuple[int, int],
    grid_position: tuple[int, int],
    displacement: tuple[int, int],
    value: int,
) -> tuple[np.ndarray, int]:
    """Draws crosses in the grid.

    Args:
        image: NumPy array representing the image to draw.
        grid_shape: Tuple of integers representing the shape of the grid.
        grid_position: Tuple of integers representing the position into the grid.
        displacement: Tuple of integers representing the displacement respecting the grid cell.
        value: Integer representing the value to draw.

    Returns:
        NumPy array representing the image with the crosses drawn.
        Area of the crosses drawn.
    """
    dx, dy = displacement
    grid_x, grid_y = grid_position

    if dx < (grid_shape[0] // 2) or dy < (grid_shape[1] // 2):
        min_distance = max((dx, dy))
    else:
        min_distance = min((dx, dy))

    long_side = random.randint(10, max(grid_shape) - min_distance)
    short_side = long_side // 2

    image = image + drawing.cross(
        image.shape, ((dx + grid_x), (dy + grid_y)), (long_side, short_side), value
    )

    return image, (((short_side * long_side) * 2) - short_side**2)


RAND_FIGS = [circle, square, cross]
