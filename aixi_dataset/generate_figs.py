"""Module for the figure generation

Written by Miquel Miró Nicolau, 2025
"""

import math
import random
import warnings

import cv2
import numpy as np
from aixi_dataset import random_figs

ITER_MAX = 1000


def grid_calculation(
    used: set, num_grid: int, grid_shape: tuple[int, int]
) -> tuple[tuple[int, int], set]:
    """Calculates the position of the grid.

    Args:
        used: Set of integers representing the positions already used.
        num_grid: Integer representing the total number of cell of the grid.
        grid_shape: Tuple of integers representing the shape of the grid.

    Returns:
        Tuple of integers representing the position of the grid.
        Set of integers representing the positions already used.
    """
    i = random.randint(0, (num_grid**2) - 1)

    tries = 0
    while (i in used) and (tries < ITER_MAX):
        i = random.randint(0, (num_grid**2) - 1)
        tries += 1

    if tries == ITER_MAX:
        warnings.warn("Max iteration tried")
        return (None, None), used

    used.add(i)

    x = i % num_grid
    y = i // num_grid

    grid_x = x * grid_shape[0]
    grid_y = y * grid_shape[1]

    return (grid_x, grid_y), used


def polygons(
    image: np.ndarray,
    grid_divs: int,
    num_circles: int,
    num_squares: int,
    num_crosses: int,
    value: list[int | float],
) -> tuple[np.ndarray, dict[str, float], tuple, dict[str, int], bool]:
    """Generates synthetic image by the superposition of circles, squares and crosses.

    Args:
        image: NumPy array representing the image.
        grid_divs: Integer, number of division to build the grid.
        num_circles: Integer number of circles to draw.
        num_squares: Integer number of squares to draw.
        num_crosses: Integer number of crosses to draw.
        value: Integer representing the value to draw.

    Returns:
        NumPy array representing the image with the circles, squares and crosses drawn.
        Dictionary with the number of circles, squares and crosses drawn.
    """
    image = np.copy(image)

    figure_image = {"c": np.copy(image), "s": np.copy(image), "cr": np.copy(image)}

    num_grid = grid_divs
    num_figures = {"c": num_circles, "s": num_squares, "cr": num_crosses}

    grid_shape = (image.shape[0] // num_grid), (image.shape[1] // num_grid)

    draw_elems = {"c": 0, "s": 0, "cr": 0}
    draw_px = {"c": 0.0, "s": 0, "cr": 0}
    used: set[int] = set()
    value_idx = 0

    for (key, num_figure), draw_fn in zip(
        num_figures.items(), tuple(random_figs.RAND_FIGS)
    ):
        while draw_elems[key] < num_figure:
            grid_positions, used = grid_calculation(used, num_grid, grid_shape)
            displacement = random.randint(5, (grid_shape[0] - 5) // 2), random.randint(
                5, (grid_shape[1] - 5) // 2
            )

            figure_image[key], area = draw_fn(
                figure_image[key],
                grid_shape,
                grid_positions,
                displacement,
                value[value_idx],
            )

            value_idx = (value_idx + 1) % len(value)
            draw_px[key] += area
            draw_elems[key] += 1

    for fig_img in figure_image.values():
        image = image + fig_img

    overlapped = image.max() > max(value)
    image[image > 255] = 255

    return image, draw_px, tuple(figure_image.values()), draw_elems, overlapped


def equally_probable(
    image: np.ndarray, values: list[int], percentage: float
) -> np.ndarray:
    """Draw value on a percentage of the image.

    The final result of this function will be an image with the percentage of pixels of the value
    passed as parameter. Another condition is that these pixels we want to be most possible densely
    distributed on the image.

    Args:
        image: Numpy array representing the image.
        values: List of integers with the values to draw.
        percentage: Float percentage of the image to draw the value.

    Returns:
        Numpy array representing the image with the pixels drawn.
    """
    image = np.copy(image)

    for val in values:
        number_of_values = np.count_nonzero(image == val)
        real_percentage = number_of_values / image.size
        num_px = image.size * percentage

        side = math.sqrt(num_px)

        while (percentage - real_percentage) > 0.05:
            pos_x, pos_y = (
                random.randint(0, image.size),
                random.randint(0, image.size),
            )  # Num of pixels to draw
            random_viewport = random.uniform(0.2, 0.8)  # Proportion of width

            width = int(random_viewport * side)
            height = int((1 - random_viewport) * side)

            image[pos_x : pos_x + width, pos_y : pos_y + height][
                image[pos_x : pos_x + width, pos_y : pos_y + height] == 0
            ] = val
            real_percentage = number_of_values / image.size

    return image


def circles(
    image: np.ndarray,
    values: list[int],
    num_of_circles: int,
    overlapped: bool = True,
    return_count: bool = False,
    margin: int = 0,
    max_iter: int = -1,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Generates synthetic image by the superposition of circles.

    Args:
        image: NumPy array representing the image.
        values: List of integers with the values to draw.
        num_of_circles: Integer number of circles to draw.
        overlapped: Boolean indicating if the circles can be overlapped.
        return_count: Boolean indicating if the function should return the number of circles drawn.
            used
        margin: Integer, space between different objects. Default=0.
        max_iter: Integer, maximum number of tries
    Returns:
        NumPy array representing the image with the circles drawn.
    """
    image = np.copy(image)
    count = {}

    drawn_circles = 0
    img_used = []
    iterations = 0
    while (drawn_circles < num_of_circles) and (iterations < max_iter or max_iter < 0):
        value = random.choice(values)

        radius = random.randint(10, min(image.shape) // 4)

        center_x = random.randint(0, image.shape[0])
        center_y = random.randint(0, image.shape[1])

        radius_and_margin = radius + margin

        if (
            image[
                max(center_y - radius_and_margin, 0) : center_y + radius_and_margin,
                max(center_x - radius_and_margin, 0) : center_x + radius_and_margin,
            ].max()
            == 0
        ) or overlapped:
            img_used.append(((center_x, center_y), radius, value))

            image = cv2.circle(image, (center_x, center_y), radius, value, thickness=-1)
            drawn_circles += 1
            if value not in count:
                count[value] = 0

            count[value] += 1
        iterations = iterations + 1
    if return_count:
        return image, count

    return image
