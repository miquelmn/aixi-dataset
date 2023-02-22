""" Module containing multiple drawing functions.

Written by: Miquel Miró Nicolau (UIB), 2022
"""
import math
import random

import cv2
import numpy as np

random.seed(42)


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
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Generates synthetic image by the superposition of circles.

    Args:
        image: NumPy array representing the image.
        values: List of integers with the values to draw.
        num_of_circles: Integer number of circles to draw.
        overlapped: Boolean indicating if the circles can be overlapped.
        return_count: Boolean indicating if the function should return the number of circles drawn.
        used
    Returns:
        NumPy array representing the image with the circles drawn.
    """
    image = np.copy(image)
    count = {}

    drawn_circles = 0
    img_used = []
    while drawn_circles < num_of_circles:
        value = random.choice(values)

        radius = random.randint(10, min(image.shape) // 4)

        center_x = random.randint(0, image.shape[0])
        center_y = random.randint(0, image.shape[1])

        if (
            image[
                max(center_y - radius, 0) : center_y + radius,
                max(center_x - radius, 0) : center_x + radius,
            ].max()
            == 0
        ) or overlapped:
            img_used.append(((center_x, center_y), radius, value))

            image = cv2.circle(image, (center_x, center_y), radius, value, thickness=-1)
            drawn_circles += 1
            if value not in count:
                count[value] = 0

            count[value] += 1

    if return_count:
        return image, count
    else:
        return image


def __draw_circles(
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

    assert ((radius * 2) + max((dx, dy))) <= max(grid_shape), "Circle out of cell grid"

    image = cv2.circle(
        image, (grid_y + dy + radius, grid_x + dx + radius), radius, value, thickness=-1
    )

    return image, math.pi * (radius**2)


def __draw_square(
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

    image[dx + grid_x : dx + grid_x + side, dy + grid_y : dy + grid_y + side] = value

    return image, side**2


def __draw_crosses(
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

    long_side = random.randint(1, max(grid_shape) - min_distance)
    short_side = random.randint(5, max(6, long_side // 2))

    image[
        (dx + grid_x) : (dx + grid_x) + long_side,
        (dy + grid_y)
        + (long_side // 2)
        - (short_side // 2) : (dy + grid_y)
        + (short_side // 2)
        + (long_side // 2),
    ] = value

    image[
        (dx + grid_x)
        + (long_side // 2)
        - (short_side // 2) : (dx + grid_x)
        + (short_side // 2)
        + (long_side // 2),
        (dy + grid_y) : (dy + grid_y) + long_side,
    ] = value

    short_side, long_side = sorted((short_side, long_side))

    return image, (((short_side * long_side) * 2) - short_side**2)


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
    while i in used:
        i = random.randint(0, num_grid**2)

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
) -> tuple[
    np.ndarray,
    dict[str, float],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    dict[str, int],
]:
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

    circles_image = np.copy(image)
    square_image = np.copy(image)
    crosses_image = np.copy(image)

    num_grid = grid_divs

    grid_shape = (image.shape[0] // (num_grid + 1)), (image.shape[1] // (num_grid + 1))

    draw_elems = {"c": 0, "s": 0, "cr": 0}
    draw_px = {"c": 0.0, "s": 0, "cr": 0}
    used: set[int] = set()
    value_idx = 0

    while draw_elems["c"] < num_circles:
        grid_positions, used = grid_calculation(used, num_grid, grid_shape)
        displacement = random.randint(5, (grid_shape[0] - 5) // 2), random.randint(
            5, (grid_shape[1] - 5) // 2
        )

        circles_image, area = __draw_circles(
            circles_image, grid_shape, grid_positions, displacement, value[value_idx]
        )

        value_idx = (value_idx + 1) % len(value)
        draw_px["c"] += area
        draw_elems["c"] += 1

    while draw_elems["s"] < num_squares:
        grid_positions, used = grid_calculation(used, num_grid, grid_shape)
        displacement = random.randint(5, (grid_shape[0] - 5) // 2), random.randint(
            5, (grid_shape[1] - 5) // 2
        )

        square_image, area = __draw_square(
            square_image, grid_shape, grid_positions, displacement, value[value_idx]
        )

        value_idx = (value_idx + 1) % len(value)

        draw_px["s"] += area
        draw_elems["s"] += 1

    while draw_elems["cr"] < num_crosses:
        grid_positions, used = grid_calculation(used, num_grid, grid_shape)
        displacement = random.randint(5, (grid_shape[0] - 5) // 2), random.randint(
            5, (grid_shape[1] - 5) // 2
        )

        crosses_image, area = __draw_crosses(
            crosses_image, grid_shape, grid_positions, displacement, value[value_idx]
        )

        value_idx = (value_idx + 1) % len(value)

        draw_px["cr"] += area
        draw_elems["cr"] += 1

    image = square_image + crosses_image + circles_image
    image[image > 255] = 255

    return image, draw_px, (circles_image, square_image, crosses_image), draw_elems
