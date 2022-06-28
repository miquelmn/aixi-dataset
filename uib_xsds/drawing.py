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


def circles(image: np.ndarray, values: list[int], num_of_circles: int) -> np.ndarray:
    """Generates synthetic image by the superposition of circles.

    Args:
        image: NumPy array representing the image.
        values: List of integers with the values to draw.
        num_of_circles: Integer number of circles to draw.

    Returns:
        NumPy array representing the image with the circles drawn.
    """
    image = np.copy(image)

    for _ in range(num_of_circles):
        value = random.choice(values)

        center_x = random.randint(0, image.shape[0])
        center_y = random.randint(0, image.shape[1])
        radius = random.randint(10, min(image.shape) // 2)

        image = cv2.circle(image, (center_x, center_y), radius, value, thickness=-1)

    return image
