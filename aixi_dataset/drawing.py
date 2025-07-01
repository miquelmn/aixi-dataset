"""Module for the drawing of a single polygon.

Written by Miquel Miró Nicolau, 2025
"""

import numpy as np
import cv2


def circle(shape_img, center, size, value):
    """Draw a circle.

    Args:
        shape_img: Tuple, shape of the image.
        center: Tuple, center of the circle (y, x).
        size: Integer, radius of the circle.
        value: Integer, value of the circle.

    Returns:
        Image with a circle drawn.
    """
    image = np.zeros(shape_img)

    return cv2.circle(
        image,
        center,
        size,
        value,
        thickness=-1,
    )


def square(shape_img, center, size, value):
    """Draw a square.

    Args:
        shape_img: Tuple, shape of the image.
        center: Tuple, center of the square (y, x).
        size: Integer, size of the square (width and height).
        value: Integer, value of the square.

    Returns:
        Returns a numpy array with a square drawn.
    """
    image = np.zeros(shape_img)

    image[
        center[0] - (size // 2) : center[0] + (size // 2),
        center[1] - (size // 2) : center[1] + (size // 2),
    ] = value

    return image


def cross(shape_img, center, size, value):
    """Draw a cross.

    Args:
        shape_img: Tuple, shape of the image.
        center: Tuple, center of the cross.
        size: Tuple, sizes of the cross.
        value: Integer with the value.

    Returns:
        Numpy array with a cross.
    """
    image = np.zeros(shape_img)
    image[
        center[0] : center[0] + size[0],
        center[1]
        + (size[0] // 2)
        - (size[1] // 2) : center[1]
        + (size[1] // 2)
        + (size[0] // 2),
    ] = value

    image[
        center[0]
        + (size[0] // 2)
        - (size[1] // 2) : center[0]
        + (size[1] // 2)
        + (size[0] // 2),
        center[1] : center[1] + size[0],
    ] = value

    return image
