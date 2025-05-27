import numpy as np
import cv2


def circle(shape_img, center, radius, value):
    """Draw a circle.

    Args:
        shape_img: Tuple, shape of the image.
        center: Tuple, center of the circle (y, x).
        radius: Integer, radius of the circle.
        value: Integer, value of the circle.

    Returns:
        Image with a circle drawn.
    """
    image = np.zeros(shape_img)

    return cv2.circle(
        image,
        center,
        radius,
        value,
        thickness=-1,
    )


def square(shape_img, top_left_corner, size, value):
    """Draw a square.

    Args:
        shape_img: Tuple, shape of the image.
        top_left_corner: Tuple, top left corner of the square (y, x).
        size: Integer, size of the square (width and height).
        value: Integer, value of the square.

    Returns:
        Returns a numpy array with a square drawn.
    """
    image = np.zeros(shape_img)

    image[
        top_left_corner[0] : top_left_corner[0] + size,
        top_left_corner[1] : top_left_corner[1] + size,
    ] = value

    return image


def cross(shape_img, positions, size, value):
    """Draw a cross.

    Args:
        shape_img: Tuple, shape of the image.
        positions: Tuple, center of the cross.
        size: Tuple, sizes of the cross.
        value: Integer with the value.

    Returns:
        Numpy array with a cross.
    """
    image = np.zeros(shape_img)
    image[
        positions[0] : positions[0] + size[0],
        positions[1]
        + (size[0] // 2)
        - (size[1] // 2) : positions[1]
        + (size[1] // 2)
        + (size[0] // 2),
    ] = value

    image[
        positions[0]
        + (size[0] // 2)
        - (size[1] // 2) : positions[0]
        + (size[1] // 2)
        + (size[0] // 2),
        positions[1] : positions[1] + size[0],
    ] = value

    return image
