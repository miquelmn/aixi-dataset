"""Suite of tests for the drawing module.

Written by: Miquel Miró Nicolau (UIB), 2022
"""

import unittest

import numpy as np

from aixi_dataset import drawing


class TestCircles(unittest.TestCase):
    """Test suite for the circles function."""

    @staticmethod
    def __draw_image(
        size: tuple[int, int], values: list[int], n_circles: int
    ) -> np.ndarray:
        image = np.zeros(size)
        image = drawing.circles(image, values, num_of_circles=n_circles)

        return image

    def test_shape(self) -> None:
        image = TestCircles.__draw_image((128, 128), [1, 2], 10)

        self.assertTupleEqual(image.shape, (128, 128))

    def test_distribution(self) -> None:
        counts = {0: 0, 1: 0, 2: 0}

        for _ in range(2500):
            image = TestCircles.__draw_image((128, 128), [1, 2], 10)
            res = np.unique(image, return_counts=True)
            res = zip(*res)

            for key, val in res:
                counts[key] += val
        diff = np.abs(counts[1] - counts[2])
        proportion = diff / (counts[1] + counts[2])

        self.assertLess(proportion, 0.01)
