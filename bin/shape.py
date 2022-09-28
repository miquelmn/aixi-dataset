""" Generates the XSDS shape images

Written by: Miquel Miró Nicolau (UIB), 2022
"""
import os
import random

import cv2
import numpy as np
import pandas as pd
from skimage import util
from tqdm import auto

from uib_xsds import drawing

NUM_OBJECTS = 5
GRID_SIDE = 5


def main() -> None:
    folder = "./out/xsds_shape_count_v3/"
    info = {"train": 50000, "val": 2000}

    for k, v in info.items():
        folder_divided = folder + f"{k}/"
        folder_gt = folder_divided + "/gt/"

        list_of_areas: list = []
        counts: list = []

        os.makedirs(folder, exist_ok=True)
        os.makedirs(folder_gt, exist_ok=True)

        for i in auto.tqdm(range(v)):
            image, areas, gts, counting = drawing.polygons(
                np.zeros((512, 512)),
                GRID_SIDE,
                random.randint(0, NUM_OBJECTS),
                random.randint(0, NUM_OBJECTS),
                random.randint(0, NUM_OBJECTS),
                [random.randint(128, 255) for _ in range(15)],
            )
            image = image.astype(np.float64) / 255
            image = util.random_noise(image, "gaussian", seed=42, clip=True) * 255
            image = image.astype(np.uint8)

            list_of_areas.append(areas)
            counts.append(counting)

            cv2.imwrite(folder_divided + f"{str(i).zfill(5)}.png", image)
            cv2.imwrite(folder_gt + f"{str(i).zfill(5)}.png", np.dstack(gts))

        df = pd.DataFrame(counts)
        df.to_csv(folder_divided + "dades.csv", sep=";")


if __name__ == "__main__":
    main()
