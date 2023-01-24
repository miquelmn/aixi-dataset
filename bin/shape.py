""" Generates the AIXI shape images

Written by: Miquel Miró Nicolau (UIB), 2022
"""
import os
import random

import cv2
import numpy as np
import pandas as pd
from tqdm import auto

from uib_xsds import drawing

NUM_OBJECTS = 2
GRID_SIDE = 5

AREA_PX = True


def main() -> None:
    if AREA_PX:
        folder = "./out/aixi_shape_area_128_px/"
    else:
        folder = "./out/aixi_shape_128_px/"

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
                np.zeros((128, 128)),
                GRID_SIDE,
                random.randint(0, NUM_OBJECTS),
                random.randint(0, NUM_OBJECTS),
                random.randint(0, NUM_OBJECTS),
                [1],
            )

            if AREA_PX:
                res = np.unique(image, return_counts=True)
                counting = dict(zip(*res))

            list_of_areas.append(areas)
            counts.append(counting)

            cv2.imwrite(folder_divided + f"{str(i).zfill(5)}.png", image)
            cv2.imwrite(folder_gt + f"{str(i).zfill(5)}.png", np.dstack(gts))

        df = pd.DataFrame(counts)
        df.to_csv(folder_divided + "dades.csv", sep=";")


if __name__ == "__main__":
    main()
