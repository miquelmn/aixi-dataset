""" Generates the XSDS shape images

Written by: Miquel Miró Nicolau (UIB), 2022
"""
import os
import random

import cv2
import numpy as np
import pandas as pd
from tqdm import auto

from uib_xsds import drawing


def main() -> None:
    folder = "./out/xsds_shape/"
    folder_gt = folder + "/gt/"

    os.makedirs(folder, exist_ok=True)
    os.makedirs(folder_gt, exist_ok=True)

    counts: list = []

    for i in auto.tqdm(range(5000)):
        image, areas, gts = drawing.polygons(
            np.zeros((512, 512)),
            4,
            random.randint(0, 2),
            random.randint(0, 2),
            random.randint(0, 2),
            255,
        )
        counts.append(areas)

        cv2.imwrite(folder + f"{str(i).zfill(5)}.png", image)
        cv2.imwrite(folder_gt + f"{str(i).zfill(5)}.png", np.dstack(gts))

    df = pd.DataFrame(counts)
    df.to_csv(folder + "dades.csv", sep=";")


if __name__ == "__main__":
    main()
