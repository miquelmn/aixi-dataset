""" Main module, generates the images

Written by: Miquel Miró Nicolau (UIB), 2022
"""
import os

import cv2
import numpy as np
import pandas as pd
from tqdm import auto

from uib_xsds import drawing


def main() -> None:
    folder = "./out/xsds_shape/"
    os.makedirs(folder, exist_ok=True)
    counts: list = []

    for i in auto.tqdm(range(5000)):
        image, areas = drawing.polygons(np.zeros((512, 512)), 4, 2, 2, 2, 255)
        counts.append(areas)

        cv2.imwrite(folder + f"{str(i).zfill(5)}.png", image)

    df = pd.DataFrame(counts)
    df.to_csv(folder + "dades.csv", sep=";")


if __name__ == "__main__":
    main()
