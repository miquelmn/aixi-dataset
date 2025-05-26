"""Generates the AIXI shape images

Written by: Miquel Miró Nicolau (UIB), 2022
"""

import glob
import os
import random

import cv2
import numpy as np
import pandas as pd

from aixi_dataset import drawing

NUM_OBJECTS = 2  # Per forma
GRID_SIDE = 5

AREA_PX = False
TEXTURE_BG = [["./in/texture.jpg"], glob.glob("./in/textures/**/*.jpg"), None][-1]
SIZE_IMG = (128, 128)


def main() -> None:
    folder = f"./out/aixi_shape_{SIZE_IMG[0]}"

    if AREA_PX:
        folder += "_area"

    if TEXTURE_BG is not None:
        folder += "_texture"

    info = {"train": 50000, "val": 2000}

    for k, v in info.items():
        folder_divided = folder + f"/{k}/"
        folder_gt = folder_divided + "/gt/"

        counts: list = []

        os.makedirs(folder, exist_ok=True)
        os.makedirs(folder_gt, exist_ok=True)

        i = 0
        while i < v:
            if TEXTURE_BG is not None:
                texture_path = random.choice(TEXTURE_BG)
                background = cv2.imread(texture_path, cv2.IMREAD_GRAYSCALE)
                background = cv2.resize(background, SIZE_IMG)

                background = background / background.max()
            else:
                background = np.zeros(SIZE_IMG)

            image, areas, gts, counting, overlapped = drawing.polygons(
                np.zeros(SIZE_IMG),
                GRID_SIDE,
                random.randint(0, NUM_OBJECTS),
                random.randint(0, NUM_OBJECTS),
                random.randint(0, NUM_OBJECTS),
                [1],
            )

            if not overlapped:
                image = image + background
                image[image > 1] = 1

                if AREA_PX:
                    counting = areas

                counts.append(counting)

                cv2.imwrite(folder_divided + f"{str(i).zfill(5)}.png", image * 255)
                cv2.imwrite(folder_gt + f"{str(i).zfill(5)}.png", np.dstack(gts))
                i += 1
                print(f"Image {i} / {v} saved")

        df = pd.DataFrame(counts)
        df.to_csv(folder_divided + "dades.csv", sep=";")


if __name__ == "__main__":
    main()
