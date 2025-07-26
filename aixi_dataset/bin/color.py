"""Generates the images of AIXI-Color

Written by: Miquel Miró Nicolau (UIB), 2022
"""

import os

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from aixi_dataset import random_figs

AREA_PX = False
NORMALIZE_VALUES = True


def main() -> None:
    folder = "./out/aixi_color_128_px/"
    info = {"train": 50000, "val": 2000}

    values = [1, 2, 3]

    if NORMALIZE_VALUES:
        values = [v / max(values) for v in values]
    for k, v in info.items():
        folder_divided = folder + f"{k}/"
        folder_gt = folder_divided + "/gt/"

        os.makedirs(folder_divided, exist_ok=True)
        os.makedirs(folder_gt, exist_ok=True)

        counts = {k: [] for k in values}  # {1: [], 2: [], 3: []}

        for i in tqdm(range(v)):
            image, img_count = random_figs.circles(
                np.zeros((128, 128), dtype=np.float32),
                values=values,
                num_of_circles=6,
                overlapped=False,
                return_count=True,
                margin=5,
                max_iter=20,
            )

            if AREA_PX:
                res = np.unique(image, return_counts=True)
                img_count = dict(zip(*res))

            gts = []
            for key in counts.keys():
                aux = (image == key).astype(np.float32)
                gts.append(aux)
                if key in img_count:
                    counts[key].append(img_count[key])
                else:
                    counts[key].append(0)

            cv2.imwrite(folder_divided + f"{str(i).zfill(6)}.png", image * 255)
            cv2.imwrite(folder_gt + f"{str(i).zfill(6)}.png", np.dstack(gts) * 255)

        df = pd.DataFrame(counts)
        df.to_csv(folder_divided + "dades.csv", sep=";")


if __name__ == "__main__":
    main()
