""" Main module, generates the images

Written by: Miquel Miró Nicolau (UIB), 2022
"""
import os

import cv2
import numpy as np
import pandas as pd
from tqdm import auto

from uib_xsds import drawing

AREA_PX = False


def main() -> None:
    folder = "./out/aixi_v_area_128_px/"
    info = {"train": 50000, "val": 2000}
    for k, v in info.items():
        folder_divided = folder + f"{k}/"
        folder_gt = folder_divided + "/gt/"

        os.makedirs(folder_divided, exist_ok=True)
        os.makedirs(folder_gt, exist_ok=True)

        counts: dict[int, list[int]] = {1: [], 2: [], 3: []}

        for i in auto.tqdm(range(v)):
            image, img_count = drawing.circles(
                np.zeros((128, 128), dtype=np.uint8),
                values=[1, 2, 3],
                num_of_circles=6,
                overlapped=False,
                return_count=True,
            )

            if AREA_PX:
                res = np.unique(image, return_counts=True)
                img_count = dict(zip(*res))

            gts = []
            for key in counts.keys():
                aux = (image == key).astype(np.uint8)
                gts.append(aux)
                if key in img_count:
                    counts[key].append(img_count[key])
                else:
                    counts[key].append(0)

            cv2.imwrite(folder_divided + f"{str(i).zfill(6)}.png", image)
            cv2.imwrite(folder_gt + f"{str(i).zfill(6)}.png", np.dstack(gts))

        df = pd.DataFrame(counts)
        df.to_csv(folder_divided + "dades.csv", sep=";")


if __name__ == "__main__":
    main()
