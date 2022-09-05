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
    folder = "./out/xsds/"
    os.makedirs(folder, exist_ok=True)
    counts: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}

    for i in auto.tqdm(range(5000)):
        image = drawing.circles(np.zeros((512, 512)), [1, 2, 3], 10)
        res = np.unique(image, return_counts=True)
        img_count = dict(zip(*res))

        for key in counts.keys():
            if key in img_count:
                counts[key].append(img_count[key])
            else:
                counts[key].append(0)

        cv2.imwrite(folder + f"{str(i).zfill(5)}.png", image)

    df = pd.DataFrame(counts)
    df.to_csv(folder + "dades.csv", sep=";")


if __name__ == "__main__":
    main()
