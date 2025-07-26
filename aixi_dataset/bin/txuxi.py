"""Generates the TXUXI images

Written by: Miquel Miró Nicolau (UIB), 2023
"""

import argparse
import glob
import logging
import os.path
import pathlib
import random

import cv2
import numpy as np
import pandas as pd
from tqdm import auto

from aixi_dataset import generate_figs

BACKGROUNDS = [
    None,
    ["./in/texture.jpg"],
    ["./in/texture_v2.jpg"],
    glob.glob("./in/textures/**/*.jpg"),
]


def main(
    num_objects: int,
    grid_side: int,
    area_px: bool,
    ver: int,
    size_img: tuple[int, int],
    train_size: int,
    val_size: int,
    output_folder: str,
    extension: str,
    save_background: bool,
) -> None:
    quality = [int(cv2.IMWRITE_JPEG_QUALITY), 75] if extension == "jpg" else None
    texture_bg = BACKGROUNDS[ver]

    folder: pathlib.Path = pathlib.Path(output_folder).joinpath(
        f"shape_v{ver}_{size_img[0]}px_n{num_objects}"
    )

    logger.info(f"Saving to folder {folder}")

    if area_px:
        folder = folder.with_name(folder.stem + "_area")

    if texture_bg is not None:
        folder.with_name(folder.stem + "_texture")

    info: dict[str, int] = {"train": train_size, "val": val_size}
    for k, v in info.items():
        folder_divided: pathlib.Path = folder / k
        folder_gt: pathlib.Path = folder_divided / "gt"
        counts: list = []

        folder.mkdir(parents=True, exist_ok=True)
        folder_gt.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Folders created for {k} set")

        used_background = []

        for i in auto.tqdm(range(v)):
            if texture_bg is not None:
                texture_path = random.choice(texture_bg)
                background: np.ndarray = cv2.imread(texture_path, cv2.IMREAD_GRAYSCALE)
                background = cv2.resize(background, size_img)
                background = background / background.max()

                if save_background:
                    texture_path = os.path.join(*texture_path.split(os.path.sep)[-2:])
                    used_background.append(texture_path)

                logger.debug(f"Background generated for image {i}")
            else:
                background: np.ndarray = np.zeros(size_img)

            image, areas, gts, counting, overlapped = generate_figs.polygons(
                np.zeros(size_img),
                grid_side,
                random.randint(0, num_objects),
                random.randint(0, num_objects),
                random.randint(0, num_objects),
                [1],
            )

            image = image + background
            image[image > 1] = 1

            if area_px:
                counting = areas

            counts.append(counting)

            cv2.imwrite(
                str(folder_divided / f"{str(i).zfill(5)}.{extension}"),
                image * 255,
                quality,
            )
            cv2.imwrite(
                str(folder_gt / f"{str(i).zfill(5)}.{extension}"),
                np.dstack(gts),
                quality,
            )
            logger.debug(f"Images saved for {k} set, image {i}")

        df = pd.DataFrame(counts)
        df.to_csv(str(folder_divided / "dades.csv"), sep=";")

        if save_background:
            df = pd.DataFrame(used_background)
            df.to_csv(str(folder_divided / "used_background.csv"), sep=";")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AIXI shape images.")
    parser.add_argument("-b", action="store_true", help="Save background images.")
    parser.add_argument(
        "--num-objects", type=int, default=2, help="Number of objects in the images."
    )
    parser.add_argument("--grid-side", type=int, default=4, help="Grid side size.")
    parser.add_argument(
        "--area-px",
        action="store_true",
        help="Count the area of the objects in pixels.",
    )
    parser.add_argument(
        "--ver", type=int, default=0, help="Type of texture used in the background."
    )
    parser.add_argument(
        "--size-img", nargs=2, type=int, default=[128, 128], help="Image size."
    )
    parser.add_argument(
        "--train-size", type=int, default=50000, help="Number of training images."
    )
    parser.add_argument(
        "--val-size", type=int, default=2000, help="Number of validation images."
    )
    parser.add_argument(
        "--output-folder", type=str, default="./out/", help="Path to the output folder."
    )

    parser.add_argument(
        "--extension", type=str, default="png", help="Extension of the files."
    )

    args = parser.parse_args()

    # configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    main(
        num_objects=args.num_objects,
        grid_side=args.grid_side,
        area_px=args.area_px,
        ver=args.ver,
        size_img=args.size_img,
        train_size=args.train_size,
        val_size=args.val_size,
        output_folder=args.output_folder,
        extension=args.extension,
        save_background=args.b,
    )
