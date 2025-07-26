import pathlib
import random

import cv2
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

from aixi_dataset import drawing


def main(output_folder: str, train_size: int, val_size: int, image_size: tuple):
    folder = pathlib.Path(output_folder).joinpath("one_obj")

    info: dict[str, int] = {"train": train_size, "val": val_size}
    for k, v in info.items():
        folder_divided: pathlib.Path = folder / k
        folder_divided.mkdir(parents=True, exist_ok=True)

        for _ in tqdm(range(v)):
            center_position = [(pos // 2) for pos in image_size]
            # position = center_position
            position = (
                random.randint(image_size[0] // 8, (image_size[0] // 8) * 7),
                random.randint(image_size[0] // 8, (image_size[0] // 8) * 7),
            )

            size = random.randint(min(position) // 8, min(image_size) // 4)

            mult = (random.randint(150, 190)) / 100

            image = drawing.square(image_size, position, int(size * mult), 1)

            image_c = drawing.circle(image_size, position, size, 1)

            image = image + image_c

            rotation_matrix = cv2.getRotationMatrix2D(
                center_position, random.randint(0, 180), 1
            )
            image = cv2.warpAffine(
                image, rotation_matrix, (image.shape[1], image.shape[0])
            )

            image[image > 1] = 1

            plt.imshow(image)
            plt.show()
        break


if __name__ == "__main__":
    main("./out", 50, 50, (128, 128))
