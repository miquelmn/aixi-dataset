import os
import math
import random
from glob import glob

import cv2
import numpy as np
from tqdm.auto import tqdm

N_IMAGES = {"train": 50000, "val": 2000}
SIL_4_IMG = 4
IMG_SIZE = (512, 512, 3)
SIL_SIZE = 128
GRID_SIZE = 4

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

def image_resize(image, size, inter=cv2.INTER_AREA):
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=inter)
    return resized


def create_image(silhouette_paths, img_size, sil_4_image, sil_size):
    image = np.zeros(img_size, dtype=np.uint8)
    cat_map = np.zeros(img_size[:2])
    dog_map = np.zeros_like(cat_map)

    grid_positions = np.arange(GRID_SIZE ** 2)
    chosen_positions = np.random.choice(grid_positions, sil_4_image, replace=False)

    for i in range(chosen_positions):
        path = random.choice(silhouette_paths)
        cat = os.path.basename(path)[0].isupper()  # Boolean for class

        sil = cv2.imread(path)
        sil = image_resize(sil, random.randint(sil_size // 2, sil_size))

        # Calculate padding only once
        pad_top = (sil_size - sil.shape[0]) // 2
        pad_bottom = sil_size - sil.shape[0] - pad_top
        pad_left = (sil_size - sil.shape[1]) // 2
        pad_right = sil_size - sil.shape[1] - pad_left

        sil = cv2.copyMakeBorder(
            sil, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0
        )

        row, col = (i // GRID_SIZE) * sil_size, (i % GRID_SIZE) * sil_size
        mask = sil > 0  # Faster boolean mask

        if cat:
            cat_map[row:row + sil_size, col:col + sil_size][mask] = 1
        else:
            dog_map[row:row + sil_size, col:col + sil_size][mask] = 1

        image[row:row + sil_size, col:col + sil_size, :] = sil

    gt = np.dstack((cat_map, dog_map, np.zeros_like(cat_map, dtype=np.uint8)))
    return image, gt


def main():
    silhouette_paths = os.path.join("..", "in", "zoo", "silhouettes", "*.png")
    silhouette_paths = glob(silhouette_paths)

    # Split into cats and dogs
    dog_path = [p for p in silhouette_paths if not os.path.split(p)[-1][0].isupper()]
    cat_path = [p for p in silhouette_paths if os.path.split(p)[-1][0].isupper()]

    # Balance classes
    dog_path = random.sample(dog_path, len(cat_path))
    silhouette_paths = dog_path + cat_path


    for phase, n_images in N_IMAGES.items():
        out_path = os.path.join("..", "out", "zaixi", phase)
        gt_path = os.path.join(out_path, "gt")

        os.makedirs(out_path, exist_ok=True)
        os.makedirs(gt_path, exist_ok=True)

        for img_idx in tqdm(range(n_images), desc=f"Phase {phase}"):
            image, gt = create_image(silhouette_paths, IMG_SIZE, SIL_4_IMG, SIL_SIZE)

            filename = f"{img_idx:06d}.png"

            cv2.imwrite(os.path.join(out_path, filename), image)
            cv2.imwrite(os.path.join(os.path.join(gt_path, filename)), gt)

if __name__ == '__main__':
    main()
