""" Implementation of the Guidotti[1] method to generate SABs.

[1] Guidotti, R. (2021). Evaluating local explanation methods on ground truth. Artificial Intelligence, 291, 103428.

Written by: Miquel Miró Nicolau (UIB), 2025
"""
import random
import os

from tqdm.auto import tqdm
import numpy as np
import cv2

SIZE_IMG = (32, 32, 3)
SIZE_CELL = (4, 4)
SIZE_PATTERN = (8, 8)


def generate_pattern_one_c(pattern_size, cell_size):
    """ Generate a pattern on one channel.

    Generate a pattern with the given size and cell size, a pattern is a matrix with 0s and 1s.

    Args:
        pattern_size: Tuple with the size of the pattern.
        cell_size: Tuple with the size of the cell.

    Returns:
        NumPy array with the pattern.
    """
    pattern = np.zeros(pattern_size)

    H, W = pattern_size
    h, w = cell_size

    # Ensure dimensions are divisible by (h, w)
    assert H % h == 0 and W % w == 0, "Array dimensions must be divisible by patch size"

    shapes_of_cells = (H // h, h, W // w, w)  # (n cells h, h, n cells w, w)
    patches = pattern.reshape(*shapes_of_cells)

    random = np.random.rand(H // h, W // w)
    random = (random > 0.5).astype(np.uint8)

    patches[:] = random[:, np.newaxis, :, np.newaxis]  # Shape (2,1,2,1)
    patches = patches.reshape(pattern_size)

    return patches


def generate_pattern(pattern_size, cell_size):
    """ Generate a pattern with 3 channels.

    Args:
        pattern_size: Tuple with the size of the pattern.
        cell_size: Tuple with the size of the cell.

    Returns:
        Numpy array with the pattern.
    """
    H, W = pattern_size
    h, w = cell_size

    assert H % h == 0 and W % w == 0, "Array dimensions must be divisible by patch size"

    mask = generate_pattern_one_c(pattern_size, cell_size)

    shapes_of_cells = (H // h, h, W // w, w)

    mask = mask.reshape(shapes_of_cells)
    mask = np.stack((mask,) * 3, axis=-1)

    random_channel = np.random.rand(H // h, W // w, 3)
    max_chanel = np.max(random_channel, axis=-1)

    use_channel = (random_channel >= max_chanel[:, :, np.newaxis]).astype(np.uint8)
    use_channel = use_channel[:, np.newaxis, :, np.newaxis, :]

    mask = mask * use_channel
    mask = mask.reshape(H, W, 3)

    return mask


def main():
    folder = f"./out/seneca_sp_{SIZE_PATTERN}_sc_{SIZE_CELL}/"
    info = {"train": 2000, "val": 1000}

    gt_pattern = generate_pattern(SIZE_PATTERN, SIZE_CELL)
    for k, v in info.items():
        folder_divided = folder + f"/{k}/"
        folder_gt = folder_divided + "/gt/"

        os.makedirs(folder, exist_ok=True)
        os.makedirs(folder_gt, exist_ok=True)

        for i in tqdm(range(v)):
            resultat = np.zeros(SIZE_IMG)

            H, W, C = SIZE_IMG
            h, w = SIZE_PATTERN

            assert H % h == 0 and W % w == 0, "Image dimensions must be divisible by pattern size"

            resultat = resultat.reshape(H // h, h, W // w, w, C)

            for pos_x in range(resultat.shape[0]):
                for pos_y in range(resultat.shape[2]):
                    if random.randint(0, 3) == 3:
                        resultat[pos_x, :, pos_y, :, :] = generate_pattern(SIZE_PATTERN, SIZE_CELL)

            contains_pattern = random.randint(0, 1)
            gt_sal_map = np.zeros((H // h, h, W // w, w))

            if contains_pattern:
                pos_pattern = random.randint(0, (resultat.shape[0] * resultat.shape[2]) - 1)
                size = resultat.shape[0]

                resultat[pos_pattern // size, :, pos_pattern % size, :, :] = 0
                resultat[pos_pattern // size, :, pos_pattern % size, :, :] = gt_pattern

                gt_sal_map[pos_pattern // size, :, pos_pattern % size, :] = np.sum(gt_pattern > 0, axis=-1)
            resultat = resultat.reshape(SIZE_IMG)
            gt_sal_map = gt_sal_map.reshape((SIZE_IMG[0], SIZE_IMG[1]))

            cv2.imwrite(folder_divided + f"{str(i).zfill(5)}_{contains_pattern}.png", resultat)
            cv2.imwrite(folder_gt + f"{str(i).zfill(5)}.png", gt_sal_map)


if __name__ == '__main__':
    main()
