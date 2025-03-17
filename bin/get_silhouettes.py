import os
from glob import glob
from PIL import Image

import cv2
import numpy as np
from tqdm import auto as tqdm


def get_silhouettes(annotation, image):
    silhouette = np.copy(image)
    silhouette[annotation == 2] = 0

    segmentation = np.where(annotation != 2)
    if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
        x_min, x_max = int(np.min(segmentation[0])), int(np.max(segmentation[0]))
        y_min, y_max = int(np.min(segmentation[1])), int(np.max(segmentation[1]))

        silhouette = silhouette[x_min:x_max, y_min:y_max, :]

    return silhouette


def main():
    path_annot = os.path.join("..", "in", "zoo", "annotations", "trimaps", "*.png")
    path_images = os.path.join("..", "in", "zoo", "images", "*.jpg")

    annotations = glob(path_annot)
    images = glob(path_images)

    output_path = os.path.join("..", "in", "zoo", "silhouettes")
    os.makedirs(output_path, exist_ok=True)
    for img_path, mask_path in tqdm.tqdm(zip(images, annotations), desc="Siluetes generades"):
        _, name = os.path.split(mask_path)

        mask = np.array(Image.open(mask_path))
        image = np.array(Image.open(img_path).convert('RGB'))[:, :, ::-1]
        silhouette = get_silhouettes(mask, image)

        cv2.imwrite(os.path.join(output_path, name), silhouette)


if __name__ == '__main__':
    main()
