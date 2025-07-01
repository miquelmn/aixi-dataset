"""Module to generate tabular synthetic tabular data.

Written by: Miquel Miró Nicolau (UIB), 2025
"""

import random
import argparse
from tqdm.auto import tqdm

import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed = 42


def main(samples, features, test_size):
    res = np.zeros((samples, features))

    for i in tqdm(range(samples)):
        for j in range(features):
            res[i, j] = random.uniform(0, 1)

    train, test = train_test_split(res, test_size=test_size, random_state=42)

    np.savetxt("./out/train.csv", train, delimiter=";")
    np.savetxt("./out/test.csv", test, delimiter=";")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DAIXI data.")
    parser.add_argument(
        "--samples", type=int, default=50000, help="Number of samples to generate."
    )
    parser.add_argument(
        "--features", type=int, default=3, help="Number of features to generate."
    )
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size.")
    args = parser.parse_args()

    main(args.samples, args.features, args.test_size)
