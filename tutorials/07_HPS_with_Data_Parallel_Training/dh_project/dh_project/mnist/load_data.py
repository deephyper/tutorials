import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))


def load_data_train_test():
    """Loads the MNIST dataset Training and Test sets with normalized pixels.

    Returns:
        tuple: (train_X, train_y), (test_X, test_y)
    """
    data_path = os.path.join(HERE, "mnist.npz")

    (train_X, train_y), (test_X, test_y) = tf.keras.datasets.mnist.load_data(
        path=data_path
    )

    train_X = train_X / 255
    test_X = test_X / 255

    return (train_X, train_y), (test_X, test_y)


def load_data_train_valid(prop=0.33, verbose=0):
    """Loads the MNIST dataset Training and Validation sets with normalized pixels.

    Returns:
        tuple: (train_X, train_y), (valid_X, valid_y)
    """

    (X, y), _ = load_data_train_test()

    train_X, valid_X, train_y, valid_y = train_test_split(
        X, y, test_size=prop, random_state=42
    )

    if verbose:
        print(f"train_X shape: {np.shape(train_X)}")
        print(f"train_y shape: {np.shape(train_y)}")
        print(f"valid_X shape: {np.shape(valid_X)}")
        print(f"valid_y shape: {np.shape(valid_y)}")

    return (train_X, train_y), (valid_X, valid_y)


if __name__ == "__main__":
    load_data_train_valid(verbose=1)
