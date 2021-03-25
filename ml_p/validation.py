from .blueprints import *
import numpy as np


def train_test_split(x, y, size=0.8, seed=None):
    if seed is not None:
        np.random.seed(seed)
    n = y.size
    split = int(n*size)
    index = np.random.permutation(n)
    X = x[index, :]
    Y = y[index]
    train_x = X[:split, :]
    test_x = X[split:, :]
    train_y = Y[:split]
    test_y = Y[split:]
    return train_x, test_x, train_y, test_y
