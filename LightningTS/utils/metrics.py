import numpy as np


def calculate_mae(x, y):
    return np.sum(np.abs(y - x)) / len(x)


def calculate_mape(x, y):
    return np.sum(np.abs(y - x) / x) / len(x)
