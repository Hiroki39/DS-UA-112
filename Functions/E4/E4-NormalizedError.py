# ------------------------------------------------------------------------------
# E4-NormalizedError.py
# Oct 31, 2020
# Hongyi Zheng
# Calculates mean error between two columns of the input with specified order.
# Accepts Nx2 2D numpy array.
# ------------------------------------------------------------------------------

import numpy as np

writtenBy = "Hongyi Zheng"


def normalizedError(data, order):
    if len(data.shape) < 2 or data.shape[1] != 2:
        raise ValueError("Requires Nx2 numpy array")
    return np.power(np.mean(np.power(abs(data[:, 0] - data[:, 1]), order)),
                    1.0 / order)  # formula to calculate normalized error
