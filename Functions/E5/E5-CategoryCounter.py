# ------------------------------------------------------------------------------
# E5-CategoryCounter.py
# Oct 31, 2020
# Hongyi Zheng
# Get the unique values and their frequency in the input and return them as an
# Nx2 numpy array.
# Accepts any 1D iterable object.
# ------------------------------------------------------------------------------

import numpy as np

writtenBy = "Hongyi Zheng"


def categoryCounter(data):
    return np.vstack(np.unique(data, return_counts=True)).T
