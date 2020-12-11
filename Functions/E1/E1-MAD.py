# ------------------------------------------------------------------------------
# E1-MAD.py
# Oct 11, 2020
# Hongyi Zheng
# Calculates the mean average deviation of a set of values.
# Accepts all types of 1D iterable object.
# ------------------------------------------------------------------------------


import numpy as np

writtenBy = "Hongyi Zheng"


def calcMAD(input):
    mean = np.mean(input)
    return np.mean(abs(input - mean))


calcMAD([1, 2, 3])
