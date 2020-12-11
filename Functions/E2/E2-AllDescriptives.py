# ------------------------------------------------------------------------------
# E2-AllDescriptives.py
# Oct 16, 2020
# Hongyi Zheng
# Calculates and all descriptives of a set of values and return them as a numpy
# array.
# Accepts all types of 1D iterable object.
# ------------------------------------------------------------------------------

import numpy as np

writtenBy = "Hongyi Zheng"


def allDescriptives(input):
    n = len(input)
    mean = np.mean(input)
    median = np.median(input)
    std = np.std(input)
    mad = np.mean(abs(input - mean))
    sem = std / np.sqrt(n)
    return np.array([mean, median, std, mad, n, sem])
