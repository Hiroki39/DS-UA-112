# ------------------------------------------------------------------------------
# E8-EmpiricalSampleBounds.py
# Dec 7, 2020
# Hongyi Zheng
# Output the value corresponds to the lower/upper bound percentile
# Accepts any 1D iterable object.
# ------------------------------------------------------------------------------

import numpy as np

writtenBy = "Hongyi Zheng"


def empiricalSampleBounds(data, percentile):
    data = np.array(data)
    data = np.sort(data)
    lower_index = round(len(data) * (50 - percentile / 2) / 100) - 1
    upper_index = round(len(data) * (50 + percentile / 2) / 100) - 1
    return data[lower_index], data[upper_index]
