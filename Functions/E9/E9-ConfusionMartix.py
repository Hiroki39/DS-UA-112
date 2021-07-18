# ------------------------------------------------------------------------------
# E9-EmpiricalSampleBounds.py
# Dec 12, 2020
# Hongyi Zheng
# Output the confusion matrix as 2x2 numpy array
# The function requires four inputs: first three inputs are 1d numpy arrays
# representing x values, the null distribution, and the signal distribution
# respectively. The fourth input is threshold value. This function assume that
# x values are sorted
# ------------------------------------------------------------------------------

import numpy as np

writtenBy = "Hongyi Zheng"


def confusionMatrix(x, y0, y1, threshold):
    threshold_index = np.where(x <= threshold)[0][-1]
    true_positive = np.sum(y1[threshold_index + 1:])
    false_positive = np.sum(y0[threshold_index + 1:])
    true_negative = np.sum(y0[:threshold_index + 1])
    false_negative = np.sum(y1[:threshold_index + 1])
    return np.array([[true_positive, false_positive],
                     [false_negative, true_negative]])
