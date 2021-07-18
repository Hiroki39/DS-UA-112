# ------------------------------------------------------------------------------
# E7-BayesCalculator.py
# Nov 23, 2020
# Hongyi Zheng
# Output the posterior probabilitty of A given B
# Accepts input with 3 numeric values and an integer flag 1 or 2
# ------------------------------------------------------------------------------

writtenBy = "Hongyi Zheng"


def bayesCalculator(priorA, input2, likelihood, flag):
    if flag == 1:
        return priorA * likelihood / input2
    else:
        return priorA * likelihood / (priorA * likelihood +
                                      input2 * (1 - priorA))
