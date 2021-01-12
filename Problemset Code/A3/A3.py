import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import sys
sys.path.appsend(sys.path[0] + '/../Recitation/code/session5')
from simple_linear_regress_func import simple_linear_regress_func


def calc_partial_correlation(independent, dependent, control):
    control = np.array(control)
    regr1 = linear_model.LinearRegression()
    X = np.transpose(control)
    Y = dependent
    regr1.fit(X, Y)  # use fit method

    betas = regr1.coef_  # m
    y_int = regr1.intercept_  # b
    # multiply betas vector with data vector
    y_hat = np.dot(betas, control) + y_int
    residuals_1 = dependent - y_hat

    regr2 = linear_model.LinearRegression()
    Y = independent
    regr2.fit(X, Y)  # use fit method

    betas = regr2.coef_  # m
    y_int = regr2.intercept_  # b
    # multiply betas vector with data vector
    y_hat = np.dot(betas, control) + y_int
    residuals_2 = independent - y_hat

    return np.corrcoef(residuals_1, residuals_2)[0, 1]


data = np.genfromtxt('kepler.txt', delimiter='   ')  # load file as data

r_caste_iq = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
r_caste_iq

r_caste_iq_controllbrain = calc_partial_correlation(
    data[:, 0], data[:, 1], [data[:, 2]])
r_caste_iq_controllbrain

simple_linear_regress_func(np.transpose([data[:, 2], data[:, 1]]))

r_income_caste = np.corrcoef(data[:, 4], data[:, 0])[0, 1]
r_income_caste

r_income_caste_controllIQ = calc_partial_correlation(
    data[:, 4], data[:, 0], [data[:, 1]])
r_income_caste_controllIQ

r_income_caste_controllwork = calc_partial_correlation(
    data[:, 4], data[:, 0], [data[:, 3]])
r_income_caste_controllwork

r_income_caste_controllIQ_and_work = calc_partial_correlation(
    data[:, 0], data[:, 4], [data[:, 1], data[:, 3]])
r_income_caste_controllIQ_and_work

simple_linear_regress_func(np.transpose([data[:, 1], data[:, 4]]))
simple_linear_regress_func(np.transpose([data[:, 3], data[:, 4]]))

# Model: IQ and hours worked
X = np.transpose([data[:, 1], data[:, 3]])  # IQ, hours worked
Y = data[:, 4]  # income
# linearRegression function from linear_model
regr = linear_model.LinearRegression()
regr.fit(X, Y)  # use fit method
r_sqr = regr.score(X, Y)
betas = regr.coef_  # m
y_int = regr.intercept_  # b
# Visualize: actual vs. predicted income (from model)
y_hat = betas[0] * data[:, 1] + betas[1] * data[:, 3] + y_int
plt.plot(y_hat, data[:, 3], 'o', markersize=.75)  # y_hat, income
plt.xlabel('Prediction from model')
plt.ylabel('Actual income')
plt.title('R^2: {:.3f}'.format(r_sqr))
120 * betas[0] + 50 * betas[1] + y_int
