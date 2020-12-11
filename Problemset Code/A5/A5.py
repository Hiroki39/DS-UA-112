# %% Import
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# %% Klonopin
klo_data = np.genfromtxt('klonopin.txt')
dosage, count = np.unique(klo_data[:, 1], return_counts=True)

data_by_dosage = [klo_data[klo_data[:, 1] == dosage[i]][:, 0]
                  for i in range(len(dosage))]

# %%
container_klo = np.empty([len(dosage), 4])
container_klo[:] = np.NaN

for i in range(len(dosage)):
    container_klo[i, 0] = np.mean(data_by_dosage[i])  # mu
    container_klo[i, 1] = np.std(data_by_dosage[i])  # sigma
    container_klo[i, 2] = len(data_by_dosage[i])  # n
    container_klo[i, 3] = container_klo[i, 1] / \
        np.sqrt(container_klo[i, 2])  # sem

grand_mean = np.mean(klo_data[:, 0])
SS_total = np.sum((klo_data[:, 0] - grand_mean)**2)
SS_between = np.sum(
    (container_klo[i, 0] - grand_mean)**2 for i in range(len(dosage)))
SS_within = SS_total - SS_between

MS_within = SS_within / (klo_data.size - len(dosage))
MS_between = SS_between / (len(dosage) - 1)
MS_between + MS_within

# %%
f, p = stats.f_oneway(data_by_dosage[0], data_by_dosage[1], data_by_dosage[2],
                      data_by_dosage[3], data_by_dosage[4], data_by_dosage[5])


# %%
x = ['0mg', '0.125mg', '0.25mg', '0.5mg', '1mg', '2mg']  # labels for the bars
x_pos = np.array([1, 2, 3, 4, 5, 6])  # x-values for the bars
plt.bar(
    x_pos, container_klo[:, 0],
    width=0.5, yerr=container_klo[:, 3])  # bars + error
plt.xticks(x_pos, x)  # label the x_pos with the labels
plt.xlabel('Dosage')
plt.ylabel('STAI')  # add y-label
plt.title('f = {:.3f}'.format(f) + ', p = {:.3f}'.format(p))

# %% Rats
rats_data = np.genfromtxt('rats.txt')
housing, housing_count = np.unique(rats_data[:, 1], return_counts=True)
exercise, exercise_count = np.unique(rats_data[:, 2], return_counts=True)

housing, exercise

# %%
data_by_housing = [rats_data[rats_data[:, 1] == housing[i]][:, 0]
                   for i in range(len(housing))]

data_by_exercise = [rats_data[rats_data[:, 2] == exercise[i]][:, 0]
                    for i in range(len(exercise))]

# %%
f_housing, p_housing = stats.f_oneway(data_by_housing[0],
                                      data_by_housing[1], data_by_housing[2])

# %%
f_exercise, p_exercise = stats.f_oneway(data_by_exercise[0],
                                        data_by_exercise[1],
                                        data_by_exercise[2],
                                        data_by_exercise[3])
# %%
container_housing = np.empty([len(housing), 4])
container_housing[:] = np.NaN

for i in range(len(housing)):
    container_housing[i, 0] = np.mean(data_by_housing[i])  # mu
    container_housing[i, 1] = np.std(data_by_housing[i])  # sigma
    container_housing[i, 2] = len(data_by_housing[i])  # n
    container_housing[i, 3] = container_housing[i, 1] / \
        np.sqrt(container_housing[i, 2])  # sem

container_exercise = np.empty([len(exercise), 4])
container_exercise[:] = np.NaN

for i in range(len(exercise)):
    container_exercise[i, 0] = np.mean(data_by_exercise[i])  # mu
    container_exercise[i, 1] = np.std(data_by_exercise[i])  # sigma
    container_exercise[i, 2] = len(data_by_exercise[i])  # n
    container_exercise[i, 3] = container_exercise[i, 1] / \
        np.sqrt(container_exercise[i, 2])  # sem
# %%
x = ['0', '1', '2']  # labels for the bars
x_pos = np.array([0, 1, 2])  # x-values for the bars
plt.bar(
    x_pos, container_housing[:, 0],
    width=0.5, yerr=container_housing[:, 3])  # bars + error
plt.xticks(x_pos, x)  # label the x_pos with the labels
plt.xlabel('Housing')
plt.ylabel('Cortical Thickness')  # add y-label
plt.title('f = {:.3f}'.format(f_housing) + ', p = {:.3f}'.format(p_housing))

plt.figure()
x = ['0min', '30min', '60min', '90min']  # labels for the bars
x_pos = np.array([0, 1, 2, 3])  # x-values for the bars
plt.bar(
    x_pos, container_exercise[:, 0],
    width=0.5, yerr=container_exercise[:, 3])  # bars + error
plt.xticks(x_pos, x)  # label the x_pos with the labels
plt.xlabel('Exercise')
plt.ylabel('Cortical Thickness')  # add y-label
plt.title('f = {:.3f}'.format(f_exercise) + ', p = {:.3f}'.format(p_exercise))
plt.plot()

# %%
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

rats_df = pd.DataFrame(data=rats_data, columns=[
                       "cortical_thickness", "housing", "exercise"])
model = ols(
    'cortical_thickness ~ housing + exercise + exercise*housing', data=rats_df).fit()
sm.stats.anova_lm(model, typ=2)

# %%
social_data = np.genfromtxt('socialstress.txt')
social_df = pd.DataFrame(data=social_data, columns=[
                         "cortisol", "sleep", "classes", "disposition", "socialsetting"])
model = ols(
    'cortisol ~ sleep + classes + disposition + socialsetting + sleep * classes + sleep * disposition + sleep * socialsetting + classes * disposition + classes * socialsetting + disposition * socialsetting + sleep * classes * disposition + sleep * classes * socialsetting + sleep * disposition * socialsetting + classes * disposition * socialsetting + sleep * classes * disposition * socialsetting', data=social_df).fit()
table = sm.stats.anova_lm(model, typ=2)

sum(table["sum_sq"][:-1]) / sum(table["sum_sq"])
table

# %%
blog_data = np.genfromtxt("blogData.txt")
month, month_count = np.unique(blog_data, return_counts=True)
chi_sqr_blog, p_blog = stats.chisquare(month_count)

# %%
birth_data = np.genfromtxt("births.txt")
date, date_count = np.unique(birth_data, return_counts=True)
chi_sqr_birth, p_birth = stats.chisquare(date_count)

# %%
happiness_data = np.genfromtxt("happiness.txt")
treatment = happiness_data[happiness_data[:, 1] == 1][:, 0]
control = happiness_data[happiness_data[:, 1] == 0][:, 0]
u_happiness, p_happiness = stats.mannwhitneyu(treatment, control)
p_happiness

# %%
