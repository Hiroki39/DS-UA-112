# %% import packages and read data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

school_data = pd.read_csv("middleSchoolData.csv")

# %% Q1
X = school_data["applications"].values.reshape(-1, 1)
Y = school_data["acceptances"]

regr = LinearRegression().fit(X, Y)
r_sqr = regr.score(X, Y)
r = np.sqrt(r_sqr)
betas = regr.coef_  # m
y_int = regr.intercept_  # b

plt.plot(school_data["applications"], school_data["acceptances"], 'o',
         markersize=2)  # Plot applications against acceptances
plt.xlabel('applications')
plt.ylabel('acceptances')

# Add regression line to visualization:
y_hat = betas[0] * school_data["applications"] + \
    y_int  # slope-intercept form, y = mx + b
plt.plot(school_data["applications"], y_hat, color='orange', linewidth=0.5)
# add title, r-squared rounded to nearest thousandth
plt.title('R^2: {:.3f}, R: {:.3f}'.format(r_sqr, r))

# we get r = 0.802

# %% Q2

# drop row with missing school size in tmp
tmp = school_data[["applications", "school_size", "acceptances"]]
tmp = tmp.dropna()

X = (tmp["applications"] /
     tmp["school_size"]).values.reshape(-1, 1)
Y = tmp["acceptances"]

regr = LinearRegression().fit(X, Y)
r_sqr = regr.score(X, Y)
r = np.sqrt(r_sqr)
betas = regr.coef_  # m
y_int = regr.intercept_  # b

plt.plot(tmp["applications"] /
         tmp["school_size"], tmp["acceptances"], 'o',
         markersize=2)  # Plot application rates against acceptances
plt.xlabel('application rates')
plt.ylabel('acceptances')

# 4. Add regression line to visualization:
y_hat = betas[0] * tmp["applications"] / tmp["school_size"] + \
    y_int  # slope-intercept form, y = mx + b
plt.plot(tmp["applications"] / tmp["school_size"],
         y_hat, color='orange', linewidth=0.5)
# add title, r-squared rounded to nearest thousandth
plt.title('R^2: {:.3f}, R: {:.3f}'.format(r_sqr, r))  # r = 0.802

# After comparing several datas, it turns out that the correlation coefficient
# between number of applications and acceptances is greater, which means the
# number of applications might be a better predictor

# %% Q3

# drop row with missing school size in tmp
tmp = school_data[["school_name", "school_size", "acceptances"]]
tmp = tmp.dropna()

# greater probability always result in greater odd, so we only need to find out
# schools with the highest proportion of student being admitted to HSPHS

tmp.loc[(tmp["acceptances"] / tmp["school_size"]).idxmax()]
# THE CHRISTA MCAULIFFE SCHOOL\I.S. 187 has the best per student odds of
# sending someone to HSPHS

# %% Q4

# "school climate factors", drop row with missing data
climate = school_data.iloc[:, 11:17]
climate = climate.dropna()

# Z-score the data:
zscored_cilmate = stats.zscore(climate)

# do a PCA to reduce number of dimension to 1
pca = PCA()
pca.fit(zscored_cilmate)

eig_vals = pca.explained_variance_
loadings = pca.components_
rotated_data = pca.fit_transform(zscored_cilmate)

# What a scree plot is: Plotting a bar graph of the sorted Eigenvalues
plt.bar(np.linspace(1, 6, 6), eig_vals)
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.plot([1, 6], [1, 1], color='red',
         linewidth=1)  # Kaiser criterion line

# add back to the temporary dataframe first to restore the index in the original
# dataframe (we drop some data but want to keep orginal index)

# sometime PCA will flip the sign, we need to flip it back to make this rating
# intuitive (higher is better)
climate["climate_rating"] = -rotated_data[:, 0]

# insert to original data frame
school_data["climate_rating"] = climate["climate_rating"]

# "measures of achievement", drop row with missing data
achievement = school_data.iloc[:, 21:24]
achievement = achievement.dropna()

# Z-score the data:
zscored_achievement = stats.zscore(achievement)

# do a PCA to reduce number of dimension to 1
pca = PCA()
pca.fit(zscored_achievement)

eig_vals = pca.explained_variance_
loadings = pca.components_
rotated_data = pca.fit_transform(zscored_achievement)

# What a scree plot is: Plotting a bar graph of the sorted Eigenvalues
plt.figure()
plt.bar(np.linspace(1, 3, 3), eig_vals)
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.plot([1, 3], [1, 1], color='red',
         linewidth=1)  # Kaiser criterion line

# add back to the temporary dataframe first to restore the index in the original
# dataframe (we drop some data but want to keep orginal index)
achievement["achievement_rating"] = rotated_data[:, 0]

# insert to original data frame
school_data["achievement_rating"] = achievement["achievement_rating"]

# finally, do a simple regression
tmp = school_data[["climate_rating", "achievement_rating"]]
tmp = tmp.dropna()

X = tmp["climate_rating"].values.reshape(-1, 1)
Y = tmp["achievement_rating"]

regr = LinearRegression().fit(X, Y)
r_sqr = regr.score(X, Y)
r = np.sqrt(r_sqr)
betas = regr.coef_  # m
y_int = regr.intercept_  # b

plt.figure()
plt.plot(tmp["climate_rating"], tmp["achievement_rating"], 'o',
         markersize=2)  # Plot application rates against acceptances
plt.xlabel('climate ratings')
plt.ylabel('achievement ratings')

# 4. Add regression line to visualization:
y_hat = betas[0] * tmp["climate_rating"] + \
    y_int  # slope-intercept form, y = mx + b
plt.plot(tmp["climate_rating"], y_hat, color='orange', linewidth=0.5)

# add title, r-squared rounded to nearest thousandth
plt.title('R^2: {:.3f}, R: {:.3f}'.format(r_sqr, r))

# There is a relationship between how students perceive their school and how
# the school performs on objective measures of achievement (R = 0.368 and R^2 =
# 0.136).

# %% Q5

# use acceptance rate instead of raw acceptance numbers
school_data["acceptance_rate"] = school_data["acceptances"] / \
    school_data["applications"]

# create a binary indicator for poor and rich schools
median_spending = school_data["per_pupil_spending"].median()
school_data["rich_school"] = school_data["per_pupil_spending"] > median_spending

# drop row with missing data in tmp
tmp = school_data[["rich_school", "achievement_rating"]]
tmp = tmp.dropna()

rich = tmp[tmp["rich_school"] == 1]["achievement_rating"]
poor = tmp[tmp["rich_school"] == 0]["achievement_rating"]
u_rich_achievement, p_rich_achievement = stats.mannwhitneyu(rich, poor)
p_rich_achievement

# drop row with missing data in tmp
tmp = school_data[["rich_school", "acceptance_rate"]]
tmp = tmp.dropna()

rich = tmp[tmp["rich_school"] == 1]["acceptance_rate"]
poor = tmp[tmp["rich_school"] == 0]["acceptance_rate"]
u_rich_acceptance, p_rich_acceptance = stats.mannwhitneyu(rich, poor)
p_rich_acceptance

# After conducting mann-whittney u test on the performance of rich and poor
# schools on achievement rating and accpetance rates, we got exteremely small
# p-values and conclude that poor schools and rich schools are extremely likely
# to perform differently when it comes to objective measures of achievements
# and admission to HSPHS

# %% Q6

# per pupil spending against achievement ratings
# drop row with missing data in tmp
tmp = school_data[["per_pupil_spending", "achievement_rating"]]
tmp = tmp.dropna()

X = tmp["per_pupil_spending"].values.reshape(-1, 1)
Y = tmp["achievement_rating"]

regr = LinearRegression().fit(X, Y)
r_sqr = regr.score(X, Y)
r = np.sqrt(r_sqr)
betas = regr.coef_  # m
y_int = regr.intercept_  # b

plt.figure()
plt.plot(tmp["per_pupil_spending"], tmp["achievement_rating"], 'o',
         markersize=2)  # Plot per pupil spending against achievement ratings
plt.xlabel('spending per pupil')
plt.ylabel('achievement ratings')

# Add regression line to visualization:
y_hat = betas[0] * tmp["per_pupil_spending"] + \
    y_int  # slope-intercept form, y = mx + b
plt.plot(tmp["per_pupil_spending"], y_hat, color='orange', linewidth=0.5)

# add title, r-squared rounded to nearest thousandth
plt.title('R^2: {:.3f}, R: {:.3f}'.format(r_sqr, r))


# per pupil spending against acceptance rate
# drop row with missing data in tmp
tmp = school_data[["per_pupil_spending", "acceptance_rate"]]
tmp = tmp.dropna()

X = tmp["per_pupil_spending"].values.reshape(-1, 1)
Y = tmp["acceptance_rate"]

regr = LinearRegression().fit(X, Y)
r_sqr = regr.score(X, Y)
r = np.sqrt(r_sqr)
betas = regr.coef_  # m
y_int = regr.intercept_  # b

plt.figure()
plt.plot(tmp["per_pupil_spending"], tmp["acceptance_rate"], 'o',
         markersize=2)  # Plot per pupil spending against acceptance rate
plt.xlabel('spending per pupil')
plt.ylabel('acceptance rate')

# Add regression line to visualization:
y_hat = betas[0] * tmp["per_pupil_spending"] + \
    y_int  # slope-intercept form, y = mx + b
plt.plot(tmp["per_pupil_spending"], y_hat, color='orange', linewidth=0.5)

# add title, r-squared rounded to nearest thousandth
plt.title('R^2: {:.3f}, R: {:.3f}'.format(r_sqr, r))


# average class sizes against achievement ratings
# drop row with missing data in tmp
tmp = school_data[["avg_class_size", "achievement_rating"]]
tmp = tmp.dropna()

X = tmp["avg_class_size"].values.reshape(-1, 1)
Y = tmp["achievement_rating"]

regr = LinearRegression().fit(X, Y)
r_sqr = regr.score(X, Y)
r = np.sqrt(r_sqr)
betas = regr.coef_  # m
y_int = regr.intercept_  # b

plt.figure()
plt.plot(tmp["avg_class_size"], tmp["achievement_rating"], 'o',
         markersize=2)  # Plot average class sizes against achievement ratings
plt.xlabel('average class size')
plt.ylabel('achievement ratings')

# Add regression line to visualization:
y_hat = betas[0] * tmp["avg_class_size"] + \
    y_int  # slope-intercept form, y = mx + b
plt.plot(tmp["avg_class_size"], y_hat, color='orange', linewidth=0.5)

# add title, r-squared rounded to nearest thousandth
plt.title('R^2: {:.3f}, R: {:.3f}'.format(r_sqr, r))


# average class sizes against acceptance rate
# drop row with missing data in tmp
tmp = school_data[["avg_class_size", "acceptance_rate"]]
tmp = tmp.dropna()

X = tmp["avg_class_size"].values.reshape(-1, 1)
Y = tmp["acceptance_rate"]

regr = LinearRegression().fit(X, Y)
r_sqr = regr.score(X, Y)
r = np.sqrt(r_sqr)
betas = regr.coef_  # m
y_int = regr.intercept_  # b

plt.figure()
plt.plot(tmp["avg_class_size"], tmp["acceptance_rate"], 'o',
         markersize=2)  # Plot average class sizes against acceptances
plt.xlabel('average class size')
plt.ylabel('acceptance rate')

# Add regression line to visualization:
y_hat = betas[0] * tmp["avg_class_size"] + \
    y_int  # slope-intercept form, y = mx + b
plt.plot(tmp["avg_class_size"], y_hat, color='orange', linewidth=0.5)

# add title, r-squared rounded to nearest thousandth
plt.title('R^2: {:.3f}, R: {:.3f}'.format(r_sqr, r))

# According to the plots, there's no evidence that the availability of material
# resources impacts object measures of achievement: the smaller the class and
# higher spending per pupil, the lower the achievement ratings and HSPHS
# acceptance rate, which is very unexpected. It is possible that there are some
# confounding factors we didn't rule out lead to this result. It is also
# possible that class size and per pupil spending from school does not matter
# too much because the availability of material resources for students is
# typically not determined by school, but  by their family. Perhaps the
# percentage of students living in households below the poverty line will
# better represent students' access to material resources in this case. (This
# is verified later in Q8)

# %% Q7
sorted_acceptance = school_data[["acceptances",
                                 "school_name"]].sort_values(by="acceptances",
                                                             ascending=False)

# normalize to get the percentage
normalized_sorted_accptance = sorted_acceptance["acceptances"] / \
    sorted_acceptance["acceptances"].sum()

plt.bar(sorted_acceptance["school_name"][:20], normalized_sorted_accptance[:20])
plt.xlabel('school')
plt.ylabel('% of total acceptance')

labels = sorted_acceptance["school_name"][:20]
plt.xticks(np.linspace(0, 19, 20), labels, rotation="vertical")

accumulated_percentage = normalized_sorted_accptance.cumsum()

sum(accumulated_percentage < 0.9) / len(accumulated_percentage)
# about 20.5% of schools account for 905 of students accepted to HSPHSs

# %% Q8

# ethnicity identity is not included because one of these columns could be
# determined by other colunmns and it is not possible to reduce the dimension
# on these coulumns. They may break the validity of this prediction model.
glm_data = school_data[["per_pupil_spending",
                        "avg_class_size", "disability_percent",
                        "climate_rating", "poverty_percent", "ESL_percent",
                        "school_size", "climate_rating", "acceptance_rate",
                        "achievement_rating"]]
glm_data = glm_data.dropna()

# to tell which factor is most important, we need to z-score the data
zscored_glm_data = stats.zscore(glm_data)

X = zscored_glm_data[:, :-2]
Y = zscored_glm_data[:, -2]

regr = LinearRegression().fit(X, Y)
betas = regr.coef_  # m
y_int = regr.intercept_  # b

plt.bar(np.linspace(0, 8, 8), betas)
plt.xlabel('variables')
plt.ylabel('betas')

labels = ["per_pupil_spending",
          "avg_class_size", "disability_percent",
          "climate_rating", "poverty_percent", "ESL_percent",
          "school_size", "climate_rating"]
plt.xticks(np.linspace(0, 8, 8), labels, rotation=45)
plt.title("Beta coefficients of each variable for prediting acceptance rate")


Y = zscored_glm_data[:, -1]

regr = LinearRegression().fit(X, Y)
betas = regr.coef_  # m
y_int = regr.intercept_  # b

plt.figure()
plt.bar(np.linspace(0, 8, 8), betas)
plt.xlabel('variables')
plt.ylabel('betas')

labels = ["per_pupil_spending",
          "avg_class_size", "disability_percent",
          "climate_rating", "poverty_percent", "ESL_percent",
          "school_size", "climate_rating"]
plt.xticks(np.linspace(0, 8, 8), labels, rotation=45)
plt.title(
    "Beta coefficients of each variable for predicting academic achievement")

# After taking all factors into accont, we conclude from the plot that the
# poverty rate is the determining factor on the HSPHS admission and achieving
# highs cores on objective measures of acceptance.

# %% Q9 & Q10

# As mentioned in Q8, based on the multiple regression model, the poverty rate
# seems to be the most relevant charateristic in determining acceptance of
# their students to HSPHS: the lower the poverty rate, the higher the HSPHS
# acception rate. To increase the number of applicants to HSPHS accepted from
# this schools, a feasible solution is to subsidize students living below the p
# poverty line and trying to make HSPHS adopt an admission criterion that is
# more friendly to students from low-income groups and students with
# disabilities (which is also an important factor according to Q8). Subsidizing
# students from low-income households and stuents with disabilities should also
# imporve object measures of achievement because the top two factors impacting
# the scores on objective measure of achiement are also poverty rate and
# disablity rate.
