import numpy as np
from scipy import stats

# load file as data
data1 = np.genfromtxt('Sadex1.txt', delimiter='\t')
data2 = np.genfromtxt('Sadex2.txt', delimiter='\t')
data3 = np.genfromtxt('Sadex3.txt', delimiter='\t')
data4 = np.genfromtxt('Sadex4.txt', delimiter='\t')

treatment1 = data1[data1[:, 1] == 1]
control1 = data1[data1[:, 1] == 0]

ate1 = np.mean(treatment1[:, 0]) - np.mean(control1[:, 0])
ate1

t1, p1 = stats.ttest_ind(treatment1[:, 0], control1[:, 0])
p1

ate2 = np.mean(data2[:, 1]) - np.mean(data2[:, 0])
ate2

t2, p2 = stats.ttest_rel(data2[:, 1], data2[:, 0])
p2

treatment3 = data3[data3[:, 1] == 1]
control3 = data3[data3[:, 1] == 0]

t3, p3 = stats.ttest_ind(treatment3[:, 0], control3[:, 0])
p3

ate4 = np.mean(data4[:, 1]) - np.mean(data4[:, 0])
ate4

t4, p4 = stats.ttest_rel(data4[:, 1], data4[:, 0])
p4
